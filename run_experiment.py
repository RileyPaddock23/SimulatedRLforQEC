"""
RL Drift-Tracking Experiment on the Rotated Surface Code
========================================================
Closed-loop PEPG steering of drifting control parameters, driven only by
detector error rates (EDR). Time advances with cumulative physical shots
(a "shot clock"): each policy update costs batch_size * shots_per_candidate
shots, and the drift keeps moving while candidates are being measured.

Compares three policies at periodic checkpoints:
  - Fixed:   parameters frozen at their t=0 calibration (mu = 0)
  - RL:      the agent's current policy mean mu(t)
  - Optimal: the ground-truth optimum p_opt(t) (oracle lower bound)

Examples:
    python run_experiment.py --drift sinusoidal
    python run_experiment.py --drift arma --arma-alpha 0.67
    python run_experiment.py --drift arma --arma-alpha 1.0 --arma-white 0.2 --plot
"""

import argparse
import json
import time
from typing import Dict

import numpy as np

from rl_agent import RLControlAgent
from surface_code_env import RotatedSurfaceCodeEnv

# Reference epoch used to convert shots -> environment time: 20 candidates * 350 shots
SHOTS_PER_EPOCH_REF = 7000.0


def run_trial(
    env: RotatedSurfaceCodeEnv,
    agent: RLControlAgent,
    total_shots: int,
    shots_per_candidate: int,
    batch_size: int,
    eval_interval: int,
    eval_shots: int,
    verbose: bool = True,
) -> Dict:
    fixed_policy = np.zeros(env.num_params, dtype=np.float32)
    history = {k: [] for k in (
        "shot", "epoch", "ler_fixed", "ler_rl", "ler_opt",
        "edr_fixed", "edr_rl", "edr_opt", "tracking_rms", "mu_0", "p_opt_0",
    )}

    if verbose:
        header = (f"{'Shot':>9} | {'EDR fixed':>9} | {'EDR RL':>8} | {'EDR opt':>8} | "
                  f"{'LER fixed':>9} | {'LER RL':>8} | {'LER opt':>8} | {'RMS err':>7}")
        print(header)
        print("-" * len(header))

    cur_shot = 0
    next_eval = 0
    update_shots = batch_size * shots_per_candidate

    while True:
        # Periodic evaluation checkpoint (static snapshot at the current physical time)
        if cur_shot >= next_eval:
            t_ep = cur_shot / SHOTS_PER_EPOCH_REF
            p_opt = env.get_drift_opt(t_ep)
            ev_f = env.evaluate_policy(fixed_policy, t=t_ep, shots=eval_shots, decode=True)
            ev_r = env.evaluate_policy(agent.mu, t=t_ep, shots=eval_shots, decode=True)
            ev_o = env.evaluate_policy(p_opt, t=t_ep, shots=eval_shots, decode=True)
            rms = float(np.sqrt(np.mean((agent.mu - p_opt) ** 2)))

            history["shot"].append(cur_shot)
            history["epoch"].append(t_ep)
            for tag, ev in (("fixed", ev_f), ("rl", ev_r), ("opt", ev_o)):
                history[f"ler_{tag}"].append(ev["logical_error_rate"])
                history[f"edr_{tag}"].append(ev["mean_edr"])
            history["tracking_rms"].append(rms)
            history["mu_0"].append(float(agent.mu[0]))
            history["p_opt_0"].append(float(p_opt[0]))

            if verbose:
                print(f"{cur_shot:>9,} | {ev_f['mean_edr']:>9.4f} | {ev_r['mean_edr']:>8.4f} | {ev_o['mean_edr']:>8.4f} | "
                      f"{ev_f['logical_error_rate']:>9.4f} | {ev_r['logical_error_rate']:>8.4f} | "
                      f"{ev_o['logical_error_rate']:>8.4f} | {rms:>7.3f}")
            next_eval += eval_interval

        if cur_shot + update_shots > total_shots:
            break

        # Candidates are measured sequentially along the shot timeline
        cands = agent.sample_policies(batch_size=batch_size, antithetic=True)
        obs = np.zeros((batch_size, env.num_detectors), dtype=np.float32)
        for k in range(batch_size):
            t_cand = (cur_shot + (k + 0.5) * shots_per_candidate) / SHOTS_PER_EPOCH_REF
            obs[k] = env.evaluate_policy(cands[k], t=t_cand, shots=shots_per_candidate)["edr_vector"]

        agent.update(cands, obs)
        cur_shot += update_shots

    # Steady-state summary over the last third of checkpoints
    tail = slice(-max(1, len(history["shot"]) // 3), None)
    s = {k: float(np.mean(history[k][tail])) for k in (
        "ler_fixed", "ler_rl", "ler_opt", "edr_fixed", "edr_rl", "edr_opt", "tracking_rms")}
    # Steering advantage: fraction of the fixed-vs-optimal EDR gap closed by RL
    s["steering_advantage"] = (s["edr_fixed"] - s["edr_rl"]) / max(1e-5, s["edr_fixed"] - s["edr_opt"])
    s["ler_reduction_pct"] = 100.0 * (s["ler_fixed"] - s["ler_rl"]) / max(1e-6, s["ler_fixed"])
    s["num_updates"] = total_shots // update_shots
    return {"summary": s, "history": history}


def plot_history(history: Dict, title: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array(history["shot"]) / 1000.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    styles = {"fixed": ("#d62728", "-", "Fixed"), "rl": ("#1f77b4", "-", "RL"), "opt": ("#2ca02c", "--", "Optimal")}

    for tag, (c, ls, lab) in styles.items():
        axes[0].plot(x, history[f"edr_{tag}"], color=c, ls=ls, label=lab)
        axes[1].plot(x, history[f"ler_{tag}"], color=c, ls=ls, label=lab)
    axes[0].set_title("Mean detection event rate")
    axes[1].set_title("Logical error rate (PyMatching)")
    axes[2].plot(x, history["p_opt_0"], color="k", ls=":", label="$p^{opt}_0(t)$")
    axes[2].plot(x, history["mu_0"], color="#1f77b4", label="$\\mu_0(t)$")
    axes[2].set_title("Parameter 0 tracking")
    for ax in axes:
        ax.set_xlabel("Physical shots (thousands)")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Plot saved to '{path}'")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Environment
    p.add_argument("--distance", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--params-per-gate", type=int, default=1)
    p.add_argument("--drift", choices=["sinusoidal", "arma"], default="sinusoidal")
    p.add_argument("--drift-frequency", type=float, default=1.0 / 150.0, help="sinusoid cycles per reference epoch")
    p.add_argument("--drift-amplitude", type=float, default=0.6)
    p.add_argument("--arma-alpha", type=float, default=0.67, help="PSD exponent: 0=white, 1=flicker, 2=random walk")
    p.add_argument("--arma-white", type=float, default=0.1, help="fraction of drift variance in the white noise floor")
    p.add_argument("--arma-sigma", type=float, default=0.6)
    p.add_argument("--arma-common", type=float, default=0.08, help="chip-wide common-mode fraction")
    # Training / evaluation
    p.add_argument("--total-shots", type=int, default=245_000)
    p.add_argument("--shots-per-candidate", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=10_000)
    p.add_argument("--eval-shots", type=int, default=800)
    p.add_argument("--seed", type=int, default=163)
    p.add_argument("--plot", action="store_true", help="save a PNG of the run")
    p.add_argument("--out", type=str, default=None, help="JSON output path (default: results_<drift>.json)")
    args = p.parse_args()

    t0 = time.time()
    env = RotatedSurfaceCodeEnv(
        distance=args.distance,
        rounds=args.rounds,
        params_per_gate=args.params_per_gate,
        drift_profile=args.drift,
        drift_frequency=args.drift_frequency,
        drift_amplitude=args.drift_amplitude,
        arma_alpha=args.arma_alpha,
        arma_white_fraction=args.arma_white,
        arma_sigma=args.arma_sigma,
        arma_common_fraction=args.arma_common,
        max_epochs=int(np.ceil(args.total_shots / SHOTS_PER_EPOCH_REF)) + 1,
        seed=args.seed,
    )
    agent = RLControlAgent(
        num_params=env.num_params,
        num_detectors=env.num_detectors,
        mask_matrix=env.M,
        lr_mu=0.05,
        lr_sigma=0.01,
        lr_baseline=0.15,
        entropy_reg=0.005,
        sigma_init=0.25,
        sigma_min=0.08,
        sigma_max=0.35,
        use_adam=True,
        seed=args.seed,
    )

    drift_desc = (f"sinusoidal f={args.drift_frequency:.4g}/epoch A={args.drift_amplitude}" if args.drift == "sinusoidal"
                  else f"1/f^{args.arma_alpha} + {args.arma_white:.0%} white floor, sigma={args.arma_sigma}")
    print(f"d={args.distance} rounds={args.rounds} | {env.num_params} params, {env.num_detectors} detectors, "
          f"M sparsity {1.0 - env.M.mean():.3f} | setup {time.time() - t0:.1f}s")
    print(f"Drift: {drift_desc}")
    print(f"Shot budget {args.total_shots:,} | S={args.shots_per_candidate} x B={args.batch_size} per update\n")

    result = run_trial(
        env, agent,
        total_shots=args.total_shots,
        shots_per_candidate=args.shots_per_candidate,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_shots=args.eval_shots,
    )

    s = result["summary"]
    print(f"\nSteady state (last third of checkpoints), {s['num_updates']} updates, {time.time() - t0:.1f}s total")
    print(f"  LER  fixed {s['ler_fixed']:.4f} | RL {s['ler_rl']:.4f} | opt {s['ler_opt']:.4f}  "
          f"(RL reduction {s['ler_reduction_pct']:.1f}%)")
    print(f"  EDR  fixed {s['edr_fixed']:.4f} | RL {s['edr_rl']:.4f} | opt {s['edr_opt']:.4f}  "
          f"(steering advantage {s['steering_advantage']:.2f})")
    print(f"  Tracking RMS error |mu - p_opt|: {s['tracking_rms']:.3f}")

    out = args.out or f"results_{args.drift}.json"
    with open(out, "w") as f:
        json.dump({"config": vars(args), **result}, f, indent=2)
    print(f"Results saved to '{out}'")

    if args.plot:
        plot_history(result["history"], f"RL drift tracking: {drift_desc}", out.replace(".json", ".png"))


if __name__ == "__main__":
    main()
