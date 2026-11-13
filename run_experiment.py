"""
RL Drift-Tracking Experiment on a QEC Code
==========================================
Closed-loop PEPG steering of drifting control parameters, driven only by
detector error rates (EDR). Time advances with cumulative physical shots
(a "shot clock"): each policy update costs batch_size * shots_per_candidate
shots, and the drift keeps moving while candidates are being measured.

Compares three policies at periodic checkpoints:
  - Fixed:   parameters frozen at their (already stale) t=0 calibration (mu = 0)
  - RL:      the agent's current policy mean mu(t)
  - Optimal: the ground-truth optimum p_opt(t) (oracle lower bound)
plus the in-situ EDR of the probes the agent actually runs, which pays for its exploration.

Parameters are in tolerance units (see qec_env.py): a gate-parameter mis-set of 1 spends one
error budget, by default doubling that gate's reference error.

Examples:
    python run_experiment.py --drift sinusoidal
    python run_experiment.py --code surface --drift sinusoidal
    python run_experiment.py --drift arma --arma-alpha 0.67
    python run_experiment.py --drift arma --arma-alpha 1.0 --arma-white 0.2 --plot
"""

import argparse
import json
import time
from typing import Dict

import numpy as np

from rl_agent import RLControlAgent
from code_envs.qec_env import DEFAULT_INITIAL_OFFSET, DEFAULT_SENSITIVITY, QECCodeEnv
from code_envs.noise_envs import Ar1Drift, ArmaDrift, SinusoidalDrift
from code_envs.surface_code_env import RotatedSurfaceCodeEnv
from code_envs.color_code_env import ColorCodeEnv

CODES = {"surface": RotatedSurfaceCodeEnv, "color": ColorCodeEnv}

# Reference epoch used to convert shots -> environment time: 20 candidates * 350 shots
SHOTS_PER_EPOCH_REF = 7000.0


def sensitivity_arg(text: str):
    """Parses a sensitivity flag: one Omega scale ("1.5") or one per source kind ("2q=1,meas=2")."""
    if "=" not in text:
        return float(text)
    return {kind.strip(): float(value) for kind, value in (item.split("=") for item in text.split(","))}


# Agent defaults, in tolerance units. A probe of width sigma costs Omega * sigma^2 of extra error on
# every source the parameter touches, so sigma stays well below 1: 0.25 costs ~6%, 0.05 ~0.3%.
AGENT_DEFAULTS = dict(
    lr_mu=0.18,
    lr_sigma=0.02,
    lr_baseline=0.15,
    entropy_reg=0.005,
    sigma_init=0.25,
    sigma_min=0.05,
    sigma_max=0.5,
)


def make_agent(env: QECCodeEnv, seed: int, **overrides) -> RLControlAgent:
    """The masked PEPG agent with the default hyperparameters used by every experiment."""
    return RLControlAgent(
        num_params=env.num_params,
        num_detectors=env.num_detectors,
        mask_matrix=env.M,
        use_adam=True,
        seed=seed,
        **{**AGENT_DEFAULTS, **overrides},
    )


def drift_diagnostics(mu: np.ndarray, p_opt: np.ndarray, sigma: np.ndarray, grad: np.ndarray, offset: np.ndarray,
                      omega: np.ndarray, window: int) -> Dict:
    """
    Per-parameter diagnostics in the spirit of Wu et al. (arXiv:2610.11477, Table I), from the
    agent's trajectory, one row per update: mu, p_opt and sigma after it, its raw gradient estimate
    grad, and the offset mu - p_opt the candidates were measured at.

      - detectability D_i = Delta_i / sigma_i: Delta_i is the rms drift of p_opt_i over one reference
        epoch (`window` updates) and sigma_i the precision with which one epoch of the agent's own
        data locates the optimum. Each gradient estimate responds to its offset as
        g_i = -k_i e_i + noise; a fit past the first third of the run gives
        sigma_i = std(noise) / |k_i| / sqrt(window). D < 1 means an epoch of data can't resolve an
        epoch of drift, so the agent mostly chases noise; a parameter that gives it no signal
        (k_i ~ 0) has D ~ 0.
      - budget fractions R_i = Omega_i <e_i^2> over the last third: the error budgets the tracking
        error e = mu - p_opt spends on each of the parameter's sources, and R_probe_i =
        Omega_i <sigma_i^2>, the extra budgets its exploration spends. Omega_i is in units of the
        error budget for additive parameters and of the base rate for relative ones (times its
        per-kind weight).
    """
    n = len(mu)
    tail = slice(-max(1, n // 3), None)
    err = (mu - p_opt)[tail]
    out = {"tracking_rms": np.sqrt(np.mean(err ** 2, axis=0)),
           "budget_fraction": omega * np.mean(err ** 2, axis=0),
           "budget_fraction_probe": omega * np.mean(sigma[tail] ** 2, axis=0)}
    w = max(1, int(window))
    if n - n // 3 > 2 * w:
        drift = np.sqrt(np.mean((p_opt[w:] - p_opt[:-w]) ** 2, axis=0))
        e, g = offset[n // 3:], grad[n // 3:]
        e, g = e - e.mean(axis=0), g - g.mean(axis=0)
        k = -np.sum(e * g, axis=0) / np.maximum(np.sum(e ** 2, axis=0), 1e-12)
        precision = np.std(g + k * e, axis=0) / np.maximum(np.abs(k), 1e-12) / np.sqrt(w)
        out["detectability"] = drift / precision
        out["gradient_response"] = k
    else:
        out["detectability"] = np.full(mu.shape[1], np.nan)
    return {key: v.tolist() for key, v in out.items()}


def run_trial(
    env: QECCodeEnv,
    agent: RLControlAgent,
    total_shots: int,
    shots_per_candidate: int,
    batch_size: int,
    eval_interval: int,
    eval_shots: int,
    verbose: bool = True,
) -> Dict:
    fixed_policy = np.zeros(env.num_params, dtype=np.float32)
    # Tracking error is also reported per parameter block (gate vs. qubit) that the env has
    blocks = {name: sl for name, sl in (("gate", env.gate_param_slice), ("qubit", env.qubit_param_slice)) if sl.stop > sl.start}
    history = {k: [] for k in (
        "shot", "epoch", "ler_fixed", "ler_rl", "ler_opt",
        "edr_fixed", "edr_rl", "edr_opt", "tracking_rms", "mu_0", "p_opt_0",
        *(f"tracking_rms_{name}" for name in blocks),
    )}

    if verbose:
        header = (f"{'Shot':>9} | {'EDR fixed':>9} | {'EDR RL':>8} | {'EDR opt':>8} | "
                  f"{'LER fixed':>9} | {'LER RL':>8} | {'LER opt':>8} | {'RMS err':>7}")
        print(header)
        print("-" * len(header))

    cur_shot = 0
    next_eval = 0
    update_shots = batch_size * shots_per_candidate
    traj = {k: [] for k in ("mu", "p_opt", "sigma", "grad", "offset", "edr_probe")}  # one row per update

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
            for name, sl in blocks.items():
                history[f"tracking_rms_{name}"].append(float(np.sqrt(np.mean((agent.mu[sl] - p_opt[sl]) ** 2))))
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

        offset = agent.mu - env.get_drift_opt((cur_shot + 0.5 * update_shots) / SHOTS_PER_EPOCH_REF)
        agent.update(cands, obs)
        cur_shot += update_shots
        traj["grad"].append(agent.last_grad_mu)
        traj["offset"].append(offset)
        traj["mu"].append(agent.mu.copy())
        traj["p_opt"].append(env.get_drift_opt(cur_shot / SHOTS_PER_EPOCH_REF))
        traj["sigma"].append(agent.sigma.copy())
        traj["edr_probe"].append(float(obs.mean()))

    # Per-parameter diagnostics over the last third of updates (no extra shots needed)
    if traj["mu"]:
        per_param = drift_diagnostics(*(np.array(traj[k]) for k in ("mu", "p_opt", "sigma", "grad", "offset")),
                                      env.param_sensitivities, window=round(SHOTS_PER_EPOCH_REF / update_shots))
    else:
        per_param = {"tracking_rms": [0.0] * env.num_params}

    # Steady-state summary over the last third of checkpoints
    tail = slice(-max(1, len(history["shot"]) // 3), None)
    s = {k: float(np.mean(history[k][tail])) for k in (
        "ler_fixed", "ler_rl", "ler_opt", "edr_fixed", "edr_rl", "edr_opt", "tracking_rms",
        *(f"tracking_rms_{name}" for name in blocks))}
    # Steering advantage: fraction of the fixed-vs-optimal EDR gap closed by RL
    s["steering_advantage"] = (s["edr_fixed"] - s["edr_rl"]) / max(1e-5, s["edr_fixed"] - s["edr_opt"])
    s["ler_reduction_pct"] = 100.0 * (s["ler_fixed"] - s["ler_rl"]) / max(1e-6, s["ler_fixed"])
    # In situ: the probes the hardware actually ran, so exploration counts against the agent
    probe_tail = traj["edr_probe"][-max(1, len(traj["edr_probe"]) // 3):]
    s["edr_probe"] = float(np.mean(probe_tail)) if probe_tail else float("nan")
    s["steering_advantage_in_situ"] = (s["edr_fixed"] - s["edr_probe"]) / max(1e-5, s["edr_fixed"] - s["edr_opt"])
    s["num_updates"] = total_shots // update_shots
    history["edr_probe"] = traj["edr_probe"]
    return {"summary": s, "history": history, "per_param": per_param,
            "per_param_tracking_rms": per_param["tracking_rms"]}


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


def add_env_arguments(p: argparse.ArgumentParser, code_default: str = "color", codes=tuple(CODES)):
    """Adds the environment flags shared by run_experiment.py and run_streaming_experiment.py (see `build_env`)."""
    p.add_argument("--code", choices=sorted(codes), default=code_default)
    p.add_argument("--distance", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--base-error", type=float, default=0.002, help="base error rate of a scale-1 (2q) source")
    p.add_argument("--params-per-gate", type=int, default=1)
    p.add_argument("--drift", choices=["sinusoidal", "arma", "ar1"], default="sinusoidal")
    p.add_argument("--drift-frequency", type=float, default=1.0 / 150.0, help="sinusoid cycles per reference epoch")
    p.add_argument("--drift-amplitude", type=float, default=2.0, help="in tolerance units")
    p.add_argument("--arma-alpha", type=float, default=0.67, help="PSD exponent: 0=white, 1=flicker, 2=random walk")
    p.add_argument("--arma-white", type=float, default=0.1, help="fraction of drift variance in the white noise floor")
    p.add_argument("--arma-sigma", type=float, default=2.0, help="in tolerance units")
    p.add_argument("--arma-common", type=float, default=0.08, help="chip-wide common-mode fraction")
    p.add_argument("--ar1-tau", type=float, nargs=2, default=[1.0, 50.0],
                   help="AR(1) drift: range of correlation times (epochs), drawn log-uniformly per parameter")
    p.add_argument("--ar1-sigma", type=float, nargs=2, default=[0.3, 1.5],
                   help="AR(1) drift: range of stationary stds (tolerance units), drawn log-uniformly per parameter")
    p.add_argument("--initial-offset", type=float, default=DEFAULT_INITIAL_OFFSET,
                   help="rms offset of the optimum from the t=0 calibration (a stale calibration), in tolerance units")
    p.add_argument("--error-model", choices=["additive", "relative"], default="additive",
                   help="gate parameters. additive: eps = base + budget * Omega e^2; relative: eps = base * (1 + Omega e^2)")
    p.add_argument("--error-budget", type=float, default=0.002,
                   help="error a unit mis-set adds to a 2q gate under the additive model (other kinds scale by SI1000)")
    p.add_argument("--sensitivity", type=sensitivity_arg, default=DEFAULT_SENSITIVITY,
                   help="gate-parameter Omega: one value, or per kind, e.g. 2q=1,meas=2 "
                        "(error budgets spent per unit squared mis-set)")
    p.add_argument("--sensitivity-spread", type=float, default=0.4, help="each Omega_i = type scale * U(1 - s, 1 + s)")
    # Qubit parameters (e.g. idle frequency): each affects every gate and idle on its qubit. Unset drift flags copy the gate ones.
    p.add_argument("--params-per-qubit", type=int, default=1, help="0 = gate parameters only")
    p.add_argument("--qubit-drift", choices=["sinusoidal", "arma", "ar1"], default=None)
    p.add_argument("--qubit-drift-frequency", type=float, default=None)
    p.add_argument("--qubit-drift-amplitude", type=float, default=None)
    p.add_argument("--qubit-arma-alpha", type=float, default=None)
    p.add_argument("--qubit-arma-white", type=float, default=None)
    p.add_argument("--qubit-arma-sigma", type=float, default=None)
    p.add_argument("--qubit-arma-common", type=float, default=None)
    p.add_argument("--qubit-sensitivity", type=sensitivity_arg, default=None,
                   help="qubit-parameter Omega on each source kind it touches, e.g. 2q=1,idle=2.5,idle_mr=1 "
                        "(default: same as --sensitivity)")
    p.add_argument("--qubit-error-model", choices=["additive", "relative"], default="relative")
    p.add_argument("--qubit-initial-offset", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=163)


def build_env(args, max_epochs: int) -> QECCodeEnv:
    """
    The env described by `add_env_arguments` flags. Also takes the saved "config" of a
    results JSON (as an argparse.Namespace); flags missing from older configs get their defaults.
    """
    get = lambda name, default=None: getattr(args, name, default)
    return CODES[args.code](
        distance=args.distance,
        rounds=args.rounds,
        base_error=get("base_error", 0.002),
        params_per_gate=get("params_per_gate", 1),
        sensitivity=get("sensitivity", DEFAULT_SENSITIVITY),
        sensitivity_spread=get("sensitivity_spread", 0.4),
        drift_profile=args.drift,
        drift_frequency=get("drift_frequency", 1.0 / 150.0),
        drift_amplitude=get("drift_amplitude", 2.0),
        arma_alpha=get("arma_alpha", 0.67),
        arma_white_fraction=get("arma_white", 0.1),
        arma_sigma=get("arma_sigma", 2.0),
        arma_common_fraction=get("arma_common", 0.08),
        ar1_tau_range=tuple(get("ar1_tau", (1.0, 50.0))),
        ar1_sigma_range=tuple(get("ar1_sigma", (0.3, 1.5))),
        max_epochs=max_epochs,
        seed=args.seed,
        params_per_qubit=get("params_per_qubit", 1),
        qubit_drift_profile=get("qubit_drift"),
        qubit_drift_frequency=get("qubit_drift_frequency"),
        qubit_drift_amplitude=get("qubit_drift_amplitude"),
        qubit_arma_alpha=get("qubit_arma_alpha"),
        qubit_arma_white_fraction=get("qubit_arma_white"),
        qubit_arma_sigma=get("qubit_arma_sigma"),
        qubit_arma_common_fraction=get("qubit_arma_common"),
        qubit_sensitivity=get("qubit_sensitivity"),
        error_model=get("error_model", "additive"),
        qubit_error_model=get("qubit_error_model", "relative"),
        error_budget=get("error_budget", 0.002),
        initial_offset=get("initial_offset", DEFAULT_INITIAL_OFFSET),
        qubit_initial_offset=get("qubit_initial_offset"),
    )


def describe_drift(d) -> str:
    """One-line description of a drift model, for printouts and plot titles."""
    if isinstance(d, SinusoidalDrift):
        return f"sinusoidal f={d.frequency:.4g}/epoch A={d.amplitude}"
    if isinstance(d, ArmaDrift):
        return f"1/f^{d.alpha} + {d.white_fraction:.0%} white floor, sigma={d.sigma}"
    if isinstance(d, Ar1Drift):
        return f"AR(1) tau {d.tau_range[0]:g}-{d.tau_range[1]:g} epochs, sigma {d.sigma_range[0]:g}-{d.sigma_range[1]:g}"
    return type(d).__name__


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_arguments(p)
    # Training / evaluation
    p.add_argument("--total-shots", type=int, default=245_000)
    p.add_argument("--shots-per-candidate", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=10_000)
    p.add_argument("--eval-shots", type=int, default=800)
    p.add_argument("--plot", action="store_true", help="save a PNG of the run")
    p.add_argument("--out", type=str, default=None, help="JSON output path (default: results_<drift>.json)")
    args = p.parse_args()

    t0 = time.time()
    env = build_env(args, max_epochs=int(np.ceil(args.total_shots / SHOTS_PER_EPOCH_REF)) + 1)
    agent = make_agent(env, seed=args.seed)

    drift_desc = describe_drift(env.drift)
    print(f"{args.code} code d={args.distance} rounds={args.rounds} | {env.num_params} params "
          f"({env.num_gate_params} gate + {env.num_qubit_params} qubit), {env.num_detectors} detectors, "
          f"M sparsity {1.0 - env.M.mean():.3f} | setup {time.time() - t0:.1f}s")
    print(f"Drift: {drift_desc}" + (f" | qubit drift: {describe_drift(env.qubit_drift)}" if env.num_qubit_params else ""))
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
    print(f"  EDR  in situ (probes) {s['edr_probe']:.4f}  (in-situ steering advantage {s['steering_advantage_in_situ']:.2f})")
    pp = result["per_param"]
    if "detectability" in pp:
        D = np.array(pp["detectability"])
        print(f"  Per parameter: detectability D median {np.nanmedian(D):.2f} ({np.mean(D < 1):.0%} below 1) | "
              f"budget fraction R tracking {np.mean(pp['budget_fraction']):.3f}, probing {np.mean(pp['budget_fraction_probe']):.3f}")
    print(f"  Tracking RMS error |mu - p_opt|: {s['tracking_rms']:.3f}"
          + "".join(f" | {name} {s[f'tracking_rms_{name}']:.3f}" for name in ("gate", "qubit") if f"tracking_rms_{name}" in s))

    out = args.out or f"results_{args.drift}.json"
    with open(out, "w") as f:
        json.dump({"config": vars(args), **result}, f, indent=2)
    print(f"Results saved to '{out}'")

    if args.plot:
        plot_history(result["history"], f"RL drift tracking: {drift_desc}", out.replace(".json", ".png"))


if __name__ == "__main__":
    main()
