"""
Steerability frequency scan with qubit parameters
=================================================
Same scan as `run_frequency_scan` in benchmark_steerability.py (sinusoidal drift, d=5,
30 epochs, same agent settings), but for either code and any parameter mode:
  - gate:  one parameter per gate (the original benchmark)
  - qubit: one parameter per qubit (e.g. its frequency), affecting every gate on it
  - both:  gate + qubit parameters

Each (code, mode, frequency) point is independent, so points can run in parallel:
    python benchmark_steerability_qubit.py scan --code color --mode qubit --freq-index 3
    python benchmark_steerability_qubit.py plot
"""

import argparse
import glob
import json
import os

import numpy as np

from rl_agent import RLControlAgent
from code_envs.surface_code_env import RotatedSurfaceCodeEnv
from code_envs.color_code_env import ColorCodeEnv

SEED = 163
FREQS = [1/1000, 1/500, 1/250, 1/150, 1/100, 1/50, 1/25, 1/10]
CODES = {"surface": RotatedSurfaceCodeEnv, "color": ColorCodeEnv}
MODES = {
    "gate": dict(params_per_gate=1, params_per_qubit=0),
    "qubit": dict(params_per_gate=0, params_per_qubit=1),
    "both": dict(params_per_gate=1, params_per_qubit=1),
}
OUT_DIR = "qubit_steerability_runs"


def scan_point(code: str, mode: str, f: float, lr_mu: float = 0.05) -> dict:
    env = CODES[code](
        distance=5, rounds=5, base_error=0.002, sensitivity=12.5,
        drift_frequency=f, drift_amplitude=0.6, seed=SEED, **MODES[mode],
    )
    agent = RLControlAgent(
        env.num_params, env.num_detectors, env.M,
        lr_mu=lr_mu, lr_sigma=0.01, lr_baseline=0.15, entropy_reg=0.005,
        sigma_init=0.25, sigma_min=0.08, sigma_max=0.35, use_adam=True, seed=SEED,
    )
    fixed_policy = np.zeros(env.num_params, dtype=np.float32)
    ep_ler_f, ep_ler_r, ep_ler_o = [], [], []
    ep_edr_f, ep_edr_r, ep_edr_o = [], [], []
    for ep in range(30):
        cands = agent.sample_policies(batch_size=20, antithetic=True)
        obs = np.array([env.evaluate_policy(c, t=ep, shots=350, decode=False)["edr_vector"] for c in cands])
        agent.update(cands, obs)
        p_opt = env.get_drift_opt(ep)
        ev_f = env.evaluate_policy(fixed_policy, t=ep, shots=800, decode=True)
        ev_r = env.evaluate_policy(agent.mu, t=ep, shots=800, decode=True)
        ev_o = env.evaluate_policy(p_opt, t=ep, shots=800, decode=True)
        ep_ler_f.append(ev_f["logical_error_rate"])
        ep_ler_r.append(ev_r["logical_error_rate"])
        ep_ler_o.append(ev_o["logical_error_rate"])
        ep_edr_f.append(ev_f["mean_edr"])
        ep_edr_r.append(ev_r["mean_edr"])
        ep_edr_o.append(ev_o["mean_edr"])
    mean_lf, mean_lr = np.mean(ep_ler_f), np.mean(ep_ler_r)
    mean_ef, mean_er, mean_eo = np.mean(ep_edr_f), np.mean(ep_edr_r), np.mean(ep_edr_o)
    return {
        "code": code, "mode": mode, "freq": f, "lr_mu": lr_mu,
        "num_gate_params": env.num_gate_params, "num_qubit_params": env.num_qubit_params,
        "ler_red_pct": float((mean_lf - mean_lr) / (mean_lf + 1e-6) * 100.0),
        "steering_r": float((mean_ef - mean_er) / (mean_ef - mean_eo + 1e-6)),
        "ler_fixed": float(mean_lf), "ler_rl": float(mean_lr), "ler_opt": float(np.mean(ep_ler_o)),
        "edr_fixed": float(mean_ef), "edr_rl": float(mean_er), "edr_opt": float(mean_eo),
    }


def plot_lr(out_dir: str, path: str):
    """Steering advantage and LER suppression vs. frequency, one line per lr_mu, one column per mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.load(open(p)) for p in glob.glob(os.path.join(out_dir, "*.json"))]
    lrs = sorted({r["lr_mu"] for r in rows})
    # lr_mu is ordinal, so one hue light -> dark (blue ramp steps 300 / 400 / 550 / 700)
    ramp = ["#6da7ec", "#3987e5", "#1c5cab", "#0d366b"][-len(lrs):] if len(lrs) <= 4 else plt.cm.Blues(np.linspace(0.4, 1, len(lrs)))
    modes = [m for m in MODES if any(r["mode"] == m for r in rows)]
    titles = {"gate": "Gate only", "qubit": "Qubit only", "both": "Gate + qubit"}

    fig, axes = plt.subplots(2, len(modes), figsize=(5.2 * len(modes), 8.5), dpi=150, sharex=True, sharey="row", squeeze=False)
    for j, mode in enumerate(modes):
        for lr, c in zip(lrs, ramp):
            pts = sorted((r for r in rows if r["mode"] == mode and r["lr_mu"] == lr), key=lambda r: r["freq"])
            f = [r["freq"] for r in pts]
            axes[0, j].semilogx(f, [r["steering_r"] for r in pts], color=c, marker="o", ms=5, lw=2, label=f"lr$_\mu$ = {lr:g}")
            axes[1, j].semilogx(f, [r["ler_red_pct"] for r in pts], color=c, marker="o", ms=5, lw=2, label=f"lr$_\mu$ = {lr:g}")
            # Adam slew-rate limit: max step ~ lr_mu per update vs. max drift speed 2 pi f A per update
            f_slew = lr / (2 * np.pi * 0.6)
            if min(f) <= f_slew <= max(f):
                for ax in axes[:, j]:
                    ax.axvline(f_slew, color=c, ls=":", lw=1.4)
        axes[0, j].set_title(titles[mode], fontsize=11, fontweight="bold")
        for ax in axes[:, j]:
            ax.grid(True, ls="--", alpha=0.4)
            ax.axhline(0.0, color="#7f7f7f", ls="--", lw=1.2)
        axes[0, j].axhline(1.0, color="#7f7f7f", ls=":", lw=1.2)
        axes[1, j].set_xlabel("Drift Frequency $f$ (cycles / update)")
    axes[0, 0].set_ylabel("Steering advantage $r$")
    axes[1, 0].set_ylabel("LER suppression (%)")
    axes[0, 0].legend(loc="lower left", fontsize=8, title="dotted: $f^* = $lr$_\mu / 2\pi A$", title_fontsize=8)
    fig.suptitle("Surface code: learning-rate sweep", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved '{path}'")


def plot(out_dir: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.load(open(p)) for p in glob.glob(os.path.join(out_dir, "*.json"))]
    # Fixed categorical order (blue, orange, aqua) plus a marker per mode, so identity isn't color-alone
    styles = {"gate": ("#2a78d6", "o", "Gate only"), "qubit": ("#eb6834", "s", "Qubit only"), "both": ("#1baf7a", "^", "Gate + qubit")}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150, sharex=True)
    for i, code in enumerate(CODES):
        for mode, (c, mk, lab) in styles.items():
            pts = sorted((r for r in rows if r["code"] == code and r["mode"] == mode), key=lambda r: r["freq"])
            if not pts:
                continue
            f = [r["freq"] for r in pts]
            for j, key in enumerate(("ler_red_pct", "steering_r")):
                axes[i, j].semilogx(f, [r[key] for r in pts], color=c, marker=mk, ms=6, lw=2, label=lab)
        for j in range(2):
            ax = axes[i, j]
            ax.grid(True, ls="--", alpha=0.4)
            ax.axhline(0.0, color="#7f7f7f", ls="--", lw=1.2)
            ax.axvline(1 / 130, color="#7f7f7f", ls="-.", lw=1.2, label="Google limit ($f_c \\sim 1/130$)")
            if j == 1:
                ax.axhline(1.0, color="#7f7f7f", ls=":", lw=1.2, label="Optimal tracking ($r=1$)")
            ax.legend(loc="lower left", fontsize=8)
        axes[i, 0].set_ylabel(f"{code.capitalize()} code\nLER suppression (%)")
        axes[i, 1].set_ylabel("Steering advantage $r$")
    axes[0, 0].set_title("Logical Error Suppression vs. Drift Frequency", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Steering Advantage vs. Drift Frequency", fontsize=11, fontweight="bold")
    for ax in axes[1]:
        ax.set_xlabel("Drift Frequency $f$ (cycles / epoch)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved '{path}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["scan", "plot", "plot-lr"])
    p.add_argument("--code", choices=sorted(CODES), default="surface")
    p.add_argument("--mode", choices=sorted(MODES), default="qubit")
    p.add_argument("--freq-index", type=int, default=None, help="run one frequency from FREQS (default: all)")
    p.add_argument("--freq", type=float, default=None, help="run one arbitrary frequency instead")
    p.add_argument("--lr-mu", type=float, default=0.05)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--plot-path", default="qubit_steerability_comparison.png")
    args = p.parse_args()

    if args.action == "plot":
        plot(args.out_dir, args.plot_path)
    elif args.action == "plot-lr":
        plot_lr(args.out_dir, args.plot_path)
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        if args.freq is not None:
            points = [(f"{args.freq:.6g}", args.freq)]
        else:
            points = [(str(i), FREQS[i]) for i in (range(len(FREQS)) if args.freq_index is None else [args.freq_index])]
        for tag, freq in points:
            r = scan_point(args.code, args.mode, freq, lr_mu=args.lr_mu)
            lr_tag = "" if args.lr_mu == 0.05 else f"_lr{args.lr_mu:g}"
            with open(os.path.join(args.out_dir, f"{args.code}_{args.mode}{lr_tag}_f{tag}.json"), "w") as fh:
                json.dump(r, fh, indent=2)
            print(f"{args.code:7s} {args.mode:5s} lr {args.lr_mu:<5g} Freq {r['freq']:7.4f} | LER Red: {r['ler_red_pct']:+6.1f}% | Steering r: {r['steering_r']:5.2f}")
