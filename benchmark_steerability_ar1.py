"""
Steerability scan under AR(1) drift
===================================
Same scan as benchmark_steerability_qubit.py (d=5, 30 epochs, same agent and shots, both codes,
gate / qubit / both parameter modes), but every optimum follows an independent AR(1)
(Ornstein-Uhlenbeck) process instead of a shared sinusoid, and the scan is over its correlation
time tau (epochs). Each point uses a single tau for every parameter (tau_range = (tau, tau)) and a
stationary std matched to the sinusoid's rms (A / sqrt 2 with the default A = 2), so the two
scans move the optimum by the same amount on average.

AR(1) drift is random, so each point is averaged over --seeds drift realizations (the sinusoidal
scan used one).

The top axis is the AR(1) spectral corner frequency f_c = 1 / (2 pi tau), in cycles per epoch:
the drift's power spectrum is a Lorentzian, flat below f_c and falling as 1/f^2 above it. It is a
spectral analog, not an equivalence: analytics/sin_vs_ar1_sweep.py found that the AR(1) tau and
sinusoid period giving the same tracking error scale as tau ~ T^1.3, not by a constant factor.

    python benchmark_steerability_ar1.py scan            # every (code, mode, tau), in parallel
    python benchmark_steerability_ar1.py plot
"""

import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

from benchmark_steerability_qubit import CODES, MODES
from run_experiment import make_agent

SEED = 163
FREQS = [1/1000, 1/500, 1/250, 1/150, 1/100, 1/50, 1/25, 1/10]  # the sinusoidal scan's frequencies
TAUS = [1.0 / (2 * np.pi * f) for f in FREQS]  # so each tau's corner frequency lands on one of them
SIGMA = 2.0 / np.sqrt(2.0)  # stationary std = rms of the default sinusoid (A = 2)
EPOCHS = 30
OUT_DIR = "runs/steerability_ar1"


def scan_point(code: str, mode: str, tau: float, seed: int, decode_every: int = 1) -> dict:
    """
    LER and EDR of the fixed, RL and optimal policies, averaged over the run, for one drift realization.
    EDR is measured every epoch; LER (which needs the decoder) every `decode_every` epochs.
    """
    env = CODES[code](
        distance=5, rounds=5, base_error=0.002, seed=seed, drift_profile="ar1",
        ar1_tau_range=(tau, tau), ar1_sigma_range=(SIGMA, SIGMA), max_epochs=EPOCHS + 1, **MODES[mode],
    )
    agent = make_agent(env, seed)
    fixed_policy = np.zeros(env.num_params, dtype=np.float32)
    keys = ("ler_fixed", "ler_rl", "ler_opt", "edr_fixed", "edr_rl", "edr_opt")
    h = {k: [] for k in keys}
    for ep in range(EPOCHS):
        cands = agent.sample_policies(batch_size=20, antithetic=True)
        obs = np.array([env.evaluate_policy(c, t=ep, shots=350, decode=False)["edr_vector"] for c in cands])
        agent.update(cands, obs)
        decode = ep % decode_every == 0
        for tag, policy in (("fixed", fixed_policy), ("rl", agent.mu), ("opt", env.get_drift_opt(ep))):
            ev = env.evaluate_policy(policy, t=ep, shots=800, decode=decode)
            h[f"edr_{tag}"].append(ev["mean_edr"])
            if decode:
                h[f"ler_{tag}"].append(ev["logical_error_rate"])
    return {"code": code, "mode": mode, "tau": tau, "seed": seed, "decode_every": decode_every,
            **{k: float(np.mean(v)) for k, v in h.items()}}


def _run(job):
    r = scan_point(*job)
    print(f"  {r['code']:7s} {r['mode']:5s} tau {r['tau']:7.2f} seed {r['seed']}: "
          f"EDR fixed {r['edr_fixed']:.4f} RL {r['edr_rl']:.4f} opt {r['edr_opt']:.4f}", flush=True)
    return r


def scan(args):
    seeds = list(range(SEED, SEED + args.seeds))
    jobs = [(code, mode, tau, seed, args.decode_every) for code in args.codes for mode in MODES for tau in TAUS for seed in seeds]
    with Pool(min(args.workers, len(jobs))) as pool:
        rows = pool.map(_run, jobs)
    os.makedirs(args.out_dir, exist_ok=True)
    for code in args.codes:
        with open(os.path.join(args.out_dir, f"{code}.json"), "w") as f:
            json.dump([r for r in rows if r["code"] == code], f, indent=2)


def aggregate(rows: list) -> dict:
    """Per (code, mode, tau): LER suppression and steering advantage from the seed-averaged rates, as in the sinusoidal scan."""
    out = {}
    for key in sorted({(r["code"], r["mode"], r["tau"]) for r in rows}):
        pts = [r for r in rows if (r["code"], r["mode"], r["tau"]) == key]
        m = {k: np.mean([r[k] for r in pts]) for k in ("ler_fixed", "ler_rl", "edr_fixed", "edr_rl", "edr_opt")}
        out[key] = {
            "ler_red_pct": (m["ler_fixed"] - m["ler_rl"]) / (m["ler_fixed"] + 1e-6) * 100.0,
            "steering_r": (m["edr_fixed"] - m["edr_rl"]) / (m["edr_fixed"] - m["edr_opt"] + 1e-6),
        }
    return out


def plot(out_dir: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for p in glob.glob(os.path.join(out_dir, "*.json")) for r in json.load(open(p))]
    agg = aggregate(rows)
    codes = [c for c in CODES if any(r["code"] == c for r in rows)]
    styles = {"gate": ("#2a78d6", "o", "Gate only"), "qubit": ("#eb6834", "s", "Qubit only"), "both": ("#1baf7a", "^", "Gate + qubit")}
    n_seeds = len({r["seed"] for r in rows})

    fig, axes = plt.subplots(len(codes), 2, figsize=(13, 4.6 * len(codes)), dpi=150, sharex=True, squeeze=False)
    for i, code in enumerate(codes):
        for mode, (c, mk, lab) in styles.items():
            taus = sorted(t for (cc, m, t) in agg if cc == code and m == mode)
            if not taus:
                continue
            for j, key in enumerate(("ler_red_pct", "steering_r")):
                axes[i, j].semilogx(taus, [agg[(code, mode, t)][key] for t in taus], color=c, marker=mk, ms=6, lw=2, label=lab)
        for j in range(2):
            ax = axes[i, j]
            ax.grid(True, ls="--", alpha=0.4)
            ax.axhline(0.0, color="#7f7f7f", ls="--", lw=1.2)
            if j == 1:
                ax.axhline(1.0, color="#7f7f7f", ls=":", lw=1.2, label="Optimal tracking ($r=1$)")
            ax.legend(loc="lower right", fontsize=8)
            # Frequency analog on top: the AR(1) corner frequency f_c = 1 / (2 pi tau)
            to_f = lambda x: 1 / (2 * np.pi * np.maximum(x, 1e-12))  # its own inverse; guarded against 0 at the axis edge
            top = ax.secondary_xaxis("top", functions=(to_f, to_f))
            if i == 0:
                top.set_xlabel("Corner frequency $f_c = 1/2\\pi\\tau$ (cycles / epoch)")
        axes[i, 0].set_ylabel(f"{code.capitalize()} code\nLER suppression (%)")
        axes[i, 1].set_ylabel("Steering advantage $r$")
    axes[0, 0].set_title("Logical Error Suppression vs. Correlation Time", fontsize=11, fontweight="bold", pad=10)
    axes[0, 1].set_title("Steering Advantage vs. Correlation Time", fontsize=11, fontweight="bold", pad=10)
    for ax in axes[-1]:
        ax.set_xlabel("AR(1) correlation time $\\tau$ (epochs)")
    fig.suptitle(f"AR(1) drift, stationary std {SIGMA:.2f}, {EPOCHS} epochs, mean of {n_seeds} drift realizations",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved '{path}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["scan", "plot"])
    p.add_argument("--codes", nargs="+", choices=list(CODES), default=list(CODES))
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--decode-every", type=int, default=1,
                   help="decode LER every k epochs (BP-OSD on the color code costs ~30 s per evaluation)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--plot-path", default="figures/steerability/steerability_ar1.png")
    args = p.parse_args()
    if args.action == "scan":
        scan(args)
    plot(args.out_dir, args.plot_path)
