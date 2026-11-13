"""
Signal Share: why higher-weight codes are harder to steer
=========================================================
A detector fires on the XOR of every error source in its detecting region. Higher-weight
checks (the 6.6.6 color code's weight-6 plaquettes vs. the surface code's weight-4) pool more
sources into each detector, so each detector fires more often and any one control parameter
causes a smaller fraction of what it sees. The local quantity is a parameter's signal share

    s_i = sum_{j in M_i} dq_j/d(Omega_i e_i^2)  /  sum_{j in M_i} q_j

the fraction of the detection events in its own detecting region M_i that a unit mis-set of
parameter i adds, with q_j the detector firing rates at the calibrated (base) error rates and
dq_j the first-order change, both read off stim's detector error model. It is a property of
the circuit alone, computed before any RL is run.

The claim, tested in two ways on the surface and color codes (distance 5, 5 rounds, SI1000
relative noise, gate + qubit parameters):

  (b) drift tracking: the per-parameter tracking RMS from analytics/per_param_tracking.py
      falls on one curve against s_i for both codes, and a "which code" term adds nothing
      once s_i is known
  (c, d) static calibration: no drift, every optimum mis-set by N(0, offset^2), the agent
      starts at p = 0. Per-parameter |mu - p_opt| against cumulative shots gives the
      learning speed (shots to halve the error) and the floor (error at the end), each
      against s_i

    python -m analytics.signal_share --seeds 6
    python -m analytics.signal_share --replot
"""

import argparse
import json
import os
import time
import types

import numpy as np
from scipy.stats import spearmanr

from analytics.per_param_tracking import CODES, make_env
from run_experiment import make_agent

TRACKING_JSON = "runs/per_param/per_param_tracking_f0.0067_si1000.json"


# ---------------------------------------------------------------------- signal share (circuit only)

def detector_rates(env, eps: np.ndarray) -> np.ndarray:
    """First-order firing probability of every detector, sum of the DEM error probabilities that flip it."""
    dem = env.backend.build_circuit(eps).detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    q = np.zeros(env.num_detectors)
    for inst in dem.flattened():
        if inst.type == "error":
            p = inst.args_copy()[0]
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    q[t.val] += p
    return q


def signal_features(env) -> dict:
    """Per parameter: signal share s_i and masked SNR per shot; per detector: fan-in and base firing rate."""
    eps0 = env.param_base_errors.astype(np.float64)
    q0 = detector_rates(env, eps0)
    M = env.M > 0
    share, snr = np.zeros(env.num_params), np.zeros(env.num_params)
    for i in range(env.num_params):
        # Omega_i is divided out so the share is pure geometry and base rates
        dq = detector_rates(env, eps0 + env.curvature[:, i] / env.param_sensitivities[i]) - q0
        share[i] = dq[M[:, i]].sum() / q0[M[:, i]].sum()
        snr[i] = dq[M[:, i]].sum() / np.sqrt(q0[M[:, i]].sum())
    return {"signal_share": share.tolist(), "masked_snr": snr.tolist(),
            "detector_fan_in": M.sum(axis=1).tolist(), "detector_rate": q0.tolist()}


# ---------------------------------------------------------------------- static calibration

def calibrate(job) -> dict:
    """One static calibration run; returns |mu - p_opt| per parameter at every checkpoint."""
    code, seed, args = job
    env = CODES[code](distance=args.distance, rounds=args.rounds, params_per_qubit=1,
                      drift_amplitude=0.0, initial_offset=args.offset, error_model="relative", seed=seed)
    agent = make_agent(env, seed)
    p_opt = env.get_drift_opt(0.0)
    update_shots = args.batch_size * args.shots_per_candidate
    shots, err = [0], [np.abs(agent.mu - p_opt)]
    for u in range(args.total_shots // update_shots):
        cands = agent.sample_policies(batch_size=args.batch_size, antithetic=True)
        obs = np.array([env.evaluate_policy(c, t=0.0, shots=args.shots_per_candidate)["edr_vector"] for c in cands])
        agent.update(cands, obs)
        if (u + 1) % args.checkpoint_every == 0:
            shots.append((u + 1) * update_shots)
            err.append(np.abs(agent.mu - p_opt))
    print(f"  {code:7s} seed={seed}: RMS {np.sqrt(np.mean(err[0] ** 2)):.2f} -> {np.sqrt(np.mean(err[-1] ** 2)):.3f}", flush=True)
    return {"shots": shots, "abs_err": np.array(err).tolist(), "offset": (-p_opt).tolist()}


def shots_to_fraction(shots: np.ndarray, err: np.ndarray, start: np.ndarray, frac: float) -> np.ndarray:
    """First checkpoint at which each parameter's (seed-averaged) error falls below frac x its start; nan if never."""
    below = err <= frac * start[None, :]
    first = np.where(below.any(axis=0), below.argmax(axis=0), -1)
    return np.where(first >= 0, shots[np.maximum(first, 0)], np.nan)


def run(args) -> dict:
    from multiprocessing import Pool

    seeds = list(range(163, 163 + args.seeds))
    jobs = [(code, seed, args) for code in CODES for seed in seeds]
    with Pool(min(args.workers, len(jobs))) as pool:
        runs = pool.map(calibrate, jobs)

    with open(TRACKING_JSON) as f:
        tracking = json.load(f)
    targs = types.SimpleNamespace(**tracking["config"])

    out = {}
    for c, code in enumerate(CODES):
        env = make_env(code, 1, seeds[0], targs)  # M and base-rate geometry are fixed across seeds
        feats = signal_features(env)
        per_seed = runs[c * len(seeds):(c + 1) * len(seeds)]
        shots = np.array(per_seed[0]["shots"])
        err = np.array([r["abs_err"] for r in per_seed])  # (seeds, checkpoints, params)
        rms_curve = np.sqrt((err ** 2).mean(axis=0))       # per parameter, rms over seeds
        n_tail = max(1, len(shots) // 5)
        track = next(r for r in tracking["runs"] if r["code"] == code and r["params_per_qubit"] == 1)
        out[code] = {
            "num_detectors": env.num_detectors,
            "type": track["type"],
            **feats,
            "tracking_rms": track["tracking_rms"],
            "shots": shots.tolist(),
            "calib_rms_curve": rms_curve.tolist(),
            "calib_floor": rms_curve[-n_tail:].mean(axis=0).tolist(),
            "shots_to_half": shots_to_fraction(shots, rms_curve, rms_curve[0], 0.5).tolist(),
        }
    return out


# ---------------------------------------------------------------------- summary & plot

def pooled_fit(data: dict, y_key: str):
    """log y = a + b log s (+ c * is_color): slope b, the code term exp(c), and R^2 with and without it."""
    s = np.concatenate([data[c]["signal_share"] for c in CODES])
    y = np.concatenate([data[c][y_key] for c in CODES])
    is_color = np.concatenate([[c == "color"] * len(data[c]["signal_share"]) for c in CODES]).astype(float)
    ok = np.isfinite(y) & (y > 0)
    x, y, is_color = np.log(s[ok]), np.log(y[ok]), is_color[ok]
    A = np.c_[np.ones_like(x), x]
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    b2, *_ = np.linalg.lstsq(np.c_[A, is_color], y, rcond=None)
    r2 = lambda A_, b_: 1 - ((y - A_ @ b_) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"intercept": b[0], "slope": b[1], "r2": r2(A, b), "code_factor": np.exp(b2[2]),
            "r2_with_code": r2(np.c_[A, is_color], b2), "rho": spearmanr(x, y).statistic, "n": int(ok.sum())}


def summarize(data: dict):
    for code in CODES:
        d = data[code]
        print(f"{code:7s}: {d['num_detectors']} detectors, fan-in median {np.median(d['detector_fan_in']):.0f}, "
              f"detector rate median {np.median(d['detector_rate']):.3f}, signal share median {np.median(d['signal_share']):.3f}, "
              f"tracking RMS {np.sqrt(np.mean(np.square(d['tracking_rms']))):.3f}, "
              f"calib floor {np.sqrt(np.mean(np.square(d['calib_floor']))):.3f}, "
              f"shots to half (median) {np.nanmedian(d['shots_to_half']):,.0f} "
              f"({np.mean(np.isnan(d['shots_to_half'])):.0%} never)")
    for key in ("tracking_rms", "calib_floor", "shots_to_half"):
        f = pooled_fit(data, key)
        print(f"  {key:13s} ~ s^{f['slope']:+.2f}  rho {f['rho']:+.2f}  R2 {f['r2']:.2f}  "
              f"code term x{f['code_factor']:.2f} (R2 {f['r2_with_code']:.2f})  n={f['n']}")


STYLE = {"surface": dict(color="C0", marker="o", label="Surface code"),
         "color": dict(color="C1", marker="s", label="Color code")}


def plot(data: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))

    # (a) per-parameter signal share, one strip per code, with its median
    ax = axes[0]
    rng = np.random.default_rng(0)
    for k, code in enumerate(CODES):
        s = np.array(data[code]["signal_share"])
        ax.scatter(s, k + rng.uniform(-0.18, 0.18, len(s)), s=14, alpha=0.6, color=STYLE[code]["color"],
                   marker=STYLE[code]["marker"])
        med = np.median(s)
        ax.plot([med, med], [k - 0.3, k + 0.3], color="black", lw=2, label="median" if k == 0 else None)
    ax.set_xscale("log")
    ax.set_yticks([0, 1], ["Surface", "Color"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Signal share $s_i$")
    ax.set_title("Signal share per parameter")
    ax.legend(loc="center left")
    ax.grid(True, alpha=0.3)

    # (b), (c), (d): one outcome each against signal share, both codes, with one power-law fit to both
    panels = [("tracking_rms", "Drift tracking error", "Tracking RMS $|\\mu_i - p^{opt}_i|$"),
              ("calib_floor", "Static calibration floor", "Final $|\\mu_i - p^{opt}_i|$ (no drift)"),
              ("shots_to_half", "Shots to learn", "Shots to halve the initial mis-set")]
    for ax, (key, title, ylabel) in zip(axes[1:], panels):
        for code in CODES:
            d = data[code]
            y = np.array(d[key], dtype=float)
            ok = np.isfinite(y)
            ax.scatter(np.array(d["signal_share"])[ok], y[ok], s=16, alpha=0.6, **STYLE[code])
        f = pooled_fit(data, key)
        xs = np.geomspace(min(min(data[c]["signal_share"]) for c in CODES), max(max(data[c]["signal_share"]) for c in CODES))
        ax.plot(xs, np.exp(f["intercept"]) * xs ** f["slope"], color="black", lw=1.5,
                label=f"fit $\\propto s^{{{f['slope']:.2f}}}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Signal share $s_i$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Figure saved to '{path}'")


def plot_by_type(data: dict, path: str):
    """Signal share per parameter, one strip per code, styled like per_param_tracking: color = type, marker = bulk / boundary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from analytics.per_param_tracking import COLORS, TYPES

    with open(TRACKING_JSON) as f:  # bulk / boundary flags, in the same parameter order
        boundary = {r["code"]: np.array(r["boundary"]) for r in json.load(f)["runs"] if r["params_per_qubit"] == 1}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    rng = np.random.default_rng(0)
    for k, code in enumerate(CODES):
        s, kind = np.array(data[code]["signal_share"]), np.array(data[code]["type"])
        y = k + rng.uniform(-0.2, 0.2, len(s))
        for t in TYPES:
            for b, marker in ((False, "o"), (True, "^")):
                sel = (kind == t) & (boundary[code] == b)
                ax.scatter(s[sel], y[sel], marker=marker, color=COLORS[t], s=18, alpha=0.8)
        med = np.median(s)
        ax.plot([med, med], [k - 0.32, k + 0.32], color="black", lw=2)
    ax.set_xscale("log")
    ax.set_yticks([0, 1], ["Surface", "Color"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Signal share $s_i$")
    ax.set_title("Signal share per parameter")
    ax.grid(True, alpha=0.3)

    handles = [Line2D([], [], ls="", marker="o", color=COLORS[t], label=f"{t} parameter") for t in TYPES]
    handles += [Line2D([], [], ls="", marker="o", color="gray", label="bulk"),
                Line2D([], [], ls="", marker="^", color="gray", label="boundary"),
                Line2D([], [], color="black", lw=2, label="median")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Figure saved to '{path}'")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distance", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--offset", type=float, default=1.5, help="rms static mis-set the calibration starts from")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--total-shots", type=int, default=300_000)
    p.add_argument("--shots-per-candidate", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--checkpoint-every", type=int, default=5, help="updates between checkpoints")
    p.add_argument("--out", type=str, default="runs/per_param/signal_share.json")
    p.add_argument("--fig", type=str, default="figures/per_param/signal_share.png")
    p.add_argument("--replot", action="store_true", help="redraw the figure from --out without rerunning")
    args = p.parse_args()

    if args.replot:
        with open(args.out) as f:
            data = json.load(f)["codes"]
    else:
        t0 = time.time()
        data = run(args)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "codes": data}, f)
        print(f"Results saved to '{args.out}' ({time.time() - t0:.0f}s)")
    summarize(data)
    plot(data, args.fig)
    plot_by_type(data, args.fig.replace(".png", "_by_type.png"))


if __name__ == "__main__":
    main()
