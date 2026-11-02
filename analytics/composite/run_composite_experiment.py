"""
Trackers under Composite Drift (minimal Gaussian model, long time)
==================================================================
One phase-type control parameter per replica (independent chips), driven by the composite
drift of composite_drift.py: 1/f background, telegraph fluctuators, diffusing TLS (dispersive
shifts and T1 drops), TLS scrambling, high-energy bursts, and a diurnal cycle. One candidate is
a measurement of 0.1 s (Google's QEC time per candidate without dead time).

Reading (gaussian model of MinimalControlEnv, with the base error scaled by eps0_mult = m):

    y = m y0 + H_l min(e^2, e_cap^2) + sqrt(m) sigma_1 / sqrt(S) n,   e_cap^2 = (0.15 - m eps0) / Omega

so the gate error is capped at 0.15 and the detection-event noise grows with the error rate.
The trackers do not know m.

Controllers, tuned on a separate training day of the full model at strength 1:
  PEPG          plain antithetic SGD (B = 20), gain tuned for the lowest tracking error
  AR(1)         Kalman tracker (B = 20), tau minimizing the tracking error (equivalently the mean
                detection rate; the innovation likelihood is hijacked by the unmodeled eps0 jumps)
  AR(1) B=2     the same filter re-centering mu after every antithetic pair
  ARMA(K,K-1)   Kalman tracker (B = 20) with a sum of K AR(1)s fitted to the autocovariance
                of the training drift

Evaluation (24 h per condition after a 1 h burn-in): each component alone, and all components
at strengths 0.5, 1, 2 and 4 (amplitudes scaled, rates unchanged).
c = E[(mu - p*)^2] in parameter units^2 (one unit adds Omega = 0.025 to the gate error);
c / c_fixed compares with leaving the parameter at 0.

Example:
    python -m analytics.composite.run_composite_experiment --plot
    python -m analytics.composite.run_composite_experiment --replot
"""

import argparse
import json
import os
import time
from typing import Dict

import numpy as np

from analytics.composite.composite_drift import COMPONENTS, CompositeDrift
from analytics.minimal_control_env import MinimalControlEnv
from analytics.run_filter_experiment import (GAINS, PEPGController, _plain_log, _pyplot, _save, _style_axes, make_kf,
                                             run_closed_loop)
from kalman_tracker import OUSumModel, fit_multi_ar1

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")

CANDIDATE_S = 0.1
METHODS = ["PEPG", "AR(1)", "AR(1) B=2", "Multi-AR(1)"]
STYLE = {"PEPG": ("#eb6834", "s", "-"), "AR(1)": ("#eda100", "v", "--"),
         "AR(1) B=2": ("#e87ba4", "^", ":"), "Multi-AR(1)": ("#1baf7a", "D", "--")}
ABLATION = [("background",), ("telegraph",), ("tls",), ("tls", "scrambling"), ("bursts",), ("diurnal",), COMPONENTS]
STRENGTHS = [0.5, 1.0, 2.0, 4.0]


def make_data(args, components, strength, seed, hours):
    d = CompositeDrift(duration_s=3600.0 * hours, candidate_s=CANDIDATE_S,
                       shots_per_candidate=args.shots_per_candidate, components=components, strength=strength,
                       chip_wide=False, seed=seed).generate(args.replicas)
    d["noise"] = np.random.default_rng(seed + 7).standard_normal(d["p_star"].shape)
    return d


def reader(env, d, S):
    eps0, omega = float(env.eps0[0]), float(env.omega[0])
    m_all, noise = d["eps0_mult"], d["noise"]

    def read(k, e):
        m = m_all[k].astype(float)
        e_cap2 = np.maximum(0.15 - m * eps0, 0.0) / omega
        return m * env.y0 + env.H_l * np.minimum(e ** 2, e_cap2) + np.sqrt(m) * env.sigma_1 / np.sqrt(S) * noise[k]
    return read


def empirical_cov(p: np.ndarray):
    """Autocovariance of the training drift (averaged over replicas) as a function of candidate lag."""
    x = p - p.mean(axis=0)
    n = len(x)
    f = np.fft.rfft(x, n=2 * n, axis=0)
    ac = np.fft.irfft(np.abs(f) ** 2, axis=0)[:n].mean(axis=1) / (n - np.arange(n))
    return lambda lags: np.interp(lags, np.arange(n), ac)


def tune(env, args) -> Dict:
    """AR(1) tau by likelihood, multi-AR(1) by autocovariance fit, PEPG gain by tracking error,
    all on a training day of the full model at strength 1 (time unit: seconds)."""
    S, B, sp, R = args.shots_per_candidate, args.batch_size, args.sigma_p, args.replicas
    d = make_data(args, COMPONENTS, 1.0, args.seed, args.train_hours)
    p, t = d["p_star"].astype(float), d["t"]
    read, burn = reader(env, d, S), min(int(3600 / CANDIDATE_S), len(p) // 4)
    var0 = float(p.var())
    run = lambda ctrl, b=B: run_closed_loop(ctrl, p, t, read, b, burn)
    # tau by the lowest tracking error (= lowest mean detection rate, observable on hardware), as the
    # PEPG gain. The innovation likelihood is recorded too: unmodeled eps0 jumps make it favor tau -> 0.
    scan = {}
    for tau in np.geomspace(1.0, 1e5, 11):
        scan[tau] = run(make_kf(env, OUSumModel([tau], [var0]), R, S, sp, args.seed))
        print(f"  AR(1) tau {tau:9.1f} s: c {scan[tau]['c']:.4f}  mean log-likelihood {scan[tau]['loglik']:.4f}")
    best = min(scan, key=lambda x: scan[x]["c"])
    for tau in (best / 1.8, best / 1.34, best * 1.34, best * 1.8):
        scan[tau] = run(make_kf(env, OUSumModel([tau], [var0]), R, S, sp, args.seed))
    tau = min(scan, key=lambda x: scan[x]["c"])
    tau_ml = max(scan, key=lambda x: scan[x]["loglik"])
    m_taus, m_vars = fit_multi_ar1(empirical_cov(p), 1.0 / CANDIDATE_S, np.geomspace(0.5, 3 * 3600.0, 7),
                                   len(p) / 4)
    gain_c = {g: run(PEPGController(R, env.H_l, g, sp, args.seed))["c"] for g in GAINS}
    gain = min(gain_c, key=gain_c.get)
    print(f"  -> AR(1) tau = {tau:.1f} s (likelihood would pick {tau_ml:.1f} s; var {var0:.3f}), multi-AR(1) tau {np.round(m_taus, 1).tolist()} s, "
          f"PEPG gain {gain}")
    return {"tau": float(tau), "tau_ml": float(tau_ml), "var0": var0, "multi_taus": m_taus.tolist(), "multi_vars": m_vars.tolist(),
            "gain": float(gain), "gain_scan": {str(k): v for k, v in gain_c.items()}}


def evaluate(env, args, tuned, components, strength, seed, trace=False) -> Dict:
    S, B, sp, R = args.shots_per_candidate, args.batch_size, args.sigma_p, args.replicas
    d = make_data(args, components, strength, seed, args.hours)
    p, t = d["p_star"].astype(float), d["t"]
    read, burn = reader(env, d, S), min(int(3600 / CANDIDATE_S), len(p) // 4)
    ar1 = lambda: OUSumModel([tuned["tau"]], [tuned["var0"]])
    ctrls = {
        "PEPG": (PEPGController(R, env.H_l, tuned["gain"], sp, args.seed + 1), B),
        "AR(1)": (make_kf(env, ar1(), R, S, sp, args.seed + 1), B),
        "AR(1) B=2": (make_kf(env, ar1(), R, S, sp, args.seed + 1), 2),
        "Multi-AR(1)": (make_kf(env, OUSumModel(tuned["multi_taus"], tuned["multi_vars"]), R, S, sp, args.seed + 1), B),
    }
    c_fixed = float(np.mean(p[burn:] ** 2))
    out = {"components": list(components), "strength": strength, "c_fixed": c_fixed,
           "p_std": float(p.std()), "eps0_mult_mean": float(d["eps0_mult"].mean()), "results": {}}
    traces = {"t": t, "p_star": d["p_star"][:, 0], "eps0_mult": d["eps0_mult"][:, 0]}
    for name, (ctrl, b) in ctrls.items():
        r = run_closed_loop(ctrl, p, t, read, b, burn, trace=0 if trace else None)
        if trace:
            traces[name] = r.pop("mu_trace").astype(np.float32)
        r["c_ratio"] = r["c"] / c_fixed if c_fixed > 0 else float("nan")  # bursts alone leave p* = 0
        out["results"][name] = r
    return out, traces


def print_row(r: Dict):
    label = "+".join(r["components"]) if len(r["components"]) < len(COMPONENTS) else "all"
    print(f"  {label:<28} x{r['strength']:<3g} | p* std {r['p_std']:.3f} | c_fixed {r['c_fixed']:.4f} | " + " | ".join(
        f"{n} {r['results'][n]['c']:.4f} ({r['results'][n]['c_ratio']:.2f}) p99 {r['results'][n]['e_p99']:.2f}"
        for n in METHODS))


# ------------------------------------------------------------------ plots
def label_of(name, out):
    k = len(out["tuned"]["multi_taus"])
    return f"ARMA({k},{k - 1})" if name == "Multi-AR(1)" else name


def plot_ablation(out):
    plt = _pyplot()
    rows = out["ablation"]
    names = ["background", "telegraph", "TLS", "TLS +\nscrambling", "bursts", "diurnal", "all"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(x, [r["c_fixed"] if r["c_fixed"] > 0 else np.nan for r in rows], "_", color="#888888", ms=28, mew=2,
                 label="no steering (μ = 0)")
    for j, n in enumerate(METHODS):
        color, marker, _ = STYLE[n]
        dx = (j - 1.5) * 0.12
        axes[0].plot(x + dx, [r["results"][n]["c"] for r in rows], marker, color=color, ms=8, label=label_of(n, out))
        axes[1].plot(x + dx, [r["results"][n]["e_p99"] for r in rows], marker, color=color, ms=8)
    axes[0].set_ylabel(r"tracking error $E[(\mu - p^*)^2]$ (units²)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("99th percentile |μ − p*| (units)")
    for ax in axes:
        ax.set_xticks(x, names)
        _style_axes(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("Each drift component alone (strength 1, controllers tuned on the full model)")
    _save(plt, fig, "composite_ablation.png", FIG_DIR)


def plot_strength(out):
    plt = _pyplot()
    rows = out["strength"]
    x = [r["strength"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for n in METHODS:
        color, marker, ls = STYLE[n]
        axes[0].plot(x, [r["results"][n]["c_ratio"] for r in rows], marker=marker, ls=ls, color=color, lw=2, ms=8,
                     label=label_of(n, out))
        axes[1].plot(x, [r["results"][n]["e_p99"] for r in rows], marker=marker, ls=ls, color=color, lw=2, ms=8)
    axes[0].axhline(1.0, color="#888888", lw=1, ls="--")
    axes[0].set_ylabel("tracking error $c / c_{fixed}$")
    axes[1].set_ylabel("99th percentile |μ − p*| (units)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [f"{v:g}×" for v in x])
        ax.axvline(1.0, color="#cccccc", lw=6, zorder=0)
        ax.set_xlabel("amplitude of every component (tuned at 1×, shaded)")
        _style_axes(ax)
        _plain_log(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("All components, scaled beyond the tuning regime")
    _save(plt, fig, "composite_strength.png", FIG_DIR)


def plot_example(out, traces):
    plt = _pyplot()
    t = traces["t"] / 3600.0
    n = min(len(traces[m]) for m in METHODS)
    t, p_star, eps = t[:n], traces["p_star"][:n], traces["eps0_mult"][:n]
    per_min = int(60 / CANDIDATE_S)
    binned = lambda x: x[: len(x) // per_min * per_min].reshape(-1, per_min).mean(axis=1)
    tb = binned(t)
    pb = binned(p_star)
    b0 = min(60, len(pb) // 4)  # skip the 1 h burn-in
    k = (b0 + int(np.argmax(np.abs(np.diff(pb[b0:]))))) * per_min  # largest change of the 1-minute means
    win = slice(max(0, k - int(300 / CANDIDATE_S)), min(n, k + int(900 / CANDIDATE_S)))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 3, 1.6]})
    for ax, sl, tt, tr in ((axes[0], slice(None), tb, binned), (axes[1], win, t[win] * 60, lambda x: x)):
        ax.plot(tt, tr(p_star[sl]), color="#222222", lw=1.5, label="p* (optimum)")
        for m in METHODS:
            ax.plot(tt, tr(traces[m][:n][sl]), color=STYLE[m][0], ls=STYLE[m][2], lw=1.5, label=label_of(m, out))
        ax.set_ylabel("parameter (units)")
        _style_axes(ax)
    axes[0].axvspan(t[win][0], t[win][-1], color="#dddddd", zorder=0)
    axes[0].set_title("One replica, all components at strength 1: full day (1-minute means; zoom window shaded)")
    axes[0].legend(frameon=False, ncol=5, loc="upper left")
    axes[0].set_xlabel("time (h)")
    axes[1].set_title("Zoom on the largest change of the 1-minute means (every candidate, 0.1 s)")
    axes[1].set_xlabel("time (min)")
    axes[2].plot(tb, eps[: len(eps) // per_min * per_min].reshape(-1, per_min).max(axis=1), color="#555555", lw=1)
    _plain_log(axes[2])
    axes[2].set_ylabel("ε₀ multiplier\n(1-min max)")
    axes[2].set_xlabel("time (h)")
    _style_axes(axes[2])
    _save(plt, fig, "composite_example.png", FIG_DIR)


def replot(path):
    with open(path) as f:
        out = json.load(f)
    plot_ablation(out)
    plot_strength(out)
    tr = np.load(path.replace(".json", "_trace.npz"))
    plot_example(out, {k: tr[k] for k in tr.files})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hours", type=float, default=24.0, help="evaluation run length")
    p.add_argument("--train-hours", type=float, default=6.0, help="training run length")
    p.add_argument("--replicas", type=int, default=16)
    p.add_argument("--shots-per-candidate", type=int, default=50, help="S")
    p.add_argument("--batch-size", type=int, default=20, help="B, candidates per epoch")
    p.add_argument("--sigma-p", type=float, default=0.25, help="probe rms (units)")
    p.add_argument("--seed", type=int, default=163)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--replot", action="store_true", help="only redraw the plots from --out")
    p.add_argument("--out", type=str, default=os.path.join(RES_DIR, "composite_results.json"))
    args = p.parse_args()
    if args.replot:
        replot(args.out)
        return

    t0 = time.time()
    env = MinimalControlEnv(mode="gaussian", seed=args.seed)
    print(f"Minimal model: y0={env.y0:.4f} H_l={env.H_l:.4f} sigma_1={env.sigma_1:.4f} eps0={env.eps0[0]:.4f} "
          f"Omega={env.omega[0]:.4f} | candidate {CANDIDATE_S} s, S={args.shots_per_candidate}, {args.replicas} replicas, "
          f"{args.hours:g} h per condition\n")
    print(f"=== Tuning on {args.train_hours:g} h of the full model (strength 1) ===")
    out = {"config": vars(args), "tuned": tune(env, args), "ablation": [], "strength": []}
    print(f"  ({time.time() - t0:.0f}s)\n")

    print("=== Each component alone (strength 1) | per method: c (c / c_fixed) p99 |mu - p*| ===")
    for i, comps in enumerate(ABLATION):
        r, traces = evaluate(env, args, out["tuned"], comps, 1.0, args.seed + 100 + i, trace=comps == COMPONENTS)
        print_row(r)
        out["ablation"].append(r)
        if comps == COMPONENTS:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            np.savez(args.out.replace(".json", "_trace.npz"), **traces)
            all_1 = r
    print(f"  ({time.time() - t0:.0f}s)\n")

    print("=== All components, amplitude scaled ===")
    for j, s in enumerate(STRENGTHS):
        r = all_1 if s == 1.0 else evaluate(env, args, out["tuned"], COMPONENTS, s, args.seed + 200 + j)[0]
        if s != 1.0:
            print_row(r)
        out["strength"].append(r)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to '{args.out}' ({time.time() - t0:.0f}s total)")
    if args.plot:
        replot(args.out)


if __name__ == "__main__":
    main()
