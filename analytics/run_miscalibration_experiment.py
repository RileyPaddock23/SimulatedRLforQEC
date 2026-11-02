"""
Cost of a Miscalibrated AR(1) Kalman Filter
===========================================
The drift is AR(1) (correlation time tau, stationary variance s^2) and the Kalman tracker
assumes AR(1) too, but with (tau_f, s2_f). The cost is c / c_opt, with c_opt the error of the
calibrated filter (tau_f = tau, s2_f = s^2) under the same probes, readings and drift.

Analytics:
  - full variance (minimal_control_env.tracker_ar1_c): the tracker as it runs. Its measurement
    variance R comes from its own assumptions (crosstalk ~ its variance v, pair drift ~ V_D(tau_f, s2_f)),
    so a wrong s2_f scales its prior and its R together and barely moves the gain; tau_f still sets the
    mean reversion exp(-1/tau_f). Also the epoch (mu held for B candidates), the pair measuring the
    midpoint (p_e + p_l) / 2, and the random information per pair.
  - shot noise only (minimal_control_env.mismatched_ar1_c): a per-candidate scalar filter with fixed
    information kappa sigma_p^2; for slow drift c / c_opt ~ (sqrt(r) + 1 / sqrt(r)) / 2, r = q_f / q.
    Kept as a reference; it misses the effects above.

Break-even correlation time tau*: the shortest drift correlation time at which the tracker is no worse
than the fixed policy mu = 0 (cost s^2), counting the probes' own error, since the QEC runs on mu +- delta:

    c(tau*) + sigma_p^2 = s^2.

A miscalibrated filter (tau_f = f tau*, or s2_f = f s^2) needs slower drift, so tau* grows. Device:
S = 50 shots, sigma_p = 0.25, one candidate = 0.1 s (Google-like QEC time per candidate, as
analytics/composite), drift std s swept.

Experiments:
  minimal    gaussian minimal model, tau = 30 and 300 candidates, B = 2 and 20 (+ analytic cuts)
  d5         distance-5 surface code (Stim), AR(1) drift on all 129 parameters, tau = 300 candidates, B = 20
  analytics  analytic cuts for the saved runs, break-even tau* and the tau* map (no simulation)
  all        all three

Examples:
    python -m analytics.run_miscalibration_experiment --experiment all --plot
    python -m analytics.run_miscalibration_experiment --experiment analytics --plot
    python -m analytics.run_miscalibration_experiment --replot
"""

import argparse
import json
import os
import time
import types

import numpy as np
from scipy.optimize import brentq

from analytics.minimal_control_env import MinimalControlEnv, mismatched_ar1_c, tracker_ar1_c
from analytics.run_filter_experiment import _pyplot, _save, _style_axes, make_data, make_kf, run_closed_loop
from kalman_tracker import OUSumModel

FACTORS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
MAP_FACTORS = list(2.0 ** np.linspace(-3, 3, 9))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL_SHOTS, CAL_E = 200_000, 0.6


# ------------------------------------------------------------------ analytics
def reading_minimal(env, S):
    """Reading constants of the minimal model for tracker_ar1_c (gaussian: constant shot noise)."""
    return dict(C=[[env.H_l]], R0=np.array([env.Sigma_1 / S]), y0=np.array([env.y0]), rate_scaled=False)


def reading_d5(cal, S):
    return dict(C=cal["C"], R0=cal["sigma_1"] ** 2 / S, y0=cal["y0"], rate_scaled=True)


def lead(case, B):
    """Forecast horizon of mu past the last pair: t_next is the middle of the next epoch; the d=5 loop
    times candidates at their midpoints (run_trial), the minimal loop at their starts (run_closed_loop)."""
    return B / 2 + (0.5 if case == "d5" else 1.0)


def tracker_c(rd, sp, B, ld, tau, s2, tau_f, s2_f):
    return float(np.mean(tracker_ar1_c(rd["C"], rd["R0"], rd["y0"], rd["rate_scaled"], sp, B, ld,
                                       tau, s2, tau_f, s2_f)))


def full_cuts(rd, sp, B, ld, tau, s2):
    """c / c_opt of the full-variance analytics along the tau_f and s2_f cuts, and c_opt / s^2."""
    c = lambda tf, sf: tracker_c(rd, sp, B, ld, tau, s2, tf, sf)
    c0 = c(tau, s2)
    return [c(tau * f, s2) / c0 for f in FACTORS], [c(tau, s2 * f) / c0 for f in FACTORS], c0 / s2


def shot_cuts(info, tau, s2):
    """c / c_opt along the cuts for the shot-noise-only scalar filter (information per candidate `info`,
    averaged over parameters if an array)."""
    info = np.atleast_1d(info)
    c = lambda tf, sf: np.mean([mismatched_ar1_c(i, tau, s2, tf, sf) for i in info])
    c0 = c(tau, s2)
    return [c(tau * f, s2) / c0 for f in FACTORS], [c(tau, s2 * f) / c0 for f in FACTORS]


def breakeven_tau(rd, sp, B, ld, s2, f_tau=1.0, f_s=1.0, guess=100.0, lo=1.0, hi=1e7):
    """tau* (candidates) with c(tau*) + sigma_p^2 = s^2, the filter assuming (f_tau tau, f_s s^2).
    Brackets outward from `guess` in factors of 2; nan if the tracker never breaks even."""
    g = lambda lt: (tracker_c(rd, sp, B, ld, np.exp(lt), s2, f_tau * np.exp(lt), f_s * s2) + sp ** 2) / s2 - 1.0
    if sp ** 2 >= s2:
        return float("nan")
    x, fx = np.log(guess), g(np.log(guess))
    step = np.log(2.0) if fx > 0 else -np.log(2.0)
    while lo <= np.exp(x + step) <= hi:
        y = x + step
        fy = g(y)
        if np.sign(fy) != np.sign(fx):
            return float(np.exp(brentq(g, min(x, y), max(x, y), xtol=2e-3)))
        x, fx = y, fy
    return float("nan")


def tuned_breakeven(rd, sp, B, ld, s2, guess):
    """Lowest tau* over the filter's (tau_f / tau, s2_f / s^2): Nelder-Mead in log2 factors from the calibrated
    point. The best this tracker can do on AR(1) drift, as reference for the calibrated and miscalibrated tau*."""
    from scipy.optimize import minimize

    def f(x):
        t = breakeven_tau(rd, sp, B, ld, s2, f_tau=2.0 ** x[0], f_s=2.0 ** x[1], guess=guess)
        return np.log(t) if np.isfinite(t) else 50.0
    r = minimize(f, [0.0, 0.0], method="Nelder-Mead",
                 options=dict(xatol=0.05, fatol=2e-3, initial_simplex=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    return {"tau": float(np.exp(r.fun)), "f_tau": float(2.0 ** r.x[0]), "f_s": float(2.0 ** r.x[1])}


def experiment_analytics(args, out, env=None):
    """Full-variance and shot-only cuts for the saved runs, break-even tau* per case and s, and the tau* map."""
    S, sp = args.shots_per_candidate, args.sigma_p
    env = env or MinimalControlEnv(mode="gaussian", seed=args.seed)
    cal = load_d5_calibration(args)
    cases = {f"minimal_B{B}": (reading_minimal(env, S), B, lead("minimal", B)) for B in (2, 20)}
    cases["d5"] = (reading_d5(cal, S), args.batch_size, lead("d5", args.batch_size))
    t0 = time.time()

    # Cuts at the simulated conditions
    for tau_s, row in out.get("minimal", {}).items():
        tau, s2 = float(tau_s), env.s ** 2
        row["shot_tau"], row["shot_s"] = shot_cuts(env.kappa(S) * sp ** 2, tau, s2)
        for B in (2, 20):
            rd, _, ld = cases[f"minimal_B{B}"]
            row[f"B{B}_analytic_tau"], row[f"B{B}_analytic_s"], row[f"B{B}_analytic_c0_s2"] = full_cuts(rd, sp, B, ld, tau, s2)
    if "d5" in out:
        res, s2 = out["d5"], args.d5_s ** 2
        info = 4.0 * np.diag(cal["C"]) ** 2 * S / cal["sigma_1"] ** 2 * sp ** 2
        res["shot_tau"], res["shot_s"] = shot_cuts(info, args.d5_tau, s2)
        rd, B, ld = cases["d5"]
        res["analytic_tau"], res["analytic_s"], res["analytic_c0_s2"] = full_cuts(rd, sp, B, ld, args.d5_tau, s2)
    print(f"=== Full-variance analytics at the simulated conditions ({time.time() - t0:.0f}s) ===")
    for tau_s, row in out.get("minimal", {}).items():
        for B in (2, 20):
            print(f"  minimal tau={tau_s:>3} B={B:<2}  c_opt/s^2 analytic {row[f'B{B}_analytic_c0_s2']:.3f}  "
                  f"sim {row[f'B{B}_c0_s2']:.3f}")
    if "d5" in out:
        print(f"  d5      tau={args.d5_tau:g} B={args.batch_size}  c_opt/s^2 analytic "
              f"{out['d5']['analytic_c0_s2']:.3f}  stim {out['d5']['c0_s2']:.3f} ± {out['d5']['c0_se']:.3f}")

    # Break-even tau* along the two cuts, per drift std; at d=5 also the tuned optimum
    be = {}
    for name, (rd, B, ld) in cases.items():
        be[name] = {}
        for s in (args.d5_s_sweep if name == "d5" else args.s_sweep):
            s2 = s * s
            t1 = time.time()
            t_cal = breakeven_tau(rd, sp, B, ld, s2)
            guess = t_cal if np.isfinite(t_cal) else 100.0
            row = {"tau": [breakeven_tau(rd, sp, B, ld, s2, f_tau=f, guess=guess) if f != 1 else t_cal for f in FACTORS],
                   "s": [breakeven_tau(rd, sp, B, ld, s2, f_s=f, guess=guess) if f != 1 else t_cal for f in FACTORS]}
            if name == "d5":
                row["best"] = tuned_breakeven(rd, sp, B, ld, s2, guess)
            be[name][f"{s:g}"] = row
            sec = lambda v: f"{v * args.candidate_s:7.1f}" if np.isfinite(v) else "    n/a"
            best = row.get("best")
            print(f"  break-even {name:<11} s={s:<4g} tau* = {sec(t_cal)} s calibrated | tau_f cut "
                  f"{' '.join(sec(v) for v in row['tau'])} | s2_f cut {' '.join(sec(v) for v in row['s'])}"
                  + (f" | tuned {sec(best['tau'])} s at tau_f/tau={best['f_tau']:.2f}, s2_f/s2={best['f_s']:.2f}"
                     if best else "") + f"  ({time.time() - t1:.0f}s)")
    out["breakeven"] = be

    # tau* map over (tau_f / tau, s2_f / s^2) at d=5
    rd, B, ld = cases["d5"]
    s2 = args.map_s ** 2
    t1 = time.time()
    t_cal = breakeven_tau(rd, sp, B, ld, s2)
    Z = [[breakeven_tau(rd, sp, B, ld, s2, f_tau=ft, f_s=fs, guess=t_cal) for ft in MAP_FACTORS] for fs in MAP_FACTORS]
    out["map"] = {"s": args.map_s, "factors": MAP_FACTORS, "tau_star": Z}
    print(f"  tau* map at d=5, s={args.map_s:g}: {len(MAP_FACTORS)}x{len(MAP_FACTORS)} ({time.time() - t1:.0f}s)")


# ------------------------------------------------------------------ simulations
def experiment_minimal(args, out):
    S, sp, R = args.shots_per_candidate, args.sigma_p, args.replicas
    env = MinimalControlEnv(mode="gaussian", seed=args.seed)
    s2 = env.s ** 2
    print(f"=== Minimal model (S={S}, sigma_p={sp}, Lambda={env.Lambda(S):.3f}, s={env.s}, {R} replicas, "
          f"{args.candidates} candidates) ===")
    out["minimal"] = {}
    for tau in (30.0, 300.0):
        p, t, noise = make_data(args.candidates, R, env.s, args.seed + int(tau), profile="ar1", tau=tau)
        read = lambda k, e: env.read_many(e, S, noise[k])
        burn = int(min(10 * tau, args.candidates // 4))
        row = {}
        for B in (2, 20):
            run = lambda tf, sf: run_closed_loop(make_kf(env, OUSumModel([tf], [sf]), R, S, sp, args.seed),
                                                 p, t, read, B, burn)["c"]
            c0 = run(tau, s2)
            row[f"B{B}_tau"] = [run(tau * f, s2) / c0 if f != 1 else 1.0 for f in FACTORS]
            row[f"B{B}_s"] = [run(tau, s2 * f) / c0 if f != 1 else 1.0 for f in FACTORS]
            row[f"B{B}_c0_s2"] = c0 / s2
        out["minimal"][str(int(tau))] = row
        print(f"  tau = {tau:g} candidates  (c_opt/s^2: B=2 {row['B2_c0_s2']:.4f}, B=20 {row['B20_c0_s2']:.4f})")
        print(f"    {'factor':>7} | {'tau_f = f tau: B=2':>17} {'B=20':>6} | {'s2_f = f s2: B=2':>16} {'B=20':>6}")
        for i, f in enumerate(FACTORS):
            print(f"    {f:>7g} | {row['B2_tau'][i]:>17.3f} {row['B20_tau'][i]:>6.3f} | "
                  f"{row['B2_s'][i]:>16.3f} {row['B20_s'][i]:>6.3f}")
    return env


def load_d5_calibration(args, env=None):
    cal_path = os.path.join(ROOT, "results", f"filter_calibration_d5_r5_seed{args.seed}.npz")
    if os.path.exists(cal_path):
        return dict(np.load(cal_path))
    from run_filter_comparison import calibrate
    from surface_code_env import RotatedSurfaceCodeEnv
    env = env or RotatedSurfaceCodeEnv(distance=5, rounds=5, drift_profile="sinusoidal", drift_amplitude=0.0, seed=args.seed)
    cal = calibrate(env, CAL_SHOTS, CAL_E)
    os.makedirs(os.path.dirname(cal_path), exist_ok=True)
    np.savez(cal_path, **cal)
    return cal


def experiment_d5(args, out):
    """AR(1) drift on every parameter of the d=5 code, sampled per candidate (replaces get_drift_opt)."""
    from run_filter_comparison import SHOTS_PER_DRIFT_UPDATE, make_kf as make_kf_d5, run_trial
    from surface_code_env import RotatedSurfaceCodeEnv

    S, B, tau, s = args.shots_per_candidate, args.batch_size, args.d5_tau, args.d5_s
    E = SHOTS_PER_DRIFT_UPDATE / S  # candidates per drift update; the filters' time is in drift updates
    env = RotatedSurfaceCodeEnv(distance=5, rounds=5, drift_profile="sinusoidal", drift_amplitude=0.0, seed=args.seed)
    cal = load_d5_calibration(args, env)
    env.arma_sigma = s  # run_trial reports c / arma_sigma^2

    n = args.d5_shots // S + 2 * B
    rng = np.random.default_rng(args.seed + 5)
    a = np.exp(-1.0 / tau)
    traj = np.empty((n, env.num_params))
    traj[0] = s * rng.standard_normal(env.num_params)
    for k in range(1, n):
        traj[k] = a * traj[k - 1] + s * np.sqrt(1 - a * a) * rng.standard_normal(env.num_params)
    env.get_drift_opt = lambda t: traj[min(int(t * E), n - 1)]

    targs = types.SimpleNamespace(shots_per_candidate=S, batch_size=B, sigma_p=args.sigma_p, scale=1.0, seed=args.seed,
                                  eval_shots=0, eval_interval=10 ** 9)
    run = lambda tf, sf: run_trial(env, make_kf_d5(env, cal, OUSumModel([tf / E], [sf]), targs), args.d5_shots,
                                   targs, evaluate=False)
    t0 = time.time()
    r0 = run(tau, s * s)
    c0 = r0["c_s2"]
    res = {"c0_s2": c0, "c0_se": r0["c_s2_se"], "tau": [], "s": []}
    for key, mk in (("tau", lambda f: (tau * f, s * s)), ("s", lambda f: (tau, s * s * f))):
        for f in FACTORS:
            res[key].append(1.0 if f == 1 else run(*mk(f))["c_s2"] / c0)
    out["d5"] = res
    print(f"\n=== d=5 surface code (Stim), AR(1) tau = {tau:g} candidates, s = {s}, {args.d5_shots:,} shots per run, "
          f"B={B} ({time.time() - t0:.0f}s) ===")
    print(f"  c_opt/s^2 = {c0:.4f} ± {r0['c_s2_se']:.4f}  (standard error over parameters; stim noise differs per run)")
    print(f"    {'factor':>7} | {'tau_f = f tau: stim':>19} | {'s2_f = f s2: stim':>17}")
    for i, f in enumerate(FACTORS):
        print(f"    {f:>7g} | {res['tau'][i]:>19.3f} | {res['s'][i]:>17.3f}")


# ------------------------------------------------------------------ plots
# s sweep: one-hue ordinal ramp (light -> dark = small -> large drift), plus a marker per level
S_COLORS = ["#86b6ef", "#2a78d6", "#104281"]
S_MARKERS = ["o", "s", "D"]
CUT_LABELS = (("tau", "assumed / true correlation time  τ_f / τ", "Wrong correlation time (s² correct)"),
              ("s", "assumed / true variance  s²_f / s²", "Wrong variance (τ correct)"))


def _factor_axis(ax, xlabel):
    x = np.array(FACTORS)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [f"{v:g}" for v in x])
    ax.set_xlabel(xlabel)
    ax.axvline(1.0, color="#cccccc", lw=1, zorder=0)
    _style_axes(ax)


def breakeven_freq(tau_star, B):
    """Break-even corner frequency f* = 1 / (2 pi tau*) in cycles per policy update (one epoch of B candidates),
    tau* in candidates. Comparable to Google's f_c ~ 1/130 cycles per update."""
    return B / (2.0 * np.pi * np.asarray(tau_star, dtype=float))


def _breakeven_row(axes, be, B, s_sweep):
    from matplotlib.ticker import FuncFormatter, LogLocator
    x = np.array(FACTORS)
    for ax, (key, xlabel, title) in zip(axes, CUT_LABELS):
        for i, s in enumerate(s_sweep):
            y = breakeven_freq(be[f"{s:g}"][key], B)
            ax.plot(x, y, color=S_COLORS[i % 3], marker=S_MARKERS[i % 3], ms=8, lw=2,
                    label=f"s = {s:g}   (calibrated f* = 1/{1 / y[3]:.3g} per update)")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        _factor_axis(ax, xlabel)
        ax.set_title(f"Break-even f*: {title[0].lower() + title[1:]}")
    axes[0].set_ylabel(f"break-even frequency  f* = 1 / (2π τ*)  (cycles / update)\n"
                       f"[update = one epoch of B = {B} candidates]")
    axes[0].legend(frameon=False, title="drift std (units)", loc="center", fontsize=9)


def plot_case(path, title, validation, be, B, s_sweep):
    """Top: c / c_opt at the simulated conditions, full-variance analytics vs simulation.
    Bottom: break-even frequency 1 / (2 pi tau*) in cycles per policy update for the s sweep. `validation` = [(row, key prefix, label, color, marker)]."""
    plt = _pyplot()
    x = np.array(FACTORS)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, (key, xlabel, ttl) in zip(axes[0], CUT_LABELS):
        for row, pre, label, color, marker in validation:
            ax.plot(x, row[f"{pre}analytic_{key}"], color=color, lw=2, label=f"full-variance analytics, {label}")
            ax.plot(x, row[f"{pre}{key}"], ls="none", marker=marker, color=color, mec="white", mew=1.5, ms=9,
                    label=f"simulation, {label}")
        _factor_axis(ax, "")
        ax.set_title(f"{ttl}, at the simulated τ")
    axes[0, 0].set_ylabel("cost  c / c_opt")
    axes[0, 0].legend(frameon=False, loc="upper center", fontsize=8.5)
    ymax = max(max(max(r[f"{p}analytic_{k}"]), max(r[f"{p}{k}"]))
               for r, p, *_ in validation for k in ("tau", "s"))
    for ax in axes[0]:
        ax.set_ylim(0.95, ymax * 1.05)
    _breakeven_row(axes[1], be, B, s_sweep)
    for ax, (_, xlabel, _) in zip(axes[1], CUT_LABELS):
        ax.set_xlabel(xlabel)
    fig.suptitle(title)
    _save(plt, fig, path)


def plot_map(m, B):
    from matplotlib.colors import LogNorm
    plt = _pyplot()
    g = np.array(m["factors"])
    Z = breakeven_freq(m["tau_star"], B)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    levels = np.geomspace(np.nanmin(Z), np.nanmax(Z), 9)
    cs = ax.contourf(g, g, Z, levels=levels, cmap="Blues", norm=LogNorm())
    ax.contour(g, g, Z, levels=levels[1:-1], colors="#333333", linewidths=0.6)
    ax.plot(g, g, color="#eb6834", lw=2, ls="--", label="same q:  s²_f / τ_f = s² / τ")
    ax.plot([1], [1], marker="*", color="#eb6834", ms=14, mec="white", ls="none",
            label=f"calibrated: f* = 1/{1 / Z[len(g) // 2, len(g) // 2]:.3g} per update")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ticks = np.array(FACTORS)
    ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
    ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
    ax.set_xlabel("τ_f / τ")
    ax.set_ylabel("s²_f / s²")
    ax.set_title(f"Break-even f* = 1 / (2π τ*) (cycles / update), d=5 surface code, s = {m['s']:g}")
    ax.legend(frameon=False, loc="upper left")
    cb = fig.colorbar(cs, ax=ax, label=f"break-even frequency  f*  (cycles / update, B = {B} candidates)")
    cb.set_ticks(levels, labels=[f"{v:.3g}" for v in levels])
    _save(plt, fig, "miscalibration_map.png")


def plot(out):
    cfg = out["config"]
    if "minimal" in out and "breakeven" in out:
        for B, marker in ((2, "o"), (20, "s")):
            val = [(out["minimal"]["30"], f"B{B}_", "τ = 30", "#eb6834", marker),
                   (out["minimal"]["300"], f"B{B}_", "τ = 300", "#2a78d6", marker)]
            plot_case(f"miscalibration_minimal_B{B}.png",
                      f"Minimal model (one d=3 coupler, gaussian), B = {B}"
                      + (" (re-centered every pair)" if B == 2 else " (μ held per epoch)"),
                      val, out["breakeven"][f"minimal_B{B}"], B, cfg["s_sweep"])
    if "d5" in out and "breakeven" in out:
        plot_case("miscalibration_d5.png", f"d=5 surface code (Stim validation at τ = {cfg['d5_tau']:g} candidates, "
                  f"s = {cfg['d5_s']:g}), B = {cfg['batch_size']}",
                  [(out["d5"], "", "Stim", "#1baf7a", "D")], out["breakeven"]["d5"], cfg["batch_size"],
                  cfg.get("d5_s_sweep", cfg["s_sweep"]))
    if "map" in out:
        plot_map(out["map"], cfg["batch_size"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=["minimal", "d5", "analytics", "all"], default="all")
    p.add_argument("--shots-per-candidate", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=20, help="B, candidates per epoch, for the d=5 runs")
    p.add_argument("--sigma-p", type=float, default=0.25)
    p.add_argument("--replicas", type=int, default=32)
    p.add_argument("--candidates", type=int, default=20_000, help="candidates per minimal run")
    p.add_argument("--d5-tau", type=float, default=300.0, help="AR(1) tau in candidates at d=5")
    p.add_argument("--d5-s", type=float, default=0.6, help="AR(1) std at d=5")
    p.add_argument("--d5-shots", type=int, default=490_000, help="shots per d=5 run")
    p.add_argument("--s-sweep", type=float, nargs="+", default=[0.3, 0.6, 1.0], help="drift stds for break-even tau*")
    p.add_argument("--d5-s-sweep", type=float, nargs="+", default=[0.3, 0.45, 0.6],
                   help="drift stds for the d=5 break-even tau* (with the tuned optimum)")
    p.add_argument("--map-s",type=float, default=0.6, help="drift std of the d=5 tau* map")
    p.add_argument("--candidate-s", type=float, default=0.1, help="seconds per candidate (Google-like)")
    p.add_argument("--seed", type=int, default=163)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--replot", action="store_true")
    p.add_argument("--out", type=str, default=os.path.join(ROOT, "analytics", "results", "miscalibration_results.json"))
    args = p.parse_args()
    if args.replot:
        with open(args.out) as f:
            plot(json.load(f))
        return

    out = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            out = json.load(f)
    out["config"] = {**out.get("config", {}), **{k: v for k, v in vars(args).items() if k not in ("plot", "replot", "out")}}
    env = None
    if args.experiment in ("minimal", "all"):
        env = experiment_minimal(args, out)
    if args.experiment in ("d5", "all"):
        experiment_d5(args, out)
    experiment_analytics(args, out, env)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to '{args.out}'")
    if args.plot:
        plot(out)


if __name__ == "__main__":
    main()
