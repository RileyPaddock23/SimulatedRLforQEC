"""
Kalman Trackers vs PEPG under Power-Law Drift (minimal model)
=============================================================
One drifting CX coupler of a d=3 surface code (MinimalControlEnv), run as R independent
replicas. Every controller uses the same probe schedule: B antithetic candidates mu +- delta
(probe rms sigma_p) per epoch, S shots each, mu fixed for the epoch.
Replicas share the drift and reading noise across methods (common random numbers).

Controllers (KalmanTracker from kalman_tracker.py, except PEPG):
  AR(1) lag-1   AR(1) matched to the variance and one-candidate autocorrelation of the drift
  AR(1) ML      AR(1) whose tau maximizes the innovation likelihood on a training run
  AR(1) oracle  AR(1) whose tau minimizes the tracking error on a training run
  Multi-AR(1)   sum of AR(1)s (tau = 0.25 .. 256 drift updates) fitted to the drift autocovariance
  Exact         the true drift model (1/f^alpha + white floor, per-drift-update, interpolated)
  PEPG          RLControlAgent as plain antithetic SGD, gain tuned on the training run

The drift is generate_power_law_noise(normalize="analytic") with std s, sampled once per
drift update and linearly interpolated, as in RotatedSurfaceCodeEnv. candidates_per_drift_update = 140
matches the full code at S = 50 (7000 shots per drift update). Training and evaluation use
different seeds. c = E[(mu - p*)^2] after a burn-in of a quarter of the run.

The Kalman trackers measure each antithetic pair's difference (KalmanTracker.update), the same
statistic PEPG's gradient uses, so neither uses the pair sum (|mu - p|^2 information).

Filter diagnostics (Kalman trackers, one value per pair):
  NEES        realized / reported variance of the predicted p before each pair (1 = calibrated)
  NIS         mean squared normalized innovation (1 = calibrated)
  innov corr  mean autocorrelation of nu = sign(J) r / sqrt(S) at lags 1..B/2 (0 = nothing left to learn)

Experiments:
  validate  AR(1) drift: AR(1) filter re-centered every pair vs the Cramer-Rao c*, then the
            controller c of the AR(1) filter and PEPG at batch size B
  alpha     sweep alpha at candidates_per_drift_update = 140 and 1
  lambda    sweep S (information budget Lambda) at alpha = 0.67
  white     sweep the white-floor fraction at alpha = 0.67
  stim      alpha = 0.67 on stim readings, with the settings tuned on the gaussian model

Examples:
    python -m analytics.run_filter_experiment --experiment validate --plot
    python -m analytics.run_filter_experiment --experiment all --plot
"""

import argparse
import json
import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from analytics.minimal_control_env import MinimalControlEnv, cramer_rao_c, drift_trajectories
from kalman_tracker import (KalmanTracker, OUSumModel, PowerLawModel, fit_ar1_lag1, fit_multi_ar1,
                            interpolated_autocov, power_law_autocov)
from rl_agent import RLControlAgent

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")

METHODS = ["Exact", "PEPG", "Multi-AR(1)", "AR(1) oracle", "AR(1) ML", "AR(1) lag-1"]
# Categorical colors in fixed order; markers give a second channel besides color
STYLE = {
    "Exact": ("#2a78d6", "o", "-"),
    "PEPG": ("#eb6834", "s", "-"),
    "Multi-AR(1)": ("#1baf7a", "D", "--"),
    "AR(1) ML": ("#eda100", "v", "--"),
    "AR(1) oracle": ("#e87ba4", "^", "-"),
    "AR(1) lag-1": ("#008300", "x", ":"),
}
# Plotted subset and plot labels (a sum of K AR(1)s sampled on a grid is an ARMA(K, K-1))
PLOT_METHODS = ["Exact", "PEPG", "Multi-AR(1)", "AR(1) ML"]
PLOT_LABELS = {"Multi-AR(1)": "ARMA(4,3)"}
GAINS = [0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0]
MULTI_TAUS = np.geomspace(0.25, 256.0, 6)  # drift updates


# ------------------------------------------------------------------ controllers
class PEPGController:
    """RLControlAgent as plain antithetic SGD (fixed probe width, no Adam or clipping) on R
    independent replicas: num_params = num_detectors = R with M = I."""

    def __init__(self, R: int, H_l: float, gain: float, sigma_p: float, seed: int):
        lr = gain / (2.0 * H_l * sigma_p ** 2)
        self.agent = RLControlAgent(
            num_params=R, num_detectors=R, mask_matrix=np.eye(R),
            lr_mu=lr * sigma_p ** 2,  # the agent divides by sigma^2
            lr_sigma=0.0, entropy_reg=0.0, sigma_init=sigma_p, sigma_min=sigma_p, sigma_max=sigma_p,
            grad_clip=0.0, use_adam=False, seed=seed,
        )

    @property
    def mu(self):
        return self.agent.mu

    def sample_policies(self, batch_size, antithetic=True):
        return self.agent.sample_policies(batch_size, antithetic=antithetic)

    def update(self, actions, observations, times, t_next=None):
        self.agent.update(actions, observations)
        return None


def make_kf(env: MinimalControlEnv, model, R: int, S: int, sigma_p: float, seed: int) -> KalmanTracker:
    return KalmanTracker(
        num_params=R, num_detectors=R, mask_matrix=np.eye(R), drift_model=model,
        y0=env.y0, C=env.H_l * np.eye(R), sigma_1=env.sigma_1, shots=S, sigma_probe=sigma_p,
        rate_scaled_noise=env.mode == "stim", seed=seed,
    )


# ------------------------------------------------------------------ closed loop
def measurement_order(B: int, pair_order: str) -> np.ndarray:
    """Candidate measured k-th within an epoch. sample_policies lays out (mu + delta_i) for
    i < B/2 and their partners (mu - delta_i) at i + B/2. "split" measures in that order (partners
    B/2 candidates apart, as run_experiment.py); "adjacent" measures each partner right after its pair."""
    if pair_order == "split":
        return np.arange(B)
    return np.stack([np.arange(B // 2), np.arange(B // 2) + B // 2], axis=1).ravel()


def run_closed_loop(ctrl, p: np.ndarray, t: np.ndarray, read: Callable, B: int, burn: int,
                    pair_order: str = "split", trace: Optional[int] = None) -> Dict[str, float]:
    """Runs a controller on drift p (candidates x replicas); candidate k is read at offset c - p[k].
    `trace` = a replica index whose mu is returned for every candidate (res["mu_trace"])."""
    T, R = p.shape
    order = measurement_order(B, pair_order)
    position_of = np.empty(B, dtype=int)
    position_of[order] = np.arange(B)
    e2, truth, diag, mu_trace = [], [], {}, []
    for u in range(T // B):
        k0 = u * B
        cands = ctrl.sample_policies(B, antithetic=True)
        obs = np.empty((B, R))
        for j, ci in enumerate(order):
            obs[ci] = read(k0 + j, cands[ci] - p[k0 + j])
        err = np.asarray(ctrl.mu)[None, :] - p[k0:k0 + B]
        if trace is not None:
            mu_trace.append(np.full(B, float(np.asarray(ctrl.mu)[trace])))
        d = ctrl.update(cands, obs, t[k0 + position_of], t_next=t[min(k0 + B + B // 2, T - 1)])
        if k0 >= burn:
            e2.append(err ** 2)
            if d is not None:
                truth.append(p[k0 + position_of[d.pop("late")]])
                for key, val in d.items():
                    diag.setdefault(key, []).append(val)
    e2 = np.concatenate(e2)
    res = {"c": float(e2.mean()), "c_se": float(e2.mean(axis=0).std(ddof=1) / np.sqrt(R)),
           "e_p99": float(np.sqrt(np.quantile(e2, 0.99)))}
    if trace is not None:
        res["mu_trace"] = np.concatenate(mu_trace)
    if diag:
        d = {k: np.concatenate(v) for k, v in diag.items()}
        err_pred = (d["p_pred"] - np.concatenate(truth)) ** 2
        nu = d["nu"]
        ac = [np.mean(nu[:-lag] * nu[lag:]) / np.mean(nu ** 2) for lag in range(1, B // 2 + 1)]
        res.update({
            "pred_mse": float(err_pred.mean()), "nees": float(err_pred.mean() / d["v_pred"].mean()),
            "nis": float(np.mean(nu ** 2)), "loglik": float(d["loglik"].mean()), "innov_corr": float(np.mean(ac)),
        })
    return res


def riccati_diff_c(info: float, q2: float, a: float) -> float:
    """Steady predicted variance of a scalar AR(1) filter receiving information `info` per candidate:
    root of  info c^2 + (1 - a^2 - info q^2) c - q^2 = 0."""
    b = 1.0 - a * a - info * q2
    return float((-b + np.sqrt(b * b + 4.0 * info * q2)) / (2.0 * info))


def make_data(n_candidates, R, s, seed, **drift):
    rng = np.random.default_rng(seed)
    p, t = drift_trajectories(n_candidates, R, s, rng, **drift)
    return p, t, rng.standard_normal(p.shape)


# ------------------------------------------------------------------ one condition
def compare(env: MinimalControlEnv, alpha: float, white: float, E: float, S: int, args,
            tuned: Optional[Dict] = None, R: Optional[int] = None, n_candidates: Optional[int] = None) -> Dict:
    """All six controllers on one power-law condition. Tunes tau / gain on a training seed unless
    `tuned` is given; evaluates on a separate seed with the current env.mode."""
    R = R or (args.replicas if E > 1 else args.replicas_e1)
    n_candidates = n_candidates or (args.candidates if E > 1 else args.candidates_e1)
    B, sp, s = args.batch_size, args.sigma_p, env.s
    burn = n_candidates // 4
    drift = dict(profile="powerlaw", alpha=alpha, white_fraction=white, candidates_per_drift_update=E)
    gauss = env.mode == "gaussian"

    def reader(noise):
        return (lambda k, e: env.read_many(e, S, noise[k])) if gauss else (lambda k, e: env.read_many(e, S))

    cov = lambda lags: interpolated_autocov(power_law_autocov(alpha, white, s), E, lags)
    tau_l1, var0 = fit_ar1_lag1(cov, E)
    m_taus, m_vars = fit_multi_ar1(cov, E, MULTI_TAUS, 2 * 256 * E)
    ar1 = lambda tau: OUSumModel([tau], [var0])
    kf = lambda model, seed: make_kf(env, model, R, S, sp, seed)

    if tuned is None:
        p, t, noise = make_data(n_candidates, R, s, args.seed, **drift)
        run = lambda ctrl: run_closed_loop(ctrl, p, t, reader(noise), B, burn, args.pair_order)
        # tau grid in drift updates (2 candidates .. 4 L drift updates), then a finer pass around each optimum
        grid = list(np.geomspace(2.0 / E, 1024.0, 12))
        scan = {tau: run(kf(ar1(tau), args.seed)) for tau in grid}
        for key, sign in (("loglik", -1.0), ("c", 1.0)):
            best = min(scan, key=lambda x: sign * scan[x][key])
            for f in (0.6, 0.8, 1.25, 1.6):
                if best * f not in scan:
                    scan[best * f] = run(kf(ar1(best * f), args.seed))
        tau_ml = max(scan, key=lambda x: scan[x]["loglik"])
        tau_or = min(scan, key=lambda x: scan[x]["c"])
        gain_c = {g: run(PEPGController(R, env.H_l, g, sp, args.seed))["c"] for g in GAINS}
        tuned = {"tau_ml": float(tau_ml), "tau_oracle": float(tau_or), "gain": float(min(gain_c, key=gain_c.get))}

    p, t, noise = make_data(n_candidates, R, s, args.seed + 1000, **drift)
    run = lambda ctrl: run_closed_loop(ctrl, p, t, reader(noise), B, burn, args.pair_order)
    seed = args.seed + 1
    ctrls = {
        "Exact": kf(PowerLawModel(alpha, white, s), seed),
        "PEPG": PEPGController(R, env.H_l, tuned["gain"], sp, seed),
        "Multi-AR(1)": kf(OUSumModel(m_taus, m_vars), seed),
        "AR(1) oracle": kf(ar1(tuned["tau_oracle"]), seed),
        "AR(1) ML": kf(ar1(tuned["tau_ml"]), seed),
        "AR(1) lag-1": kf(ar1(tau_l1), seed),
    }
    res = {name: run(ctrl) for name, ctrl in ctrls.items()}
    s2, c_ex = float(np.mean(p ** 2)), res["Exact"]["c"]
    for r in res.values():
        r["c_s2"], r["ratio"] = r["c"] / s2, r["c"] / c_ex
    return {
        "alpha": alpha, "white": white, "candidates_per_drift_update": E, "S": S, "Lambda": env.Lambda(S), "s2": s2,
        "mode": env.mode, "tau_lag1": tau_l1, "multi_taus": list(map(float, m_taus)),
        "multi_vars": list(map(float, m_vars)), **tuned, "results": res,
    }


def print_condition(r: Dict):
    E = r["candidates_per_drift_update"]
    print(f"\n  alpha={r['alpha']:g}  white={r['white']:g}  candidates/drift update={E:g}  S={r['S']}  Lambda={r['Lambda']:.3f}  "
          f"({r['mode']})  s^2={r['s2']:.4f}")
    print(f"    {'method':<13} | {'setting':>16} | {'c/s^2':>15} | {'c/c_exact':>9} | {'NEES':>5} | {'NIS':>5} | {'innov corr':>10}")
    setting = {
        "PEPG": f"gain {r['gain']:.3f}", "AR(1) oracle": f"tau {r['tau_oracle'] * E:.3g} sl",
        "AR(1) ML": f"tau {r['tau_ml'] * E:.3g} sl", "AR(1) lag-1": f"tau {r['tau_lag1'] * E:.3g} sl",
        "Multi-AR(1)": f"{len(r['multi_taus'])} components", "Exact": "true model",
    }
    for name in METHODS:
        x = r["results"][name]
        diag = (f"{x['nees']:>5.2f} | {x['nis']:>5.2f} | {x['innov_corr']:>+10.3f}" if "nees" in x
                else f"{'':>5} | {'':>5} | {'':>10}")
        print(f"    {name:<13} | {setting[name]:>16} | {x['c_s2']:.4f} ± {x['c_se'] / r['s2']:.4f} | "
              f"{x['ratio']:>9.3f} | {diag}")


# ------------------------------------------------------------------ experiments
def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888",
                         "axes.labelcolor": "#333333", "xtick.color": "#555555", "ytick.color": "#555555"})
    return plt


def _style_axes(ax):
    ax.grid(True, color="#dddddd", lw=0.8)
    ax.set_axisbelow(True)


def _plain_log(ax):
    """Log axis with plain tick labels (1.2, 2 rather than 1.2 x 10^0)."""
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1.0, 10.0, 0.5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _plot_methods(ax, x, rows, key, labels=None, annotate_k=False):
    labels = {**PLOT_LABELS, **(labels or {})}
    for name in PLOT_METHODS:
        color, marker, ls = STYLE[name]
        y = [r["results"][name][key] for r in rows]
        ax.plot(x, y, marker=marker, ls=ls, color=color, lw=2, ms=8, label=labels.get(name, name))
        if annotate_k and name == "Multi-AR(1)":
            for xi, yi, r in zip(x, y, rows):
                ax.annotate(f"K={len(r['multi_taus'])}", (xi, yi), xytext=(6, -12), textcoords="offset points",
                            fontsize=8, color="#555555")


def _save(plt, fig, path, fig_dir=FIG_DIR):
    path = os.path.join(fig_dir, path)
    os.makedirs(fig_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Plot saved to '{path}'")


def experiment_validate(env, args, out):
    S, B, sp, s = args.shots_per_candidate, args.batch_size, args.sigma_p, env.s
    R, n_candidates = args.replicas, args.candidates_validate
    print(f"=== Validation on AR(1) drift (S={S}, B={B}, sigma_p={sp}, Lambda={env.Lambda(S):.3f}) ===")
    print("  'pred' = E[(p_hat - p*)^2] before each pair, filter re-centered every pair (B=2).")
    print("  c*_diff = Riccati bound with the pair-difference information kappa sigma_p^2 per candidate (what a")
    print("  difference-based tracker can use); c* also counts kappa e^2 from the pair sum.")
    print(f"  {'tau':>6} | {'c*/s^2':>7} | {'c*_diff':>7} | {'KF pred/s^2':>11} | {'pred/c*_diff':>12} | {'NEES':>5} | "
          f"{'KF c/s^2 (B)':>12} | {'PEPG c/s^2 (B)':>14} | {'gain':>5}")
    rows = []
    for tau in [float(x) for x in args.validate_taus.split(",")]:
        a = np.exp(-1.0 / tau)
        c_star = cramer_rao_c(env.kappa(S), s ** 2 * (1 - a * a), a, sp ** 2)
        c_diff = riccati_diff_c(env.kappa(S) * sp ** 2, s ** 2 * (1 - a * a), a)
        drift = dict(profile="ar1", tau=tau)
        p, t, noise = make_data(n_candidates, R, s, args.seed + 1000, **drift)
        read = lambda k, e: env.read_many(e, S, noise[k])
        burn = int(min(5 * tau, n_candidates // 4))
        model = lambda: OUSumModel([tau], [s ** 2])
        kf2 = run_closed_loop(make_kf(env, model(), R, S, sp, args.seed), p, t, read, 2, burn)
        kfB = run_closed_loop(make_kf(env, model(), R, S, sp, args.seed), p, t, read, B, burn, args.pair_order)
        p_tr, t_tr, n_tr = make_data(n_candidates, R, s, args.seed, **drift)
        read_tr = lambda k, e: env.read_many(e, S, n_tr[k])
        gain_c = {g: run_closed_loop(PEPGController(R, env.H_l, g, sp, args.seed), p_tr, t_tr, read_tr, B, burn,
                                  args.pair_order)["c"]
                  for g in GAINS}
        g = min(gain_c, key=gain_c.get)
        pepg = run_closed_loop(PEPGController(R, env.H_l, g, sp, args.seed), p, t, read, B, burn, args.pair_order)
        s2 = s ** 2
        rows.append({"tau": tau, "c_star": c_star / s2, "c_diff": c_diff / s2, "kf_pred": kf2["pred_mse"] / s2,
                     "nees": kf2["nees"], "kf_c": kfB["c"] / s2, "pepg_c": pepg["c"] / s2, "gain": g})
        print(f"  {tau:>6g} | {c_star / s2:>7.4f} | {c_diff / s2:>7.4f} | {kf2['pred_mse'] / s2:>11.4f} | "
              f"{kf2['pred_mse'] / c_diff:>12.3f} | "
              f"{kf2['nees']:>5.2f} | {kfB['c'] / s2:>12.4f} | {pepg['c'] / s2:>14.4f} | {g:>5.3f}")
    out["validate"] = rows
    if args.plot:
        plot_validate(rows, B)


def plot_validate(rows, B):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    taus = [r["tau"] for r in rows]
    ax.plot(taus, [r["c_star"] for r in rows], color="#555555", ls="--", lw=2, label="Cramér–Rao $c^*$")
    ax.plot(taus, [r["c_diff"] for r in rows], color="#555555", ls=":", lw=2,
            label="Riccati bound, pair differences only")
    ax.plot(taus, [r["kf_pred"] for r in rows], marker="o", color=STYLE["Exact"][0], lw=2, ms=8,
            label="AR(1) filter, prediction error (B=2)")
    ax.plot(taus, [r["kf_c"] for r in rows], marker="D", ls="--", color=STYLE["Multi-AR(1)"][0], lw=2, ms=8,
            label=f"AR(1) filter, control error (B={B})")
    ax.plot(taus, [r["pepg_c"] for r in rows], marker="s", color=STYLE["PEPG"][0], lw=2, ms=8,
            label=f"PEPG, control error (B={B})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("AR(1) drift correlation time τ (candidates)")
    ax.set_ylabel("$E[(\\mu - p^*)^2] / s^2$")
    ax.set_title("Matched model: AR(1) drift, AR(1) filter")
    _style_axes(ax)
    ax.legend(frameon=False)
    _save(plt, fig, "filter_validate.png")


def _sweep(env, args, out, key, conditions, label):
    rows = []
    for cond in conditions:
        t0 = time.time()
        r = compare(env, **cond, args=args)
        print_condition(r)
        print(f"    ({time.time() - t0:.0f}s)")
        rows.append(r)
    out[key] = rows
    return rows


def experiment_alpha(env, args, out):
    alphas = [float(x) for x in args.alphas.split(",")]
    S, w = args.shots_per_candidate, args.white
    panels = {}
    for E in (args.candidates_per_drift_update, 1.0):
        print(f"\n=== Alpha sweep (white={w}, candidates/drift update={E:g}, S={S}, B={args.batch_size}, sigma_p={args.sigma_p}) ===")
        panels[E] = _sweep(env, args, out, f"alpha_E{E:g}",
                           [dict(alpha=a, white=w, E=E, S=S) for a in alphas], "alpha")
    if args.plot:
        plot_alpha(panels)


def plot_alpha(panels):
    """panels: {candidates_per_drift_update: rows}. The multi-AR(1) fit keeps a different number K of
    components per alpha, so it is labeled ARMA(K, K-1) with K at each point."""
    plt = _pyplot()
    rows0 = next(iter(panels.values()))[0]
    w, S, lam = rows0["white"], rows0["S"], rows0["Lambda"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for col, (E, rows) in enumerate(panels.items()):
        x = [r["alpha"] for r in rows]
        arma = {"Multi-AR(1)": "ARMA(K, K−1)"}
        _plot_methods(axes[0, col], x, rows, "c_s2", labels=arma, annotate_k=True)
        _plot_methods(axes[1, col], x, rows, "ratio", labels=arma)
        axes[0, col].set_title(f"{E:g} candidate{'s' if E != 1 else ''} per drift update "
                               f"({'interpolated drift' if E > 1 else 'no interpolation'})")
        axes[0, col].set_ylabel("control error $c / s^2$")
        axes[1, col].set_ylabel("$c / c_{Exact}$")
        axes[1, col].set_xlabel("power-law exponent α")
    for ax in axes.flat:
        _style_axes(ax)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Kalman trackers vs PEPG across α (white floor {w:.0%}, S={S}, Λ={lam:.2f})")
    _save(plt, fig, "filter_alpha.png")


def experiment_lambda(env, args, out):
    S_grid = [int(x) for x in args.S_grid.split(",")]
    print(f"\n=== Lambda sweep (alpha={args.alpha}, white={args.white}, candidates/drift update={args.candidates_per_drift_update:g}) ===")
    rows = _sweep(env, args, out, "lambda",
                  [dict(alpha=args.alpha, white=args.white, E=args.candidates_per_drift_update, S=S) for S in S_grid], "S")
    if args.plot:
        plot_lambda(rows)


def plot_lambda(rows):
    plt = _pyplot()
    alpha, white = rows[0]["alpha"], rows[0]["white"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = [r["Lambda"] for r in rows]
    _plot_methods(axes[0], x, rows, "c_s2")
    _plot_methods(axes[1], x, rows, "ratio")
    axes[0].set_ylabel("control error $c / s^2$")
    axes[1].set_ylabel("$c / c_{Exact}$")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("information budget Λ = κ s⁴ (varied through S)")
        _style_axes(ax)
    _plain_log(axes[0])
    axes[0].legend(frameon=False)
    fig.suptitle(f"Information budget sweep (α={alpha:g}, white floor {white:.0%})")
    _save(plt, fig, "filter_lambda.png")


def experiment_white(env, args, out):
    whites = [float(x) for x in args.white_grid.split(",")]
    S = args.shots_per_candidate
    print(f"\n=== White-floor sweep (alpha={args.alpha}, candidates/drift update={args.candidates_per_drift_update:g}, S={S}) ===")
    rows = _sweep(env, args, out, "white",
                  [dict(alpha=args.alpha, white=w, E=args.candidates_per_drift_update, S=S) for w in whites], "white")
    if args.plot:
        plot_white(rows)


def plot_white(rows):
    plt = _pyplot()
    alpha, S = rows[0]["alpha"], rows[0]["S"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = [r["white"] for r in rows]
    _plot_methods(axes[0], x, rows, "c_s2")
    _plot_methods(axes[1], x, rows, "ratio")
    axes[0].set_ylabel("control error $c / s^2$")
    axes[1].set_ylabel("$c / c_{Exact}$")
    for ax in axes:
        ax.set_xlabel("white-floor fraction of the drift variance")
        _style_axes(ax)
    axes[0].legend(frameon=False)
    fig.suptitle(f"White-floor sweep (α={alpha:g}, S={S})")
    _save(plt, fig, "filter_white.png")


def experiment_stim(env, args, out):
    S, E = args.shots_per_candidate, args.candidates_per_drift_update
    print(f"\n=== Stim readings (alpha={args.alpha}, white={args.white}, candidates/drift update={E:g}, S={S}) ===")
    mode = env.mode
    env.mode = "gaussian"
    g = compare(env, args.alpha, args.white, E, S, args, R=args.stim_replicas, n_candidates=args.stim_candidates)
    print_condition(g)
    env.mode = "stim"
    tuned = {k: g[k] for k in ("tau_ml", "tau_oracle", "gain")}
    r = compare(env, args.alpha, args.white, E, S, args, tuned=tuned, R=args.stim_replicas, n_candidates=args.stim_candidates)
    env.mode = mode
    print_condition(r)
    out["stim"] = {"gaussian": g, "stim": r}


def replot(path: str):
    """Redraws every plot whose results are in the JSON file."""
    with open(path) as f:
        out = json.load(f)
    if "validate" in out:
        plot_validate(out["validate"], out["config"]["batch_size"])
    alpha = {float(k[len("alpha_E"):]): v for k, v in out.items() if k.startswith("alpha_E")}
    if alpha:
        plot_alpha(dict(sorted(alpha.items(), reverse=True)))
    if "lambda" in out:
        plot_lambda(out["lambda"])
    if "white" in out:
        plot_white(out["white"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=["validate", "alpha", "lambda", "white", "stim", "all"], default="all")
    p.add_argument("--distance", type=int, default=3)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--param", type=int, default=0, help="index of the drifting CX coupler")
    p.add_argument("--sensitivity", type=float, default=0.025)
    p.add_argument("--drift-std", type=float, default=0.5, help="stationary drift std s")
    p.add_argument("--shots-per-candidate", type=int, default=50, help="S (one candidate)")
    p.add_argument("--batch-size", type=int, default=20, help="B, candidates per epoch (even)")
    p.add_argument("--sigma-p", type=float, default=0.25, help="probe rms")
    p.add_argument("--alpha", type=float, default=0.67)
    p.add_argument("--white", type=float, default=0.1, help="white-floor fraction")
    p.add_argument("--candidates-per-drift-update", type=float, default=140.0, help="7000 shots / S=50, as the full code")
    p.add_argument("--pair-order", choices=["split", "adjacent"], default="split",
                   help="split: partners B/2 candidates apart (as run_experiment.py); adjacent: consecutive candidates")
    p.add_argument("--replicas", type=int, default=32)
    p.add_argument("--candidates", type=int, default=30_000, help="candidates per run (candidates/drift update > 1)")
    p.add_argument("--candidates-e1", type=int, default=12_000, help="candidates per run at 1 candidate per drift update")
    p.add_argument("--replicas-e1", type=int, default=16, help="replicas at 1 candidate per drift update (the exact filter folds every candidate)")
    p.add_argument("--alphas", type=str, default="0,0.5,0.67,1,1.5,2")
    p.add_argument("--S-grid", type=str, default="12,50,200,800")
    p.add_argument("--white-grid", type=str, default="0,0.1,0.3,0.6")
    p.add_argument("--validate-taus", type=str, default="30,100,300,1000", help="AR(1) tau in candidates")
    p.add_argument("--candidates-validate", type=int, default=20_000)
    p.add_argument("--stim-replicas", type=int, default=8)
    p.add_argument("--stim-candidates", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=163)
    p.add_argument("--plot", action="store_true", help="save filter_<experiment>.png")
    p.add_argument("--replot", action="store_true", help="only redraw the plots from --out (no simulation)")
    p.add_argument("--out", type=str, default=os.path.join(RES_DIR, "filter_results.json"))
    args = p.parse_args()
    if args.batch_size % 2:
        raise ValueError("--batch-size must be even (antithetic pairs)")
    if args.replot:
        replot(args.out)
        return

    t0 = time.time()
    env = MinimalControlEnv(mode="gaussian", distance=args.distance, rounds=args.rounds, param_index=args.param,
                            sensitivity=args.sensitivity, drift_std=args.drift_std, seed=args.seed)
    c = env.summary()
    S = args.shots_per_candidate
    print(f"=== Calibration (d={args.distance}, rounds={args.rounds}, CX parameter {args.param}) ===")
    print(f"  y = {c['y0']:.4f} + H_l e^2:  H_l={c['H_l']:.4f}   sigma_1={c['sigma_1']:.4f}")
    print(f"  S={S}: kappa={env.kappa(S):.2f}  Lambda=kappa s^4={env.Lambda(S):.3f}  (s={env.s})")
    print(f"  setup {time.time() - t0:.1f}s\n")

    # Merge into an existing results file, so experiments can be (re)run one at a time
    out = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            out = json.load(f)
    out.update({"config": vars(args), "calibration": c})
    todo = ["validate", "alpha", "lambda", "white", "stim"] if args.experiment == "all" else [args.experiment]
    fns = {"validate": experiment_validate, "alpha": experiment_alpha, "lambda": experiment_lambda,
           "white": experiment_white, "stim": experiment_stim}
    for name in todo:
        fns[name](env, args, out)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    print(f"\nResults saved to '{args.out}'.  Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
