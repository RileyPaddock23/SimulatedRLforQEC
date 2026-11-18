"""
Sinusoidal vs AR(1) Drift on the Minimal Control Model
======================================================
One drifting CX coupler i of a d=3 surface code (stim mode), steered by the minimal antithetic
SGD controller. The K couplers sharing the most detectors with M_i sit at a fixed offset
ebar_nbr: they never move and are never probed, they only raise the detector-flip rates and so
the shot noise of each reading.

Two drift families for p*_i(t) (t in candidate times; one candidate time = S shots, one update = B candidate times):
  ar1:  x_{t+1} = a x_t + w_t,  a = exp(-1/tau),  stationary std s           (rms = s)
  sin:  A sin(2 pi t / T + phi),  phi uniform                                  (rms = A / sqrt 2)

Both are swept over (timescale, rms) on the same grids, with the gain tuned per point. The metric
is c = E[(mu - p*)^2], reported normalized as c / rms^2 (1 = no better than sitting at the mean).

Matching: for each sinusoid point we find the tau whose AR(1) error at the same rms equals the
sinusoid's error, tau_eq(T). A constant factor T = alpha tau would show up as slope 1 on log-log.

Usage:
    python -m analytics.sin_vs_ar1_sweep            # run + plot
    python -m analytics.sin_vs_ar1_sweep --plot-only
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from analytics.minimal_control_env import MinimalControlEnv
from rl_agent import RLControlAgent

OUT_DIR = os.path.join("figures", "sin_vs_ar1")
RESULTS = os.path.join(OUT_DIR, "sin_vs_ar1_results.json")

CFG = dict(
    S=50, B=20, sigma_p=0.9, n_neighbors=4, ebar_nbr=2.0,
    timescales=[10, 30, 100, 300, 1000, 3000, 10000],  # tau (ar1) and T (sin), in candidate times
    rms=[0.5, 1.0, 1.75, 2.5, 3.5],
    gains=[0.03, 0.06, 0.12, 0.25, 0.5, 1.0],
    reps=4, seed=163,
)

_ENV = None


def make_env(cfg, seed):
    probe = MinimalControlEnv(mode="stim", calib_shots=20_000, seed=seed)
    n_cx = probe.code.backend.num_cx_gates
    overlap = probe.code.M[probe.M_i, :n_cx].sum(axis=0)
    overlap[probe.i] = -1
    nbrs = [int(j) for j in np.argsort(-overlap, kind="stable")[: cfg["n_neighbors"]]]
    return MinimalControlEnv(mode="stim", neighbor_index=nbrs, seed=seed, calib_shots=100_000)


def trajectory(kind, scale, rms, n, rng):
    """p*_i(t) for t = 0..n-1."""
    if kind == "sin":
        return np.sqrt(2) * rms * np.sin(2 * np.pi * np.arange(n) / scale + rng.uniform(0, 2 * np.pi))
    a = np.exp(-1.0 / scale)
    w = rms * np.sqrt(1 - a * a) * rng.standard_normal(n)
    x = np.empty(n)
    x[0] = rms * rng.standard_normal()
    for t in range(1, n):
        x[t] = a * x[t - 1] + w[t]
    return x


def n_steps_for(kind, scale):
    if kind == "sin":
        return int(min(max(8 * scale, 20_000), 80_000))
    return int(min(max(30 * scale, 20_000), 150_000))


def closed_loop(env, cfg, p, gain, seed, return_trace=False):
    """Runs the controller on p*(t); returns c over the candidate times after burn-in (and optionally mu(t))."""
    S, B, sp = cfg["S"], cfg["B"], cfg["sigma_p"]
    lr = gain / (2.0 * env.H_l * sp ** 2)
    agent = RLControlAgent(
        num_params=1, num_detectors=1, mask_matrix=np.ones((1, 1)), lr_mu=lr * sp ** 2,
        lr_sigma=0.0, entropy_reg=0.0, sigma_init=sp, sigma_min=sp, sigma_max=sp,
        grad_clip=0.0, use_adam=False, seed=seed,
    )
    agent.mu[:] = p[0]
    e_nbr = np.full(env.n - 1, cfg["ebar_nbr"])
    n = (len(p) // B) * B
    mu = np.empty(n)
    obs = np.empty((B, 1), dtype=np.float32)
    for u in range(n // B):
        cands = agent.sample_policies(B, antithetic=True)
        for k in range(B):
            t = u * B + k
            mu[t] = agent.mu[0]
            obs[k, 0] = env.read(np.concatenate([[cands[k, 0] - p[t]], e_nbr]), S)
        agent.update(cands, obs)
    burn = n // 5
    c = float(np.mean((mu[burn:] - p[burn:n]) ** 2))
    return (c, mu) if return_trace else c


def _init(cfg, seed):
    global _ENV
    _ENV = make_env(cfg, seed)


def run_point(job):
    kind, scale, rms, cfg, seed = job
    rng = np.random.default_rng(seed)
    _ENV.rng = np.random.default_rng(seed + 1)
    n = n_steps_for(kind, scale)
    # tune the gain on one realization, then evaluate on fresh ones
    p_tune = trajectory(kind, scale, rms, n // 2, rng)
    tuned = [closed_loop(_ENV, cfg, p_tune, g, seed) for g in cfg["gains"]]
    g = cfg["gains"][int(np.argmin(tuned))]
    cs = [closed_loop(_ENV, cfg, trajectory(kind, scale, rms, n, rng), g, seed + 10 + r) for r in range(cfg["reps"])]
    return dict(kind=kind, scale=scale, rms=rms, gain=g, c=float(np.mean(cs)),
                c_se=float(np.std(cs, ddof=1) / np.sqrt(len(cs))), c_tune=tuned)


def run(cfg, workers):
    jobs, k = [], 0
    for kind in ("ar1", "sin"):
        for scale in cfg["timescales"]:
            for rms in cfg["rms"]:
                jobs.append((kind, scale, rms, cfg, cfg["seed"] + 1000 * k))
                k += 1
    # longest jobs first so the pool finishes evenly
    jobs.sort(key=lambda j: -n_steps_for(j[0], j[1]))
    t0 = time.time()
    out = []
    with Pool(workers, initializer=_init, initargs=(cfg, cfg["seed"])) as pool:
        for r in pool.imap_unordered(run_point, jobs):
            out.append(r)
            print(f"[{len(out):3d}/{len(jobs)}  {time.time() - t0:6.0f}s] {r['kind']:>3} scale={r['scale']:>6} "
                  f"rms={r['rms']:.2f}  gain={r['gain']:.2f}  c/rms^2={r['c'] / r['rms'] ** 2:.3f}", flush=True)

    # one example trajectory per family for the illustration panel
    env = make_env(cfg, cfg["seed"])
    rng = np.random.default_rng(7)
    examples = {}
    for kind in ("ar1", "sin"):
        best = {r["scale"]: r["gain"] for r in out if r["kind"] == kind and r["rms"] == 1.75}
        p = trajectory(kind, 1000, 1.75, 6000, rng)
        _, mu = closed_loop(env, cfg, p, best[1000], 5, return_trace=True)
        examples[kind] = dict(p=p.tolist(), mu=mu.tolist(), gain=best[1000])
    env_info = {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item()) for k, v in env.summary().items()}
    env_info.update(neighbors=env.neighbors, C_nbr=env.C[1:].tolist(),
                    sigma_1_with_nbrs=float(np.std(env.sample_shots(np.r_[0.0, [cfg["ebar_nbr"]] * (env.n - 1)], 100_000))))
    return dict(cfg=cfg, env=env_info, results=out, examples=examples)


# --------------------------------------------------------------------- analysis
def grid(data, kind):
    cfg = data["cfg"]
    G = np.full((len(cfg["rms"]), len(cfg["timescales"])), np.nan)
    E = np.full_like(G, np.nan)
    for r in data["results"]:
        if r["kind"] == kind:
            i, j = cfg["rms"].index(r["rms"]), cfg["timescales"].index(r["scale"])
            G[i, j] = r["c"] / r["rms"] ** 2
            E[i, j] = r["c_se"] / r["rms"] ** 2
    return G, E


def tau_equivalent(data, max_err=0.8):
    """For each (rms, T): the AR(1) tau with the same normalized error at that rms (log-log interpolation
    along the AR(1) curve, which decreases monotonically in tau). NaN when outside the AR(1) range, or
    when the sinusoid is not tracked (c / rms^2 > max_err): there both errors sit at ~1 and the match is noise."""
    cfg = data["cfg"]
    ts = np.log(cfg["timescales"])
    Gar, _ = grid(data, "ar1")
    Gsin, _ = grid(data, "sin")
    out = np.full_like(Gsin, np.nan)
    for i in range(len(cfg["rms"])):
        y = np.log(np.minimum.accumulate(Gar[i]))  # enforce monotone (noise at the flat ends)
        for j in range(len(cfg["timescales"])):
            target = np.log(Gsin[i, j])
            if Gsin[i, j] > max_err or not (y[-1] <= target <= y[0]):
                continue
            k = int(np.argmax(y <= target))
            if k == 0:
                out[i, j] = cfg["timescales"][0]
                continue
            f = (y[k - 1] - target) / (y[k - 1] - y[k]) if y[k - 1] != y[k] else 0.0
            out[i, j] = np.exp(ts[k - 1] + f * (ts[k] - ts[k - 1]))
    return out


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    cfg = data["cfg"]
    ts, rms = np.array(cfg["timescales"], float), np.array(cfg["rms"], float)
    Gar, Ear = grid(data, "ar1")
    Gsin, Esin = grid(data, "sin")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    paths = []

    # --- Fig 1: example trajectories
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, kind, title in zip(axes, ("ar1", "sin"), ("AR(1): τ = 1000 candidate times, s = 1.75", "Sinusoid: T = 1000 candidate times, A = 1.75·√2")):
        ex = data["examples"][kind]
        ax.plot(ex["p"], color=colors[0], lw=1.2, label="optimum $p^*(t)$")
        ax.plot(ex["mu"], color=colors[1], lw=1.2, label=f"controller $\\mu(t)$ (gain {ex['gain']})")
        ax.set_title(title)
        ax.set_xlabel("candidate time")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("offset (tolerance units)")
    fig.tight_layout()
    paths.append(os.path.join(OUT_DIR, "examples.png"))
    fig.savefig(paths[-1], dpi=140)
    plt.close(fig)

    # --- Fig 2: side-by-side heatmaps, shared color scale
    vmin = np.nanmin([Gar, Gsin])
    vmax = max(np.nanmax([Gar, Gsin]), 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2), sharey=True)
    for ax, G, title, xl in zip(axes[:2], (Gar, Gsin), ("AR(1)", "Sinusoid"),
                                ("correlation time τ (candidate times)", "period T (candidate times)")):
        im = ax.pcolormesh(np.arange(len(ts) + 1), np.arange(len(rms) + 1), G, norm=LogNorm(vmin, vmax), cmap="viridis_r")
        for i in range(len(rms)):
            for j in range(len(ts)):
                ax.text(j + 0.5, i + 0.5, f"{G[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if G[i, j] > np.sqrt(vmin * vmax) else "black")
        ax.set_xticks(np.arange(len(ts)) + 0.5, [f"{t:g}" for t in ts])
        ax.set_yticks(np.arange(len(rms)) + 0.5, [f"{r:g}" for r in rms])
        ax.set_xlabel(xl)
        ax.set_title(title)
    axes[0].set_ylabel("drift rms  (s  or  A/√2)")
    fig.colorbar(im, ax=axes[:2], label="tracking error  $c\\,/\\,\\mathrm{rms}^2$", shrink=0.9, pad=0.02)

    # difference panel: sinusoid - AR(1) at the same (timescale, rms); diverging map centered at 0
    from matplotlib.colors import TwoSlopeNorm
    D = Gsin - Gar
    lim = np.nanmax(np.abs(D))
    ax = axes[2]
    imd = ax.pcolormesh(np.arange(len(ts) + 1), np.arange(len(rms) + 1), D,
                        norm=TwoSlopeNorm(0.0, -lim, lim), cmap="RdBu_r")
    for i in range(len(rms)):
        for j in range(len(ts)):
            ax.text(j + 0.5, i + 0.5, f"{D[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(D[i, j]) > 0.6 * lim else "black")
    ax.set_xticks(np.arange(len(ts)) + 0.5, [f"{t:g}" for t in ts])
    ax.set_xlabel("τ = T (candidate times)")
    ax.set_title("Sinusoid − AR(1)")
    fig.colorbar(imd, ax=ax, label="Δ $c\\,/\\,\\mathrm{rms}^2$  (red: sinusoid worse)", shrink=0.9, pad=0.02)
    paths.append(os.path.join(OUT_DIR, "heatmaps.png"))
    fig.savefig(paths[-1], dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 3: normalized error vs timescale, per rms; both families on each panel
    sel = [0, 2, 4] if len(rms) >= 5 else list(range(len(rms)))
    fig, axes = plt.subplots(1, len(sel), figsize=(4.2 * len(sel), 3.8), sharey=True)
    for ax, i in zip(np.atleast_1d(axes), sel):
        ax.errorbar(ts, Gar[i], yerr=Ear[i], fmt="o-", color=colors[0], capsize=2, label="AR(1) vs τ")
        ax.errorbar(ts, Gsin[i], yerr=Esin[i], fmt="s--", color=colors[1], capsize=2, label="sinusoid vs T")
        ax.axhline(1.0, color="k", lw=0.8, ls=":", label="no tracking")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("τ or T (candidate times)")
        ax.set_title(f"rms = {rms[i]:g}")
        ax.grid(True, ls="--", alpha=0.5, which="both")
    np.atleast_1d(axes)[0].set_ylabel("$c\\,/\\,\\mathrm{rms}^2$")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    paths.append(os.path.join(OUT_DIR, "error_vs_timescale.png"))
    fig.savefig(paths[-1], dpi=140)
    plt.close(fig)

    # --- Fig 4: equivalence maps. (a) tau_eq vs T, (b) absolute c vs rms at matched timescales
    teq = tau_equivalent(data)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    allT, allTau = [], []
    for i, r in enumerate(rms):
        m = np.isfinite(teq[i])
        ax.plot(ts[m], teq[i][m], "o-", color=colors[i % len(colors)], label=f"rms = {r:g}")
        allT += list(ts[m]); allTau += list(teq[i][m])
    Td = np.geomspace(ts[0], ts[-1], 50)
    if len(allT) >= 3:
        slope, icpt = np.polyfit(np.log(allT), np.log(allTau), 1)
        Tf = np.geomspace(min(allT), max(allT), 50)
        ax.plot(Tf, np.exp(icpt) * Tf ** slope, "k--", lw=1, label=f"fit: τ ∝ T$^{{{slope:.2f}}}$")
        # linear-loop prediction tau_eq ∝ T^(4/3), anchored at the geometric mean of the matched points
        k43 = np.exp(np.mean(np.log(allTau) - 4 / 3 * np.log(allT)))
        ax.plot(Tf, k43 * Tf ** (4 / 3), color="gray", ls="-.", lw=1, label="theory slope 4/3")
        data["fit"] = dict(slope=float(slope), prefactor=float(np.exp(icpt)))
    ax.plot(Td, Td, color="gray", ls=":", lw=1, label="τ = T (constant factor)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sinusoid period T (candidate times)")
    ax.set_ylabel("AR(1) τ with equal error (candidate times)")
    ax.set_title("(a) equal-error timescale (tracked points only)")
    ax.grid(True, ls="--", alpha=0.5, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    # normalized error vs rms at the two longest timescales, with the linear-loop power laws
    slopes = {}
    for j, jt in enumerate([len(ts) - 2, len(ts) - 1]):
        for G, fmt, name, sym in ((Gar, "o-", "AR(1)", "τ"), (Gsin, "s--", "sin", "T")):
            sl = float(np.polyfit(np.log(rms), np.log(G[:, jt]), 1)[0])
            slopes[f"{name}_{ts[jt]:g}"] = sl
            ax.plot(rms, G[:, jt], fmt, color=colors[j],
                    label=f"{name}, {sym} = {ts[jt]:g}  (slope {sl:.2f})")
    r0 = rms[len(rms) // 2]
    jt = len(ts) - 1
    ax.plot(rms, Gar[len(rms) // 2, jt] * (rms / r0) ** -1, color="gray", ls=":", lw=1, label="theory AR(1): rms$^{-1}$")
    ax.plot(rms, Gsin[len(rms) // 2, jt] * (rms / r0) ** (-4 / 3), color="gray", ls="-.", lw=1, label="theory sin: rms$^{-4/3}$")
    data["rms_slopes"] = slopes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("drift rms  (s  or  A/√2)")
    ax.set_ylabel("$c\\,/\\,\\mathrm{rms}^2$")
    ax.set_title("(b) error vs drift size (tracked regime)")
    ax.grid(True, ls="--", alpha=0.5, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    paths.append(os.path.join(OUT_DIR, "equivalence.png"))
    fig.savefig(paths[-1], dpi=140)
    plt.close(fig)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.plot_only:
        with open(RESULTS) as f:
            data = json.load(f)
    else:
        data = run(CFG, args.workers)
    paths = plot(data)
    with open(RESULTS, "w") as f:
        json.dump(data, f)
    print("saved:", *paths, RESULTS, sep="\n  ")
    if "fit" in data:
        print(f"tau_eq ~ {data['fit']['prefactor']:.3g} T^{data['fit']['slope']:.2f}")


if __name__ == "__main__":
    main()
