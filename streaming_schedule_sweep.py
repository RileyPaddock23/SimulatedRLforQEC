"""
Streaming Schedule Sweep: batch size x cycles per candidate x drift period
==========================================================================
How should a streaming controller split its data between candidates (B) and cycles per
candidate (S), and how slow must the drift be for a given N = B * S to steer it?

Two modes, both on the d=5 surface code at noise scale --scale (base error = error budget),
steering 2q, 1q and qubit parameters (reset and readout drift but stay frozen):

  loop    Closed-loop single-shot streaming runs, one per (B, S, drift period, seed, lr,
          optimizer). Schedule per update: B candidates x S cycles, then the update is
          computed and loaded after a fixed --latency (cycles), during which the hardware
          holds the old mean ("wait"); the new mean applies from the next batch. The probe
          width sigma is held fixed. Drift: every parameter's optimum is a sinusoid of
          amplitude A and period P cycles with its own random phase. Runs start calibrated
          (mu = p_opt(0)) and are scored after --burn updates.

  cosine  Frozen-time gradient quality per (B, S): the steered parameters are mis-set by a
          known random e (rms --e-rms), one batch is measured, and the PEPG gradient is
          compared with the direction to the optimum, -e.

Loop metrics (over the scored updates; tracked = steered parameters any cycle detector sees):
  - tracking_rms: rms over time and parameters of mu - p_opt, overall and per kind.
  - r_mean / r_insitu: steering advantage in total excess physical error,
        r = 1 - excess(policy) / excess(unsteered),
    computed exactly from the quadratic error model (excess = sum_s curvature[s, i] e_i^2).
    "unsteered" is a device calibrated at a random time and never touched again, whose
    long-run <e_i^2> is A^2 for this drift. r_insitu also pays for exploration (+ sigma^2
    per steered parameter). r = 1 is perfect tracking, r = 0 no better than not steering,
    r < 0 worse. Excess is sampled at random times within each update period.
  - r_hold: the in-situ advantage against holding the t = 0 calibration for the same run
    (scored updates only). For drift slow next to the run this is the relevant reference:
    it goes negative when steering jitter and probing cost more than the drift does.
  - cos_loop: cosine between each update's raw gradient and -(mu - p_opt) at the batch's midpoint.

Results append to a JSON-lines file as jobs finish; rerunning skips finished jobs.

  long    One schedule (default B=80 x S=2000, period 1e8 cycles) run for several full drift
          periods, so the loop reaches steady state and has to follow the optimum across its
          whole range (the sweep's runs cover a small fraction of a slow period). Several
          controllers in parallel: plain per-type SGD gains (--sgd-gains, needs --response)
          and Adam (--adam-lr). Steady-state numbers use the updates after the first period;
          tracking-rms references: calibrated once at a random time (rms A, the sweep's r = 0),
          fixed at the drift center (rms A/sqrt 2), and holding the t = 0 calibration.

    python streaming_schedule_sweep.py loop --dry-run
    python streaming_schedule_sweep.py loop
    python streaming_schedule_sweep.py cosine
    python streaming_schedule_sweep.py plot
    python streaming_schedule_sweep.py long            # run + plot
    python streaming_schedule_sweep.py long --replot   # plot from the saved .npz files
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

CPE = 1_000_000.0  # env time unit: one "epoch" = 1e6 cycles, so frequency in 1/cycle = drift_frequency / CPE
STEER = ("2q", "1q", "qubit")
KINDS = ("2q", "1q", "qubit")
PER_PROC_THROUGHPUT = 5.0e4  # cycles/s per process with ~18 running (measured, decoder off)

DEFAULTS = dict(
    B=[20, 40, 80, 160, 320, 640],
    S=[250, 500, 1000, 2000, 4000, 8000],
    periods=[1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9],
    seeds=[1, 2],
)
LONG_DEFAULTS = dict(B=[80], S=[2000], periods=[1e8], seeds=[1])
LONG_OUT = "runs/streaming/long_run"
LONG_FIG = "figures/streaming/long_run.png"


# ---------------------------------------------------------------------- setup

def make_env(seed: int, period: float, amplitude: float, scale: float, drift: str = "sin",
             sensitivity: str = "equalized", max_cycles: float = 1e6):
    """
    drift "sin": A sin(2 pi t / P + phi_i), random phase per parameter. drift "ar1": AR(1)
    with correlation time tau = P / (2 pi) (same corner frequency 1/P) and stationary std
    A / sqrt(2) (same rms as the sinusoid, so the long-run unsteered <e^2> is A^2 for both).
    sensitivity "equalized": a unit mis-set of any parameter adds one error budget of total
    physical error (env.equalize_sensitivities); "raw": the env's per-kind SI1000 scaling
    with the default Omega spread.
    """
    from run_experiment import build_env
    from code_envs.noise_envs import Ar1Drift, SinusoidalDrift
    a = argparse.Namespace(code="surface", distance=5, rounds=5, base_error=scale, error_budget=scale,
                           drift="sinusoidal", drift_frequency=CPE / period, drift_amplitude=amplitude,
                           initial_offset=0.0, qubit_initial_offset=0.0, seed=seed)
    env = build_env(a, max_epochs=2)
    # Re-draw each block's drift (build_env has no flags for random phases or a single AR(1) tau)
    rng = np.random.default_rng([seed, 2])
    if drift == "sin":
        model = SinusoidalDrift(CPE / period, amplitude, random_phase=True)
    elif drift == "ar1":
        tau, sd = period / (2 * np.pi * CPE), amplitude / np.sqrt(2)
        model = Ar1Drift(tau_range=(tau, tau), sigma_range=(sd, sd), max_epochs=int(np.ceil(max_cycles / CPE)) + 1)
    else:
        raise ValueError(f"unknown drift '{drift}'")
    env._drift_opts = [model.build(b.num_params, b.params_per_owner, rng) for b in env.blocks]
    if sensitivity == "equalized":
        env.equalize_sensitivities()
    elif sensitivity != "raw":
        raise ValueError(f"unknown sensitivity mode '{sensitivity}'")
    return env


def schedule_length(cfg):
    """Updates per run and how many of them are burn-in: at least cfg['updates'] and cfg['min_cycles'] cycles."""
    T_u = cfg["B"] * cfg["S"] + cfg["latency"]
    n = max(cfg["updates"], int(np.ceil(cfg["min_cycles"] / T_u)))
    return n, max(cfg["burn"], n // 5)


def make_stream(env, seed: int, period: float):
    from code_envs.streaming_env import StreamingQECEnv
    du = int(max(1, min(1000, period // 200)))  # >= 200 drift updates per period
    return StreamingQECEnv(env, cycles_per_epoch=CPE, drift_update_cycles=du, decode=False,
                           rebase_cycles=2048, seed=seed + 7919)


def make_sweep_agent(stream, seed, mask, sigma, lr, optimizer, kinds=None, response=None):
    """
    optimizer "adam": the agent's Adam step with learning rate lr. "sgd": plain gradient steps
    with a per-kind gain lr / k_kind, where k_kind is the measured mean response of the
    gradient to a mis-set (E[g_i] = -k e_i, see `response`), so each update corrects a
    fraction lr of the expected mis-set; no gradient clipping.
    """
    from run_experiment import make_agent
    agent = make_agent(stream, seed=seed, steer_mask=mask, sigma_init=sigma, sigma_min=sigma, sigma_max=sigma,
                       lr_mu=lr, lr_sigma=0.0)
    if optimizer == "sgd":
        agent.use_adam = False
        agent.grad_clip = 0.0
        agent.lr_mu = np.array([lr / response[k] if k in response else 0.0 for k in kinds], dtype=np.float32)
    return agent


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def job_key(cfg):
    return json.dumps({k: cfg[k] for k in sorted(cfg)}, sort_keys=True)


# ---------------------------------------------------------------------- closed loop

def run_loop(cfg):
    from run_streaming_experiment import param_kinds, steer_mask
    t0 = time.time()
    B, S, P, seed = cfg["B"], cfg["S"], cfg["period"], cfg["seed"]
    sigma, A, L = cfg["sigma"], cfg["amplitude"], cfg["latency"]
    updates, burn = schedule_length(cfg)
    env = make_env(seed, P, A, cfg["scale"], cfg["drift"], cfg["sensitivity"], updates * (B * S + L))
    stream = make_stream(env, seed, P)
    mask = steer_mask(env, STEER)
    tracked = mask & stream.active_params
    kinds = param_kinds(env)
    agent = make_sweep_agent(stream, seed, mask, sigma, cfg["lr"], cfg["optimizer"], kinds, cfg.get("response"))
    rng = np.random.default_rng([seed, 3])

    curv = env.curvature[:, tracked].sum(axis=0)  # total excess error per unit e_i^2
    excess_ref = float(curv.sum() * A ** 2)
    excess_probe = float(curv.sum() * sigma ** 2)

    mu0 = env.get_drift_opt(0.0)  # calibrated at t = 0
    agent.mu = mu0.astype(np.float32).copy()  # unsteered parameters stay at this calibration
    hw_mu = mu0.astype(np.float32).copy()
    stream.reset(hw_mu)

    rows = {"excess": [], "excess_hold": [], "e2": [], "e2_kind": [], "cos": [], "cos_kind": [], "edr": []}
    trace = {"cycle": [], "mu": [], "p_opt": [], "e2": [], "e2_hold": []} if cfg.get("trace") else None
    for n in range(updates):
        c_start = stream.cycle
        z = agent.sample_directions(B, antithetic=True)
        cands = hw_mu + sigma * z
        obs = np.empty((B, stream.num_detectors), dtype=np.float32)
        edr = 0.0
        for k in range(B):
            block = stream.run_cycles(cands[k], S)
            obs[k] = block.detection_events[cfg["discard"]:].mean(axis=(0, 1))
            edr += float(block.detection_events.mean())
        e_mid = (hw_mu - env.get_drift_opt(stream.time_of((c_start + stream.cycle) / 2)))[tracked]
        agent.update(cands, obs)
        g = agent.last_grad_mu[tracked]
        stream.run_cycles(hw_mu, L)  # latency: the old mean holds until the update is loaded
        # hw_mu was active from c_start to now: sample its excess at random times
        ts = rng.uniform(c_start, stream.cycle, size=cfg["time_samples"])
        p_ts = [env.get_drift_opt(stream.time_of(c))[tracked] for c in ts]
        e2 = np.mean([(hw_mu[tracked] - q) ** 2 for q in p_ts], axis=0)
        e2_hold = np.mean([(mu0[tracked] - q) ** 2 for q in p_ts], axis=0)
        if trace is not None:  # every update, burn-in included: time, applied mean, optimum at the period's midpoint
            c_mid = (c_start + stream.cycle) / 2
            trace["cycle"].append(c_mid)
            trace["mu"].append(hw_mu[tracked].copy())
            trace["p_opt"].append(env.get_drift_opt(stream.time_of(c_mid))[tracked])
            trace["e2"].append(e2)
            trace["e2_hold"].append(e2_hold)
        hw_mu = agent.mu.copy()
        if n < burn:
            continue
        rows["excess"].append(float(curv @ e2))
        rows["excess_hold"].append(float(curv @ e2_hold))
        rows["e2"].append(float(e2.mean()))
        rows["e2_kind"].append({kd: float(e2[kinds[tracked] == kd].mean()) for kd in KINDS})
        rows["cos"].append(cosine(g, -e_mid))
        rows["cos_kind"].append({kd: cosine(g[kinds[tracked] == kd], -e_mid[kinds[tracked] == kd]) for kd in KINDS})
        rows["edr"].append(edr / B)

    excess = float(np.mean(rows["excess"]))
    excess_hold = float(np.mean(rows["excess_hold"]))
    return {**cfg, "mode": "loop", "N": B * S, "T_update": B * S + L, "updates_run": updates, "burn_run": burn,
            "r_mean": 1.0 - excess / excess_ref, "r_insitu": 1.0 - (excess + excess_probe) / excess_ref,
            "r_hold": 1.0 - (excess + excess_probe) / max(excess_hold, 1e-30), "excess_hold": excess_hold,
            "tracking_rms": float(np.sqrt(np.mean(rows["e2"]))),
            "tracking_rms_kind": {kd: float(np.sqrt(np.mean([r[kd] for r in rows["e2_kind"]]))) for kd in KINDS},
            "cos_loop": float(np.nanmean(rows["cos"])),
            "cos_loop_kind": {kd: float(np.nanmean([r[kd] for r in rows["cos_kind"]])) for kd in KINDS},
            "edr_insitu": float(np.mean(rows["edr"])), "excess_ref": excess_ref, "excess_probe": excess_probe,
            "per_update": {k: rows[k] for k in ("excess", "e2", "cos")},
            "cycles": stream.cycle, "seconds": time.time() - t0,
            **({"trace": {k: np.asarray(v) for k, v in trace.items()}, "kinds": kinds[tracked]} if trace else {})}


# ---------------------------------------------------------------------- frozen-time gradient quality

def run_cosine(cfg):
    from run_streaming_experiment import param_kinds, steer_mask
    t0 = time.time()
    B, S, seed = cfg["B"], cfg["S"], cfg["seed"]
    env = make_env(seed, 1e9, cfg["amplitude"], cfg["scale"], "sin", cfg["sensitivity"])
    stream = make_stream(env, seed, 1e9)
    stream.time_of = lambda c: 0.0  # time frozen
    mask = steer_mask(env, STEER)
    tracked = mask & stream.active_params
    kinds = param_kinds(env)[tracked]
    agent = make_sweep_agent(stream, seed, mask, cfg["sigma"], 0.1, "adam")
    rng = np.random.default_rng([seed, 4])
    e = np.where(tracked, cfg["e_rms"] * rng.standard_normal(env.num_params), 0.0).astype(np.float32)
    mu = (env.get_drift_opt(0.0) + e).astype(np.float32)
    agent.mu = mu.copy()
    stream.reset(mu)
    z = agent.sample_directions(B, antithetic=True)
    cands = mu + cfg["sigma"] * z
    obs = np.stack([stream.run_cycles(cands[k], S).detection_events[cfg["discard"]:].mean(axis=(0, 1))
                    for k in range(B)])
    agent.update(cands, obs)
    g, t = agent.last_grad_mu[tracked], -e[tracked]
    return {**cfg, "mode": "cosine", "N": B * S, "cos": cosine(g, t),
            "cos_kind": {kd: cosine(g[kinds == kd], t[kinds == kd]) for kd in KINDS},
            "g": g.tolist(), "e": e[tracked].tolist(), "kinds": kinds.tolist(), "seconds": time.time() - t0}


def run_job(cfg):
    return run_loop(cfg) if cfg["mode"] == "loop" else run_cosine(cfg)


# ---------------------------------------------------------------------- long run (several drift periods)

LONG_KIND_STYLE = {"2q": ("#2a78d6", "-"), "1q": ("#eb6834", "--"), "qubit": ("#c2448a", "-.")}
LONG_RUN_COLORS = ["#2a78d6", "#2f9e6e", "#eb6834", "#6b6a63"]


def long_tag(cfg):
    return f"{cfg['optimizer']}{cfg['lr']:g}"


def long_job(cfg):
    res = run_loop(cfg)
    tr = res.pop("trace")
    np.savez_compressed(os.path.join(LONG_OUT, f"{long_tag(cfg)}.npz"), kinds=res.pop("kinds"), **tr)
    res.pop("per_update", None)
    with open(os.path.join(LONG_OUT, f"{long_tag(cfg)}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] {long_tag(cfg)} done in {res['seconds']:.0f}s", flush=True)
    return long_tag(cfg)


def long_steady(d, P):
    """Steady-state rms per kind (and overall) from updates after the first period."""
    sel = d["cycle"] > P
    kinds = d["kinds"]
    out = {}
    for name, key in (("steer", "e2"), ("hold", "e2_hold")):
        e2 = d[key][sel]
        out[name] = {kd: float(np.sqrt(e2[:, kinds == kd].mean())) for kd in KINDS}
        out[name]["all"] = float(np.sqrt(e2.mean()))
    return out


def _rolling(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def long_plot(tags, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P, A = cfg["period"], cfg["amplitude"]
    data = {t: dict(np.load(os.path.join(LONG_OUT, f"{t}.npz"))) for t in tags}
    best = min(tags, key=lambda t: long_steady(data[t], P)["steer"]["all"])
    d = data[best]
    x = d["cycle"] / P
    kinds = d["kinds"]

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    # Row 1: one example parameter per kind
    rng = np.random.default_rng(0)
    for j, kd in enumerate(KINDS):
        ax = fig.add_subplot(gs[0, j])
        i = rng.choice(np.flatnonzero(kinds == kd))
        c, _ = LONG_KIND_STYLE[kd]
        ax.plot(x, d["p_opt"][:, i], color="#1f1f1e", lw=1.2, label="optimum $p_{opt}(t)$")
        ax.plot(x, d["mu"][:, i], color=c, lw=1.0, alpha=0.9, label=f"applied mean $\\mu$ ({best})")
        ax.axvspan(0, 1, color="#6b6a63", alpha=0.08, lw=0)
        ax.set_title(f"Example {kd} parameter (#{i})")
        ax.set_xlabel("time (drift periods)")
        ax.set_ylabel("parameter (tolerance units)")
        ax.grid(True, ls="--", alpha=0.35)
        if j == 0:
            ax.legend(fontsize=8, loc="lower left")

    w = 25
    xr = _rolling(x, w)
    # Row 2 left: per-kind rolling rms for the best run
    ax = fig.add_subplot(gs[1, 0:2])
    for kd in KINDS:
        c, ls = LONG_KIND_STYLE[kd]
        m = kinds == kd
        ax.plot(xr, np.sqrt(_rolling(d["e2"][:, m].mean(axis=1), w)), color=c, ls=ls, lw=1.8, label=f"{kd} ({best})")
    ax.plot(xr, np.sqrt(_rolling(d["e2_hold"].mean(axis=1), w)), color="#1f1f1e", lw=1.2, alpha=0.6,
            label="hold t=0 calibration (all)")
    for t, c in zip(tags, LONG_RUN_COLORS):
        if t != best:
            ax.plot(xr, np.sqrt(_rolling(data[t]["e2"].mean(axis=1), w)), color=c, lw=1.0, alpha=0.55,
                    ls=":", label=f"all params, {t}")
    ax.axhline(A, color="#6b6a63", ls="--", lw=1, label="calibrated once at random time (rms A)")
    ax.axhline(A / np.sqrt(2), color="#6b6a63", ls=(0, (1, 2)), lw=1, label="fixed at drift center (rms A/$\\sqrt{2}$)")
    ax.axvspan(0, 1, color="#6b6a63", alpha=0.08, lw=0)
    ax.text(0.5, 0.97, "burn-in (excluded)", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8, color="#6b6a63")
    ax.set_yscale("log")
    ax.set_xlabel("time (drift periods)")
    ax.set_ylabel(f"tracking rms, rolling {w} updates (tolerance)")
    ax.set_title("Tracking error over time")
    ax.grid(True, ls="--", alpha=0.35, which="both")
    ax.legend(fontsize=8, ncol=2, loc="lower right")

    # Row 2 right: steady-state rms per kind per controller
    ax = fig.add_subplot(gs[1, 2])
    cats = list(KINDS) + ["all"]
    width = 0.8 / len(tags)
    for k, (t, c) in enumerate(zip(tags, LONG_RUN_COLORS)):
        s = long_steady(data[t], P)["steer"]
        xs = np.arange(len(cats)) + (k - (len(tags) - 1) / 2) * width
        ax.bar(xs, [s[q] for q in cats], width * 0.92, color=c, label=t)
    ax.axhline(A, color="#6b6a63", ls="--", lw=1)
    ax.axhline(A / np.sqrt(2), color="#6b6a63", ls=(0, (1, 2)), lw=1)
    ax.set_xticks(range(len(cats)), cats)
    ax.set_ylabel("steady-state tracking rms (tolerance)")
    ax.set_title("Steady state (after the first period)")
    ax.grid(True, ls="--", alpha=0.35, axis="y")
    ax.legend(fontsize=8, title="controller")

    fig.suptitle(f"Streaming long run: d=5 surface code, B={cfg['B']} x S={cfg['S']} (N={cfg['B'] * cfg['S']:,}), "
                 f"latency {cfg['latency']:,}, drift period {P:.0e} cycles x {cfg['periods_run']:g}, A={A}, "
                 f"sigma={cfg['sigma']}, {cfg['sensitivity']} sensitivity")
    plt.tight_layout()
    os.makedirs(os.path.dirname(LONG_FIG), exist_ok=True)
    plt.savefig(LONG_FIG, dpi=130)
    plt.close()
    print(f"Plot saved to '{LONG_FIG}'")

    print(f"\nSteady state (cycles > 1 period), tracking rms; references: random-time calibration {A}, "
          f"drift center {A / np.sqrt(2):.3f}")
    for t in tags:
        s = long_steady(data[t], P)
        print(f"  {t:>10}: " + "  ".join(f"{q} {s['steer'][q]:.3f}" for q in cats)
              + f"   | hold t=0: {s['hold']['all']:.3f}   r(vs rms A) = {1 - s['steer']['all'] ** 2 / A ** 2 - (cfg['sigma'] / A) ** 2:+.3f}")


def run_long(args):
    B, S, P, seed = args.B[0], args.S[0], args.periods[0], args.seeds[0]
    response = None
    if args.sgd_gains:
        with open(args.response) as f:
            response = json.load(f)["k"]
    T_u = B * S + args.latency
    updates = int(np.ceil(args.periods_run * P / T_u))
    base = dict(mode="loop", B=B, S=S, period=P, seed=seed, drift=args.drift, min_cycles=0, latency=args.latency,
                updates=updates, burn=0, time_samples=16, sigma=args.sigma, amplitude=args.amplitude, scale=args.scale,
                discard=args.discard, sensitivity=args.sensitivity, trace=True)
    cfgs = [dict(base, optimizer="sgd", lr=g, response=response) for g in args.sgd_gains]
    cfgs += [dict(base, optimizer="adam", lr=lr, response=None) for lr in args.adam_lr]
    if not args.replot:
        os.makedirs(LONG_OUT, exist_ok=True)
        print(f"{len(cfgs)} runs x {updates} updates x {T_u:,} cycles = {updates * T_u:,} cycles each", flush=True)
        with Pool(len(cfgs)) as pool:
            pool.map(long_job, cfgs)
    long_plot([long_tag(c) for c in cfgs], dict(base, periods_run=args.periods_run))


# ---------------------------------------------------------------------- driver

def build_jobs(args):
    common = dict(sigma=args.sigma, amplitude=args.amplitude, scale=args.scale, discard=args.discard,
                  sensitivity=args.sensitivity)
    response = None
    if "sgd" in args.optimizer:
        with open(args.response) as f:
            response = json.load(f)["k"]
    jobs = []
    if args.mode == "loop":
        for B in args.B:
            for S in args.S:
                for P in args.periods:
                    for seed in args.seeds:
                        for lr in args.lr:
                            for opt in args.optimizer:
                                jobs.append(dict(mode="loop", B=B, S=S, period=float(P), seed=seed, lr=lr, optimizer=opt,
                                                 drift=args.drift, min_cycles=args.min_cycles,
                                                 response=response if opt == "sgd" else None,
                                                 latency=args.latency, updates=args.updates, burn=args.burn,
                                                 time_samples=16, **common))
    else:
        for B in args.B:
            for S in args.S:
                for rep in range(args.reps):
                    jobs.append(dict(mode="cosine", B=B, S=S, seed=100 + rep, e_rms=args.e_rms, **common))
    return jobs


def job_cycles(cfg):
    return schedule_length(cfg)[0] * (cfg["B"] * cfg["S"] + cfg["latency"]) if cfg["mode"] == "loop" else cfg["B"] * cfg["S"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("loop", "cosine", "plot", "long"))
    p.add_argument("--B", type=int, nargs="+", default=None, help="default: DEFAULTS (sweep) or LONG_DEFAULTS (long)")
    p.add_argument("--S", type=int, nargs="+", default=None)
    p.add_argument("--periods", type=float, nargs="+", default=None, help="drift periods, in cycles")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--lr", type=float, nargs="+", default=[0.18])
    p.add_argument("--optimizer", nargs="+", choices=("adam", "sgd"), default=["adam"])
    p.add_argument("--sigma", type=float, default=0.1, help="fixed probe width (tolerance units)")
    p.add_argument("--amplitude", type=float, default=1.0, help="drift amplitude A (tolerance units)")
    p.add_argument("--scale", type=float, default=8e-4)
    p.add_argument("--latency", type=int, default=10_000, help="cycles from a batch's end to its update loading")
    p.add_argument("--updates", type=int, default=50)
    p.add_argument("--burn", type=int, default=10, help="scored from update max(burn, updates // 5) on")
    p.add_argument("--min-cycles", type=float, default=0, help="run at least this many cycles (more updates for small N)")
    p.add_argument("--drift", choices=("sin", "ar1"), default="sin")
    p.add_argument("--sensitivity", choices=("equalized", "raw"), default="equalized")
    p.add_argument("--response", default="runs/streaming/schedule_sweep_response.json",
                   help="per-kind gradient response k (from analytics/gradient_information.py), for --optimizer sgd")
    p.add_argument("--discard", type=int, default=1)
    p.add_argument("--reps", type=int, default=8, help="cosine mode: repetitions per (B, S)")
    p.add_argument("--e-rms", type=float, default=0.3, help="cosine mode: rms mis-set")
    p.add_argument("--procs", type=int, default=18)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    # long mode
    p.add_argument("--periods-run", type=float, default=3.0, help="long: drift periods to run")
    p.add_argument("--sgd-gains", type=float, nargs="*", default=[0.01, 0.02, 0.04], help="long: per-type SGD gains")
    p.add_argument("--adam-lr", type=float, nargs="*", default=[0.03], help="long: Adam learning rates")
    p.add_argument("--replot", action="store_true", help="long: plot from the saved .npz files")
    args = p.parse_args()
    for key, value in (LONG_DEFAULTS if args.mode == "long" else DEFAULTS).items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.mode == "long":
        run_long(args)
        return

    if args.mode == "plot":
        from analytics.plot_schedule_sweep import main as plot_main
        plot_main(args.out)
        return

    out = args.out or f"runs/streaming/schedule_sweep_{args.mode}.jsonl"
    jobs = build_jobs(args)
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                r = json.loads(line)
                done.add(job_key({k: r[k] for k in jobs[0]}) if jobs else None)
    todo = sorted([j for j in jobs if job_key(j) not in done], key=job_cycles, reverse=True)  # longest first
    cyc = np.array([job_cycles(j) for j in todo], dtype=float)
    agg = PER_PROC_THROUGHPUT * args.procs
    est = max(cyc.sum() / agg, (cyc.max() if len(cyc) else 0) / PER_PROC_THROUGHPUT)
    print(f"{len(jobs)} jobs ({len(jobs) - len(todo)} done), {cyc.sum():.3g} cycles to simulate; "
          f"longest job {cyc.max() if len(cyc) else 0:.3g} cycles (~{(cyc.max() if len(cyc) else 0) / PER_PROC_THROUGHPUT / 3600:.1f} h); "
          f"estimated {est / 3600:.1f} h on {args.procs} processes")
    if args.dry_run or not todo:
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t0 = time.time()
    with Pool(args.procs, maxtasksperchild=4) as pool, open(out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(run_job, todo, chunksize=1)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            tag = (f"B={r['B']} S={r['S']} P={r['period']:.0e} seed={r['seed']} lr={r['lr']} {r['optimizer']}: "
                   f"r_insitu {r['r_insitu']:+.3f} rms {r['tracking_rms']:.3f} cos {r['cos_loop']:.2f}"
                   if r["mode"] == "loop" else f"B={r['B']} S={r['S']} seed={r['seed']}: cos {r['cos']:.3f}")
            print(f"[{i + 1}/{len(todo)} {time.time() - t0:7.0f}s] {tag} ({r['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
