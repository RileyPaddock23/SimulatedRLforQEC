"""
Steerability scans
==================
How much of the gap between a fixed calibration and the drifting optimum the RL agent closes,
as one property of the drift (or of the device) is swept. Every point runs the same short
epoch-based trial: d=5, 5 rounds, 30 epochs; each epoch the agent measures 20 antithetic
candidates at 350 shots (EDR only) and updates, then the Fixed (mu = 0, the stale t = 0
calibration), RL (agent mean) and Optimal (drift oracle) policies are scored at 800 shots.
Points are averaged over --seeds drift realizations, for every code and parameter mode
(gate / qubit / both).

Metrics, from the seed- and run-averaged rates:
  - LER suppression (%)   100 (LER_fixed - LER_rl) / LER_fixed
  - steering advantage r  (EDR_fixed - EDR_rl) / (EDR_fixed - EDR_opt); 1 = perfect tracking

Sweeps (see SWEEPS):
  freq          sinusoid frequency f (cycles / epoch), amplitude 2
  tau           AR(1) correlation time tau, stationary std matched to the sinusoid's rms;
                top axis: corner frequency 1 / (2 pi tau)
  alpha         1/f^alpha exponent (ARMA filter, no white floor or common mode), same std
  amp-sin       sinusoid amplitude A, f = 1/150, Omega = 1
  amp-ar1       AR(1) std sigma, tau = 150 / 2 pi (corner at f = 1/150), Omega = 1
  omega-sin     sensitivity scale Omega of every parameter, sinusoid A = 1
  omega-ar1     the same with AR(1) sigma = 1

A mis-set e costs Omega e^2 error budgets, so the amp-* and omega-* sweeps move the drift's
cost the same way (Omega A^2) but differ for the agent: a bigger A moves the optimum faster in
parameter units; a bigger Omega makes both the signal and each probe's cost larger.

Results: runs/steerability/<sweep>.json, one row per (code, mode, value, seed, lr_mu). Rerun
points replace old ones; everything else is kept.

    python steerability.py scan --sweep freq                 # every code, mode, value, seed
    python steerability.py scan --sweep alpha --values 0 0.5 --codes surface
    python steerability.py scan --sweep amp-sin amp-ar1      # the amp-* sweeps default to surface
    python steerability.py scan --sweep freq --modes gate --lr-mu 0.05 0.1 0.3
    python steerability.py static --sweep alpha              # add calibrate-at-t=0 / hindsight-best-fixed refs
    python steerability.py plot --sweep alpha --reference bestfixed
    python steerability.py plot --sweep amp-sin amp-ar1      # stacked rows, one figure
    python steerability.py plot-lr --sweep freq --code surface
    python steerability.py track --sweep freq --value 0.0067 --epochs 300
    python steerability.py trajectories                      # example 1/f^alpha drift realizations
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from run_experiment import AGENT_DEFAULTS, CODES, make_agent

SEED = 163
EPOCHS = 30
SIGMA = 2.0 / np.sqrt(2.0)  # stationary std matched to the rms of the default sinusoid (A = 2)
FREQS = [1 / 1000, 1 / 500, 1 / 250, 1 / 150, 1 / 100, 1 / 50, 1 / 25, 1 / 10]
FREQ_AMP = 1.0 / 150.0  # fixed timescale of the amp-* / omega-* sweeps, near the Google f_c ~ 1/130
TAU_AMP = 1.0 / (2 * np.pi * FREQ_AMP)
AMPLITUDES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
OMEGAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
OUT_DIR = "runs/steerability"
FIG_DIR = "figures/steerability"

MODES = {
    "gate": dict(params_per_gate=1, params_per_qubit=0),
    "qubit": dict(params_per_gate=0, params_per_qubit=1),
    "both": dict(params_per_gate=1, params_per_qubit=1),
}
# Fixed categorical order (blue, orange, aqua) plus a marker per mode, so identity isn't color-alone
MODE_STYLE = {"gate": ("#2a78d6", "o", "Gate only"), "qubit": ("#eb6834", "s", "Qubit only"),
              "both": ("#1baf7a", "^", "Gate + qubit")}


def _sin(f, a=2.0, omega=None):
    return dict(drift_profile="sinusoidal", drift_frequency=f, drift_amplitude=a,
                **({} if omega is None else {"sensitivity": omega}))


def _ar1(tau, sigma=SIGMA, omega=None):
    return dict(drift_profile="ar1", ar1_tau_range=(tau, tau), ar1_sigma_range=(sigma, sigma),
                **({} if omega is None else {"sensitivity": omega}))


def _arma(alpha):
    return dict(drift_profile="arma", arma_alpha=alpha, arma_sigma=SIGMA, arma_white_fraction=0.0,
                arma_common_fraction=0.0)


# name -> values, env kwargs from a value, x label, log x, default codes, extra plot decorations
SWEEPS = {
    "freq": dict(values=FREQS, env=_sin, xlabel="Drift frequency $f$ (cycles / epoch)", logx=True,
                 title="Sinusoidal drift, $A$ = 2", vline=(1 / 130, "Google limit ($f_c \\sim 1/130$)")),
    "tau": dict(values=[1 / (2 * np.pi * f) for f in FREQS], env=_ar1, logx=True, top_corner=True,
                xlabel="AR(1) correlation time $\\tau$ (epochs)", title=f"AR(1) drift, stationary std {SIGMA:.2f}"),
    "alpha": dict(values=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], env=_arma, logx=False,
                  xlabel="Drift PSD exponent $\\alpha$  ($S(f) \\propto 1/f^\\alpha$; larger = slower)",
                  title=f"$1/f^\\alpha$ drift, stationary std {SIGMA:.2f}"),
    "amp-sin": dict(values=AMPLITUDES, env=lambda a: _sin(FREQ_AMP, a, omega=1.0), logx=True, codes=["surface"],
                    xlabel="Sinusoid amplitude $A$ (tolerance units)", title=f"Sinusoidal, $f$ = 1/{1 / FREQ_AMP:.0f}, $\\Omega$ = 1"),
    "amp-ar1": dict(values=AMPLITUDES, env=lambda s: _ar1(TAU_AMP, s, omega=1.0), logx=True, codes=["surface"],
                    xlabel="AR(1) stationary std $\\sigma$ (tolerance units)", title=f"AR(1), $\\tau$ = {TAU_AMP:.1f}, $\\Omega$ = 1"),
    "omega-sin": dict(values=OMEGAS, env=lambda w: _sin(FREQ_AMP, 1.0, omega=w), logx=True, codes=["surface"],
                      xlabel="Sensitivity $\\Omega$", title=f"Sinusoidal, $f$ = 1/{1 / FREQ_AMP:.0f}, $A$ = 1"),
    "omega-ar1": dict(values=OMEGAS, env=lambda w: _ar1(TAU_AMP, 1.0, omega=w), logx=True, codes=["surface"],
                      xlabel="Sensitivity $\\Omega$", title=f"AR(1), $\\tau$ = {TAU_AMP:.1f}, $\\sigma$ = 1"),
}
POLICIES = ("fixed", "rl", "opt")


def make_env(sweep: str, code: str, mode: str, value: float, seed: int, epochs: int = EPOCHS):
    return CODES[code](distance=5, rounds=5, base_error=0.002, seed=seed, max_epochs=epochs + 1,
                       **SWEEPS[sweep]["env"](value), **MODES[mode])


def row_key(r):
    return (r["code"], r["mode"], r["value"], r["seed"], r["lr_mu"])


# ---------------------------------------------------------------------- scan

def scan_point(sweep: str, code: str, mode: str, value: float, seed: int, lr_mu: float = AGENT_DEFAULTS["lr_mu"],
               decode_every: int = 1, epochs: int = EPOCHS, history: bool = False) -> dict:
    """
    LER and EDR of the Fixed, RL and Optimal policies, averaged over the run, for one drift realization.
    EDR is measured every epoch; LER (which needs the decoder) every `decode_every` epochs.
    With `history`, also returns the per-epoch series and the agent / optimum trajectories.
    """
    env = make_env(sweep, code, mode, value, seed, epochs)
    agent = make_agent(env, seed, lr_mu=lr_mu)
    fixed = np.zeros(env.num_params, dtype=np.float32)
    h = {f"{m}_{tag}": [] for m in ("ler", "edr") for tag in POLICIES}
    mu, p_opt = [], []
    for ep in range(epochs):
        cands = agent.sample_policies(batch_size=20, antithetic=True)
        obs = np.array([env.evaluate_policy(c, t=ep, shots=350, decode=False)["edr_vector"] for c in cands])
        agent.update(cands, obs)
        opt = env.get_drift_opt(ep)
        decode = ep % decode_every == 0
        for tag, policy in (("fixed", fixed), ("rl", agent.mu), ("opt", opt)):
            ev = env.evaluate_policy(policy, t=ep, shots=800, decode=decode)
            h[f"edr_{tag}"].append(ev["mean_edr"])
            if decode:
                h[f"ler_{tag}"].append(ev["logical_error_rate"])
        mu.append(agent.mu.copy())
        p_opt.append(opt)
    row = {"sweep": sweep, "code": code, "mode": mode, "value": float(value), "seed": seed, "lr_mu": lr_mu,
           "decode_every": decode_every, "epochs": epochs, "drift_amplitude": getattr(env.drift, "amplitude", None),
           **{k: float(np.mean(v)) for k, v in h.items()}}
    if history:
        row["history"] = {**h, "mu": np.array(mu), "p_opt": np.array(p_opt)}
    return row


def _run(job):
    r = scan_point(*job)
    print(f"  {r['sweep']:9s} {r['code']:7s} {r['mode']:5s} {r['value']:8.4g} seed {r['seed']} lr {r['lr_mu']:g}: "
          f"EDR fixed {r['edr_fixed']:.4f} RL {r['edr_rl']:.4f} opt {r['edr_opt']:.4f}", flush=True)
    return r


def load(out_dir: str, sweep: str) -> list:
    path = os.path.join(out_dir, f"{sweep}.json")
    return json.load(open(path)) if os.path.exists(path) else []


def save(out_dir: str, sweep: str, rows: list):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{sweep}.json"), "w") as f:
        json.dump(sorted(rows, key=row_key), f, indent=1)


def scan(args):
    seeds = list(range(SEED, SEED + args.seeds))
    jobs = [(sweep, code, mode, v, seed, lr, args.decode_every)
            for sweep in args.sweep for code in (args.codes or SWEEPS[sweep].get("codes", list(CODES)))
            for mode in args.modes for v in (args.values or SWEEPS[sweep]["values"]) for seed in seeds for lr in args.lr_mu]
    print(f"{len(jobs)} points on {min(args.workers, len(jobs))} workers", flush=True)
    with Pool(min(args.workers, len(jobs))) as pool:
        rows = pool.map(_run, jobs, chunksize=1)
    for sweep in args.sweep:
        new = [r for r in rows if r["sweep"] == sweep]
        done = {row_key(r) for r in new}
        save(args.out_dir, sweep, [r for r in load(args.out_dir, sweep) if row_key(r) not in done] + new)


# ---------------------------------------------------------------------- static reference policies

def _policy_initcal(env, epochs):
    """Calibrated once at t = 0 (p = p_opt(0)) and never touched again: needs no knowledge of the future."""
    return env.get_drift_opt(0)


def _policy_bestfixed(env, epochs):
    """
    Hindsight-optimal constant p: errors are quadratic in the mis-set, so (ignoring the max_error clip) the
    run-averaged error is minimized by the time-average of p_opt. An oracle bound for static calibration.
    """
    return np.mean([env.get_drift_opt(ep) for ep in range(epochs)], axis=0)


REFERENCES = {
    "initcal": ("Calibrated at $t=0$, then held", _policy_initcal),
    "bestfixed": ("Best fixed policy (hindsight)", _policy_bestfixed),
}


def reference_point(job) -> dict:
    """
    Evaluates static reference `ref` on a scan row's drift realization (same env seed and LER decode schedule)
    and adds edr_<ref> (and ler_<ref> if decode_ler) to the row. Without decoding only the EDR is measured;
    the LER is then estimated from it at plot time (see estimate_ler_from_edr).
    """
    row, ref, decode_ler = job
    epochs = row.get("epochs", EPOCHS)
    env = make_env(row["sweep"], row["code"], row["mode"], row["value"], row["seed"], epochs)
    policy = REFERENCES[ref][1](env, epochs).astype(np.float32)
    edr, ler = [], []
    for ep in range(epochs):
        decode = decode_ler and ep % row["decode_every"] == 0
        ev = env.evaluate_policy(policy, t=ep, shots=800, decode=decode)
        edr.append(ev["mean_edr"])
        if decode:
            ler.append(ev["logical_error_rate"])
    out = {**row, f"edr_{ref}": float(np.mean(edr))}
    if decode_ler:
        out[f"ler_{ref}"] = float(np.mean(ler))
    return out


def scan_static(args):
    """Adds every static reference's results to each row (of the requested codes) that lacks them."""
    for sweep in args.sweep:
        rows = load(args.out_dir, sweep)
        codes = args.codes or sorted({r["code"] for r in rows})
        for ref in REFERENCES:
            todo = [(r, ref, r["code"] not in args.estimate_ler) for r in rows
                    if r["code"] in codes and f"edr_{ref}" not in r]
            if todo:
                with Pool(min(args.workers, len(todo))) as pool:
                    done = {row_key(r): r for r in pool.map(reference_point, todo, chunksize=1)}
                rows = [done.get(row_key(r), r) for r in rows]
                save(args.out_dir, sweep, rows)
            print(f"  {sweep}: added '{ref}' results to {len(todo)} rows", flush=True)


def fit_edr_to_ler(rows: list, code: str, mode: str):
    """
    Run-averaged LER vs run-averaged EDR for one (code, mode), fit as log LER = quadratic in EDR over every
    decoded (fixed, RL, optimal) policy. Returns (predict, s), s the rms log-residual (~68% band per run).
    """
    pts = [(r[f"edr_{t}"], r[f"ler_{t}"]) for r in rows if (r["code"], r["mode"]) == (code, mode)
           for t in POLICIES if r[f"ler_{t}"] > 0]
    x, y = np.array(pts).T
    coef = np.polyfit(x, np.log(y), 2)
    return (lambda e: np.polyval(coef, e)), float(np.std(np.log(y) - np.polyval(coef, x)))


def estimate_ler_from_edr(rows: list, ref: str) -> list:
    """Fills ler_<ref>_lo/mid/hi from edr_<ref> (and the EDR -> LER fit) for rows with no decoded LER."""
    out, fits = [], {}
    for r in rows:
        if f"edr_{ref}" in r and f"ler_{ref}" not in r:
            key = (r["code"], r["mode"])
            if key not in fits:
                fits[key] = fit_edr_to_ler(rows, *key)
            pred, s = fits[key]
            m = float(pred(r[f"edr_{ref}"]))
            r = {**r, f"ler_{ref}_mid": np.exp(m), f"ler_{ref}_lo": np.exp(m - s), f"ler_{ref}_hi": np.exp(m + s)}
        out.append(r)
    return out


# ---------------------------------------------------------------------- aggregate and plot

def aggregate(rows: list, ref: str = None) -> dict:
    """
    Per (code, mode, value): seed-averaged rates, then LER suppression and steering advantage from them, for the
    RL agent and (as <metric>_ref, or _ref_lo/mid/hi when LER is estimated) for static reference `ref`.
    """
    if ref:
        rows = estimate_ler_from_edr(rows, ref)
    out = {}
    for key in sorted({(r["code"], r["mode"], r["value"]) for r in rows}):
        pts = [r for r in rows if (r["code"], r["mode"], r["value"]) == key]
        m = {f"{q}_{t}": float(np.mean([r[f"{q}_{t}"] for r in pts])) for q in ("ler", "edr") for t in POLICIES}
        red = lambda ler: (m["ler_fixed"] - ler) / (m["ler_fixed"] + 1e-6) * 100.0
        adv = lambda edr: (m["edr_fixed"] - edr) / (m["edr_fixed"] - m["edr_opt"] + 1e-6)
        m.update(ler_red_pct=red(m["ler_rl"]), steering_r=adv(m["edr_rl"]), n=len(pts))
        if ref and all(f"edr_{ref}" in r for r in pts):
            m["steering_r_ref"] = adv(np.mean([r[f"edr_{ref}"] for r in pts]))
            if all(f"ler_{ref}" in r for r in pts):  # decoded
                m["ler_red_pct_ref"] = red(np.mean([r[f"ler_{ref}"] for r in pts]))
            else:  # estimated from EDR; band = fit residual, taken fully correlated across seeds (conservative)
                for b in ("lo", "mid", "hi"):
                    m[f"ler_red_pct_ref_{b}"] = red(np.mean([r[f"ler_{ref}_{b}"] for r in pts]))
        out[key] = m
    return out


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot(sweeps: list, out_dir: str, path: str, ref: str = None):
    """
    One row per (sweep, code) with results: LER suppression, steering advantage, and the raw LERs of the
    RL (solid), Fixed (dashed) and Optimal (dotted) policies. Only default-lr_mu rows are shown.
    """
    plt = _plt()
    panels = []
    for sweep in sweeps:
        rows = [r for r in load(out_dir, sweep) if r["lr_mu"] == AGENT_DEFAULTS["lr_mu"]]
        for code in [c for c in CODES if any(r["code"] == c for r in rows)]:
            panels.append((sweep, code, aggregate([r for r in rows if r["code"] == code], ref)))
    if not panels:
        print(f"No results for {sweeps} in '{out_dir}'")
        return

    fig, axes = plt.subplots(len(panels), 3, figsize=(18, 4.6 * len(panels)), dpi=150, squeeze=False)
    for i, (sweep, code, agg) in enumerate(panels):
        spec = SWEEPS[sweep]
        for mode, (c, mk, lab) in MODE_STYLE.items():
            xs = sorted(v for (_, m, v) in agg if m == mode)
            if not xs:
                continue
            pts = [agg[(code, mode, v)] for v in xs]
            for j, key in enumerate(("ler_red_pct", "steering_r")):
                axes[i, j].plot(xs, [p[key] for p in pts], color=c, marker=mk, ms=6, lw=2, label=lab)
                dec = [(x, p[f"{key}_ref"]) for x, p in zip(xs, pts) if f"{key}_ref" in p]
                if dec:
                    axes[i, j].plot(*zip(*dec), color=c, marker=mk, ms=5, mfc="white", lw=1.4, ls="--")
                est = [(x, p) for x, p in zip(xs, pts) if f"{key}_ref_mid" in p]
                if est:  # LER estimated from EDR: shaded band around the dashed estimate
                    ex = [x for x, _ in est]
                    g = lambda b: [p[f"{key}_ref_{b}"] for _, p in est]
                    axes[i, j].fill_between(ex, g("hi"), g("lo"), color=c, alpha=0.15, lw=0)
                    axes[i, j].plot(ex, g("mid"), color=c, marker=mk, ms=5, mfc="white", lw=1.4, ls="--")
            for tag, ls in (("rl", "-"), ("fixed", "--"), ("opt", ":")):
                axes[i, 2].plot(xs, [p[f"ler_{tag}"] for p in pts], color=c, ls=ls, lw=2 if tag == "rl" else 1.3,
                                marker=mk if tag == "rl" else None, ms=5, alpha=1 if tag == "rl" else 0.8)
        for j, ax in enumerate(axes[i]):
            ax.grid(True, which="both" if j == 2 else "major", ls="--", alpha=0.4)
            ax.set_xlabel(spec["xlabel"])
            if spec["logx"]:
                ax.set_xscale("log")
            if spec.get("vline"):
                ax.axvline(spec["vline"][0], color="#7f7f7f", ls="-.", lw=1.2, label=spec["vline"][1] if j < 2 else None)
            if spec.get("top_corner"):  # AR(1) corner frequency f_c = 1 / (2 pi tau), its own inverse
                to_f = lambda x: 1 / (2 * np.pi * np.maximum(x, 1e-12))
                ax.secondary_xaxis("top", functions=(to_f, to_f)).set_xlabel("Corner frequency $f_c = 1/2\\pi\\tau$")
            if j < 2:
                ax.axhline(0.0, color="#7f7f7f", ls="--", lw=1.2)
            if j == 1:
                ax.axhline(1.0, color="#7f7f7f", ls=":", lw=1.2, label="Optimal tracking ($r=1$)")
            if j < 2 and ref:
                ax.plot([], [], color="#7f7f7f", marker="o", mfc="white", ls="--", lw=1.4, label=REFERENCES[ref][0])
            if j == 2:
                ax.set_yscale("log")
                for ls, lab in (("-", "RL"), ("--", "Fixed"), (":", "Optimal")):
                    ax.plot([], [], color="#7f7f7f", ls=ls, label=lab)
            ax.legend(loc="best", fontsize=7)
        n = max(m["n"] for m in agg.values())
        axes[i, 0].set_ylabel(f"{spec['title']}\n{code} code, {n} seeds\nLER suppression (%)", fontsize=9)
        axes[i, 1].set_ylabel("Steering advantage $r$")
        axes[i, 2].set_ylabel("Logical error rate (per 5 rounds)")
    axes[0, 0].set_title("Logical error suppression", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Steering advantage", fontsize=11, fontweight="bold")
    axes[0, 2].set_title("LER: RL (solid), Fixed (dashed), Optimal (dotted)", fontsize=11, fontweight="bold")
    fig.suptitle(f"Steerability, d=5, {EPOCHS} epochs per point", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved '{path}'")


def plot_lr(sweep: str, code: str, out_dir: str, path: str):
    """Steering advantage and LER suppression vs the swept value, one line per lr_mu, one column per mode."""
    plt = _plt()
    rows = [r for r in load(out_dir, sweep) if r["code"] == code]
    lrs = sorted({r["lr_mu"] for r in rows})
    if len(lrs) < 2:
        print(f"Only lr_mu {lrs} in '{sweep}' for {code}: run scan with several --lr-mu first")
        return
    # lr_mu is ordinal, so one hue light -> dark
    ramp = ["#6da7ec", "#3987e5", "#1c5cab", "#0d366b"][-len(lrs):] if len(lrs) <= 4 else plt.cm.Blues(np.linspace(0.4, 1, len(lrs)))
    modes = [m for m in MODES if any(r["mode"] == m for r in rows)]
    spec = SWEEPS[sweep]
    fig, axes = plt.subplots(2, len(modes), figsize=(5.2 * len(modes), 8.5), dpi=150, sharex=True, sharey="row", squeeze=False)
    for j, mode in enumerate(modes):
        for lr, c in zip(lrs, ramp):
            agg = aggregate([r for r in rows if r["mode"] == mode and r["lr_mu"] == lr])
            xs = sorted(v for (_, _, v) in agg)
            if not xs:
                continue
            for i, key in enumerate(("steering_r", "ler_red_pct")):
                axes[i, j].plot(xs, [agg[(code, mode, v)][key] for v in xs], color=c, marker="o", ms=5, lw=2,
                                label=f"lr$_\\mu$ = {lr:g}")
            amp = next((r["drift_amplitude"] for r in rows if r.get("drift_amplitude")), None)
            if sweep == "freq" and amp:  # Adam slew limit: max step ~ lr_mu per update vs max drift speed 2 pi f A
                f_slew = lr / (2 * np.pi * amp)
                if min(xs) <= f_slew <= max(xs):
                    for ax in axes[:, j]:
                        ax.axvline(f_slew, color=c, ls=":", lw=1.4)
        axes[0, j].set_title(MODE_STYLE[mode][2], fontsize=11, fontweight="bold")
        for ax in axes[:, j]:
            ax.grid(True, ls="--", alpha=0.4)
            ax.axhline(0.0, color="#7f7f7f", ls="--", lw=1.2)
            if spec["logx"]:
                ax.set_xscale("log")
        axes[0, j].axhline(1.0, color="#7f7f7f", ls=":", lw=1.2)
        axes[1, j].set_xlabel(spec["xlabel"])
    axes[0, 0].set_ylabel("Steering advantage $r$")
    axes[1, 0].set_ylabel("LER suppression (%)")
    axes[0, 0].legend(loc="lower left", fontsize=8,
                      title="dotted: $f^* = $lr$_\\mu / 2\\pi A$" if sweep == "freq" else None, title_fontsize=8)
    fig.suptitle(f"{code.capitalize()} code, {sweep} sweep: learning-rate comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved '{path}'")


def plot_track(sweep: str, code: str, mode: str, value: float, epochs: int, path: str):
    """Agent mean mu vs the drifting optimum, and the resulting EDR, over every epoch of one run."""
    plt = _plt()
    r = scan_point(sweep, code, mode, value, SEED, epochs=epochs, history=True)
    h = r["history"]
    ep = np.arange(epochs)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), dpi=150, sharex=True)
    for ax in (ax1, ax2):
        ax.grid(True, linestyle="--", alpha=0.5)
    ax1.fill_between(ep, h["mu"].min(axis=1), h["mu"].max(axis=1), color="#2a78d6", alpha=0.2,
                     label="Agent $\\mu$ (range over params)")
    ax1.plot(ep, h["mu"].mean(axis=1), color="#2a78d6", lw=1.8, label="Agent $\\mu$ (mean over params)")
    ax1.plot(ep, h["p_opt"].mean(axis=1), color="black", lw=1.5, ls="--", label="Optimum $p_{\\mathrm{opt}}(t)$ (mean)")
    ax1.set_ylabel("Parameter value (tolerance units)")
    ax1.set_title("Parameter tracking", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right")
    for tag, c, lab in (("fixed", "#7f7f7f", "Fixed policy"), ("rl", "#2a78d6", "RL agent"), ("opt", "#1baf7a", "Optimal policy")):
        ax2.plot(ep, h[f"edr_{tag}"], color=c, lw=1, label=lab)
    agg = aggregate([r])[(code, mode, float(value))]
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean EDR")
    ax2.set_title(f"Error detection rate (steering r = {agg['steering_r']:.2f}, LER reduction = {agg['ler_red_pct']:+.1f}%)",
                  fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right")
    fig.suptitle(f"{code.capitalize()} code (d=5), {mode} params, {sweep} = {value:g}, {epochs} epochs",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved '{path}'")


def plot_trajectories(path: str, alphas=(0.0, 0.5, 1.0, 1.5, 2.0), n_real: int = 3, long_epochs: int = 1000):
    """
    Example 1/f^alpha drift trajectories p_opt(t), generated as in the alpha sweep (std SIGMA, no white
    floor, no common mode). Left: the first EPOCHS epochs the agent sees; right: a long window.
    """
    plt = _plt()
    from code_envs.noise_envs import generate_fractional_noise

    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axes = plt.subplots(len(alphas), 2, figsize=(13, 1.9 * len(alphas) + 0.8), dpi=150, sharey=True,
                             gridspec_kw={"width_ratios": [1, 2.2]}, squeeze=False)
    names = {0.0: "white", 1.0: "flicker / pink", 2.0: "Brownian"}
    for i, a in enumerate(alphas):
        for k in range(n_real):
            x = generate_fractional_noise(long_epochs + 1, a, sigma=SIGMA, rng=np.random.default_rng(SEED + k))
            axes[i, 0].plot(np.arange(EPOCHS + 1), x[:EPOCHS + 1], color=colors[k], lw=1.4, marker="o", ms=2.5)
            axes[i, 1].plot(np.arange(long_epochs + 1), x, color=colors[k], lw=0.8)
        axes[i, 1].axvspan(0, EPOCHS, color="#6b6a63", alpha=0.12, lw=0)
        axes[i, 0].set_ylabel(f"$\\alpha$ = {a:g}" + (f"\n({names[a]})" if a in names else ""), fontsize=10)
        for ax in axes[i]:
            ax.grid(True, ls="--", alpha=0.35)
            for s in (-1, 1):
                ax.axhline(s * SIGMA, color="#7f7f7f", ls=":", lw=1.0)
    axes[0, 0].set_title(f"First {EPOCHS} epochs (the scan's horizon)", fontsize=11, fontweight="bold")
    axes[0, 1].set_title(f"{long_epochs} epochs (shaded: first {EPOCHS})", fontsize=11, fontweight="bold")
    axes[-1, 0].set_xlabel("Epoch")
    axes[-1, 1].set_xlabel("Epoch")
    fig.suptitle(f"$1/f^\\alpha$ drift trajectories of the optimum, stationary std {SIGMA:.2f} (dotted: $\\pm\\sigma$), "
                 f"{n_real} realizations per $\\alpha$", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved '{path}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["scan", "static", "plot", "plot-lr", "track", "trajectories"])
    p.add_argument("--sweep", nargs="+", choices=list(SWEEPS), default=["freq"])
    p.add_argument("--codes", nargs="+", choices=list(CODES), default=None,
                   help="default: the sweep's codes (both, or surface for amp-* / omega-*)")
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    p.add_argument("--values", nargs="+", type=float, default=None, help="values to (re)run; default: the sweep's grid")
    p.add_argument("--seeds", type=int, default=4, help="drift realizations per point (seeds 163, 164, ...)")
    p.add_argument("--lr-mu", nargs="+", type=float, default=[AGENT_DEFAULTS["lr_mu"]])
    p.add_argument("--decode-every", type=int, default=1,
                   help="decode LER every k epochs (BP-OSD on the color code costs ~30 s per evaluation)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--plot-path", default=None, help=f"default: {FIG_DIR}/steerability_<sweep>.png")
    p.add_argument("--reference", choices=list(REFERENCES), default=None, help="static reference drawn dashed on the plot")
    p.add_argument("--estimate-ler", nargs="*", choices=list(CODES), default=["color"],
                   help="static: codes whose reference LER is estimated from EDR instead of decoded (BP-OSD is slow)")
    # track / plot-lr
    p.add_argument("--code", choices=list(CODES), default="surface")
    p.add_argument("--mode", choices=list(MODES), default="gate")
    p.add_argument("--value", type=float, default=1 / 150)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    args = p.parse_args()

    name = "_".join(args.sweep)
    if args.action == "trajectories":
        plot_trajectories(args.plot_path or f"{FIG_DIR}/alpha_trajectories.png")
    elif args.action == "track":
        plot_track(args.sweep[0], args.code, args.mode, args.value, args.epochs,
                   args.plot_path or f"{FIG_DIR}/tracking_{args.sweep[0]}{args.value:g}_{args.code}_{args.mode}_{args.epochs}ep.png")
    elif args.action == "plot-lr":
        plot_lr(args.sweep[0], args.code, args.out_dir, args.plot_path or f"{FIG_DIR}/lr_{args.sweep[0]}_{args.code}.png")
    else:
        if args.action == "scan":
            scan(args)
        elif args.action == "static":
            scan_static(args)
            args.reference = args.reference or "initcal"
        plot(args.sweep, args.out_dir, args.plot_path or f"{FIG_DIR}/steerability_{name}.png", args.reference)
