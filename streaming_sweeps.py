"""
Single-Shot Streaming Sweeps
============================
Sweeps of the full streaming controller (`run_streaming_trial`): one long memory shot per
run (code_envs/streaming_env.py), decoded in real time, on the d=5 surface code. The agent
steers the 2q, 1q and qubit parameters; readout and reset drift uncontrolled, so the Optimal
reference is the reachable optimum (p_opt on the steered parameters, the t = 0 calibration
elsewhere). The Fixed and Optimal shots are shared by every run on the same drift and seed.

Clock: 1 us QEC cycles, a reference epoch of 35,000 cycles (35 ms, the same physical time as
run_experiment.py), candidates of 250 cycles in batches of 20, so a batch takes 5 ms.

latency   In-situ steering advantage against a uniform update latency T, for both schedules
          (interleave probes into the gap, or hold the mean until the update lands), under
          sinusoidal and 1/f^2 drift. T = 1 ms is also run with every kind steered.
          Metrics over the last third of the shot: in-situ advantage
          (EDR_fixed - EDR_rl) / (EDR_fixed - EDR_opt), EDR_rl covering every cycle the
          hardware ran, probes included; logical failures per cycle.

jump      Recovery from TLS jumps on the hardware-realistic drift of hardware_drift/: gate
          parameters at a stale calibration with Wu-rate random-walk drift, qubit parameters on
          the same walk plus `JumpDrift` events (default at 2.0, 3.5, 5.0 and 6.5 s; three
          qubit parameters each jump by +-1.5 tolerance). Interleave schedule; compares the
          agent's own PEPG sigma step (cap 0.5 or 0.3) with `AdamProbeSizer` (floor 0.1 or
          0.25), under three latency profiles. Per jumped parameter, with e = mu_hw - p_opt
          sampled at every batch end:
            - recovery time: first time after the jump at which |e| (20 ms moving average)
              is <= 0.5, i.e. the excess error on the qubit is down to 25% of its base rate
            - paid fraction: mean(e^2 - e_pre^2) over the second after the jump, over J^2
          Plus the mean probe width and probe cost Omega sigma^2 in the 400 ms before each jump.

Run from the repo root:
    python streaming_sweeps.py latency [--seeds 163 164 165]
    python streaming_sweeps.py jump [--seeds 163 164 165]
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from hardware_drift.profile import HARDWARE_LATENCY_MS, JUMP_AMPLITUDE, hardware_env
from rl_agent import AdamProbeSizer
from run_experiment import CODES, SHOTS_PER_EPOCH_REF, make_agent
from run_streaming_experiment import kind_latency, run_policy_shot, run_streaming_trial, steer_mask
from code_envs.streaming_env import StreamingQECEnv

DISTANCE, ROUNDS = 5, 5
CYCLE_US = 1.0
CYCLES_PER_EPOCH = SHOTS_PER_EPOCH_REF * ROUNDS  # 35 ms at 1 us per cycle
CYCLES_PER_CANDIDATE, BATCH_SIZE = 250, 20
BATCH_MS = BATCH_SIZE * CYCLES_PER_CANDIDATE * CYCLE_US / 1000  # 5 ms
DRIFT_UPDATE_CYCLES = 50  # 50 us; every drift here moves far slower
GATES_AND_QUBITS = ("2q", "1q", "qubit")
ALL_KINDS = ("2q", "1q", "reset", "meas", "qubit")

# latency experiment
DRIFTS = {"sinusoidal": dict(drift_profile="sinusoidal"), "1/f^2": dict(drift_profile="arma", arma_alpha=2.0)}
LATENCIES_MS = (0, 1, 5, 20, 50)
SCHEDULES = ("interleave", "wait")

# jump experiment
JUMP = JUMP_AMPLITUDE
JUMP_PROFILES = {  # name -> (default ms, per-kind ms)
    "uniform 1 ms": (1.0, {}),
    "hardware": (1.0, HARDWARE_LATENCY_MS),
    "hardware, qubit 50 ms": (1.0, {**HARDWARE_LATENCY_MS, "qubit": 50.0}),
}
SIZINGS = {  # name -> (agent overrides, AdamProbeSizer range or None for the agent's own sigma step)
    "PEPG, sigma <= 0.5": ({}, None),
    "PEPG, sigma <= 0.3": ({"sigma_max": 0.3}, None),
    "Adam-sized, 0.1-0.5": ({}, (0.1, 0.5)),
    "Adam-sized, 0.25-0.5": ({}, (0.25, 0.5)),
}
JUMP_SIZINGS = tuple(SIZINGS)
DEFAULT_SIZING = "PEPG, sigma <= 0.5"  # the agent's own sigma step at its default cap

# ---------------------------------------------------------------------- jobs (shared)
# Jobs: ("rl", drift, seed, schedule, latency, kinds, sizing) | ("fixed", drift, seed) | ("opt", drift, seed, kinds)
# drift: a DRIFTS name, or "hardware" (jump experiment); latency: uniform ms, or a JUMP_PROFILES name.

SETTINGS = {}


def init_worker(settings):
    SETTINGS.update(settings)


def make_env(drift: str, seed: int):
    if drift == "hardware":
        return hardware_env(seed, SETTINGS["total_s"], SETTINGS["events_s"])
    return CODES["surface"](distance=DISTANCE, rounds=ROUNDS, seed=seed,
                            max_epochs=int(np.ceil(SETTINGS["total_cycles"] / CYCLES_PER_EPOCH)) + 1, **DRIFTS[drift])


def latency_cycles(env, latency):
    ms_to_cycles = lambda ms: int(round(ms * 1000 / CYCLE_US))
    if isinstance(latency, str):
        default, by_kind = JUMP_PROFILES[latency]
        return kind_latency(env, {k: ms_to_cycles(v) for k, v in by_kind.items()}, default=ms_to_cycles(default))
    return ms_to_cycles(latency)


def run_job(job):
    kind, drift, seed, *rest = job
    env = make_env(drift, seed)
    common = dict(total_cycles=SETTINGS["total_cycles"], eval_interval=SETTINGS["eval_interval"],
                  cycles_per_epoch=CYCLES_PER_EPOCH, drift_update_cycles=DRIFT_UPDATE_CYCLES)
    t0 = time.time()
    if kind == "rl":
        schedule, latency, kinds, sizing = rest
        mask = steer_mask(env, kinds)
        overrides, sizer_range = SIZINGS[sizing]
        agent = make_agent(StreamingQECEnv(env, cycles_per_epoch=CYCLES_PER_EPOCH), seed=seed, steer_mask=mask, **overrides)
        res = run_streaming_trial(env, agent, cycles_per_candidate=CYCLES_PER_CANDIDATE, batch_size=BATCH_SIZE,
                                  latency=latency_cycles(env, latency), schedule=schedule, baselines=False,
                                  probe_sizer=AdamProbeSizer(*sizer_range) if sizer_range else None,
                                  seed=seed * 10, verbose=False, **common)
        out = {"summary": res["summary"]}
        if SETTINGS.get("events_s"):
            out["jumps"] = analyse_jumps(res["trajectory"], mask, env.param_sensitivities, SETTINGS["events_s"])
    elif kind == "fixed":
        out = run_policy_shot(env, np.zeros(env.num_params, dtype=np.float32), seed=seed * 10 + 1, **common)["summary"]
    else:
        mask = steer_mask(env, rest[0])
        opt = lambda t: np.where(mask, env.get_drift_opt(t), 0.0).astype(np.float32)
        out = run_policy_shot(env, opt, seed=seed * 10 + 2, **common)["summary"]
    print(f"  done {job} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return job, out


def run_all(jobs, settings, workers):
    jobs = sorted(jobs, key=lambda j: j[0] != "rl")  # the RL shots are the slowest; start them first
    print(f"{len(jobs)} single-shot runs of {settings['total_cycles']:,} cycles on {min(len(jobs), workers)} workers")
    t0 = time.time()
    with Pool(min(len(jobs), workers), initializer=init_worker, initargs=(settings,)) as pool:
        out = dict(pool.imap_unordered(run_job, jobs))
    print(f"all runs done in {(time.time() - t0) / 60:.1f} min\n")
    return out


def with_references(out, drift, seed, rl_job):
    """The RL run's summary plus its in-situ advantage against the shared Fixed and reachable-Optimal shots."""
    s = out[rl_job]["summary"]
    fixed, opt = out[("fixed", drift, seed)], out[("opt", drift, seed, rl_job[5])]
    return {**s, "steering_advantage_in_situ": (fixed["edr"] - s["edr_rl"]) / max(1e-5, fixed["edr"] - opt["edr"]),
            "edr_fixed": fixed["edr"], "edr_opt": opt["edr"], "ler_per_cycle_fixed": fixed["ler_per_cycle"],
            "ler_per_cycle_opt": opt["ler_per_cycle"]}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _config(args, **extra):
    return {**vars(args), "distance": DISTANCE, "rounds": ROUNDS, "cycle_us": CYCLE_US, "cycles_per_epoch": CYCLES_PER_EPOCH,
            "cycles_per_candidate": CYCLES_PER_CANDIDATE, "batch_size": BATCH_SIZE,
            "drift_update_cycles": DRIFT_UPDATE_CYCLES, **extra}


# ---------------------------------------------------------------------- latency sweep

def latency_cases():
    return ([("interleave", 0, GATES_AND_QUBITS)]
            + [(s, T, GATES_AND_QUBITS) for s in SCHEDULES for T in LATENCIES_MS[1:]]
            + [("interleave", 1, ALL_KINDS)])


def run_latency(args):
    settings = {"total_cycles": args.total_cycles, "eval_interval": args.eval_interval}
    jobs = []
    for drift in DRIFTS:
        for seed in args.seeds:
            jobs += [("rl", drift, seed, s, T, kinds, DEFAULT_SIZING) for s, T, kinds in latency_cases()]
            jobs += [("fixed", drift, seed), ("opt", drift, seed, GATES_AND_QUBITS), ("opt", drift, seed, ALL_KINDS)]
    out = run_all(jobs, settings, args.workers)

    keys = ("steering_advantage_in_situ", "edr_rl", "edr_fixed", "edr_opt", "ler_per_cycle_rl", "ler_per_cycle_fixed",
            "ler_per_cycle_opt", "tracking_rms", "num_updates", "stale_fraction", "hold_fraction")
    table = {}
    print(f"{'drift':10s} {'schedule':16s} {'T ms':>4s} {'in-situ adv':>11s} {'LER/cyc situ':>12s} {'fixed':>8s} "
          f"{'opt':>8s} {'RMS':>5s} {'upd':>4s} {'stale':>5s} {'hold':>5s}")
    for drift in DRIFTS:
        for schedule, T, kinds in latency_cases():
            runs = [with_references(out, drift, seed, ("rl", drift, seed, schedule, T, kinds, DEFAULT_SIZING)) for seed in args.seeds]
            a = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in keys}
            label = schedule + (" (all)" if kinds == ALL_KINDS else "")
            table[f"{drift} | {label} | {T}"] = a
            print(f"{drift:10s} {label:16s} {T:>4d} {a['steering_advantage_in_situ'][0]:5.2f}+-{a['steering_advantage_in_situ'][1]:.2f} "
                  f"{a['ler_per_cycle_rl'][0]:12.2e} {a['ler_per_cycle_fixed'][0]:8.2e} {a['ler_per_cycle_opt'][0]:8.2e} "
                  f"{a['tracking_rms'][0]:5.2f} {a['num_updates'][0]:4.0f} {a['stale_fraction'][0]:5.2f} "
                  f"{a['hold_fraction'][0]:5.2f}")
    with open(args.out + ".json", "w") as f:
        json.dump({"config": _config(args), "results": table}, f, indent=1)

    plt = _plt()
    colors = {"interleave": "#2a78d6", "wait": "#eb6834"}
    labels = {"interleave": "Interleave probes", "wait": "Hold mean while waiting"}
    x = np.array(LATENCIES_MS, dtype=float)
    xplot = np.where(x == 0, 0.3, x)  # T = 0 drawn at 0.3 ms on the log axis
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, drift in zip(axes, DRIFTS):
        for schedule in SCHEDULES:
            m = [table[f"{drift} | {'interleave' if T == 0 else schedule} | {T}"]["steering_advantage_in_situ"]
                 for T in LATENCIES_MS]
            ax.errorbar(xplot, [v[0] for v in m], yerr=[v[1] for v in m], color=colors[schedule], lw=2, marker="o",
                        ms=8, capsize=3, label=labels[schedule])
        ax.axvline(1.0, color="#888888", ls=":", lw=1)
        ax.text(1.05, 0.02, "T = 1 ms", transform=ax.get_xaxis_transform(), color="#555555", fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(xplot, ["0"] + [f"{t:g}" for t in LATENCIES_MS[1:]])
        ax.set_xlabel(f"Update latency T (ms); batch = {BATCH_MS:g} ms")
        ax.set_title(f"{drift} drift", fontsize=11)
        ax.grid(True, ls="--", alpha=0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("In-situ steering advantage")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Single-shot streaming RL, surface d={DISTANCE}, steering 2q/1q/qubit params "
                 "(reset & readout drift uncontrolled)", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out + ".png", dpi=140)
    print(f"Saved '{args.out}.json' and '{args.out}.png'")


# ---------------------------------------------------------------------- TLS-jump recovery

def analyse_jumps(traj, mask, omega, events_s):
    """Recovery after each jump and probe cost before it, from the per-batch trajectory (see module docstring)."""
    ctrl = np.flatnonzero(mask)
    t_ms = np.asarray(traj["cycle"]) * CYCLE_US / 1000.0
    p_opt = np.asarray(traj["p_opt"])[:, ctrl]
    e = np.asarray(traj["mu_hw"])[:, ctrl] - p_opt
    sigma = np.asarray(traj["sigma_hw"])[:, ctrl]
    jumps = np.argwhere(np.abs(np.diff(p_opt, axis=0)) > 0.5 * JUMP)  # (batch before the jump, param)
    rel = np.arange(-40, 200)  # batches around the jump: -200 ms .. +1 s
    kernel = np.ones(4) / 4
    curves, sig_curves, rec, paid = [], [], [], []
    for b, i in jumps:
        b += 1
        if b + rel[-1] >= len(t_ms) or b + rel[0] < 0:
            continue
        ei = np.abs(e[b + rel, i])
        curves.append(ei)
        sig_curves.append(sigma[b + rel, i])
        smooth = np.convolve(ei, kernel, mode="full")[:len(ei)]
        after = np.flatnonzero((rel >= len(kernel) - 1) & (smooth <= 0.5))  # average over post-jump batches only
        rec.append(rel[after[0]] * BATCH_MS if len(after) else np.inf)
        paid.append((np.mean(ei[rel >= 0] ** 2) - np.mean(ei[rel < 0] ** 2)) / JUMP ** 2)
    quiet = np.zeros(len(t_ms), dtype=bool)
    for t in events_s:
        quiet |= (t_ms > t * 1000 - 400) & (t_ms <= t * 1000)
    return {
        "rel_ms": (rel * BATCH_MS).tolist(),
        "abs_err": np.mean(curves, axis=0).tolist() if curves else [],
        "sigma_jumped": np.mean(sig_curves, axis=0).tolist() if sig_curves else [],
        "recovery_ms": rec, "paid_fraction": paid,
        "quiet_sigma": float(np.mean(sigma[quiet])),
        "quiet_probe_cost": float(np.mean(omega[ctrl] * sigma[quiet] ** 2)),
        "quiet_rms": float(np.sqrt(np.mean(e[quiet] ** 2))),
    }


def run_jump(args):
    settings = {"total_cycles": int(round(args.total_s * 1e6 / CYCLE_US)), "eval_interval": args.eval_interval,
                "total_s": args.total_s, "events_s": tuple(args.events_s)}
    kinds = GATES_AND_QUBITS
    jobs = [("rl", "hardware", seed, "interleave", lat, kinds, sz) for lat in JUMP_PROFILES for sz in JUMP_SIZINGS
            for seed in args.seeds]
    jobs += [j for seed in args.seeds for j in (("fixed", "hardware", seed), ("opt", "hardware", seed, kinds))]
    out = run_all(jobs, settings, args.workers)

    rows = {}
    print(f"{'latency':22s} {'probes':21s} {'recovery ms (med, p90)':>23s} {'paid':>6s} {'unrec':>6s} | "
          f"{'quiet sigma':>11s} {'probe cost':>10s} {'quiet RMS':>9s} | {'in-situ adv':>11s} {'LER/cyc':>8s}")
    for lat in JUMP_PROFILES:
        for sizing in JUMP_SIZINGS:
            keys = [("rl", "hardware", seed, "interleave", lat, kinds, sizing) for seed in args.seeds]
            jumps = [out[k]["jumps"] for k in keys]
            sums = [with_references(out, "hardware", k[2], k) for k in keys]
            rec = np.concatenate([a["recovery_ms"] for a in jumps])
            paid = np.concatenate([a["paid_fraction"] for a in jumps])
            r = {
                "rel_ms": jumps[0]["rel_ms"],
                "abs_err": np.mean([a["abs_err"] for a in jumps], axis=0).tolist(),
                "sigma_jumped": np.mean([a["sigma_jumped"] for a in jumps], axis=0).tolist(),
                "recovery_median_ms": float(np.median(rec)), "recovery_p90_ms": float(np.percentile(rec, 90)),
                "unrecovered": float(np.mean(~np.isfinite(rec))), "paid_fraction": float(np.mean(paid)),
                **{k: float(np.mean([a[k] for a in jumps])) for k in ("quiet_sigma", "quiet_probe_cost", "quiet_rms")},
                **{k: float(np.mean([s[k] for s in sums])) for k in (
                    "steering_advantage_in_situ", "ler_per_cycle_rl", "ler_per_cycle_fixed", "ler_per_cycle_opt",
                    "stale_fraction")},
                "num_jumps": int(len(rec)),
            }
            rows[(lat, sizing)] = r
            print(f"{lat:22s} {sizing:21s} {r['recovery_median_ms']:>11.0f}, {r['recovery_p90_ms']:>9.0f} "
                  f"{r['paid_fraction']:6.3f} {r['unrecovered']:6.0%} | {r['quiet_sigma']:11.3f} "
                  f"{r['quiet_probe_cost']:10.3f} {r['quiet_rms']:9.3f} | {r['steering_advantage_in_situ']:11.2f} "
                  f"{r['ler_per_cycle_rl']:8.2e}")
    print(f"(fixed-policy LER/cycle {np.mean([r['ler_per_cycle_fixed'] for r in rows.values()]):.2e}, "
          f"reachable optimum {np.mean([r['ler_per_cycle_opt'] for r in rows.values()]):.2e})")
    with open(args.out + ".json", "w") as f:
        json.dump({"config": _config(args, schedule="interleave", control=kinds),
                   "results": {f"{lat} | {sizing}": r for (lat, sizing), r in rows.items()}}, f, indent=1)
    plot_jump(rows, args.out + ".png")


def plot_jump(rows, path):
    plt = _plt()
    colors = dict(zip(JUMP_SIZINGS, ("#2a78d6", "#1baf7a", "#eb6834", "#e87ba4")))
    fig, axes = plt.subplots(2, len(JUMP_PROFILES), figsize=(15, 7), sharex=True, sharey="row")
    for col, lat in enumerate(JUMP_PROFILES):
        for sizing in JUMP_SIZINGS:
            r = rows[(lat, sizing)]
            kw = dict(color=colors[sizing], lw=1.8, label=sizing)
            axes[0, col].plot(r["rel_ms"], r["abs_err"], **kw)
            axes[1, col].plot(r["rel_ms"], r["sigma_jumped"], **kw)
        axes[0, col].axhline(0.5, color="#888888", ls=":", lw=1)
        axes[0, col].set_title(lat, fontsize=11)
        axes[1, col].set_xlabel("Time since jump (ms)")
    axes[0, 0].set_ylabel(f"|mu - p_opt| of jumped qubit params\n(jump = {JUMP}; dotted: recovered)")
    axes[1, 0].set_ylabel("Probe width sigma of jumped params")
    for ax in axes.flat:
        ax.axvline(0, color="#888888", lw=1)
        ax.grid(True, ls="--", alpha=0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, -1].legend(fontsize=8, frameon=False)
    fig.suptitle("TLS-jump recovery, single-shot streaming RL on surface d=5 (mean over jumps and seeds)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"Saved '{path}'")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=("latency", "jump"))
    p.add_argument("--seeds", type=int, nargs="+", default=[163, 164, 165])
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--total-cycles", type=int, default=1_225_000, help="latency: default 35 reference epochs (1.225 s)")
    p.add_argument("--total-s", type=float, default=8.0, help="jump: length of each shot in seconds")
    p.add_argument("--events-s", type=float, nargs="+", default=[2.0, 3.5, 5.0, 6.5], help="jump: TLS jump times (s)")
    p.add_argument("--eval-interval", type=int, default=None, help="default: 50k cycles (latency), 250k (jump)")
    p.add_argument("--out", default=None, help="output path without extension")
    args = p.parse_args()
    if args.experiment == "latency":
        args.eval_interval = args.eval_interval or 50_000
        args.out = args.out or "figures/streaming/latency_sweep_single_shot"
    else:
        args.eval_interval = args.eval_interval or 250_000
        args.out = args.out or "figures/streaming/jump_recovery_single_shot"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    (run_latency if args.experiment == "latency" else run_jump)(args)


if __name__ == "__main__":
    main()
