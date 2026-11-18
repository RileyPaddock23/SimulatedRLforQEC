"""
Real-Time RL Drift Tracking in a Single-Shot QEC Experiment
===========================================================
The streaming counterpart of run_experiment.py: the whole experiment is ONE long memory
shot (see code_envs/streaming_env.py). Time is measured in QEC cycles, and the drift
moves the error rates every `--drift-update-cycles` cycles while the shot runs.

The controller repeats a fixed schedule:

    | sampling window: B candidates x S cycles each | wait: L cycles at the old mean | new mean applies ...

  - Sampling window: candidate k of the PEPG batch drives the hardware for S cycles; its
    observation is the per-detector detection rate over those cycles (the first
    `--discard-cycles` of each slice are dropped, since a cycle's detectors also see the
    previous slice's errors).
  - Wait: the update is being computed and loaded, so the hardware keeps running the old
    policy mean for L cycles. The new mean takes effect when the next window starts.

With `--schedule interleave` the controller doesn't wait: the next sampling window starts
at once around the old mean, and the update re-centres the candidates still to run when it
lands L cycles later (between antithetic pairs). `run_streaming_trial` also takes a
per-parameter latency (see `kind_latency`) and a probe-width rule; the latency sweep and
jump-recovery experiments (streaming_sweeps.py) use them.

The real-time decoder runs alongside: every cycle's detection events stream into a
sliding-window matcher, so the decoder and controller see the same data at the same time.

Two more single-shot experiments run in lock step on the same drift: Fixed (parameters
frozen at the stale t=0 calibration) and Optimal (the drift oracle p_opt(t)). Logical
failures are counted by probing each shot with a virtual readout every `--probe-interval`
cycles (default: one decoder window) and counting how often the answer changes; the
final readout is the last probe.

`--report` switches to the report preset (see PRESETS): a 1 s shot at 1 us per cycle and a
35 ms reference epoch, plus two more lock-step shots, "RL mean" (the mean loaded on the
hardware, no exploration) and "Optimal (reachable)" (the oracle on the steered parameters
only), and the three-panel figure. Explicit flags override the preset.

Examples:
    python run_streaming_experiment.py
    python run_streaming_experiment.py --drift arma --arma-alpha 2.0 --plot
    python run_streaming_experiment.py --latency-cycles 200 --drift-update-cycles 10
    python run_streaming_experiment.py --latency-cycles 200 --schedule interleave
    python run_streaming_experiment.py --report
    python run_streaming_experiment.py --report --latency-cycles 20000 --scale 1e-3
    python run_streaming_experiment.py --replot figures/streaming/<run>.json
"""

import argparse
import json
import time
from collections import deque
from typing import Callable, Dict, Optional, Union

import numpy as np

from rl_agent import RLControlAgent
from run_experiment import add_env_arguments, build_env, describe_drift, make_agent
from code_envs.qec_env import QECCodeEnv
from code_envs.streaming_env import StreamingQECEnv

# Sliding-window matching needs a graphlike detector error model, so color codes are out for now
STREAM_CODES = ("surface",)
# Parameter kinds the agent can be allowed to steer: gate parameters by gate kind, plus the per-qubit parameters
STEER_KINDS = ("2q", "1q", "reset", "meas", "qubit")
# What the hardware runs while an update is in flight (see run_streaming_trial)
SCHEDULES = ("wait", "interleave")


def param_kinds(env: QECCodeEnv) -> np.ndarray:
    """Each parameter's kind (see STEER_KINDS): its gate's kind, or "qubit"."""
    gate_block = env.blocks[0]
    kinds = np.empty(env.num_params, dtype=object)
    kinds[env.gate_param_slice] = np.repeat([src.kind for src in gate_block.owners], gate_block.params_per_owner)
    kinds[env.qubit_param_slice] = "qubit"
    return kinds.astype(str)


def steer_mask(env: QECCodeEnv, kinds) -> np.ndarray:
    """True for the parameters of the given kinds (see STEER_KINDS); the rest stay at their t=0 calibration."""
    unknown = set(kinds) - set(STEER_KINDS)
    if unknown:
        raise ValueError(f"unknown parameter kinds {sorted(unknown)} (use {STEER_KINDS})")
    return np.isin(param_kinds(env), list(kinds))


def kind_latency(env: QECCodeEnv, cycles_by_kind: Dict[str, int], default: int = 0) -> np.ndarray:
    """Per-parameter latency in cycles from a per-kind table, e.g. {"2q": 10_000, "qubit": 5_000}."""
    return np.array([cycles_by_kind.get(k, default) for k in param_kinds(env)], dtype=np.int64)


def steady_state(values) -> float:
    """Mean over the last third of checkpoints, leaving out the final one (it only covers a partial interval)."""
    n = len(values)
    tail = slice(max(0, n - 1 - max(1, (n - 1) // 3)), max(1, n - 1))
    return float(np.nanmean(np.asarray(values, dtype=float)[tail]))


def run_streaming_trial(
    env: QECCodeEnv,
    agent: RLControlAgent,
    total_cycles: int,
    cycles_per_candidate: int,
    batch_size: int,
    latency: Union[int, np.ndarray],
    eval_interval: int,
    cycles_per_epoch: float,
    schedule: str = "wait",
    drift_update_cycles: int = 2,
    discard_cycles: int = 1,
    window: Optional[int] = None,
    commit: Optional[int] = None,
    probe_interval: Optional[int] = None,
    num_streams: int = 1,
    baselines: bool = True,
    probe_sizer: Optional[Callable[[RLControlAgent], None]] = None,
    seed: int = 0,
    verbose: bool = True,
    mean_shot: bool = False,
    extra_policies: Optional[Dict[str, Union[np.ndarray, Callable[[float], np.ndarray]]]] = None,
) -> Dict:
    """
    One single-shot run of the controller (see the module docstring). `latency` is in
    cycles, one value or one per parameter (see `kind_latency`): each parameter's new mean
    and probe width reach the hardware that long after the batch ends. `schedule`:

      - "wait": hold the loaded mean until every parameter of the update has landed, then
        start the next batch.
      - "interleave": start the next batch at once. Its perturbation directions z are drawn
        when the previous batch ends and applied as hw_mu + hw_sigma * z with whatever is
        loaded when the candidate runs, so a landing update re-centres the candidates still
        to run. Updates only land between antithetic pairs, so both halves of a pair share
        a mean. The update uses the mean and width each candidate was actually drawn with.

    `probe_sizer`, if given, sets agent.sigma after every update (in place of the agent's
    own sigma step). Fixed and Optimal (the drift oracle) run in lock step when `baselines`,
    plus one shot per `extra_policies` entry (tag -> vector or function of t). With
    `mean_shot`, a "mean" shot runs whatever mean the hardware has loaded (hw_mu, no
    exploration) in lock step with the RL shot, candidate by candidate: the policy the
    controller would deploy, on the same drift with independent noise. The result's
    "windows" log the RL shot per candidate slice (end cycle, detection rate, logical
    failures, whether an update was in flight).
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule '{schedule}' (use {SCHEDULES})")
    if discard_cycles >= cycles_per_candidate:
        raise ValueError("discard_cycles must be smaller than cycles_per_candidate")
    make = lambda k: StreamingQECEnv(env, cycles_per_epoch=cycles_per_epoch, drift_update_cycles=drift_update_cycles,
                                     window=window, commit=commit, num_streams=num_streams, seed=seed + k,
                                     probe_interval=probe_interval)
    shots = {"rl": make(0)}
    policies = {"fixed": np.zeros(env.num_params, dtype=np.float32), "opt": env.get_drift_opt}
    if baselines:
        shots.update(fixed=make(1), opt=make(2))
    if mean_shot:
        shots["mean"] = make(3)
    for j, (tag, policy) in enumerate((extra_policies or {}).items()):
        policies[tag] = policy
        shots[tag] = make(4 + j)
    baseline_tags = [tag for tag in shots if tag in policies]
    follows_mu = [tag for tag in ("rl", "mean") if tag in shots]
    rl = shots["rl"]
    for tag, shot in shots.items():
        shot.reset(agent.mu.copy() if tag in follows_mu else policies[tag])

    steered = agent.steer_mask > 0
    latency = np.broadcast_to(np.asarray(latency, dtype=np.int64), (env.num_params,)).copy()
    hold = int(latency[steered].max()) if steered.any() and schedule == "wait" else 0
    batch_cycles = batch_size * cycles_per_candidate
    hw_mu, hw_sigma = agent.mu.copy(), agent.sigma.copy()  # what the hardware registers hold
    pending = deque()  # (ready cycle per parameter, mu, sigma) of updates not yet fully loaded, oldest first
    counts = {"candidates": 0, "stale": 0, "hold_cycles": 0}

    # Tracking error over the parameters the agent steers and can see; frozen ones only drift away
    tracked = steered & rl.active_params
    history = {k: [] for k in ("cycle", "epoch", "tracking_rms", "frozen_rms", "mu_0", "p_opt_0", "edr_rl_mean_policy",
                               *(f"{m}_{tag}" for m in ("edr", "failures") for tag in shots))}
    # Per-checkpoint accumulators: detection events and cycles since the last checkpoint
    acc = {tag: [0.0, 0] for tag in shots}
    acc_mean_policy = [0.0, 0]  # RL while holding the loaded mean, i.e. the policy mean without exploration
    traj = {"cycle": [], "mu": [], "mu_hw": [], "sigma_hw": [], "p_opt": [], "edr_window": []}
    windows = {"cycle": [], "edr": [], "failures": [], "stale": []}  # RL shot, one row per candidate slice

    def add(tag, block):
        acc[tag][0] += float(block.detection_events.sum()) / block.detection_events.shape[2]
        acc[tag][1] += block.detection_events.shape[0] * num_streams

    def follow_mean():
        """Brings the mean shot up to the RL shot's cycle at the currently loaded mean."""
        if "mean" in shots and shots["mean"].cycle < rl.cycle:
            add("mean", shots["mean"].run_cycles(hw_mu.copy(), rl.cycle - shots["mean"].cycle))

    def land_ready():
        """Loads each parameter of the pending updates whose latency has elapsed, oldest first."""
        for ready, mu, sigma in pending:
            due = ready <= rl.cycle
            hw_mu[due], hw_sigma[due] = mu[due], sigma[due]
        while pending and np.all(pending[0][0] <= rl.cycle):
            pending.popleft()

    def checkpoint(final_readout: bool = False):
        p_opt = env.get_drift_opt(rl.t)
        for tag, shot in shots.items():
            if final_readout:  # the real readout counts as the last probe
                shot.finish(hw_mu if tag in follows_mu else policies[tag])
            history[f"edr_{tag}"].append(acc[tag][0] / max(1, acc[tag][1]))
            history[f"failures_{tag}"].append(int(shot.logical_failures.sum()))
            acc[tag] = [0.0, 0]
        history["edr_rl_mean_policy"].append(acc_mean_policy[0] / max(1, acc_mean_policy[1]))
        acc_mean_policy[:] = [0.0, 0]
        history["cycle"].append(rl.cycle)
        history["epoch"].append(rl.t)
        # Over the parameters that act during cycles; the others (e.g. data readout) get no signal
        history["tracking_rms"].append(float(np.sqrt(np.mean((hw_mu - p_opt)[tracked] ** 2))))
        history["frozen_rms"].append(float(np.sqrt(np.mean((hw_mu - p_opt)[~steered] ** 2))) if (~steered).any() else 0.0)
        history["mu_0"].append(float(hw_mu[0]))
        history["p_opt_0"].append(float(p_opt[0]))
        if verbose:
            edr = " | ".join(f"{history[f'edr_{tag}'][-1]:>8.4f}" for tag in shots)
            fails = " | ".join(f"{history[f'failures_{tag}'][-1]:>6d}" for tag in shots)
            print(f"{rl.cycle:>9,} | {edr} | {fails} | {history['tracking_rms'][-1]:>7.3f}")

    if verbose:
        names = list(shots)
        header = (f"{'Cycle':>9} | " + " | ".join(f"{'EDR ' + n:>8}" for n in names) + " | "
                  + " | ".join(f"{'LF ' + n:>6}" for n in names) + f" | {'RMS err':>7}")
        print(header + "\n" + "-" * len(header))
        print("(EDR = detection rate since the previous row; LF = cumulative logical failures)")

    next_eval = eval_interval
    while rl.cycle + batch_cycles + hold <= total_cycles:
        # Sampling window: candidates drive the hardware one after another, around whatever mean is loaded
        z = agent.sample_directions(batch_size, antithetic=True)
        cands = np.empty((batch_size, env.num_params), dtype=np.float32)
        centers, sigmas = np.empty_like(cands), np.empty_like(cands)
        obs = np.zeros((batch_size, rl.num_detectors), dtype=np.float32)
        for k in range(batch_size):
            if k % 2 == 0:
                land_ready()  # an update only lands between antithetic pairs
            counts["candidates"] += 1
            counts["stale"] += bool(pending)
            centers[k], sigmas[k] = hw_mu, hw_sigma
            cands[k] = hw_mu + hw_sigma * z[k]
            failures_before = int(rl.logical_failures.sum())
            block = rl.run_cycles(cands[k], cycles_per_candidate)
            obs[k] = block.detection_events[discard_cycles:].mean(axis=(0, 1))
            add("rl", block)
            follow_mean()
            windows["cycle"].append(rl.cycle)
            windows["edr"].append(float(block.detection_events.mean()))
            windows["failures"].append(int(rl.logical_failures.sum()) - failures_before)
            windows["stale"].append(bool(pending))
        # The update is computed now and reaches each parameter's register `latency` cycles later
        agent.update(cands, obs, centers=centers, sigmas=sigmas)
        if probe_sizer is not None:
            probe_sizer(agent)
        pending.append((rl.cycle + latency, agent.mu.copy(), agent.sigma.copy()))
        if schedule == "wait":  # hold the loaded mean until every parameter of the update has landed
            for ready in np.unique(pending[-1][0][steered]):
                land_ready()
                n = int(ready) - rl.cycle
                if n > 0:
                    block = rl.run_cycles(hw_mu.copy(), n)
                    add("rl", block)
                    acc_mean_policy[0] += float(block.detection_events.sum()) / rl.num_detectors
                    acc_mean_policy[1] += n * num_streams
                    counts["hold_cycles"] += n
                    follow_mean()
            land_ready()
        for tag in baseline_tags:
            add(tag, shots[tag].run_cycles(policies[tag], rl.cycle - shots[tag].cycle))

        traj["cycle"].append(rl.cycle)
        traj["mu"].append(agent.mu.copy())
        traj["mu_hw"].append(hw_mu.copy())
        traj["sigma_hw"].append(hw_sigma.copy())
        traj["p_opt"].append(env.get_drift_opt(rl.t))
        traj["edr_window"].append(float(obs.mean()))
        if rl.cycle >= next_eval:
            checkpoint()
            next_eval += eval_interval
    checkpoint(final_readout=True)

    s = {f"edr_{tag}": steady_state(history[f"edr_{tag}"]) for tag in shots}
    s["edr_rl_mean_policy"] = steady_state(history["edr_rl_mean_policy"]) if counts["hold_cycles"] else float("nan")
    s["tracking_rms"] = steady_state(history["tracking_rms"])
    s["frozen_rms"] = steady_state(history["frozen_rms"])
    cycles_run = rl.cycle * num_streams
    for tag, shot in shots.items():
        s[f"logical_failures_{tag}"] = int(shot.logical_failures.sum())
        s[f"ler_per_cycle_{tag}"] = s[f"logical_failures_{tag}"] / cycles_run
        st = shot.decoder.stats
        s[f"decode_us_per_cycle_{tag}"] = 1e6 * st["decode_seconds"] / max(1, st["chunks"] * num_streams)
    if baselines:
        s["steering_advantage"] = (s["edr_fixed"] - s["edr_rl"]) / max(1e-5, s["edr_fixed"] - s["edr_opt"])
    s["num_updates"] = len(traj["mu"])
    s["stale_fraction"] = counts["stale"] / max(1, counts["candidates"])  # candidates run with an update in flight
    s["hold_fraction"] = counts["hold_cycles"] / max(1, rl.cycle)
    s["cycles"] = rl.cycle
    s["epochs"] = rl.t
    return {"summary": s, "history": history, "windows": windows,
            "trajectory": {k: np.asarray(v).tolist() for k, v in traj.items()}}


def run_policy_shot(
    env: QECCodeEnv,
    policy: Union[np.ndarray, Callable[[float], np.ndarray]],
    total_cycles: int,
    eval_interval: int,
    cycles_per_epoch: float,
    drift_update_cycles: int = 2,
    window: Optional[int] = None,
    commit: Optional[int] = None,
    probe_interval: Optional[int] = None,
    seed: int = 0,
) -> Dict:
    """
    One single shot of a fixed policy (a vector, or a function of t such as the drift
    oracle), checkpointed every `eval_interval` cycles. Lets many controller runs on the
    same drift (same env seed) share one Fixed and one Optimal shot.
    """
    shot = StreamingQECEnv(env, cycles_per_epoch=cycles_per_epoch, drift_update_cycles=drift_update_cycles,
                           window=window, commit=commit, seed=seed, probe_interval=probe_interval)
    shot.reset(policy)
    history = {"cycle": [], "edr": [], "failures": []}
    while shot.cycle < total_cycles:
        block = shot.run_cycles(policy, min(eval_interval - shot.cycle % eval_interval, total_cycles - shot.cycle))
        if shot.cycle >= total_cycles:
            shot.finish(policy)
        history["cycle"].append(shot.cycle)
        history["edr"].append(float(block.detection_events.mean()))
        history["failures"].append(int(shot.logical_failures.sum()))
    return {"summary": {"edr": steady_state(history["edr"]), "logical_failures": history["failures"][-1],
                        "ler_per_cycle": history["failures"][-1] / shot.cycle, "cycles": shot.cycle},
            "history": history}


SEGMENT = 5  # cycles per reported logical-error segment
# Per-kind colors (the repo's per-parameter palette); line style is the secondary encoding
KIND_STYLE = {"2q": ("#2a78d6", "-", "2q gate"), "1q": ("#eb6834", "--", "1q gate"),
              "reset": ("#1baf7a", "-.", "reset"), "meas": ("#eda100", ":", "readout"),
              "qubit": ("#c2448a", (0, (5, 1, 1, 1)), "qubit")}


def per_segment(failures, cycles):
    """Probability of >= 1 logical failure in SEGMENT cycles from a failure count over `cycles` cycles."""
    p = np.asarray(failures, dtype=float) / np.maximum(np.asarray(cycles, dtype=float), 1)
    return 1.0 - (1.0 - np.clip(p, 0, 1)) ** SEGMENT


def reachable_opt(env: QECCodeEnv, mask: np.ndarray) -> Callable[[float], np.ndarray]:
    """The drift oracle on the steered parameters, the t = 0 calibration elsewhere."""
    return lambda t: np.where(mask, env.get_drift_opt(t), 0.0).astype(np.float32)


def plot_report(res: dict, path: str, title: str, cycle_us: float = 1.0, smooth: int = 20, rms_updates: int = 5):
    """
    Three panels over time (ms): detection rate and logical error rate per SEGMENT-cycle segment of every
    shot (Fixed, RL mean, reachable and full Optimal, whichever ran, plus the RL shot in situ as a rolling
    mean over `smooth` candidate slices), and the per-kind tracking RMS of the steered parameters.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w, tr = res["history"], res["windows"], res["trajectory"]
    ms = lambda c: np.asarray(c, dtype=float) * cycle_us / 1000.0
    cyc = np.asarray(h["cycle"], dtype=float)
    span = np.diff(np.concatenate([[1.0], cyc]))  # cycles in each checkpoint interval (cycle 0 is the prefix)
    keep = span > 0  # the final readout adds an empty checkpoint when the shot ends on an eval boundary
    x_mid = ms(cyc - span / 2)[keep]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    series = [("fixed", "#d62728", "-", "Fixed"), ("mean", "#1f77b4", "-", "RL mean"),
              ("opt_reach", "#2ca02c", "--", "Optimal (reachable)"), ("opt", "#7f7f7f", ":", "Optimal (all params)")]
    for tag, c, ls, lab in series:
        if f"edr_{tag}" not in h:
            continue
        axes[0].plot(x_mid, np.asarray(h[f"edr_{tag}"])[keep], color=c, ls=ls, lw=1.6, label=lab)
        fails = np.diff(np.concatenate([[0], h[f"failures_{tag}"]]))
        axes[1].plot(x_mid, per_segment(fails, span)[keep], color=c, ls=ls, lw=1.6, label=lab)

    # In situ: the RL shot itself, per candidate slice, as a rolling mean over `smooth` slices
    wc = np.asarray(w["cycle"], dtype=float)
    width = np.diff(np.concatenate([[1.0], wc]))
    smooth = max(1, min(smooth, len(wc)))
    kern = np.ones(smooth)
    roll = lambda v: np.convolve(np.asarray(v, dtype=float), kern, mode="valid")
    xw = ms(wc[smooth - 1:] - roll(width) / 2)
    axes[0].plot(xw, roll(np.asarray(w["edr"]) * width) / roll(width), color="#ff7f0e", lw=0.8, alpha=0.85,
                 label=f"In situ ({smooth}-window mean)")
    axes[1].plot(xw, per_segment(roll(w["failures"]), roll(width)), color="#ff7f0e", lw=0.8, alpha=0.85,
                 label=f"In situ ({smooth}-window mean)")

    # Tracking: |mu_i - p_opt_i| of each steered, cycle-visible parameter, RMS over `rms_updates` updates
    tracked = np.asarray(res["tracked"], dtype=bool)
    kinds = np.asarray(res["param_kinds"])[tracked]
    err2 = (np.asarray(tr["mu_hw"]) - np.asarray(tr["p_opt"]))[:, tracked] ** 2
    k = max(1, min(rms_updates, len(err2)))
    err2 = np.stack([err2[i:i + k].mean(axis=0) for i in range(len(err2) - k + 1)])
    xt = ms(np.asarray(tr["cycle"], dtype=float)[k - 1:])
    ax = axes[2]
    for kind in [kd for kd in KIND_STYLE if kd in set(kinds)]:
        c, ls, lab = KIND_STYLE[kind]
        sel = kinds == kind
        ax.plot(xt, np.sqrt(err2[:, sel]), color=c, lw=0.6, alpha=0.08)
        ax.plot(xt, np.sqrt(err2[:, sel].mean(axis=1)), color=c, ls=ls, lw=2.2, label=f"{lab} (RMS over {int(sel.sum())})")
    ax.plot(xt, np.sqrt(err2.mean(axis=1)), color="#1f1f1e", lw=1.4, label=f"All steered ({len(kinds)})")
    ax.set_ylim(bottom=0)

    axes[0].set_title("Mean detection event rate")
    axes[1].set_title(f"Logical error rate per {SEGMENT}-cycle segment")
    axes[2].set_title(r"Tracking RMS $|\mu_i - p^{opt}_i|$ (steered params)")
    for a in axes:
        a.set_xlabel("Time (ms)")
        a.grid(True, ls="--", alpha=0.5)
        a.legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Plot saved to '{path}'")


# Defaults of the flags that `--report` changes. The report preset: a 1 s shot at 1 us per cycle and a
# 35 ms reference epoch (run_experiment.py's shot clock), interleaving 20 x 500-cycle candidates with a
# 5 ms latency at noise scale 8e-4, steering gates and qubits, with the RL-mean and reachable-Optimal shots.
PRESETS = {
    "default": dict(total_cycles=50_000, cycles_per_epoch=1500.0, drift_update_cycles=2, cycles_per_candidate=40,
                    batch_size=10, latency_cycles=20, schedule="wait", eval_interval=2_000, steer=STEER_KINDS,
                    scale=None, mean_shot=False, reachable_opt=False),
    "report": dict(total_cycles=1_000_001, cycles_per_epoch=35_000.0, drift_update_cycles=50, cycles_per_candidate=500,
                   batch_size=20, latency_cycles=5000, schedule="interleave", eval_interval=50_000,
                   steer=("2q", "1q", "qubit"), scale=8e-4, mean_shot=True, reachable_opt=True, plot=True),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_arguments(p, code_default="surface", codes=STREAM_CODES)
    p.add_argument("--report", action="store_true", help="use the report preset (see PRESETS); explicit flags still win")
    p.add_argument("--scale", type=float, help="sets --base-error and --error-budget together")
    # Streaming schedule (unset flags take the preset's value)
    p.add_argument("--total-cycles", type=int, help="length of the single shot, in QEC cycles")
    p.add_argument("--cycles-per-epoch", type=float, help="time compression: QEC cycles per reference drift epoch")
    p.add_argument("--cycle-us", type=float, default=1.0, help="physical length of a QEC cycle (plot time axis)")
    p.add_argument("--drift-update-cycles", type=int, help="error rates follow the drift every this many cycles")
    p.add_argument("--cycles-per-candidate", type=int, help="S: cycles each candidate drives the hardware")
    p.add_argument("--batch-size", type=int, help="B: candidates per sampling window")
    p.add_argument("--latency-cycles", "--wait-cycles", dest="latency_cycles", type=int,
                   help="L: latency from a batch's end to its update applying")
    p.add_argument("--schedule", choices=SCHEDULES,
                   help="wait: hold the old mean for L cycles; interleave: keep sampling while the update is in flight")
    p.add_argument("--discard-cycles", type=int, default=1, help="cycles dropped at the start of each candidate's slice")
    # Real-time decoder
    p.add_argument("--window", type=int, default=None, help="decoder window in cycles (default 2d)")
    p.add_argument("--commit", type=int, default=None, help="cycles committed per window (default window // 2)")
    # Evaluation
    p.add_argument("--eval-interval", type=int, help="cycles between table rows")
    p.add_argument("--probe-interval", type=int, default=None,
                   help="cycles between virtual logical readouts that count failures (default: the decoder window)")
    p.add_argument("--num-streams", type=int, default=1,
                   help="independent copies of each shot (same schedule; the controller averages over them)")
    p.add_argument("--steer", type=lambda t: tuple(k.strip() for k in t.split(",")),
                   help=f"comma-separated parameter kinds the agent steers, from {','.join(STEER_KINDS)}; "
                        "the others stay at their t=0 calibration but still drift (default: all)")
    p.add_argument("--no-baselines", action="store_true", help="skip the Fixed and Optimal shots (3x faster)")
    p.add_argument("--mean-shot", action="store_true", default=None,
                   help="also run the loaded mean without exploration in lock step (RL mean)")
    p.add_argument("--reachable-opt", action="store_true", default=None,
                   help="also run the drift oracle on the steered parameters only (Optimal, reachable)")
    p.add_argument("--plot", action="store_true", default=None, help="save the three-panel figure of the run")
    p.add_argument("--out", type=str, default=None, help="JSON output path")
    p.add_argument("--replot", type=str, default=None, help="redraw the figure of a saved results JSON")
    args = p.parse_args()

    if args.replot:
        with open(args.replot) as f:
            res = json.load(f)
        plot_report(res, args.replot.replace(".json", ".png"), res["title"], res["config"].get("cycle_us", 1.0))
        return
    for key, value in PRESETS["report" if args.report else "default"].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if args.scale is not None:
        args.base_error = args.error_budget = args.scale

    t0 = time.time()
    env = build_env(args, max_epochs=int(np.ceil(args.total_cycles / args.cycles_per_epoch)) + 1)
    # The agent sees one cycle's detectors, so it gets the streaming factor graph
    probe_env = StreamingQECEnv(env, cycles_per_epoch=args.cycles_per_epoch)
    mask = steer_mask(env, args.steer)
    agent = make_agent(probe_env, seed=args.seed, steer_mask=mask)
    kinds = param_kinds(env)
    probe_interval = args.probe_interval or probe_env.decoder.window

    drift_desc = describe_drift(env.drift)
    period = args.batch_size * args.cycles_per_candidate + (args.latency_cycles if args.schedule == "wait" else 0)
    print(f"{args.code} code d={args.distance} | {env.num_params} params ({int(probe_env.active_params.sum())} act "
          f"during cycles), {probe_env.num_detectors} detectors per cycle | setup {time.time() - t0:.1f}s")
    for kd in STEER_KINDS:
        sel = kinds == kd
        print(f"  {kd:>5}: {int(sel.sum()):3d} params, {int((sel & probe_env.active_params).sum()):3d} act in cycles, "
              f"{'steered' if kd in args.steer else 'frozen'}")
    print(f"Drift: {drift_desc}" + (f" | qubit drift: {describe_drift(env.qubit_drift)}" if env.num_qubit_params else ""))
    print(f"Single shot of {args.total_cycles:,} cycles ({args.total_cycles / args.cycles_per_epoch:.1f} epochs, "
          f"{args.total_cycles * args.cycle_us / 1000:g} ms), drift updated every {args.drift_update_cycles} cycles")
    print(f"Schedule '{args.schedule}': sample B={args.batch_size} x S={args.cycles_per_candidate} cycles, "
          f"latency L={args.latency_cycles} -> {period} cycles per update, {period / args.cycles_per_epoch:.2f} epochs")
    print(f"Real-time decoder: window {probe_env.decoder.window}, commit {probe_env.decoder.commit} cycles | "
          f"logical readout probed every {probe_interval} cycles\n")

    result = run_streaming_trial(
        env, agent,
        total_cycles=args.total_cycles,
        cycles_per_candidate=args.cycles_per_candidate,
        batch_size=args.batch_size,
        latency=args.latency_cycles,
        schedule=args.schedule,
        eval_interval=args.eval_interval,
        cycles_per_epoch=args.cycles_per_epoch,
        drift_update_cycles=args.drift_update_cycles,
        discard_cycles=args.discard_cycles,
        window=args.window,
        commit=args.commit,
        probe_interval=probe_interval,
        num_streams=args.num_streams,
        baselines=not args.no_baselines,
        seed=args.seed,
        mean_shot=args.mean_shot,
        extra_policies={"opt_reach": reachable_opt(env, mask)} if args.reachable_opt else None,
    )
    result["tracked"] = (mask & probe_env.active_params).tolist()
    result["param_kinds"] = kinds.tolist()
    result["title"] = (f"Streaming RL, d={args.distance}, {args.schedule}: B={args.batch_size}x{args.cycles_per_candidate} "
                       f"cycles, latency {args.latency_cycles * args.cycle_us / 1000:g} ms; {drift_desc}"
                       + (f"; scale {args.scale:g}" if args.scale else "") + f"; steering {','.join(args.steer)}")
    result["wall_seconds"] = time.time() - t0

    s = result["summary"]
    tags = [t for t in ("fixed", "rl", "mean", "opt_reach", "opt") if f"edr_{t}" in s]
    print(f"\n{s['num_updates']} updates over {s['cycles']:,} cycles ({s['epochs']:.1f} epochs), {time.time() - t0:.1f}s total")
    print("  Steady state (last third of the shot):")
    print("    EDR  " + " | ".join(f"{t} {s[f'edr_{t}']:.4f}" for t in tags)
          + (f"  (steering advantage {s['steering_advantage']:.2f})" if "steering_advantage" in s else ""))
    if not np.isnan(s["edr_rl_mean_policy"]):
        print(f"    EDR  RL during wait cycles (policy mean, no exploration) {s['edr_rl_mean_policy']:.4f}")
    if args.schedule == "interleave":
        print(f"    Candidates run with an update in flight: {s['stale_fraction']:.0%}")
    print(f"    Tracking RMS error |mu - p_opt|: steered parameters {s['tracking_rms']:.3f} | frozen {s['frozen_rms']:.3f}")
    print("  Whole shot, real-time decoder:")
    print("    Logical failures  " + " | ".join(f"{t} {s[f'logical_failures_{t}']}" for t in tags))
    print("    LER per cycle     " + " | ".join(f"{t} {s[f'ler_per_cycle_{t}']:.2e}" for t in tags))
    print(f"    LER per {SEGMENT} cycles  " + " | ".join(f"{t} {per_segment(s[f'logical_failures_{t}'], s['cycles']):.4f}" for t in tags))
    print("    Decode time       " + " | ".join(f"{t} {s[f'decode_us_per_cycle_{t}']:.0f} us/cycle" for t in tags))

    if args.out:
        out = args.out
    elif args.report:
        out = (f"figures/streaming/report_{args.schedule}_B{args.batch_size}S{args.cycles_per_candidate}"
               f"_L{args.latency_cycles}_scale{args.scale:g}.json")
    else:
        out = f"results_streaming_{args.drift}.json"
    result["config"] = {**vars(args), "probe_interval_used": probe_interval, "decoder_window": probe_env.decoder.window,
                        "decoder_commit": probe_env.decoder.commit}
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"Results saved to '{out}'")
    if args.plot:
        plot_report(result, out.replace(".json", ".png"), result["title"], args.cycle_us)


if __name__ == "__main__":
    main()
