"""
Per-Parameter Tracking Error vs. Detecting-Region Structure
===========================================================
Runs the masked PEPG agent on the surface and color codes and breaks the steady-state
tracking error |mu_i - p_opt_i| down by parameter, against two properties of the
factor-graph matrix M:

  - column weight     k_i = |M_i|, the number of detectors parameter i can flip
  - neighbourhood     n_i = #{j != i : M_i and M_j share a detector}, the parameters whose
                      perturbations leak into i's masked advantage

plus its signal strength, the number of noisy operations in the circuit it controls,
and by parameter class: CX coupler vs. single-qubit (1q) vs. qubit parameter, and bulk vs.
boundary (a qubit is on the boundary if it has fewer couplers than the most-connected qubit
of its role, data or measure; a coupler is on the boundary if either of its qubits is).

Each code is run gate-only (params_per_qubit = 0) and with qubit parameters (= 1). The
geometry (M) is fixed across seeds while the sensitivities and drift realizations are
redrawn, so averaging over seeds isolates the effect of the geometry.

    python -m analytics.per_param_tracking --seeds 4
    python -m analytics.per_param_tracking --drift-frequency 0.02 --out figures/per_param/per_param_tracking_f0.02.json
"""

import argparse
import collections
import json
import os
import time

import numpy as np
import stim
from scipy.stats import spearmanr

from code_envs.surface_code_env import RotatedSurfaceCodeEnv
from code_envs.color_code_env import ColorCodeEnv
from run_experiment import SHOTS_PER_EPOCH_REF, make_agent, run_trial

CODES = {"surface": RotatedSurfaceCodeEnv, "color": ColorCodeEnv}
TYPES = ("CX", "1q", "qubit")


def parameter_features(env) -> dict:
    """Column weight, neighbourhood size, class, role and boundary flag for every parameter."""
    M = env.M > 0
    shared = M.T.astype(np.int32) @ M.astype(np.int32)  # (P x P) shared detector counts
    np.fill_diagonal(shared, 0)

    # Qubit roles (measure qubits are measured every round) and boundary flags from coupler degree
    backend = env.backend
    meas = collections.Counter()
    for op in backend.flat_instructions:
        if stim.gate_data(op.name).produces_measurements:
            meas.update(t.value for t in op.targets_copy() if t.is_qubit_target)
    degree = collections.Counter(q for pair in backend.cx_pairs for q in pair)
    role = {q: "measure" if meas[q] > 1 else "data" for q in set(meas) | set(degree)}
    max_degree = {r: max(degree[q] for q in role if role[q] == r) for r in ("data", "measure")}
    on_boundary = {q: degree[q] < max_degree[role[q]] for q in role}

    kind, roles, boundary = [], [], []
    ppg = env.num_gate_params // backend.num_sources if env.num_gate_params else 0
    for src in backend.sources:
        for _ in range(ppg):
            kind.append("CX" if src.kind == "2q" else "1q")
            roles.append("coupler" if src.kind == "2q" else role[src.qubits[0]])
            boundary.append(any(on_boundary[q] for q in src.qubits))
    ppq = env.num_qubit_params // max(1, len(env.param_qubits or []))
    for q in env.param_qubits or []:
        for _ in range(ppq):
            kind.append("qubit")
            roles.append(role[q])
            boundary.append(on_boundary[q])

    # Signal strength: the noisy operations in the circuit each parameter controls
    W = np.concatenate([np.repeat(b.weights, b.params_per_owner, axis=1) for b in env.blocks], axis=1)
    noisy_ops = backend.source_instances @ (W > 0)

    return {
        "noisy_ops": noisy_ops.tolist(),
        "column_weight": M.sum(axis=0).tolist(),
        "neighbourhood": (shared > 0).sum(axis=1).tolist(),
        "type": kind,
        "role": roles,
        "boundary": boundary,
    }


def make_env(code: str, params_per_qubit: int, seed: int, args):
    return CODES[code](
        distance=args.distance, rounds=args.rounds, params_per_qubit=params_per_qubit,
        drift_profile=args.drift, drift_frequency=args.drift_frequency,
        max_epochs=int(np.ceil(args.total_shots / SHOTS_PER_EPOCH_REF)) + 1, seed=seed,
    )


def run_one(job) -> list:
    """One trial; returns the per-parameter tracking RMS."""
    code, params_per_qubit, seed, args = job
    env = make_env(code, params_per_qubit, seed, args)
    result = run_trial(env, make_agent(env, seed), total_shots=args.total_shots,
                       shots_per_candidate=args.shots_per_candidate, batch_size=args.batch_size,
                       eval_interval=args.total_shots, eval_shots=args.eval_shots, verbose=False)
    print(f"  {code:7s} ppq={params_per_qubit} seed={seed}: overall RMS {result['summary']['tracking_rms']:.3f}", flush=True)
    return result["per_param_tracking_rms"]


def run_configs(args) -> list:
    """Runs every (code, params_per_qubit, seed) trial in parallel and averages each config over seeds."""
    from multiprocessing import Pool

    seeds = list(range(163, 163 + args.seeds))
    configs = [(code, ppq) for code in args.codes for ppq in (0, 1)]
    jobs = [(code, ppq, seed, args) for code, ppq in configs for seed in seeds]
    with Pool(min(args.workers, len(jobs))) as pool:
        rms = pool.map(run_one, jobs)

    out = []
    for c, (code, ppq) in enumerate(configs):
        env = make_env(code, ppq, seeds[0], args)  # M (and so every feature) is the same for every seed
        per_seed = rms[c * len(seeds):(c + 1) * len(seeds)]
        out.append({
            "code": code,
            "params_per_qubit": ppq,
            "num_detectors": env.num_detectors,
            "seeds": seeds,
            **parameter_features(env),
            "tracking_rms": np.mean(per_seed, axis=0).tolist(),  # per parameter, averaged over seeds
            "tracking_rms_seed_std": np.std(per_seed, axis=0).tolist(),
        })
    return out


def summarize(cfg: dict):
    """Prints per-class means and within-class Spearman correlations."""
    rms, ops, k, n = (np.array(cfg[key]) for key in ("tracking_rms", "noisy_ops", "column_weight", "neighbourhood"))
    kind, boundary = np.array(cfg["type"]), np.array(cfg["boundary"])
    se = np.median(np.array(cfg["tracking_rms_seed_std"]) / np.sqrt(len(cfg["seeds"])))
    print(f"\n{cfg['code']} code, params_per_qubit={cfg['params_per_qubit']} "
          f"({len(rms)} params, {cfg['num_detectors']} detectors, overall RMS {np.sqrt(np.mean(rms ** 2)):.3f}, "
          f"median per-param s.e. {se:.3f})")
    print(f"  {'class':16s} {'count':>5s} {'RMS':>6s} {'ops':>5s} {'k':>6s} {'n':>6s} | "
          f"{'rho(RMS,ops)':>12s} {'rho(RMS,k)':>10s} {'rho(RMS,n)':>10s}")
    for t in (*TYPES, None):
        for label, sel in (("all", kind == t), ("bulk", (kind == t) & ~boundary), ("boundary", (kind == t) & boundary)):
            if t is None:  # pooled over every parameter
                label, sel = "pooled", np.ones(len(rms), dtype=bool)
            if sel.sum() == 0:
                continue
            corr = lambda x: spearmanr(x[sel], rms[sel]).statistic if sel.sum() > 3 and np.ptp(x[sel]) > 0 else np.nan
            print(f"  {(t or '') + ' ' + label:16s} {sel.sum():5d} {rms[sel].mean():6.3f} {ops[sel].mean():5.1f} "
                  f"{k[sel].mean():6.1f} {n[sel].mean():6.1f} | {corr(ops):12.2f} {corr(k):10.2f} {corr(n):10.2f}")
            if t is None:
                break


# ---------------------------------------------------------------------- plotting

COLORS = {"CX": "C0", "1q": "C1", "qubit": "C2"}


def plot(configs, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    codes = sorted({c["code"] for c in configs}, key=list(CODES).index)
    fig, axes = plt.subplots(1, len(codes), figsize=(6 * len(codes), 4.8), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)

    # Per-parameter RMS vs. signal strength, in the env with gate and qubit parameters
    for ax, code in zip(axes[0], codes):
        full = next(c for c in configs if c["code"] == code and c["params_per_qubit"] > 0)
        rms, kind, boundary = np.array(full["tracking_rms"]), np.array(full["type"]), np.array(full["boundary"])
        x = np.array(full["noisy_ops"], dtype=float) * np.exp(rng.uniform(-0.06, 0.06, size=len(rms)))  # jitter
        for t in TYPES:
            for b, marker in ((False, "o"), (True, "^")):
                sel = (kind == t) & (boundary == b)
                ax.scatter(x[sel], rms[sel], marker=marker, color=COLORS[t], alpha=0.8)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 40])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Noisy operations controlled")
        rho = spearmanr(full["noisy_ops"], rms).statistic
        ax.set_title(f"{code.capitalize()} code (Spearman rho = {rho:.2f})")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Tracking RMS |mu - p_opt|")

    handles = [Line2D([], [], ls="", marker="o", color=COLORS[t], label=f"{t} parameter") for t in TYPES]
    handles += [Line2D([], [], ls="", marker="o", color="gray", label="bulk"),
                Line2D([], [], ls="", marker="^", color="gray", label="boundary")]
    axes[0, -1].legend(handles=handles, loc="upper right")
    fig.suptitle("Per-parameter tracking error vs. signal strength")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Figure saved to '{path}'")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codes", nargs="+", choices=list(CODES), default=list(CODES))
    p.add_argument("--distance", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--drift", choices=["sinusoidal", "arma"], default="sinusoidal",
                   help="sinusoidal moves every parameter in phase; arma drifts them independently")
    p.add_argument("--drift-frequency", type=float, default=1.0 / 150.0, help="sinusoidal drift frequency per epoch")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--total-shots", type=int, default=245_000)
    p.add_argument("--shots-per-candidate", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--eval-shots", type=int, default=400)
    p.add_argument("--out", type=str, default="figures/per_param/per_param_tracking.json")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    configs = run_configs(args)
    for cfg in configs:
        summarize(cfg)

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "runs": configs}, f, indent=2)
    print(f"\nResults saved to '{args.out}' ({time.time() - t0:.0f}s)")
    plot(configs, args.out.replace(".json", ".png"))


if __name__ == "__main__":
    main()
