"""
Kalman Trackers vs PEPG on the Full Surface Code
================================================
Closed-loop steering of every control parameter of a distance-d rotated surface code
(129 parameters at d=5) under 1/f^alpha + white-floor drift, comparing four controllers
that see the same detector error rates on the same shot clock as run_experiment.py:

  AR(1)        KalmanTracker with one AR(1) per parameter, tau tuned on a training drift
  Multi-AR(1)  KalmanTracker with a sum of AR(1)s fitted to the drift autocovariance
  Exact        KalmanTracker with the true drift model (PowerLawModel)
  PEPG         RLControlAgent as plain antithetic SGD (fixed probe width), gain tuned on
               the training drift

All four draw B antithetic candidates mu +- delta (same probe width, same seed) per epoch
and hold mu for the epoch. The filters are per parameter: the reading of parameter i is
the mean EDR over its detectors M_i (as the agent's masked advantage), and crosstalk from
the other probed parameters enters through the calibrated curvature matrix C.

The drift uses arma_normalize="analytic", so the Exact filter's model is exact for each
parameter's marginal drift; the 8% chip-wide common mode is ignored by every filter.

Example:
    python run_filter_comparison.py --plot
"""

import argparse
import json
import os
import time
from typing import Dict

import numpy as np

from kalman_tracker import (KalmanTracker, OUSumModel, PowerLawModel, fit_multi_ar1, interpolated_autocov,
                            power_law_autocov)
from rl_agent import RLControlAgent
from surface_code_env import RotatedSurfaceCodeEnv

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")

SHOTS_PER_DRIFT_UPDATE = 7000.0  # as run_experiment.py
METHODS = ["Exact", "PEPG", "Multi-AR(1)", "AR(1)"]
COLORS = {"Exact": "#2a78d6", "PEPG": "#eb6834", "Multi-AR(1)": "#1baf7a", "AR(1)": "#eda100"}
GAINS = [0.06, 0.12, 0.25, 0.5, 1.0, 2.0]
MULTI_TAUS = np.geomspace(0.25, 256.0, 6)  # drift updates


# ------------------------------------------------------------------ calibration
def calibrate(env: RotatedSurfaceCodeEnv, shots: int, e_cal: float) -> Dict[str, np.ndarray]:
    """Reading model of each parameter i (mean EDR over M_i): y0_i at zero offsets, single-shot
    std sigma_1_i, and curvatures C_ik = d^2 y_i / d e_k^2 / 2 from +-e_cal offsets of parameter k."""
    P = env.num_params
    p0 = env.get_drift_opt(0.0)
    proj = env.M / np.maximum(env.M.sum(axis=0, keepdims=True), 1.0)  # (O, P)

    def detectors(offsets, n):
        return env.build_noisy_circuit(p0 + offsets, 0.0).compile_detector_sampler().sample(n).astype(np.float32)

    det0 = detectors(np.zeros(P), 4 * shots)
    y_shot = det0 @ proj
    E0 = det0.mean(axis=0)
    D = np.zeros((env.num_detectors, P))
    for k in range(P):
        e = np.zeros(P)
        e[k] = e_cal
        Ep, Em = detectors(e, shots).mean(axis=0), detectors(-e, shots).mean(axis=0)
        D[:, k] = np.maximum((Ep + Em - 2.0 * E0) / (2.0 * e_cal ** 2), 0.0) * env.M[:, k]
    return {"y0": y_shot.mean(axis=0), "sigma_1": y_shot.std(axis=0), "C": proj.T @ D}


def load_calibration(env, args) -> Dict[str, np.ndarray]:
    path = os.path.join(RES_DIR, f"filter_calibration_d{args.distance}_r{args.rounds}_seed{args.seed}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    t0 = time.time()
    cal = calibrate(env, args.calib_shots, args.arma_sigma)
    os.makedirs(RES_DIR, exist_ok=True)
    np.savez(path, **cal)
    print(f"Calibration saved to '{path}' ({time.time() - t0:.0f}s)")
    return cal


# ------------------------------------------------------------------ controllers
class PEPGController:
    """RLControlAgent as plain antithetic SGD in rescaled coordinates q = p / scale (as the
    control-space rescaling of arXiv:2511.08493, Fig. S3): an isotropic policy with probe rms
    sigma_p and one learning rate in q, so parameter i probes with sigma_p * scale_i and its
    gain is g H_i scale_i^2 / H. With scale = 1 this is the plain agent, gain g at curvature H."""

    def __init__(self, env, H: float, gain: float, sigma_p: float, seed: int, scale=1.0):
        lr = gain / (2.0 * H * sigma_p ** 2)
        self.scale = np.broadcast_to(np.asarray(scale, dtype=np.float32), (env.num_params,)).copy()
        self.agent = RLControlAgent(
            num_params=env.num_params, num_detectors=env.num_detectors, mask_matrix=env.M,
            lr_mu=lr * sigma_p ** 2, lr_sigma=0.0, entropy_reg=0.0, sigma_init=sigma_p,
            sigma_min=sigma_p, sigma_max=sigma_p, grad_clip=0.0, use_adam=False, seed=seed,
        )

    @property
    def mu(self):
        return self.agent.mu * self.scale

    def sample_policies(self, batch_size, antithetic=True):
        self._q = self.agent.sample_policies(batch_size, antithetic=antithetic)
        return self._q * self.scale

    def update(self, actions, observations, times, t_next=None):
        self.agent.update(self._q, observations)
        return None


def control_scale(H: np.ndarray, num_cx: int, mode: str) -> np.ndarray:
    """Per-parameter scale s_i with H_i s_i^2 equalized to the median H:
    "param" per parameter, "type" per parameter type (CX couplers vs single-qubit), as Fig. S3
    of arXiv:2511.08493 (one sensitivity per type), "none" s = 1."""
    if mode == "none":
        return np.ones_like(H)
    H_ref = np.median(H)
    if mode == "type":
        is_cx = np.arange(len(H)) < num_cx
        H = np.where(is_cx, H[is_cx].mean(), H[~is_cx].mean())
    return np.sqrt(H_ref / H)


def make_kf(env, cal, model, args) -> KalmanTracker:
    return KalmanTracker(
        num_params=env.num_params, num_detectors=env.num_detectors, mask_matrix=env.M, drift_model=model,
        y0=cal["y0"], C=cal["C"], sigma_1=cal["sigma_1"], shots=args.shots_per_candidate,
        sigma_probe=args.sigma_p * args.scale, rate_scaled_noise=True, seed=args.seed,
    )


# ------------------------------------------------------------------ closed loop
def run_trial(env, ctrl, total_shots: int, args, evaluate: bool = True) -> Dict:
    """Shot-clock loop of run_experiment.py. Tracking error is recorded at every candidate after
    the first third of the shots; EDR / LER of mu at checkpoints every eval_interval shots."""
    S, B = args.shots_per_candidate, args.batch_size
    update_shots = S * B
    burn = total_shots // 3
    e2, ckpt, diag = [], {"shot": [], "edr": [], "ler": []}, {}
    cur_shot, next_eval = 0, burn
    while cur_shot + update_shots <= total_shots:
        if evaluate and cur_shot >= next_eval:
            ev = env.evaluate_policy(np.asarray(ctrl.mu), t=cur_shot / SHOTS_PER_DRIFT_UPDATE, shots=args.eval_shots,
                                     decode=True)
            ckpt["shot"].append(cur_shot)
            ckpt["edr"].append(ev["mean_edr"])
            ckpt["ler"].append(ev["logical_error_rate"])
            next_eval += args.eval_interval
        cands = ctrl.sample_policies(B, antithetic=True)
        times = (cur_shot + (np.arange(B) + 0.5) * S) / SHOTS_PER_DRIFT_UPDATE
        obs = np.stack([env.evaluate_policy(cands[k], t=times[k], shots=S)["edr_vector"] for k in range(B)])
        p_opt = np.stack([env.get_drift_opt(t) for t in times])
        err = np.asarray(ctrl.mu)[None, :] - p_opt
        d = ctrl.update(cands, obs, times, t_next=(cur_shot + 1.5 * update_shots) / SHOTS_PER_DRIFT_UPDATE)
        if cur_shot >= burn:
            e2.append(err ** 2)
            if d is not None:
                late = d.pop("late")
                d["err_pred"] = (d["p_pred"] - p_opt[late]) ** 2
                for key, val in d.items():
                    diag.setdefault(key, []).append(val)
        cur_shot += update_shots

    e2 = np.concatenate(e2)
    s2 = env.arma_sigma ** 2
    res = {"c_s2": float(e2.mean() / s2), "c_s2_se": float(e2.mean(axis=0).std(ddof=1) / np.sqrt(e2.shape[1]) / s2),
           "tracking_rms": float(np.sqrt(e2.mean()))}
    if diag:
        d = {k: np.concatenate(v) for k, v in diag.items()}
        res.update({"nees": float(d["err_pred"].mean() / d["v_pred"].mean()), "nis": float(np.mean(d["nu"] ** 2))})
    if evaluate:
        res.update({"edr": float(np.mean(ckpt["edr"])), "ler": float(np.mean(ckpt["ler"])), "checkpoints": ckpt})
    return res


def reference_policies(env, total_shots: int, args) -> Dict:
    """EDR / LER of the fixed (mu = 0) and optimal (mu = p_opt) policies at the same checkpoints."""
    out = {}
    for name in ("Fixed", "Optimal"):
        edr, ler = [], []
        for shot in range(total_shots // 3, total_shots - args.shots_per_candidate * args.batch_size + 1,
                          args.eval_interval):
            t = shot / SHOTS_PER_DRIFT_UPDATE
            params = np.zeros(env.num_params) if name == "Fixed" else env.get_drift_opt(t)
            ev = env.evaluate_policy(params, t=t, shots=args.eval_shots, decode=True)
            edr.append(ev["mean_edr"])
            ler.append(ev["logical_error_rate"])
        out[name] = {"edr": float(np.mean(edr)), "ler": float(np.mean(ler))}
    return out


# ------------------------------------------------------------------ main
def plot_results(results: Dict, refs: Dict, title: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    names = METHODS
    y = np.arange(len(names))
    vals = [results[n]["c_s2"] for n in names]
    axes[0].barh(y, vals, xerr=[results[n]["c_s2_se"] for n in names], color=[COLORS[n] for n in names],
                 height=0.6, error_kw={"ecolor": "#555555", "lw": 1})
    for yi, v, n in zip(y, vals, names):
        axes[0].text(v + results[n]["c_s2_se"], yi, f"  {v:.3f}", va="center", ha="left", color="#333333", fontsize=9)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("tracking error $E[(\\mu - p^*)^2] / \\sigma^2$")
    axes[0].set_title("Tracking error (lower is better)")

    names2 = ["Fixed"] + METHODS + ["Optimal"]
    lers = [refs[n]["ler"] if n in refs else results[n]["ler"] for n in names2]
    cols = ["#9a9a95" if n in refs else COLORS[n] for n in names2]
    y2 = np.arange(len(names2))
    axes[1].barh(y2, lers, color=cols, height=0.6)
    for yi, v in zip(y2, lers):
        axes[1].text(v, yi, f" {v:.4f}", va="center", ha="left", color="#333333", fontsize=9)
    axes[1].set_yticks(y2, names2)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("logical error rate (PyMatching)")
    axes[1].set_title("Logical error rate (lower is better)")
    for ax in axes:
        ax.grid(True, axis="x", color="#dddddd", lw=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Plot saved to '{path}'")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distance", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--arma-alpha", type=float, default=0.67)
    p.add_argument("--arma-white", type=float, default=0.1)
    p.add_argument("--arma-sigma", type=float, default=0.6)
    p.add_argument("--arma-common", type=float, default=0.08)
    p.add_argument("--total-shots", type=int, default=490_000)
    p.add_argument("--tune-shots", type=int, default=150_000, help="shot budget of each tuning run")
    p.add_argument("--shots-per-candidate", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--sigma-p", type=float, default=0.25, help="probe rms (all controllers)")
    p.add_argument("--rescale", choices=["none", "type", "param"], default="none",
                   help="control-space rescaling (probe widths of all controllers, PEPG coordinates): per parameter "
                        "type as arXiv:2511.08493 Fig. S3, per parameter, or none")
    p.add_argument("--eval-interval", type=int, default=10_000)
    p.add_argument("--eval-shots", type=int, default=2000)
    p.add_argument("--calib-shots", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=163)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", type=str, default=os.path.join(RES_DIR, "results_filters.json"))
    args = p.parse_args()

    t0 = time.time()
    n_drift_updates = int(np.ceil(max(args.total_shots, args.tune_shots) / SHOTS_PER_DRIFT_UPDATE)) + 2
    env = RotatedSurfaceCodeEnv(
        distance=args.distance, rounds=args.rounds, drift_profile="arma", arma_alpha=args.arma_alpha,
        arma_white_fraction=args.arma_white, arma_sigma=args.arma_sigma, arma_common_fraction=args.arma_common,
        arma_normalize="analytic", max_drift_updates=n_drift_updates, seed=args.seed,
    )
    cal = load_calibration(env, args)
    H = np.diag(cal["C"])
    args.scale = control_scale(H, env.num_cx_gates, args.rescale)
    print(f"d={args.distance} rounds={args.rounds} | {env.num_params} params, {env.num_detectors} detectors | "
          f"setup {time.time() - t0:.0f}s")
    print(f"Calibration: H_i median {np.median(H):.4f} (range {H.min():.4f}..{H.max():.4f}), "
          f"sigma_1 median {np.median(cal['sigma_1']):.4f}, off-diagonal |C| / diagonal = "
          f"{np.abs(cal['C'] - np.diag(H)).sum(axis=1).mean() / H.mean():.2f}")
    drift_desc = f"1/f^{args.arma_alpha} + {args.arma_white:.0%} white, sigma={args.arma_sigma}, common {args.arma_common:.0%}"
    print(f"Drift: {drift_desc} (analytic normalization)")
    print(f"Rescaling: {args.rescale}  (probe rms sigma_p * s_i, s_i in {args.scale.min():.2f}..{args.scale.max():.2f})\n")

    # Drift models (time in drift updates; candidate k of a drift update sits at (k + 0.5) / E drift updates)
    E = SHOTS_PER_DRIFT_UPDATE / args.shots_per_candidate
    sig = args.arma_sigma
    cov = lambda lags: interpolated_autocov(power_law_autocov(args.arma_alpha, args.arma_white, sig), E, lags, phase=0.5)
    var0 = float(cov(np.array([0.0]))[0])
    m_taus, m_vars = fit_multi_ar1(cov, E, MULTI_TAUS, 2 * 256 * E)
    print(f"Multi-AR(1) fit: tau (drift updates) {np.round(m_taus, 3).tolist()}, variance {np.round(m_vars, 4).tolist()}")

    # Tuning on a separate drift draw
    env.resample_drift(args.seed + 1000)
    print(f"Tuning on a training drift ({args.tune_shots:,} shots per run)")
    tau_c = {}
    for tau in np.geomspace(0.02, 256.0, 8):
        tau_c[tau] = run_trial(env, make_kf(env, cal, OUSumModel([tau], [var0]), args), args.tune_shots, args, False)["c_s2"]
        print(f"  AR(1) tau {tau:8.3f} drift updates ({tau * E:9.1f} candidates): c/s^2 {tau_c[tau]:.4f}")
    best = min(tau_c, key=tau_c.get)
    for tau in (best / 1.8, best * 1.8):
        tau_c[tau] = run_trial(env, make_kf(env, cal, OUSumModel([tau], [var0]), args), args.tune_shots, args, False)["c_s2"]
        print(f"  AR(1) tau {tau:8.3f} drift updates ({tau * E:9.1f} candidates): c/s^2 {tau_c[tau]:.4f}")
    tau_ar1 = min(tau_c, key=tau_c.get)
    gain_c = {}
    for g in GAINS:
        gain_c[g] = run_trial(env, PEPGController(env, np.median(H), g, args.sigma_p, args.seed, args.scale), args.tune_shots, args,
                              False)["c_s2"]
        print(f"  PEPG gain {g:5.3f}: c/s^2 {gain_c[g]:.4f}")
    best = min(gain_c, key=gain_c.get)
    for g in (best / 1.4, best * 1.4):
        gain_c[g] = run_trial(env, PEPGController(env, np.median(H), g, args.sigma_p, args.seed, args.scale), args.tune_shots, args,
                              False)["c_s2"]
        print(f"  PEPG gain {g:5.3f}: c/s^2 {gain_c[g]:.4f}")
    gain = min(gain_c, key=gain_c.get)
    print(f"  -> AR(1) tau = {tau_ar1:.3f} drift updates, PEPG gain = {gain}  ({time.time() - t0:.0f}s)\n")

    # Evaluation drift
    env.resample_drift(args.seed + 1)
    ctrls = {
        "Exact": lambda: make_kf(env, cal, PowerLawModel(args.arma_alpha, args.arma_white, sig), args),
        "PEPG": lambda: PEPGController(env, np.median(H), gain, args.sigma_p, args.seed, args.scale),
        "Multi-AR(1)": lambda: make_kf(env, cal, OUSumModel(m_taus, m_vars), args),
        "AR(1)": lambda: make_kf(env, cal, OUSumModel([tau_ar1], [var0]), args),
    }
    results = {}
    for name, mk in ctrls.items():
        t1 = time.time()
        results[name] = run_trial(env, mk(), args.total_shots, args)
        print(f"  {name:<12} done ({time.time() - t1:.0f}s)")
    refs = reference_policies(env, args.total_shots, args)

    c_ex = results["Exact"]["c_s2"]
    print(f"\nSteady state (after the first third of {args.total_shots:,} shots), {time.time() - t0:.0f}s total")
    print(f"  {'policy':<12} | {'c/s^2':>15} | {'c/c_exact':>9} | {'RMS':>5} | {'NEES':>5} | {'NIS':>5} | {'EDR':>7} | {'LER':>7}")
    for name in ["Fixed"] + METHODS + ["Optimal"]:
        if name in refs:
            r = refs[name]
            print(f"  {name:<12} | {'':>15} | {'':>9} | {'':>5} | {'':>5} | {'':>5} | {r['edr']:.5f} | {r['ler']:.5f}")
            continue
        r = results[name]
        diag = f"{r['nees']:>5.2f} | {r['nis']:>5.2f}" if "nees" in r else f"{'':>5} | {'':>5}"
        print(f"  {name:<12} | {r['c_s2']:.4f} ± {r['c_s2_se']:.4f} | {r['c_s2'] / c_ex:>9.3f} | "
              f"{r['tracking_rms']:.3f} | {diag} | {r['edr']:.5f} | {r['ler']:.5f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": {k: v for k, v in vars(args).items() if k != "scale"}, "scale": args.scale.tolist(), "tau_ar1": tau_ar1, "gain": gain, "tau_scan": {str(k): v for k, v in tau_c.items()},
                   "gain_scan": gain_c, "multi_taus": m_taus.tolist(), "multi_vars": m_vars.tolist(),
                   "results": results, "references": refs}, f, indent=2)
    print(f"Results saved to '{args.out}'")
    if args.plot:
        plot_results(results, refs, f"d={args.distance}: Kalman trackers vs PEPG ({drift_desc}; rescale: {args.rescale})",
                     os.path.join(FIG_DIR, os.path.basename(args.out).replace(".json", ".png")))


if __name__ == "__main__":
    main()
