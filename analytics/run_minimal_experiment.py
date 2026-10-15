"""
Minimal Single-Parameter Check of the Steering Analytics
========================================================
One drifting CX coupler i of a d=3 surface code, read through the mean EDR of its
detectors M_i, steered by the PEPG agent reduced to plain antithetic SGD (fixed probe
width sigma_p, no Adam, no clipping). Three separate experiments:

  tracking:  closed-loop c = E[(mu - p*)^2] across tau vs the Cramer-Rao bound c*, the
             factor k = c/c* - 1, and tau_lim for a target R_t. The gain is tuned per tau
             on the Gaussian model (fast) and then run on the chosen mode.
  variance:  measured E[dmu], Var[dmu] of one update vs the closed form (drift off / on),
             and the S that minimizes Var[dmu] at a fixed physical tau.
  neighbor:  Var[dmu_i] when a neighbor coupler l (sharing detectors with M_i) sits at an
             offset ebar_l and is probed too, i.e. L_nbr = 4 C_il^2 <ebar_l^2> sigma_p^2 > 0.

The controller gain g = 2 lr H_l sigma_p^2 is the per-update contraction of E[dmu].

The bound uses the information of the readings actually taken: a reading at the probed
setting mu +- delta carries kappa (e^2 + sigma_p^2), not kappa e^2.

Examples:
    python run_minimal_experiment.py --experiment tracking --plot
    python run_minimal_experiment.py --experiment variance --plot
    python run_minimal_experiment.py --experiment neighbor --plot
    python run_minimal_experiment.py --experiment tracking --mode gaussian
"""

import argparse
import time
from typing import Dict, List, Optional

import numpy as np

from analytics.minimal_control_env import MinimalControlEnv, cramer_rao_c, tau_lim, update_stats_theory, v_pair
from rl_agent import RLControlAgent


def make_controller(env: MinimalControlEnv, gain: float, sigma_p: float, seed: int):
    """RLControlAgent as plain antithetic SGD: dmu = (lr / B) sum_k adv_k delta_k. Only coupler i
    learns from the M_i reading; a neighbor (if modeled) is probed but its mean stays frozen."""
    lr = gain / (2.0 * env.H_l * sigma_p ** 2)
    mask = np.zeros((1, env.n))
    mask[0, 0] = 1.0
    agent = RLControlAgent(
        num_params=env.n, num_detectors=1, mask_matrix=mask,
        lr_mu=lr * sigma_p ** 2,  # the agent divides by sigma^2
        lr_sigma=0.0, entropy_reg=0.0, sigma_init=sigma_p, sigma_min=sigma_p, sigma_max=sigma_p,
        grad_clip=0.0, use_adam=False, seed=seed,
    )
    return agent, lr


def run_update(env: MinimalControlEnv, agent: RLControlAgent, S: int, B: int) -> np.ndarray:
    """Measures B candidates in consecutive slots (drift moves between slots), then updates.
    Returns mu - p* per slot and coupler, shape (B, n) (mu before the update)."""
    cands = agent.sample_policies(B, antithetic=True)  # partners k and k + B/2
    obs = np.empty((B, 1), dtype=np.float32)
    e_mu = np.empty((B, env.n))
    for k in range(B):
        e_mu[k] = agent.mu - env.p_star
        obs[k, 0] = env.read(cands[k] - env.p_star, S)
        env.step()
    agent.update(cands, obs)
    return e_mu


# ------------------------------------------------------------- update statistics
def expected_ebar(ebar0: float, s: float, a: float, B: int):
    """<ebar> and <ebar^2> when mu = p*(0) + ebar0 is frozen and p* follows the AR(1)."""
    k = np.arange(B // 2)
    m = k + B // 2
    var = s ** 2 * (1.5 + 0.5 * a ** (B // 2) - a ** k - a ** m) if a < 1 else np.zeros(B // 2)
    return ebar0, ebar0 ** 2 + float(np.mean(var))


def update_statistics(env, S, B, gain, sigma_p, ebar0, reps, tau, seed,
                      ebar_nbr: float = 0.0, Sigma_1: Optional[float] = None) -> Dict[str, float]:
    """Repeats one update from mu = p* + (ebar0, ebar_nbr) and compares dmu_i with the closed form."""
    env.set_drift(tau, env.s)
    agent, lr = make_controller(env, gain, sigma_p, seed)
    offsets = np.array([ebar0, ebar_nbr])[:env.n]
    dmu, cond_mean, eb, eb2, ebl2 = (np.zeros(reps) for _ in range(5))
    for r in range(reps):
        env.reset_drift()
        agent.mu[:] = env.p_star + offsets
        mu0 = float(agent.mu[0])
        pair = (lambda e: 0.5 * (e[:B // 2] + e[B // 2:]))(run_update(env, agent, S, B))
        dmu[r] = float(agent.mu[0]) - mu0
        eb[r], eb2[r] = pair[:, 0].mean(), np.mean(pair[:, 0] ** 2)
        if env.n > 1:
            ebl2[r] = np.mean(pair[:, 1] ** 2)
        cond_mean[r] = -2.0 * lr * env.H_l * sigma_p ** 2 * eb[r]
    common = dict(lr=lr, sigma_p=sigma_p, B=B, H_l=env.H_l, ebar_mean=eb.mean(), ebar_sq=eb2.mean(),
                  Sigma_1=env.Sigma_1 if Sigma_1 is None else Sigma_1, S=S, V_pair=v_pair(env.s, B, tau))
    th = update_stats_theory(**common, C_il=env.C_il, ebar_l_sq=ebl2.mean())
    th0 = update_stats_theory(**common)
    # Var is conditional on the realized pair offsets: subtract each update's conditional mean
    resid = dmu - cond_mean
    var_emp = float(np.var(resid, ddof=1))
    # Standard error of the sample variance from the fourth moment (dmu is heavy-tailed, not Gaussian)
    m4 = float(np.mean((resid - resid.mean()) ** 4))
    var_se = float(np.sqrt(max(m4 - (reps - 3) / (reps - 1) * var_emp ** 2, 0.0) / reps))
    return {
        "mean_emp": float(dmu.mean()), "mean_se": float(dmu.std(ddof=1) / np.sqrt(reps)), "mean_th": th["mean"],
        "var_emp": var_emp, "var_se": var_se, "var_th": th["var"],
        "var_th_no_nbr": th0["var"], "L": th["L"], "L_nbr": th["L_nbr"],
    }


def var_theory_curve(env, S_grid, B, gain, sigma_p, ebar0, tau_shots):
    """Closed-form Var[dmu] vs S at a fixed physical tau (in shots)."""
    lr = gain / (2.0 * env.H_l * sigma_p ** 2)
    out = []
    for S in S_grid:
        tau = tau_shots / S
        eb, eb2 = expected_ebar(ebar0, env.s, np.exp(-1.0 / tau), B)
        out.append(update_stats_theory(lr, sigma_p, B, env.H_l, eb, eb2, env.Sigma_1, S, v_pair(env.s, B, tau))["var"])
    return np.array(out)


def shot_variance_at(env, offsets, sigma_p, draws=100, shots=4000) -> float:
    """Single-shot variance Sigma_1 averaged over probe settings around `offsets` (stim noise
    grows with the offsets; the Gaussian model's is constant)."""
    if env.mode == "gaussian":
        return env.Sigma_1
    offsets = np.asarray(offsets, dtype=float)
    return float(np.mean([np.var(env.sample_shots(offsets + sigma_p * env.rng.standard_normal(env.n), shots))
                          for _ in range(draws)]))


# ------------------------------------------------------------------ closed loop
def closed_loop(env, tau, S, B, gain, sigma_p, n_slots, seed) -> float:
    """Runs the controller on AR(1) drift; returns c = E[(mu - p*)^2] after a burn-in."""
    env.set_drift(tau, env.s)
    agent, _ = make_controller(env, gain, sigma_p, seed)
    agent.mu[:] = env.p_star
    burn = int(min(5 * tau, n_slots // 2))
    e2: List[float] = []
    for u in range((burn + n_slots) // B):
        e_mu = run_update(env, agent, S, B)
        if u * B >= burn:
            e2.extend(e_mu[:, 0] ** 2)
    return float(np.mean(e2))


# ------------------------------------------------------------------ experiments
def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(plt, fig, axes, path):
    for ax in np.atleast_1d(axes):
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Plot saved to '{path}'")


def experiment_tracking(env, args):
    S, B, sp = args.shots_per_candidate, args.batch_size, args.sigma_p
    s2, lam = env.s ** 2, env.Lambda(S)
    taus = [float(x) for x in args.taus.split(",")]
    gains = [float(x) for x in args.gains.split(",")]
    print(f"=== Closed loop vs Cramer-Rao with probe information kappa (e^2 + sigma_p^2) "
          f"(S={S}, B={B}, sigma_p={sp}, Lambda={lam:.3f}) ===")
    print(f"  {'tau':>6} | {'gain':>5} | {'c/s^2':>7} | {'c*/s^2':>7} | {'k':>6} | {'c>=c*':>5}")
    rows = []
    for tau in taus:
        n_slots = int(max(30 * tau, 10_000))
        a = np.exp(-1.0 / tau)
        c_star = cramer_rao_c(env.kappa(S), s2 * (1 - a * a), a, sp ** 2)
        # Tune the gain on the Gaussian model with shorter runs, then run the chosen mode
        mode = env.mode
        env.mode = "gaussian"
        tuned = [closed_loop(env, tau, S, B, g, sp, n_slots // 3, args.seed) for g in gains]
        env.mode = mode
        g_best = gains[int(np.argmin(tuned))]
        c = closed_loop(env, tau, S, B, g_best, sp, n_slots, args.seed + 1)
        k = c / c_star - 1.0
        rows.append({"c": c / s2, "k": k})
        print(f"  {tau:>6g} | {g_best:>5.2f} | {c / s2:>7.3f} | {c_star / s2:>7.3f} | {k:>6.2f} | {'yes' if k >= 0 else 'NO':>5}")

    # Steerable limit: where measured c/s^2 crosses R_t vs tau_lim from the measured k
    cs = np.array([r["c"] for r in rows])
    ks = np.array([r["k"] for r in rows])
    lt = np.log(taus)
    v = sp ** 2 / s2
    print(f"\n  Steerable limit for R_t = {args.target}:")
    below = np.flatnonzero(cs <= args.target)
    if len(below) and below[0] > 0:
        j = below[0]
        f = (np.log(cs[j - 1]) - np.log(args.target)) / (np.log(cs[j - 1]) - np.log(cs[j]))
        tau_cross = float(np.exp(lt[j - 1] + f * (lt[j] - lt[j - 1])))
        k_near = float(np.exp(np.interp(np.log(tau_cross), lt, np.log(np.maximum(ks, 1e-3)))))
        print(f"    measured crossing                  tau = {tau_cross:8.1f} slots")
        print(f"    tau_lim with k at crossing ({k_near:.2f})  tau = {tau_lim(lam, args.target, k_near, v):8.1f} slots")
    else:
        print("    measured c/s^2 does not cross R_t inside the tau sweep")
    k_med = float(np.median(ks))
    print(f"    tau_lim with median k ({k_med:.2f})       tau = {tau_lim(lam, args.target, k_med, v):8.1f} slots")
    print(f"    tau_lim at the bound (k=0)          tau = {tau_lim(lam, args.target, 0.0, v):8.1f} slots")

    # Limits of the probe-free Cramer-Rao cubic
    q2 = 1e-4
    print("\n  Cramer-Rao limits at sigma_p = 0 (q^2 = 1e-4):")
    print(f"    Lambda->inf (kappa=1e12, a=0.99):  c* = {cramer_rao_c(1e12, q2, 0.99):.3e}   vs q^2 = {q2:.1e}")
    print(f"    Lambda->0   (kappa=1e-9, a=0.99):  c* = {cramer_rao_c(1e-9, q2, 0.99):.3e}   vs s^2 = {q2 / (1 - 0.99 ** 2):.3e}")
    print(f"    random walk (kappa=10,   a=1):     c* = {cramer_rao_c(10.0, q2, 1.0):.3e}   vs (q^2/kappa)^(1/3) = {(q2 / 10) ** (1 / 3):.3e}")

    if args.plot:
        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        tau_dense = np.geomspace(min(taus) / 2, max(taus) * 2, 200)
        bound = [cramer_rao_c(env.kappa(S), s2 * (1 - np.exp(-2 / t)), np.exp(-1 / t), sp ** 2) / s2 for t in tau_dense]
        ax.plot(tau_dense, bound, color="k", ls="--", label="Cramér–Rao $c^*/s^2$")
        ax.plot(taus, cs, "o-", color="#1f77b4", label=f"controller $c/s^2$ ({args.mode})")
        ax.axhline(args.target, color="#d62728", ls=":", label=f"$R_t$ = {args.target}")
        ax.axvline(tau_lim(lam, args.target, k_med, v), color="#2ca02c", ls="-.", label=f"$\\tau_{{lim}}$ (k={k_med:.1f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("drift correlation time τ (slots)")
        ax.set_ylabel("$E[(\\mu - p^*)^2] / s^2$")
        ax.set_title(f"Tracking vs bound (Λ={lam:.2f}, $\\sigma_p$={sp})")
        _save(plt, fig, ax, "minimal_tracking.png")


def experiment_variance(env, args):
    S, B, sp = args.shots_per_candidate, args.batch_size, args.sigma_p
    print(f"=== Update statistics (B={B}, sigma_p={sp}, gain={args.gain}, ebar0={args.ebar}, {args.reps} updates) ===")
    print(f"  {'case':<24} | {'E[dmu] emp':>16} | {'E[dmu] th':>9} | {'Var emp':>18} | {'Var th':>9}")
    for label, tau in (("drift off", None), (f"drift on, tau={args.tau_shots / S:g} slots", args.tau_shots / S)):
        r = update_statistics(env, S, B, args.gain, sp, args.ebar, args.reps, tau, args.seed)
        print(f"  {label:<24} | {r['mean_emp']:>+9.5f}±{r['mean_se']:.5f} | {r['mean_th']:>+9.5f} | "
              f"{r['var_emp']:.3e}±{r['var_se']:.1e} | {r['var_th']:.3e}")

    S_grid = [int(x) for x in args.S_grid.split(",")]
    print(f"\n  S sweep at fixed physical tau = {args.tau_shots:g} shots")
    print(f"  {'S':>5} | {'Var emp':>18} | {'Var th':>9}")
    var_emp, var_se = [], []
    for Sk in S_grid:
        r = update_statistics(env, Sk, B, args.gain, sp, args.ebar, args.reps, args.tau_shots / Sk, args.seed)
        var_emp.append(r["var_emp"])
        var_se.append(r["var_se"])
        print(f"  {Sk:>5} | {r['var_emp']:.3e}±{r['var_se']:.1e} | {var_theory_curve(env, [Sk], B, args.gain, sp, args.ebar, args.tau_shots)[0]:.3e}")
    S_dense = np.geomspace(min(S_grid) / 2, max(S_grid) * 2, 400)
    var_dense = var_theory_curve(env, S_dense, B, args.gain, sp, args.ebar, args.tau_shots)
    S_star = float(S_dense[np.argmin(var_dense)])
    print(f"  S* (theory) = {S_star:.0f}   S* (empirical, on grid) = {S_grid[int(np.argmin(var_emp))]}")

    if args.plot:
        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.errorbar(S_grid, var_emp, yerr=var_se, fmt="o", color="#1f77b4", label=f"measured ({args.mode})")
        ax.plot(S_dense, var_dense, color="k", ls="--", label="theory")
        ax.axvline(S_star, color="#2ca02c", ls="-.", label=f"$S^*$ = {S_star:.0f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("shots per candidate S")
        ax.set_ylabel("Var[Δμ]")
        ax.set_title(f"Update variance at τ = {args.tau_shots:g} shots")
        _save(plt, fig, ax, "minimal_variance.png")


def experiment_neighbor(env, args):
    S, B, sp = args.nbr_shots, args.batch_size, args.sigma_p
    grid = [float(x) for x in args.ebar_nbr_grid.split(",")]
    print(f"=== Neighbor term L_nbr (coupler {env.neighbor}: C_il={env.C_il:.4f}, shares {env.overlap}/{len(env.M_i)} "
          f"of M_i; S={S}, B={B}, sigma_p={sp}, gain={args.gain}, ebar_i={args.ebar}, {args.reps} updates) ===")
    print("  Theory uses Sigma_1 averaged over the probe settings (stim shot noise grows with the offsets).")
    print(f"  {'case':<10} | {'ebar_l':>6} | {'2L_nbr/(Sig/S)':>14} | {'Var emp':>18} | {'Var th':>9} | {'th, L_nbr=0':>11}")
    res = {}
    for label, tau in (("drift off", None), ("drift on", args.tau_shots / S)):
        res[label] = []
        for el in grid:
            sig = shot_variance_at(env, [args.ebar, el], sp)
            r = update_statistics(env, S, B, args.gain, sp, args.ebar, args.reps, tau, args.seed, ebar_nbr=el, Sigma_1=sig)
            res[label].append(r)
            print(f"  {label:<10} | {el:>6.2f} | {2 * r['L_nbr'] / (sig / S):>14.2f} | "
                  f"{r['var_emp']:.3e}±{r['var_se']:.1e} | {r['var_th']:.3e} | {r['var_th_no_nbr']:>11.3e}")
    print(f"  (drift on: tau = {args.tau_shots:g} shots = {args.tau_shots / S:g} slots, neighbor drifts independently)")

    if args.plot:
        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for (label, rs), color in zip(res.items(), ("#1f77b4", "#d62728")):
            ax.errorbar(grid, [r["var_emp"] for r in rs], yerr=[r["var_se"] for r in rs], fmt="o", color=color,
                        label=f"measured, {label}")
            ax.plot(grid, [r["var_th"] for r in rs], "-", color=color, label=f"theory, {label}")
            ax.plot(grid, [r["var_th_no_nbr"] for r in rs], ":", color=color, label=f"theory without $L_{{nbr}}$, {label}")
        ax.set_xlabel("neighbor offset $\\bar e_l$")
        ax.set_ylabel("Var[Δμ$_i$]")
        ax.set_title(f"Neighbor term (C_il/C_ii = {env.C_il / env.H_l:.2f}, S={S})")
        _save(plt, fig, ax, "minimal_neighbor.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=["tracking", "variance", "neighbor"], default="tracking")
    p.add_argument("--mode", choices=["stim", "gaussian"], default="stim")
    p.add_argument("--distance", type=int, default=3)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--param", type=int, default=0, help="index of the drifting CX coupler")
    p.add_argument("--sensitivity", type=float, default=0.025)
    p.add_argument("--drift-std", type=float, default=0.5, help="stationary drift std s")
    p.add_argument("--shots-per-candidate", type=int, default=50, help="S (one slot)")
    p.add_argument("--batch-size", type=int, default=20, help="B (even)")
    p.add_argument("--sigma-p", type=float, default=0.25, help="probe rms")
    # tracking
    p.add_argument("--taus", type=str, default="30,100,300,1000,3000", help="tau in slots")
    p.add_argument("--gains", type=str, default="0.06,0.12,0.25,0.5,1.0", help="gains tried per tau")
    p.add_argument("--target", type=float, default=0.5, help="R_t: target c/s^2")
    # variance / neighbor
    p.add_argument("--gain", type=float, default=0.2, help="g = 2 lr H_l sigma_p^2")
    p.add_argument("--ebar", type=float, default=0.3, help="frozen offset mu_i - p*_i")
    p.add_argument("--reps", type=int, default=800, help="updates per statistics point")
    p.add_argument("--tau-shots", type=float, default=5000.0, help="physical tau in shots")
    p.add_argument("--S-grid", type=str, default="25,50,100,200,400,800")
    p.add_argument("--neighbor", type=int, default=None, help="neighbor CX coupler (default: most shared detectors)")
    p.add_argument("--nbr-shots", type=int, default=200, help="S for the neighbor experiment")
    p.add_argument("--ebar-nbr-grid", type=str, default="0,0.33,0.67,1.0", help="frozen neighbor offsets")
    p.add_argument("--seed", type=int, default=163)
    p.add_argument("--plot", action="store_true", help="save minimal_<experiment>.png")
    args = p.parse_args()
    if args.batch_size % 2:
        raise ValueError("--batch-size must be even (antithetic pairs)")

    t0 = time.time()
    neighbor = (args.neighbor if args.neighbor is not None else "auto") if args.experiment == "neighbor" else None
    env = MinimalControlEnv(
        mode=args.mode, distance=args.distance, rounds=args.rounds, param_index=args.param, neighbor_index=neighbor,
        sensitivity=args.sensitivity, drift_std=args.drift_std, seed=args.seed,
    )
    c = env.summary()
    S = args.shots_per_candidate
    print(f"=== Calibration (d={args.distance}, rounds={args.rounds}, CX parameter {args.param}, mode={args.mode}) ===")
    print(f"  Omega={c['omega']:.4f}  eps_0={c['eps0']:.4f}  |M_i|={c['M_i']}  N_det={c['N_det']}")
    print(f"  y = {c['y0']:.4f} + H_l e^2:  H_l={c['H_l']:.4f}  (fit residual {c['calib_rel_resid']:.1%} of H_l e_max^2)")
    if env.neighbor is not None:
        print(f"  neighbor coupler {c['neighbor']}: C_il={c['C_il']:.4f}  (C_il/C_ii = {c['C_il'] / c['H_l']:.2f})")
    print(f"  H_g = |M_i|/N_det H_l = {c['H_g']:.4f}   sigma_1={c['sigma_1']:.4f}  Sigma_1={env.Sigma_1:.3e}")
    print(f"  S={S}: kappa={env.kappa(S):.2f}  Lambda=kappa s^4={env.Lambda(S):.3f}  (s={env.s})")
    print(f"  setup {time.time() - t0:.1f}s\n")

    {"tracking": experiment_tracking, "variance": experiment_variance, "neighbor": experiment_neighbor}[args.experiment](env, args)
    print(f"\nTotal {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
