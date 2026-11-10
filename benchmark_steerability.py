import sys
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
from rl_agent import RLControlAgent
from code_envs.surface_code_env import RotatedSurfaceCodeEnv
SEED = 163


def run_calibration(num_epochs=40):
    env = RotatedSurfaceCodeEnv(
        distance=5, rounds=5, base_error=0.002, sensitivity=12.5,
        drift_frequency=0.0, drift_amplitude=0.0, seed=SEED,
    )
    rng = np.random.default_rng(SEED)
    p_init = rng.uniform(-2.7, 2.7, size=env.num_params).astype(np.float32)
    agent = RLControlAgent(
        env.num_params, env.num_detectors, env.M,
        lr_mu=0.08, lr_sigma=0.01, lr_baseline=0.15, entropy_reg=0.005,
        sigma_init=0.40, sigma_min=0.08, sigma_max=0.50, use_adam=True, seed=SEED,
    )
    agent.mu = p_init.copy()
    history = {"epochs": [], "ler": [], "edr": [], "phys_err": []}
    for ep in range(num_epochs):
        cands = agent.sample_policies(batch_size=20, antithetic=True)
        obs = np.array([env.evaluate_policy(c, t=0, shots=400, decode=False)["edr_vector"] for c in cands])
        agent.update(cands, obs)
        eval_ep = env.evaluate_policy(agent.mu, t=0, shots=1200, decode=True)
        history["epochs"].append(ep)
        history["ler"].append(eval_ep["logical_error_rate"])
        history["edr"].append(eval_ep["mean_edr"])
        history["phys_err"].append(float(np.mean(env.compute_physical_error_rates(agent.mu, 0))))
        print(f"Calib Ep {ep:2d} | LER: {eval_ep['logical_error_rate']:.4f} | EDR: {eval_ep['mean_edr']:.4f}")
    return history


def scan_frequency(f, num_epochs=30, return_history=False):
    """Steering advantage r and LER reduction (%) at one drift frequency, averaged over num_epochs."""
    env = RotatedSurfaceCodeEnv(
        distance=5, rounds=5, base_error=0.002, sensitivity=12.5,
        drift_frequency=f, drift_amplitude=0.6, seed=SEED,
    )
    agent = RLControlAgent(
        env.num_params, env.num_detectors, env.M,
        lr_mu=0.05, lr_sigma=0.01, lr_baseline=0.15, entropy_reg=0.005,
        sigma_init=0.25, sigma_min=0.08, sigma_max=0.35, use_adam=True, seed=SEED,
    )
    fixed_policy = np.zeros(env.num_params, dtype=np.float32)
    ep_ler_f, ep_ler_r = [], []
    ep_edr_f, ep_edr_r, ep_edr_o = [], [], []
    ep_mu, ep_opt = [], []
    for ep in range(num_epochs):
        cands = agent.sample_policies(batch_size=20, antithetic=True)
        obs = np.array([env.evaluate_policy(c, t=ep, shots=350, decode=False)["edr_vector"] for c in cands])
        agent.update(cands, obs)
        p_opt = env.get_drift_opt(ep)
        ev_f = env.evaluate_policy(fixed_policy, t=ep, shots=800, decode=True)
        ev_r = env.evaluate_policy(agent.mu, t=ep, shots=800, decode=True)
        ev_o = env.evaluate_policy(p_opt, t=ep, shots=800, decode=True)
        ep_ler_f.append(ev_f["logical_error_rate"])
        ep_ler_r.append(ev_r["logical_error_rate"])
        ep_edr_f.append(ev_f["mean_edr"])
        ep_edr_r.append(ev_r["mean_edr"])
        ep_edr_o.append(ev_o["mean_edr"])
        ep_mu.append(agent.mu.copy())
        ep_opt.append(p_opt)
    mean_lf, mean_lr = np.mean(ep_ler_f), np.mean(ep_ler_r)
    mean_ef, mean_er, mean_eo = np.mean(ep_edr_f), np.mean(ep_edr_r), np.mean(ep_edr_o)
    red = ((mean_lf - mean_lr) / (mean_lf + 1e-6)) * 100.0
    r = (mean_ef - mean_er) / (mean_ef - mean_eo + 1e-6)
    print(f"Freq {f:7.4f} | LER Red: {red:+6.1f}% | Steering r: {r:5.2f}", flush=True)
    if return_history:
        return {"mu": np.array(ep_mu), "p_opt": np.array(ep_opt), "edr_fixed": ep_edr_f, "edr_rl": ep_edr_r,
                "edr_opt": ep_edr_o, "ler_fixed": ep_ler_f, "ler_rl": ep_ler_r, "red": red, "r": r}
    return red, r


def run_frequency_scan(num_epochs=30):
    freqs = [1/1000, 1/500, 1/250, 1/150, 1/100, 1/50, 1/25, 1/10]
    with Pool(len(freqs)) as pool:
        results = pool.starmap(scan_frequency, [(f, num_epochs) for f in freqs])
    ler_red_pct, r_advantage = (list(x) for x in zip(*results))
    return freqs, ler_red_pct, r_advantage


def plot_figure(freqs, red_pct, r_vals, num_epochs, path="from_scratch_and_steerability.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    fig.patch.set_facecolor("#fafafa")
    for ax in axes.flat:
        ax.set_facecolor("#ffffff")
        ax.grid(True, linestyle="--", alpha=0.5)
    # 3. LER Suppression vs Frequency
    ax3 = axes[0]
    ax3.semilogx(freqs, red_pct, color="#1f77b4", lw=2.5, marker="s", ms=6, label="RL LER Reduction (%)")
    ax3.axhline(0.0, color="#d62728", ls="--", lw=1.5, label="Zero Benefit Line (0%)")
    ax3.axvline(1 / 130, color="#ff7f0e", ls="-.", lw=1.8, label="Google Paper Limit ($f_c \sim 1/130$)")
    ax3.set_title("Logical Error Suppression vs. Drift Frequency", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Drift Frequency $f$ (cycles / epoch)")
    ax3.set_ylabel("LER Suppression (%)")
    ax3.legend(loc="lower left")
    # 4. Steering Advantage r vs Frequency
    ax4 = axes[1]
    ax4.semilogx(freqs, r_vals, color="#2ca02c", lw=2.5, marker="^", ms=6, label="Steering Advantage $r$")
    ax4.axhline(1.0, color="#7f7f7f", ls=":", lw=1.2, label="Optimal Tracking ($r=1$)")
    ax4.axhline(0.0, color="#d62728", ls="--", lw=1.5, label="Steerability Boundary ($r=0$)")
    ax4.axvline(1 / 130, color="#ff7f0e", ls="-.", lw=1.8, label="Google Limit ($1/130$)")
    ax4.set_title(r"Steering Advantage $r(f) = (E_{\mathrm{fix}} - E_{\mathrm{RL}})/(E_{\mathrm{fix}} - E_{\mathrm{opt}})$", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Drift Frequency $f$ (cycles / epoch)")
    ax4.set_ylabel("Steering Advantage $r$")
    ax4.legend(loc="lower left")
    fig.suptitle(f"Surface code (d=5) steerability, {num_epochs} epochs per frequency", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved '{path}'")
def plot_tracking(f, num_epochs, path):
    """Agent mean mu vs. the drifting optimum, and the resulting EDR, over every epoch at one frequency."""
    h = scan_frequency(f, num_epochs, return_history=True)
    ep = np.arange(num_epochs)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), dpi=150, sharex=True)
    for ax in (ax1, ax2):
        ax.grid(True, linestyle="--", alpha=0.5)
    ax1.fill_between(ep, h["mu"].min(axis=1), h["mu"].max(axis=1), color="#1f77b4", alpha=0.2, label="Agent $\mu$ (range over params)")
    ax1.plot(ep, h["mu"].mean(axis=1), color="#1f77b4", lw=1.8, label="Agent $\mu$ (mean over params)")
    ax1.plot(ep, h["p_opt"].mean(axis=1), color="black", lw=1.5, ls="--", label="Optimum $p_{\mathrm{opt}}(t)$")
    ax1.set_ylabel("Parameter value")
    ax1.set_title("Parameter tracking", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right")
    ax2.plot(ep, h["edr_fixed"], color="#7f7f7f", lw=1, label="Fixed policy")
    ax2.plot(ep, h["edr_rl"], color="#1f77b4", lw=1, label="RL agent")
    ax2.plot(ep, h["edr_opt"], color="#2ca02c", lw=1, label="Optimal policy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean EDR")
    ax2.set_title(f"Error detection rate (steering r = {h['r']:.2f}, LER reduction = {h['red']:+.1f}%)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right")
    fig.suptitle(f"Surface code (d=5) tracking at f = {f:g} cycles/epoch, {num_epochs} epochs", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved '{path}'")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "track":  # python benchmark_steerability.py track <freq> <epochs>
        f, num_epochs = float(sys.argv[2]), int(sys.argv[3])
        plot_tracking(f, num_epochs, f"figures/steerability/tracking_f{f:g}_{num_epochs}ep.png")
        sys.exit()
    num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    freqs, red_pct, r_vals = run_frequency_scan(num_epochs)
    plot_figure(freqs, red_pct, r_vals, num_epochs, f"figures/steerability/steerability_{num_epochs}ep.png")