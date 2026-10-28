import matplotlib.pyplot as plt
import numpy as np
#Need to fix imports now that it's in example folder
from rl_agent import RLControlAgent
from surface_code_env import RotatedSurfaceCodeEnv
SEED = 163


def run_calibration(num_epochs=40):
    env = RotatedSurfaceCodeEnv(
        distance=5, rounds=5, base_error=0.002, sensitivity=0.025,
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


def run_frequency_scan():
    freqs = [1/1000, 1/500, 1/250, 1/150, 1/100, 1/50, 1/25, 1/10]
    ler_red_pct, r_advantage = [], []
    for f in freqs:
        env = RotatedSurfaceCodeEnv(
            distance=5, rounds=5, base_error=0.002, sensitivity=0.025,
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
        for ep in range(30):
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
        mean_lf, mean_lr = np.mean(ep_ler_f), np.mean(ep_ler_r)
        mean_ef, mean_er, mean_eo = np.mean(ep_edr_f), np.mean(ep_edr_r), np.mean(ep_edr_o)
        red = ((mean_lf - mean_lr) / (mean_lf + 1e-6)) * 100.0
        r = (mean_ef - mean_er) / (mean_ef - mean_eo + 1e-6)
        ler_red_pct.append(red)
        r_advantage.append(r)
        print(f"Freq {f:7.4f} | LER Red: {red:+6.1f}% | Steering r: {r:5.2f}")
    return freqs, ler_red_pct, r_advantage


def plot_figure(h_cal, freqs, red_pct, r_vals):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    fig.patch.set_facecolor("#fafafa")
    for ax in axes.flat:
        ax.set_facecolor("#ffffff")
        ax.grid(True, linestyle="--", alpha=0.5)
    # 1. From-Scratch LER
    ax1 = axes[0, 0]
    ax1.plot(h_cal["epochs"], h_cal["ler"], color="#d62728", lw=2.5, marker="o", ms=4, label="Decoded LER (d=5)")
    ax1.axhline(0.10, color="#ff7f0e", ls=":", lw=1.8, label="Surface Code Threshold (~0.10)")
    ax1.axhline(0.002, color="#2ca02c", ls="--", lw=1.8, label="Target Noise Floor")
    ax1.set_title("From-Scratch Calibration: LER Trajectory (d=5)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Logical Error Rate $\epsilon_L$")
    ax1.legend(loc="upper right")
    # 2. EDR & Physical Error
    ax2 = axes[0, 1]
    ax2.plot(h_cal["epochs"], h_cal["edr"], color="#1f77b4", lw=2, label="Error Detection Rate (EDR)")
    ax2.plot(h_cal["epochs"], h_cal["phys_err"], color="#9467bd", lw=2, label="Mean Physical Error $\overline{\epsilon}$")
    ax2.set_title("Defect Density & Physical Infidelity Convergence", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Rate")
    ax2.legend(loc="upper right")
    # 3. LER Suppression vs Frequency
    ax3 = axes[1, 0]
    ax3.semilogx(freqs, red_pct, color="#1f77b4", lw=2.5, marker="s", ms=6, label="RL LER Reduction (%)")
    ax3.axhline(0.0, color="#d62728", ls="--", lw=1.5, label="Zero Benefit Line (0%)")
    ax3.axvline(1 / 130, color="#ff7f0e", ls="-.", lw=1.8, label="Google Paper Limit ($f_c \sim 1/130$)")
    ax3.set_title("Logical Error Suppression vs. Drift Frequency", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Drift Frequency $f$ (cycles / epoch)")
    ax3.set_ylabel("LER Suppression (%)")
    ax3.legend(loc="lower left")
    # 4. Steering Advantage r vs Frequency
    ax4 = axes[1, 1]
    ax4.semilogx(freqs, r_vals, color="#2ca02c", lw=2.5, marker="^", ms=6, label="Steering Advantage $r$")
    ax4.axhline(1.0, color="#7f7f7f", ls=":", lw=1.2, label="Optimal Tracking ($r=1$)")
    ax4.axhline(0.0, color="#d62728", ls="--", lw=1.5, label="Steerability Boundary ($r=0$)")
    ax4.axvline(1 / 130, color="#ff7f0e", ls="-.", lw=1.8, label="Google Limit ($1/130$)")
    ax4.set_title(r"Steering Advantage $r(f) = (E_{\mathrm{fix}} - E_{\mathrm{RL}})/(E_{\mathrm{fix}} - E_{\mathrm{opt}})$", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Drift Frequency $f$ (cycles / epoch)")
    ax4.set_ylabel("Steering Advantage $r$")
    ax4.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("from_scratch_and_steerability.png", dpi=150)
    plt.close()
    print("Saved 'from_scratch_and_steerability.png'")
if __name__ == "__main__":
    h_cal = run_calibration(num_epochs=40)
    freqs, red_pct, r_vals = run_frequency_scan()
    plot_figure(h_cal, freqs, red_pct, r_vals)