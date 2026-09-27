# RL drift tracking for the surface code

An RL agent keeps a rotated surface code's control parameters calibrated while the hardware drifts. It only sees detector error rates, never the logical outcome or the true optimum. The method follows Google Quantum AI, [arXiv:2511.08493](https://arxiv.org/abs/2511.08493).

## Files

| File | Contents |
|---|---|
| `surface_code_env.py` | Stim environment for the rotated surface code with sinusoidal or 1/f^α drift. |
| `rl_agent.py` | The PEPG agent (`RLControlAgent`). |
| `run_experiment.py` | Command-line script that runs one closed-loop trial and prints Fixed, RL and Optimal side by side. |

### The model

- **Parameters.** Every CX coupler has control parameters. With `include_single_qubit_noise=True`, so does every qubit (its H, reset and readout). For d=5 that makes 129 parameters and 120 detectors.
- **Error rates.** Each gate's depolarizing rate grows quadratically as its parameters move away from the drifting optimum:
  `eps_g = base_g + sum_m Omega_gm * (p_gm - p_opt_gm(t))^2`
- **Factor graph `M` (detectors × parameters).** `M[j, i] = 1` if an error from parameter *i* can flip detector *j*. It is built from Stim's detector error model.
- **Agent.** The agent keeps a Gaussian policy `N(mu, sigma)` over the parameters and draws antithetic samples from it. Its reward is `-EDR` for each detector. Before each gradient step, `M` masks the advantages, so a parameter only learns from the detectors it can affect.
- **Time.** Time runs on a physical shot clock. Each update costs `batch_size × shots_per_candidate` shots, and the drift keeps moving while the candidates are measured. One reference epoch is 7000 shots.

### Drift profiles

- **`sinusoidal`:** `p_opt(t) = A sin(2π f t)`, where `f` is in cycles per reference epoch (`--drift-frequency`, `--drift-amplitude`).
- **`arma`:** 1/f^α colored noise on top of a white noise floor.
  - `--arma-alpha` sets the power-spectrum exponent: 0 is white, 1 is flicker, 2 is a random walk.
  - `--arma-white` sets the share of the drift variance that comes from the white floor (default 0.1).
  - `--arma-sigma` sets the overall size of the drift.
  - Every trajectory mixes a chip-wide component with a per-parameter component (`--arma-common`, default 8% shared).

## Setup

```bash
pip install -r requirements.txt
```

## Testing the approach

Run the commands from this folder. With default settings (d=5, 245k shots, S=50, B=20, 245 policy updates) each run takes about 10 s.

```bash
# Sanity check: slow sinusoidal drift. RL should close almost the whole gap to optimal.
python run_experiment.py --drift sinusoidal

# 1/f^alpha drift with a 10% white floor
python run_experiment.py --drift arma --arma-alpha 0.67   # pink
python run_experiment.py --drift arma --arma-alpha 2.0    # random walk
python run_experiment.py --drift arma --arma-alpha 2.0 --arma-white 0.5   # heavier white floor
python run_experiment.py --drift arma --arma-alpha 0 --arma-white 0       # pure white, untrackable

# Add --plot to save a PNG (EDR, LER, and parameter 0 vs. its target)
python run_experiment.py --drift arma --arma-alpha 2.0 --plot
```

The script prints a table at each checkpoint, then a steady-state summary averaged over the last third of checkpoints. It also writes `results_<drift>.json`, or the path given by `--out`.

### Reading the output

- **Fixed:** parameters frozen at their t=0 calibration (`mu = 0`).
- **RL:** the agent's current policy mean.
- **Optimal:** the true `p_opt(t)`, which gives the best achievable EDR and LER.
- **Steering advantage** = `(EDR_fixed − EDR_RL) / (EDR_fixed − EDR_opt)`. 1 means RL matches the optimum; 0 means RL does no better than the fixed policy.
- **Tracking RMS** = `‖mu − p_opt‖` (RMS over parameters).

### Reference numbers

These use the default settings with seed 163. They are single runs, so expect ±0.1 variation in steering advantage.

| Drift | LER fixed → RL (opt) | Steering advantage | Tracking RMS |
|---|---|---|---|
| sinusoidal, f=1/150 | 0.040 → 0.0005 (0.0002) | 0.98 | 0.08 |
| 1/f², 10% white | 0.066 → 0.016 (0.001) | 0.59 | 0.36 |
| 1/f², 50% white | 0.047 → 0.029 (0.002) | 0.28 | 0.46 |
| 1/f¹, 10% white | 0.053 → 0.031 (0.001) | 0.19 | 0.51 |
| 1/f^0.67, 10% white | 0.045 → 0.039 (0.001) | ~0 | 0.52 |
| pure white | 0.027 → 0.041 (0.001) | < 0 | 0.53 |

RL tracks well when most of the drift power is at low frequencies: large α and a small white floor. Power above the agent's update bandwidth cannot be tracked. Pure white noise is worse than leaving the parameters alone, because the agent chases noise.

### Useful knobs

- `--shots-per-candidate` trades policy lag against gradient signal-to-noise. Fewer shots means faster updates but noisier gradients.
- `--drift-frequency` (sinusoid) and `--arma-alpha` / `--arma-white` (ARMA) set how fast the target moves.
- `--distance 3` gives a faster, smaller problem (41 parameters).
- `--params-per-gate` gives each gate several drifting knobs.
- The agent's hyperparameters are set in `main()` in `run_experiment.py`.

### Using the pieces directly

```python
import numpy as np
from surface_code_env import RotatedSurfaceCodeEnv
from rl_agent import RLControlAgent

env = RotatedSurfaceCodeEnv(distance=5, drift_profile="arma", arma_alpha=2.0, arma_white_fraction=0.1)
agent = RLControlAgent(env.num_params, env.num_detectors, env.M)

for epoch in range(30):
    cands = agent.sample_policies(batch_size=20)
    obs = [env.evaluate_policy(c, t=epoch, shots=350)["edr_vector"] for c in cands]
    agent.update(cands, np.array(obs))

print(env.evaluate_policy(agent.mu, t=30, shots=2000, decode=True)["logical_error_rate"])
```
