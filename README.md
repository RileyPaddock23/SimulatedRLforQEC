# RL drift tracking for the surface code

An RL agent keeps a rotated surface code's control parameters calibrated while the hardware drifts. It only sees detector error rates, never the logical outcome or the true optimum. The method follows Google Quantum AI, [arXiv:2511.08493](https://arxiv.org/abs/2511.08493).

## Files

| File | Contents |
|---|---|
| `qec_env.py` | `QECCodeEnv`: generic Stim environment for any QEC circuit, with pluggable noise placement, drift (sinusoidal or 1/f^α) and decoder. |
| `surface_code_env.py` | `RotatedSurfaceCodeEnv`: rotated surface code on top of `QECCodeEnv`, decoded with PyMatching. |
| `color_code_env.py` | `ColorCodeEnv`: color code on top of `QECCodeEnv`, decoded with BP+OSD (`stimbposd`). |
| `rl_agent.py` | The PEPG agent (`RLControlAgent`). |
| `run_experiment.py` | Command-line script (`--code surface` or `--code color`) that runs one closed-loop trial and prints Fixed, RL and Optimal side by side. |
| `streaming/` | Single-shot, many-cycle version of the loop with update latency, and its experiments. See [streaming/README.md](streaming/README.md). |
| `hardware_drift/` | Hardware-realistic drift (slow random walk, stale calibration, TLS jumps) and the parameter choices behind it. See [hardware_drift/README.md](hardware_drift/README.md). |
| `minimal_control_env.py` | Single-parameter control model and the analytic bounds. |
| `run_minimal_experiment.py` | Checks the analytic results against the minimal model. |

### Using another code

`QECCodeEnv` takes any noiseless Stim memory circuit with `DETECTOR`s and `OBSERVABLE_INCLUDE`s. Any noise already in the circuit is stripped, because the environment supplies its own.

```python
import stim
from qec_env import QECCodeEnv, NoiseModel, ArmaDrift

env = QECCodeEnv(
    circuit=stim.Circuit.from_file("my_code.stim"),
    decoder="bposd",                       # "pymatching", "bposd", or a callable: noisy circuit -> object with decode_batch
    noise_model=NoiseModel(),              # which gates get drift-controlled noise (defaults are inferred from stim's gate data)
    drift=ArmaDrift(alpha=2.0, max_epochs=100),
)
```

The agent only needs `env.num_params`, `env.num_detectors`, `env.M` and `env.evaluate_policy`, so `RLControlAgent` and `run_trial` work unchanged.

### The model

- **Parameters.** Every CX coupler has control parameters. With `include_single_qubit_noise=True`, so does every qubit (its H, reset and readout). For d=5 that makes 129 parameters and 120 detectors.
- **Error rates.** A mis-set gate parameter adds a coherent error, quadratic in the mis-set, on top of the gate's base depolarizing rate:
  `eps_g = base_g + budget_g * sum_m Omega_gm * (p_gm - p_opt_gm(t))^2`
  Here `budget_g` is a fixed error budget per gate kind: `--error-budget` (0.002) times the kind's SI1000 multiplier, i.e. the reference base rate. Parameters are therefore in **tolerance units**. With the default Omega = 1, a mis-set of 1 doubles the gate's reference error and a mis-set of 0.3 adds about 10%. The budget doesn't move with `base_error`, so lowering the error floor doesn't make a mis-set any cheaper. This is the form used by the two calibration papers this repo draws on (Wu et al., [arXiv:2610.11477](https://arxiv.org/abs/2610.11477), App. D; Magann et al., [arXiv:2512.07815](https://arxiv.org/abs/2512.07815)). `--error-model relative` switches the gate parameters to `eps_g = base_g * (1 + sum_m Omega_gm * (...)^2)`.
- **Stale calibration.** Each gate parameter's optimum starts off the t=0 calibration `p = 0` by a static offset drawn from N(0, 1) (`--initial-offset`, in tolerance units). This is roughly the drift Wu et al. measure for CZ coupling, CZ phase and qubit frequency over a 6 h recalibration interval. Qubit parameters get no offset by default (`--qubit-initial-offset`).
- **Factor graph `M` (detectors × parameters).** `M[j, i] = 1` if an error from parameter *i* can flip detector *j*. It is built from Stim's detector error model.
- **Agent.** The agent keeps a Gaussian policy `N(mu, sigma)` over the parameters and draws antithetic samples from it. Its reward is `-EDR` for each detector. Before each gradient step, `M` masks the advantages, so a parameter only learns from the detectors it can affect.
- **Time.** Time runs on a physical shot clock. Each update costs `batch_size × shots_per_candidate` shots, and the drift keeps moving while the candidates are measured. One reference epoch is 7000 shots.

### Qubit parameters

`--params-per-qubit N` gives every qubit N extra parameters, such as its idle frequency, which sets T1 through nearby TLS defects. A mis-set qubit parameter raises the error rate of **every** source on that qubit: its couplers and its own single-qubit, reset, readout and idle noise. Because it scales an existing mechanism (T1), it is relative to the source's base rate by default, `eps_g = base_g * (1 + Omega e^2)` on top of the gate term, so Omega = 1 with a mis-set of 1 doubles every error on the qubit (`--qubit-error-model additive` switches it). Sensitivities are set per parameter type: `--sensitivity` for the gate parameters of each gate kind (`2q`, `1q`, `reset`, `meas`) and `--qubit-sensitivity` for a qubit parameter on each kind of source it touches (those plus `idle`, `idle_mr`), e.g. `--qubit-sensitivity 2q=1,idle=2.5,idle_mr=1`, and `--sensitivity` is in error budgets per squared tolerance (e.g. `--sensitivity 2q=1,meas=2`). Each parameter draws its own Omega in `[1 - s, 1 + s]` times its type's value (`--sensitivity-spread`). The placement of Stim channels stays the same.

- Qubit parameters come after the gate parameters in the parameter vector (`env.qubit_param_slice`).
- `--params-per-gate 0` leaves only the qubit parameters.
- The `--qubit-*` drift flags fall back to the gate drift flags when they aren't set.
- A qubit parameter's column of `M` is the union of the detectors of all its gates, so its mask is less local than a gate parameter's.

### Drift profiles

Drift sizes are in tolerance units. The default of 2 is the old 0.6 under the old Omega = 12.5. It also matches the ±0.3 MHz telegraph switching of qubit frequencies in Wu et al., but it is well beyond their hour-scale gate-parameter drift (see the note under *Reference numbers*).

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

- **Fixed:** parameters frozen at their t=0 calibration (`mu = 0`), which is already off by the stale-calibration offset.
- **RL:** the agent's current policy mean.
- **Optimal:** the true `p_opt(t)`, which gives the best achievable EDR and LER.
- **Steering advantage** = `(EDR_fixed − EDR_RL) / (EDR_fixed − EDR_opt)`. 1 means RL matches the optimum; 0 means RL does no better than the fixed policy.
- **In-situ steering advantage:** the same ratio with the mean EDR of the probes the agent actually ran in place of `EDR_RL`. This is what a processor steered in place pays, so it counts the cost of exploration.
- **Tracking RMS** = `‖mu − p_opt‖` (RMS over parameters, in tolerance units).
- **Per-parameter diagnostics** (in the JSON under `per_param`), modeled on Wu et al.'s Table I:
  - **Detectability `D_i`:** the drift of `p_opt_i` over one reference epoch, divided by the precision with which one epoch of the agent's own gradient estimates locates `p_opt_i`. It falls toward 0 for a parameter that gives the agent no signal. D grows with the window (about `w^1.5` for smooth drift), so read it as a ranking.
  - **Budget fraction `R_i = Omega_i <e_i^2>`:** the error budgets its tracking error spends on each of its sources.
  - **`R_probe_i = Omega_i <sigma_i^2>`:** what its exploration spends.

### Reference numbers

These come from `--code surface` (about 15 s per run) with all other settings at their defaults, seed 163. They are single runs, so expect ±0.1 variation in steering advantage. The default color code is far slower here: BP+OSD takes 13–20 s per 200 shots at the fixed policy's error rates, so one run takes over 30 minutes.

| Drift | LER fixed → RL (opt) | Steering advantage | In situ | Tracking RMS |
|---|---|---|---|---|
| sinusoidal, f=1/150 | 0.44 → 0.008 (0.006) | 0.98 | 0.89 | 0.42 |
| 1/f², 10% white | 0.45 → 0.31 (0.012) | 0.39 | 0.31 | 1.70 |
| 1/f², 50% white | 0.45 → 0.37 (0.015) | 0.17 | 0.10 | 1.79 |
| 1/f¹, 10% white | 0.41 → 0.40 (0.014) | 0.07 | 0.02 | 1.95 |
| 1/f^0.67, 10% white | 0.40 → 0.41 (0.013) | 0.02 | −0.03 | 1.92 |
| pure white | 0.38 → 0.40 (0.011) | −0.04 | −0.08 | 1.91 |

RL tracks well when most of the drift power is at low frequencies: large α and a small white floor. Power above the agent's update bandwidth cannot be tracked. Pure white noise is worse than leaving the parameters alone, because the agent chases noise.

The fixed LER of about 0.4 means the default drift (2 tolerances on every gate and qubit parameter, on top of the stale offset) is far harsher than hardware drift on the time scale of one run. Wu et al.'s hour-scale gate drift is about 0.6 tolerances per √hour in these units, which is only ~0.05 over a run of tens of seconds. Against that drift the stale offset is what matters, and a drift of 2 only fits fast noise such as TLS or telegraph switching of qubit frequencies. The tables before the tolerance-unit change used Omega = 12.5 (relative) or 0.025 (absolute) with drift 0.6. That is the same physics as Omega = 1 with drift 2.1 and `--initial-offset 0`, but with a different agent.

### Useful knobs

- `--shots-per-candidate` trades policy lag against gradient signal-to-noise. Fewer shots means faster updates but noisier gradients.
- `--drift-frequency` (sinusoid) and `--arma-alpha` / `--arma-white` (ARMA) set how fast the target moves.
- `--distance 3` gives a faster, smaller problem (41 parameters).
- `--params-per-gate` gives each gate several drifting knobs.
- The agent's hyperparameters are `AGENT_DEFAULTS` in `run_experiment.py`, also in tolerance units. A probe of width σ adds `Omega σ²` to every source the parameter touches, so σ is capped at 0.5 (+25%) where the old settings explored at about 0.9 (+80%). In a sweep on the surface code, this cap raised the in-situ advantage from 0.79 to 0.90 (sinusoid) and from 0.30 to 0.36 (1/f²). Narrower caps (0.35) or a smaller `entropy_reg` spend less on probes but lose enough tracking signal that the 1/f² result gets worse. With the default `entropy_reg`, σ sits at the cap.

### Streaming and hardware-realistic drift

Two folders build on the code above without changing it. [`streaming/`](streaming/README.md) runs the agent inside one long memory experiment, with updates landing some latency after each batch. [`hardware_drift/`](hardware_drift/README.md) defines slower, hardware-scale drift and documents the values chosen.

### Minimal single-parameter model

`minimal_control_env.py` reduces the problem to one drifting CX coupler of a d=3 code, with every other parameter at its optimum. The reading is the mean EDR over the coupler's detectors, `y = y0 + H_l e^2 + σ n`, and the drift is AR(1) per slot. In `stim` mode (the default) the readings come from the real circuit. In `gaussian` mode they come from the idealized model, using the constants calibrated from stim. The file also holds the closed-form results: the Cramér–Rao bound `c*`, `tau_lim`, and the mean and variance of one update.

```bash
python run_minimal_experiment.py --experiment tracking --plot   # c vs the Cramér–Rao bound and tau_lim (~1 min)
python run_minimal_experiment.py --experiment variance --plot   # E[Δμ], Var[Δμ] and S* (~1 min)
python run_minimal_experiment.py --experiment neighbor --plot   # Var[Δμ_i] with L_nbr > 0 (~1 min)
python run_minimal_experiment.py --experiment tracking --mode gaussian   # idealized reading model
```

The agent is `RLControlAgent` reduced to plain antithetic SGD. Each reading is taken at the probed setting `μ ± δ`, so it carries information `κ(e² + σ_p²)`, and the bound `c*` on `E[(μ − p*)²]` uses that. The `neighbor` experiment adds the coupler that shares the most detectors with `M_i`. That coupler drifts independently and is probed but not steered, and its `C_il` is calibrated like `H_l`.

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
