"""
Detector-sensitivity normalization of the control space
=======================================================
Google's rescaling (Fig. S3 of their supplement): perturb every parameter of one type by
N(0, sigma^2), fit the device-wide detection rate as DR = DR0 + (sigma / sigma0)^2, and
rescale each type by its sigma0, so that one unit of the policy's Gaussian raises the
detection rate by the same amount whichever type of parameter it moves.

Here sigma0 is exact rather than fitted. To first order (see the derivation in the
amplitude / sensitivity benchmark notes),

    1 / sigma0_T^2 = d DR / d sigma^2 = sum_{i in T} sum_j dq_j / d e_i^2 / num_detectors

with q_j the detector firing rates read off stim's detector error model at the base error
rates; under the relative error model this is Omega_T times the share of the baseline
detection rate that type T's sources produce.

The agent then works in p' = p / s, with s_i = sigma0_T(i) / sigma0_ref: the reference type
(2q gates by default) keeps its tolerance units, so the agent's defaults (sigma_init etc.)
keep their meaning for it. `NormalizedEnv` wraps any ControlEnv to present p'.
Diagnostics computed in p' units convert back to tolerance units by multiplying by s
(tracking errors); ratios (detectability) and budget fractions Omega e^2 need nothing,
since the wrapper reports Omega' = Omega s^2.
"""

from typing import Dict, Tuple

import numpy as np


def param_types(env) -> np.ndarray:
    """Type of every parameter: its gate kind ("2q", "1q", "reset", "meas") or "qubit"."""
    gate = [src.kind for src in env.blocks[0].owners for _ in range(env.blocks[0].params_per_owner)]
    return np.array(gate + ["qubit"] * env.num_qubit_params)


def type_sigma0(env, h: float = 1e-3) -> Dict[str, float]:
    """sigma0 per parameter type, in tolerance units, from an exact first-order finite difference of DR."""
    from analytics.signal_share import detector_rates  # deferred: signal_share imports per_param_tracking, which imports this
    eps0 = env.param_base_errors.astype(np.float64)
    dr0 = detector_rates(env, eps0).mean()
    p_opt = env.get_drift_opt(0.0)
    kinds = param_types(env)
    out = {}
    for t in dict.fromkeys(kinds):
        p = p_opt.copy()
        p[kinds == t] += h
        slope = (detector_rates(env, env.compute_physical_error_rates(p, 0.0).astype(np.float64)).mean() - dr0) / h ** 2
        out[t] = float(1.0 / np.sqrt(slope))
    return out


def param_scales(env, ref: str = "2q") -> Tuple[np.ndarray, Dict[str, float]]:
    """Per-parameter scale s_i = sigma0_T(i) / sigma0_ref, and the sigma0 of every type."""
    s0 = type_sigma0(env)
    return np.array([s0[t] / s0[ref] for t in param_types(env)], dtype=np.float32), s0


class NormalizedEnv:
    """Presents a ControlEnv in normalized coordinates p' = p / scales; everything else passes through."""

    def __init__(self, env, scales: np.ndarray):
        self.env = env
        self.scales = np.asarray(scales, dtype=np.float32)
        self.param_sensitivities = env.param_sensitivities * self.scales ** 2

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_drift_opt(self, t: float) -> np.ndarray:
        return self.env.get_drift_opt(t) / self.scales

    def compute_physical_error_rates(self, params: np.ndarray, t: float) -> np.ndarray:
        return self.env.compute_physical_error_rates(self.scales * params, t)

    def evaluate_policy(self, params: np.ndarray, t: float, shots: int = 1000, decode: bool = False):
        return self.env.evaluate_policy(self.scales * params, t, shots=shots, decode=decode)


def normalize(env, ref: str = "2q") -> NormalizedEnv:
    scales, _ = param_scales(env, ref)
    return NormalizedEnv(env, scales)
