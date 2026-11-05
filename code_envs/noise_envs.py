from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import stim


def generate_fractional_noise(
    n_steps: int,
    alpha: float,
    sigma: float = 0.6,
    n_taps: int = 256,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generates 1/f^alpha fractional noise with unit variance scaled by sigma.
    For alpha <= 0.01, generates pure white noise.
    """
    if rng is None:
        rng = np.random.default_rng(163)
    if alpha <= 0.01:
        return rng.normal(0.0, sigma, size=n_steps).astype(np.float32)

    d = float(alpha) / 2.0
    b = np.empty(n_taps, dtype=np.float64)
    b[0] = 1.0
    for k in range(1, n_taps):
        b[k] = b[k - 1] * (k - 1 + d) / k
    norm = np.sqrt(np.sum(b ** 2))
    if norm > 1e-12:
        b /= norm

    burn_in = min(n_taps, 500)
    total = n_steps + burn_in
    x = rng.standard_normal(total + len(b))
    y = np.convolve(x, b, mode="full")[len(b) - 1: len(b) - 1 + total]
    traj = y[burn_in:burn_in + n_steps]
    std_val = float(np.std(traj))
    if std_val > 1e-12:
        traj = (traj - float(np.mean(traj))) / std_val
    return (float(sigma) * traj).astype(np.float32)


def generate_power_law_noise(
    n_steps: int,
    alpha: float = 0.67,
    white_fraction: float = 0.1,
    sigma: float = 0.60,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generates 1/f^alpha noise on top of a white noise floor.
    `white_fraction` is the fraction of the total variance carried by the white floor.
    """
    if rng is None:
        rng = np.random.default_rng(163)
    w = float(np.clip(white_fraction, 0.0, 1.0))
    colored = generate_fractional_noise(n_steps, alpha=alpha, sigma=1.0, rng=rng)
    white = rng.normal(0.0, 1.0, size=n_steps).astype(np.float32)
    combined = np.sqrt(1.0 - w) * colored + np.sqrt(w) * white
    std_val = float(np.std(combined))
    if std_val > 1e-12:
        combined = (combined - float(np.mean(combined))) / std_val
    return (float(sigma) * combined).astype(np.float32)


# ---------------------------------------------------------------------- drift models
# A drift model has `build(num_params, params_per_owner, rng)`, returning a function
# t -> p_opt(t) of shape (num_params,). Parameters are laid out owner-major (an owner is
# a gate, a qubit, ...), so parameter m of owner g is index g * params_per_owner + m.

class SinusoidalDrift:
    """p_opt(t) = A * sin(2 pi f_m t), with f_m = f * (1 + 0.5 m) for the m-th parameter of each owner."""

    def __init__(self, frequency: float = 1.0 / 150.0, amplitude: float = 0.6):
        self.frequency = frequency
        self.amplitude = amplitude

    def build(self, num_params: int, params_per_owner: int, rng: np.random.Generator) -> Callable[[float], np.ndarray]:
        def p_opt(t: float) -> np.ndarray:
            out = np.zeros(num_params, dtype=np.float32)
            for m in range(params_per_owner):
                freq_m = self.frequency * (1.0 + 0.5 * m)
                out[m::params_per_owner] = self.amplitude * np.sin(2.0 * np.pi * freq_m * t)
            return out
        return p_opt


class ArmaDrift:
    """
    1/f^alpha colored noise plus a white noise floor, with a shared chip-wide
    common-mode component. Sampled once per epoch and linearly interpolated in between.
    """

    def __init__(
        self,
        alpha: float = 0.67,  # 0 = white, 1 = flicker, 2 = random walk
        white_fraction: float = 0.1,  # fraction of drift variance in the white floor
        sigma: float = 0.60,
        common_fraction: float = 0.08,
        max_epochs: int = 500,
    ):
        self.alpha = float(alpha)
        self.white_fraction = float(np.clip(white_fraction, 0.0, 1.0))
        self.sigma = float(sigma)
        self.common_fraction = float(np.clip(common_fraction, 0.0, 1.0))
        self.max_epochs = max(10, int(max_epochs))

    def build(self, num_params: int, params_per_owner: int, rng: np.random.Generator) -> Callable[[float], np.ndarray]:
        # Each parameter's trajectory = sqrt(c) * shared + sqrt(1 - c) * local
        n = self.max_epochs + 1
        gen = lambda: generate_power_law_noise(
            n, alpha=self.alpha, white_fraction=self.white_fraction, sigma=self.sigma, rng=rng
        )

        c = self.common_fraction
        shared = gen()
        trajectories = np.zeros((num_params, n), dtype=np.float32)
        for pid in range(num_params):
            trajectories[pid] = np.sqrt(c) * shared + np.sqrt(1.0 - c) * gen()

        max_epochs = self.max_epochs

        def p_opt(t: float) -> np.ndarray:
            t_clamped = max(0.0, min(float(t), float(max_epochs)))
            idx = int(t_clamped)
            frac = t_clamped - idx
            if frac > 1e-5 and idx + 1 <= max_epochs:
                return (1.0 - frac) * trajectories[:, idx] + frac * trajectories[:, idx + 1]
            return trajectories[:, idx].copy()
        p_opt.trajectories = trajectories
        return p_opt


def make_drift(profile: str, frequency: float, amplitude: float, alpha: float, white_fraction: float,
               sigma: float, common_fraction: float, max_epochs: int):
    """Builds a drift model from flat settings: "sinusoidal" uses frequency/amplitude, "arma" the rest."""
    if profile == "sinusoidal":
        return SinusoidalDrift(frequency=frequency, amplitude=amplitude)
    if profile == "arma":
        return ArmaDrift(alpha=alpha, white_fraction=white_fraction, sigma=sigma,
                         common_fraction=common_fraction, max_epochs=max_epochs)
    raise ValueError(f"Unknown drift profile '{profile}' (use 'sinusoidal' or 'arma')")


# ---------------------------------------------------------------------- noise model

def _default_gate_sets() -> Tuple[Set[str], Set[str], Set[str]]:
    """Classifies stim gates into (two-qubit unitaries, single-qubit unitaries/resets, single-qubit measurements)."""
    two_qubit, after, before = set(), set(), set()
    for name, g in stim.gate_data().items():
        if name == "MPAD" or name.startswith("HERALDED_") or (g.is_noisy_gate and not g.produces_measurements):
            continue
        if g.is_two_qubit_gate and g.is_unitary:
            two_qubit.add(name)
        elif g.is_single_qubit_gate:
            if g.is_unitary or g.is_reset:
                after.add(name)
            if g.produces_measurements:
                before.add(name)
    return two_qubit, after, before


@dataclass
class NoiseModel:
    """
    Where the drift-controlled depolarizing channels are placed. A gate set left as
    None is inferred from stim's gate metadata, so any circuit is covered by default.
    """
    two_qubit_gates: Optional[Set[str]] = None  # DEPOLARIZE2 after each target pair (default: all 2-qubit unitaries)
    after_gates: Optional[Set[str]] = None  # DEPOLARIZE1 after (default: 1-qubit unitaries and resets)
    before_gates: Optional[Set[str]] = None  # DEPOLARIZE1 before, i.e. readout error (default: 1-qubit measurements)
    include_single_qubit_noise: bool = True
    max_error: float = 0.15  # error rates are clipped to [base_s, max_error]

    def resolved(self) -> Tuple[Set[str], Set[str], Set[str]]:
        two_qubit, after, before = _default_gate_sets()
        return (
            set(self.two_qubit_gates) if self.two_qubit_gates is not None else two_qubit,
            set(self.after_gates) if self.after_gates is not None else after,
            set(self.before_gates) if self.before_gates is not None else before,
        )

