from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.signal as si
import stim


@lru_cache(maxsize=64)
def pink_arma_filter(alpha: float, wl: float = 0.001 * np.pi, wh: float = 0.999 * np.pi) -> Tuple[np.ndarray, int]:
    """
    ARMA filter whose output PSD follows 1/f^alpha between the normalized angular frequencies
    wl and wh (rad/sample), after S. Plaszczynski, Fluctuation and Noise Letters 7, R1 (2007).
    Poles and zeros alternate on a log-spaced grid (~2.5 per decade); below wl the spectrum is
    white, above wh it falls as 1/f^2. Mapped to discrete time with the bilinear transform.

    Returns (sos, burn_in): second-order sections scaled so that unit-variance white input
    gives unit-variance stationary output, and the number of samples to discard so the filter
    state reaches stationarity (10 time constants of the slowest pole).
    """
    if not 0.0 < alpha <= 2.0:
        raise ValueError(f"alpha must be in (0, 2], got {alpha}")
    lw, hw = np.log10(wl), np.log10(wh)
    nf = int(np.ceil(2.5 * (hw - lw)))
    delp = (hw - lw) / nf
    log_p = lw + 0.5 * (1.0 - alpha / 2.0) * delp + np.arange(nf) * delp
    log_z = log_p + alpha / 2.0 * delp
    p, z = 10.0 ** log_p, 10.0 ** log_z
    sos = si.zpk2sos((1.0 - z) / (1.0 + z), (1.0 - p) / (1.0 + p), 1.0)

    # Exact stationary variance from the impulse response (sum of h^2).
    slowest = float(np.max((1.0 - p) / (1.0 + p)))
    tau = 1.0 / (1.0 - slowest)
    impulse = np.zeros(int(40 * tau) + 1)
    impulse[0] = 1.0
    h = si.sosfilt(sos, impulse)
    sos = sos.copy()
    sos[0, :3] /= np.sqrt(np.sum(h ** 2))
    return sos, int(np.ceil(10 * tau))


def generate_fractional_noise(
    n_steps: int,
    alpha: float,
    sigma: float = 0.6,
    rng: Optional[np.random.Generator] = None,
    wl: float = 0.001 * np.pi,
    wh: float = 0.999 * np.pi,
) -> np.ndarray:
    """
    1/f^alpha noise (alpha in (0, 2]) from the Plaszczynski ARMA filter (see pink_arma_filter),
    with stationary std sigma. sigma is the ensemble std: individual trajectories are not
    re-standardized, so a finite window of long-memory noise shows its natural spread in
    sample std (typically below sigma). For alpha <= 0.01, generates pure white noise.
    """
    if rng is None:
        rng = np.random.default_rng(163)
    if alpha <= 0.01:
        return rng.normal(0.0, sigma, size=n_steps).astype(np.float32)
    sos, burn_in = pink_arma_filter(float(alpha), float(wl), float(wh))
    y = si.sosfilt(sos, rng.standard_normal(n_steps + burn_in))[burn_in:]
    return (float(sigma) * y).astype(np.float32)


def generate_power_law_noise(
    n_steps: int,
    alpha: float = 0.67,
    white_fraction: float = 0.1,
    sigma: float = 0.60,
    rng: Optional[np.random.Generator] = None,
    wl: float = 0.001 * np.pi,
    wh: float = 0.999 * np.pi,
) -> np.ndarray:
    """
    Generates 1/f^alpha noise on top of a white noise floor, with stationary (ensemble) std sigma.
    `white_fraction` is the fraction of the total variance carried by the white floor.
    """
    if rng is None:
        rng = np.random.default_rng(163)
    w = float(np.clip(white_fraction, 0.0, 1.0))
    colored = generate_fractional_noise(n_steps, alpha=alpha, sigma=1.0, rng=rng, wl=wl, wh=wh)
    white = rng.normal(0.0, 1.0, size=n_steps).astype(np.float32)
    return (float(sigma) * (np.sqrt(1.0 - w) * colored + np.sqrt(w) * white)).astype(np.float32)


# ---------------------------------------------------------------------- drift models
# A drift model has `build(num_params, params_per_owner, rng)`, returning a function
# t -> p_opt(t) of shape (num_params,). Parameters are laid out owner-major (an owner is
# a gate, a qubit, ...), so parameter m of owner g is index g * params_per_owner + m.

class SinusoidalDrift:
    """
    p_opt(t) = A * sin(2 pi f_m t + phi), with f_m = f * (1 + 0.5 m) for the m-th parameter of
    each owner. phi = 0 for every parameter (all in phase, the default), or with random_phase
    drawn U(0, 2 pi) per parameter, so the parameters don't move together and every stretch
    of time sees the same mix of drift speeds.
    """

    def __init__(self, frequency: float = 1.0 / 150.0, amplitude: float = 2.0, random_phase: bool = False):
        self.frequency = frequency
        self.amplitude = amplitude
        self.random_phase = random_phase

    def build(self, num_params: int, params_per_owner: int, rng: np.random.Generator) -> Callable[[float], np.ndarray]:
        phase = rng.uniform(0.0, 2.0 * np.pi, size=num_params) if self.random_phase else np.zeros(num_params)
        freq = self.frequency * (1.0 + 0.5 * (np.arange(num_params) % max(1, params_per_owner)))

        def p_opt(t: float) -> np.ndarray:
            return (self.amplitude * np.sin(2.0 * np.pi * freq * t + phase)).astype(np.float32)
        return p_opt


class ArmaDrift:
    """
    1/f^alpha colored noise plus a white noise floor, with a shared chip-wide
    common-mode component. Sampled once per epoch and linearly interpolated in between.
    """

    def __init__(
        self,
        alpha: float = 0.67,  # 0 = white, 1 = flicker, 2 = random walk (max 2)
        white_fraction: float = 0.1,  # fraction of drift variance in the white floor
        sigma: float = 2.0,
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


class Ar1Drift:
    """
    Independent AR(1) (Ornstein-Uhlenbeck) drift per parameter, each with its own
    correlation time tau_i (epochs) and stationary std sigma_i, both drawn log-uniformly
    from the given ranges:

        x(t + dt) = phi_i x(t) + sigma_i sqrt(1 - phi_i^2) n,   phi_i = exp(-dt / tau_i)

    Each trajectory starts from its stationary distribution. It is sampled every
    dt = min(tau) / steps_per_tau epochs and linearly interpolated in between. The drawn
    taus and sigmas are attached to the returned function as `p_opt.taus`, `p_opt.sigmas`.
    """

    def __init__(self, tau_range: Tuple[float, float] = (1.0, 50.0), sigma_range: Tuple[float, float] = (0.3, 1.5),
                 max_epochs: int = 500, steps_per_tau: int = 20):
        self.tau_range = tuple(float(v) for v in tau_range)
        self.sigma_range = tuple(float(v) for v in sigma_range)
        self.max_epochs = max(1, int(max_epochs))
        self.steps_per_tau = max(2, int(steps_per_tau))

    def build(self, num_params: int, params_per_owner: int, rng: np.random.Generator) -> Callable[[float], np.ndarray]:
        loguniform = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), size=num_params))
        taus = loguniform(*self.tau_range)
        sigmas = loguniform(*self.sigma_range)
        dt = float(taus.min()) / self.steps_per_tau if num_params else 1.0
        n = int(np.ceil(self.max_epochs / dt)) + 2
        phi = np.exp(-dt / taus)
        x = np.empty((n, num_params))
        x[0] = sigmas * rng.standard_normal(num_params)
        kicks = sigmas * np.sqrt(1.0 - phi ** 2) * rng.standard_normal((n - 1, num_params))
        for k in range(1, n):
            x[k] = phi * x[k - 1] + kicks[k - 1]
        x = x.astype(np.float32)

        def p_opt(t: float) -> np.ndarray:
            u = min(max(float(t), 0.0) / dt, n - 1.0)
            i = min(int(u), n - 2)
            f = u - i
            return (1.0 - f) * x[i] + f * x[i + 1]
        p_opt.taus, p_opt.sigmas = taus, sigmas
        return p_opt


def make_drift(profile: str, frequency: float, amplitude: float, alpha: float, white_fraction: float,
               sigma: float, common_fraction: float, max_epochs: int,
               ar1_tau_range: Tuple[float, float] = (1.0, 50.0), ar1_sigma_range: Tuple[float, float] = (0.3, 1.5)):
    """
    Builds a drift model from flat settings: "sinusoidal" uses frequency/amplitude, "arma"
    alpha/white_fraction/sigma/common_fraction, "ar1" the AR(1) tau and sigma ranges.
    """
    if profile == "sinusoidal":
        return SinusoidalDrift(frequency=frequency, amplitude=amplitude)
    if profile == "arma":
        return ArmaDrift(alpha=alpha, white_fraction=white_fraction, sigma=sigma,
                         common_fraction=common_fraction, max_epochs=max_epochs)
    if profile == "ar1":
        return Ar1Drift(tau_range=ar1_tau_range, sigma_range=ar1_sigma_range, max_epochs=max_epochs)
    raise ValueError(f"Unknown drift profile '{profile}' (use 'sinusoidal', 'arma' or 'ar1')")


# ---------------------------------------------------------------------- noise model
# SI1000-style noise (Gidney, Newman & McEwen 2021, "Benchmarking the planar honeycomb
# code"): with p the two-qubit gate error rate,
#   two-qubit gate         DEPOLARIZE2(p) after
#   single-qubit gate      DEPOLARIZE1(p / 10) after
#   reset                  X_ERROR(2p) after (Z_ERROR for X-basis resets)
#   measurement            X_ERROR(5p) before (Z_ERROR for X-basis measurements)
#   idle, gate moment      DEPOLARIZE1(p / 10) on qubits a TICK-delimited moment leaves untouched
#   idle, readout moment   DEPOLARIZE1(2p) on untouched qubits in a moment with a measurement or reset

def _default_gate_sets() -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """Classifies stim gates into (two-qubit unitaries, single-qubit unitaries, resets, measurements)."""
    two_qubit, single_qubit, reset, measure = set(), set(), set(), set()
    for name, g in stim.gate_data().items():
        if name == "MPAD" or name.startswith("HERALDED_") or (g.is_noisy_gate and not g.produces_measurements):
            continue
        if g.is_two_qubit_gate and g.is_unitary:
            two_qubit.add(name)
        elif g.is_single_qubit_gate:
            if g.is_unitary:
                single_qubit.add(name)
            if g.is_reset:
                reset.add(name)
            if g.produces_measurements:
                measure.add(name)
    return two_qubit, single_qubit, reset, measure


def flip_channel(gate: str) -> str:
    """The classical flip error for a reset or measurement: Z_ERROR in the X basis, X_ERROR in the Y or Z basis."""
    return "Z_ERROR" if gate.endswith("X") else "X_ERROR"


@dataclass
class NoiseModel:
    """
    Where the drift-controlled error channels are placed, and each channel's base rate as
    a multiple of the two-qubit rate p (SI1000 by default). A gate set left as None is
    inferred from stim's gate metadata, so any circuit is covered by default. A scale of 0
    drops that kind of channel.
    """
    two_qubit_gates: Optional[Set[str]] = None  # default: all 2-qubit unitaries
    single_qubit_gates: Optional[Set[str]] = None  # default: all 1-qubit unitaries
    reset_gates: Optional[Set[str]] = None  # default: all 1-qubit resets (incl. MR)
    measure_gates: Optional[Set[str]] = None  # default: all 1-qubit measurements (incl. MR)
    include_single_qubit_noise: bool = True  # False keeps only the two-qubit channels
    scale_2q: float = 1.0
    scale_1q: float = 0.1
    scale_reset: float = 2.0
    scale_measure: float = 5.0
    scale_idle: float = 0.1  # idle during a moment of gates
    scale_idle_mr: float = 2.0  # idle during a moment with a measurement or reset
    max_error: float = 0.15  # error rates are clipped to [base_s, max_error]

    def resolved(self) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
        defaults = _default_gate_sets()
        given = (self.two_qubit_gates, self.single_qubit_gates, self.reset_gates, self.measure_gates)
        return tuple(set(g) if g is not None else d for g, d in zip(given, defaults))

    def scales(self) -> Dict[str, float]:
        """Base-rate multiplier of each source kind, without the kinds switched off."""
        scales = {"2q": self.scale_2q}
        if self.include_single_qubit_noise:
            scales.update({"1q": self.scale_1q, "reset": self.scale_reset, "meas": self.scale_measure,
                           "idle": self.scale_idle, "idle_mr": self.scale_idle_mr})
        return {kind: float(s) for kind, s in scales.items() if s > 0}
