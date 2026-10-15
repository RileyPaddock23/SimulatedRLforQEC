"""
Minimal Single-Parameter Control Model
======================================
One drifting control parameter p (a single CX coupler i of the rotated surface code)
with optimum p*(t). Optionally one neighbor coupler l, which shares detectors with i,
drifts too (independently, same tau and s). Every other parameter sits at its optimum.
The offsets are e = p - p*, and each gate's error rate is

    eps = eps_0 + Omega * e^2        (clipped at 0.15)

Reading: the mean EDR over the detectors M_i that parameter i can flip,

    y = y_0 + C_ii e_i^2 + C_il e_l^2 + sigma * n,   sigma = sigma_1 / sqrt(S),   n ~ N(0, 1)

with C_ii = H_l. Drift: AR(1) per slot (one slot = S shots), x_{t+1} = a x_t + w_t,
a = exp(-1/tau), w_t ~ N(0, q^2), stationary variance s^2 = q^2 / (1 - a^2).

Reading modes:
  - "stim":     detection events of the real circuit. Only the modeled couplers' rates
                change with e, so the baseline circuit is sampled once and each coupler's
                excess depolarizing is added exactly on top (depolarizing channels compose
                multiplicatively), using a precomputed Pauli -> detector-flip table.
  - "gaussian": the idealized model above, with y_0, C, sigma_1 calibrated from stim.

Also provides the closed-form results: the Cramer-Rao bound c*, the steerable
limit tau_lim, and the mean / variance of one antithetic policy update.
"""

from typing import Dict, Optional, Union
import numpy as np
import stim

from surface_code_env import RotatedSurfaceCodeEnv

PAULIS = "IXYZ"


class MinimalControlEnv:
    def __init__(
        self,
        mode: str = "stim",  # "stim" or "gaussian"
        distance: int = 3,
        rounds: int = 3,
        param_index: int = 0,
        neighbor_index: Union[None, int, str] = None,  # None, a CX index, or "auto" (most shared detectors)
        base_error: float = 0.002,
        sensitivity: float = 0.025,
        tau: Optional[float] = 1000.0,  # drift correlation time in slots (None = static)
        drift_std: float = 0.5,  # stationary drift std s
        calib_shots: int = 200_000,
        seed: int = 163,
    ):
        if mode not in ("stim", "gaussian"):
            raise ValueError(f"Unknown mode '{mode}' (use 'stim' or 'gaussian')")
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        # Full code with drift switched off: all parameters at their optimum except the modeled ones
        self.code = RotatedSurfaceCodeEnv(
            distance=distance, rounds=rounds, base_error=base_error, sensitivity=sensitivity,
            drift_profile="sinusoidal", drift_amplitude=0.0, seed=seed,
        )
        n_cx = self.code.num_cx_gates
        if not 0 <= param_index < n_cx:
            raise ValueError(f"param_index must be a CX coupler parameter (0..{n_cx - 1})")
        self.i = param_index
        self.M_i = np.flatnonzero(self.code.M[:, param_index])
        self.num_detectors = self.code.num_detectors

        overlap = self.code.M[self.M_i, :n_cx].sum(axis=0)
        if neighbor_index == "auto":
            overlap[param_index] = -1
            neighbor_index = int(np.argmax(overlap))
        if neighbor_index is not None and (neighbor_index == param_index or overlap[neighbor_index] <= 0):
            raise ValueError("neighbor_index must be a different CX coupler sharing detectors with M_i")
        self.neighbor = neighbor_index
        self.overlap = 0 if neighbor_index is None else int(overlap[neighbor_index])

        # Modeled couplers: index 0 = i, index 1 = neighbor l (if any)
        self.idx = [param_index] + ([neighbor_index] if neighbor_index is not None else [])
        self.n = len(self.idx)
        self.omega = self.code.param_sensitivities[self.idx].astype(float)
        self.eps0 = self.code.param_base_errors[self.idx].astype(float)

        self._setup_fast_sampler()
        self.set_drift(tau, drift_std)
        self.calibrate(shots=calib_shots)

    # ------------------------------------------------------------------ sampling
    def _flip_table(self, gate: int, lines) -> np.ndarray:
        """Pauli -> M_i detector-flip table, shape (instances, 16, |M_i|), for one CX coupler."""
        pair = self.code.cx_pairs[gate]
        positions = []
        for pos, op in enumerate(self.code.flat_instructions):
            if op.name == "CX":
                t = op.targets_copy()
                if any((t[k].value, t[k + 1].value) == pair for k in range(0, len(t), 2)):
                    positions.append(pos)

        # Index 0 = II (no flip). Errors are inserted as probability-1 noise so they are
        # excluded from Stim's noiseless reference and show up as detection events.
        paulis = [(a, b) for a in PAULIS for b in PAULIS]
        flips = np.zeros((len(positions), 16, len(self.M_i)), dtype=bool)
        for n, pos in enumerate(positions):
            for k, (pa, pb) in enumerate(paulis[1:], start=1):
                ins = [f"{p}_ERROR(1) {q}" for p, q in ((pa, pair[0]), (pb, pair[1])) if p != "I"]
                circ = stim.Circuit("\n".join(lines[:pos + 1] + ins + lines[pos + 1:]))
                flips[n, k] = circ.compile_detector_sampler().sample(1)[0, self.M_i]
        return flips

    def _setup_fast_sampler(self):
        """Flip tables for each modeled coupler, plus a baseline sampler."""
        lines = [str(op) for op in self.code.flat_instructions]
        self._flips = [self._flip_table(g, lines) for g in self.idx]
        if not self._flips[0].any(axis=(0, 1)).all():
            raise RuntimeError("Flip table does not cover every detector in M_i")

        base = self.code.build_noisy_circuit(np.zeros(self.code.num_params, dtype=np.float32), 0.0)
        self._base_sampler = base.compile_detector_sampler(seed=int(self.rng.integers(2**31)))
        self._base_buf = np.zeros((0, len(self.M_i)), dtype=bool)

    def _base_shots(self, shots: int) -> np.ndarray:
        """Baseline detection events on M_i (all gates at their base rate), drawn from a buffer."""
        if len(self._base_buf) < shots:
            fresh = self._base_sampler.sample(max(shots, 100_000))[:, self.M_i]
            self._base_buf = np.concatenate([self._base_buf, fresh])
        out, self._base_buf = self._base_buf[:shots].copy(), self._base_buf[shots:]
        return out

    def _offsets(self, e) -> np.ndarray:
        """Offsets of the modeled couplers as an array of shape (n,); a scalar sets coupler i only."""
        e = np.atleast_1d(np.asarray(e, dtype=float))
        return e if len(e) == self.n else np.concatenate([e, np.zeros(self.n - len(e))])

    def error_rate(self, e) -> np.ndarray:
        """Gate error rates eps = eps_0 + Omega e^2 (clipped at 0.15) of the modeled couplers."""
        return np.minimum(self.eps0 + self.omega * self._offsets(e) ** 2, 0.15)

    def sample_shots(self, e, shots: int) -> np.ndarray:
        """Per-shot readings (fraction of M_i detectors that fired) from the stim circuit at offsets e."""
        det = self._base_shots(shots)
        # DEPOLARIZE2(p) = apply a uniform random 2-qubit Pauli (of 16) with prob r = 16p/15;
        # composing channels multiplies (1 - r), so the excess over eps_0 has r_x below.
        r0 = 16.0 / 15.0 * self.eps0
        r = 16.0 / 15.0 * self.error_rate(e)
        r_x = 1.0 - (1.0 - r) / (1.0 - r0)
        for flips, rx in zip(self._flips, r_x):
            n_inst = flips.shape[0]
            idx = self.rng.integers(0, 16, size=(shots, n_inst)) * (self.rng.random((shots, n_inst)) < rx)
            for n in range(n_inst):
                det ^= flips[n][idx[:, n]]
        return det.mean(axis=1)

    def read(self, e, shots: int) -> float:
        """One reading y (mean EDR over M_i) of `shots` shots at offsets e."""
        if self.mode == "gaussian":
            return self.y0 + float(self.C @ self._offsets(e) ** 2) + self.sigma_1 / np.sqrt(shots) * self.rng.standard_normal()
        return float(self.sample_shots(e, shots).mean())

    def full_circuit_reading(self, e, shots: int) -> float:
        """Same reading from the full RotatedSurfaceCodeEnv circuit (slow; for cross-checking)."""
        params = np.zeros(self.code.num_params, dtype=np.float32)
        params[self.idx] = self._offsets(e)
        return float(self.code.evaluate_policy(params, 0.0, shots=shots)["edr_vector"][self.M_i].mean())

    def calibrate(self, shots: int = 200_000, e_max: Optional[float] = None):
        """Fits y = y_0 + C_c e_c^2 for each modeled coupler c on the stim circuit, and
        measures the single-shot noise sigma_1 at e = 0."""
        e_max = 2.5 * self.s if e_max is None else e_max
        grid = np.linspace(-e_max, e_max, 11)
        A = np.stack([np.ones_like(grid), grid ** 2], axis=1)
        self.C = np.zeros(self.n)
        for c in range(self.n):
            y = np.array([self.sample_shots(np.eye(self.n)[c] * e, shots).mean() for e in grid])
            (y0, self.C[c]), *_ = np.linalg.lstsq(A, y, rcond=None)
            if c == 0:
                self.y0 = float(y0)
                self.calib_rel_resid = float(np.sqrt(np.mean((A @ [y0, self.C[0]] - y) ** 2)) / (self.C[0] * e_max ** 2))
        self.H_l = float(self.C[0])
        self.C_il = float(self.C[1]) if self.n > 1 else 0.0
        self.sigma_1 = float(np.std(self.sample_shots(0.0, shots)))
        self.Sigma_1 = self.sigma_1 ** 2

    # --------------------------------------------------------------------- drift
    def set_drift(self, tau: Optional[float], drift_std: float):
        """AR(1) drift with correlation time tau (slots) and stationary std s; tau=None is static."""
        self.tau = tau
        self.s = float(drift_std)
        self.a = 1.0 if tau is None or np.isinf(tau) else float(np.exp(-1.0 / tau))
        self.q = self.s * np.sqrt(1.0 - self.a ** 2)
        self.reset_drift()

    def reset_drift(self):
        """Draws p* (shape (n,)) from the stationary distribution (p* = 0 when static)."""
        self.p_star = self.s * self.rng.standard_normal(self.n) if self.a < 1.0 else np.zeros(self.n)

    def step(self):
        """Advances the drift by one slot."""
        self.p_star = self.a * self.p_star + self.q * self.rng.standard_normal(self.n)

    # ------------------------------------------------------------- derived constants
    @property
    def H_g(self) -> float:
        """Curvature of the full cost (mean EDR over all detectors)."""
        return len(self.M_i) / self.num_detectors * self.H_l

    def kappa(self, S: int) -> float:
        """Information per reading, kappa = 4 H_l^2 / sigma^2 with sigma^2 = Sigma_1 / S."""
        return 4.0 * self.H_l ** 2 * S / self.Sigma_1

    def Lambda(self, S: int) -> float:
        """Information budget Lambda = kappa s^4."""
        return self.kappa(S) * self.s ** 4

    def summary(self) -> Dict[str, float]:
        return {
            "omega": float(self.omega[0]), "eps0": float(self.eps0[0]), "M_i": len(self.M_i), "N_det": self.num_detectors,
            "y0": self.y0, "H_l": self.H_l, "H_g": self.H_g, "C_il": self.C_il, "neighbor": self.neighbor,
            "overlap": self.overlap, "sigma_1": self.sigma_1, "calib_rel_resid": self.calib_rel_resid,
        }


# ---------------------------------------------------------------------- analytics
def cramer_rao_c(kappa: float, q2: float, a: float, sigma_p2: float = 0.0) -> float:
    """Cramer-Rao c* = E[(mu - p*)^2] for readings taken at mu +- delta (probe variance sigma_p2).
    Each reading carries information kappa (e^2 + sigma_p^2), so c* is the positive root of

        kappa c^3 + kappa (sigma_p^2 - q^2) c^2 + (1 - a^2 - kappa q^2 sigma_p^2) c - q^2 = 0,

    which reduces to kappa c^3 - kappa q^2 c^2 + (1 - a^2) c - q^2 = 0 at sigma_p = 0."""
    if q2 <= 0.0:
        return 0.0
    eps = 1.0 - a * a
    f = lambda c: kappa * c ** 3 + kappa * (sigma_p2 - q2) * c ** 2 + (eps - kappa * q2 * sigma_p2) * c - q2
    # f(q^2) < 0 and f(s^2) >= 0, so the root lies in [q^2, s^2]
    lo = q2
    hi = q2 / eps if eps > 0 else max(q2, (q2 / kappa) ** (1.0 / 3.0))
    while f(hi) < 0:
        hi *= 2.0
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return float(np.sqrt(lo * hi))


def tau_lim(Lam: float, R_t: float, k: float, sigma_p2_s2: float = 0.0) -> float:
    """Shortest steerable tau (slots) for a controller (1 + k)x above the bound to reach c/s^2 = R_t.
    sigma_p2_s2 = sigma_p^2 / s^2 is the probe variance (0 recovers the probe-free bound)."""
    rho = R_t / (1.0 + k)
    v = sigma_p2_s2
    eps = Lam * rho ** 2 * (rho + v) / (1.0 - rho + Lam * rho ** 2 + Lam * v * rho)
    if eps >= 1.0:
        return 0.0
    return float(-2.0 / np.log(1.0 - eps))


def v_pair(s: float, B: int, tau: Optional[float]) -> float:
    """Variance of the drift between antithetic partners B/2 slots apart (tau in slots)."""
    if tau is None or np.isinf(tau):
        return 0.0
    return 2.0 * s ** 2 * (1.0 - np.exp(-B / (2.0 * tau)))


def update_stats_theory(
    lr: float, sigma_p: float, B: int, H_l: float, ebar_mean: float, ebar_sq: float,
    Sigma_1: float, S: int, V_pair: float, C_il: float = 0.0, ebar_l_sq: float = 0.0,
) -> Dict[str, float]:
    """E[dmu_i] and Var[dmu_i] of one update. The neighbor term L_nbr = 4 C_il^2 <ebar_l^2> sigma_p^2
    is zero for a single parameter."""
    L = 4.0 * H_l ** 2 * ebar_sq * sigma_p ** 2
    L_nbr = 4.0 * C_il ** 2 * ebar_l_sq * sigma_p ** 2
    mean = -2.0 * lr * H_l * sigma_p ** 2 * ebar_mean
    var = lr ** 2 * sigma_p ** 2 / B * (
        4.0 * L + 2.0 * L_nbr + Sigma_1 / S + (L + L_nbr) * V_pair / (2.0 * sigma_p ** 2)
    )
    return {"mean": mean, "var": var, "L": L, "L_nbr": L_nbr}
