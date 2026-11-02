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

with C_ii = H_l. Drift: AR(1) per candidate (one candidate = S shots), x_{t+1} = a x_t + w_t,
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

from surface_code_env import RotatedSurfaceCodeEnv, generate_power_law_noise

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
        tau: Optional[float] = 1000.0,  # drift correlation time in candidates (None = static)
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

    def read_many(self, e, shots: int, noise: Optional[np.ndarray] = None) -> np.ndarray:
        """Readings for independent copies of coupler i at offsets e, shape (R,) (any neighbor at
        its optimum). `noise` optionally fixes the gaussian mode's standard normals."""
        e = np.asarray(e, dtype=float)
        if self.mode == "gaussian":
            n = self.rng.standard_normal(e.shape) if noise is None else noise
            return self.y0 + self.H_l * e ** 2 + self.sigma_1 / np.sqrt(shots) * n
        return np.array([self.sample_shots(x, shots).mean() for x in e])

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
        """AR(1) drift with correlation time tau (candidates) and stationary std s; tau=None is static."""
        self.tau = tau
        self.s = float(drift_std)
        self.a = 1.0 if tau is None or np.isinf(tau) else float(np.exp(-1.0 / tau))
        self.q = self.s * np.sqrt(1.0 - self.a ** 2)
        self.reset_drift()

    def reset_drift(self):
        """Draws p* (shape (n,)) from the stationary distribution (p* = 0 when static)."""
        self.p_star = self.s * self.rng.standard_normal(self.n) if self.a < 1.0 else np.zeros(self.n)

    def step(self):
        """Advances the drift by one candidate."""
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


def drift_trajectories(
    n_candidates: int, replicas: int, s: float, rng: np.random.Generator, profile: str = "ar1",
    tau: Optional[float] = None, alpha: float = 0.67, white_fraction: float = 0.1, candidates_per_drift_update: float = 1.0,
):
    """p*(candidate) for independent replicas, shape (n_candidates, replicas), and the candidate times (drift updates).
      - "ar1":      AR(1) per candidate with correlation time tau (candidates) and stationary std s, as step().
                    Times are in candidates (one drift update = one candidate).
      - "powerlaw": generate_power_law_noise(normalize="analytic") per drift update, std s, linearly
                    interpolated; candidate k sits at t = k / candidates_per_drift_update."""
    if profile == "ar1":
        a = np.exp(-1.0 / tau)
        p = np.empty((n_candidates, replicas))
        p[0] = s * rng.standard_normal(replicas)
        w = s * np.sqrt(1.0 - a * a) * rng.standard_normal((n_candidates, replicas))
        for k in range(1, n_candidates):
            p[k] = a * p[k - 1] + w[k]
        return p, np.arange(n_candidates, dtype=float)
    if profile != "powerlaw":
        raise ValueError(f"Unknown drift profile '{profile}' (use 'ar1' or 'powerlaw')")
    t = np.arange(n_candidates) / candidates_per_drift_update
    n_ep = int(np.floor(t[-1])) + 2
    traj = np.stack([generate_power_law_noise(n_ep, alpha=alpha, white_fraction=white_fraction, sigma=s, rng=rng,
                                              normalize="analytic") for _ in range(replicas)], axis=1)
    n = np.floor(t).astype(int)
    f = (t - n)[:, None]
    return (1.0 - f) * traj[n] + f * traj[n + 1], t


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


def mismatched_ar1_c(info: float, tau: float, s2: float, tau_f: float, s2_f: float) -> float:
    """Prediction error E[(x - m)^2] of a steady-state scalar Kalman filter that assumes AR(1)
    drift (tau_f, s2_f) while the drift is AR(1) (tau, s2), with one reading of information `info`
    per candidate (tau in candidates). Exact: the stationary covariance of the joint (truth, estimate) system
        x' = a x + w,   m' = a_f (1 - K) m + a_f K (x + v),   K the filter's own steady gain,
    from vec(Sigma) = (I - F kron F)^-1 vec(Q). With tau_f = tau, s2_f = s2 it is the Riccati value."""
    R = 1.0 / info
    a, a_f = np.exp(-1.0 / tau), np.exp(-1.0 / tau_f)
    q, q_f = s2 * (1.0 - a * a), s2_f * (1.0 - a_f * a_f)
    b = R * (1.0 - a_f * a_f) - q_f  # filter's prior variance P: P^2 + b P - q_f R = 0
    P = (-b + np.sqrt(b * b + 4.0 * q_f * R)) / 2.0
    K = P / (P + R)
    F = np.array([[a, 0.0], [a_f * K, a_f * (1.0 - K)]])
    Q = np.diag([q, a_f * a_f * K * K * R])
    S = np.linalg.solve(np.eye(4) - np.kron(F, F), Q.ravel()).reshape(2, 2)
    return float(S[0, 0] + S[1, 1] - 2.0 * S[0, 1])


def tracker_ar1_c(
    C: np.ndarray, R0: np.ndarray, y0: np.ndarray, rate_scaled: bool, sigma_p: float, B: int, lead: float,
    tau: float, s2: float, tau_f: float, s2_f: float, n_quad: int = 40, tol: float = 1e-8, max_epochs: int = 100_000,
) -> np.ndarray:
    """E[(mu - p*)^2] per parameter of KalmanTracker with one AR(1) model (tau_f, s2_f) per parameter, on
    independent AR(1) drift (tau, s2) of every parameter. Times in candidates. Includes what mismatched_ar1_c
    leaves out:
      - the epoch: mu is held for B candidates and set to the forecast `lead` candidates past the last pair;
        pairs are split (candidate i with i + B/2), so each pair difference measures the midpoint
        (p_e + p_l) / 2, not p_l
      - the tracker's measurement variance R, computed from its own variance v and its own increment
        variance V_D(tau_f, s2_f): crosstalk 16 sigma_p^2 sum_k!=i C_ik^2 v_k, pair drift
        sum_k C_ik^2 (4 V_D (sigma_p^2 + v) + 2 V_D^2), and the shot noise (scaled by the predicted
        reading if rate_scaled). The true innovation has the same terms at the true errors
      - the random information of each pair, J^2 = 16 C_ii^2 delta^2 with delta ~ N(0, sigma_p^2): gains are
        averaged over delta (Gauss-Hermite) and second moments propagated exactly
    The filter's own variance is propagated in expectation. Parameters couple only through the
    crosstalk variance, so the state per parameter is (x, m, mu, x_0 .. x_{B/2-1}), the last being the
    epoch's early truths; its covariance is iterated epoch by epoch to the periodic steady state.

    Args:
        C: curvatures C_ik (P x P); R0: single-reading noise variance sigma_1^2 / S (P,); y0: base readings (P,)
    """
    C = np.atleast_2d(np.asarray(C, dtype=float))
    n = len(C)
    Cd, C2, Crow = np.diag(C), C ** 2, C.sum(axis=1)
    sp2 = sigma_p ** 2
    J2 = 16.0 * Cd ** 2 * sp2  # mean pair information; a pair's is J2 u, u ~ chi^2_1
    z, w = np.polynomial.hermite_e.hermegauss(n_quad)
    u, w = z * z, w / w.sum()
    a, af = np.exp(-1.0 / tau), np.exp(-1.0 / tau_f)
    q = s2 * (1.0 - a * a)
    h = B // 2
    VD, VD_f = 2.0 * s2 * (1.0 - a ** h), 2.0 * s2_f * (1.0 - af ** h)
    dts = np.diff(np.concatenate([[-1], np.arange(h, B)]))  # filter steps to each later reading

    def shot(var):
        return 2.0 * R0 * (1.0 + C @ (sp2 + var) / y0) if rate_scaled else 2.0 * R0

    # Filter's own periodic Riccati: gain moments E[g], E[g^2], E[g^2 / u] of each pair
    P = np.full(n, float(s2_f))
    for _ in range(max_epochs):
        P_start, moments = P.copy(), []
        for dt in dts:
            F2 = af ** (2 * dt)
            P = F2 * P + s2_f * (1.0 - F2)
            rho = (shot(P) + 16.0 * sp2 * (C2 @ P - Cd ** 2 * P) + C2 @ (4.0 * VD_f * (sp2 + P) + 2.0 * VD_f ** 2)) / J2
            g = u[None, :] * P[:, None] / (u[None, :] * P[:, None] + rho[:, None])
            moments.append((g @ w, (g * g) @ w, (g * g / u[None, :]) @ w))
            P = P - (g @ w) * P
        if np.max(np.abs(P - P_start) / P_start) < tol:
            break

    X, M, U = 0, 1, 2
    D = 3 + h
    quad = lambda S, v: np.einsum("i,nij,j->n", v, S, v)
    Sig = np.zeros((n, D, D))
    Sig[:, X, X] = s2
    p_err = np.full((h, n), 0.5 * s2)  # E[(p_hat - p_mid)^2] at each pair: crosstalk residual
    mu_err = np.full((h, n), 0.5 * s2)  # E[(mu - p_mid)^2] at each pair: pair drift, rate scaling
    e_mu = np.zeros(D)
    e_mu[U], e_mu[X] = 1.0, -1.0
    c_prev = None
    for _ in range(max_epochs):
        Sig[:, U, :] = af ** lead * Sig[:, M, :]
        Sig[:, :, U] = af ** lead * Sig[:, :, M]
        c = np.zeros(n)
        for j in range(B):
            c += quad(Sig, e_mu)
            if j < h:  # keep the early truth x_j for its partner
                b = 3 + j
                Sig[:, b, :], Sig[:, :, b] = Sig[:, X, :], Sig[:, :, X]
            else:
                k = j - h
                F = af ** dts[k]
                mid = np.zeros(D)
                mid[X] = mid[3 + k] = 0.5
                ep, em = -mid, -mid.copy()
                ep[M], em[U] = F, 1.0
                pe, me = p_err[k], mu_err[k]
                p_err[k], mu_err[k] = quad(Sig, ep), quad(Sig, em)
                # true innovation variance / J2; the own 2 C delta D term is part of the midpoint signal
                rho = (shot(me) + 16.0 * sp2 * (C2 @ pe - Cd ** 2 * pe) + C2 @ (4.0 * VD * (sp2 + me) + 2.0 * VD ** 2)
                       - 4.0 * Cd ** 2 * VD * sp2 + (Crow * VD) ** 2) / J2
                Eg, Eg2, Egu = moments[k]
                L0, L1 = np.eye(D), np.zeros((D, D))
                L0[M, M] = F
                L1[M], L1[M, M] = mid, -F  # m <- F m + g (p_mid - F m + noise)
                A = L0 @ Sig @ L1.T
                Sig = (L0 @ Sig @ L0.T + Eg[:, None, None] * (A + A.transpose(0, 2, 1))
                       + Eg2[:, None, None] * (L1 @ Sig @ L1.T))
                Sig[:, M, M] += Egu * rho
            Sig[:, X, :] *= a
            Sig[:, :, X] *= a
            Sig[:, X, X] += q
        c /= B
        if c_prev is not None and np.max(np.abs(c - c_prev) / c) < tol:
            break
        c_prev = c
    return c


def tau_lim(Lam: float, R_t: float, k: float, sigma_p2_s2: float = 0.0) -> float:
    """Shortest steerable tau (candidates) for a controller (1 + k)x above the bound to reach c/s^2 = R_t.
    sigma_p2_s2 = sigma_p^2 / s^2 is the probe variance (0 recovers the probe-free bound)."""
    rho = R_t / (1.0 + k)
    v = sigma_p2_s2
    eps = Lam * rho ** 2 * (rho + v) / (1.0 - rho + Lam * rho ** 2 + Lam * v * rho)
    if eps >= 1.0:
        return 0.0
    return float(-2.0 / np.log(1.0 - eps))


def v_pair(s: float, B: int, tau: Optional[float]) -> float:
    """Variance of the drift between antithetic partners B/2 candidates apart (tau in candidates)."""
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
