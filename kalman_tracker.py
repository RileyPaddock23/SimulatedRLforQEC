"""
Kalman Drift Tracker
====================
Model-based counterpart of the PEPG agent (rl_agent.py). Each control parameter's
optimum p*(t) is tracked by a Kalman filter built on an assumed linear-Gaussian drift
model, and the policy mean mu is set to the filter's forecast.

Key Features:
1. Same interface as RLControlAgent: sample_policies() draws antithetic probes mu +- delta,
   update() consumes the measured detector error rates (plus their measurement times)
2. Same masked reading as the agent: y_i = mean EDR over the detectors M_i of parameter i
3. One filter per parameter. The reading y_i = y0_i + sum_k C_ik (c_k - p_k)^2 + noise is
   quadratic, so a single reading cannot tell which side of the probe p lies on (a per-reading
   EKF loses lock once |p - p_hat| >> delta). Each antithetic pair is used instead: its
   difference is linear in p_i with the right sign at any offset, as in the PEPG gradient.
   Crosstalk from the other probed parameters and the drift between partners enter the
   measurement variance, not the state
4. Pluggable drift models (time t in drift updates, as RotatedSurfaceCodeEnv.get_drift_opt):
     - OUSumModel:    AR(1) (one component) or a sum of AR(1)s
     - PowerLawModel: the exact 1/f^alpha + white-floor process of surface_code_env
                      (normalize="analytic"), sampled per drift update and linearly interpolated
5. Fitting helpers that approximate the power-law drift by one AR(1) or a sum of AR(1)s
"""

from itertools import combinations
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from surface_code_env import fractional_taps


# ------------------------------------------------------------------ drift models
class OUSumModel:
    """p(t) = sum_k x_k(t): independent Ornstein-Uhlenbeck components with correlation times
    `taus` (drift updates) and stationary variances `variances`. One component is an AR(1)."""

    def __init__(self, taus: Sequence[float], variances: Sequence[float]):
        self.taus = np.atleast_1d(np.asarray(taus, dtype=float))
        self.var = np.atleast_1d(np.asarray(variances, dtype=float))
        self.dim = len(self.taus)

    def init(self, n: int) -> Dict:
        return {"m": np.zeros((n, self.dim)), "P": np.tile(np.diag(self.var), (n, 1, 1)), "t": 0.0}

    def predict(self, st: Dict, t: float):
        dt = t - st["t"]
        if dt > 0:
            F = np.exp(-dt / self.taus)
            st["m"] *= F
            st["P"] = st["P"] * np.outer(F, F) + np.diag(self.var * (1.0 - F ** 2))
            st["t"] = t

    def moments(self, st: Dict):
        return st["m"].sum(axis=1), st["P"].sum(axis=(1, 2))

    def update(self, st: Dict, J: np.ndarray, r: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Scalar update per parameter for an innovation r = J (p - p_hat) + noise (variance R)."""
        Ph = st["P"].sum(axis=2)  # P h with h = 1
        S = J ** 2 * Ph.sum(axis=1) + R
        st["m"] += Ph * (J * r / S)[:, None]
        st["P"] -= Ph[:, :, None] * Ph[:, None, :] * (J ** 2 / S)[:, None, None]
        return S

    def forecast(self, st: Dict, t: float) -> np.ndarray:
        return (st["m"] * np.exp(-max(t - st["t"], 0.0) / self.taus)).sum(axis=1)

    def increment_var(self, t1: float, t2: float) -> float:
        """Var[p(t2) - p(t1)]."""
        return float(np.sum(2.0 * self.var * (1.0 - np.exp(-abs(t2 - t1) / self.taus))))


class PowerLawModel:
    """Exact state-space form of generate_power_law_noise(normalize="analytic"):

        p_n  = sigma (sqrt(1 - w) sum_k b_k u_{n-k} + sqrt(w) v_n),   u, v ~ N(0, 1)
        p(t) = (1 - f) p_n + f p_{n+1},   n = floor(t), f = t - n

    The state holds the L+1 innovations u_{n+1-L} .. u_{n+1} (a ring buffer) and v_n, v_{n+1}.
    Within a drift update p(t) depends only on z = (p_n, p_{n+1}), so readings update a 2-D Gaussian
    over z, which is folded back into the full state when the drift update ends."""

    def __init__(self, alpha: float, white_fraction: float, sigma: float, n_taps: int = 256):
        b = fractional_taps(alpha, n_taps)
        w = float(np.clip(white_fraction, 0.0, 1.0))
        self.L = len(b)
        self.cb = sigma * np.sqrt(1.0 - w) * b
        self.cw = sigma * np.sqrt(w)
        self.dim = self.L + 3
        self.gamma = power_law_autocov(alpha, w, sigma, n_taps)  # per-drift-update autocovariance

    def _h(self, j: int, known_upto: Optional[int] = None) -> np.ndarray:
        """p_j = h . x in ring coordinates; innovations after `known_upto` (mean 0) are left out."""
        h = np.zeros(self.dim)
        k = np.arange(self.L)
        if known_upto is not None:
            k = k[j - k <= known_upto]
        h[(j - k) % (self.L + 1)] = self.cb[k]
        if known_upto is None or j <= known_upto:
            h[self.L + 1 + j % 2] = self.cw
        return h

    def init(self, n: int) -> Dict:
        st = {"m": np.zeros((n, self.dim)), "P": np.tile(np.eye(self.dim), (n, 1, 1)), "drift_update": 0, "f": 0.0}
        self._open_drift_update(st)
        return st

    def _open_drift_update(self, st: Dict):
        n = st["drift_update"]
        H = np.stack([self._h(n), self._h(n + 1)])  # (2, dim)
        G = st["P"] @ H.T  # (N, dim, 2)
        S0 = H @ G
        S0 = 0.5 * (S0 + S0.transpose(0, 2, 1))
        st["W"] = G @ np.linalg.inv(S0)
        st["z0"] = st["m"] @ H.T
        st["S0"] = S0
        st["z"], st["S"] = st["z0"].copy(), S0.copy()

    def _close_drift_update(self, st: Dict):
        W = st["W"]
        st["m"] += (W @ (st["z"] - st["z0"])[:, :, None])[:, :, 0]
        st["P"] -= (W @ (st["S0"] - st["S"])) @ W.transpose(0, 2, 1)

    def predict(self, st: Dict, t: float):
        while st["drift_update"] < int(np.floor(t)):
            self._close_drift_update(st)
            e = st["drift_update"]
            # u_{e+1-L} -> u_{e+2} and v_e -> v_{e+2} share ring positions: marginalize, then insert fresh N(0, 1)
            for pos in ((e + 2) % (self.L + 1), self.L + 1 + e % 2):
                st["m"][:, pos] = 0.0
                st["P"][:, pos, :] = 0.0
                st["P"][:, :, pos] = 0.0
                st["P"][:, pos, pos] = 1.0
            st["drift_update"] += 1
            self._open_drift_update(st)
        st["f"] = t - st["drift_update"]

    def moments(self, st: Dict):
        hz = np.array([1.0 - st["f"], st["f"]])
        return st["z"] @ hz, np.einsum("k,nkl,l->n", hz, st["S"], hz)

    def update(self, st: Dict, J: np.ndarray, r: np.ndarray, R: np.ndarray) -> np.ndarray:
        hz = np.array([1.0 - st["f"], st["f"]])
        Sh = st["S"] @ hz  # (N, 2)
        S = J ** 2 * (Sh @ hz) + R
        st["z"] += Sh * (J * r / S)[:, None]
        st["S"] -= Sh[:, :, None] * Sh[:, None, :] * (J ** 2 / S)[:, None, None]
        return S

    def forecast(self, st: Dict, t: float) -> np.ndarray:
        n = st["drift_update"]
        m_now = st["m"] + (st["W"] @ (st["z"] - st["z0"])[:, :, None])[:, :, 0]

        def mean_p(j):
            if j <= n + 1:
                return st["z"][:, j - n]
            return m_now @ self._h(j, known_upto=n + 1)

        j = int(np.floor(t))
        f = t - j
        return (1.0 - f) * mean_p(j) + f * mean_p(j + 1)

    def increment_var(self, t1: float, t2: float) -> float:
        """Var[p(t2) - p(t1)] of the interpolated process."""
        def weights(t):
            n = int(np.floor(t))
            return [(n, 1.0 - (t - n)), (n + 1, t - n)]

        g = lambda m: self.gamma[abs(m)] if abs(m) < len(self.gamma) else 0.0
        cov = lambda ta, tb: sum(wa * wb * g(na - nb) for na, wa in weights(ta) for nb, wb in weights(tb))
        return float(cov(t1, t1) + cov(t2, t2) - 2.0 * cov(t1, t2))


# ------------------------------------------------------------------ fitting helpers
def power_law_autocov(alpha: float, white_fraction: float, sigma: float, n_taps: int = 256) -> np.ndarray:
    """Per-drift update autocovariance gamma(m), m = 0 .. L-1 (zero beyond), of the power-law drift."""
    b = fractional_taps(alpha, n_taps)
    w = float(np.clip(white_fraction, 0.0, 1.0))
    g = sigma ** 2 * (1.0 - w) * np.correlate(b, b, mode="full")[len(b) - 1:]
    g[0] += sigma ** 2 * w
    return g


def interpolated_autocov(gamma: np.ndarray, candidates_per_drift_update: float, lags: np.ndarray, phase: float = 0.0) -> np.ndarray:
    """Autocovariance at candidate lags of the linearly interpolated drift, averaged over the position
    in the drift update. Candidate k sits at t = (k + phase) / candidates_per_drift_update."""
    E = float(candidates_per_drift_update)
    gam = lambda m: np.where(np.abs(m) < len(gamma), gamma[np.minimum(np.abs(m), len(gamma) - 1)], 0.0)
    f0 = ((np.arange(max(1, int(round(E)))) + phase) / E) % 1.0
    t1 = f0[None, :] + np.asarray(lags, dtype=float)[:, None] / E
    n1 = np.floor(t1).astype(int)
    f1 = t1 - n1
    a0, a1 = 1.0 - f0[None, :], f0[None, :]
    c = a0 * (1 - f1) * gam(n1) + a0 * f1 * gam(n1 + 1) + a1 * (1 - f1) * gam(n1 - 1) + a1 * f1 * gam(n1)
    return c.mean(axis=1)


def fit_ar1_lag1(cov: Callable[[np.ndarray], np.ndarray], candidates_per_drift_update: float):
    """AR(1) matching the variance and the one-candidate autocorrelation (= ML on noise-free drift).
    Returns (tau in drift updates, stationary variance)."""
    c0, c1 = cov(np.array([0.0, 1.0]))
    rho = max(c1 / c0, 1e-6)
    return float(-1.0 / (candidates_per_drift_update * np.log(rho))) if rho < 1 else np.inf, float(c0)


def fit_multi_ar1(cov: Callable[[np.ndarray], np.ndarray], candidates_per_drift_update: float, taus: Sequence[float],
                  max_lag_candidates: float):
    """Sum of AR(1)s with fixed correlation times `taus` (drift updates) and non-negative variances fitted
    by least squares to the autocovariance on a log-spaced lag grid (lag 0 weighted x10).
    Returns (taus, variances) with zero-variance components dropped."""
    lags = np.unique(np.concatenate([[0.0], np.round(np.geomspace(1.0, max_lag_candidates, 60))]))
    target = cov(lags)
    A = np.exp(-lags[:, None] / (candidates_per_drift_update * np.asarray(taus, dtype=float)[None, :]))
    wts = np.where(lags == 0, 10.0, 1.0)[:, None]
    Aw, tw = A * wts, target * wts[:, 0]
    best, best_err = None, np.inf
    for k in range(1, len(taus) + 1):  # exhaustive NNLS over supports (few components)
        for sup in combinations(range(len(taus)), k):
            x, *_ = np.linalg.lstsq(Aw[:, sup], tw, rcond=None)
            if np.all(x >= 0):
                err = np.sum((Aw[:, sup] @ x - tw) ** 2)
                if err < best_err:
                    best, best_err = (np.asarray(taus, dtype=float)[list(sup)], x), err
    return best


# ------------------------------------------------------------------ tracker
class KalmanTracker:
    """
    Kalman-filter controller that steers the control parameters to the forecast of their
    drifting optimum, from the same antithetic EDR readings the PEPG agent uses.
    """

    def __init__(
        self,
        num_params: int,
        num_detectors: int,
        mask_matrix: np.ndarray,
        drift_model,
        y0: np.ndarray,
        C: np.ndarray,
        sigma_1: np.ndarray,
        shots: int,
        sigma_probe: float = 0.25,
        rate_scaled_noise: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            rate_scaled_noise: scale the shot-noise variance by the predicted reading over y0 (stim
                readings: detection-event noise grows with the error rate); False keeps it constant
            mask_matrix: factor graph M (O x P); the reading of parameter i is the mean EDR over M_i
            drift_model: OUSumModel or PowerLawModel, shared by every parameter (one filter each)
            y0: reading of each parameter with every offset at zero, shape (P,)
            C: curvature of reading i in offset k, shape (P, P)
            sigma_1: single-shot std of each reading, shape (P,)
            shots: shots per candidate, so the reading noise variance is sigma_1^2 / shots
            sigma_probe: probe rms, a scalar or one value per parameter
        """
        self.num_params = num_params
        self.num_detectors = num_detectors
        self.M = mask_matrix.astype(np.float64)
        self.M_col_sums = np.maximum(np.sum(self.M, axis=0, keepdims=True), 1.0)
        self.model = drift_model
        self.y0 = np.broadcast_to(np.asarray(y0, dtype=float), (num_params,)).copy()
        self.C = np.asarray(C, dtype=float)
        self.C_diag = np.diag(self.C).copy()
        self.C2 = self.C ** 2
        self.R0 = np.broadcast_to(np.asarray(sigma_1, dtype=float) ** 2 / shots, (num_params,)).copy()
        self.sigma = np.broadcast_to(np.asarray(sigma_probe, dtype=np.float32), (num_params,)).copy()
        self.rate_scaled_noise = rate_scaled_noise
        self.rng = np.random.default_rng(seed)

        self.state = self.model.init(num_params)
        self.mu = np.zeros(num_params)

    def sample_policies(self, batch_size: int, antithetic: bool = True) -> np.ndarray:
        """Candidates around mu, drawn exactly as RLControlAgent.sample_policies (same seed, same probes)."""
        if antithetic:
            half_batch = (batch_size + 1) // 2
            noise = self.rng.normal(0.0, 1.0, size=(half_batch, self.num_params)).astype(np.float32)
            delta = noise * self.sigma
            return np.vstack([self.mu + delta, self.mu - delta])[:batch_size]
        noise = self.rng.normal(0.0, 1.0, size=(batch_size, self.num_params)).astype(np.float32)
        return self.mu + noise * self.sigma

    def update(
        self,
        actions: np.ndarray,
        observations: np.ndarray,
        times: np.ndarray,
        t_next: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Filters the epoch pair by pair, then moves mu to the forecast at t_next.

        Each antithetic pair (candidates i and i + B/2, as laid out by sample_policies) is one
        measurement, z = y(c_e, t_e) - y(c_l, t_l) for its earlier (e) and later (l) reading.
        For parameter i, with Dc = c_e - c_l, mu = (c_e + c_l) / 2 and D = p(t_l) - p(t_e),

            z_i = sum_k C_ik [2 Dc_k (mu_k - p_k(t_l)) + 2 D_k (c_e,k - p_k) + D_k^2] + noise,

        which is linear in p_i(t_l) with slope J_i = -2 C_ii Dc_i, whatever the size of the offset.
        Crosstalk (k != i) and the drift between the partners (D) enter as measurement noise.

        Args:
            actions: candidate parameter vectors, shape (B, P)
            observations: detector error rates, shape (B, O)
            times: measurement time of each candidate (drift updates), shape (B,)
            t_next: time at which the new mu should be centered (default: the last reading)

        Returns:
            dict of per-pair diagnostics, each of shape (B/2, P): the predicted p(t_l) and its variance
            before the pair, the normalized innovation nu = sign(J) r / sqrt(S) and the log-likelihood;
            plus "late", the candidate index of each pair's later reading (in processing order)
        """
        y = (observations @ self.M) / self.M_col_sums  # (B, P)
        times = np.asarray(times, dtype=float)
        half = len(actions) // 2
        pairs = [(i, i + half) if times[i] <= times[i + half] else (i + half, i) for i in range(half)]
        pairs.sort(key=lambda pr: times[pr[1]])
        out = {k: np.empty((half, self.num_params)) for k in ("p_pred", "v_pred", "nu", "loglik")}
        out["late"] = np.array([l for _, l in pairs])
        for n, (e, l) in enumerate(pairs):
            self.model.predict(self.state, float(times[l]))
            p_hat, v = self.model.moments(self.state)
            V_D = self.model.increment_var(float(times[e]), float(times[l]))
            dc = actions[e] - actions[l]
            mid = 0.5 * (actions[e] + actions[l])
            z_hat = self.C @ (2.0 * dc * (mid - p_hat))
            J = -2.0 * self.C_diag * dc
            shot_var = 2.0 * self.R0
            if self.rate_scaled_noise:
                level = lambda c: self.y0 + self.C @ ((c - p_hat) ** 2 + v)
                shot_var = self.R0 * (level(actions[e]) + level(actions[l])) / self.y0
            R = (shot_var + self.C2 @ (4.0 * dc ** 2 * v) - self.C_diag ** 2 * 4.0 * dc ** 2 * v
                 + self.C2 @ (4.0 * V_D * ((actions[e] - p_hat) ** 2 + v) + 2.0 * V_D ** 2))
            r = (y[e] - y[l]) - z_hat
            S = self.model.update(self.state, J, r, R)
            out["p_pred"][n], out["v_pred"][n] = p_hat, v
            out["nu"][n] = np.sign(J) * r / np.sqrt(S)
            out["loglik"][n] = -0.5 * (np.log(2.0 * np.pi * S) + r ** 2 / S)
        self.mu = self.model.forecast(self.state, float(times.max() if t_next is None else t_next))
        return out

    def get_policy_stats(self) -> Dict[str, np.ndarray]:
        """Returns the current policy mean and probe width."""
        return {"mu": self.mu.copy(), "sigma": self.sigma.copy()}
