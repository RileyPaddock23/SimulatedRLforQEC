"""
Composite Drift Model
=====================
Literature-anchored drift of a phase-type control parameter (e.g. a CZ post-phase or
idle Z correction, which tracks the qubit frequency over a ~1 us QEC cycle), built from
independent components that can each be included, excluded, or rescaled.

Two outputs per time step and parameter:
  p_star     optimum of the control parameter, in units where an offset e adds Omega e^2
             to the gate error. A qubit-frequency shift df moves it by 2 pi df t_acc, and
             with error ~ theta^2 / 4 one unit is theta = 2 sqrt(Omega) rad, i.e.
             khz_per_unit = 2 sqrt(Omega) / (2 pi t_acc) ~ 50 kHz (Omega = 0.025, t_acc = 1 us).
  eps0_mult  multiplier of the uncontrollable base error eps0. A fraction t1_fraction of eps0
             is T1-limited at T1 = t1_us; extra relaxation rates scale that part.

Components (DEFAULTS holds every parameter; the numbers come from the cited papers unless
marked "assumption"):
  background  1/f^alpha qubit-frequency noise as a sum of Ornstein-Uhlenbeck processes with
              log-spaced rates. alpha = 1.1 [Schlor PRL 2019]. Level: in Schlor Fig. 3 the 1/f part
              at 1 mHz is ~0.075 x the single-fluctuator plateau; for 5 kHz jumps at 1 mHz that is
              S(1 mHz) ~ 1e9 Hz^2/Hz (~3 kHz rms over a day)
  telegraph   two-level fluctuators switching the qubit frequency by 5-140 kHz, rates
              log-uniform (1/Gamma distributed) in 50 uHz - 5 mHz        [Schlor 2019; Klimov PRL 2018]
  tls         TLS defects near the qubit frequency (1 per +-50 MHz, ~ the density in Klimov 2018).
              Each diffuses (D = 2.5 MHz/sqrt(hr)) and jumps telegraphically (1-60 MHz,
              50 uHz - 5 mHz), with coupling g = 50-500 kHz and linewidth 0.5-20 MHz [Klimov 2018].
              Effects: dispersive shift g^2 D / (D^2 + gamma^2) on p_star, resonant relaxation
              4 pi g^2 gamma / (gamma^2 + D^2) on eps0_mult, capped so T1 drops at most t1_drop_max = 10x
              ("T1 can vary by up to an order of magnitude", Klimov 2018).
  scrambling  Poisson events that redraw every TLS frequency at once
              [Thorbeck PRX Quantum 2023; rate: assumption, 1 per hour]
  bursts      high-energy impacts: Poisson rate 1 / 10 s, chip-wide T1 drop to 0.7-15 us
              decaying with 25 ms                                         [McEwen Nat. Phys. 2021]
  diurnal     slow sinusoidal drift (control-electronics temperature), 15 kHz, 24 h period
              [assumption; drift sources named in arXiv:2511.08493]

resolution="candidate" returns one value per candidate: bursts are averaged exactly over
the candidate, the slower components are taken at the candidate midpoint. resolution="shot" returns
one value per shot. Events are chip-wide (shared by all parameters) unless chip_wide=False,
which makes every column an independent chip.

Example:
    drift = CompositeDrift(duration_s=3600, components=("background", "telegraph", "bursts"))
    out = drift.generate(n_params=8)   # out["t"], out["p_star"], out["eps0_mult"]
"""

import copy
from typing import Dict, Optional, Sequence

import numpy as np

COMPONENTS = ("background", "telegraph", "tls", "scrambling", "bursts", "diurnal")

DEFAULTS: Dict[str, Dict] = {
    "background": dict(alpha=1.1, S_1mHz=1e9, per_decade=2),  # S in Hz^2/Hz at 1 mHz
    "telegraph": dict(n=2, amp_khz=(5.0, 140.0), rate_hz=(5e-5, 5e-3)),
    "tls": dict(n=1, window_mhz=50.0, g_khz=(50.0, 500.0), gamma_mhz=(0.5, 20.0), D_mhz_sqrt_hr=2.5,
                jump_mhz=(1.0, 60.0), jump_rate_hz=(5e-5, 5e-3), t1_drop_max=10.0),
    "scrambling": dict(rate_hz=1.0 / 3600.0),
    "bursts": dict(rate_hz=0.1, tau_s=0.025, t1_peak_us=(0.7, 15.0)),
    "diurnal": dict(amp_khz=15.0, period_s=86400.0),
}


def _log_uniform(rng, lo_hi, size):
    lo, hi = lo_hi
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def _telegraph(rng, rate_hz, dt, n_steps):
    """Symmetric +-1 telegraph switching at rate_hz in each direction, shape (n_steps,)."""
    flips = rng.random(n_steps) < 0.5 * (1.0 - np.exp(-2.0 * rate_hz * dt))
    return rng.choice([-1.0, 1.0]) * (1.0 - 2.0 * (np.cumsum(flips) % 2))


def _poisson_times(rng, rate_hz, duration_s):
    n = rng.poisson(rate_hz * duration_s)
    return np.sort(rng.uniform(0.0, duration_s, n))


class CompositeDrift:
    def __init__(
        self,
        duration_s: float = 86400.0,
        candidate_s: float = 0.1,
        shots_per_candidate: int = 50,
        resolution: str = "candidate",  # "candidate" (one value per candidate) or "shot"
        components: Sequence[str] = COMPONENTS,
        params: Optional[Dict[str, Dict]] = None,  # per-component overrides of DEFAULTS
        strength: float = 1.0,  # multiplies every amplitude (frequency shifts and extra relaxation)
        omega: float = 0.025,  # gate-error curvature of the control parameter
        t_acc_s: float = 1e-6,  # phase accumulation time (one QEC cycle)
        t1_us: float = 50.0,  # baseline T1 (assumption)
        t1_fraction: float = 0.5,  # fraction of eps0 that is T1-limited (assumption)
        chip_wide: bool = True,
        seed: int = 0,
    ):
        unknown = set(components) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown components {unknown} (choose from {COMPONENTS})")
        if "scrambling" in components and "tls" not in components:
            raise ValueError("'scrambling' redraws TLS frequencies, so it needs 'tls'")
        if resolution not in ("candidate", "shot"):
            raise ValueError("resolution must be 'candidate' or 'shot'")
        self.duration_s, self.candidate_s, self.shots_per_candidate = duration_s, candidate_s, shots_per_candidate
        self.resolution, self.components = resolution, tuple(components)
        self.p = copy.deepcopy(DEFAULTS)
        for name, over in (params or {}).items():
            self.p[name].update(over)
        self.strength, self.chip_wide = float(strength), chip_wide
        self.khz_per_unit = 2.0 * np.sqrt(omega) / (2.0 * np.pi * t_acc_s) / 1e3
        self.gamma1_base, self.t1_fraction = 1e6 / t1_us, t1_fraction
        self.rng = np.random.default_rng(seed)

        # Background as a sum of OU processes: rates log-spaced over the run's band
        b = self.p["background"]
        f_lo, f_hi = 1.0 / duration_s, 0.5 / candidate_s
        n_k = int(np.ceil(np.log10(f_hi / f_lo) * b["per_decade"])) + 1
        self.bg_rates = 2.0 * np.pi * np.geomspace(f_lo, f_hi, n_k)  # 1/s
        psd = lambda f, v: np.sum(4.0 * v * self.bg_rates / (self.bg_rates ** 2 + (2.0 * np.pi * f) ** 2))
        v = self.bg_rates ** (1.0 - b["alpha"])
        self.bg_var_khz2 = v * b["S_1mHz"] * 1e-6 / psd(1e-3, v) * self.strength ** 2  # kHz^2

    # ------------------------------------------------------------------ helpers
    def background_model(self):
        """(taus in s, variances in units^2) of the background: the exact calm drift as a sum of AR(1)s."""
        return 1.0 / self.bg_rates, self.bg_var_khz2 / self.khz_per_unit ** 2

    def times(self) -> np.ndarray:
        n_candidates = int(round(self.duration_s / self.candidate_s))
        if self.resolution == "candidate":
            return (np.arange(n_candidates) + 0.5) * self.candidate_s
        dt = self.candidate_s / self.shots_per_candidate
        return (np.arange(n_candidates * self.shots_per_candidate) + 0.5) * dt

    # ------------------------------------------------------------------ generation
    def generate(self, n_params: int = 1) -> Dict[str, np.ndarray]:
        """Returns t (s), p_star (units) and eps0_mult, the latter two of shape (n_steps, n_params)."""
        t = self.times()
        n, dt, rng, s = len(t), t[1] - t[0], self.rng, self.strength
        shift_khz = np.zeros((n, n_params))
        gamma_extra = np.zeros((n, n_params))  # extra relaxation rate, 1/s
        n_chips = 1 if self.chip_wide else n_params
        chip_col = (lambda x: np.repeat(x, n_params, axis=1)) if self.chip_wide else (lambda x: x)

        if "background" in self.components:
            a = np.exp(-self.bg_rates * dt)[:, None]
            q = np.sqrt(self.bg_var_khz2 * (1.0 - a[:, 0] ** 2))[:, None]
            x = np.sqrt(self.bg_var_khz2)[:, None] * rng.standard_normal((len(self.bg_rates), n_params))
            for k in range(n):
                x = a * x + q * rng.standard_normal(x.shape)
                shift_khz[k] += x.sum(axis=0)

        if "telegraph" in self.components:
            c = self.p["telegraph"]
            for col in range(n_params):
                for _ in range(c["n"]):
                    amp = _log_uniform(rng, c["amp_khz"], 1)[0] * s
                    shift_khz[:, col] += 0.5 * amp * _telegraph(rng, _log_uniform(rng, c["rate_hz"], 1)[0], dt, n)

        if "tls" in self.components:
            c = self.p["tls"]
            W = c["window_mhz"]
            # Scrambling times are chip-wide: shared by every TLS on the chip
            scramble = [_poisson_times(rng, self.p["scrambling"]["rate_hz"], self.duration_s)
                        if "scrambling" in self.components else np.array([]) for _ in range(n_chips)]
            for col in range(n_params):
                for _ in range(c["n"]):
                    walk = np.cumsum(rng.standard_normal(n) * 2.0 * c["D_mhz_sqrt_hr"] * np.sqrt(dt / 3600.0))
                    jumps = 0.5 * _log_uniform(rng, c["jump_mhz"], 1)[0] * _telegraph(
                        rng, _log_uniform(rng, c["jump_rate_hz"], 1)[0], dt, n)
                    # Uniform offset, redrawn at every scrambling event
                    seg = np.searchsorted(scramble[0 if self.chip_wide else col], t)
                    offset = rng.uniform(-W, W, seg.max() + 1)[seg]
                    delta = (offset + walk + jumps + W) % (2.0 * W) - W  # TLS - qubit detuning, MHz
                    g2 = (_log_uniform(rng, c["g_khz"], 1)[0] * 1e-3) ** 2 * s  # MHz^2
                    gam = _log_uniform(rng, c["gamma_mhz"], 1)[0]
                    lor = 1.0 / (delta ** 2 + gam ** 2)
                    shift_khz[:, col] += 1e3 * g2 * delta * lor  # dispersive shift
                    gamma_extra[:, col] += np.minimum(1e6 * 4.0 * np.pi * g2 * gam * lor,  # resonant relaxation
                                                      (c["t1_drop_max"] - 1.0) * self.gamma1_base)

        if "bursts" in self.components:
            c = self.p["bursts"]
            tau, burst = c["tau_s"], np.zeros((n, n_chips))
            span = int(np.ceil(12.0 * tau / dt)) + 1  # steps a burst lasts
            for chip in range(n_chips):
                te = _poisson_times(rng, c["rate_hz"], self.duration_s)
                peak = 1e6 / _log_uniform(rng, c["t1_peak_us"], len(te)) * s
                k = (te // dt).astype(int)[:, None] + np.arange(span)[None, :]  # step k covers [k dt, (k+1) dt)
                lo = np.maximum(k * dt, te[:, None]) - te[:, None]
                if self.resolution == "candidate":  # exact average over the candidate
                    val = tau * (np.exp(-lo / tau) - np.exp(-((k + 1) * dt - te[:, None]) / tau)) / dt
                else:  # value at the shot time
                    val = np.exp(-(t[np.minimum(k, n - 1)] - te[:, None]) / tau) * (t[np.minimum(k, n - 1)] >= te[:, None])
                ok = k < n
                np.add.at(burst[:, chip], k[ok], (peak[:, None] * val)[ok])
            gamma_extra += chip_col(burst)

        if "diurnal" in self.components:
            c = self.p["diurnal"]
            phase = rng.uniform(0.0, 2.0 * np.pi, n_params)
            shift_khz += c["amp_khz"] * s * np.sin(2.0 * np.pi * t[:, None] / c["period_s"] + phase)

        eps0_mult = 1.0 + self.t1_fraction * gamma_extra / self.gamma1_base
        return {"t": t, "p_star": (shift_khz / self.khz_per_unit).astype(np.float32),
                "eps0_mult": eps0_mult.astype(np.float32)}
