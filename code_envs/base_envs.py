"""
Control Environment Base Classes
================================
A `ControlEnv` couples drifting control parameters to the error sources of a simulation
`Backend`. The backend owns everything about how errors become detection events (a stim
circuit, a detector error model, ...); the control env owns the parameters, their drift,
and the map from parameters to per-source error rates.

Parameters come in `ParamBlock`s (e.g. one per gate, one per qubit). Block parameter i
belongs to owner j and raises the error rate of source s with weight W[s, j], quadratically
in its mis-set e_i = p_i - p_opt_i(t):

    eps_s(t, p) = base_s + sum_i W[s, owner(i)] * Omega_i * unit_s * e_i^2

clipped to [base_s, max_error]. A block's `error_model` sets the unit:

  - "additive" (the default): unit_s = error_budget * scale_s, a fixed error budget per
    source kind. Parameters are then in tolerance units: a mis-set of 1 / sqrt(Omega_i)
    spends one budget, and with the default budget (the reference base rate) Omega = 1 makes
    a mis-set of 1 double the error and one of 0.3 add about 10%. The budget does not move
    with base_error, so lowering the floor doesn't make a mis-set any cheaper. This is the
    form of a coherent control error, eps = eps_0 + kappa e^2 (Wu et al., arXiv:2610.11477,
    App. D; Magann et al., arXiv:2512.07815).
  - "relative": unit_s = base_s, so eps_s = base_s * (1 + sum ...). This is the form of a
    parameter that scales an existing error mechanism, e.g. a qubit frequency setting T1
    through nearby TLS defects.

Each source's base rate is base_error times the source's `scale` (e.g. its SI1000
multiplier) times a static spread U(0.8, 1.2). Each parameter's Omega_i is its owner's
sensitivity times a spread U(1 - s, 1 + s), so the sensitivity can be set per owner (e.g.
per gate kind), and W can weight one owner differently on each of its sources. A block's
optimum can sit off the t = 0 calibration (p = 0) by a static offset N(0, initial_offset^2)
per parameter, as for a calibration that is already hours old. The factor-graph matrix M in
{0, 1}^(O x P) follows from the backend's source footprint: parameter i sees every
detector any of its sources can flip.

The agent stays outside the environment: it only needs `num_params`, `num_detectors`,
`M` and `evaluate_policy`.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from code_envs.noise_envs import SinusoidalDrift

ERROR_MODELS = ("relative", "additive")


@dataclass(frozen=True)
class Source:
    """An independently scaled error source of a backend."""
    kind: str  # e.g. "2q" for a coupler, "meas" for a qubit's readout, "idle" for its idling
    qubits: Tuple[int, ...]
    scale: float = 1.0  # base rate as a multiple of the env's base_error


class Backend:
    """Base class turning per-source error rates into detection events. Subclasses implement `sample`."""

    def __init__(self, sources: List[Source], footprint: np.ndarray, has_observables: bool = False):
        self.sources = sources
        self.footprint = footprint  # (num_detectors x num_sources), 1 if source s can flip detector j
        self.num_sources = len(sources)
        self.num_detectors = footprint.shape[0]
        self.has_observables = has_observables

    def prepare_decoder(self, eps: np.ndarray):
        """Builds the decoder's priors from nominal error rates eps. No-op for backends that don't decode."""

    def sample(self, eps: np.ndarray, shots: int, decode: bool) -> Tuple[np.ndarray, Optional[float]]:
        """Samples `shots` shots at error rates eps; returns the per-detector EDR and the LER (None unless decode)."""
        raise NotImplementedError


@dataclass
class ParamBlock:
    """A group of drifting control parameters, `params_per_owner` for each owner (a column of `weights`)."""
    weights: np.ndarray  # (num_sources x num_owners): how strongly a mis-set owner raises each source's error rate
    params_per_owner: int = 1
    drift: object = field(default_factory=SinusoidalDrift)  # SinusoidalDrift, ArmaDrift, or any object with `build`
    sensitivity: Union[float, np.ndarray] = 1.0  # Omega scale, one value or one per owner
    sensitivity_spread: float = 0.4  # each parameter draws Omega_i = sensitivity * U(1 - spread, 1 + spread)
    owners: Optional[list] = None  # optional labels, one per owner (e.g. the qubit index)
    error_model: str = "relative"  # "relative" (of the base rate) or "additive" (in units of the error budget)
    initial_offset: float = 0.0  # rms static offset of p_opt from the t = 0 calibration p = 0

    @property
    def num_params(self) -> int:
        return self.weights.shape[1] * self.params_per_owner

    @classmethod
    def per_source(
        cls,
        sources: List[Source],
        kinds: Optional[Tuple[str, ...]] = None,
        sensitivity: Union[float, Dict[str, float]] = 1.0,
        **kwargs,
    ) -> "ParamBlock":
        """
        One owner per source (only those of the given kinds, if any), raising only its own
        error rate. `sensitivity` is one Omega scale, or one per source kind.
        """
        cols = [s for s, src in enumerate(sources) if kinds is None or src.kind in kinds]
        weights = np.zeros((len(sources), len(cols)), dtype=np.float32)
        weights[cols, np.arange(len(cols))] = 1.0
        if isinstance(sensitivity, dict):
            sensitivity = np.array([sensitivity[sources[s].kind] for s in cols], dtype=np.float32)
        return cls(weights=weights, owners=[sources[s] for s in cols], sensitivity=sensitivity, **kwargs)

    @classmethod
    def per_qubit(cls, sources: List[Source], kind_weights: Optional[Dict[str, float]] = None, **kwargs) -> "ParamBlock":
        """One owner per qubit, raising every source on that qubit with weight kind_weights[source kind] (default 1)."""
        kind_weights = kind_weights or {}
        qubits = sorted({q for src in sources for q in src.qubits})
        col = {q: j for j, q in enumerate(qubits)}
        weights = np.zeros((len(sources), len(qubits)), dtype=np.float32)
        for s, src in enumerate(sources):
            for q in src.qubits:
                weights[s, col[q]] = kind_weights.get(src.kind, 1.0)
        return cls(weights=weights, owners=qubits, **kwargs)


class ControlEnv:
    """Links drifting control parameters to the error sources of any simulation backend."""

    def __init__(
        self,
        backend: Backend,
        blocks: List[ParamBlock],
        base_error: float = 0.002,
        max_error: float = 0.15,
        seed: int = 163,
        error_budget: float = 0.002,  # error a unit mis-set adds to a scale-1 source under the additive model
    ):
        for block in blocks:
            if block.error_model not in ERROR_MODELS:
                raise ValueError(f"Unknown error_model '{block.error_model}' (use {ERROR_MODELS})")
        self.backend = backend
        self.blocks = blocks
        self.base_error = base_error
        self.max_error = max_error
        self.error_budget = error_budget
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.num_detectors = backend.num_detectors

        # Parameter layout: blocks in order, each owner-major
        self.block_slices: List[slice] = []
        start = 0
        for block in blocks:
            self.block_slices.append(slice(start, start + block.num_params))
            start += block.num_params
        self.num_params = start
        if self.num_params == 0:
            raise ValueError("the parameter blocks are all empty, so there is nothing to control")

        # Baseline irreducible error per source, then each block's sensitivities Omega_i and drift
        scales = np.array([src.scale for src in backend.sources], dtype=np.float32)
        self.param_base_errors = (base_error * scales * self.rng.uniform(0.8, 1.2, size=backend.num_sources)).astype(np.float32)
        sens, self._drift_opts = [], []
        for block in blocks:
            owner_sens = np.broadcast_to(np.asarray(block.sensitivity, dtype=np.float32), (block.weights.shape[1],))
            spread = block.sensitivity_spread
            sens.append((np.repeat(owner_sens, block.params_per_owner)
                         * self.rng.uniform(1.0 - spread, 1.0 + spread, size=block.num_params)).astype(np.float32))
            self._drift_opts.append(block.drift.build(block.num_params, block.params_per_owner, self.rng)
                                    if block.num_params else lambda t: np.zeros(0, dtype=np.float32))
        self.param_sensitivities = np.concatenate(sens)

        # Static offsets of the optimum from the t = 0 calibration, from their own stream so the drift stays put
        offset_rng = np.random.default_rng([self.seed, 1])
        self.initial_offsets = np.concatenate([offset_rng.normal(0.0, b.initial_offset, size=b.num_params)
                                               for b in blocks]).astype(np.float32)

        # W[s, i] = weight of parameter i's owner on source s; unit[s, i] = error per unit W * Omega_i * e^2 on s
        W = np.concatenate([np.repeat(b.weights, b.params_per_owner, axis=1) for b in blocks], axis=1)
        relative = np.concatenate([np.full(b.num_params, b.error_model == "relative") for b in blocks])
        budgets = (error_budget * scales).astype(np.float32)
        self._unit = np.where(relative[None, :], self.param_base_errors[:, None], budgets[:, None])
        self._build_curvature(W)

        # M[j, i] = 1 if detector j is in the detecting region of parameter i
        self.M = ((backend.footprint @ (W > 0)) > 0).astype(np.float32)

        # The decoder assumes the static policy (params = 0) at t = 0
        backend.prepare_decoder(self.compute_physical_error_rates(np.zeros(self.num_params, dtype=np.float32), 0.0))

    def scale_sensitivities(self, factors: np.ndarray):
        """Multiplies each Omega_i by factors[i], keeping the drawn sensitivities and drift realization."""
        self.param_sensitivities = (self.param_sensitivities * np.asarray(factors, dtype=np.float32)).astype(np.float32)
        self._build_curvature(np.concatenate([np.repeat(b.weights, b.params_per_owner, axis=1) for b in self.blocks], axis=1))
        self.backend.prepare_decoder(self.compute_physical_error_rates(np.zeros(self.num_params, dtype=np.float32), 0.0))

    def _build_curvature(self, W: np.ndarray):
        """curvature[s, i] = d eps_s / d e_i^2 = W[s, i] * Omega_i * unit[s, i], in absolute error per unit squared mis-set."""
        self.curvature = (W * self.param_sensitivities[None, :] * self._unit).astype(np.float32)

    def get_drift_opt(self, t: float) -> np.ndarray:
        """Returns the ground-truth optimal parameter vector at time t (epochs)."""
        return np.concatenate([p_opt(t) for p_opt in self._drift_opts]) + self.initial_offsets

    def compute_physical_error_rates(self, params: np.ndarray, t: float) -> np.ndarray:
        """
        Maps control parameters p to per-source error rates eps_s(t, p) (see module docstring);
        under the default relative model eps_s = base_s * (1 + sum_i W[s, i] Omega_i (p_i - p_opt_i(t))^2).
        """
        dev = params - self.get_drift_opt(t)
        return np.clip(self.param_base_errors + self.curvature @ (dev ** 2), self.param_base_errors, self.max_error)

    def evaluate_policy(
        self,
        params: np.ndarray,
        t: float,
        shots: int = 1000,
        decode: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Samples `shots` detection events with the error rates induced by `params` at (static) time t.

        Returns:
            dict containing:
                'edr_vector': empirical per-detector error detection rate o in [0, 1]^O
                'mean_edr': scalar average EDR
                'logical_error_rate': decoded LER (if decode=True, else None)
        """
        edr_vector, ler = self.backend.sample(self.compute_physical_error_rates(params, t), shots, decode)
        return {
            "edr_vector": edr_vector,
            "mean_edr": float(np.mean(edr_vector)),
            "logical_error_rate": ler,
        }
