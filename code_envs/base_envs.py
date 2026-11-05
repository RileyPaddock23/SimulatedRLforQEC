"""
Control Environment Base Classes
================================
A `ControlEnv` couples drifting control parameters to the error sources of a simulation
`Backend`. The backend owns everything about how errors become detection events (a stim
circuit, a detector error model, ...); the control env owns the parameters, their drift,
and the map from parameters to per-source error rates.

Parameters come in `ParamBlock`s (e.g. one per gate, one per qubit). Block parameter i
belongs to owner j and raises the error rate of source s with weight W[s, j]:

    eps_s(t, p) = base_s + sum_i W[s, owner(i)] * Omega_i * (p_i - p_opt_i(t))^2

clipped to [base_s, max_error]. The factor-graph matrix M in {0, 1}^(O x P) follows from
the backend's source footprint: parameter i sees every detector any of its sources can flip.

The agent stays outside the environment: it only needs `num_params`, `num_detectors`,
`M` and `evaluate_policy`.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from code_envs.noise_envs import SinusoidalDrift


@dataclass(frozen=True)
class Source:
    """An independently scaled error source of a backend."""
    kind: str  # e.g. "2q" for a coupler, "1q" for a qubit's single-qubit / reset / readout channels
    qubits: Tuple[int, ...]


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
    sensitivity: float = 0.025  # Omega scale; each parameter draws Omega_i = sensitivity * U(0.6, 1.4)
    owners: Optional[list] = None  # optional labels, one per owner (e.g. the qubit index)

    @property
    def num_params(self) -> int:
        return self.weights.shape[1] * self.params_per_owner

    @classmethod
    def per_source(cls, sources: List[Source], **kwargs) -> "ParamBlock":
        """One owner per source, raising only its own error rate."""
        return cls(weights=np.eye(len(sources), dtype=np.float32), owners=list(sources), **kwargs)

    @classmethod
    def per_qubit(cls, sources: List[Source], weight_2q: float = 1.0, weight_1q: float = 1.0, **kwargs) -> "ParamBlock":
        """One owner per qubit, raising every source on that qubit (weight_2q for couplers, weight_1q otherwise)."""
        qubits = sorted({q for src in sources for q in src.qubits})
        col = {q: j for j, q in enumerate(qubits)}
        weights = np.zeros((len(sources), len(qubits)), dtype=np.float32)
        for s, src in enumerate(sources):
            for q in src.qubits:
                weights[s, col[q]] = weight_2q if len(src.qubits) == 2 else weight_1q
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
    ):
        self.backend = backend
        self.blocks = blocks
        self.base_error = base_error
        self.max_error = max_error
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
        self.param_base_errors = (base_error * self.rng.uniform(0.8, 1.2, size=backend.num_sources)).astype(np.float32)
        sens, self._drift_opts = [], []
        for block in blocks:
            sens.append((block.sensitivity * self.rng.uniform(0.6, 1.4, size=block.num_params)).astype(np.float32))
            self._drift_opts.append(block.drift.build(block.num_params, block.params_per_owner, self.rng)
                                    if block.num_params else lambda t: np.zeros(0, dtype=np.float32))
        self.param_sensitivities = np.concatenate(sens)

        # W[s, i] = weight of parameter i's owner on source s; A[s, i] = W[s, i] * Omega_i
        W = np.concatenate([np.repeat(b.weights, b.params_per_owner, axis=1) for b in blocks], axis=1)
        self._A = W * self.param_sensitivities[None, :]

        # M[j, i] = 1 if detector j is in the detecting region of parameter i
        self.M = ((backend.footprint @ (W > 0)) > 0).astype(np.float32)

        # The decoder assumes the static policy (params = 0) at t = 0
        backend.prepare_decoder(self.compute_physical_error_rates(np.zeros(self.num_params, dtype=np.float32), 0.0))

    def get_drift_opt(self, t: float) -> np.ndarray:
        """Returns the ground-truth optimal parameter vector at time t (epochs)."""
        return np.concatenate([p_opt(t) for p_opt in self._drift_opts])

    def compute_physical_error_rates(self, params: np.ndarray, t: float) -> np.ndarray:
        """Maps control parameters p to per-source error rates eps_s(t, p) (see module docstring)."""
        dev = params - self.get_drift_opt(t)
        eps = self.param_base_errors + self._A @ (dev ** 2)
        return np.clip(eps, self.param_base_errors, self.max_error)

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
