"""
Generic QEC Code Environment with Drift Injection
=================================================
Stim-based environment for any QEC memory experiment given as a stim circuit with
DETECTORs and OBSERVABLE_INCLUDEs. `CircuitBackend` places SI1000-style error sources
(see noise_envs.py): one per two-qubit coupler, and per qubit one each for its
single-qubit gates, resets, measurements, and idling. Every copy of a gate in the circuit
(e.g. the H on qubit 2 in every round) feeds the same source, so it has one error rate
and one set of parameters. `QECCodeEnv` is the `ControlEnv` (see base_envs.py) that
steers them.

Each gate source g (coupler, single-qubit gates, reset, or readout of a qubit) has control
parameters p_g (gate time, pulse amplitude, ...) whose optimum p_opt_g(t) drifts in time.
Mis-set parameters scale that source's base error rate, quadratically in the mis-set (the
"relative" model, see base_envs.py):

    eps_g(t, p_g) = base_g * (1 + sum_m Omega_{g,m} * (p_{g,m} - p_opt_{g,m}(t))^2)

where base_g = base_error * scale_g * U(0.8, 1.2), with scale_g the SI1000 multiplier of g's
kind (1 for a coupler, 1/10 for single-qubit gates, 2 for resets, 5 for readout, see
NoiseModel). Gate parameters are in tolerance units: with Omega = 1 a mis-set of 1
doubles the source's error rate and one of 0.3 adds about 10%, for every kind.

Each qubit q also has its own parameters p_q (e.g. its idle frequency, whose optimum
drifts as TLS defects wander and which sets T1). A mis-set qubit parameter scales the error
rate of every source on q, idling included, on top of the gate term:

    eps_g(t, p) = base_g * (1 + sum_m Omega_{g,m} * (p_{g,m} - p_opt_{g,m}(t))^2
                              + sum_{q in g} sum_m Omega_{q,m}^{kind(g)} * (p_{q,m} - p_opt_{q,m}(t))^2)

Idle sources have no gate term, only the qubit term. Either block can instead use the
"additive" model, eps_g = base_g + error_budget * scale_g * Omega * e^2, whose penalty
doesn't shrink with base_error (`error_model`, `qubit_error_model`).

Sensitivities are set per parameter type: `sensitivity` is the Omega scale of the gate
parameters of each gate kind ("2q", "1q", "reset", "meas"), and `qubit_sensitivity` the
Omega scale of a qubit parameter on each kind of source it touches (those plus "idle" and
"idle_mr"), so T1 drift can hit a qubit's idling harder than its gates. Either can be one
value for every kind; kinds left out of a dict get DEFAULT_SENSITIVITY. Each parameter then
draws its own factor U(1 - sensitivity_spread, 1 + sensitivity_spread) around its type's
scale; a qubit parameter draws one factor, shared by all its sources.

Under the static policy, a gate parameter with sinusoidal drift of amplitude A swings its
source between base and base + budget * Omega * A^2, so a measured swing d_eps gives
Omega = d_eps / (budget * A^2). On top of the drift, each parameter's optimum starts off
the t = 0 calibration by a static offset of rms `initial_offset`: the calibration the fixed
policy keeps is already stale (Wu et al. measure roughly 1 tolerance of drift, in these
units, over a 6 h recalibration interval for CZ coupling, CZ phase and qubit frequency).

The quadratic is a local model around a single optimum: the real T1(f) landscape has a
narrow dip at each TLS, so it doesn't capture a TLS crossing the qubit or the dephasing /
frequency-collision cost of moving far from the sweet spot.

The pluggable pieces are:
  - the circuit:     any noiseless stim circuit (existing noise channels are stripped);
                     idle noise needs its moments separated by TICKs
  - the noise model: `NoiseModel`, where the drift-controlled channels sit and their base rates
  - the drifts:      `SinusoidalDrift`, `ArmaDrift`, or any object with the same `build`,
                     one for the gate parameters and one for the qubit parameters
plus the decoder used for logical error rates ("pymatching", "bposd", or a factory).
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import stim
import pymatching

from code_envs.base_envs import Backend, ControlEnv, ParamBlock, Source
from code_envs.noise_envs import NoiseModel, SinusoidalDrift, flip_channel, make_drift


# ---------------------------------------------------------------------- decoders

def _make_decoder(decoder: Union[str, Callable[[stim.Circuit], object]], circuit: stim.Circuit):
    """Builds a decoder exposing `decode_batch(det_shots) -> predicted_obs` from the nominal noisy circuit."""
    if callable(decoder):
        return decoder(circuit)
    if decoder == "pymatching":
        return pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    if decoder == "bposd":
        # https://github.com/oscarhiggott/stimbposd a BP+OSD decoder for use with stim!
        import stimbposd
        # BP+OSD takes the undecomposed DEM, so it also works for codes without a matching structure (e.g. color codes)
        return stimbposd.BPOSD(circuit.detector_error_model(), max_bp_iters=300)
    raise ValueError(f"Unknown decoder '{decoder}' (use 'pymatching', 'bposd', or a callable circuit -> decoder)")


# ---------------------------------------------------------------------- backend

# Source kinds that have gate parameters; idle sources only see their qubit's parameters
GATE_KINDS = ("2q", "1q", "reset", "meas")
_KIND_ORDER = ("2q", "1q", "reset", "meas", "idle", "idle_mr")

DEFAULT_SENSITIVITY = 1.0  # a unit mis-set spends one error budget (additive) or doubles the base rate (relative)
DEFAULT_INITIAL_OFFSET = 1.0  # rms stale-calibration offset, in tolerance units
Sensitivity = Union[float, Dict[str, float]]  # one Omega scale, or one per source kind


def _by_kind(sensitivity: Sensitivity, kinds: Tuple[str, ...]) -> Dict[str, float]:
    """Expands a sensitivity to one Omega scale per kind; kinds a dict leaves out get DEFAULT_SENSITIVITY."""
    if not isinstance(sensitivity, dict):
        return {kind: float(sensitivity) for kind in kinds}
    unknown = set(sensitivity) - set(kinds)
    if unknown:
        raise ValueError(f"unknown source kinds {sorted(unknown)} in sensitivity (use {kinds})")
    return {kind: float(sensitivity.get(kind, DEFAULT_SENSITIVITY)) for kind in kinds}


class CircuitBackend(Backend):
    """
    Backend sampling a stim circuit with SI1000-style error sources (see `NoiseModel`).

    A source is keyed by (kind, qubits), so all copies of a gate share one source. Source
    ids run kind by kind in `_KIND_ORDER`: two-qubit couplers first (0 .. num_cx_gates - 1,
    in order of first appearance), then the single-qubit kinds sorted by qubit. The `cx_*`
    names are kept for compatibility but cover every two-qubit gate in
    `NoiseModel.two_qubit_gates`, not only CX. A circuit without OBSERVABLE_INCLUDEs still
    gives detection events, but can't be decoded.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        noise_model: Optional[NoiseModel] = None,
        decoder: Union[str, Callable[[stim.Circuit], object]] = "pymatching",
    ):
        self.noise_model = noise_model if noise_model is not None else NoiseModel()
        self.include_single_qubit_noise = self.noise_model.include_single_qubit_noise
        self._decoder_spec = decoder
        self.decoder = None

        # The backend supplies all noise itself
        self.base_circuit = circuit.without_noise()
        self.flat_instructions = self.base_circuit.flattened()
        self.num_observables = self.base_circuit.num_observables
        if self.base_circuit.num_detectors == 0:
            raise ValueError("circuit has no DETECTORs, so there is nothing for the agent to observe")

        sources = self._setup_fast_circuit_template()
        super().__init__(sources, self._compute_footprint(len(sources)), has_observables=self.num_observables > 0)

    @staticmethod
    def _qubit_pairs(op: stim.CircuitInstruction) -> List[Tuple[int, int]]:
        """Target pairs of a two-qubit gate, skipping classically controlled ones (rec / sweep targets)."""
        targets = op.targets_copy()
        return [
            (targets[k].value, targets[k + 1].value)
            for k in range(0, len(targets), 2)
            if targets[k].is_qubit_target and targets[k + 1].is_qubit_target
        ]

    def _acts(self, name: str) -> bool:
        """Whether gate `name` gets noise channels under the noise model."""
        return any(name in gates for gates in self.noise_model.resolved())

    def _place_channels(self, flat_ops, all_qubits: List[int]) -> List[Tuple[str, Optional[Tuple[str, Tuple[int, ...]]]]]:
        """
        Walks flattened instructions moment by moment (moments are separated by TICKs),
        placing each source's channels. Returns (line, source key) pairs; key None marks a
        noiseless line from the base circuit. `all_qubits` are the qubits that can idle.
        """
        two_qubit, single_qubit, reset, measure = self.noise_model.resolved()
        scales = self.noise_model.scales()
        acts = lambda name: name in two_qubit or name in single_qubit or name in reset or name in measure
        lines: List[Tuple[str, Optional[Tuple[str, Tuple[int, ...]]]]] = []

        def add(kind: str, channel: str, qubits: Tuple[int, ...]):
            if kind in scales:
                lines.append((f"{channel}({{:.8g}}) {' '.join(map(str, qubits))}\n", (kind, qubits)))

        def qubit_targets(op) -> List[int]:
            return [t.value for t in op.targets_copy() if t.is_qubit_target]

        touched, readout_moment = set(), False

        def end_moment():
            # Qubits the moment left alone idle through it, for longer if it holds a readout or reset
            if touched:
                for q in all_qubits:
                    if q not in touched:
                        add("idle_mr" if readout_moment else "idle", "DEPOLARIZE1", (q,))

        for op in flat_ops:
            if op.name == "TICK":
                end_moment()
                touched, readout_moment = set(), False
                lines.append(("TICK\n", None))
                continue

            # 1. Readout error: a classical flip before the measurement
            if op.name in measure:
                for q in qubit_targets(op):
                    add("meas", flip_channel(op.name), (q,))

            lines.append((str(op) + "\n", None))

            # 2. Gate / reset error after the operation
            if op.name in two_qubit:
                for pair in self._qubit_pairs(op):
                    add("2q", "DEPOLARIZE2", pair)
            elif op.name in reset:
                for q in qubit_targets(op):
                    add("reset", flip_channel(op.name), (q,))
            elif op.name in single_qubit:
                for q in qubit_targets(op):
                    add("1q", "DEPOLARIZE1", (q,))

            if acts(op.name):
                touched.update(qubit_targets(op))
                readout_moment |= op.name in reset or op.name in measure
        end_moment()
        return lines

    def _setup_fast_circuit_template(self) -> List[Source]:
        """
        Places each source's channels in the circuit and precomputes a circuit string
        template and source-id map so noisy circuits can be assembled with a single
        str.format call. Returns the sources.
        """
        scales = self.noise_model.scales()
        self.all_qubits = sorted({t.value for op in self.flat_instructions if self._acts(op.name)
                                  for t in op.targets_copy() if t.is_qubit_target})
        lines = self._place_channels(self.flat_instructions, self.all_qubits)

        # Source ids, kind by kind: couplers in order of first appearance, single-qubit kinds by qubit
        keys = list(dict.fromkeys(key for _, key in lines if key is not None))
        if not keys:
            raise ValueError("noise model places no channels in this circuit, so there are no parameters to control")
        keys.sort(key=lambda k: (_KIND_ORDER.index(k[0]), () if k[0] == "2q" else k[1]))
        self.source_ids: Dict[Tuple[str, Tuple[int, ...]], int] = {k: sid for sid, k in enumerate(keys)}

        self.cx_pairs: List[Tuple[int, int]] = [qubits for kind, qubits in keys if kind == "2q"]
        self.pair_to_id: Dict[Tuple[int, int], int] = {pair: sid for sid, pair in enumerate(self.cx_pairs)}
        self.num_cx_gates = len(self.cx_pairs)

        self._template_lines: List[Tuple[str, int]] = [(line, -1 if key is None else self.source_ids[key])
                                                       for line, key in lines]
        self._fast_template_str = "".join(line for line, _ in self._template_lines)
        self._template_source_ids = [sid for _, sid in self._template_lines if sid >= 0]
        # Noisy channel instances per source, i.e. how many operations in the circuit each source covers
        self.source_instances = np.bincount(self._template_source_ids, minlength=len(keys))
        return [Source(kind, qubits, scales[kind]) for kind, qubits in keys]

    def segment_template(self, segment: stim.Circuit) -> List[Tuple[str, int]]:
        """
        Noisy template lines (line, source id; -1 for a noiseless line) for a piece of the
        circuit (e.g. one QEC cycle), using this backend's source ids. Idle noise is placed
        as if the segment started and ended on a moment boundary; where that creates a
        source the whole circuit doesn't have (e.g. ancillas idling through a final data
        readout that, in the whole circuit, shares a moment with their own readout), the
        channel is left out.
        """
        lines = self._place_channels(segment.without_noise().flattened(), self.all_qubits)
        return [(line, -1 if key is None else self.source_ids[key]) for line, key in lines
                if key is None or key in self.source_ids]

    def _compute_footprint(self, num_sources: int, template_lines: Optional[List[Tuple[str, int]]] = None,
                           rows: Optional[slice] = None, fill_unobserved: bool = True) -> np.ndarray:
        """
        Source-level detector incidence matrix in {0, 1}^(O x num_sources): the noiseless
        circuit plus only source sid's channels, at a probe rate of 0.01. `template_lines`
        defaults to the whole circuit; `rows` keeps only those detectors.
        """
        template_lines = self._template_lines if template_lines is None else template_lines
        num_detectors = stim.Circuit("".join(line for line, s in template_lines if s < 0)).num_detectors
        footprint = np.zeros((num_detectors, num_sources), dtype=np.float32)

        for sid in range(num_sources):
            c = stim.Circuit("".join(
                line if s < 0 else line.format(0.01) for line, s in template_lines if s < 0 or s == sid
            ))
            dem = c.detector_error_model(decompose_errors=False)
            for instr in dem.flattened():
                if instr.type == "error":
                    for target in instr.targets_copy():
                        if target.is_relative_detector_id():
                            footprint[target.val, sid] = 1.0

        footprint = footprint[rows] if rows is not None else footprint
        # Safety check: ensure every source is observed by at least one detector
        if fill_unobserved:
            for sid in range(num_sources):
                if np.sum(footprint[:, sid]) == 0:
                    footprint[sid % footprint.shape[0], sid] = 1.0
        return footprint

    def build_circuit(self, eps: np.ndarray) -> stim.Circuit:
        """Constructs the stim circuit with per-source depolarizing rates eps."""
        return stim.Circuit(self._fast_template_str.format(*(eps[sid] for sid in self._template_source_ids)))

    def prepare_decoder(self, eps: np.ndarray):
        if self.has_observables:
            self.decoder = _make_decoder(self._decoder_spec, self.build_circuit(eps))

    def sample(self, eps: np.ndarray, shots: int, decode: bool) -> Tuple[np.ndarray, Optional[float]]:
        sampler = self.build_circuit(eps).compile_detector_sampler()

        if decode:
            if self.decoder is None:
                raise ValueError("circuit has no OBSERVABLE_INCLUDEs, so there is no logical error rate to decode")
            det_shots, actual_obs = sampler.sample(shots=shots, separate_observables=True)
            predicted_obs = self.decoder.decode_batch(det_shots)
            ler = float(np.mean(np.any(actual_obs != predicted_obs, axis=1)))
        else:
            det_shots = sampler.sample(shots=shots, separate_observables=False)
            ler = None

        return np.mean(det_shots, axis=0), ler  # EDR shape (O,)


# ---------------------------------------------------------------------- environment

class QECCodeEnv(ControlEnv):
    """
    Parameterized environment for an arbitrary stim QEC circuit with continuous
    hardware drift and local factor-graph masking.

    Time `t` is measured in (reference) epochs; the drift model defines what that means
    for its own parameters.

    Parameter layout: the gate block comes first (gate source g's m-th parameter at
    g * params_per_gate + m, see `gate_param_slice`; gate sources are the first source
    ids, so g is also the source id), then the qubit block (the m-th
    parameter of param_qubits[j] at num_gate_params + j * params_per_qubit + m, see
    `qubit_param_slice`). Either block can be switched off by setting its count to 0.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        decoder: Union[str, Callable[[stim.Circuit], object]] = "pymatching",
        noise_model: Optional[NoiseModel] = None,
        drift=None,  # gate-parameter drift: SinusoidalDrift (default), ArmaDrift, or any object with `build`
        base_error: float = 0.002,
        sensitivity: Sensitivity = DEFAULT_SENSITIVITY,  # gate-parameter Omega scale, per gate kind (GATE_KINDS)
        sensitivity_spread: float = 0.4,  # each parameter's Omega is its type's scale * U(1 - spread, 1 + spread)
        params_per_gate: int = 1,
        seed: int = 163,
        # Qubit parameters (e.g. idle frequency), each affecting every gate and idle on its qubit
        params_per_qubit: int = 1,
        qubit_drift=None,  # qubit-parameter drift (default SinusoidalDrift)
        qubit_sensitivity: Optional[Sensitivity] = None,  # per source kind it touches (default: sensitivity)
        error_model: str = "relative",  # gate parameters: "relative" or "additive", see base_envs.py
        qubit_error_model: str = "relative",
        error_budget: float = 0.002,  # error a unit mis-set adds to a scale-1 source (additive model)
        initial_offset: float = DEFAULT_INITIAL_OFFSET,  # rms static offset of the gate-parameter optimum from p = 0
        qubit_initial_offset: Optional[float] = None,  # default: initial_offset
    ):
        self.noise_model = noise_model if noise_model is not None else NoiseModel()
        self.drift = drift if drift is not None else SinusoidalDrift()
        self.qubit_drift = qubit_drift if qubit_drift is not None else SinusoidalDrift()
        self.gate_sensitivity = _by_kind(sensitivity, GATE_KINDS)
        self.qubit_sensitivity = _by_kind(sensitivity if qubit_sensitivity is None else qubit_sensitivity, _KIND_ORDER)

        backend = CircuitBackend(circuit, self.noise_model, decoder)
        gate_block = ParamBlock.per_source(
            backend.sources, kinds=GATE_KINDS, params_per_owner=max(0, int(params_per_gate)), drift=self.drift,
            sensitivity=self.gate_sensitivity, sensitivity_spread=sensitivity_spread,
            error_model=error_model, initial_offset=initial_offset,
        )
        # A qubit parameter's Omega scale differs per source kind, so it sits in the weights
        qubit_block = ParamBlock.per_qubit(
            backend.sources, kind_weights=self.qubit_sensitivity,
            params_per_owner=max(0, int(params_per_qubit)), drift=self.qubit_drift,
            sensitivity=1.0, sensitivity_spread=sensitivity_spread, error_model=qubit_error_model,
            initial_offset=initial_offset if qubit_initial_offset is None else qubit_initial_offset,
        )
        super().__init__(backend, [gate_block, qubit_block], base_error=base_error,
                         max_error=self.noise_model.max_error, seed=seed, error_budget=error_budget)

        self.gate_param_slice, self.qubit_param_slice = self.block_slices
        self.num_gate_params = gate_block.num_params
        self.num_qubit_params = qubit_block.num_params
        self.param_qubits = qubit_block.owners

    def build_noisy_circuit(self, params: np.ndarray, t: float) -> stim.Circuit:
        """Constructs a Stim circuit with the physical error rates induced by params at time t."""
        return self.backend.build_circuit(self.compute_physical_error_rates(params, t))


class GeneratedCodeEnv(QECCodeEnv):
    """
    QECCodeEnv for a circuit from `stim.Circuit.generated(code_task, ...)`, configured
    with flat drift keyword arguments. Subclasses set `code_task` and `default_decoder`,
    or override `make_circuit` to supply a hand-built circuit.
    """
    code_task: str = ""
    default_decoder: str = "pymatching"

    def __init__(
        self,
        distance: int = 5,
        rounds: int = 5,
        base_error: float = 0.002,
        sensitivity: Sensitivity = DEFAULT_SENSITIVITY,  # gate-parameter Omega scale, per gate kind
        sensitivity_spread: float = 0.4,
        params_per_gate: int = 1,
        include_single_qubit_noise: bool = True,
        drift_profile: str = "sinusoidal",  # "sinusoidal", "arma" or "ar1"
        # Sinusoidal drift
        drift_frequency: float = 1.0 / 150.0,
        drift_amplitude: float = 2.0,
        # ARMA drift
        arma_alpha: float = 0.67,  # 0 = white, 1 = flicker, 2 = random walk
        arma_white_fraction: float = 0.1,  # fraction of drift variance in the white floor
        arma_sigma: float = 2.0,
        arma_common_fraction: float = 0.08,
        # AR(1) drift: log-uniform ranges of each parameter's correlation time (epochs) and std
        ar1_tau_range: Tuple[float, float] = (1.0, 50.0),
        ar1_sigma_range: Tuple[float, float] = (0.3, 1.5),
        max_epochs: int = 500,
        seed: int = 163,
        decoder: Union[None, str, Callable[[stim.Circuit], object]] = None,  # None = default_decoder
        noise_model: Optional[NoiseModel] = None,  # overrides include_single_qubit_noise when given
        # Qubit parameters; each qubit_* drift setting left as None copies the gate drift setting
        params_per_qubit: int = 1,
        qubit_drift_profile: Optional[str] = None,
        qubit_drift_frequency: Optional[float] = None,
        qubit_drift_amplitude: Optional[float] = None,
        qubit_arma_alpha: Optional[float] = None,
        qubit_arma_white_fraction: Optional[float] = None,
        qubit_arma_sigma: Optional[float] = None,
        qubit_arma_common_fraction: Optional[float] = None,
        qubit_sensitivity: Optional[Sensitivity] = None,  # qubit-parameter Omega scale, per source kind
        error_model: str = "relative",  # gate parameters: "relative" or "additive", see base_envs.py
        qubit_error_model: str = "relative",
        error_budget: float = 0.002,
        initial_offset: float = DEFAULT_INITIAL_OFFSET,  # rms stale-calibration offset of the gate parameters
        qubit_initial_offset: Optional[float] = None,  # default: initial_offset
    ):
        self.distance = distance
        self.rounds = rounds
        self.drift_profile = str(drift_profile).lower()
        self.qubit_drift_profile = self.drift_profile if qubit_drift_profile is None else str(qubit_drift_profile).lower()
        pick = lambda qubit_value, gate_value: gate_value if qubit_value is None else qubit_value

        super().__init__(
            circuit=self.make_circuit(),
            decoder=decoder if decoder is not None else self.default_decoder,
            noise_model=noise_model if noise_model is not None else NoiseModel(include_single_qubit_noise=include_single_qubit_noise),
            drift=make_drift(self.drift_profile, drift_frequency, drift_amplitude, arma_alpha,
                             arma_white_fraction, arma_sigma, arma_common_fraction, max_epochs,
                             ar1_tau_range, ar1_sigma_range),
            base_error=base_error,
            sensitivity=sensitivity,
            sensitivity_spread=sensitivity_spread,
            params_per_gate=params_per_gate,
            seed=seed,
            params_per_qubit=params_per_qubit,
            qubit_drift=make_drift(
                self.qubit_drift_profile,
                pick(qubit_drift_frequency, drift_frequency),
                pick(qubit_drift_amplitude, drift_amplitude),
                pick(qubit_arma_alpha, arma_alpha),
                pick(qubit_arma_white_fraction, arma_white_fraction),
                pick(qubit_arma_sigma, arma_sigma),
                pick(qubit_arma_common_fraction, arma_common_fraction),
                max_epochs,
                ar1_tau_range,
                ar1_sigma_range,
            ),
            qubit_sensitivity=qubit_sensitivity,
            error_model=error_model,
            qubit_error_model=qubit_error_model,
            error_budget=error_budget,
            initial_offset=initial_offset,
            qubit_initial_offset=qubit_initial_offset,
        )

    def make_circuit(self) -> stim.Circuit:
        """The noiseless memory circuit for `self.distance` and `self.rounds`."""
        return stim.Circuit.generated(self.code_task, distance=self.distance, rounds=self.rounds)
