"""
Generic QEC Code Environment with Drift Injection
=================================================
Stim-based environment for any QEC memory experiment given as a stim circuit with
DETECTORs and OBSERVABLE_INCLUDEs. `CircuitBackend` places one depolarizing error source
on each physical gate (two-qubit coupler, and optionally each qubit's single-qubit/reset/
readout operations); `QECCodeEnv` is the `ControlEnv` (see base_envs.py) that steers them.

Each gate g has control parameters p_g whose optimum p_opt_g(t) drifts in time. Mis-set
parameters raise that gate's depolarizing error rate quadratically:

    eps_g(t, p_g) = base_g + sum_m Omega_{g,m} * (p_{g,m} - p_opt_{g,m}(t))^2

Optionally, each qubit q also has its own parameters p_q (e.g. its idle frequency, whose
optimum drifts as TLS defects wander and which sets T1). A mis-set qubit parameter raises
the error rate of every gate that touches q, without changing where the channels sit:

    eps_g(t, p) = base_g + sum_m Omega_{g,m} * (p_{g,m} - p_opt_{g,m}(t))^2
                         + sum_{q in g} w_g sum_m Omega_{q,m} * (p_{q,m} - p_opt_{q,m}(t))^2

with w_g = qubit_weight_2q for couplers and qubit_weight_1q for q's own single-qubit /
reset / readout channels. The quadratic is a local model around a single optimum: the
real T1(f) landscape has a narrow dip at each TLS, so it doesn't capture a TLS crossing
the qubit or the dephasing / frequency-collision cost of moving far from the sweet spot.

The pluggable pieces are:
  - the circuit:     any noiseless stim circuit (existing noise channels are stripped)
  - the noise model: `NoiseModel`, where the drift-controlled channels are placed
  - the drifts:      `SinusoidalDrift`, `ArmaDrift`, or any object with the same `build`,
                     one for the gate parameters and one for the qubit parameters
plus the decoder used for logical error rates ("pymatching", "bposd", or a factory).
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import stim
import pymatching

from code_envs.base_envs import Backend, ControlEnv, ParamBlock, Source
from code_envs.noise_envs import NoiseModel, SinusoidalDrift, make_drift


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

class CircuitBackend(Backend):
    """
    Backend sampling a stim circuit with one depolarizing error source per physical gate.

    Source ids: two-qubit couplers come first (0 .. num_cx_gates - 1, in order of first
    appearance), then one id per qubit for its single-qubit/reset/readout noise. The
    `cx_*` names are kept for compatibility but cover every two-qubit gate in
    `NoiseModel.two_qubit_gates`, not only CX. A circuit without OBSERVABLE_INCLUDEs
    still gives detection events, but can't be decoded.
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

        sources = self._setup_sources()
        self._setup_fast_circuit_template()
        super().__init__(sources, self._compute_footprint(len(sources)), has_observables=self.num_observables > 0)

    def _setup_sources(self) -> List[Source]:
        """Finds the distinct 2-qubit couplers and the qubits carrying single-qubit noise."""
        self._two_qubit_gates, self._after_gates, self._before_gates = self.noise_model.resolved()
        self.cx_pairs: List[Tuple[int, int]] = []
        self.pair_to_id: Dict[Tuple[int, int], int] = {}
        qubits_used = set()

        for op in self.flat_instructions:
            if op.name in self._two_qubit_gates:
                for pair in self._qubit_pairs(op):
                    qubits_used.update(pair)
                    if pair not in self.pair_to_id:
                        self.pair_to_id[pair] = len(self.cx_pairs)
                        self.cx_pairs.append(pair)
            elif op.name in self._after_gates or op.name in self._before_gates:
                for t in op.targets_copy():
                    if t.is_qubit_target:
                        qubits_used.add(t.value)

        self.num_cx_gates = len(self.cx_pairs)
        self.qubit_list = sorted(qubits_used) if self.include_single_qubit_noise else []
        self.qubit_to_id = {q: idx + self.num_cx_gates for idx, q in enumerate(self.qubit_list)}
        sources = [Source("2q", pair) for pair in self.cx_pairs] + [Source("1q", (q,)) for q in self.qubit_list]
        if not sources:
            raise ValueError("noise model places no channels in this circuit, so there are no parameters to control")
        return sources

    @staticmethod
    def _qubit_pairs(op: stim.CircuitInstruction) -> List[Tuple[int, int]]:
        """Target pairs of a two-qubit gate, skipping classically controlled ones (rec / sweep targets)."""
        targets = op.targets_copy()
        return [
            (targets[k].value, targets[k + 1].value)
            for k in range(0, len(targets), 2)
            if targets[k].is_qubit_target and targets[k + 1].is_qubit_target
        ]

    def _setup_fast_circuit_template(self):
        """
        Precomputes a circuit string template and source-id map so noisy circuits
        can be assembled with a single str.format call.
        """
        # (line, source id) pairs; source id -1 marks a noiseless line from the base circuit
        self._template_lines: List[Tuple[str, int]] = []
        single_qubit = self.include_single_qubit_noise

        def add_single_qubit_noise(op):
            for t_gate in op.targets_copy():
                if t_gate.is_qubit_target and t_gate.value in self.qubit_to_id:
                    self._template_lines.append((f"DEPOLARIZE1({{:.8g}}) {t_gate.value}\n", self.qubit_to_id[t_gate.value]))

        for op in self.flat_instructions:
            # 1. Readout discrimination noise applied before measurement
            if single_qubit and op.name in self._before_gates:
                add_single_qubit_noise(op)

            self._template_lines.append((str(op) + "\n", -1))

            # 2. Two-qubit gate noise
            if op.name in self._two_qubit_gates:
                for pair in self._qubit_pairs(op):
                    self._template_lines.append((f"DEPOLARIZE2({{:.8g}}) {pair[0]} {pair[1]}\n", self.pair_to_id[pair]))

            # 3. Single-qubit unitary / reset / state preparation noise applied after gate or reset
            elif single_qubit and op.name in self._after_gates:
                add_single_qubit_noise(op)

        self._fast_template_str = "".join(line for line, _ in self._template_lines)
        self._template_source_ids = [sid for _, sid in self._template_lines if sid >= 0]
        # Noisy channel instances per source, i.e. how many operations in the circuit each source covers
        self.source_instances = np.bincount(self._template_source_ids, minlength=len(self.pair_to_id) + len(self.qubit_list))

    def _compute_footprint(self, num_sources: int) -> np.ndarray:
        """
        Source-level detector incidence matrix in {0, 1}^(O x num_sources): the noiseless
        circuit plus only source sid's channels, at a probe rate of 0.01.
        """
        num_detectors = self.base_circuit.num_detectors
        footprint = np.zeros((num_detectors, num_sources), dtype=np.float32)

        for sid in range(num_sources):
            c = stim.Circuit("".join(
                line if s < 0 else line.format(0.01) for line, s in self._template_lines if s < 0 or s == sid
            ))
            dem = c.detector_error_model(decompose_errors=False)
            for instr in dem:
                if instr.type == "error":
                    for target in instr.targets_copy():
                        if target.is_relative_detector_id():
                            footprint[target.val, sid] = 1.0

        # Safety check: ensure every source is observed by at least one detector
        for sid in range(num_sources):
            if np.sum(footprint[:, sid]) == 0:
                footprint[sid % num_detectors, sid] = 1.0
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

    Parameter layout: the gate block comes first (gate g's m-th parameter at
    g * params_per_gate + m, see `gate_param_slice`), then the qubit block (the m-th
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
        sensitivity: float = 0.025,
        params_per_gate: int = 1,
        seed: int = 163,
        # Qubit parameters (e.g. idle frequency), each affecting every gate on its qubit
        params_per_qubit: int = 0,
        qubit_drift=None,  # qubit-parameter drift (default SinusoidalDrift)
        qubit_sensitivity: Optional[float] = None,  # Omega scale for qubit parameters (default: sensitivity)
        qubit_weight_2q: float = 1.0,  # w_g for couplers the qubit takes part in
        qubit_weight_1q: float = 1.0,  # w_g for the qubit's own single-qubit / reset / readout channels
    ):
        self.noise_model = noise_model if noise_model is not None else NoiseModel()
        self.drift = drift if drift is not None else SinusoidalDrift()
        self.qubit_drift = qubit_drift if qubit_drift is not None else SinusoidalDrift()

        backend = CircuitBackend(circuit, self.noise_model, decoder)
        gate_block = ParamBlock.per_source(
            backend.sources, params_per_owner=max(0, int(params_per_gate)), drift=self.drift, sensitivity=sensitivity,
        )
        qubit_block = ParamBlock.per_qubit(
            backend.sources, weight_2q=qubit_weight_2q, weight_1q=qubit_weight_1q,
            params_per_owner=max(0, int(params_per_qubit)), drift=self.qubit_drift,
            sensitivity=sensitivity if qubit_sensitivity is None else qubit_sensitivity,
        )
        super().__init__(backend, [gate_block, qubit_block], base_error=base_error,
                         max_error=self.noise_model.max_error, seed=seed)

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
    with flat drift keyword arguments. Subclasses set `code_task` and `default_decoder`.
    """
    code_task: str = ""
    default_decoder: str = "pymatching"

    def __init__(
        self,
        distance: int = 5,
        rounds: int = 5,
        base_error: float = 0.002,
        sensitivity: float = 0.025,
        params_per_gate: int = 1,
        include_single_qubit_noise: bool = True,
        drift_profile: str = "sinusoidal",  # "sinusoidal" or "arma"
        # Sinusoidal drift
        drift_frequency: float = 1.0 / 150.0,
        drift_amplitude: float = 0.6,
        # ARMA drift
        arma_alpha: float = 0.67,  # 0 = white, 1 = flicker, 2 = random walk
        arma_white_fraction: float = 0.1,  # fraction of drift variance in the white floor
        arma_sigma: float = 0.60,
        arma_common_fraction: float = 0.08,
        max_epochs: int = 500,
        seed: int = 163,
        decoder: Union[None, str, Callable[[stim.Circuit], object]] = None,  # None = default_decoder
        noise_model: Optional[NoiseModel] = None,  # overrides include_single_qubit_noise when given
        # Qubit parameters; each qubit_* drift setting left as None copies the gate drift setting
        params_per_qubit: int = 0,
        qubit_drift_profile: Optional[str] = None,
        qubit_drift_frequency: Optional[float] = None,
        qubit_drift_amplitude: Optional[float] = None,
        qubit_arma_alpha: Optional[float] = None,
        qubit_arma_white_fraction: Optional[float] = None,
        qubit_arma_sigma: Optional[float] = None,
        qubit_arma_common_fraction: Optional[float] = None,
        qubit_sensitivity: Optional[float] = None,
        qubit_weight_2q: float = 1.0,
        qubit_weight_1q: float = 1.0,
    ):
        self.distance = distance
        self.rounds = rounds
        self.drift_profile = str(drift_profile).lower()
        self.qubit_drift_profile = self.drift_profile if qubit_drift_profile is None else str(qubit_drift_profile).lower()
        pick = lambda qubit_value, gate_value: gate_value if qubit_value is None else qubit_value

        super().__init__(
            circuit=stim.Circuit.generated(self.code_task, distance=distance, rounds=rounds),
            decoder=decoder if decoder is not None else self.default_decoder,
            noise_model=noise_model if noise_model is not None else NoiseModel(include_single_qubit_noise=include_single_qubit_noise),
            drift=make_drift(self.drift_profile, drift_frequency, drift_amplitude, arma_alpha,
                             arma_white_fraction, arma_sigma, arma_common_fraction, max_epochs),
            base_error=base_error,
            sensitivity=sensitivity,
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
            ),
            qubit_sensitivity=qubit_sensitivity,
            qubit_weight_2q=qubit_weight_2q,
            qubit_weight_1q=qubit_weight_1q,
        )
