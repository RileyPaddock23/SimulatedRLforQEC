"""
Generic QEC Code Environment with Drift Injection
=================================================
Stim-based environment for any QEC memory experiment given as a stim circuit with
DETECTORs and OBSERVABLE_INCLUDEs. Each physical gate (two-qubit coupler, and optionally
each qubit's single-qubit/reset/readout operations) has control parameters p_g whose
optimum p_opt_g(t) drifts in time. Mis-set parameters raise that gate's depolarizing
error rate quadratically:

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

The environment also exposes the bipartite factor-graph matrix M in {0, 1}^(O x P)
connecting parameters to the detectors they can flip.

The pluggable pieces are:
  - the circuit:     any noiseless stim circuit (existing noise channels are stripped)
  - the noise model: `NoiseModel`, where the drift-controlled channels are placed
  - the drifts:      `SinusoidalDrift`, `ArmaDrift`, or any object with the same `build`,
                     one for the gate parameters and one for the qubit parameters
plus the decoder used for logical error rates ("pymatching", "bposd", or a factory).

The agent stays outside the environment: it only needs `num_params`, `num_detectors`,
`M` and `evaluate_policy`.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import stim
import pymatching


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
# A drift model has `build(num_params, params_per_gate, rng)`, returning a function
# t -> p_opt(t) of shape (num_params,). Parameters are laid out gate-major, so
# parameter m of gate g is index g * params_per_gate + m.

class SinusoidalDrift:
    """p_opt(t) = A * sin(2 pi f_m t), with f_m = f * (1 + 0.5 m) for the m-th parameter of each gate."""

    def __init__(self, frequency: float = 1.0 / 150.0, amplitude: float = 0.6):
        self.frequency = frequency
        self.amplitude = amplitude

    def build(self, num_params: int, params_per_gate: int, rng: np.random.Generator) -> Callable[[float], np.ndarray]:
        def p_opt(t: float) -> np.ndarray:
            out = np.zeros(num_params, dtype=np.float32)
            for m in range(params_per_gate):
                freq_m = self.frequency * (1.0 + 0.5 * m)
                out[m::params_per_gate] = self.amplitude * np.sin(2.0 * np.pi * freq_m * t)
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

    def build(self, num_params: int, params_per_gate: int, rng: np.random.Generator) -> Callable[[float], np.ndarray]:
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
    max_error: float = 0.15  # depolarizing rates are clipped to [base_g, max_error]

    def resolved(self) -> Tuple[Set[str], Set[str], Set[str]]:
        two_qubit, after, before = _default_gate_sets()
        return (
            set(self.two_qubit_gates) if self.two_qubit_gates is not None else two_qubit,
            set(self.after_gates) if self.after_gates is not None else after,
            set(self.before_gates) if self.before_gates is not None else before,
        )


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


# ---------------------------------------------------------------------- environment

class QECCodeEnv:
    """
    Parameterized environment for an arbitrary stim QEC circuit with continuous
    hardware drift and local factor-graph masking.

    Time `t` is measured in (reference) epochs; the drift model defines what that means
    for its own parameters.

    Gate ids: two-qubit couplers come first (0 .. num_cx_gates - 1, in order of first
    appearance), then one id per qubit for its single-qubit/reset/readout noise. The
    `cx_*` names are kept for compatibility but cover every two-qubit gate in
    `NoiseModel.two_qubit_gates`, not only CX.

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
        self.base_error = base_error
        self.sensitivity = sensitivity
        self.params_per_gate = max(0, int(params_per_gate))
        self.params_per_qubit = max(0, int(params_per_qubit))
        self.qubit_sensitivity = sensitivity if qubit_sensitivity is None else qubit_sensitivity
        self.qubit_weight_2q = float(qubit_weight_2q)
        self.qubit_weight_1q = float(qubit_weight_1q)
        self.noise_model = noise_model if noise_model is not None else NoiseModel()
        self.include_single_qubit_noise = self.noise_model.include_single_qubit_noise
        self.drift = drift if drift is not None else SinusoidalDrift()
        self.qubit_drift = qubit_drift if qubit_drift is not None else SinusoidalDrift()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # 1. Base circuit; the environment supplies all noise itself
        self.base_circuit = circuit.without_noise()
        self.flat_instructions = self.base_circuit.flattened()
        self.num_detectors = self.base_circuit.num_detectors
        self.num_observables = self.base_circuit.num_observables
        if self.num_detectors == 0:
            raise ValueError("circuit has no DETECTORs, so there is nothing for the agent to observe")

        # 2. Identify physical qubit layout and distinct gates
        self._setup_parameter_mapping()

        # 3. Initialize sensitivities, baseline errors, and the drift trajectory
        self._setup_drift_profiles()

        # 4. Pre-compile circuit template for fast noisy-circuit assembly
        self._setup_fast_circuit_template()

        # 5. Construct bipartite factor-graph dependency matrix M in {0, 1}^(O x P)
        self.M = self._compute_factor_graph_mask()

        # 6. Pre-compile decoder for validation
        self._setup_decoder(decoder)

    def _setup_parameter_mapping(self):
        """Maps distinct 2-qubit couplers and single-qubit gates to parameter indices."""
        self._two_qubit_gates, self._after_gates, self._before_gates = self.noise_model.resolved()
        self.cx_pairs: List[Tuple[int, int]] = []
        self.pair_to_id: Dict[Tuple[int, int], int] = {}
        self.qubits_used = set()

        for op in self.flat_instructions:
            if op.name in self._two_qubit_gates:
                for pair in self._qubit_pairs(op):
                    self.qubits_used.update(pair)
                    if pair not in self.pair_to_id:
                        self.pair_to_id[pair] = len(self.cx_pairs)
                        self.cx_pairs.append(pair)
            elif op.name in self._after_gates or op.name in self._before_gates:
                for t in op.targets_copy():
                    if t.is_qubit_target:
                        self.qubits_used.add(t.value)

        self.num_cx_gates = len(self.cx_pairs)
        self.qubit_list = sorted(list(self.qubits_used))
        self.num_qubit_gates = len(self.qubit_list) if self.include_single_qubit_noise else 0
        self.num_gates = self.num_cx_gates + self.num_qubit_gates
        if self.num_gates == 0:
            raise ValueError("noise model places no channels in this circuit, so there are no parameters to control")

        self.qubit_to_id = {q: idx + self.num_cx_gates for idx, q in enumerate(self.qubit_list)}

        # Gate block
        self.num_cx_params = self.num_cx_gates * self.params_per_gate
        self.num_1q_params = self.num_qubit_gates * self.params_per_gate
        self.num_gate_params = self.num_gates * self.params_per_gate

        # Qubit block: qubits that own at least one noisy gate, and W[g, j] = w_g if gate g touches param_qubits[j]
        coupler_qubits = {q for pair in self.cx_pairs for q in pair}
        self.param_qubits = [q for q in self.qubit_list if q in coupler_qubits or self.include_single_qubit_noise]
        param_qubit_idx = {q: j for j, q in enumerate(self.param_qubits)}
        self.qubit_gate_weights = np.zeros((self.num_gates, len(self.param_qubits)), dtype=np.float32)
        if self.params_per_qubit:
            for gid, pair in enumerate(self.cx_pairs):
                for q in pair:
                    self.qubit_gate_weights[gid, param_qubit_idx[q]] = self.qubit_weight_2q
            if self.include_single_qubit_noise:
                for q in self.qubit_list:
                    self.qubit_gate_weights[self.qubit_to_id[q], param_qubit_idx[q]] = self.qubit_weight_1q
        self.num_qubit_params = len(self.param_qubits) * self.params_per_qubit

        self.num_params = self.num_gate_params + self.num_qubit_params
        self.gate_param_slice = slice(0, self.num_gate_params)
        self.qubit_param_slice = slice(self.num_gate_params, self.num_params)
        if self.num_params == 0:
            raise ValueError("params_per_gate and params_per_qubit are both 0, so there is nothing to control")

    @staticmethod
    def _qubit_pairs(op: stim.CircuitInstruction) -> List[Tuple[int, int]]:
        """Target pairs of a two-qubit gate, skipping classically controlled ones (rec / sweep targets)."""
        targets = op.targets_copy()
        return [
            (targets[k].value, targets[k + 1].value)
            for k in range(0, len(targets), 2)
            if targets[k].is_qubit_target and targets[k + 1].is_qubit_target
        ]

    def _setup_drift_profiles(self):
        """Initializes drift sensitivities, baseline errors, and the drift trajectories."""
        # Gate block first, then the qubit block, so a gate-only env draws the same random numbers as before
        # Heterogeneous sensitivity Omega_i for each parameter (some parameters drift worse than others)
        gate_sens = (self.sensitivity * self.rng.uniform(0.6, 1.4, size=self.num_gate_params)).astype(np.float32)
        # Baseline irreducible error per physical gate
        self.param_base_errors = (self.base_error * self.rng.uniform(0.8, 1.2, size=self.num_gates)).astype(np.float32)
        self._gate_drift_opt = self._build_drift(self.drift, self.num_gate_params, self.params_per_gate)

        qubit_sens = np.zeros(0, dtype=np.float32)
        if self.num_qubit_params:
            qubit_sens = (self.qubit_sensitivity * self.rng.uniform(0.6, 1.4, size=self.num_qubit_params)).astype(np.float32)
        self._qubit_drift_opt = self._build_drift(self.qubit_drift, self.num_qubit_params, self.params_per_qubit)
        self.param_sensitivities = np.concatenate([gate_sens, qubit_sens])

        # A_qubit[g, i] = w_g * Omega_i for qubit parameter i on a qubit that gate g touches
        self._A_qubit = np.repeat(self.qubit_gate_weights, self.params_per_qubit, axis=1) * qubit_sens[None, :]

    def _build_drift(self, drift, num_params: int, per_owner: int) -> Callable[[float], np.ndarray]:
        if num_params == 0:
            return lambda t: np.zeros(0, dtype=np.float32)
        return drift.build(num_params, per_owner, self.rng)

    def get_drift_opt(self, t: float) -> np.ndarray:
        """Returns the ground-truth optimal parameter vector at time t (epochs)."""
        if not self.num_qubit_params:
            return self._gate_drift_opt(t)
        if not self.num_gate_params:
            return self._qubit_drift_opt(t)
        return np.concatenate([self._gate_drift_opt(t), self._qubit_drift_opt(t)])

    def compute_physical_error_rates(self, params: np.ndarray, t: float) -> np.ndarray:
        """
        Maps control parameters p to physical gate error rates:
        eps_g(t, p) = base_g + sum_{m} Omega_{g,m} * (p_{g,m} - p_opt_{g,m}(t))^2
                             + sum_{q in g} w_g sum_m Omega_{q,m} * (p_{q,m} - p_opt_{q,m}(t))^2
        """
        p_opt = self.get_drift_opt(t)
        dev = params - p_opt
        eps = self.param_base_errors
        if self.num_gate_params:
            dev_2d = dev[self.gate_param_slice].reshape((self.num_gates, self.params_per_gate))
            sens_2d = self.param_sensitivities[self.gate_param_slice].reshape((self.num_gates, self.params_per_gate))
            eps = eps + np.sum(sens_2d * (dev_2d ** 2), axis=1)
        if self.num_qubit_params:
            eps = eps + self._A_qubit @ (dev[self.qubit_param_slice] ** 2)
        # Clamp to physical depolarizing noise boundaries [base_error, max_error]
        return np.clip(eps, self.param_base_errors, self.noise_model.max_error)

    def _setup_fast_circuit_template(self):
        """
        Precomputes a circuit string template and gate-id map so noisy circuits
        can be assembled with a single str.format call.
        """
        # (line, gate id) pairs; gate id -1 marks a noiseless line from the base circuit
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
        self._template_gate_ids = [gid for _, gid in self._template_lines if gid >= 0]

    def _compute_factor_graph_mask(self) -> np.ndarray:
        """
        Computes the sparse bipartite factor-graph matrix M in {0, 1}^(O x P).
        M[j, i] = 1 if detector D_j is in the detecting region of parameter p_i.
        """
        # 1. Gate-level detector incidence matrix M_gate in {0, 1}^(O x num_gates):
        #    the noiseless circuit plus only gate gid's channels, at a probe rate of 0.01
        M_gate = np.zeros((self.num_detectors, self.num_gates), dtype=np.float32)

        for gid in range(self.num_gates):
            c = stim.Circuit("".join(
                line if g < 0 else line.format(0.01) for line, g in self._template_lines if g < 0 or g == gid
            ))
            dem = c.detector_error_model(decompose_errors=False)
            for instr in dem:
                if instr.type == "error":
                    for target in instr.targets_copy():
                        if target.is_relative_detector_id():
                            M_gate[target.val, gid] = 1.0

        # Safety check: ensure every gate is observed by at least one detector
        for gid in range(self.num_gates):
            if np.sum(M_gate[:, gid]) == 0:
                M_gate[gid % self.num_detectors, gid] = 1.0

        # 2. Expand to parameter mask matrix M in {0, 1}^(O x P). A gate parameter sees its gate's
        #    detectors; a qubit parameter sees the union over every gate on its qubit.
        M_gates = M_gate if self.params_per_gate == 1 else np.repeat(M_gate, self.params_per_gate, axis=1)
        if not self.num_qubit_params:
            return M_gates
        M_qubit = ((M_gate @ (self.qubit_gate_weights > 0)) > 0).astype(np.float32)
        M_qubits = np.repeat(M_qubit, self.params_per_qubit, axis=1)
        return np.concatenate([M_gates, M_qubits], axis=1)

    def build_noisy_circuit(self, params: np.ndarray, t: float) -> stim.Circuit:
        """Constructs a Stim circuit with the physical error rates induced by params at time t."""
        eps = self.compute_physical_error_rates(params, t)
        rates = [eps[gid] for gid in self._template_gate_ids]
        return stim.Circuit(self._fast_template_str.format(*rates))

    def _setup_decoder(self, decoder):
        """Initializes the decoder from the error model of the static policy (params = 0) at t = 0."""
        nominal_params = np.zeros(self.num_params, dtype=np.float32)
        self.noisy_circuit = self.build_noisy_circuit(nominal_params, t=0.0)
        self.decoder = _make_decoder(decoder, self.noisy_circuit)

    def evaluate_policy(
        self,
        params: np.ndarray,
        t: float,
        shots: int = 1000,
        decode: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Samples `shots` detection events from the circuit compiled with `params`
        at (static) time t.

        Returns:
            dict containing:
                'edr_vector': empirical per-detector error detection rate o in [0, 1]^O
                'mean_edr': scalar average EDR
                'logical_error_rate': decoded LER (if decode=True, else None)
        """
        sampler = self.build_noisy_circuit(params, t).compile_detector_sampler()

        if decode:
            det_shots, actual_obs = sampler.sample(shots=shots, separate_observables=True)
            predicted_obs = self.decoder.decode_batch(det_shots)
            ler = float(np.mean(np.any(actual_obs != predicted_obs, axis=1)))
        else:
            det_shots = sampler.sample(shots=shots, separate_observables=False)
            ler = None

        edr_vector = np.mean(det_shots, axis=0)  # Shape (O,)
        return {
            "edr_vector": edr_vector,
            "mean_edr": float(np.mean(edr_vector)),
            "logical_error_rate": ler,
        }


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


def make_drift(profile: str, frequency: float, amplitude: float, alpha: float, white_fraction: float,
               sigma: float, common_fraction: float, max_epochs: int):
    """Builds a drift model from flat settings: "sinusoidal" uses frequency/amplitude, "arma" the rest."""
    if profile == "sinusoidal":
        return SinusoidalDrift(frequency=frequency, amplitude=amplitude)
    if profile == "arma":
        return ArmaDrift(alpha=alpha, white_fraction=white_fraction, sigma=sigma,
                         common_fraction=common_fraction, max_epochs=max_epochs)
    raise ValueError(f"Unknown drift profile '{profile}' (use 'sinusoidal' or 'arma')")
