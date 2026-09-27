"""
Rotated Surface Code Environment with Drift Injection (minimal)
===============================================================
Stim-based environment for a distance-d rotated surface code memory experiment.
Each physical gate (CX coupler, and optionally each qubit's single-qubit/reset/readout
operations) has control parameters p_g whose optimum p_opt_g(t) drifts in time.
Mis-set parameters raise that gate's depolarizing error rate quadratically:

    eps_g(t, p_g) = base_g + sum_m Omega_{g,m} * (p_{g,m} - p_opt_{g,m}(t))^2

The environment also exposes the bipartite factor-graph matrix M in {0, 1}^(O x P)
connecting parameters to the detectors they can flip.

Supported drift profiles:
  - "sinusoidal": p_opt(t) = A * sin(2 pi f t)
  - "arma":       1/f^alpha colored noise plus a white noise floor, with a shared
                  chip-wide common-mode component.
"""

from typing import Dict, List, Optional, Tuple
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


class RotatedSurfaceCodeEnv:
    """
    Parameterized environment for a rotated surface code with continuous
    hardware drift and local factor-graph masking.

    Time `t` is measured in (reference) epochs. `drift_frequency` is in cycles per
    epoch, and ARMA trajectories are sampled once per epoch and linearly
    interpolated in between.
    """

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
    ):
        self.distance = distance
        self.rounds = rounds
        self.base_error = base_error
        self.sensitivity = sensitivity
        self.params_per_gate = max(1, int(params_per_gate))
        self.include_single_qubit_noise = include_single_qubit_noise
        self.drift_profile = str(drift_profile).lower()
        if self.drift_profile not in ("sinusoidal", "arma"):
            raise ValueError(f"Unknown drift_profile '{drift_profile}' (use 'sinusoidal' or 'arma')")
        self.drift_frequency = drift_frequency
        self.drift_amplitude = drift_amplitude
        self.arma_alpha = float(arma_alpha)
        self.arma_white_fraction = float(np.clip(arma_white_fraction, 0.0, 1.0))
        self.arma_sigma = float(arma_sigma)
        self.arma_common_fraction = float(np.clip(arma_common_fraction, 0.0, 1.0))
        self.max_epochs = max(10, int(max_epochs))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # 1. Generate base topological circuit from Stim
        self.base_circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=self.distance,
            rounds=self.rounds,
        )
        self.flat_instructions = self.base_circuit.flattened()
        self.num_detectors = self.base_circuit.num_detectors
        self.num_observables = self.base_circuit.num_observables

        # 2. Identify physical qubit layout and distinct gates
        self._setup_parameter_mapping()

        # 3. Initialize parameter drift trajectories
        self._setup_drift_profiles()

        # 4. Construct bipartite factor-graph dependency matrix M in {0, 1}^(O x P)
        self.M = self._compute_factor_graph_mask()

        # 5. Pre-compile circuit template for fast noisy-circuit assembly
        self._setup_fast_circuit_template()

        # 6. Pre-compile decoder for validation (PyMatching)
        self._setup_decoder()

    def _setup_parameter_mapping(self):
        """Maps distinct 2-qubit CX couplers and single-qubit gates to parameter indices."""
        self.cx_pairs: List[Tuple[int, int]] = []
        self.pair_to_id: Dict[Tuple[int, int], int] = {}
        self.qubits_used = set()

        for op in self.flat_instructions:
            if op.name == "CX":
                targets = op.targets_copy()
                for k in range(0, len(targets), 2):
                    pair = (targets[k].value, targets[k + 1].value)
                    self.qubits_used.add(pair[0])
                    self.qubits_used.add(pair[1])
                    if pair not in self.pair_to_id:
                        self.pair_to_id[pair] = len(self.cx_pairs)
                        self.cx_pairs.append(pair)
            elif op.name in ("R", "M", "MR", "H"):
                for t in op.targets_copy():
                    if t.is_qubit_target:
                        self.qubits_used.add(t.value)

        self.num_cx_gates = len(self.cx_pairs)
        self.qubit_list = sorted(list(self.qubits_used))
        self.num_qubit_gates = len(self.qubit_list) if self.include_single_qubit_noise else 0
        self.num_gates = self.num_cx_gates + self.num_qubit_gates

        self.qubit_to_id = {q: idx + self.num_cx_gates for idx, q in enumerate(self.qubit_list)}

        self.num_cx_params = self.num_cx_gates * self.params_per_gate
        self.num_qubit_params = self.num_qubit_gates * self.params_per_gate
        self.num_params = self.num_gates * self.params_per_gate

    def _setup_drift_profiles(self):
        """Initializes drift sensitivities, baseline errors, and ARMA trajectories."""
        # Heterogeneous sensitivity Omega_i for each parameter (some parameters drift worse than others)
        self.param_sensitivities = (self.sensitivity * self.rng.uniform(0.6, 1.4, size=self.num_params)).astype(np.float32)
        # Baseline irreducible error per physical gate
        self.param_base_errors = (self.base_error * self.rng.uniform(0.8, 1.2, size=self.num_gates)).astype(np.float32)

        if self.drift_profile != "arma":
            return

        # Each parameter's trajectory = sqrt(c) * shared + sqrt(1 - c) * local
        n = self.max_epochs + 1
        gen = lambda: generate_power_law_noise(
            n, alpha=self.arma_alpha, white_fraction=self.arma_white_fraction, sigma=self.arma_sigma, rng=self.rng
        )

        c = self.arma_common_fraction
        shared = gen()
        self.arma_trajectories = np.zeros((self.num_params, n), dtype=np.float32)
        for pid in range(self.num_params):
            self.arma_trajectories[pid] = np.sqrt(c) * shared + np.sqrt(1.0 - c) * gen()

    def get_drift_opt(self, t: float) -> np.ndarray:
        """Returns the ground-truth optimal parameter vector at time t (epochs)."""
        if self.drift_profile == "sinusoidal":
            p_opt = np.zeros(self.num_params, dtype=np.float32)
            for m in range(self.params_per_gate):
                freq_m = self.drift_frequency * (1.0 + 0.5 * m)
                p_opt[m::self.params_per_gate] = self.drift_amplitude * np.sin(2.0 * np.pi * freq_m * t)
            return p_opt

        # ARMA: linear interpolation between per-epoch samples
        t_clamped = max(0.0, min(float(t), float(self.max_epochs)))
        idx = int(t_clamped)
        frac = t_clamped - idx
        if frac > 1e-5 and idx + 1 <= self.max_epochs:
            return (1.0 - frac) * self.arma_trajectories[:, idx] + frac * self.arma_trajectories[:, idx + 1]
        return self.arma_trajectories[:, idx].copy()

    def compute_physical_error_rates(self, params: np.ndarray, t: float) -> np.ndarray:
        """
        Maps control parameters p to physical gate error rates:
        eps_g(t, p_g) = base_g + sum_{m} Omega_{g,m} * (p_{g,m} - p_opt_{g,m}(t))^2
        """
        p_opt = self.get_drift_opt(t)
        dev = params - p_opt
        dev_2d = dev.reshape((self.num_gates, self.params_per_gate))
        sens_2d = self.param_sensitivities.reshape((self.num_gates, self.params_per_gate))
        quad_err = np.sum(sens_2d * (dev_2d ** 2), axis=1)
        eps = self.param_base_errors + quad_err
        # Clamp to physical depolarizing noise boundaries [base_error, 0.15]
        return np.clip(eps, self.param_base_errors, 0.15)

    def _compute_factor_graph_mask(self) -> np.ndarray:
        """
        Computes the sparse bipartite factor-graph matrix M in {0, 1}^(O x P).
        M[j, i] = 1 if detector D_j is in the detecting region of parameter p_i.
        """
        # 1. Gate-level detector incidence matrix M_gate in {0, 1}^(O x num_gates)
        M_gate = np.zeros((self.num_detectors, self.num_gates), dtype=np.float32)

        for gid in range(self.num_gates):
            c = stim.Circuit()
            for op in self.flat_instructions:
                # Pre-measurement readout noise
                if gid >= self.num_cx_gates:
                    target_q = self.qubit_list[gid - self.num_cx_gates]
                    if op.name in ("M", "MR"):
                        for t_gate in op.targets_copy():
                            if t_gate.is_qubit_target and t_gate.value == target_q:
                                c.append("DEPOLARIZE1", [target_q], 0.01)

                c.append(op)

                if gid < self.num_cx_gates:
                    # 2-qubit gate parameter
                    if op.name == "CX":
                        targets = op.targets_copy()
                        for k in range(0, len(targets), 2):
                            pair = (targets[k].value, targets[k + 1].value)
                            if self.pair_to_id[pair] == gid:
                                c.append("DEPOLARIZE2", [pair[0], pair[1]], 0.01)
                else:
                    # Single-qubit gate / reset / state preparation noise (after H, R, MR)
                    target_q = self.qubit_list[gid - self.num_cx_gates]
                    if op.name in ("H", "R", "MR"):
                        for t_gate in op.targets_copy():
                            if t_gate.is_qubit_target and t_gate.value == target_q:
                                c.append("DEPOLARIZE1", [target_q], 0.01)

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

        # 2. Expand to parameter mask matrix M in {0, 1}^(O x P)
        if self.params_per_gate == 1:
            return M_gate
        return np.repeat(M_gate, self.params_per_gate, axis=1)

    def _setup_fast_circuit_template(self):
        """
        Precomputes a circuit string template and gate-id map so noisy circuits
        can be assembled with a single str.format call.
        """
        template_parts = []
        self._template_gate_ids = []

        for op in self.flat_instructions:
            # 1. Readout discrimination noise applied before measurement
            if self.include_single_qubit_noise and op.name in ("M", "MR"):
                for t_gate in op.targets_copy():
                    if t_gate.is_qubit_target and t_gate.value in self.qubit_to_id:
                        template_parts.append(f"DEPOLARIZE1({{:.8g}}) {t_gate.value}\n")
                        self._template_gate_ids.append(self.qubit_to_id[t_gate.value])

            template_parts.append(str(op) + "\n")

            # 2. Two-qubit gate noise
            if op.name == "CX":
                targets = op.targets_copy()
                for k in range(0, len(targets), 2):
                    pair = (targets[k].value, targets[k + 1].value)
                    if pair in self.pair_to_id:
                        template_parts.append(f"DEPOLARIZE2({{:.8g}}) {pair[0]} {pair[1]}\n")
                        self._template_gate_ids.append(self.pair_to_id[pair])

            # 3. Single-qubit unitary / reset / state preparation noise applied after gate or reset
            elif self.include_single_qubit_noise and op.name in ("H", "R", "MR"):
                for t_gate in op.targets_copy():
                    if t_gate.is_qubit_target and t_gate.value in self.qubit_to_id:
                        template_parts.append(f"DEPOLARIZE1({{:.8g}}) {t_gate.value}\n")
                        self._template_gate_ids.append(self.qubit_to_id[t_gate.value])

        self._fast_template_str = "".join(template_parts)

    def build_noisy_circuit(self, params: np.ndarray, t: float) -> stim.Circuit:
        """Constructs a Stim circuit with the physical error rates induced by params at time t."""
        eps = self.compute_physical_error_rates(params, t)
        rates = [eps[gid] for gid in self._template_gate_ids]
        return stim.Circuit(self._fast_template_str.format(*rates))

    def _setup_decoder(self):
        """Initializes a PyMatching decoder from the error model of the static policy (params = 0) at t = 0."""
        nominal_params = np.zeros(self.num_params, dtype=np.float32)
        self.noisy_circuit = self.build_noisy_circuit(nominal_params, t=0.0)
        dem = self.noisy_circuit.detector_error_model(decompose_errors=True)
        self.matcher = pymatching.Matching.from_detector_error_model(dem)

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
            predicted_obs = self.matcher.decode_batch(det_shots)
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
