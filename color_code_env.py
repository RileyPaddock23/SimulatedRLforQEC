"""
Color Code Environment with Drift Injection (minimal)
=====================================================
Distance-d triangular color code memory experiment (stim's "color_code:memory_xyz")
on top of the generic `QECCodeEnv` (see qec_env.py for the noise and drift model).
Color code errors don't decompose into matchable edges, so it is decoded with BP+OSD.
"""

import stim

from qec_env import GeneratedCodeEnv
from qec_env import QECCodeEnv
from qec_env import NoiseModel
from qec_env import make_drift
from typing import Callable, Optional, Union

from color_code_gen_488 import square_octogon_color_code as cc488


class ColorCodeEnv(GeneratedCodeEnv):
    """Color code (XYZ memory) with continuous hardware drift and local factor-graph masking."""
    code_task = "color_code:memory_xyz"
    default_decoder = "bposd"  # T.S 10/14 swapped out pymatching for a stimbposd decoder

class ColorCode488Env(QECCodeEnv):
    """4.8.8 (square-octogon) color code with continuous hardware drift and local factor-graph masking."""
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
            qubit_weight_1q: float = 1.0
                 ):

            pick = lambda qubit_value, gate_value: gate_value if qubit_value is None else qubit_value
            self.drift_profile = str(drift_profile).lower()
            self.qubit_drift_profile = self.drift_profile if qubit_drift_profile is None else str(qubit_drift_profile).lower()

            cc488_builder = cc488(5)
            
            super().__init__(
                circuit=cc488_builder.build_circuit(), 
                decoder="bposd",
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
