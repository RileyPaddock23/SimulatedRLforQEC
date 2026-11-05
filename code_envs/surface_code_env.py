"""
Rotated Surface Code Environment with Drift Injection (minimal)
===============================================================
Distance-d rotated surface code memory experiment on top of the generic `QECCodeEnv`
(see qec_env.py for the noise and drift model), decoded with PyMatching.
"""

from code_envs.qec_env import GeneratedCodeEnv


class RotatedSurfaceCodeEnv(GeneratedCodeEnv):
    """Rotated surface code (Z-basis memory) with continuous hardware drift and local factor-graph masking."""
    code_task = "surface_code:rotated_memory_z"
    default_decoder = "pymatching"
