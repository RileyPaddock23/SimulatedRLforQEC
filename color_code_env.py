"""
Color Code Environment with Drift Injection (minimal)
=====================================================
Distance-d triangular color code memory experiment (stim's "color_code:memory_xyz")
on top of the generic `QECCodeEnv` (see qec_env.py for the noise and drift model).
Color code errors don't decompose into matchable edges, so it is decoded with BP+OSD.
"""

from qec_env import GeneratedCodeEnv


class ColorCodeEnv(GeneratedCodeEnv):
    """Color code (XYZ memory) with continuous hardware drift and local factor-graph masking."""
    code_task = "color_code:memory_xyz"
    default_decoder = "bposd"  # T.S 10/14 swapped out pymatching for a stimbposd decoder
