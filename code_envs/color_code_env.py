"""
Color Code Environments with Drift Injection (minimal)
======================================================
Color code memory experiments on top of the generic `QECCodeEnv` (see qec_env.py for
the noise and drift model):
  - `ColorCodeEnv`: distance-d triangular (6.6.6) color code, stim's "color_code:memory_xyz"
  - `ColorCode488Env`: distance-5 square-octagon (4.8.8) color code, hand-built by `cc488`
Color code errors don't decompose into matchable edges, so both are decoded with BP+OSD.
"""

import stim

from code_envs.qec_env import GeneratedCodeEnv

#for the generation of memory experiment stim circuit from parity check matrices
from qldpc import codes, circuits
from qldpc.objects import Pauli


class ColorCodeEnv(GeneratedCodeEnv):
    """Color code (XYZ memory) with continuous hardware drift and local factor-graph masking."""
    code_task = "color_code:memory_xyz"
    default_decoder = "bposd"  # T.S 10/14 swapped out pymatching for a stimbposd decoder


class cc488:
    """
    Stim circuit for the distance-5 square-octagon (4.8.8) color code, Z-basis memory.
    Only the Z stabilizers are measured for now, so only X errors are detected.
    """

    def __init__(self, rounds: int):
       self.H = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #1
                 [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #2
                 [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], #3
                 [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0], #4
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0], #5
                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1], #6
                 [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], #7
                ]
       self.code = codes.CSSCode(code_x=self.H, code_z=self.H)


    def build_circuit(self) -> stim.Circuit:
        """The noiseless memory circuit; noise is added by the backend."""
        return circuits.get_memory_experiment(self.code, num_rounds=5, basis=Pauli.Z)

class ColorCode488Env(GeneratedCodeEnv):
    """4.8.8 (square-octagon) color code (Z memory) with continuous hardware drift and local factor-graph masking."""
    default_decoder = "bposd"

    def make_circuit(self) -> stim.Circuit:
        if self.distance != 5:
            raise ValueError(f"cc488 only implements distance 5, got distance={self.distance}")
        return cc488(self.rounds).build_circuit()
