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
        self.rounds = rounds
        self.num_data_qubits = 17
        self.data_qubit_indices = range(0, 17)
        self.Z_ancilla_indices = range(17, 25)
        self.X_ancilla_indices = range(25, 33)  # not used until X stabilizers are added

        # dictionary where the key is the CNOT (data, ancilla) pair and the value is the timeslice to draw it in
        self.all_stabs = {(0, 17): 1, (1, 17): 2, (2, 17): 3, (3, 17): 4,       # red stab 1 (square)
                          (4, 19): 1, (5, 19): 2, (8, 19): 3, (9, 19): 4,       # red stab 2 (square)
                          (6, 21): 1, (7, 21): 2, (10, 21): 3, (11, 21): 4,     # red stab 3 (square)
                          (0, 18): 4, (2, 18): 6, (4, 18): 7, (5, 18): 8,       # green stab 1 (boundary octagon)
                          (8, 22): 1, (9, 22): 2, (12, 22): 3, (13, 22): 4,     # green stab 2 (boundary octagon)
                          (10, 23): 1, (11, 23): 2, (14, 23): 3, (15, 23): 4,   # green stab 3 (boundary octagon)
                          (2, 20): 1, (3, 20): 2, (5, 20): 3, (6, 20): 4,       # blue stab 1 (full octagon)
                          (9, 20): 5, (10, 20): 6, (13, 20): 7, (14, 20): 8,
                          (7, 24): 1, (11, 24): 5, (15, 24): 7, (16, 24): 8,    # blue stab 2 (boundary octagon)
                          }

        # data qubits in each Z stabilizer, keyed by its ancilla
        self.Z_stabs = {a: sorted(d for d, anc in self.all_stabs if anc == a) for a in self.Z_ancilla_indices}

    def Zstabilizers(self, first_round: bool = False) -> stim.Circuit:
        """
        One round of Z stabilizer measurements, each compared to the previous round. In the
        first round the ancillas start from a reset rather than a measurement, so the
        stabilizers are deterministically +1 and each detector is the measurement alone.
        The readout gets its own moment and the round ends on a TICK, so idling is placed right.
        """
        c = stim.Circuit()
        sorted_stabs = sorted(self.all_stabs.items(), key=lambda item: item[1])

        prev_time = sorted_stabs[0][1]
        for (control, target), time in sorted_stabs:
            if time != prev_time:
                c.append("TICK")  # next timeslice
                prev_time = time
            c.append("CNOT", (control, target))

        c.append("TICK")
        c.append("MR", self.Z_ancilla_indices)

        num_Z = len(self.Z_ancilla_indices)
        for i in range(num_Z):
            if first_round:
                c.append("DETECTOR", [stim.target_rec(-i - 1)])
            else:
                c.append("DETECTOR", [stim.target_rec(-i - 1), stim.target_rec(-i - num_Z - 1)])
        c.append("TICK")

        return c

    def final_round(self, c: stim.Circuit) -> stim.Circuit:
        """
        Appends the data qubit readout to c (which must hold the earlier rounds, since the
        detectors look back at the last stabilizer measurements) and returns it.
        """
        c.append("M", self.data_qubit_indices)

        # each stabilizer's data parity, compared to its ancilla's last measurement
        num_Z = len(self.Z_ancilla_indices)
        for k, ancilla in enumerate(self.Z_ancilla_indices):
            target_recs = [stim.target_rec(i - self.num_data_qubits) for i in self.Z_stabs[ancilla]]
            target_recs.append(stim.target_rec(k - num_Z - self.num_data_qubits))
            c.append("DETECTOR", target_recs)

        # the logical Z is the parity of all data qubits
        obs_rec = [stim.target_rec(-i - 1) for i in range(self.num_data_qubits)]
        c.append("OBSERVABLE_INCLUDE", obs_rec, 0)

        return c

    def build_circuit(self) -> stim.Circuit:
        """The noiseless memory circuit; noise is added by the backend."""
        if self.rounds < 1:
            raise ValueError("cc488 needs at least one round of stabilizer measurements")
        c = stim.Circuit()
        # a reset rather than MR: a noisy reference measurement would mimic a data error on
        # corner qubit 16 (in only one stabilizer), giving weight-2 logical errors
        c.append("R", list(self.data_qubit_indices) + list(self.Z_ancilla_indices))
        c.append("TICK")
        c.append(self.Zstabilizers(first_round=True))
        c.append(self.Zstabilizers() * (self.rounds - 1))
        return self.final_round(c)


class ColorCode488Env(GeneratedCodeEnv):
    """4.8.8 (square-octagon) color code (Z memory) with continuous hardware drift and local factor-graph masking."""
    default_decoder = "bposd"

    def make_circuit(self) -> stim.Circuit:
        if self.distance != 5:
            raise ValueError(f"cc488 only implements distance 5, got distance={self.distance}")
        return cc488(self.rounds).build_circuit()
