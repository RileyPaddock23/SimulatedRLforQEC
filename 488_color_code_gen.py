"""
A class to create a Stim circuit for the square-octogon (4.8.8) color code. for now, its just for distance 5
"""
import stim
import numpy as np


class square_octogon_color_code():
    """
    #red stabilizers (all squares)
    redZ_1 = {(0,17):1, (1,17):2, (2,17):3, (3,17):4}
    redZ_2 = {(4,19):1, (5,19):2, (8,19):3, (9,19):4}
    redZ_3 = {(6,21):1, (7,21):2, (10,21):3, (11,21):4}

    #green stabilizers (all boundary octogons)
    greenZ_1 = {(0,18):4, (2,18):6, (4,18):7, (5,18):8}
    greenZ_2 = {(8,22):1, (9,22):2, (12,22):3, (13,22):4}
    greenZ_3 = {(10,23):1, (11,23):2, (14,23):3, (15,23):4}

    #blue stabilizers (one full octogon, one boundary octogon)
    blueZ_1 = {(2,20):1, (3,20):2, (5,20):3, (6,20):4, (9,20):5, (10,20):6, (13,20):7, (14,20):8}
    blueZ_2 = {(7,24):1, (11,24):5, (15,24):7, (16,24):8}
    """

    def __init__(self, rounds):
        self.circuit = stim.circuit()

        self.Z_ancilla_indices = range(17,25)
        self.X_ancilla_indices = range(25,33)
        self.data_qubit_indices = range(0,17)
        self.rounds = rounds

        #dictionary where the key CNOT pair and the value is the timeslice to draw it in
        self.all_stabs = {(0,17):1, (1,17):2, (2,17):3, (3,17):4,     #red stab 1
                          (4,19):1, (5,19):2, (8,19):3, (9,19):4,     #red stab 2
                          (6,21):1, (7,21):2, (10,21):3, (11,21):4,   #red stab 3
                          (0,18):4, (2,18):6, (4,18):7, (5,18):8,     #green stab 1 
                          (8,22):1, (9,22):2, (12,22):3, (13,22):4,   #green stab 2
                          (10,23):1, (11,23):2, (14,23):3, (15,23):4, #green stab 3
                          (2,20):1, (3,20):2, (5,20):3, (6,20):4, (9,20):5, (10,20):6, (13,20):7, (14,20):8, #blue stab 1
                          (7,24):1, (11,24):5, (15,24):7, (16,24):8   #blue stab 2
                          }

    def Zstabilizers(self):
        c = stim.Circuit()
        sorted_stabs = dict(sorted(self.all_stabs.items(), key=lambda item: item[1]))

        prev_time = 1
        for pair,time in sorted_stabs:
            if time != prev_time: c.append("TICK") #next timeslice
            c.append("CNOT", pair)

        return c

    def build_circuit(self):
        c = stim.Circuit()
        
        c.append(self.Zstabilizers() * self.rounds)
        c.append("M", self.Z_ancilla_indices)





