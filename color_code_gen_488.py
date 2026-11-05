"""
A class to create a Stim circuit for the square-octogon (4.8.8) color code. for now, its just for distance 5
"""
import stim
import numpy as np


class square_octogon_color_code(): 
    """
    Only Z stabilizers implemented for now
    """

    def __init__(self, rounds):
        self.circuit = stim.Circuit()

        self.Z_ancilla_indices = range(17,25)
        self.num_data_qubits = 17
        self.num_ancilla_qubits = 16 #8 each
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
        for control,target in sorted_stabs:
            if sorted_stabs[(control,target)] != prev_time: 
                c.append("TICK") #next timeslice
                prev_time = sorted_stabs[(control,target)]
            c.append("CNOT", (control, target))
            #c.append("DEPOLARIZE2", (control, target), p) #simple circuit level (for testing the circuit)

        #c.append("DEPOLARIZE1", self.Z_ancilla_indices, p) #phenomenological (for testing the circuit)
        c.append("MR", self.Z_ancilla_indices)

        for i in range(len(self.Z_ancilla_indices)):
            c.append("DETECTOR", [stim.target_rec(-i-1), stim.target_rec(-i-8-1)])

        return c

    def final_round(self, circ):
        c = stim.Circuit()

        #will get errors without the whole circuit here. need to be able to look back at all the measurements
        c.append(circ)

        #red stabilizers (all squares)
        redZ_1 = [0, 1, 2, 3]
        redZ_2 = [4, 5, 8, 9] 
        redZ_3 = [6, 7, 10, 11] 
    
        #green stabilizers (all boundary octogons)
        greenZ_1 = [0, 2, 4, 5] 
        greenZ_2 = [8, 9, 12, 13]
        greenZ_3 = [10, 11, 14, 15]
    
        #blue stabilizers (one full octogon, one boundary octogon)
        blueZ_1 = [2, 3, 5, 6, 9, 10, 13, 14]
        blueZ_2 = [7, 11, 15, 16]

        Z_stabs = [redZ_1, redZ_2, redZ_3, greenZ_1, greenZ_2, greenZ_3, blueZ_1, blueZ_2]
    
        c.append("M", self.data_qubit_indices)

        #final round detectors
        num_stabs_meas = 0
        for stab in Z_stabs:
            target_recs = []
            for i in stab:
                target_recs.append(stim.target_rec(i - self.num_data_qubits))
            #comapre to previous measurement of this stabilizer
            target_recs.append(stim.target_rec(num_stabs_meas - self.num_data_qubits - (int)(self.num_ancilla_qubits/2)))
            c.append("DETECTOR", target_recs)
            num_stabs_meas += 1

        obs_rec = []
        for i in range(self.num_data_qubits): #compare all data qubits as the observable
            obs_rec.append(stim.target_rec(-i-1))

        c.append("OBSERVABLE_INCLUDE", obs_rec, 0)

        return c


    def build_circuit(self):
        c = stim.Circuit()

        c.append("MR", self.data_qubit_indices)
        c.append("MR", self.Z_ancilla_indices)

        #c.append("DEPOLARIZE1", self.data_qubit_indices, p) #code cap (for testing the circuit)

        c.append(self.Zstabilizers() * self.rounds)

        c = self.final_round(c) #set equal rather than append because c gets passed into final_round and added to

        return c






