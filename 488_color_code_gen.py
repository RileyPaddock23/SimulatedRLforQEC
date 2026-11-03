"""
A class to create a Stim circuit for the square-octogon (4.8.8) color code. for now, its just for distance 5
"""
import stim
import numpy as np


class square_octogon_color_code():

    def __init__(self):
        self.circuit = stim.circuit()
        
        #red stabilizers (all squares)
        self.redZ_1 = [0,17, 1,17, 2,17, 3,17]
        self.redZ_2 = [4,19, 5,19, 8,19, 9,19]
        self.redZ_3 = [6,21, 7,21, 10,21, 11,21]

        #green stabilizers (all boundary octogons)
        self.greenZ_1 = [0,18, 2,18, 4,18, 5,18]
        self.greenZ_2 = [8,22, 9,22, 12,22, 13,22]
        self.greenZ_3 = [10,23, 11,23, 14,23, 15,23]

        #blue stabilizers (one full octogon, one boundary octogon)
        self.blueZ_1
