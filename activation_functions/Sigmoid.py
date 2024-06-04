'''
Andre Palacio Braga Tivo 13835534
João Pedro Gonçalves Vilela 13731070
Lucas Muniz de Lima 13728941
Lucas Toupitzen Ferracin Garcia 11804164
'''


import math
from .Activation_interface import Activation_interface
import numpy as np

class SigmoidActivation(Activation_interface):

    def forward(self, inputs):

        outputs = []
        for input_x in inputs:
            output = self.sigmoid_function(input_x)
            outputs.append(output)

        self.output = outputs

    def make_derivatives(self, inputs):

        outputs = []
        for input in inputs:
            output = self.sigmoid_derivative(input)
            outputs.append(output)

        self.derivatives = outputs

    @classmethod
    def sigmoid_function(cls, input_x):

        return (1 / (1 + math.exp(-input_x)))
    
    @classmethod
    def sigmoid_derivative(cls, input):

        return cls.sigmoid_function(input)*(1 - cls.sigmoid_function(input))