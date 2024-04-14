import math
from .Activation_interface import Activation_interface

class SigmoidActivation(Activation_interface):

    def forward(self, inputs):

        outputs = []
        for input in inputs:
            output = self.sigmoid_function(input)
            outputs.append(output)

        self.output = outputs

    def make_derivatives(self, inputs):

        outputs = []
        for input in inputs:
            output = self.sigmoid_derivative(input)
            outputs.append(output)

        self.derivatives = outputs

    @classmethod
    def sigmoid_function(cls, input):

        return 1 / (1 + math.exp(-input))
    
    @classmethod
    def sigmoid_derivative(cls, input):

        return cls.sigmoid_function(input)*(1 - cls.sigmoid_function(input))