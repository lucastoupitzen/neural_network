from .Activation_interface import Activation_interface

class ReluActivation(Activation_interface):

    def forward(self, inputs):
        outputs = []
        for input in inputs:
            output = input if input > 0 else 0 
            outputs.append(output)

        self.output = outputs

    def make_derivatives(self, inputs):

        outputs = []
        for input in inputs:
            output = 1 if input > 0 else 0
            outputs.append(output)

        self.derivatives = outputs
