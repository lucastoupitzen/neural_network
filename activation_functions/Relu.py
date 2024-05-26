from .Activation_interface import Activation_interface
import numpy as np

class ReluActivation(Activation_interface):

    
    def forward(self, inputs):
        # Ensure inputs is a numpy array for vectorized operations
        inputs = np.array(inputs)
        # Use numpy to perform vectorized ReLU
        self.output = np.maximum(0, inputs).tolist()

    def make_derivatives(self, inputs):
        # Ensure inputs is a numpy array for vectorized operations
        inputs = np.array(inputs)
        # Use numpy to perform vectorized derivative of ReLU
        self.derivatives = np.where(inputs > 0, 1, 0).tolist()
