import math
from .Activation_interface import Activation_interface
import numpy as np

class SoftmaxActivation(Activation_interface):
    
    def forward(self, inputs):
        self.output = self.softmax_function(inputs)
        
    def make_derivatives(self, inputs):
        self.derivatives = self.softmax_derivative(inputs)
        
    @classmethod
    def softmax_function(cls, inputs):

        
        exp_values = np.exp(inputs - np.max(inputs)).tolist() 
        sum_exp_values = sum(exp_values)
        return [exp_val / sum_exp_values for exp_val in exp_values]
    
    @classmethod
    def softmax_derivative(cls, softmax_output):

        derivatives = []
        for i in range(len(softmax_output)):
            derivative = softmax_output[i] * (1 - softmax_output[i])
            derivatives.append(derivative)
        return derivatives
    

 