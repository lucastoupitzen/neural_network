import random

def random_matrix(rows, cols):

    random.seed(10)
   
    matrix = []
    for _ in range(rows):
        row = [random.random() for _ in range(cols)]
        matrix.append(row)
    return matrix

def biases_array(inicialization_value, size):

    array = []
    for _ in range(size):
        array.append(inicialization_value)
    return array


class Layer():

    def __init__(self, input_dimension, n_neurons) -> None:
        self.n_neurons = n_neurons
        self.weights = random_matrix(input_dimension, n_neurons)
        self.biases = biases_array(1, n_neurons)

    def forward(self, input_signal):
       
        layer_in = []
        for j in range(self.n_neurons):
            weight_i = 0
            for i in range(len(input_signal)):
                weight_i += input_signal[i]*self.weights[i][j]
            layer_in.append(weight_i + self.biases[j])

        self.output = layer_in #z_in ou y_in

    def set_weights(self, weights):
        self.weights = weights

    def set_biases(self, biases):
        self.biases = biases