'''
Andre Palacio Braga Tivo 13835534
João Pedro Gonçalves Vilela 13731070
Lucas Muniz de Lima 13728941
Lucas Toupitzen Ferracin Garcia 11804164
'''




import random


# cria a matriz de pesos randômicos
def random_matrix(rows, cols):

    random.seed(10)
   
    matrix = []
    for _ in range(rows):
        row = [random.random() for _ in range(cols)]
        matrix.append(row)
    return matrix

# Inicializa os vetores de bias
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

    # Método para lidar com a passagem do sinal de input pela camada
    # Aplica os pesos e bias sobre o input e reserva o resultado na
    # propriedade output (pré função de ativação)
    def forward(self, input_signal):
       
        layer_in = []
        for j in range(self.n_neurons):
            weight_i = 0
            for i in range(len(input_signal)):
                weight_i += input_signal[i]*self.weights[i][j]
            layer_in.append(weight_i + self.biases[j])

        self.output = layer_in #z_in ou y_in

    # Métodos para aproveitar pesos e biases já calculados que serão carregados na camada
    # utilizado para realização dos testes
    def set_weights(self, weights):
        self.weights = weights

    def set_biases(self, biases):
        self.biases = biases