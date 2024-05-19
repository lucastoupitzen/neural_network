import json
import matplotlib.pyplot as plt
from Layer import Layer
from activation_functions.Relu import ReluActivation
from activation_functions.Sigmoid import SigmoidActivation
from reading_letters_fausett import read_csv_data_letters_fausset
from reading_final_project import read_database
import numpy as np
# file_path = "caracteres-limpo.csv"
# X, y = read_csv_data_letters_fausset(file_path)

# Define a method to shuffle the training data
def shuffle_data(X, y):
    

    X = np.array(X)
    y = np.array(y)
    # Generate random indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Shuffle X and y using the same indices
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    return X_shuffled, y_shuffled

X, y = read_database()

# input possui 63 atributos
hidden_layer = Layer(120, 50)
# output possui 7 atributos
output_layer = Layer(50, 26)

# Define o mínimo erro quadrático aceitável
min_error_threshold = 0.001

# Define o limite de épocas de treinamento
current_epoch = 0
max_epochs = 600

# guarda os erros obtidos em cada época
error_history = []

while True:
    # Perform training epoch
    X, y = shuffle_data(X, y)
    error = 0
    for index in range(len(X)):
        
        hidden_layer.forward(X[index])

        #próximo passo é aplicar a função de ativação ao output dessa camada
        hidden_activation = SigmoidActivation()
        hidden_activation.forward(hidden_layer.output) #f(z_in)

        #hidden_activation.output é o input da próxima camada
        output_layer.forward(hidden_activation.output)

        #output.layer.output = y_in
        output_activation = SigmoidActivation()
        output_activation.forward(output_layer.output) #f(y_in)

        #output_activation.output é a saída da rede neural
        result = output_activation.output

        error_vector = []
        
        for i in range(len(y[index])):
            error_vector.append(y[index][i] - result[i])
            error += ((y[index][i] - result[i])**2)
    
        #vetor das derivadas das entradas na output layer
        y_in = output_layer.output
        output_activation.make_derivatives(y_in)
        derivatives_vector = output_activation.derivatives

        #delta_k vector - reserva o termo de informação do erro de cada neurônio de saída
        delta_k = []
        for i in range(len(error_vector)):
            delta_k.append(error_vector[i] * derivatives_vector[i])

        #termo de correção do erro (delta_Wjk)

        learning_rate = 0.2 #taxa de aprendizado
        ## delta_Wjk = learning_rate * delta_k * f(z_in)

        delta_Wjk = []
        for j in range(len(hidden_activation.output)):
            Wjk = []
            for k in range(len(delta_k)):
                Wjk.append(learning_rate * delta_k[k] * hidden_activation.output[j])
            delta_Wjk.append(Wjk)

        #termo de correção do bias 
        delta_bias = []
        for i in range(len(delta_k)):
            delta_bias.append(learning_rate * delta_k[i])

        #calculando o delta_inj 
        delta_inj = []
        for i in range(hidden_layer.n_neurons):
            somatorio = 0
            for j in range(len(delta_k)):
                somatorio += output_layer.weights[i][j] * delta_k[j]

            delta_inj.append(somatorio)

        #multiplicando pela derivada da função de ativação f'(z_in)
        z_in = hidden_layer.output
        hidden_activation.make_derivatives(z_in)
        derivatives_vector = hidden_activation.derivatives

        delta_j = []
        for i in range(len(delta_inj)):
            delta_j.append(delta_inj[i] * derivatives_vector[i])

        #calcular o termo de correção do peso delta_Vij
        delta_Vij = []
        for i in range(len(X[index])):
            Vij = []
            for j in range(len(delta_j)):
                Vij.append(learning_rate * delta_j[j]* X[index][i])
            delta_Vij.append(Vij)

        #calcular o termo de correção do bias na hidden_layer
        delta_V0j = []
        for i in range(len(delta_j)):
            delta_V0j.append(learning_rate * delta_j[i])

        #atualização dos pesos da camada de output e da oculta
        for j in range(len(output_layer.weights)):
            for z in range(len(output_layer.weights[0])):
                output_layer.weights[j][z] += delta_Wjk[j][z]

        for j in range(len(hidden_layer.weights)):
            for z in range(len(hidden_layer.weights[0])):
                hidden_layer.weights[j][z] += delta_Vij[j][z]

        #atualização dos bias 
        for i in range(len(output_layer.biases)):
            output_layer.biases[i] += delta_bias[i]

        for i in range(len(hidden_layer.biases)):
            hidden_layer.biases[i] += delta_V0j[i]
    
    #acabou a época
    quadratic_error = error / len(X)
  
    # Compute training loss and update model parameters
    
    
    # Checa se o erro já está no nível aceitável
    if quadratic_error < min_error_threshold:
        print("Training converged: Minimum error threshold reached.")
        break
    
    # Checa se já atingiu o máximo de épocas
    if current_epoch >= max_epochs:
        print("Training terminated: Maximum number of epochs reached.")
        break

    error_history.append(quadratic_error)
    
    if current_epoch % 10 == 0:
        print(f"Currenty epoch: {current_epoch} quadratic_error:{quadratic_error}")

        json_filename = "output.json"

        # Create a dictionary to store the results
        results = {
            "epoca": current_epoch,
            "Pesos da camada escondida": hidden_layer.weights,
            "Bias da camada escondida": hidden_layer.biases,
            "Pesos da camada de output": output_layer.weights,
            "Bias da camada de output": output_layer.biases,
            "Erro quadrático": quadratic_error
        }

        # Write the results to the JSON file
        with open(json_filename, 'w') as json_file:
            json.dump(results, json_file, indent=4)
            
    # Increment epoch counter
    current_epoch += 1

    


# Plot the learning curve
plt.plot(range(1, current_epoch + 1), error_history, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Average Error')
plt.legend()
plt.grid(True)
plt.show()


json_filename = "output.json"

# Create a dictionary to store the results
results = {
    "Pesos da camada escondida": hidden_layer.weights,
    "Bias da camada escondida": hidden_layer.biases,
    "Pesos da camada de output": output_layer.weights,
    "Bias da camada de output": output_layer.biases,
    "Erro quadrático": quadratic_error
}

# Write the results to the JSON file
with open(json_filename, 'w') as json_file:
    json.dump(results, json_file, indent=4)
