import json
import matplotlib.pyplot as plt
from Layer import Layer
from activation_functions.Sigmoid import SigmoidActivation
from reading_final_project import read_database
import numpy as np
from testing import testing
from fpdf import FPDF


# Método para embaralhar a entrada no treinamento a cada nova época
# utilizado para minimizar a chance de mínimos locais
def shuffle_data(X, y):
    

    X = np.array(X)
    y = np.array(y)
    # Generate random indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Shuffle X and y using the same indices
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    return X_shuffled.tolist(), y_shuffled.tolist()

X, y = read_database()

number_of_neurons = 76

# input possui 63 atributos
hidden_layer = Layer(120, number_of_neurons)
# output possui 7 atributos
output_layer = Layer(number_of_neurons, 26)

# Define the filename for the JSON file
# json_filename = "output63n.json"

# # Read the results from the JSON file
# with open(json_filename, 'r') as json_file:
#     results = json.load(json_file)

# # Extract the weights and biases from the results dictionary
# hidden_layer_weights = results["Pesos da camada escondida"]
# hidden_layer_biases = results["Bias da camada escondida"]
# output_layer_weights = results["Pesos da camada de output"]
# output_layer_biases = results["Bias da camada de output"]

# hidden_layer.set_weights(hidden_layer_weights)
# hidden_layer.set_biases(hidden_layer_biases)

# output_layer.set_weights(output_layer_weights)
# output_layer.set_biases(output_layer_biases)

# Define o mínimo erro quadrático aceitável
min_error_threshold = 0.005

# Define o limite de épocas de treinamento
current_epoch = 0
max_epochs = 300

# guarda os erros obtidos em cada época
error_historic = []
accuracy_historic = []



#momentum
alpha = 0.8
momentum_output = [[0 for _ in range(len(output_layer.weights[0]))] for _ in range(len(output_layer.weights))]
momentum_hidden = [[0 for _ in range(len(hidden_layer.weights[0]))] for _ in range(len(hidden_layer.weights))]
momentum_bias_output = [0] * len(output_layer.biases)
momentum_bias_hidden = [0] * len(hidden_layer.biases)

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
        learning_rate = 0.8
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
                output_layer.weights[j][z] += delta_Wjk[j][z] + (alpha * momentum_output[j][z])
                #momentum_output[j][z] = delta_Wjk[j][z] + (alpha * momentum_output[j][z])

        for j in range(len(hidden_layer.weights)):
            for z in range(len(hidden_layer.weights[0])):
                hidden_layer.weights[j][z] += delta_Vij[j][z] + (alpha * momentum_hidden[j][z])
                #momentum_hidden[j][z] = delta_Vij[j][z] + (alpha * momentum_hidden[j][z])

        #atualização dos bias 
        for i in range(len(output_layer.biases)):
            output_layer.biases[i] += delta_bias[i] + (alpha * momentum_bias_output[i])
            #momentum_bias_output[i] = delta_bias[i] + (alpha * momentum_bias_output[i])

        for i in range(len(hidden_layer.biases)):
            hidden_layer.biases[i] += delta_V0j[i] + (alpha * momentum_bias_hidden[i])
            #momentum_bias_hidden[i] = delta_V0j[i] + (alpha * momentum_bias_hidden[i])
    
    
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

    error_historic.append(quadratic_error)

    
    json_filename = "output.json"

    # Create a dictionary to store the results
    results = {
        "epoca": current_epoch,
        "Pesos da camada escondida": hidden_layer.weights,
        "Bias da camada escondida": hidden_layer.biases,
        "Pesos da camada de output": output_layer.weights,
        "Bias da camada de output": output_layer.biases,
        "erros": error_historic,
        "accuracies": accuracy_historic,
        "Erro quadrático": quadratic_error
    }

    # Write the results to the JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    
            
    test = testing(hidden_neurons=number_of_neurons)
    accuracy_historic.append(test)
    if test > 0.75:

        json_filename = "output_final.json"

        # Create a dictionary to store the results
        results = {
            "epoca": current_epoch,
            "Pesos da camada escondida": hidden_layer.weights,
            "Bias da camada escondida": hidden_layer.biases,
            "Pesos da camada de output": output_layer.weights,
            "Bias da camada de output": output_layer.biases,
            "erros": error_historic,
            "accuracies": accuracy_historic,
            "Erro quadrático": quadratic_error
        }

        # Write the results to the JSON file
        with open(json_filename, 'w') as json_file:
            json.dump(results, json_file, indent=4)

    


    print(f"Current epoch: {current_epoch} quadratic_error: {quadratic_error} accuracy: {test}")

    # Increment epoch counter
    current_epoch += 1

    

testing(isFinal=True, hidden_neurons=number_of_neurons)

# Plot and save the combined learning curves
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss', color=color)
ax1.plot(range(1, current_epoch + 1), error_historic, color=color, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(range(1, current_epoch + 1), accuracy_historic, color=color, label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color)


plt.title('Training Loss and Accuracy Over Epochs')
fig.tight_layout()
plt.savefig('training_loss_and_accuracy.png')
plt.clf()

# Create PDF report
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Training Report', 0, 1, 'C')
        self.ln(10)
        
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1)
        self.ln(5)
        
    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()
        
    def add_image(self, image_path):
        
        self.image(image_path, x = 20, y = 60, w = 180)

# Generate PDF
pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 12)


number_of_epochs = current_epoch

intro_text = f"""
Number of neurons: {number_of_neurons}
Number of epochs: {number_of_epochs}
learning_rate inicial: {learning_rate}


Final quadratic error: {quadratic_error}
Final acurracy level: {"{:.2%}".format(test)}

Using Momentum
alpha = {alpha}
"""
pdf.multi_cell(0, 10, intro_text)

pdf.add_page()
pdf.chapter_title("Training Loss and Accuracy Over Epochs")
pdf.add_image('training_loss_and_accuracy.png')

pdf.add_page()
pdf.chapter_title("Confusion Matrix")
pdf.add_image('confusion_matrix.png')

pdf.output(f'report_{number_of_neurons}n_{number_of_epochs}epochs.pdf', 'F')