'''
Andre Palacio Braga Tivo 13835534
João Pedro Gonçalves Vilela 13731070
Lucas Muniz de Lima 13728941
Lucas Toupitzen Ferracin Garcia 11804164
'''

import json
import matplotlib.pyplot as plt
from Layer import Layer
from activation_functions.Sigmoid import SigmoidActivation
from reading_final_project import read_database, read_database_cross_validation
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
# Implementação da estrutura de neurônios

number_of_neurons = 85

hidden_layer = Layer(120, number_of_neurons)
output_layer = Layer(number_of_neurons, 26)




# Define o mínimo erro quadrático aceitável que determina a convergência do modelo
min_error_threshold = 0.005

# Define o limite de épocas de treinamento
current_epoch = 0
max_epochs = 700

# guarda os erros e acuràcias obtidos em cada época
error_historic = []
accuracy_historic =  []
error_test = []

#momentum
# Utilizado como uma estratégia para acelerar a convergência do método e
# evitar mìnimos locais

# definição de um alpha para determinar o peso que o momentum terá na correção
alpha = 0.8
# matrizes de momentum que irão guardar o valor da última atualização de cada peso durante o treinamento
momentum_output = [[0 for _ in range(len(output_layer.weights[0]))] for _ in range(len(output_layer.weights))]
momentum_hidden = [[0 for _ in range(len(hidden_layer.weights[0]))] for _ in range(len(hidden_layer.weights))]
momentum_bias_output = [0] * len(output_layer.biases)
momentum_bias_hidden = [0] * len(hidden_layer.biases)

max_acuracia = 0

# Início do loop de treinamento
while True:

    # Embaralha os dados de input e output para diminuir a chance de 
    # mínimos locais
    X, y = shuffle_data(X, y)

    # inicialização da variável do erro quadrático calculado para cada época
    error = 0

    # Processamento da época
    for index in range(len(X)):

        ### Processo de FeedForward
        
        # Passagem do input sobre a camada escondida
        hidden_layer.forward(X[index])

        # Definição da função de ativação da camada
        hidden_activation = SigmoidActivation()

        # Aplicação da função de ativação sobre a entrada da camada escondida
        hidden_activation.forward(hidden_layer.output) #f(z_in)

        # A saída dos dados da função de ativação é a entrada dos dados para a camada de saída
        output_layer.forward(hidden_activation.output)

        

        # Definição da função de ativação da camada
        output_activation = SigmoidActivation()

        # # Aplicação da função de ativação sobre a entrada da camada de saída
        # output.layer.output = y_in
        output_activation.forward(output_layer.output) #f(y_in)

        # output_activation.output é a saída da rede neural
        result = output_activation.output


        ### Estimação do erro obtido na época

        # guarda para cada dado do conjunto de dados a diferença entre o valor esperado e o obtido
        error_vector = []
        
        for i in range(len(y[index])):
            error_vector.append(y[index][i] - result[i])
            # Erros ao quadrado -> Compõe o cálculo do erro quadrático mais adiante (linha 210)
            error += ((y[index][i] - result[i])**2)

        #vetor das derivadas das entradas na output layer
        y_in = output_layer.output
        # retorna a derivada da função de ativação para cada ponto da entrada na camada
        output_activation.make_derivatives(y_in)
        derivatives_vector = output_activation.derivatives


        #delta_k vector - reserva o termo de informação do erro de cada neurônio de saída
        delta_k = []
        for i in range(len(error_vector)):
            delta_k.append(error_vector[i] * derivatives_vector[i])

        
        ### Taxa de aprendizado
            
        # Em nossos treinamentos estamos usando taxa de aprendizagem variável
        if current_epoch < 300:
            learning_rate = 0.1
        else: learning_rate = 0.2
        


        # Cálculo do termo de correção do erro (delta_Wjk) para cada peso da camada de saída
        # delta_Wjk = learning_rate * delta_k * f(z_in)
        delta_Wjk = []
        for j in range(len(hidden_activation.output)):
            Wjk = []
            for k in range(len(delta_k)):
                Wjk.append(learning_rate * delta_k[k] * hidden_activation.output[j])
            delta_Wjk.append(Wjk)

        # termo de correção do bias 
        delta_bias = []
        for i in range(len(delta_k)):
            delta_bias.append(learning_rate * delta_k[i])

        # calculando o delta_inj 
        # as informações de erro vindas da camada acima (posterior).
        delta_inj = []
        for i in range(hidden_layer.n_neurons):
            somatorio = 0
            for j in range(len(delta_k)):
                somatorio += output_layer.weights[i][j] * delta_k[j]

            delta_inj.append(somatorio)

        #  derivada da função de ativação da camada oculta f'(z_in)
        z_in = hidden_layer.output
        hidden_activation.make_derivatives(z_in)
        derivatives_vector = hidden_activation.derivatives

        # calculando o delta j, que é o componente da correção do peso para a camada oculta 
        # que traz a informação da camada de saída
        delta_j = []
        for i in range(len(delta_inj)):
            delta_j.append(delta_inj[i] * derivatives_vector[i])

        # calcular o termo de correção dos pesos da camada oculta delta_Vij
        delta_Vij = []
        for i in range(len(X[index])):
            Vij = []
            for j in range(len(delta_j)):
                Vij.append(learning_rate * delta_j[j]* X[index][i])
            delta_Vij.append(Vij)

        # calcular o termo de correção do bias na camada oculta
        delta_V0j = []
        for i in range(len(delta_j)):
            delta_V0j.append(learning_rate * delta_j[i])

        

        # atualização dos pesos da camada de output e da oculta
        # Utilização do termo de momentum para acelerar a convergência
        for j in range(len(output_layer.weights)):
            for z in range(len(output_layer.weights[0])):
                output_layer.weights[j][z] += delta_Wjk[j][z] + (alpha * momentum_output[j][z])
                momentum_output[j][z] = delta_Wjk[j][z] + (alpha * momentum_output[j][z])

        for j in range(len(hidden_layer.weights)):
            for z in range(len(hidden_layer.weights[0])):
                hidden_layer.weights[j][z] += delta_Vij[j][z] + (alpha * momentum_hidden[j][z])
                momentum_hidden[j][z] = delta_Vij[j][z] + (alpha * momentum_hidden[j][z])

        #atualização dos bias 
        for i in range(len(output_layer.biases)):
            output_layer.biases[i] += delta_bias[i] + (alpha * momentum_bias_output[i])
            momentum_bias_output[i] = delta_bias[i] + (alpha * momentum_bias_output[i])

        for i in range(len(hidden_layer.biases)):
            hidden_layer.biases[i] += delta_V0j[i] + (alpha * momentum_bias_hidden[i])
            momentum_bias_hidden[i] = delta_V0j[i] + (alpha * momentum_bias_hidden[i])


    # Ao fim de cada época, calculamos o erro quedrático 
    quadratic_error = error / len(X) 


    ### Condições de parada
    # Checa se o erro já está no nível aceitável
    if quadratic_error < min_error_threshold:
        print("Training converged: Minimum error threshold reached.")
        break

    # Checa se já atingiu o máximo de épocas
    if current_epoch >= max_epochs:
        print("Training terminated: Maximum number of epochs reached.")
        break


    #### Mecananismos de salvamento dos resultados e plotagem de gráfico
    error_historic.append(quadratic_error)

    # Salva os pesos da época em arquivo json que será utilizado para realizar os testes de acurácia
    json_filename = f"output_final.json"

    # Create a dictionary to store the results
    results = {
        "epoca": current_epoch,
        "Pesos da camada escondida": hidden_layer.weights,
        "Bias da camada escondida": hidden_layer.biases,
        "Pesos da camada de output": output_layer.weights,
        "Bias da camada de output": output_layer.biases,
        "erros": error_historic,
        "accuracies": accuracy_historic,
        "erros_teste": error_test,
        "Erro quadrático": quadratic_error
    }

    # Write the results to the JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    # realiza os testes com os pesos obtidos na época, obtendo a acurácia 
    ### Os testes se encontram no arquivo testing.py
    acuracia, erro_teste = testing(hidden_neurons=number_of_neurons) # retorna a acurácia do modelo
    accuracy_historic.append(acuracia)

    if acuracia > max_acuracia:

        json_filename = f"output_final_max.json"

        # Create a dictionary to store the results
        results = {
            "epoca": current_epoch,
            "Pesos da camada escondida": hidden_layer.weights,
            "Bias da camada escondida": hidden_layer.biases,
            "Pesos da camada de output": output_layer.weights,
            "Bias da camada de output": output_layer.biases,
            "erros": error_historic,
            "accuracies": accuracy_historic,
            "erros_teste": error_test,
            "Erro quadrático": quadratic_error
        }

        # Write the results to the JSON file
        with open(json_filename, 'w') as json_file:
            json.dump(results, json_file, indent=4)

    error_test.append(erro_teste)

    # Log de controle do que está ocorrendo em cada época
    print(f"Current epoch: {current_epoch} quadratic_error: {quadratic_error} accuracy: {acuracia}")

    # Increment epoch counter
    current_epoch += 1


#### Teste final da modelo obtido, plotando a matriz de confusão
testing(isFinal=True, hidden_neurons=number_of_neurons)

#### Plotagem dos resultados
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

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss', color=color)
ax1.plot(range(1, current_epoch + 1), error_historic, color=color, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:green'
ax2.set_ylabel("Test's quadratic error", color=color)
ax2.plot(range(1, current_epoch + 1), error_test, color=color, label="Test's quadratic error")
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Training Loss and Test Loss Over Epochs')
fig.tight_layout()
plt.savefig('training_loss_and_test.png')
plt.clf()

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss', color=color)
ax1.plot(range(101, current_epoch + 1), error_historic[100:], color=color, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:green'
ax2.set_ylabel("Test's quadratic error", color=color)
ax2.plot(range(101, current_epoch + 1), error_test[100:], color=color, label="Test's quadratic error")
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Training Loss and Test Loss Over last Epochs')
fig.tight_layout()
plt.savefig('training_loss_and_test_last.png')
plt.clf()


# Create PDF report
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Final Training Report', 0, 1, 'C')
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

# Gera o arquivo PDF
pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 12)


number_of_epochs = current_epoch

intro_text = f"""

Andre Palacio Braga Tivo 13835534
João Pedro Gonçalves Vilela 13731070
Lucas Muniz 13728941
Lucas Toupitzen Ferracin Garcia 11804164

Number of neurons: {number_of_neurons}
Number of epochs: {number_of_epochs}
learning_rate: Váriavel: 0.1 até a época 300 e 0.2 após isso
Momentum (alpha): {alpha}


Final quadratic error: {quadratic_error}
Final acurracy level: {"{:.2%}".format(acuracia)}

"""
pdf.multi_cell(0, 10, intro_text)

pdf.add_page()
pdf.chapter_title("Training Loss and Accuracy Over Epochs")
pdf.add_image('training_loss_and_accuracy.png')

pdf.add_page()
pdf.chapter_title("Training Loss and Test Loss Over Epochs")
pdf.add_image('training_loss_and_test.png')

pdf.add_page()
pdf.chapter_title("Training Loss and Test Loss Over Last Epochs")
pdf.add_image('training_loss_and_test_last.png')

pdf.add_page()
pdf.chapter_title("Confusion Matrix")
pdf.add_image('confusion_matrix.png')

pdf.output(f'report_final.pdf', 'F')