import json
from Layer import Layer
from activation_functions.Relu import ReluActivation
from activation_functions.Sigmoid import SigmoidActivation
from activation_functions.Softmax import SoftmaxActivation
from reading_letters_fausett import read_csv_data_letters_fausset
from reading_final_project import read_test_database, letters_encoded
import seaborn as sns
import matplotlib.pyplot as plt

def testing(isFinal: bool = False, hidden_neurons = 0):

    hidden_layer = Layer(120, hidden_neurons)
    # output possui 7 atributos
    output_layer = Layer(hidden_neurons, 26)


    X, y = read_test_database()

    # Define the filename for the JSON file
    json_filename = "output.json"

    # Read the results from the JSON file
    with open(json_filename, 'r') as json_file:
        results = json.load(json_file)

    # Extract the weights and biases from the results dictionary
    hidden_layer_weights = results["Pesos da camada escondida"]
    hidden_layer_biases = results["Bias da camada escondida"]
    output_layer_weights = results["Pesos da camada de output"]
    output_layer_biases = results["Bias da camada de output"]

    hidden_layer.set_weights(hidden_layer_weights)
    hidden_layer.set_biases(hidden_layer_biases)

    output_layer.set_weights(output_layer_weights)
    output_layer.set_biases(output_layer_biases)

    sucesso = 0
    erro = 0

    confusion_matrix = [[0 for _ in range(26)] for _ in range(26)]

    for index_test in range(len(X)):


        hidden_layer.forward(X[index_test])

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

        expected_output = y[index_test]
        
        if result.index(max(result)) == expected_output.index(1): sucesso += 1
        else: erro += 1

        if isFinal:

            confusion_matrix[expected_output.index(1)][result.index(max(result))] += 1
            

    if isFinal:
        # Generate and save confusion matrix plot
        plt.figure(figsize=(14, 10))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), yticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        plt.clf()

        


    # print("Sucessos: ", sucesso)
    # print("Erro: ", erro)
    # print("Porcentagem de sucesso: ", (sucesso)/(sucesso + erro))
    

    return (sucesso)/(sucesso + erro)