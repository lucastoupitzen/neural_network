import json
from Layer import Layer
from activation_functions.Sigmoid import SigmoidActivation
from reading_final_project import read_test_database
import seaborn as sns
import matplotlib.pyplot as plt

def testing(isFinal: bool = False, hidden_neurons = 0, validation_input = None, validation_output = None, fold = 0):

    # Instancia as camadas do modelo
    hidden_layer = Layer(120, hidden_neurons)
    output_layer = Layer(hidden_neurons, 26)


    X, y = validation_input, validation_output

    # Arquivo json de onde as informações serão retiradas
    json_filename = f"output_fold_{fold}x.json"

    with open(json_filename, 'r') as json_file:
        results = json.load(json_file)

    # Extração dos pesos e dos biases obtidos em treinamento
    hidden_layer_weights = results["Pesos da camada escondida"]
    hidden_layer_biases = results["Bias da camada escondida"]
    output_layer_weights = results["Pesos da camada de output"]
    output_layer_biases = results["Bias da camada de output"]

    hidden_layer.set_weights(hidden_layer_weights)
    hidden_layer.set_biases(hidden_layer_biases)

    output_layer.set_weights(output_layer_weights)
    output_layer.set_biases(output_layer_biases)

    # cálculo da acurácia do modelo
    sucesso = 0
    erro = 0

    ## Inicia a estrutura da matriz de confusão, uma matriz[26][26] com 0 em todos os campos
    confusion_matrix = [[0 for _ in range(26)] for _ in range(26)]

    # inicialização da variável do erro quadrático calculado para cada época
    error = 0

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

        error_vector = []
            
        for i in range(len(y[index_test])):
            error_vector.append(y[index_test][i] - result[i])
            # Erros ao quadrado -> Compõe o cálculo do erro quadrático mais adiante (linha 210)
            error += ((y[index_test][i] - result[i])**2)

        expected_output = y[index_test]
        
        # As saídas esperadas estão no formato one-hot-encoded
        # Dessa forma, comparamos a posição do maior valor do vetor de saída com a posição do 1
        # no vetor de saídas esperadas, indicando a maior semelhança possível dada pelo modelo
        # Se forem iguais, temos um sucesso, caso contrário, um erro
        if result.index(max(result)) == expected_output.index(1): sucesso += 1
        else: erro += 1

        # Sendo o teste final, queremos atualizar nossa matriz de confusão
        if isFinal:
            # Soma 1 ao campo [esperado][obtido]
            confusion_matrix[expected_output.index(1)][result.index(max(result))] += 1

    quadratic_error = error / len(X) 


            
    # plotagem da matriz de confusão
    if isFinal:
        # Generate and save confusion matrix plot
        plt.figure(figsize=(14, 10))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), yticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        plt.clf()
    
    # retorna a acurácia
    return (sucesso)/(sucesso + erro) , quadratic_error