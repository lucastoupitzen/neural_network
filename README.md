# Trabalho de implementação de uma rede neural no contexto da disciplina de ACH2016 - Inteligência Artificial

# Alunos responsáveis:

Andre Palacio Braga Tivo 13835534
João Pedro Gonçalves Vilela 13731070
Lucas Muniz de Lima 13728941
Lucas Toupitzen Ferracin Garcia 11804164


# Para instalar pacotes necessários:

pip install -r requirements.txt

# Descrição dos arquivos e pastas

## Treinamento

O treinamento da rede neural está instanciado no arquivo trainning.py, utilizando as classes Layer de Layer.py e SigmoidActivation de activation_functions/Sigmoid.py.

## Leitura dos dados

O processamento dos arquivos txt para o formato de input e output utilizados pela rede neural se encontra no arquivo
reading_final_project.py

## Testes

Os testes são executados no arquivo testing.py, aonde são obtidos os valores de acurácia, erro_quadrático dos testes e também a matriz de confusão

## Relatórios de treinamento

Os arquivos pdf que resumem os treinamentos tanto final quanto os do cross-validation estão na pasta reports_final

## Pesos e bias de treinamento

Os pesos e bias de treinamento estão reservados na pasta outputs_final

## Log do console de exemplo do controle de treinamento

Arquivo log_final.txt