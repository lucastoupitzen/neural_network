def one_hot_encoder(letter):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    num_classes = len(alphabet)
    encoding = [0] * num_classes
    if letter.upper() in alphabet:
        index = alphabet.index(letter.upper())
        encoding[index] = 1
    return encoding

def letters_encoded():
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    corresp = {}
    for letter in alphabet:
        num_classes = len(alphabet)
        encoding = [0] * num_classes
        index = alphabet.index(letter.upper())
        encoding[index] = 1
        corresp[letter.upper()] = encoding
    return corresp


def read_exit_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[:-130]:
            letter = line.strip()
            encoding = one_hot_encoder(letter)
            data.append(encoding)
    return data



def read_entries_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[:-130]:  # Exclude the last 130 lines
            elements = line.strip().split(', ')
            if len(elements) == 120:  # Check if the line has 120 elements
                elements[119] = elements[119][:-1]
                data.append([int(element.strip()) for element in elements])
    return data

def read_database():

    # Example usage:
    file_path_output = "Y_letra.txt"  # Replace with your file path
    data_output = read_exit_file(file_path_output)

    file_path = "X.txt"  # Replace with your file path
    data = read_entries_file(file_path)

    return data, data_output

def read_exit_file_test(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[-130:]:
            letter = line.strip()
            encoding = one_hot_encoder(letter)
            data.append(encoding)
    return data

def read_entries_file_test(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[-130:]:  # Exclude the last 130 lines
            elements = line.strip().split(', ')
            if len(elements) == 120:  # Check if the line has 120 elements
                elements[119] = elements[119][:-1]
                data.append([int(element.strip()) for element in elements])
    return data

def read_test_database():

    # Example usage:
    file_path_output = "Y_letra.txt"  # Replace with your file path
    data_output = read_exit_file_test(file_path_output)

    file_path = "X.txt"  # Replace with your file path
    data = read_entries_file_test(file_path)

    return data, data_output

def read_entries_file_cross_validation(file_path, index):
    data = []
    validation = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Retirar as últimas 130 linhas
        lines = lines[:-130]
        
        # Calcular o tamanho da base e o tamanho do conjunto de validação
        tamanho_da_base = len(lines)
        num_folds = 10
        validation_size = int(tamanho_da_base / num_folds)
        
        # Definir o intervalo de validação
        validation_start = (validation_size * index)
        validation_end = validation_start + validation_size
        
        # Separar dados em conjuntos de treinamento e validação
        for i, line in enumerate(lines):
            elements = line.strip().split(', ')
            
            if len(elements) == 120:
                elements[119] = elements[119][:-1]
                processed_elements = [int(element.strip()) for element in elements]
                
                if validation_start <= i < validation_end:
                    validation.append(processed_elements)
                else:
                    data.append(processed_elements)
                    
    return data, validation

def read_exit_file_cross_validation(file_path, index):
    data = []
    validation = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Retirar as últimas 130 linhas
        lines = lines[:-130]
        
        # Calcular o tamanho da base e o tamanho do conjunto de validação
        tamanho_da_base = len(lines)
        num_folds = 10
        validation_size = int(tamanho_da_base / num_folds)
        
        # Definir o intervalo de validação
        validation_start = (validation_size * index)
        validation_end = validation_start + validation_size
        
        # Separar dados em conjuntos de treinamento e validação
        for i, line in enumerate(lines):
            letter = line.strip()
            encoding = one_hot_encoder(letter)
            
            if validation_start <= i < validation_end:
                validation.append(encoding)
            else:
                data.append(encoding)
                    
    return data, validation


def read_database_cross_validation(index):

    # Example usage:
    file_path_output = "Y_letra.txt"  # Replace with your file path
    data_output, validation_output = read_exit_file_cross_validation(file_path_output, index)

    file_path = "X.txt"  # Replace with your file path
    data, validation = read_entries_file_cross_validation(file_path, index)

    return data, data_output, validation, validation_output