def one_hot_encoder(letter):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    num_classes = len(alphabet)
    encoding = [0] * num_classes
    if letter.upper() in alphabet:
        index = alphabet.index(letter.upper())
        encoding[index] = 1
    return encoding

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
            elements = line.strip().split(',  ')
            if len(elements) == 120:  # Check if the line has 120 elements
                data.append([int(element.strip()[0]) for element in elements])
    return data

def read_database():

    # Example usage:
    file_path_output = "Y_letra.txt"  # Replace with your file path
    data_output = read_exit_file(file_path_output)

    file_path = "X.txt"  # Replace with your file path
    data = read_entries_file(file_path)

    return data, data_output