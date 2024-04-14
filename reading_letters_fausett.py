import csv

def read_csv_data_letters_fausset(file_path):
    X = []
    y = []


    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(file)
        for row in reader:
            # Convert each element to int and separate inputs (X) and outputs (y)
            X.append([int(x) for x in row[:63]])
            y.append([int(x) if x == "1" else 0 for x in row[63:]])

    return X, y
