import json
import os
import numpy as np
import csv
import random

def generate_and_save_binary_values():
    # Generate a list of ten random binary values (0 or 1)
    binary_values = [random.choice([0, 1]) for _ in range(10)]

    # Save them to a CSV file named test_labels.csv in the current working directory
    with open('test_labels.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(binary_values)

    return binary_values

def detect_delimiter(file_path, num_lines=10):
    with open(file_path, 'r') as file:
        lines = [file.readline().strip() for _ in range(num_lines)]

    delimiters = [',', '\t', ';', '|', ' ', '\n']
    delimiter_count = {delim: 0 for delim in delimiters}

    for line in lines:
        for delim in delimiters:
            delimiter_count[delim] += line.count(delim)

    inferred_delimiter = max(delimiter_count, key=delimiter_count.get)
    return inferred_delimiter

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def process_label_file(file_path):
    delimiter = detect_delimiter(file_path)
    data = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        for row in reader:
            if not all(is_number(field) for field in row):
                return False, []

            # Convert each field to an integer and add to the data list
            data.append([int(field) for field in row])

    return True, data[0]

def is_binary(a):
    a = np.asarray(a)
    if a.size == 0:
        return False  # Handle empty array case
    return ((a == 0) | (a == 1)).all()

def custom_json_dump(obj, indent=2):
    if isinstance(obj, dict):
        output = "{\n"
        comma = ""
        for key, value in obj.items():
            output += f"{comma}{' ' * (indent + 4)}{json.dumps(key)}: {custom_json_dump(value, indent + 4)}"
            comma = ",\n"
        output += f"\n{' ' * indent}}}"
        return output
    elif isinstance(obj, list):
        output = "["
        comma = ""
        for item in obj:
            output += f"{comma}{custom_json_dump(item, indent)}"
            comma = ", "
        output += "]"
        return output
    else:
        return json.dumps(obj)
    
def is_mat(file):
    return check_file_extension(file,".mat")

def is_mat_flist(file):
    return is_flist(file, type_ = ".mat")

def is_npy_flist(file):
    return is_flist(file, type_ = ".npy")

def is_flist(file, type_ = None):
    if (not check_file_extension(file, ".txt")) and (not check_file_extension(file, ".flist")):
        return False
    else:
        with open(file) as f:
            files = f.readlines()
            for line in files:
                if (not os.path.exists(line.strip()) or 
                    (type_ is not None and not check_file_extension(line.strip(), type_))):
                    return False
        return True
    
def is_npy(file):
    return check_file_extension(file,".npy")

def check_file_extension(filename, extension):
    return os.path.splitext(filename)[1] == extension

def load_npy_flist(self, file):
    with open(file) as f:
        arrays = np.array([np.load(fname.strip()) for fname in f.readlines()])
    return arrays
