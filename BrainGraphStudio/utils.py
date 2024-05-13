import json
import os
from itertools import compress

MAIN_PATH = os.path.abspath(__file__)
BGS_DIR = os.path.dirname(MAIN_PATH)

class MaskableList(list):
    def __init__(self, iterable=[]):
        super(MaskableList, self).__init__(iterable)
    
    def __getitem__(self, index):
        # Try to return a single item or a slice using the parent class' method
        try:
            return super(MaskableList, self).__getitem__(index)
        except TypeError:
            # Handle the case where index is an iterable
            if all(isinstance(i, bool) for i in index):
                # Treat as a boolean mask
                return MaskableList(compress(self, index))
            else:
                # Treat as a list of indices
                return MaskableList([self[i] for i in index])

def write_dict_to_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def merge_nested_dicts(d, merged=None):
    """
    Merge a nested dictionary into a single-level dictionary without modifying parent keys.
    
    :param d: The dictionary to merge
    :param merged: The resulting merged dictionary
    :return: A single-level merged dictionary
    """
    if merged is None:
        merged = {}

    for key, value in d.items():
        if isinstance(value, dict):
            merge_nested_dicts(value, merged)  # Recursively merge nested dictionaries
        else:
            merged[key] = value  # Add key to merged dictionary

    return merged

def merge_dictionaries(dict_list):
    """
    Merge a list of dictionaries into a single dictionary.
    
    :param dict_list: List of dictionaries to merge
    :return: Merged dictionary with all key-value pairs from the list
    """
    merged = {}
    for d in dict_list:
        if isinstance(d, dict):
            merged.update(d)  # Add all key-value pairs to the merged dictionary
        else:
            raise ValueError("The list should contain dictionaries only.")
    return merged

def convert_numeric_values(data):
    """
    Converts all numeric values in a possibly nested dictionary into correct types (int or float).
    
    Args:
        data (dict): A dictionary that may contain nested dictionaries or lists.
    
    Returns:
        dict: A new dictionary with numeric values converted to int or float.
    """
    if isinstance(data, dict):
        return {k: convert_numeric_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numeric_values(v) for v in data]
    elif isinstance(data, str):
        try:
            # Try to convert to an integer
            return int(data)
        except ValueError:
            try:
                # If not an integer, try to convert to a float
                return float(data)
            except ValueError:
                # If not numeric, return the original string
                return data
    else:
        # Return other data types as is
        return data
