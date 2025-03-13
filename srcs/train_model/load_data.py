import numpy as np
import pandas as pd
import os

def load_data(path: str) -> np.ndarray:
    '''
    Loads csv file to be read by the machine learning model

    Args:
    path: (str) - Path to the csv file

    Return:
    np.ndarray
    Return csv file as a np.ndarray dataframe
    '''
    # Check if path is a string
    if not isinstance(path, str):
        raise TypeError(f"{TypeError.__name__}: File path must be a string")

    # Check if the path is empty
    if not path.strip():  # Removes trailing spaces
        raise ValueError(f"{ValueError.__name__}: File path is empty")
    
    if not path.endswith(".csv"):
        raise FileNotFoundError(f"{FileNotFoundError.__name__}: '{path}'\
 is not a csv file")

    # Check if the file exists in path
    if not os.path.exists(path):
        raise FileNotFoundError(f"{FileNotFoundError.__name__}: File '{path}'\
 does not exist")

    # Check file permissions
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{PermissionError.__name__}: Permission denied:\
 Cannot read '{path}'")

    # Check if the file is empty
    if os.path.getsize(path) == 0:
        raise OSError(f"{OSError.__name__}: File '{path}'\
 is empty or corrupted")

    # Check if file path is a directory
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{FileNotFoundError.__name__}: '{path}'\
 does not exist or is a directory")
    
    data_set = pd.read_csv(path)
    # print(data_set.to_string())
    return data_set
