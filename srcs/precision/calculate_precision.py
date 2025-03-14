import os
import pandas as pd
import numpy as np


def linear_regression(w: float, b: float, x: float) -> float:
    '''Linear Regression
    Fw,b(x) = wx + b

    Args:
    w: (float) - Slope in Fw,b(x) = wx + b
    b: (float) - Y intercept in Fw,b(x) = wx + b
    x: (float) - x value of data point

    Return:
    float
    f_wb = w * x + b
    '''
    f_wb = w * x + b
    return f_wb


def calculate_precision(w: float, b: float, x: list[float | int], y: list[float | int]) -> float:
    '''Calculate precision of the trained model in percentage
    R2 = 1 - SStot/SSres

    Args:
    w: (float) - Slope in Fw,b(x) = wx + b
    b: (float) - Y intercept in Fw,b(x) = wx + b
    x: (list[float | int])	- Price data set
    y: (list[float | int])	- Mileage data set

    Return:
    float
    Returns the R2 (R-squared (R2) score)
    '''
    # Manual R² calculation
    y_mean = sum(y) / len(y)  # Compute mean of actual y values

    # Compute SS_res (sum of squared residuals)
    SS_res = sum((y[i] - linear_regression(w, b, x[i]))**2 for i in range(len(y)))

    # Compute SS_tot (total sum of squares)
    SS_tot = sum((y[i] - y_mean)**2 for i in range(len(y)))

    # Compute R²
    r2_manual = 1 - (SS_res / SS_tot)

    # Convert to percentage
    precision_percentage = r2_manual * 100

    return precision_percentage


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


def main():
    path = "../train_model/model.txt"
    if os.path.exists(path):
        model = open(path, 'r')
        w = float(model.readline()[0:-1])
        b = float(model.readline())
    else:
        print(f"1. {path} file doesn't exist")
        return -1
    path = "../../data/data.csv"
    if os.path.exists(path):
        data = load_data(path)
        x = data['km']
        y = data['price']
    else:
        print(f"2. {path} file doesn't exist")
        return -2

    precision = calculate_precision(w, b, x, y)
    print(f"Model Precision (Manual R²): {precision:.2f}%")


if __name__ == '__main__':
    main()
