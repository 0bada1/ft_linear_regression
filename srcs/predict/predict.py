import sys
import os
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


def predict_price(w, b, mileage):
    predicted_price = linear_regression(w, b, mileage)
    return predicted_price


def load_data(path: str) -> np.ndarray:
    '''
    Loads model file to read coefficients

    Args:
    path: (str) - Path to the file

    Return:
    np.ndarray
    Return file as a np.ndarray dataframe
    '''
    # Check if path is a string
    if not isinstance(path, str):
        raise TypeError(f"{TypeError.__name__}: File path must be a string")

    # Check if the path is empty
    if not path.strip():  # Removes trailing spaces
        raise ValueError(f"{ValueError.__name__}: File path is empty")

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

    file = open(path, 'r')
    w = float(file.readline()[0:-1])
    b = float(file.readline())
    # print(file.to_string())
    return w, b


def main():
    try:
        if len(sys.argv) != 2:
            raise ValueError(f"{ValueError.__name__}: Wrong number of arguments")
        if not sys.argv[1].isdigit():
            raise ValueError(f"{ValueError.__name__}: Argument must be >= 0")
        w = 0
        b = 0
        car_mileage = float(sys.argv[1])
        path = "../train_model/model.txt"
        if os.path.exists(path):
            w, b = load_data(path)

    except ValueError as ErrMsg:
        print(ErrMsg)
        return -1
    predicted_price = predict_price(w, b, car_mileage)
    if predicted_price < 0:
        predicted_price = 0
    print(f"The price of your car is {predicted_price:.2f} USD")
    return 0


if __name__ == '__main__':
    main()
