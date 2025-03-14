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
