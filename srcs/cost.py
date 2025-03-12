import numpy as np
from linear_regression import linear_regression

def squared_error_cost(m: int, w: float, b: float, x: list[int | float], y: list[int | float]) -> float:
	'''
	Squared error cost function
	J(w,b): (1/2m) * (Fw,b(x[i]) - y[i])**2

	Args:
	x: (list[float | int])	- Price data set
	y: (list[float | int])	- Mileage data set
	m: (int)				- Number of training examples
	w: (float)				- Slope in wx + b
	b: (float)				- Y intercept in wx + b

	Returns:
	float
	Returns cost value
	'''
	cost = 0
	# m = len(arr)
	for i in range(m):
		f_wb = linear_regression(w, b, x[i])
		cost += (f_wb - y[i])**2
	total_cost = (1/(2*m)) * cost
	return total_cost
