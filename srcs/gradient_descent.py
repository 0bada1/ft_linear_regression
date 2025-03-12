import numpy as np
import math
from linear_regression import linear_regression
from cost import squared_error_cost


def compute_gradient(x: list[int | float], y: list[int | float], w: float, b: float) -> list[float]:
	'''
	Computes gradient using partial differentiation of the cost function
	For w: (1/m) * Sum of training data(fw,b(x[i]) - y[i]) * x[i]
	For b: (1/m) * Sum of training data(fw,b(x[i]) - y[i])
	
	Args:
	x: (list[float | int])	- Price data set
	y: (list[float | int])	- Mileage data set
	w: (float)				- Slope in wx + b
	b: (float)				- Y intercept in wx + b

	Returns:
	list[float]
	Returns 2 variables: new w, new b
	'''
	m = x.shape[0]
	df_dw = 0
	df_db = 0
	for i in range(m):
		f_wb = linear_regression(w, b, x[i])
		tmp_df_dw = (f_wb - y[i]) * x[i]
		tmp_df_db = (f_wb - y[i])
		df_dw += tmp_df_dw
		df_db += tmp_df_db
	df_dw = df_dw/m
	df_db = df_db/m
	return df_dw, df_db


def gradient_descent(x: list[float | int], y: list[int | float], m: int, alpha: float, w: float, b: float, iterations: int=10000) -> list[float]:
	'''
	Gradient Descent to find w and b (theta0 and thetha1). It updates w,b by taking
	iterations steps with the learning rate

	Args:
	x: (list[float | int])	- Price data set
	y: (list[float | int])	- Mileage data set
	m: (int)				- Number of training examples
	alpha: (float)			- Learning rate
	w: (float)				- Slope in wx + b
	b: (float)				- Y intercept in wx + b
	iterations: (float)		- Number of iterations

	Returns:
	list[float]
	Returns 2 variables: new w, new b
	'''
	cost_history = []
	wb_history = []
	# Fw,b(x[i]) - y[i] is the cost of a single data point
	for i in range(iterations):
		df_dw, df_db = compute_gradient(x, y, w, b)
		# Simultaneous update of w (theta 0) and b (thetha 1)
		# tmp_w = w - alpha * ((1/m)*(linear_regression(w, b, y[i]) - y[i]) * x[i])  # w - α (∂/∂w J(w,b))
		# tmp_b = b - alpha * ((1/m)*(linear_regression(w, b, y[i]) - y[i]))  # b - α (∂/∂b J(w,b))
		w = w - alpha * df_dw  # w = w - α (∂/∂w J(w,b))
		b = b - alpha * df_db  # b = b - α (∂/∂b J(w,b))
		if i < 10000:  # Prevent resource exauhstion
			cost_history.append(squared_error_cost(m, w, b, x, y))
			wb_history.append([w, b])
		if i <= iterations:  # Print cost every iteration
		# if i % math.ceil(iterations/10) == 0:  # Print cost every 10 iterations
			print(f"Iteration: {i:4}: Cost: {cost_history[-1]:0.2e} ",
		 		  f"df_dw: {df_dw: 0.3e}, df_db: {df_db: 0.3e} ",
				  f"w: {w: 0.3e}, b: {b: 0.5e}")
	return w, b, cost_history, wb_history
