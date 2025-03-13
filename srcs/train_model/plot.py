import matplotlib.pyplot as plt
import numpy as np


def cost_w_b_3D_graph(wb_history, cost_history):
	'''Cost and w (theta0) and b (theta1) 3D graph'''
	pass


def cost_w_graph(cost_history, wb_history, iterations):
	'''Cost and w (theta0) 2D graph'''
	# Get w from wb_history
	w_history = [w[0] for w in wb_history]
	print(w_history)

	# Create new graph	
	plt.figure()

	# Graph cost and w
	plt.plot(w_history, cost_history)
	plt.title("Cost on w (theta 0)")
	plt.xlabel("w (theta 0)"); plt.ylabel("Cost (USD)")


def cost_iteration_graph(cost_history, iterations):
	'''Cost to iterations 2D graph'''
	fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
	ax1.plot(cost_history[:100])
	ax2.plot(1000 + np.arange(len(cost_history[1000:])), cost_history[1000:])
	ax1.set_title("Cost vs iterations (start)"); ax2.set_title("Cost vs iterations (end)")
	ax1.set_xlabel("Iteration step"); ax2.set_xlabel("Iteration step")
	ax1.set_ylabel("Cost"); ax2.set_ylabel("Cost")


def scatter_data_graph(mileage_data, price_data):
	'''Price and mileage raw data scatter graph'''
	plt.figure()
	plt.scatter(price_data, mileage_data, color='blue', marker="x", label="Data points")
	plt.xlabel("Mileage (km)")
	plt.ylabel("Price (USD)")
	plt.title("Car Price Based on Mileage")
	plt.legend()


def model_fit_on_data_graph(mileage_data, price_data, w, b):
	'''Linear regression fit on price and mileage scatter graph'''
	plt.figure()
	# Your dataset (after scaling)
	x_scaled = np.array(price_data)  # Scaled x-values (e.g., car price)
	y_scaled = np.array(mileage_data)  # Scaled y-values (e.g., mileage)

	# Generate predictions
	x_line = np.linspace(min(x_scaled), max(x_scaled), 100)  # Generate 100 points between min and max x
	y_line = w * x_line + b  # Compute predicted y values

	# Plot the data
	plt.scatter(x_scaled, y_scaled, color='blue', marker='x', label='Data Points')  # Original data
	plt.plot(x_line, y_line, color='red', label='Model Fit')  # Regression line

	# Labels and title
	plt.xlabel("Mileage (km)")
	plt.ylabel("Price (USD)")
	plt.title("Linear Regression Model Fit")
	plt.legend()


def wb_contour_graph(w, b):
	'''w and b contour graph'''
	pass


def make_graphs(mileage_data, price_data, cost_history, wb_history, w, b, iterations):
	'''Call all graph functions from here'''
	scatter_data_graph(mileage_data, price_data)
	model_fit_on_data_graph(mileage_data, price_data, w, b)
	cost_iteration_graph(cost_history, iterations)
	plt.show()
