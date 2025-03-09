import numpy as np
import matplotlib.pyplot as plt
import math
from load_data import load_data
# from cost import squared_error_cost
from gradient_descent import gradient_descent
# from linear_regression import linear_regression
# import seaborn as sns

def main():
	'''Load data, check for errors, calculate cost, calculate gradient descent
	calculate linear regression, predict new car price, draw graph'''
	data = load_data("../data/data.csv")
	# sns.pairplot(data)
	price_data = data['price']  # x data
	mileage_data = data['km']  # y data
	print(f"Mileage = \n{mileage_data}")
	print(f"Price= \n{price_data}")
	plt.scatter(price_data, mileage_data, color='blue', marker="x")
	plt.xlabel("Price (1000 USD)")
	plt.ylabel("Mileage (km)")
	plt.title("Car Price Based on Mileage")
	plt.legend(loc='lower right')
	# cost = squared_error_cost(len(data), w, b, price_data, mileage_data)
	print(f"price_data: \n{price_data}\nmileage_data: \n{mileage_data}\nm: {len(price_data)}\nalpha(learning rate): {0.01}\nw: {0}\nb: {0}\niterations: {10000}")
	print(f"Price[{len(price_data) - 1}]: {price_data[len(price_data) - 1]}")
	w, b, cost_history, wb_history = gradient_descent(mileage_data, price_data, len(price_data),1.0e-7, 0, 0, 10000)
	print(f"w: {w}, b: {b}")
	w += 16; b-= -20
	plt.plot([[w * 2000 + b, w * 16000 + b], (300000 - b)/w, (20000 - b)/w])
	# plt.plot(wb_history, cost_history)
	plt.show()

	# fig, (ax1, ax2) = plt.subplot(1, 2, constrained_layout=True, figsize=(12,4))
	# ax1.plot(cost_history[:100])
	# ax2.plot(1000, np.arange(len(cost_history[1000:])), cost_history[1000:])
	# ax1.set_title("Cost vs. i")

	return 0


if __name__ == '__main__':
	main()
