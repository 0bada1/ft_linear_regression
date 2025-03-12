import numpy as np
from load_data import load_data
from plot import make_graphs
from gradient_descent import gradient_descent
# from linear_regression import linear_regression
# import seaborn as sns

def main():
	'''Load data, check for errors, calculate cost, calculate gradient descent
	calculate linear regression, predict new car price, draw graph'''
	data = load_data("../data/data.csv")
	# sns.pairplot(data)
	price_data = data['price']  # y data
	mileage_data = data['km']  # x data

	normalized_price_data = (price_data - np.mean(price_data)) / np.std(price_data)
	normalized_mileage_data = (mileage_data - np.mean(mileage_data)) / np.std(mileage_data)
	# print(f"Mileage = \n{mileage_data}")
	# print(f"Price= \n{price_data}")

	# Test data
	# x_train = np.array([1.0, 2.0])
	# y_train = np.array([300.0, 500.0])

	iterations = 10000
	w = 0; b = 0
	alpha = 1.0e-1
	print(f"Price Data: \n{price_data}\nMileage Data: \n{mileage_data}\nm: {len(price_data)}\nalpha(learning rate): {0.01}\nw: {0}\nb: {0}\niterations: {10000}")
	print(f"Price[{len(price_data) - 1}]: {price_data[len(price_data) - 1]}")
	w, b, cost_history, wb_history = gradient_descent(normalized_price_data, normalized_mileage_data, len(price_data), alpha, 0, -1, iterations)
	w = w * (np.std(price_data) / np.std(mileage_data))
	b = (b * np.std(price_data)) +  np.mean(price_data) - (w * np.mean(mileage_data))

	make_graphs(price_data, mileage_data, cost_history, wb_history, w, b, iterations)
	# w, b, cost_history, wb_history = gradient_descent(x_train, y_train, len(x_train), 1.0e-2, 0, 0, iterations)
	print(f"w: {w}, b: {b}")
	# w += 16; b-= -20
	# plt.plot([[w * 2000 + b, w * 16000 + b], (300000 - b)/w, (20000 - b)/w])
	# plt.xlabel("w"); plt.ylabel("Cost"); plt.plot(wb_history, cost_history)

	# fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
	# ax1.plot(cost_history[:100])
	# ax2.plot(1000 + np.arange(len(cost_history[1000:])), cost_history[1000:])
	# ax1.set_title("Cost vs. i")

	return 0


if __name__ == '__main__':
	main()
