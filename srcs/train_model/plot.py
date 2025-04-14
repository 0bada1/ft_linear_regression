import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots
from cost import squared_error_cost


def cost_surface_plot(cost_function, x_data, y_data):
    w_vals = np.linspace(-1, 1, 50)
    b_vals = np.linspace(-1, 1, 50)
    W, B = np.meshgrid(w_vals, b_vals)

    Z = np.zeros_like(W)
    m = len(x_data)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j] = cost_function(m, W[i, j], B[i, j], x_data, y_data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W, B, Z, cmap='viridis')
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_zlabel('Cost')
    ax.set_title('Cost Function Surface')


def cost_w_b_3D_graph(w_history, b_history, cost_history):
    '''3D graph of cost vs. w and b over gradient descent iterations'''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert to NumPy arrays for compatibility
    w = np.array(w_history)
    b = np.array(b_history)
    cost = np.array(cost_history)

    # Plot the gradient descent path in 3D
    ax.plot3D(w, b, cost, color='blue', label='Gradient Descent Path')
    ax.scatter3D(w[-1], b[-1], cost[-1], color='red', label='Final Point')  # Final position

    # Label the axes
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_zlabel('Cost')
    ax.set_title('3D Cost Function Convergence')
    ax.legend()


def cost_w_graph(cost_history, w_history):
    '''Cost and w (theta0) 2D graph'''
    # Create new graph
    plt.figure()

    # Graph cost and w
    plt.plot(w_history, cost_history)
    plt.title("Cost on w (theta 0)")
    plt.xlabel("w (theta 0)")
    plt.ylabel("Cost (USD)")

    # The left side of gradient descent
    # plt.xticks(ticks=[-0.04, -0.02, 0], labels=['-0.04', '-0.02', '0']) # Custom x-axis ticks


def cost_iteration_graph(cost_history, iterations):
    '''Cost to iterations 2D graph'''
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))

    ax1.plot(cost_history[:100])
    ax2.plot(1000 + np.arange(len(cost_history[1000:])), cost_history[1000:])

    ax1.set_title("Cost vs iterations (start)")
    ax2.set_title("Cost vs iterations (end)")

    ax1.set_xlabel("Iteration step")
    ax2.set_xlabel("Iteration step")

    ax1.set_ylabel("Cost")
    ax2.set_ylabel("Cost")


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


def w_contour_graph(w_history, cost_history):
    '''w and cost contour graph'''
    # Create new figure/graph
    plt.figure()

    # Set title of w/cost contour graph
    plt.title("Cost and w contour graph")

    # Set x & y labels
    plt.xlabel("w")
    plt.ylabel("cost")

    # Create a meshgrind & create contour
    x, y = np.meshgrid(w_history, cost_history)
    z = np.sin(x) + np.cos(y)
    plt.contour(x, y, z)
    # plt.contourf(x, y, z)


def plot_cost_contour(x_data, y_data, cost_function, w_history=None, b_history=None):
    '''2D contour plot of the cost function over w and b'''

    # Define range of w and b
    w_vals = np.linspace(-1, 1, 100)
    b_vals = np.linspace(-1, 1, 100)
    W, B = np.meshgrid(w_vals, b_vals)

    # Compute cost at each point (w, b)
    Z = np.zeros_like(W)
    m = len(x_data)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j] = cost_function(m, W[i, j], B[i, j], x_data, y_data)

    # Plot contour
    plt.figure(figsize=(8, 6))
    cp = plt.contour(W, B, Z, levels=30, cmap='viridis')
    plt.clabel(cp, inline=True, fontsize=8)
    plt.title("Cost Function Contour Plot")
    plt.xlabel("w")
    plt.ylabel("b")

    # Optional: plot gradient descent path on top
    if w_history is not None and b_history is not None:
        plt.plot(w_history, b_history, color='red', marker='o', markersize=3, label='Gradient Descent Path')
        plt.legend()

    plt.grid(True)


def make_graphs(price_data, mileage_data, cost_history, wb_history, w, b, iterations):
    '''Call all graph functions from here'''
    # Get w from wb_history
    w_history = np.zeros(len(wb_history))
    b_history = np.zeros(len(wb_history))
    for i in range(len(wb_history)):
        w_history[i] = wb_history[i][0]  # Save w of every iteration in wb_history
        b_history[i] = wb_history[i][1]  # Save b of every iteration in wb_history
    # print(f"w_history = {w_history}\nb_history = {b_history}")

    # plot_cost_contour(mileage_data, price_data, squared_error_cost, w_history, b_history)
    cost_surface_plot(squared_error_cost, b_history, w_history)
    # cost_w_graph(cost_history, w_history)
    cost_w_b_3D_graph(w_history, b_history, cost_history)

    w_contour_graph(w_history, cost_history)
    scatter_data_graph(price_data, mileage_data)
    model_fit_on_data_graph(price_data, mileage_data, w, b)
    cost_iteration_graph(cost_history, iterations)
    plt.show()
