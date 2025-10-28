import numpy as np
import matplotlib.pyplot as plt

# Function and its derivative
def f(x):
    return x**2

def df(x):
    return 2*x

# Starting point
x = 10    # start far from minimum
learning_rate = 0.1
iterations = 25

x_history = [x]

# Gradient Descent Loop
for i in range(iterations):
    grad = df(x)
    x = x - learning_rate * grad
    x_history.append(x)

# Plot the function
x_vals = np.linspace(-10, 10, 100)
y_vals = f(x_vals)

plt.plot(x_vals, y_vals, label="f(x) = x²")
plt.scatter(x_history, f(np.array(x_history)), color='red', label="Steps")
plt.title("Gradient Descent Visualization")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
