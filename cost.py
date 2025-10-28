import numpy as np
import matplotlib.pyplot as plt

# Generate a larger dataset (100 data points)
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 2.5 * x + 5 + np.random.randn(100) * 2   # y = 2.5x + 5 + noise

# Function to calculate cost
def compute_cost(m, c, x, y):
    y_pred = m * x + c
    cost = np.sum((y_pred - y) ** 2) / (2 * len(y))
    return cost

# Create a grid of m and c values for visualization
m_values = np.linspace(0, 5, 100)
c_values = np.linspace(0, 10, 100)

M, C = np.meshgrid(m_values, c_values)
Cost = np.zeros(M.shape)

# Compute cost for every (m, c)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        Cost[i, j] = compute_cost(M[i, j], C[i, j], x, y)

# Plot the 3D Surface of Cost Function
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(M, C, Cost, cmap='viridis')
ax.set_xlabel('Slope (m)')
ax.set_ylabel('Intercept (c)')
ax.set_zlabel('Cost (Error)')
ax.set_title('Cost Function Surface for Linear Regression')

plt.show()
