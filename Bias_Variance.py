import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
np.random.seed(42)
X = np.linspace(0, 5, 50).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.randn(50) * 0.3  # Non-linear data with noise

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Try different polynomial degrees to show bias-variance tradeoff
degrees = [1, 3, 9]
plt.figure(figsize=(12, 4))

for i, degree in enumerate(degrees, 1):
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    y_pred_train = model.predict(X_poly_train)
    y_pred_test = model.predict(X_poly_test)

    # Plot
    plt.subplot(1, 3, i)
    plt.scatter(X_train, y_train, color="blue", label="Train Data")
    plt.scatter(X_test, y_test, color="green", label="Test Data")
    X_range = np.linspace(0, 5, 100).reshape(-1, 1)
    plt.plot(X_range, model.predict(poly.transform(X_range)), color="red", label=f"Degree {degree}")
    plt.title(f"Degree {degree}\nTrain MSE: {mean_squared_error(y_train, y_pred_train):.2f}, "
              f"Test MSE: {mean_squared_error(y_test, y_pred_test):.2f}")
    plt.legend()

plt.suptitle("Bias-Variance Tradeoff Visualization", fontsize=14)
plt.show()


