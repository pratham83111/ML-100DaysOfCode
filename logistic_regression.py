#  Logistic Regression with Visualization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Sample Data: [Age, EstimatedSalary, Purchased (0=No, 1=Yes)]
data = [
    [22, 20000, 0],
    [25, 25000, 0],
    [28, 30000, 0],
    [32, 40000, 0],
    [35, 50000, 1],
    [37, 55000, 1],
    [40, 60000, 1],
    [45, 80000, 1]
]

# Split data
X = np.array([[row[0]] for row in data])  # Age (1 feature for easy visualization)
y = np.array([row[2] for row in data])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(" Accuracy:", accuracy_score(y_test, y_pred))

# Sigmoid curve visualization
X_range = np.linspace(20, 50, 200).reshape(-1, 1)
probabilities = model.predict_proba(X_range)[:, 1]

plt.figure(figsize=(8,5))
plt.scatter(X, y, color="red", label="Actual Data (0/1)")
plt.plot(X_range, probabilities, color="blue", label="Logistic Regression Curve")
plt.xlabel("Age")
plt.ylabel("Probability of Purchase")
plt.title(" Logistic Regression Visualization")
plt.legend()
plt.grid(True)
plt.show()
