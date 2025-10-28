# -------------------------------
#  Regression Model Evaluation
# -------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# # Sample data
# X = np.array([800, 1000, 1200, 1500, 1800, 2000, 2300, 2500]).reshape(-1, 1)
# y = np.array([100, 120, 150, 180, 210, 240, 270, 300])  # in lakhs

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# # Model training
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Evaluation Metrics
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print(" Regression Evaluation Metrics")
# print(f"MAE  : {mae:.2f}")
# print(f"MSE  : {mse:.2f}")
# print(f"RMSE : {rmse:.2f}")
# print(f"R² Score : {r2:.2f}")

# #  Graph
# plt.scatter(X_test, y_test, color='blue', label="Actual Values")
# plt.plot(X_test, y_pred, color='red', linewidth=2, label="Predicted Line")
# plt.xlabel("House Size (sq.ft)")
# plt.ylabel("Price (in Lakhs)")
# plt.title("Regression Model Evaluation")
# plt.legend()
# plt.show()
# -------------------------------
# 📘 Classification Model Evaluation
# -------------------------------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

# 🎯 Sample dataset (binary classification)
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 🧠 Model training
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# 🔮 Predictions
y_pred = model.predict(X_test)

# 📏 Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Classification Evaluation Metrics")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1 Score  : {f1:.2f}")

# 🧮 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Classification Model")
plt.show()
