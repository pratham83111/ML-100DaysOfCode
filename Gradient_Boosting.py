# 🚀 Gradient Boosting Classifier with Visualization
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1️⃣ Load Dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# 2️⃣ Train Model
model = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42
)
model.fit(X_train, y_train)

# 3️⃣ Make Predictions
y_pred = model.predict(X_test)

# 4️⃣ Model Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Gradient Boosting Accuracy: {acc:.4f}")

# 5️⃣ Confusion Matrix Visualization
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Gradient Boosting")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 6️⃣ Feature Importance Plot
feature_importance = pd.Series(model.feature_importances_, index=data.feature_names)
top_features = feature_importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Important Features in Gradient Boosting Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.show()
