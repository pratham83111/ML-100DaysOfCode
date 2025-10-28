from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Underfitting model (very shallow tree)
model_under = DecisionTreeClassifier(max_depth=1)
model_under.fit(X_train, y_train)
print("Underfitting:")
print("Train Accuracy:", accuracy_score(y_train, model_under.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, model_under.predict(X_test)))

# Overfitting model (very deep tree)
model_over = DecisionTreeClassifier(max_depth=None)
model_over.fit(X_train, y_train)
print("\nOverfitting:")
print("Train Accuracy:", accuracy_score(y_train, model_over.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, model_over.predict(X_test)))
