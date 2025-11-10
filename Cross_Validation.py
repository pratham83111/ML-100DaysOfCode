from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np

# Load sample dataset
X, y = load_iris(return_X_y=True)

# Model
model = LogisticRegression(max_iter=1000)

# Define 5-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# Display results
print("Fold-wise Accuracy:", scores)
print("Average Accuracy: {:.2f}%".format(np.mean(scores) * 100))
