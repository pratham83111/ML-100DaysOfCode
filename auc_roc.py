# -----------------------------
# ROC Curve & AUC Example
# -----------------------------

# Step 1: Import Libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Step 2: Create Dummy Dataset
X, y = make_classification(
    n_samples=1000,      # 1000 samples
    n_features=10,       # 10 features
    n_classes=2,         # Binary classification (0 or 1)
    random_state=42
)

# Step 3: Split Dataset (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Train Model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Predict Probabilities for Test Set
y_probs = model.predict_proba(X_test)[:, 1]  # Only positive class probability

# Step 6: Calculate FPR, TPR, and Thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Step 7: Calculate AUC Score
auc_score = roc_auc_score(y_test, y_probs)
print("AUC Score:", auc_score)

# Step 8: Plot ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
plt.title('ROC Curve Example')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()
