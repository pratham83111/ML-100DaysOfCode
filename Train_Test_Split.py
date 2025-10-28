# ----------------------------------------------------------
# 📘 DAY 9 - Train-Test Split in Machine Learning
# ----------------------------------------------------------
# Author: Pratham Kumar
# Topic: Understanding and Implementing Train-Test Split
# ----------------------------------------------------------

# Step 1️⃣: Import required libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Step 2️⃣: Load a sample dataset (Iris)
iris = load_iris()

# Create a DataFrame for better visualization
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("  First 5 rows of the dataset:")
print(df.head())

# Step 3️⃣: Split the data into Features (X) and Labels (y)
X = df.drop('target', axis=1)
y = df['target']

# Step 4️⃣: Perform Train-Test Split
# 80% data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n Data Split Completed Successfully!")
print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# Step 5️⃣: Visualize the Split (for concept clarity)
split_labels = ['Training Data'] * len(X_train) + ['Testing Data'] * len(X_test)
split_data = pd.concat([X_train, X_test])
split_data['Set'] = split_labels

plt.figure(figsize=(8, 4))
sns.countplot(x='Set', data=split_data, palette='viridis')
plt.title(' Train-Test Split Visualization')
plt.xlabel('Dataset Type')
plt.ylabel('Count of Samples')
plt.show()

# Step 6️⃣: Summary Print
print("\n Summary:")
print("➡ Train-Test Split helps prevent overfitting.")
print("➡ It checks model performance on unseen data.")
print("➡ Common ratios: 80-20, 70-30, or 75-25.")
