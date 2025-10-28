# # 🔹 Feature Scaling in Machine Learning
# # Author: Pratham Kumar

# import pandas as pd
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# # 🧠 Step 1: Sample dataset
# data = pd.DataFrame({
#     'Age': [18, 25, 30, 45, 60],
#     'Salary': [20000, 35000, 50000, 80000, 120000]
# })

# print(" Original Data:\n")
# print(data)

# # ⚙️ Step 2: Standardization (Z-score Normalization)
# standard_scaler = StandardScaler()
# standardized_data = standard_scaler.fit_transform(data)

# standardized_df = pd.DataFrame(standardized_data, columns=data.columns)
# print("\n After Standardization:\n")
# print(standardized_df)

# # ⚙️ Step 3: Normalization (Min-Max Scaling)
# minmax_scaler = MinMaxScaler()
# normalized_data = minmax_scaler.fit_transform(data)

# normalized_df = pd.DataFrame(normalized_data, columns=data.columns)
# print("\n After Normalization:\n")
# print(normalized_df)



# 🔹 Feature Selection with Chi-Square Test & Visualization

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load Dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("🔸 Original Dataset Shape:", X.shape)

# Step 2: Split Data into Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Apply Feature Selection (Top 2 Features)
selector = SelectKBest(score_func=chi2, k=2)
X_new = selector.fit_transform(X_train, y_train)

    # Step 4: Get feature scores
    feature_scores = pd.DataFrame({
        'Feature': X_train.columns,
        'Score': selector.scores_
    }).sort_values(by='Score', ascending=False)

    # Step 5: Show selected features
    selected_features = X_train.columns[selector.get_support()]
    print("\n Selected Features:")
for f in selected_features:
    print("•", f)

# Step 6: Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(feature_scores['Feature'], feature_scores['Score'], color='teal')
plt.xlabel('Chi-Square Score')
plt.ylabel('Features')
plt.title('Feature Importance using Chi-Square Test')
plt.gca().invert_yaxis()  # Highest score on top
plt.show()

# Step 7: Compare before and after
print("\n Before Selection:", X_train.shape)
print(" After Selection:", X_new.shape)
