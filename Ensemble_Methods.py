
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score


data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

ab_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ab_model.fit(X_train, y_train)
ab_preds = ab_model.predict(X_test)


rf_acc = accuracy_score(y_test, rf_preds)
ab_acc = accuracy_score(y_test, ab_preds)

print("🌳 Random Forest (Bagging) Accuracy:", round(rf_acc * 100, 2), "%")
print("🚀 AdaBoost (Boosting) Accuracy:", round(ab_acc * 100, 2), "%")


if rf_acc > ab_acc:
    print("✅ Random Forest performed better!")
elif rf_acc < ab_acc:
    print("✅ AdaBoost performed better!")
else:
    print("🤝 Both models performed equally well!")
