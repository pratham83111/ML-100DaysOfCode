# Step 1: Import Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create Model
rf = RandomForestClassifier(random_state=42)

# Step 5: Define Hyperparameters to Tune
param_grid = {
    'n_estimators': [50, 100, 200],         # Number of trees
    'max_depth': [2, 4, 6, 8, None],        # Depth of trees
    'min_samples_split': [2, 5, 10],        # Minimum samples to split a node
}

# Step 6: Create GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2)

# Step 7: Fit the Model
grid_search.fit(X_train, y_train)

# Step 8: Best Parameters and Model
print("Best Hyperparameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_

# Step 9: Test Accuracy
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

