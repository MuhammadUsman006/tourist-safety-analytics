import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train = pd.read_csv("model_X_train.csv")
X_test = pd.read_csv("model_X_test.csv")
y_train = pd.read_csv("model_y_train.csv").squeeze()
y_test = pd.read_csv("model_y_test.csv").squeeze()

print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows")

# --- 27 C values, matching the 27-combination effort given to the
# other two models, for a fair comparison ---
param_grid = {
    "C": np.logspace(-3, 3, 27),
}

base_model = LogisticRegression(max_iter=1000, random_state=42)

grid_search = GridSearchCV(
    base_model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1,
)

print("\nStarting grid search - 27 C values x 3-fold CV = 81 models to train.")

grid_search.fit(X_train, y_train)

print(f"\nBest C value found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy during search: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nFinal test set accuracy with tuned model: {accuracy:.3f} ({accuracy*100:.1f}%)")

print("\nDetailed performance report:")
print(classification_report(y_test, predictions))

print("Confusion matrix (rows = actual, columns = predicted):")
print("Order: high, low, medium")
print(confusion_matrix(y_test, predictions, labels=["high", "low", "medium"]))