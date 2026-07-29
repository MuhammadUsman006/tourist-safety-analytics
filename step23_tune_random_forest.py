import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train = pd.read_csv("model_X_train.csv")
X_test = pd.read_csv("model_X_test.csv")
y_train = pd.read_csv("model_y_train.csv").squeeze()
y_test = pd.read_csv("model_y_test.csv").squeeze()

print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows")

# --- Final narrowed check ---
# The previous search found max_depth=None and min_samples_split=5 as
# clear winners (not at any edge), so we keep those fixed this time.
# n_estimators=300 DID hit the top of its tested range, so this search
# only checks whether going even higher (400, 500) improves things
# further - a smaller, faster, targeted follow-up search.
param_grid = {
    "n_estimators": [300, 400, 500],
    "max_depth": [None],
    "min_samples_split": [5],
}

base_model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    base_model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1,
)

print("\nStarting smaller follow-up grid search - only 3 combinations x 3-fold CV = 9 models. Should be quick.")

grid_search.fit(X_train, y_train)

print(f"\nBest settings found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy during search: {grid_search.best_score_:.3f}")

best = grid_search.best_params_
if best["n_estimators"] == max(param_grid["n_estimators"]):
    print("NOTE: n_estimators hit the upper edge again - could test even more trees, but returns are likely diminishing now.")

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nFinal test set accuracy with tuned model: {accuracy:.3f} ({accuracy*100:.1f}%)")

print("\nDetailed performance report:")
print(classification_report(y_test, predictions))

print("Confusion matrix (rows = actual, columns = predicted):")
print("Order: high, low, medium")
print(confusion_matrix(y_test, predictions, labels=["high", "low", "medium"]))