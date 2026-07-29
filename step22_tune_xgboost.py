import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train = pd.read_csv("model_X_train.csv")
X_test = pd.read_csv("model_X_test.csv")
y_train = pd.read_csv("model_y_train.csv").squeeze()
y_test = pd.read_csv("model_y_test.csv").squeeze()

target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(y_train)
y_test_encoded = target_encoder.transform(y_test)

print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows")

# --- WIDENED grid, based on the previous search's results ---
# The first search (n_estimators: 100-300, max_depth: 3-7,
# learning_rate: 0.05-0.2) found its best result at max_depth=7 and
# learning_rate=0.2 - both the HIGHEST values we tested. When a grid
# search's winner sits right at the edge of the tested range, that's a
# sign the true best setting might lie beyond it, so we shift the
# search window upward this time to check.
param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [7, 9, 11],
    "learning_rate": [0.2, 0.3, 0.4],
}

base_model = XGBClassifier(random_state=42, eval_metric="mlogloss")

grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

print("\nStarting grid search - this trains 81 different model versions, so it will take a few minutes. Please be patient and let it run.")

grid_search.fit(X_train, y_train_encoded)

print(f"\nBest settings found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy during search: {grid_search.best_score_:.3f}")

# --- Check whether the winning settings are still at an edge ---
# If so, we may need to widen the search once more.
best = grid_search.best_params_
if best["max_depth"] == max(param_grid["max_depth"]):
    print("NOTE: max_depth hit the upper edge again - consider testing even deeper trees.")
if best["learning_rate"] == max(param_grid["learning_rate"]):
    print("NOTE: learning_rate hit the upper edge again - consider testing even higher values.")
if best["n_estimators"] == max(param_grid["n_estimators"]):
    print("NOTE: n_estimators hit the upper edge - consider testing even more trees.")

# --- Use the best model found to evaluate on our actual held-out test set ---
best_model = grid_search.best_estimator_
predictions_encoded = best_model.predict(X_test)
predictions = target_encoder.inverse_transform(predictions_encoded)

accuracy = accuracy_score(y_test, predictions)
print(f"\nFinal test set accuracy with tuned model: {accuracy:.3f} ({accuracy*100:.1f}%)")

print("\nDetailed performance report:")
print(classification_report(y_test, predictions))

print("Confusion matrix (rows = actual, columns = predicted):")
print("Order: high, low, medium")
print(confusion_matrix(y_test, predictions, labels=["high", "low", "medium"]))