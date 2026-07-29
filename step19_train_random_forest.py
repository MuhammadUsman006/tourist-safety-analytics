import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train = pd.read_csv("model_X_train.csv")
X_test = pd.read_csv("model_X_test.csv")
y_train = pd.read_csv("model_y_train.csv").squeeze()
y_test = pd.read_csv("model_y_test.csv").squeeze()

print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows")

# --- Create and train the model ---
# Random Forest builds MANY decision trees (like flowcharts of yes/no
# questions - "is distance < 0.5km? if yes, is season summer? ...") and
# then has them all VOTE on the final answer. This lets it capture more
# complex, non-linear patterns than Logistic Regression can.
#
# n_estimators=200: build 200 separate decision trees and combine their
# votes - more trees generally means more stable, reliable predictions,
# at the cost of slightly more computing time.
model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)
print("Model trained.")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nOverall accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

print("\nDetailed performance report:")
print(classification_report(y_test, predictions))

print("Confusion matrix (rows = actual, columns = predicted):")
print("Order: high, low, medium")
print(confusion_matrix(y_test, predictions, labels=["high", "low", "medium"]))

# --- BONUS: which features mattered most to this model? ---
# Random Forest can tell us which input features it actually relied on
# most heavily to make its decisions - a nice preview of what SHAP will
# explore in much more depth later in this phase.
feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nWhich features mattered most to this model:")
print(feature_importance)