import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

X_train = pd.read_csv("model_X_train.csv")
X_test = pd.read_csv("model_X_test.csv")
y_train = pd.read_csv("model_y_train.csv").squeeze()
y_test = pd.read_csv("model_y_test.csv").squeeze()

print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows")

# --- XGBoost needs the TARGET labels as numbers too, not text ---
# Unlike scikit-learn's other models (which handle text labels like
# "low"/"medium"/"high" automatically), XGBoost specifically requires
# the target to already be numeric. We use LabelEncoder again, just
# for the target this time.
target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(y_train)
y_test_encoded = target_encoder.transform(y_test)

print("\nTarget label encoding:")
for i, label in enumerate(target_encoder.classes_):
    print(f"  {label} -> {i}")

# --- Create and train the model ---
# XGBoost ("Extreme Gradient Boosting") builds decision trees ONE AT A
# TIME, where each new tree specifically focuses on correcting the
# mistakes the previous trees made - rather than Random Forest's
# approach of building many independent trees and voting. This
# "learn from your mistakes, one step at a time" approach often gives
# XGBoost an edge, especially on complex, messy real-world data.
model = XGBClassifier(
    n_estimators=200,
    random_state=42,
    eval_metric="mlogloss",  # the specific scoring method used internally
                              # while it's learning, for multi-class
                              # problems like ours (low/medium/high)
)

model.fit(X_train, y_train_encoded)
print("Model trained.")

predictions_encoded = model.predict(X_test)

# Convert the numeric predictions back into readable text labels
predictions = target_encoder.inverse_transform(predictions_encoded)

accuracy = accuracy_score(y_test, predictions)
print(f"\nOverall accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

print("\nDetailed performance report:")
print(classification_report(y_test, predictions))

print("Confusion matrix (rows = actual, columns = predicted):")
print("Order: high, low, medium")
print(confusion_matrix(y_test, predictions, labels=["high", "low", "medium"]))

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nWhich features mattered most to this model:")
print(feature_importance)