import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

X_train = pd.read_csv("model_X_train.csv")
X_test = pd.read_csv("model_X_test.csv")
y_train = pd.read_csv("model_y_train.csv").squeeze()
y_test = pd.read_csv("model_y_test.csv").squeeze()

print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows")

# --- Define our 3 best-tuned models as the "base learners" ---
# These use the exact best settings we found during hyperparameter
# tuning earlier - stacking works FROM our already-optimised models,
# not from scratch.
base_models = [
    ("logreg", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=5, random_state=42)),
    ("xgb", XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.2, random_state=42, eval_metric="mlogloss")),
]

# --- The "meta-model" that learns how to best combine the 3 base
# models' predictions ---
# We use a simple Logistic Regression here deliberately - the meta-model
# doesn't need to be complex, since its only job is to learn sensible
# WEIGHTS for combining the three base models' outputs, not to find new
# patterns in the raw data itself.
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# --- StackingClassifier ties it all together ---
# cv=5: internally, scikit-learn trains each base model on 4/5 of the
# training data and predicts on the held-out 5th, rotating through all
# 5 combinations - this generates honest "out-of-fold" predictions for
# training the meta-model, preventing the meta-model from unfairly
# benefiting from base models that have already memorised the training
# data they're being asked to predict.
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1,
)

print("\nTraining the stacking ensemble (this trains all 3 base models multiple times internally, so it will take a few minutes)...")

stacking_model.fit(X_train, y_train)
print("Stacking ensemble trained.")

predictions = stacking_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nStacking ensemble accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

print("\nDetailed performance report:")
print(classification_report(y_test, predictions))

print("Confusion matrix (rows = actual, columns = predicted):")
print("Order: high, low, medium")
print(confusion_matrix(y_test, predictions, labels=["high", "low", "medium"]))

# --- Compare directly against our best single model (XGBoost alone) ---
print("\n--- COMPARISON ---")
print(f"XGBoost alone (from earlier): 76.0-76.1%")
print(f"Stacking ensemble: {accuracy*100:.1f}%")