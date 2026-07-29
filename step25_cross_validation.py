import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# --- Load and combine ALL the data (training + test together) ---
# Cross-validation works differently from a single train/test split: it
# uses ALL the data, repeatedly, by splitting it into several equal
# "folds" (we'll use 5). Each fold takes a turn being the test set,
# while the other folds train the model - so every single row gets
# used for both training and testing at some point, just never at the
# same time. This gives a far more reliable performance estimate than
# relying on one single lucky/unlucky 80/20 split.
X_train = pd.read_csv("model_X_train.csv")
X_test = pd.read_csv("model_X_test.csv")
y_train = pd.read_csv("model_y_train.csv").squeeze()
y_test = pd.read_csv("model_y_test.csv").squeeze()

X_full = pd.concat([X_train, X_test], ignore_index=True)
y_full = pd.concat([y_train, y_test], ignore_index=True)

print(f"Full dataset for cross-validation: {len(X_full):,} rows")

# --- Set up the SAME 5 folds for every model ---
# Using the exact same fold splits for all three models (via one shared
# StratifiedKFold object with a fixed random_state) is essential - it's
# what makes a later paired statistical comparison between models valid,
# since each model is tested on IDENTICAL data splits, not different
# random ones.
# "Stratified" means each fold keeps the same balanced 33/33/33 low/
# medium/high mix as the full dataset, avoiding an unlucky fold that's
# accidentally mostly one risk category.
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Encode the target for XGBoost (it needs numbers, not text) ---
target_encoder = LabelEncoder()
y_full_encoded = target_encoder.fit_transform(y_full)

# --- Define all 3 models, using the BEST settings found during tuning ---
models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=5, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.2,
        random_state=42, eval_metric="mlogloss"
    ),
}

fold_results = {}

for model_name, model in models.items():
    print(f"\nRunning 5-fold cross-validation for {model_name}...")

    if model_name == "XGBoost":
        # XGBoost needs the numeric-encoded target
        scores = cross_val_score(model, X_full, y_full_encoded, cv=cv_folds, scoring="accuracy")
    else:
        # Logistic Regression and Random Forest can use the text labels directly
        scores = cross_val_score(model, X_full, y_full, cv=cv_folds, scoring="accuracy")

    fold_results[model_name] = scores

    print(f"  Fold-by-fold accuracy: {np.round(scores, 3)}")
    print(f"  Mean accuracy: {scores.mean():.3f} ({scores.mean()*100:.1f}%)")
    print(f"  Standard deviation: {scores.std():.3f}")

# --- Save the fold-by-fold results for the significance test in Step 2 ---
results_df = pd.DataFrame(fold_results)
results_df.to_csv("cross_validation_fold_scores.csv", index=False)

print("\n--- SUMMARY (mean accuracy ± standard deviation across 5 folds) ---")
for model_name, scores in fold_results.items():
    print(f"{model_name}: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

print("\nSaved fold-by-fold scores -> cross_validation_fold_scores.csv")