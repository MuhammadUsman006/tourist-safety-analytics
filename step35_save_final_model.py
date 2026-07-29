"""
=====================================================================================
STEP 35 - SAVE THE FINAL TUNED MODEL FOR THE DASHBOARD
=====================================================================================

WHY THIS SCRIPT EXISTS
------------------------
Every ML script so far (step17, step18, step19, step20, and the tuning/CV
scripts) trained a model fresh each time it ran, purely to print out
accuracy/metrics - none of them actually SAVED the trained model to disk.
That was fine while we were comparing models, but the dashboard's Live
Prediction tab needs an already-trained model sitting on disk that it can
load instantly, rather than retraining from scratch every time someone
opens the app.

This script does three things:
  1. Rebuilds the exact same feature pipeline as step17 (same columns,
     same encoding, same scaling) so the saved model behaves identically
     to the 76% accuracy model you already validated.
  2. Trains ONE final XGBoost model using your confirmed best
     hyperparameters from the tuning phase (n_estimators=200, max_depth=7,
     learning_rate=0.2) on ALL the England/Wales data (not just a
     train/test split - for a "final" deployed model we use every row
     we have, since we've already proven its accuracy via the earlier
     train/test split and cross-validation).
  3. Saves everything the dashboard will need: the model itself, the
     fitted scaler, the fitted target label encoder, and the exact list
     of column names in the exact order the model expects.

RUN THIS ONCE
------------------
Save as step35_save_final_model.py in the same folder as your
_district_month_model_table_v2.csv files, then run it. It will create
a new subfolder called saved_model/ containing 4 files.
"""

import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

# -------------------------------------------------------------------------------
# STEP 1: Load and combine all 4 England/Wales cities' engineered feature tables
# (identical to step17_prepare_model_data.py)
# -------------------------------------------------------------------------------
ENGLAND_CITIES = ["london", "york", "liverpool", "birmingham"]

all_tables = []
for city_name in ENGLAND_CITIES:
    df = pd.read_csv(f"{city_name}_district_month_model_table_v2.csv")
    all_tables.append(df)

combined = pd.concat(all_tables, ignore_index=True)
print(f"Combined dataset: {len(combined):,} rows (district-month combinations across all 4 cities)")

# -------------------------------------------------------------------------------
# STEP 2: Select features (identical list to step17 v2) and drop missing rows
# -------------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "avg_distance_to_poi_km",
    "min_distance_to_poi_km",
    "poi_count_within_500m",
    "poi_count_within_1km",
    "year",
    "month_number",
    "season",
    "most_common_category",
    "city",
]

TARGET_COLUMN = "risk_label"

model_data = combined[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

before_drop = len(model_data)
model_data = model_data.dropna()
print(f"Dropped {before_drop - len(model_data)} rows with missing values")

# -------------------------------------------------------------------------------
# STEP 3: One-hot encode the categorical columns (season, crime category, city)
# -------------------------------------------------------------------------------
CATEGORICAL_COLUMNS = ["season", "most_common_category", "city"]

model_data_encoded = pd.get_dummies(model_data, columns=CATEGORICAL_COLUMNS, drop_first=False)

print(f"Total columns after encoding: {len(model_data_encoded.columns)}")

X = model_data_encoded.drop(columns=[TARGET_COLUMN])
y = model_data_encoded[TARGET_COLUMN]

# Save the exact column order NOW, before scaling - the dashboard needs this
# to line up any new user input into the same shape the model expects.
FEATURE_COLUMN_ORDER = list(X.columns)

# -------------------------------------------------------------------------------
# STEP 4: Scale the numeric columns (identical to step17 v2)
# -------------------------------------------------------------------------------
NUMERIC_COLUMNS = [
    "avg_distance_to_poi_km", "min_distance_to_poi_km",
    "poi_count_within_500m", "poi_count_within_1km",
    "year", "month_number",
]

scaler = StandardScaler()
X[NUMERIC_COLUMNS] = scaler.fit_transform(X[NUMERIC_COLUMNS])
print("Numeric features scaled.")

# -------------------------------------------------------------------------------
# STEP 5: Encode the target label (identical order to step20: high=0, low=1, medium=2)
# -------------------------------------------------------------------------------
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print("\nTarget label encoding:")
for i, label in enumerate(target_encoder.classes_):
    print(f"  {label} -> {i}")

# -------------------------------------------------------------------------------
# STEP 6: Train the FINAL model using your confirmed best hyperparameters
# -------------------------------------------------------------------------------
# These exact values (n_estimators=200, max_depth=7, learning_rate=0.2) are
# the ones your tuning phase confirmed as the best-performing, stable
# configuration (~76% accuracy, validated by 5-fold cross-validation).
final_model = XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.2,
    random_state=42,
    eval_metric="mlogloss",
)

final_model.fit(X, y_encoded)
print("\nFinal model trained on the full dataset.")

# -------------------------------------------------------------------------------
# STEP 7: Save everything the dashboard needs
# -------------------------------------------------------------------------------
OUTPUT_FOLDER = "saved_model"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

joblib.dump(final_model, f"{OUTPUT_FOLDER}/xgboost_final_model.pkl")
joblib.dump(scaler, f"{OUTPUT_FOLDER}/scaler.pkl")
joblib.dump(target_encoder, f"{OUTPUT_FOLDER}/target_encoder.pkl")
joblib.dump(FEATURE_COLUMN_ORDER, f"{OUTPUT_FOLDER}/feature_column_order.pkl")

print(f"\nSaved 4 files to '{OUTPUT_FOLDER}/':")
print("  - xgboost_final_model.pkl   (the trained model)")
print("  - scaler.pkl                (fitted StandardScaler for numeric features)")
print("  - target_encoder.pkl        (maps 0/1/2 back to high/low/medium)")
print("  - feature_column_order.pkl  (exact column order the model expects)")
print("\nStep 35 complete - the dashboard's Live Prediction tab can now load these.")