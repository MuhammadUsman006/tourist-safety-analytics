import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Features used for this test - NOTE: "city" is deliberately excluded ---
# We're specifically testing whether patterns learned in London transfer
# to other cities, so including "city" as a feature would be meaningless
# during training (it's constant - always "london") and inappropriate
# during testing (the model never learned anything genuine about other
# cities' city-identity, only about York/Liverpool's SPATIAL and CRIME
# patterns, which is exactly what we want to test).
FEATURE_COLUMNS = [
    "avg_distance_to_poi_km",
    "min_distance_to_poi_km",
    "poi_count_within_500m",
    "poi_count_within_1km",
    "year",
    "month_number",
    "season",
    "most_common_category",
]

CATEGORICAL_COLUMNS = ["season", "most_common_category"]

TARGET_COLUMN = "risk_label"


def prepare_features(df, reference_columns=None):
    """
    Selects our feature columns, one-hot encodes the categorical ones,
    and (if given a reference_columns list) aligns the result to match
    those exact columns - filling in 0 for any category that appears in
    one city but not the other. This is essential: London and York/
    Liverpool might not have identical sets of "most_common_category"
    values, so their one-hot encoded tables could otherwise end up with
    different columns, which a trained model cannot handle.
    """
    data = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()
    encoded = pd.get_dummies(data, columns=CATEGORICAL_COLUMNS)

    X = encoded.drop(columns=[TARGET_COLUMN])
    y = encoded[TARGET_COLUMN]

    if reference_columns is not None:
        # Add any missing columns (filled with 0), and drop any extra
        # ones not seen during training, then put columns in the exact
        # same order as the reference - this guarantees the test data
        # has IDENTICAL columns to what the model was trained on.
        X = X.reindex(columns=reference_columns, fill_value=0)

    return X, y


# --- STEP 1: Train exclusively on London ---
print("--- Training on London only ---")
london_df = pd.read_csv("london_district_month_model_table_v2.csv")
X_train, y_train = prepare_features(london_df)

print(f"London training data: {len(X_train):,} rows, {len(X_train.columns)} features")

target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(y_train)

model = XGBClassifier(
    n_estimators=200, max_depth=7, learning_rate=0.2,
    random_state=42, eval_metric="mlogloss",
)
model.fit(X_train, y_train_encoded)
print("Model trained on London data only.")

training_columns = X_train.columns.tolist()

# --- STEP 2: Test on York and Liverpool separately ---
TEST_CITIES = ["york", "liverpool"]

for city_name in TEST_CITIES:
    print(f"\n--- Testing on {city_name.title()} (never seen during training) ---")

    test_df = pd.read_csv(f"{city_name}_district_month_model_table_v2.csv")
    X_test, y_test = prepare_features(test_df, reference_columns=training_columns)

    print(f"{city_name.title()} test data: {len(X_test):,} rows")

    predictions_encoded = model.predict(X_test)
    predictions = target_encoder.inverse_transform(predictions_encoded)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy on {city_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")

    print("\nDetailed performance report:")
    print(classification_report(y_test, predictions))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print("Order: high, low, medium")
    print(confusion_matrix(y_test, predictions, labels=["high", "low", "medium"]))

print("\nGeographic generalisation test complete.")