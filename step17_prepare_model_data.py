import pandas as pd
from sklearn.model_selection import train_test_split

ENGLAND_CITIES = ["london", "york", "liverpool", "birmingham"]

# --- Load the v2 tables (now with POI density added) ---
all_tables = []
for city_name in ENGLAND_CITIES:
    df = pd.read_csv(f"{city_name}_district_month_model_table_v2.csv")
    all_tables.append(df)

combined = pd.concat(all_tables, ignore_index=True)
print(f"Combined dataset: {len(combined):,} rows (district-month combinations across all 4 cities)")

# --- Updated feature list: added year, poi_count_within_500m, poi_count_within_1km ---
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

CATEGORICAL_COLUMNS = ["season", "most_common_category", "city"]

model_data_encoded = pd.get_dummies(
    model_data, columns=CATEGORICAL_COLUMNS, drop_first=False
)

print(f"\nTotal columns after encoding: {len(model_data_encoded.columns)}")

X = model_data_encoded.drop(columns=[TARGET_COLUMN])
y = model_data_encoded[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.preprocessing import StandardScaler

NUMERIC_COLUMNS = [
    "avg_distance_to_poi_km", "min_distance_to_poi_km",
    "poi_count_within_500m", "poi_count_within_1km",
    "year", "month_number",
]

scaler = StandardScaler()
X_train[NUMERIC_COLUMNS] = scaler.fit_transform(X_train[NUMERIC_COLUMNS])
X_test[NUMERIC_COLUMNS] = scaler.transform(X_test[NUMERIC_COLUMNS])

print("\nNumeric features scaled (mean=0, similar spread across all numeric columns).")

print(f"\nTraining set size: {len(X_train):,} rows")
print(f"Test set size: {len(X_test):,} rows")

print("\nTraining set risk label balance:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTest set risk label balance:")
print(y_test.value_counts(normalize=True).round(3))

X_train.to_csv("model_X_train.csv", index=False)
X_test.to_csv("model_X_test.csv", index=False)
y_train.to_csv("model_y_train.csv", index=False)
y_test.to_csv("model_y_test.csv", index=False)

print("\nModel-ready data prepared and saved (v2 - with POI density + year, scaled).")