import pandas as pd

from step13_severity_weights_england import ENGLAND_SEVERITY_WEIGHTS
from step14_severity_weights_scotland import SCOTLAND_SEVERITY_WEIGHTS

WEIGHT_CRIME_FREQUENCY = 0.40
WEIGHT_SEVERITY = 0.40
WEIGHT_INVERSE_FOOTFALL = 0.20

footfall = pd.read_csv("footfall_final.csv")


def min_max_normalise(series):
    """
    Rescales a column of numbers so the smallest value becomes 0, the
    largest becomes 1, and everything else falls proportionally in
    between.
    """
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return series * 0

    return (series - min_val) / (max_val - min_val)


def calculate_england_tvs(city_name):
    df = pd.read_csv(f"{city_name}_district_month_model_table.csv")

    df["norm_crime_frequency"] = min_max_normalise(df["crime_count"])

    df["severity_score"] = df["most_common_category"].map(ENGLAND_SEVERITY_WEIGHTS)
    df["norm_severity_score"] = min_max_normalise(df["severity_score"])

    city_footfall = footfall.loc[footfall["city"] == city_name, "annual_overnight_trips_millions"].iloc[0]
    df["inverse_footfall"] = 1 / city_footfall

    df["city"] = city_name
    df["nation"] = "england"
    df["category"] = df["most_common_category"]

    return df


def calculate_scotland_tvs(city_name):
    df = pd.read_csv(f"{city_name}_yearly_model_table.csv")

    df = df[df["year"] >= 2023].copy()

    df["norm_crime_frequency"] = min_max_normalise(df["crime_count"])

    df["severity_score"] = df["category"].map(SCOTLAND_SEVERITY_WEIGHTS)
    df["norm_severity_score"] = min_max_normalise(df["severity_score"])

    city_footfall = footfall.loc[footfall["city"] == city_name, "annual_overnight_trips_millions"].iloc[0]
    df["inverse_footfall"] = 1 / city_footfall

    df["city"] = city_name
    df["nation"] = "scotland"

    return df


# --- Process all 6 cities and collect their component tables ---
all_city_tables = []

for city_name in ["london", "york", "liverpool", "birmingham"]:
    all_city_tables.append(calculate_england_tvs(city_name))

for city_name in ["edinburgh", "glasgow"]:
    all_city_tables.append(calculate_scotland_tvs(city_name))

print("Components calculated for all 6 cities.")

# --- Normalise inverse_footfall ACROSS all 6 cities together ---
combined = pd.concat(all_city_tables, ignore_index=True)

combined["norm_inverse_footfall"] = min_max_normalise(combined["inverse_footfall"])

# --- Calculate the final TVS score ---
combined["TVS"] = (
    WEIGHT_CRIME_FREQUENCY * combined["norm_crime_frequency"]
    + WEIGHT_SEVERITY * combined["norm_severity_score"]
    + WEIGHT_INVERSE_FOOTFALL * combined["norm_inverse_footfall"]
)

print("\nTVS score summary statistics (should range between 0 and 1):")
print(combined["TVS"].describe())

combined.to_csv("all_cities_tvs_detailed.csv", index=False)
print("\nSaved detailed TVS table -> all_cities_tvs_detailed.csv")

# --- Build a one-row-per-city summary (average TVS) ---
city_summary = (
    combined.groupby("city")
    .agg(
        average_tvs=("TVS", "mean"),
        max_tvs=("TVS", "max"),
        min_tvs=("TVS", "min"),
    )
    .reset_index()
    .sort_values("average_tvs", ascending=False)
)

print("\n--- CITY COMPARISON SUMMARY (sorted highest to lowest average TVS) ---")
print(city_summary)

city_summary.to_csv("city_tvs_summary.csv", index=False)
print("\nSaved city summary -> city_tvs_summary.csv")

print("\nPhase 3 (TVS calculation) complete.")