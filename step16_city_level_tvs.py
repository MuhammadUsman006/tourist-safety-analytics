import pandas as pd

# Load the detailed table we already built - it has crime_count,
# severity_score, and inverse_footfall for every row, across all 6
# cities, which is everything we need to build a properly fair
# CITY-LEVEL comparison.
detailed = pd.read_csv("all_cities_tvs_detailed.csv")


def min_max_normalise(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series * 0
    return (series - min_val) / (max_val - min_val)


# --- Aggregate RAW numbers per city (not already-normalised ones) ---
# This is the key fix: we go back to the raw crime_count and
# severity_score numbers, and only normalise them ONCE, across all 6
# cities together - so London's true crime scale actually gets
# compared against York's, Liverpool's, etc.
city_raw = (
    detailed.groupby("city")
    .apply(lambda g: pd.Series({
        "total_crime_count": g["crime_count"].sum(),
        # Weighted average severity - weights busier district-months
        # more heavily, so one quiet low-severity month doesn't count
        # equally to a busy high-severity month.
        "weighted_avg_severity": (g["crime_count"] * g["severity_score"]).sum() / g["crime_count"].sum(),
        "inverse_footfall": g["inverse_footfall"].iloc[0],  # same value repeated per city, just take one
    }))
    .reset_index()
)

print("Raw city-level aggregates (before normalising):")
print(city_raw)

# --- NOW normalise these three numbers GLOBALLY, across all 6 cities ---
city_raw["norm_crime_frequency"] = min_max_normalise(city_raw["total_crime_count"])
city_raw["norm_severity"] = min_max_normalise(city_raw["weighted_avg_severity"])
city_raw["norm_inverse_footfall"] = min_max_normalise(city_raw["inverse_footfall"])

# --- Combine using the proposal's weights ---
city_raw["city_level_TVS"] = (
    0.40 * city_raw["norm_crime_frequency"]
    + 0.40 * city_raw["norm_severity"]
    + 0.20 * city_raw["norm_inverse_footfall"]
)

city_raw = city_raw.sort_values("city_level_TVS", ascending=False)

print("\n--- CORRECTED CITY COMPARISON (globally normalised, fair across cities) ---")
print(city_raw[["city", "total_crime_count", "weighted_avg_severity", "norm_inverse_footfall", "city_level_TVS"]])

city_raw.to_csv("city_tvs_summary_corrected.csv", index=False)
print("\nSaved -> city_tvs_summary_corrected.csv")