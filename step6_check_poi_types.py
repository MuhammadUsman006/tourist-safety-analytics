import pandas as pd

# Tell pandas not to truncate long outputs, so we see every city, not
# just a hidden preview of the beginning and end
pd.set_option("display.max_rows", None)

CITY_NAMES = ["london", "york", "liverpool", "birmingham", "edinburgh", "glasgow"]

# --- Load every city's cleaned POI file and combine them into one table ---
all_poi_dataframes = []

for city_name in CITY_NAMES:
    filename = f"{city_name}_poi_cleaned.csv"
    try:
        df = pd.read_csv(filename)
        all_poi_dataframes.append(df)
    except FileNotFoundError:
        print(f"Could not find {filename} - skipping.")

combined_poi = pd.concat(all_poi_dataframes, ignore_index=True)

print(f"Total POIs across all 6 cities: {len(combined_poi)}")
print()

# --- Count how many of each poi_type exist, across ALL cities combined ---
print("Total POI type counts (all cities combined):")
print(combined_poi["poi_type"].value_counts())

print()

# --- Count how many of each poi_type exist, PER CITY ---
print("POI type counts, broken down by city:")
print(combined_poi.groupby("city")["poi_type"].value_counts())

london_only = combined_poi[combined_poi["city"] == "london"]
print()
print(f"London specifically: {len(london_only)} POIs")
print(london_only["poi_type"].value_counts())