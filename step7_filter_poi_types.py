import pandas as pd

CITY_NAMES = ["london", "york", "liverpool", "birmingham", "edinburgh", "glasgow"]

# --- The categories we consider genuine tourist-relevant locations ---
# Excluded on purpose:
#   - "artwork" - heavily skewed by inconsistent OSM mapping density
#     between cities (e.g. Birmingham has 401 vs a handful elsewhere),
#     which would unfairly bias the "distance to nearest POI" feature
#     in Birmingham's favour. This is a mapping-completeness artefact,
#     not a real difference in tourist density.
#   - "artwork:removed" / "artwork_removed" - these tags literally mean
#     the artwork no longer exists.
#   - "yes" - a vague, unspecified tourism tag with no real information.
#   - "convenience", "caravan_site" - not landmark-relevant.
KEEP_TYPES = {
    "hotel", "guest_house", "hostel", "apartment",
    "attraction", "museum", "gallery", "viewpoint",
    "theme_park", "aquarium", "picnic_site",
}

all_poi_dataframes = []
for city_name in CITY_NAMES:
    df = pd.read_csv(f"{city_name}_poi_cleaned.csv")
    all_poi_dataframes.append(df)

combined_poi = pd.concat(all_poi_dataframes, ignore_index=True)
print(f"Before filtering: {len(combined_poi)} POIs")

# Keep only rows whose poi_type is in our approved whitelist
filtered_poi = combined_poi[combined_poi["poi_type"].isin(KEEP_TYPES)].copy()
print(f"After filtering to genuine tourist-relevant types: {len(filtered_poi)} POIs")

print()
print("Remaining POI counts per city:")
print(filtered_poi.groupby("city")["poi_type"].count())

# Save each city's final, filtered POI file
for city_name in CITY_NAMES:
    city_df = filtered_poi[filtered_poi["city"] == city_name]
    output_filename = f"{city_name}_poi_final.csv"
    city_df.to_csv(output_filename, index=False)
    print(f"{city_name}: {len(city_df)} POIs saved -> {output_filename}")