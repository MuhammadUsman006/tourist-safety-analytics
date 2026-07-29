import pandas as pd
import numpy as np

# --- Only England/Wales cities have crime coordinates to work with ---
# (Scotland's data is yearly totals with no individual crime locations,
# as we established earlier - so this feature can't be built for
# Edinburgh/Glasgow)
ENGLAND_CITIES = ["london", "york", "liverpool", "birmingham"]


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the real-world distance (in kilometres) between two
    points on Earth, given their latitude/longitude. This is called the
    "Haversine formula" - it accounts for the Earth being a curved
    sphere, not a flat grid, so simple lat/lon subtraction alone would
    give the wrong distance.

    All four inputs can be single numbers OR whole numpy arrays (lists
    of numbers) - this lets us calculate the distance from ONE crime to
    MANY landmarks all at once, very quickly, instead of looping one at
    a time for every single landmark.
    """
    R = 6371  # Earth's radius in kilometres

    # Convert degrees to radians - a unit that trigonometry functions
    # (sin, cos) require internally to work correctly.
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def add_distance_feature(city_name):
    print(f"\n--- Processing {city_name.title()} ---")

    # Load this city's cleaned crime data and final POI data
    crimes = pd.read_csv(f"{city_name}_crimes_cleaned.csv")
    pois = pd.read_csv(f"{city_name}_poi_final.csv")

    print(f"Loaded {len(crimes):,} crimes and {len(pois)} POIs")

    if pois.empty:
        print(f"WARNING: no POIs available for {city_name} - skipping distance calculation.")
        return

    # Pull out the POI coordinates once, as numpy arrays, so we don't
    # have to keep re-reading them from the dataframe inside the loop
    # below - this makes the calculation much faster.
    poi_lats = pois["poi_lat"].values
    poi_lons = pois["poi_lon"].values

    nearest_distances = []

    # For every single crime, calculate its distance to EVERY landmark
    # in this city, then keep only the SMALLEST one (the nearest).
    # .iterrows() lets us go through the crimes table one row at a time.
    for _, crime_row in crimes.iterrows():
        distances_to_all_pois = haversine_distance(
            crime_row["Latitude"], crime_row["Longitude"],
            poi_lats, poi_lons
        )
        nearest_distances.append(distances_to_all_pois.min())

    crimes["distance_to_nearest_poi_km"] = nearest_distances

    print(f"Distance calculated. Average distance to nearest landmark: "
          f"{crimes['distance_to_nearest_poi_km'].mean():.3f} km")
    print(f"Closest crime-to-landmark distance: "
          f"{crimes['distance_to_nearest_poi_km'].min():.4f} km")
    print(f"Furthest crime-to-landmark distance: "
          f"{crimes['distance_to_nearest_poi_km'].max():.3f} km")

    output_filename = f"{city_name}_crimes_with_distance.csv"
    crimes.to_csv(output_filename, index=False)
    print(f"Saved -> {output_filename}")


# --- Process all 4 England/Wales cities ---
for city_name in ENGLAND_CITIES:
    add_distance_feature(city_name)

print("\nDistance feature engineering complete for all England/Wales cities.")