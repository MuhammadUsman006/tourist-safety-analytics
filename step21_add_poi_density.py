import pandas as pd
import numpy as np

ENGLAND_CITIES = ["london", "york", "liverpool", "birmingham"]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Same distance formula we used before - real-world km between two GPS points."""
    R = 6371
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def build_poi_density_for_city(city_name):
    print(f"\n--- {city_name.title()} ---")

    # Load individual crimes (has LSOA code + exact lat/lon per crime)
    crimes = pd.read_csv(f"{city_name}_crimes_with_distance.csv")
    pois = pd.read_csv(f"{city_name}_poi_final.csv")

    # --- Find each LSOA's approximate centre point ---
    # We don't have official LSOA boundary shapes, so we approximate
    # each district's location as the AVERAGE position of all the
    # crimes that occurred within it - a reasonable stand-in for "roughly
    # where this district is."
    lsoa_centres = (
        crimes.groupby("LSOA code")
        .agg(centre_lat=("Latitude", "mean"), centre_lon=("Longitude", "mean"))
        .reset_index()
    )

    print(f"Found {len(lsoa_centres)} distinct LSOA districts")

    poi_lats = pois["poi_lat"].values
    poi_lons = pois["poi_lon"].values

    density_500m = []
    density_1km = []

    # For each district, count how many landmarks fall within 500m and
    # within 1km of its approximate centre point.
    for _, row in lsoa_centres.iterrows():
        distances = haversine_distance(row["centre_lat"], row["centre_lon"], poi_lats, poi_lons)
        density_500m.append((distances <= 0.5).sum())
        density_1km.append((distances <= 1.0).sum())

    lsoa_centres["poi_count_within_500m"] = density_500m
    lsoa_centres["poi_count_within_1km"] = density_1km

    print(f"Average landmarks within 500m across all districts: {np.mean(density_500m):.2f}")
    print(f"Average landmarks within 1km across all districts: {np.mean(density_1km):.2f}")

    # --- Merge this new density info into the existing model table ---
    model_table = pd.read_csv(f"{city_name}_district_month_model_table.csv")
    model_table = model_table.merge(
        lsoa_centres[["LSOA code", "poi_count_within_500m", "poi_count_within_1km"]],
        on="LSOA code",
        how="left",
    )

    output_filename = f"{city_name}_district_month_model_table_v2.csv"
    model_table.to_csv(output_filename, index=False)
    print(f"Saved updated table -> {output_filename}")


for city_name in ENGLAND_CITIES:
    build_poi_density_for_city(city_name)

print("\nPOI density feature added for all England/Wales cities.")