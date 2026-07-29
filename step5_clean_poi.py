import json
import pandas as pd

# --- Each city's tourist-core bounding box ---
# (min_lat, min_lon, max_lat, max_lon)
# These are the SAME bounding boxes used for the crime data, so both
# datasets describe exactly the same geographic area for each city -
# this consistency matters later when we combine crime + POI data
# together to build the Tourist Vulnerability Score in Phase 3.
CITY_BBOXES = {
    "london": (51.47, -0.20, 51.55, -0.05),
    "york": (53.945, -1.11, 53.975, -1.05),
    "liverpool": (53.39, -3.02, 53.42, -2.96),
    "birmingham": (52.46, -1.93, 52.51, -1.86),
    "edinburgh": (55.93, -3.25, 55.97, -3.15),
    "glasgow": (55.84, -4.30, 55.89, -4.20),
}


def get_point_from_geometry(geometry):
    """
    OSM features can be a single Point (like a statue), or a Polygon/
    LineString (like a park boundary or a long tourist trail). We only
    need ONE representative (lat, lon) location per feature, so:
      - if it's already a Point, we use that coordinate directly
      - if it's a Polygon/LineString, we average ALL of its coordinates
        together, which gives a reasonable approximate "centre point"
        without needing any extra specialist geometry library.
    GeoJSON always stores coordinates as [longitude, latitude] - the
    OPPOSITE order to how we normally say "latitude, longitude" out
    loud - so we have to be careful to unpack them correctly.
    """
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")

    if geom_type == "Point":
        lon, lat = coords
        return lat, lon

    # For anything else (Polygon, LineString, MultiPolygon...), flatten
    # every nested coordinate pair into one simple list, then average
    # them all together.
    flat_points = []

    def flatten(item):
        # A coordinate pair looks like [lon, lat] - a plain list of
        # two numbers. Anything else is a further nested list we need
        # to dig into first.
        if len(item) == 2 and all(isinstance(x, (int, float)) for x in item):
            flat_points.append(item)
        else:
            for sub_item in item:
                flatten(sub_item)

    flatten(coords)

    if not flat_points:
        return None, None

    avg_lon = sum(p[0] for p in flat_points) / len(flat_points)
    avg_lat = sum(p[1] for p in flat_points) / len(flat_points)
    return avg_lat, avg_lon


def process_city(city_name):
    filepath = f"raw_data_poi/{city_name}_poi.geojson"

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])

    rows = []
    for feature in features:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry")

        if geometry is None:
            continue

        lat, lon = get_point_from_geometry(geometry)
        if lat is None:
            continue

        rows.append({
            "name": properties.get("name"),          # may be empty - that's OK
            "poi_type": properties.get("tourism"),   # e.g. "hotel", "museum", "attraction"
            "poi_lat": lat,
            "poi_lon": lon,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # --- Filter to this city's tourist-core bounding box ---
    min_lat, min_lon, max_lat, max_lon = CITY_BBOXES[city_name]
    df = df[
        (df["poi_lat"] >= min_lat) & (df["poi_lat"] <= max_lat) &
        (df["poi_lon"] >= min_lon) & (df["poi_lon"] <= max_lon)
    ].copy()

    df["city"] = city_name
    return df


# --- Process all 6 cities ---
for city_name in CITY_BBOXES.keys():
    print(f"\n--- {city_name.title()} ---")

    df = process_city(city_name)

    if df.empty:
        print("No usable POIs found for this city.")
        continue

    print(f"POIs after filtering to tourist-core bounding box: {len(df)}")

    output_filename = f"{city_name}_poi_cleaned.csv"
    df.to_csv(output_filename, index=False)
    print(f"Saved -> {output_filename}")

    # Show a few examples so we can eyeball the result
    print("Sample:")
    print(df[["name", "poi_type"]].head(5))

print("\nAll POI data cleaned.")