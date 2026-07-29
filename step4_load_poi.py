import json

# --- Which cities we expect to find POI files for ---
CITY_NAMES = ["london", "york", "liverpool", "birmingham", "edinburgh", "glasgow"]


def load_poi_file(city_name):
    """
    Opens one city's GeoJSON file and returns the raw parsed JSON data.
    GeoJSON is just a specific way of writing JSON to describe map
    features - Python's built-in json module can read it directly,
    no special library needed for this step.
    """
    filepath = f"raw_data_poi/{city_name}_poi.geojson"

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# --- Check every city, one at a time ---
for city_name in CITY_NAMES:
    print(f"\n--- {city_name.title()} ---")

    try:
        data = load_poi_file(city_name)
    except FileNotFoundError:
        print(f"FILE NOT FOUND: raw_data_poi/{city_name}_poi.geojson")
        print("Check the filename matches exactly (see message above).")
        continue

    # A GeoJSON file's actual content lives inside a list called
    # "features" - each entry in this list is one landmark/POI.
    features = data.get("features", [])
    print(f"Total POIs found: {len(features)}")

    # Show a sample of the first 3 POI names, just to sanity check the
    # data looks sensible (real landmark names, not empty/garbage).
    print("Sample POI names:")
    for feature in features[:3]:
        name = feature.get("properties", {}).get("name", "(no name)")
        print(f"  - {name}")

print("\nAll cities checked.")