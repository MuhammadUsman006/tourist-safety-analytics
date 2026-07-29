import osmnx as ox

# Try several query phrasings AND osmnx's "structured query" dict format,
# which sometimes resolves administrative boundaries more reliably than
# a plain text string.
QUERIES_TO_TRY = [
    "York, United Kingdom",
    "City of York, England, United Kingdom",
    "York Metropolitan Borough, United Kingdom",
    {"city": "York", "country": "United Kingdom"},
]

for query in QUERIES_TO_TRY:
    print(f"\n=== Trying: {query} ===")
    try:
        gdf = ox.geocode_to_gdf(query)
        area = gdf.geometry.area.iloc[0]
        name = gdf["display_name"].iloc[0] if "display_name" in gdf.columns else "?"
        print(f"  area={area:.6f}, display_name={name}")
    except Exception as e:
        print(f"  FAILED: {e}")