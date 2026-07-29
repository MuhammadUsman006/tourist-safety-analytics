"""
=====================================================================================
STEP 36 - FETCH REAL CITY BOUNDARY POLYGONS FOR THE TVS CHOROPLETH
=====================================================================================

WHY THIS SCRIPT EXISTS
------------------------
A genuine choropleth map needs boundary POLYGON data (the actual outline
shape of each city), not just a single centre-point coordinate. Your
project never downloaded this, because none of your crime/POI/TVS analysis
actually needed it - point coordinates were enough for everything else.

This script fetches each city's official administrative boundary directly
from OpenStreetMap (the same data source your POI collection already uses),
via osmnx's geocode_to_gdf() function - no manual shapefile download needed.

WHY THE QUERIES LOOK LIKE THIS
------------------------
A plain query like "London, UK" can accidentally match a small, unrelated
feature (e.g. "City of London" is a tiny ~1 square mile financial district,
NOT the whole city of London). Using each city's official administrative
area name - "Greater London" instead of "London", "City of York" instead
of "York" - forces Nominatim to return the correct, full-sized boundary.

IMPORTANT CAVEAT TO KNOW ABOUT
------------------------
OpenStreetMap's boundary for a city is the OFFICIAL administrative/council
boundary - this is usually a fair bit LARGER than the "tourist core"
bounding boxes your crime data was filtered down to in Phase 1. This is
completely normal and doesn't invalidate the choropleth: the map will
genuinely shade "the city of Liverpool" using "Liverpool's TVS score" -
it just means the shaded area covers more ground than your crime data
itself. Worth one sentence in your report acknowledging this, similar to
how you've already documented the England/Scotland data gap.

RUN THIS ONCE
------------------
Save as step36_get_city_boundaries.py, run it. It creates a single file:
city_boundaries.geojson - the dashboard's choropleth reads this directly.
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd

CITY_QUERIES = {
    "London": "Greater London, United Kingdom",
    "York": "York, United Kingdom",
    "Liverpool": "Liverpool, England, United Kingdom",
    "Birmingham": "Birmingham, West Midlands, United Kingdom",
    "Edinburgh": "City of Edinburgh, United Kingdom",
    "Glasgow": "Glasgow City, United Kingdom",
}

boundary_pieces = []

for city_name, query in CITY_QUERIES.items():
    print(f"Fetching boundary polygon for {city_name} ('{query}')...")
    try:
        gdf = ox.geocode_to_gdf(query)
        gdf["city"] = city_name
        boundary_pieces.append(gdf[["city", "geometry"]])
        print(f"  Success - boundary found.")
    except Exception as error:
        print(f"  FAILED for {city_name}: {error}")
        print(f"  Skipping - this city will fall back to the bubble map in the dashboard.")

if not boundary_pieces:
    print("\nNo boundaries were fetched successfully - nothing to save.")
else:
    all_boundaries = gpd.GeoDataFrame(pd.concat(boundary_pieces, ignore_index=True), crs=boundary_pieces[0].crs)
    all_boundaries.to_file("city_boundaries.geojson", driver="GeoJSON")
    print(f"\nSaved {len(all_boundaries)} city boundaries to city_boundaries.geojson")
    print("The dashboard's City Comparison tab can now render a real TVS choropleth.")