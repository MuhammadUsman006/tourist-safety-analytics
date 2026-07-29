import osmnx as ox

for query in ["York, England, United Kingdom", "Liverpool, England, United Kingdom"]:
    print(f"\n=== Candidates for '{query}' ===")
    for i in range(1, 4):
        try:
            gdf = ox.geocode_to_gdf(query, which_result=i)
            area = gdf.geometry.area.iloc[0]
            place_type = gdf.get("place", ["?"]).iloc[0] if "place" in gdf.columns else "?"
            print(f"  Result {i}: area={area:.6f}, display_name={gdf.get('display_name', ['?']).iloc[0]}")
        except Exception as e:
            print(f"  Result {i}: failed ({e})")