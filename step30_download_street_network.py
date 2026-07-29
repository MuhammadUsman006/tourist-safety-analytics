import osmnx as ox

# --- Each English city's tourist-core bounding box ---
# Same bounding boxes used throughout this whole project, for
# consistency across every data source.
CITY_BBOXES = {
    "london": (51.47, -0.20, 51.55, -0.05),
    "york": (53.945, -1.11, 53.975, -1.05),
    "liverpool": (53.39, -3.02, 53.42, -2.96),
    "birmingham": (52.46, -1.93, 52.51, -1.86),
}


def download_city_network(city_name, bbox):
    print(f"\n--- Downloading {city_name.title()}'s street network ---")

    min_lat, min_lon, max_lat, max_lon = bbox
    north, south, east, west = max_lat, min_lat, max_lon, min_lon

    graph = ox.graph_from_bbox(
        bbox=(west, south, east, north),
        network_type="walk",
    )

    print(f"Nodes (intersections): {graph.number_of_nodes()}")
    print(f"Edges (street segments): {graph.number_of_edges()}")

    output_filename = f"{city_name}_street_network.graphml"
    ox.save_graphml(graph, filepath=output_filename)
    print(f"Saved -> {output_filename}")


for city_name, bbox in CITY_BBOXES.items():
    download_city_network(city_name, bbox)

print("\nAll 4 English city street networks downloaded.")