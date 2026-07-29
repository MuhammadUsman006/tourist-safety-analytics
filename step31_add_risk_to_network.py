import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx

ENGLAND_CITIES = ["london", "york", "liverpool", "birmingham"]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Same distance formula used throughout this project."""
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


def build_lsoa_risk_lookup(city_name):
    """
    Builds a simple table of (LSOA centre latitude, LSOA centre
    longitude, average TVS score for that district) - we'll use this to
    assign a risk score to every street segment based on which district
    it's physically closest to.
    """
    # Load individual crimes (has exact lat/lon per crime, used to
    # approximate each LSOA's centre point, same method as Step 21)
    crimes = pd.read_csv(f"{city_name}_crimes_with_distance.csv")
    crimes = crimes.dropna(subset=["LSOA code"])

    lsoa_centres = (
        crimes.groupby("LSOA code")
        .agg(centre_lat=("Latitude", "mean"), centre_lon=("Longitude", "mean"))
        .reset_index()
    )

    # Load our detailed TVS table and calculate each LSOA's AVERAGE TVS
    # score across all its months (since street risk shouldn't change
    # month-to-month for route planning purposes - we want one stable
    # risk value per street).
    tvs_detailed = pd.read_csv("all_cities_tvs_detailed.csv")
    city_tvs = tvs_detailed[tvs_detailed["city"] == city_name]

    lsoa_avg_tvs = (
        city_tvs.groupby("LSOA code")
        .agg(avg_tvs=("TVS", "mean"))
        .reset_index()
    )

    # Combine the district centre coordinates with their average TVS score
    lookup = lsoa_centres.merge(lsoa_avg_tvs, on="LSOA code", how="inner")

    print(f"{city_name.title()}: built risk lookup for {len(lookup)} districts")
    return lookup


def assign_risk_to_network(city_name):
    print(f"\n--- Processing {city_name.title()} ---")

    # Load the street network we downloaded in Step 30
    graph = ox.load_graphml(f"{city_name}_street_network.graphml")
    print(f"Loaded street network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    lsoa_lookup = build_lsoa_risk_lookup(city_name)

    lsoa_lats = lsoa_lookup["centre_lat"].values
    lsoa_lons = lsoa_lookup["centre_lon"].values
    lsoa_tvs = lsoa_lookup["avg_tvs"].values

    # --- Assign a risk score to every NODE (intersection) first ---
    # It's easier to calculate risk per intersection, then apply it to
    # the streets connecting them, than to calculate it per street
    # segment directly.
    node_risk = {}
    for node_id, node_data in graph.nodes(data=True):
        node_lat = node_data["y"]  # OSMnx stores latitude as "y"
        node_lon = node_data["x"]  # and longitude as "x"

        distances = haversine_distance(node_lat, node_lon, lsoa_lats, lsoa_lons)
        nearest_index = distances.argmin()
        node_risk[node_id] = lsoa_tvs[nearest_index]

    # --- Assign each EDGE (street segment) the AVERAGE risk of its two endpoints ---
    for u, v, key, edge_data in graph.edges(keys=True, data=True):
        risk_u = node_risk[u]
        risk_v = node_risk[v]
        edge_data["risk_score"] = (risk_u + risk_v) / 2

    all_edge_risks = [data["risk_score"] for _, _, data in graph.edges(data=True)]
    print(f"Street risk scores assigned. Average: {sum(all_edge_risks)/len(all_edge_risks):.3f}, "
          f"Min: {min(all_edge_risks):.3f}, Max: {max(all_edge_risks):.3f}")

    output_filename = f"{city_name}_street_network_with_risk.graphml"
    ox.save_graphml(graph, filepath=output_filename)
    print(f"Saved -> {output_filename}")


for city_name in ENGLAND_CITIES:
    assign_risk_to_network(city_name)

print("\nRisk scores added to all 4 street networks.")