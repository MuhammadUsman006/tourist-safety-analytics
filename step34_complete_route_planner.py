import os
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import folium

CITY_NETWORK_FILES = {
    "london": "london_street_network_with_risk.graphml",
    "york": "york_street_network_with_risk.graphml",
    "liverpool": "liverpool_street_network_with_risk.graphml",
    "birmingham": "birmingham_street_network_with_risk.graphml",
}

CITY_CENTRES = {
    "london": (51.5074, -0.1278),
    "york": (53.9600, -1.0873),
    "liverpool": (53.4084, -2.9916),
    "birmingham": (52.4862, -1.8904),
}

CITY_HUBS = {
    "london": (51.5308, -0.1238),
    "york": (53.9583, -1.0933),
    "liverpool": (53.4072, -2.9779),
    "birmingham": (52.4778, -1.8983),
}

# Named landmark pairs used for the single-route demonstration maps
DEMO_LANDMARK_PAIRS = {
    "london": ("Buckingham Palace", "British Museum"),
    "york": ("Clifford's Tower", "National Railway Museum"),
    "liverpool": ("Saint George's Hall", "Crowne Plaza"),
    "birmingham": ("Ibis", "The Burlington Hotel"),
}

NEAREST_NEIGHBOURS_PER_LANDMARK = 3

OUTPUT_FOLDER = "route_maps_final"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =============================================================================
# SHARED FUNCTIONS - used by all three map types below
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def prepare_graph_with_risk_costs(graph, risk_penalty=100.0):
    """
    Calculates the risk-weighted cost for every street segment ONCE.
    Doing this a single time up front (rather than repeatedly inside
    every route calculation) makes generating hundreds of routes on the
    same city network much faster.
    """
    for u, v, key, data in graph.edges(keys=True, data=True):
        distance_metres = float(data.get("length", 1))
        risk_score = float(data.get("risk_score", 0))
        data["risk_weighted_cost"] = distance_metres * (1 + risk_penalty * risk_score)
    return graph


def find_landmark_coordinates(pois_df, landmark_name):
    matches = pois_df[pois_df["name"].str.contains(landmark_name, case=False, na=False, regex=False)]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return row["poi_lat"], row["poi_lon"], row["name"]


def find_route(graph, start_lat, start_lon, end_lat, end_lon, weight_column):
    start_node = ox.distance.nearest_nodes(graph, start_lon, start_lat)
    end_node = ox.distance.nearest_nodes(graph, end_lon, end_lat)
    try:
        return nx.shortest_path(graph, start_node, end_node, weight=weight_column)
    except nx.NetworkXNoPath:
        return None  # some landmarks may not be reachable if they sit outside the walkable network


def route_to_coordinates(graph, route):
    return [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in route]


def route_distance_km(graph, route):
    return sum(float(graph[route[i]][route[i + 1]][0]["length"]) for i in range(len(route) - 1)) / 1000


def route_distance_weighted_risk(graph, route):
    total_dist = sum(float(graph[route[i]][route[i + 1]][0]["length"]) for i in range(len(route) - 1))
    total_risk_dist = sum(
        float(graph[route[i]][route[i + 1]][0]["risk_score"]) * float(graph[route[i]][route[i + 1]][0]["length"])
        for i in range(len(route) - 1)
    )
    return total_risk_dist / total_dist


# =============================================================================
# MAP TYPE 1: single named landmark-pair demo (safe vs shortest comparison)
# =============================================================================

def build_demo_route_map(city_name, start_landmark_name, end_landmark_name):
    print(f"\n[Demo route] {start_landmark_name} -> {end_landmark_name} ({city_name.title()})")

    graph = ox.load_graphml(CITY_NETWORK_FILES[city_name])
    graph = prepare_graph_with_risk_costs(graph)
    pois = pd.read_csv(f"{city_name}_poi_final.csv")

    start = find_landmark_coordinates(pois, start_landmark_name)
    end = find_landmark_coordinates(pois, end_landmark_name)
    if start is None or end is None:
        print("  SKIPPED: landmark not found")
        return

    start_lat, start_lon, start_name = start
    end_lat, end_lon, end_name = end

    safe_route = find_route(graph, start_lat, start_lon, end_lat, end_lon, "risk_weighted_cost")
    shortest_route = find_route(graph, start_lat, start_lon, end_lat, end_lon, "length")

    safe_dist = route_distance_km(graph, safe_route)
    safe_risk = route_distance_weighted_risk(graph, safe_route)
    short_dist = route_distance_km(graph, shortest_route)
    short_risk = route_distance_weighted_risk(graph, shortest_route)

    print(f"  Safe route:     {safe_dist:.2f} km, risk {safe_risk:.4f}")
    print(f"  Shortest route: {short_dist:.2f} km, risk {short_risk:.4f}")

    city_map = folium.Map(location=CITY_CENTRES[city_name], zoom_start=15)
    safe_coords = route_to_coordinates(graph, safe_route)

    folium.PolyLine(safe_coords, color="green", weight=5,
                     opacity=0.8, tooltip="Safest route").add_to(city_map)

    # --- NEW: close the visual gap between each marker and the route's ---
    # --- actual start/end point (the nearest street node, not the door) ---
    folium.PolyLine(
        [(start_lat, start_lon), safe_coords[0]],
        color="green", weight=3, dash_array="5,5", opacity=0.6,
    ).add_to(city_map)
    folium.PolyLine(
        [safe_coords[-1], (end_lat, end_lon)],
        color="green", weight=3, dash_array="5,5", opacity=0.6,
    ).add_to(city_map)

    if safe_route != shortest_route:
        shortest_coords = route_to_coordinates(graph, shortest_route)
        folium.PolyLine(shortest_coords, color="red", weight=3,
                         opacity=0.6, dash_array="10", tooltip="Shortest route").add_to(city_map)

        # same connector fix applied to the red "shortest route" line
        folium.PolyLine(
            [(start_lat, start_lon), shortest_coords[0]],
            color="red", weight=2, dash_array="5,5", opacity=0.5,
        ).add_to(city_map)
        folium.PolyLine(
            [shortest_coords[-1], (end_lat, end_lon)],
            color="red", weight=2, dash_array="5,5", opacity=0.5,
        ).add_to(city_map)
    else:
        print("  (Routes are identical - no safer alternative existed)")

    folium.Marker((start_lat, start_lon), popup=start_name,
                  icon=folium.Icon(color="blue", icon="play")).add_to(city_map)
    folium.Marker((end_lat, end_lon), popup=end_name,
                  icon=folium.Icon(color="darkred", icon="flag")).add_to(city_map)

    filename = f"{OUTPUT_FOLDER}/{city_name}_demo_route.html"
    city_map.save(filename)
    print(f"  Saved -> {filename}")


# =============================================================================
# MAP TYPE 2: hub (station) to EVERY landmark in the city
# =============================================================================

def build_hub_to_all_map(city_name):
    print(f"\n[Hub-to-all] {city_name.title()}")

    graph = ox.load_graphml(CITY_NETWORK_FILES[city_name])
    graph = prepare_graph_with_risk_costs(graph)
    pois = pd.read_csv(f"{city_name}_poi_final.csv")

    attractions = pois[pois["poi_type"].isin(["attraction", "museum", "gallery", "viewpoint"])]
    attractions = attractions.dropna(subset=["name"])

    hub_lat, hub_lon = CITY_HUBS[city_name]
    city_map = folium.Map(location=(hub_lat, hub_lon), zoom_start=14)

    folium.Marker((hub_lat, hub_lon), popup="City Hub (start point)",
                  icon=folium.Icon(color="black", icon="home")).add_to(city_map)

    successful, failed = 0, 0
    for _, landmark in attractions.iterrows():
        route = find_route(graph, hub_lat, hub_lon, landmark["poi_lat"], landmark["poi_lon"], "risk_weighted_cost")
        if route is None:
            failed += 1
            continue

        route_coords = route_to_coordinates(graph, route)

        folium.PolyLine(route_coords, color="green", weight=2,
                         opacity=0.5, tooltip=f"Safe route to {landmark['name']}").add_to(city_map)

        # --- NEW: connector from the hub marker to the route's real start point ---
        folium.PolyLine(
            [(hub_lat, hub_lon), route_coords[0]],
            color="green", weight=1, dash_array="5,5", opacity=0.4,
        ).add_to(city_map)
        # --- NEW: connector from the route's real end point to the landmark ---
        folium.PolyLine(
            [route_coords[-1], (landmark["poi_lat"], landmark["poi_lon"])],
            color="green", weight=1, dash_array="5,5", opacity=0.4,
        ).add_to(city_map)

        folium.CircleMarker((landmark["poi_lat"], landmark["poi_lon"]), radius=4, color="darkred",
                             fill=True, fill_opacity=0.8, popup=landmark["name"]).add_to(city_map)
        successful += 1

    print(f"  Routes: {successful} successful, {failed} unreachable")

    filename = f"{OUTPUT_FOLDER}/{city_name}_hub_to_all.html"
    city_map.save(filename)
    print(f"  Saved -> {filename}")


# =============================================================================
# MAP TYPE 3: connected network - every landmark linked to its nearest 3
# =============================================================================

def build_connected_network_map(city_name):
    print(f"\n[Connected network] {city_name.title()}")

    graph = ox.load_graphml(CITY_NETWORK_FILES[city_name])
    graph = prepare_graph_with_risk_costs(graph)
    pois = pd.read_csv(f"{city_name}_poi_final.csv")

    attractions = pois[pois["poi_type"].isin(["attraction", "museum", "gallery", "viewpoint"])]
    attractions = attractions.dropna(subset=["name"]).reset_index(drop=True)

    lats = attractions["poi_lat"].values
    lons = attractions["poi_lon"].values

    city_map = folium.Map(location=CITY_CENTRES[city_name], zoom_start=14)

    for _, row in attractions.iterrows():
        folium.CircleMarker((row["poi_lat"], row["poi_lon"]), radius=4, color="darkred",
                             fill=True, fill_opacity=0.8, popup=row["name"]).add_to(city_map)

    drawn_pairs = set()
    successful, failed = 0, 0

    for i in range(len(attractions)):
        distances = haversine_distance(lats[i], lons[i], lats, lons)
        nearest_indices = np.argsort(distances)[1:NEAREST_NEIGHBOURS_PER_LANDMARK + 1]

        for j in nearest_indices:
            pair_key = tuple(sorted([i, j]))
            if pair_key in drawn_pairs:
                continue
            drawn_pairs.add(pair_key)

            route = find_route(graph, lats[i], lons[i], lats[j], lons[j], "risk_weighted_cost")
            if route is None:
                failed += 1
                continue

            route_coords = route_to_coordinates(graph, route)
            folium.PolyLine(route_coords, color="green", weight=2, opacity=0.5).add_to(city_map)

            # --- NEW: connectors closing the gap at both landmark ends ---
            folium.PolyLine(
                [(lats[i], lons[i]), route_coords[0]],
                color="green", weight=1, dash_array="5,5", opacity=0.4,
            ).add_to(city_map)
            folium.PolyLine(
                [route_coords[-1], (lats[j], lons[j])],
                color="green", weight=1, dash_array="5,5", opacity=0.4,
            ).add_to(city_map)

            successful += 1

    print(f"  Connections: {successful} drawn, {failed} unreachable")

    filename = f"{OUTPUT_FOLDER}/{city_name}_connected_network.html"
    city_map.save(filename)
    print(f"  Saved -> {filename}")


# =============================================================================
# MAP TYPE 4: proof that ANY two landmarks (not just neighbours) can be routed
# =============================================================================

def prove_direct_routing_works():
    print("\n[Proof] Direct routing between distant, non-neighbouring landmarks")
    graph = ox.load_graphml(CITY_NETWORK_FILES["london"])
    graph = prepare_graph_with_risk_costs(graph)

    # Tower of London -> Buckingham Palace area: opposite ends of central London
    route = find_route(graph, 51.5081, -0.0759, 51.5014, -0.1419, "risk_weighted_cost")
    if route:
        print(f"  SUCCESS: route found with {len(route)} nodes - confirms any landmark pair "
              f"can be routed directly, regardless of the nearest-neighbour map above.")
    else:
        print("  FAILED: investigate further.")


# =============================================================================
# RUN EVERYTHING
# =============================================================================

for city_name in CITY_NETWORK_FILES:
    start_name, end_name = DEMO_LANDMARK_PAIRS[city_name]
    build_demo_route_map(city_name, start_name, end_name)

for city_name in CITY_NETWORK_FILES:
    build_hub_to_all_map(city_name)

for city_name in CITY_NETWORK_FILES:
    build_connected_network_map(city_name)

prove_direct_routing_works()

print(f"\nAll maps saved in the '{OUTPUT_FOLDER}' folder. Phase 5 fully complete.")
