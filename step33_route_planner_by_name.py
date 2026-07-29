import pandas as pd
import osmnx as ox
import networkx as nx

CITY_NETWORK_FILES = {
    "london": "london_street_network_with_risk.graphml",
    "york": "york_street_network_with_risk.graphml",
    "liverpool": "liverpool_street_network_with_risk.graphml",
    "birmingham": "birmingham_street_network_with_risk.graphml",
}


def find_landmark_coordinates(city_name, landmark_name):
    """
    Looks up a landmark BY NAME in our real POI data, rather than
    requiring hardcoded coordinates. This lets the route planner work
    for ANY of the landmarks we actually collected in Phase 1, not just
    a few manually typed examples.
    """
    pois = pd.read_csv(f"{city_name}_poi_final.csv")

    # Case-insensitive partial match, so "buckingham palace" or
    # "Buckingham" both successfully find "Buckingham Palace"
    matches = pois[pois["name"].str.contains(landmark_name, case=False, na=False)]

    if matches.empty:
        available_names = pois["name"].dropna().unique()
        raise ValueError(
            f"No landmark found matching '{landmark_name}' in {city_name}. "
            f"Try one of these instead: {list(available_names[:15])}..."
        )

    if len(matches) > 1:
        print(f"Multiple matches found for '{landmark_name}', using the first: {matches.iloc[0]['name']}")

    row = matches.iloc[0]
    return row["poi_lat"], row["poi_lon"], row["name"]


def find_route(city_name, start_lat, start_lon, end_lat, end_lon, avoid_risk=True, risk_penalty=100.0):
    graph = ox.load_graphml(CITY_NETWORK_FILES[city_name])

    start_node = ox.distance.nearest_nodes(graph, start_lon, start_lat)
    end_node = ox.distance.nearest_nodes(graph, end_lon, end_lat)

    if avoid_risk:
        for u, v, key, data in graph.edges(keys=True, data=True):
            distance_metres = float(data.get("length", 1))
            risk_score = float(data.get("risk_score", 0))
            data["risk_weighted_cost"] = distance_metres * (1 + risk_penalty * risk_score)
        weight_column = "risk_weighted_cost"
    else:
        weight_column = "length"

    route = nx.shortest_path(graph, start_node, end_node, weight=weight_column)

    total_distance_metres = sum(
        float(graph[route[i]][route[i + 1]][0]["length"])
        for i in range(len(route) - 1)
    )
    total_risk_times_distance = sum(
        float(graph[route[i]][route[i + 1]][0]["risk_score"]) *
        float(graph[route[i]][route[i + 1]][0]["length"])
        for i in range(len(route) - 1)
    )
    distance_weighted_average_risk = total_risk_times_distance / total_distance_metres

    return {
        "route_nodes": route,
        "total_distance_km": total_distance_metres / 1000,
        "average_risk_per_segment": distance_weighted_average_risk,
        "number_of_segments": len(route) - 1,
    }


def plan_route_between_landmarks(city_name, start_landmark_name, end_landmark_name):
    """
    The main function you'll actually use: give it a city and two
    landmark NAMES (not coordinates), and it does everything - looks up
    their real locations, finds both the safe and shortest routes, and
    prints a clear comparison.
    """
    start_lat, start_lon, start_full_name = find_landmark_coordinates(city_name, start_landmark_name)
    end_lat, end_lon, end_full_name = find_landmark_coordinates(city_name, end_landmark_name)

    print(f"\n=== Route: {start_full_name} -> {end_full_name} ({city_name.title()}) ===")

    safe = find_route(city_name, start_lat, start_lon, end_lat, end_lon, avoid_risk=True)
    shortest = find_route(city_name, start_lat, start_lon, end_lat, end_lon, avoid_risk=False)

    print(f"Safest route:   {safe['total_distance_km']:.2f} km, risk {safe['average_risk_per_segment']:.4f}")
    print(f"Shortest route: {shortest['total_distance_km']:.2f} km, risk {shortest['average_risk_per_segment']:.4f}")

    if safe['route_nodes'] == shortest['route_nodes']:
        print("Result: identical path (no safer alternative available for this journey)")
    else:
        extra_dist = safe['total_distance_km'] - shortest['total_distance_km']
        risk_saved = shortest['average_risk_per_segment'] - safe['average_risk_per_segment']
        print(f"Result: safe route is {extra_dist:+.3f} km different, risk reduced by {risk_saved:.4f}")

    return safe, shortest


# --- Try it with a few different landmark pairs, using NAMES not coordinates ---
plan_route_between_landmarks("london", "Buckingham Palace", "British Museum")
plan_route_between_landmarks("york", "Clifford's Tower", "National Railway Museum")
plan_route_between_landmarks("liverpool", "Saint George's Hall", "Crowne Plaza")
plan_route_between_landmarks("birmingham", "Ibis", "The Burlington Hotel")