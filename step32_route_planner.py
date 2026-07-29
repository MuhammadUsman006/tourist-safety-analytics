import osmnx as ox
import networkx as nx

CITY_NETWORK_FILES = {
    "london": "london_street_network_with_risk.graphml",
    "york": "york_street_network_with_risk.graphml",
    "liverpool": "liverpool_street_network_with_risk.graphml",
    "birmingham": "birmingham_street_network_with_risk.graphml",
}


def find_route(city_name, start_lat, start_lon, end_lat, end_lon, avoid_risk=True, risk_penalty=100.0):
    """
    Finds a walking route between two points in a given city.
    """
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

    # --- Distance-WEIGHTED average risk, not a simple segment count average ---
    # A route with many short segments shouldn't be judged the same as
    # one with fewer long segments - what actually matters to a tourist
    # is how much of their WALKING DISTANCE passes through risky areas,
    # not how many individual street segments they crossed. This
    # calculates: (risk x distance) added up for every segment, divided
    # by the total distance walked - giving a genuine "risk exposure
    # per kilometre" figure.
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


# --- TEST 1: Buckingham Palace -> London Eye ---
print("--- Test 1: Buckingham Palace to London Eye (risk_penalty=100) ---")

BUCKINGHAM_PALACE = (51.5014, -0.1419)
LONDON_EYE = (51.5033, -0.1195)

safe_route = find_route(
    "london",
    BUCKINGHAM_PALACE[0], BUCKINGHAM_PALACE[1],
    LONDON_EYE[0], LONDON_EYE[1],
    avoid_risk=True,
    risk_penalty=100.0,
)
print(f"Safest route: {safe_route['total_distance_km']:.2f} km, "
      f"distance-weighted avg risk {safe_route['average_risk_per_segment']:.4f}, "
      f"{safe_route['number_of_segments']} segments")

shortest_route = find_route(
    "london",
    BUCKINGHAM_PALACE[0], BUCKINGHAM_PALACE[1],
    LONDON_EYE[0], LONDON_EYE[1],
    avoid_risk=False,
)
print(f"Shortest route: {shortest_route['total_distance_km']:.2f} km, "
      f"distance-weighted avg risk {shortest_route['average_risk_per_segment']:.4f}, "
      f"{shortest_route['number_of_segments']} segments")

print(f"\nDifference: {safe_route['total_distance_km'] - shortest_route['total_distance_km']:.3f} km, "
      f"risk change: {shortest_route['average_risk_per_segment'] - safe_route['average_risk_per_segment']:.4f}")

# --- TEST 2: Tower of London -> British Museum (longer route) ---
print("\n\n--- Test 2: Tower of London to British Museum (longer route, risk_penalty=100) ---")

TOWER_OF_LONDON = (51.5081, -0.0759)
BRITISH_MUSEUM = (51.5194, -0.1270)

safe_route_2 = find_route(
    "london",
    TOWER_OF_LONDON[0], TOWER_OF_LONDON[1],
    BRITISH_MUSEUM[0], BRITISH_MUSEUM[1],
    avoid_risk=True,
    risk_penalty=100.0,
)
print(f"Safest route: {safe_route_2['total_distance_km']:.2f} km, "
      f"distance-weighted avg risk {safe_route_2['average_risk_per_segment']:.4f}, "
      f"{safe_route_2['number_of_segments']} segments")

shortest_route_2 = find_route(
    "london",
    TOWER_OF_LONDON[0], TOWER_OF_LONDON[1],
    BRITISH_MUSEUM[0], BRITISH_MUSEUM[1],
    avoid_risk=False,
)
print(f"Shortest route: {shortest_route_2['total_distance_km']:.2f} km, "
      f"distance-weighted avg risk {shortest_route_2['average_risk_per_segment']:.4f}, "
      f"{shortest_route_2['number_of_segments']} segments")

print(f"\nDifference: {safe_route_2['total_distance_km'] - shortest_route_2['total_distance_km']:.3f} km, "
      f"risk change: {shortest_route_2['average_risk_per_segment'] - safe_route_2['average_risk_per_segment']:.4f}")

print("\n\n--- Diagnostic: are Test 1's two routes actually IDENTICAL paths? ---")
print(f"Safe route nodes:     {safe_route['route_nodes']}")
print(f"Shortest route nodes: {shortest_route['route_nodes']}")
print(f"Are they the exact same path? {safe_route['route_nodes'] == shortest_route['route_nodes']}")