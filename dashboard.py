"""
=====================================================================================
TOURIST SAFETY ANALYTICS DASHBOARD — Phase 6 (COM748 MSc Final Project)
=====================================================================================

Six tabs: City Comparison (with real choropleth), Seasonal & Yearly Trends
(with calendar heatmap and Scotland data-gap handling), Safe Route Planner,
Live Risk Prediction (landmark-based, contradiction-free season/month input),
City Risk Heatmap (custom gradient, zoom compensation, light/dark toggle),
and Risk Trend Forecast (anchored projection + R-squared fit-quality check).

BEFORE RUNNING THIS
----------------------
1. Run step35_save_final_model.py ONCE first (creates saved_model/ for Tab 4)
2. Run step36_get_city_boundaries.py ONCE first (creates city_boundaries.geojson
   for Tab 1's real choropleth)
3. This file, your CSVs, your _street_network_with_risk.graphml files, your
   _poi_final.csv files, and your _crimes_cleaned.csv files should all sit
   in the same folder (or edit the path variables in STEP 0 below).

HOW TO RUN
----------------------
streamlit run dashboard.py   (from the Terminal tab, NOT the Run button)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import osmnx as ox
import networkx as nx
import folium
import geopandas as gpd
import json
from folium.plugins import HeatMap, Fullscreen
from streamlit_folium import st_folium
import streamlit.components.v1 as components
import joblib

# =====================================================================================
# STEP 0 — CONFIGURATION
# =====================================================================================
CITY_TVS_SUMMARY_PATH = "city_tvs_summary_corrected.csv"
DETAILED_TVS_PATH = "all_cities_tvs_detailed.csv"
SAVED_MODEL_FOLDER = "saved_model"
CITY_BOUNDARIES_PATH = "city_boundaries.geojson"  # created by step36_get_city_boundaries.py

ENGLAND_CITIES = ["London", "York", "Liverpool", "Birmingham"]
SCOTLAND_CITIES = ["Edinburgh", "Glasgow"]
ALL_CITIES = ENGLAND_CITIES + SCOTLAND_CITIES

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

# Scotland's city centres are only needed for the TVS bubble map (Tab 1) -
# Edinburgh/Glasgow have no street network or POI files, so they're not
# added to CITY_NETWORK_FILES above.
ALL_CITY_CENTRES = {
    **CITY_CENTRES,
    "edinburgh": (55.9533, -3.1883),
    "glasgow": (55.8642, -4.2518),
}

# One consistent colour theme, used across every chart in the app so the
# whole dashboard feels like a single designed product rather than several
# separately-styled pages.
RISK_COLORS = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}
PLOTLY_TEMPLATE = "plotly_white"
ACCENT_COLOR = "#2c3e50"


# =====================================================================================
# STEP 1 — PAGE SETUP + LIGHT CUSTOM STYLING
# =====================================================================================
st.set_page_config(
    page_title="Tourist Safety Analytics — COM748",
    page_icon="🧭",
    layout="wide",
)

# A small amount of custom CSS to make the KPI cards and tab labels feel
# more like a finished product rather than raw default Streamlit styling.
# This is plain CSS injected once - nothing here touches your data or logic.
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 15px 10px;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }
    button[data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧭 Tourist Safety Analytics Dashboard")
st.caption(
    "MSc Final Project (COM748) — crime risk prediction and visualisation "
    "for tourists across six UK cities."
)


# =====================================================================================
# STEP 2 — DATA LOADING (cached)
# =====================================================================================

@st.cache_data
def load_city_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Your CSV's actual column is "city_level_TVS" - rename it to "TVS" right
    # here, once, so every other place in this file that expects a "TVS"
    # column just works without needing separate fixes everywhere.
    df = df.rename(columns={"city_level_TVS": "TVS"})
    # Your CSV has lowercase city names ("london") but city_boundaries.geojson
    # has proper capitalization ("London") - .str.strip() removes any hidden
    # leading/trailing spaces, .str.title() capitalizes the first letter of
    # each word, so both files match exactly when merged. Doing this INSIDE
    # the cached function (rather than after calling it) means it only runs
    # once - previously it re-ran on every single Streamlit rerun, including
    # ones triggered by unrelated interactions elsewhere in the app, which
    # was part of why the whole dashboard felt sluggish.
    df["city"] = df["city"].str.strip().str.title()
    return df


@st.cache_data
def load_detailed_tvs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Same lowercase-vs-capitalized mismatch as the TVS summary file -
    # standardize this file's city AND season names, stripping any hidden
    # whitespace too. Also moved inside the cached function for the same
    # performance reason as above - this file can have thousands of rows
    # (LSOA/month-level data across 4 cities), so re-cleaning it on every
    # rerun was a real, measurable cost.
    df["city"] = df["city"].str.strip().str.title()
    df["season"] = df["season"].str.strip().str.title()
    return df


@st.cache_data
def load_city_crimes(city_key: str) -> pd.DataFrame:
    """Loads the per-crime cleaned CSV (with Latitude/Longitude) for one
    England/Wales city — used by the heatmap tab."""
    return pd.read_csv(f"{city_key}_crimes_cleaned.csv")


@st.cache_data
def load_city_boundaries() -> gpd.GeoDataFrame:
    """Loads the real city boundary polygons created by
    step36_get_city_boundaries.py — used for the genuine TVS choropleth."""
    return gpd.read_file(CITY_BOUNDARIES_PATH)


try:
    city_summary_df = load_city_summary(CITY_TVS_SUMMARY_PATH)
    data_loaded = True
except FileNotFoundError:
    st.error(f"Could not find `{CITY_TVS_SUMMARY_PATH}`. Edit CITY_TVS_SUMMARY_PATH at the top of the script.")
    data_loaded = False

try:
    detailed_tvs_df = load_detailed_tvs(DETAILED_TVS_PATH)
except FileNotFoundError:
    detailed_tvs_df = None


# =====================================================================================
# STEP 3 — KPI SUMMARY CARDS (top of page, always visible)
# =====================================================================================
if data_loaded:
    highest_risk_row = city_summary_df.loc[city_summary_df["TVS"].idxmax()]
    lowest_risk_row = city_summary_df.loc[city_summary_df["TVS"].idxmin()]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Cities analysed", len(city_summary_df))
    kpi2.metric("Highest TVS", highest_risk_row["city"], f"{highest_risk_row['TVS']:.3f}")
    kpi3.metric("Lowest TVS", lowest_risk_row["city"], f"{lowest_risk_row['TVS']:.3f}")
    kpi4.metric("ML model accuracy", "76.1%", "XGBoost, 5-fold CV: ±0.6%")

st.divider()


# =====================================================================================
# STEP 4 — ROUTE PLANNER FUNCTIONS (identical to dashboard_v2.py)
# =====================================================================================

@st.cache_resource
def load_city_graph(city_name: str):
    graph = ox.load_graphml(CITY_NETWORK_FILES[city_name])
    risk_penalty = 100.0
    for u, v, key, data in graph.edges(keys=True, data=True):
        distance_metres = float(data.get("length", 1))
        risk_score = float(data.get("risk_score", 0))
        data["risk_weighted_cost"] = distance_metres * (1 + risk_penalty * risk_score)
    return graph


@st.cache_data
def load_city_pois(city_name: str) -> pd.DataFrame:
    return pd.read_csv(f"{city_name}_poi_final.csv")


def find_landmark_coordinates(pois_df, landmark_name):
    matches = pois_df[pois_df["name"].str.contains(landmark_name, case=False, na=False, regex=False)]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return row["poi_lat"], row["poi_lon"], row["name"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between one point and one or more other
    points (lat2/lon2 can be single numbers or numpy arrays) — used to
    compute real location features (distance to nearest POI, POI density)
    for the Live Risk Prediction tab, instead of asking the user to type
    in abstract GIS statistics they'd have no way to know."""
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def find_route(graph, start_lat, start_lon, end_lat, end_lon, weight_column):
    start_node = ox.distance.nearest_nodes(graph, start_lon, start_lat)
    end_node = ox.distance.nearest_nodes(graph, end_lon, end_lat)
    try:
        return nx.shortest_path(graph, start_node, end_node, weight=weight_column)
    except nx.NetworkXNoPath:
        return None


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


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Compass bearing (0-360 degrees, 0=North, going clockwise) from one
    point to another - used to work out which way the route is heading at
    each step, so turns can be described the way Google Maps does."""
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    diff_lon_rad = np.radians(lon2 - lon1)
    x = np.sin(diff_lon_rad) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(diff_lon_rad)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def bearing_to_compass(bearing):
    directions = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    index = int((bearing + 22.5) // 45) % 8
    return directions[index]


def describe_turn(angle_degrees):
    """Classifies the change in heading between two street segments into a
    Google-Maps-style instruction, using the signed angle between them."""
    if -20 <= angle_degrees <= 20:
        return "⬆️ Continue straight"
    elif 20 < angle_degrees <= 45:
        return "↗️ Turn slightly right"
    elif 45 < angle_degrees <= 135:
        return "➡️ Turn right"
    elif 135 < angle_degrees <= 180 or angle_degrees < -135:
        return "🔄 Make a sharp turn / U-turn"
    elif -45 <= angle_degrees < -20:
        return "↖️ Turn slightly left"
    else:
        return "⬅️ Turn left"


def describe_turn_plain(angle_degrees):
    """Same classification as describe_turn(), but plain text with no emoji -
    used for the live JavaScript navigation panel, to rule out any risk of
    emoji characters causing an encoding issue once embedded through
    Python -> JSON -> HTML iframe."""
    if -20 <= angle_degrees <= 20:
        return "Continue straight"
    elif 20 < angle_degrees <= 45:
        return "Turn slightly right"
    elif 45 < angle_degrees <= 135:
        return "Turn right"
    elif 135 < angle_degrees <= 180 or angle_degrees < -135:
        return "Make a sharp turn / U-turn"
    elif -45 <= angle_degrees < -20:
        return "Turn slightly left"
    else:
        return "Turn left"


def classify_unnamed_road(edge_data):
    """OSM footpaths (inside parks, pedestrian precincts, etc.) usually have
    no 'name' tag, but they DO have a 'highway' tag saying what kind of way
    it is. This gives a more honest label than a blanket "unnamed road" -
    e.g. a real park footpath reads as "a footpath", not a mystery road."""
    highway_type = edge_data.get("highway", "")
    if isinstance(highway_type, list):
        highway_type = highway_type[0] if highway_type else ""
    footpath_types = {"footway", "path", "pedestrian", "steps", "track"}
    if highway_type in footpath_types:
        return "a footpath"
    return "an unnamed road"


def build_navigation_data(graph, route):
    """Computes everything the LIVE walking-animation panel needs: the
    cumulative distance travelled at every point along the route, the
    total route distance, which turn-by-turn instruction is active at each
    point, and a per-segment risk category (low/medium/high) - this is
    what powers the risk-coloured route line and the "upcoming risk"
    warning, features Google Maps has no equivalent of since it has no
    concept of crime risk at all."""
    cumulative_m = [0.0]
    segment_lengths = []
    segment_risk_scores = []
    for i in range(len(route) - 1):
        edge_data = graph[route[i]][route[i + 1]][0]
        length = float(edge_data.get("length", 0))
        cumulative_m.append(cumulative_m[-1] + length)
        segment_lengths.append(length)
        segment_risk_scores.append(float(edge_data.get("risk_score", 0)))
    total_m = cumulative_m[-1]

    # Classify each segment's risk RELATIVE TO THIS ROUTE (using tertiles -
    # the bottom third of this route's own segments count as "low", middle
    # third "medium", top third "high"). This is deliberately relative
    # rather than a fixed absolute threshold, because raw risk_score values
    # vary city to city and route to route - a relative split guarantees
    # a meaningful, visible distinction on every single route, rather than
    # everything landing in one bucket if fixed cutoffs happen to be wrong
    # for a particular city's score distribution.
    sorted_scores = sorted(segment_risk_scores)
    n = len(sorted_scores)
    low_cutoff = sorted_scores[n // 3] if n >= 3 else sorted_scores[0]
    high_cutoff = sorted_scores[(2 * n) // 3] if n >= 3 else sorted_scores[-1]

    segment_risk_categories = []
    for score in segment_risk_scores:
        if score <= low_cutoff:
            segment_risk_categories.append("low")
        elif score <= high_cutoff:
            segment_risk_categories.append("medium")
        else:
            segment_risk_categories.append("high")

    runs = []
    for i in range(len(route) - 1):
        node_a, node_b = route[i], route[i + 1]
        edge_data = graph[node_a][node_b][0]

        street_name = edge_data.get("name", None)
        if isinstance(street_name, list):
            street_name = street_name[0] if street_name else None
        if not isinstance(street_name, str) or pd.isna(street_name):
            street_name = classify_unnamed_road(edge_data)

        segment_length = float(edge_data.get("length", 0))
        lat1, lon1 = graph.nodes[node_a]["y"], graph.nodes[node_a]["x"]
        lat2, lon2 = graph.nodes[node_b]["y"], graph.nodes[node_b]["x"]
        segment_bearing = calculate_bearing(lat1, lon1, lat2, lon2)

        if runs and runs[-1]["name"] == street_name:
            runs[-1]["distance"] += segment_length
            runs[-1]["end_bearing"] = segment_bearing
            runs[-1]["end_index"] = i + 1
        else:
            runs.append({
                "name": street_name, "distance": segment_length,
                "start_bearing": segment_bearing, "end_bearing": segment_bearing,
                "end_index": i + 1,
            })

    MIN_STEP_DISTANCE_M = 15
    merged_runs = []
    for run in runs:
        if merged_runs and run["distance"] < MIN_STEP_DISTANCE_M:
            merged_runs[-1]["distance"] += run["distance"]
            merged_runs[-1]["end_bearing"] = run["end_bearing"]
            merged_runs[-1]["end_index"] = run["end_index"]
        else:
            merged_runs.append(dict(run))
    runs = merged_runs

    steps = []
    for i, run in enumerate(runs):
        if i == 0:
            instruction = f"Head {bearing_to_compass(run['start_bearing'])} on {run['name']}"
        else:
            turn_angle = (run["start_bearing"] - runs[i - 1]["end_bearing"] + 180) % 360 - 180
            instruction = f"{describe_turn_plain(turn_angle)} onto {run['name']}"
        steps.append({"instruction": instruction, "end_index": run["end_index"]})

    return {
        "steps": steps, "cumulative_m": cumulative_m, "total_m": total_m,
        "segment_risk_categories": segment_risk_categories, "segment_lengths": segment_lengths,
    }


def build_turn_by_turn_directions(graph, route):
    """Builds a Google-Maps-style list of turn-by-turn directions from a
    route (a list of graph node IDs). Groups consecutive street segments
    that share the same real street name (from OSMnx) into a single step,
    merges very short graph segments into the previous step (so tiny
    junction nodes in the underlying street-network data don't get
    announced as their own separate "turn" the way a person walking
    wouldn't notice them), and works out the turn direction at each
    street-name change using the change in compass bearing."""
    if len(route) < 2:
        return []

    # First pass: group the route into "runs" of consecutive same-named streets
    runs = []
    for i in range(len(route) - 1):
        node_a, node_b = route[i], route[i + 1]
        edge_data = graph[node_a][node_b][0]

        street_name = edge_data.get("name", None)
        if isinstance(street_name, list):
            street_name = street_name[0] if street_name else None
        if not isinstance(street_name, str) or pd.isna(street_name):
            street_name = classify_unnamed_road(edge_data)

        segment_length = float(edge_data.get("length", 0))
        lat1, lon1 = graph.nodes[node_a]["y"], graph.nodes[node_a]["x"]
        lat2, lon2 = graph.nodes[node_b]["y"], graph.nodes[node_b]["x"]
        segment_bearing = calculate_bearing(lat1, lon1, lat2, lon2)

        if runs and runs[-1]["name"] == street_name:
            runs[-1]["distance"] += segment_length
            runs[-1]["end_bearing"] = segment_bearing
        else:
            runs.append({
                "name": street_name, "distance": segment_length,
                "start_bearing": segment_bearing, "end_bearing": segment_bearing,
            })

    # Second pass: fold any very short run (a graph junction too small to
    # notice while actually walking) into the PREVIOUS run, rather than
    # announcing a separate turn for it and then immediately turning again.
    MIN_STEP_DISTANCE_M = 15
    merged_runs = []
    for run in runs:
        if merged_runs and run["distance"] < MIN_STEP_DISTANCE_M:
            merged_runs[-1]["distance"] += run["distance"]
            merged_runs[-1]["end_bearing"] = run["end_bearing"]
        else:
            merged_runs.append(dict(run))
    runs = merged_runs

    # Third pass: turn each surviving run into a numbered instruction
    steps = []
    for i, run in enumerate(runs):
        if i == 0:
            instruction = f"🚩 Head {bearing_to_compass(run['start_bearing'])} on {run['name']}"
        else:
            turn_angle = (run["start_bearing"] - runs[i - 1]["end_bearing"] + 180) % 360 - 180
            instruction = f"{describe_turn(turn_angle)} onto {run['name']}"
        steps.append({"instruction": instruction, "distance_m": run["distance"]})

    return steps


def build_route_folium_map(graph, city_key, start_lat, start_lon, start_name,
                            end_lat, end_lon, end_name, safe_coords, safe_route, shortest_coords,
                            walker_index=0):
    """Builds the route map. walker_index (0 to len(safe_coords)-1) controls
    where the "you are here" marker and the greyed-out "already walked"
    line are drawn - driven by a Streamlit slider in Tab 3, rather than an
    injected JavaScript animation. This trades the smooth auto-play motion
    for something that reliably works: every frame is rendered fresh by
    Python on each Streamlit rerun, the same way every other chart in this
    dashboard already works, instead of depending on custom JS running
    correctly inside an embedded iframe."""
    city_map = folium.Map(location=CITY_CENTRES[city_key], zoom_start=15, tiles="cartodbpositron")

    # Computed early because the risk-coloured route segments (below) need
    # the per-segment risk categories this returns.
    nav_data = build_navigation_data(graph, safe_route)
    segment_risk_categories = nav_data["segment_risk_categories"]

    # Colour each stretch of the safe route by ITS OWN risk level, rather
    # than drawing the whole route as one flat green line. This is
    # something Google Maps has no equivalent of, since it has no concept
    # of crime risk at all - a tourist can see exactly which stretch of
    # their "safest" route is still comparatively riskier, not just trust
    # a single uniform colour for the whole path.
    #
    # Performance note: drawing ONE polyline PER RAW STREET EDGE (which the
    # earlier version did) can mean hundreds of separate map objects for a
    # longer route, and every single one has to be recreated on every
    # Streamlit rerun - including every time the walk-through slider moves.
    # That was the main cause of the dashboard feeling slow. Instead, this
    # merges consecutive edges that share the same risk category into one
    # polyline per stretch, which typically cuts the object count from
    # hundreds down to a handful, with no loss of colour information.
    RISK_SEGMENT_COLORS = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}

    merged_segments = []  # each: {"category": str, "coords": [points...]}
    for i in range(len(safe_coords) - 1):
        category = segment_risk_categories[i] if i < len(segment_risk_categories) else "low"
        if merged_segments and merged_segments[-1]["category"] == category:
            merged_segments[-1]["coords"].append(safe_coords[i + 1])
        else:
            merged_segments.append({"category": category, "coords": [safe_coords[i], safe_coords[i + 1]]})

    for segment in merged_segments:
        folium.PolyLine(
            segment["coords"],
            color=RISK_SEGMENT_COLORS[segment["category"]], weight=5, opacity=0.85,
        ).add_to(city_map)

    folium.PolyLine([(start_lat, start_lon), safe_coords[0]],
                     color="green", weight=3, dash_array="5,5", opacity=0.6).add_to(city_map)
    folium.PolyLine([safe_coords[-1], (end_lat, end_lon)],
                     color="green", weight=3, dash_array="5,5", opacity=0.6).add_to(city_map)

    # A small legend explaining the risk-segment colouring, since a route
    # made of many small coloured pieces needs a key to be understood at a
    # glance.
    legend_html = """
    <div style="position: fixed; top: 80px; left: 10px; z-index: 999;
                background-color: white; padding: 8px 12px; border-radius: 6px;
                border: 2px solid #666; font-size: 12px; box-shadow: 0 1px 5px rgba(0,0,0,0.3);">
        <div style="font-weight:bold; margin-bottom:4px;">Route risk (this street)</div>
        <div><span style="color:#2ecc71;">●</span> Lower risk</div>
        <div><span style="color:#f39c12;">●</span> Medium risk</div>
        <div><span style="color:#e74c3c;">●</span> Higher risk</div>
    </div>
    """
    city_map.get_root().html.add_child(folium.Element(legend_html))

    if shortest_coords is not None:
        folium.PolyLine(shortest_coords, color="grey", weight=3,
                         opacity=0.6, dash_array="10", tooltip="Shortest route").add_to(city_map)
        folium.PolyLine([(start_lat, start_lon), shortest_coords[0]],
                         color="grey", weight=2, dash_array="5,5", opacity=0.5).add_to(city_map)
        folium.PolyLine([shortest_coords[-1], (end_lat, end_lon)],
                         color="grey", weight=2, dash_array="5,5", opacity=0.5).add_to(city_map)

    folium.Marker((start_lat, start_lon), popup=start_name,
                  icon=folium.Icon(color="blue", icon="play")).add_to(city_map)
    folium.Marker((end_lat, end_lon), popup=end_name,
                  icon=folium.Icon(color="darkred", icon="flag")).add_to(city_map)

    # Clamp walker_index to a valid range defensively.
    walker_index = max(0, min(walker_index, len(safe_coords) - 1))

    # The "already walked" portion, greyed out - computed directly in
    # Python from the slider position, no JavaScript involved.
    if walker_index > 0:
        folium.PolyLine(
            safe_coords[: walker_index + 1], color="#9e9e9e", weight=6, opacity=0.9,
        ).add_to(city_map)

    # The "you are here" marker, placed at the slider's current position.
    folium.Marker(
        location=list(safe_coords[walker_index]),
        icon=folium.DivIcon(html="""
            <div style="background-color:#1a73e8; width:16px; height:16px;
                        border-radius:50%; border:3px solid white;
                        box-shadow:0 0 4px rgba(0,0,0,0.5);"></div>
        """),
    ).add_to(city_map)

    return city_map



# =====================================================================================
# STEP 5 — LIVE PREDICTION MODEL LOADING (cached)
# =====================================================================================

@st.cache_resource
def load_prediction_model():
    try:
        model = joblib.load(f"{SAVED_MODEL_FOLDER}/xgboost_final_model.pkl")
        scaler = joblib.load(f"{SAVED_MODEL_FOLDER}/scaler.pkl")
        target_encoder = joblib.load(f"{SAVED_MODEL_FOLDER}/target_encoder.pkl")
        column_order = joblib.load(f"{SAVED_MODEL_FOLDER}/feature_column_order.pkl")
        return model, scaler, target_encoder, column_order
    except FileNotFoundError:
        return None


prediction_assets = load_prediction_model()


# =====================================================================================
# STEP 6 — TABS
# =====================================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 City Comparison",
    "📈 Seasonal & Yearly Trends",
    "🗺️ Safe Route Planner",
    "🎯 Live Risk Prediction",
    "🔥 City Risk Heatmap",
    "🔮 Risk Trend Forecast",
])


# -------------------------------------------------------------------------------
# TAB 1 — CITY COMPARISON
# -------------------------------------------------------------------------------
with tab1:
    st.header("Tourist Vulnerability Score (TVS) — City Comparison")
    st.markdown(
        "TVS combines crime frequency (40%), severity-weighted crime type (40%), "
        "and inverse visitor footfall (20%), normalised **globally** across all "
        "six cities so scores are directly comparable."
    )

    if not data_loaded:
        st.warning("City summary data not loaded — see error above.")
    else:
        sorted_df = city_summary_df.sort_values("TVS", ascending=False)

        fig = px.bar(
            sorted_df, x="city", y="TVS", color="TVS",
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            title="Global TVS Ranking (higher = higher tourist crime risk)",
            labels={"TVS": "Tourist Vulnerability Score", "city": "City"},
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "ℹ️ **Edinburgh and Glasgow** are included here because TVS only needs "
            "yearly category totals. However, they are **excluded** from the Safe "
            "Route Planner, Live Risk Prediction, and Heatmap tabs, because Police "
            "Scotland's published data has no individual crime coordinates."
        )

        st.subheader("Full ranking table")
        st.dataframe(sorted_df[["city", "TVS"]].reset_index(drop=True), use_container_width=True)

        st.subheader("TVS Choropleth Map (all 6 cities)")

        @st.cache_resource
        def build_choropleth_map(tvs_pairs):
            """Builds the full choropleth map, cached and keyed on the
            actual (city, TVS) values. Building this from scratch is
            expensive (merging boundary polygons, building the Choropleth
            and GeoJson layers), and it never actually changes during a
            session - without this cache, it was being rebuilt on EVERY
            Streamlit rerun, including ones triggered by unrelated
            interactions elsewhere in the app (e.g. moving the Tab 3
            slider), which was a real contributor to the whole dashboard
            feeling sluggish."""
            boundaries_gdf = load_city_boundaries()
            tvs_df = pd.DataFrame(list(tvs_pairs), columns=["city", "TVS"])
            merged_gdf = boundaries_gdf.merge(tvs_df, on="city", how="left")

            choropleth_map = folium.Map(location=(54.0, -3.0), zoom_start=5, tiles="cartodbpositron")

            folium.Choropleth(
                geo_data=merged_gdf.__geo_interface__,
                data=merged_gdf,
                columns=["city", "TVS"],
                key_on="feature.properties.city",
                fill_color="RdYlGn_r",
                fill_opacity=0.7,
                line_opacity=0.6,
                line_color="black",
                legend_name="Tourist Vulnerability Score (TVS)",
                nan_fill_color="lightgrey",
            ).add_to(choropleth_map)

            folium.GeoJson(
                merged_gdf.__geo_interface__,
                style_function=lambda x: {"fillOpacity": 0, "weight": 0},
                tooltip=folium.GeoJsonTooltip(fields=["city", "TVS"], aliases=["City:", "TVS:"]),
            ).add_to(choropleth_map)

            return choropleth_map

        try:
            st.caption(
                "A genuine choropleth using real administrative boundary polygons from "
                "OpenStreetMap (fetched via step36_get_city_boundaries.py). Note: OSM's "
                "official boundary for a city is usually larger than the 'tourist core' "
                "bounding box your crime data was filtered to in Phase 1 — the shading "
                "reflects the whole administrative city, using that city's overall TVS score."
            )

            tvs_pairs = tuple(sorted_df[["city", "TVS"]].itertuples(index=False, name=None))
            choropleth_map = build_choropleth_map(tvs_pairs)
            st_folium(choropleth_map, width=None, height=550, returned_objects=[])

        except (FileNotFoundError, Exception) as error:
            st.warning(
                f"Real city boundaries not found (`{CITY_BOUNDARIES_PATH}`). Falling back to "
                f"a proportional symbol map instead. Run `step36_get_city_boundaries.py` once "
                f"to fetch real boundary polygons and unlock the full choropleth."
            )

            risk_map = folium.Map(location=(53.5, -2.5), zoom_start=5, tiles="cartodbpositron")

            for _, row in sorted_df.iterrows():
                city_key = row["city"].lower()
                if city_key not in ALL_CITY_CENTRES:
                    continue
                tvs_value = row["TVS"]
                if tvs_value >= 0.45:
                    colour = RISK_COLORS["high"]
                elif tvs_value >= 0.25:
                    colour = RISK_COLORS["medium"]
                else:
                    colour = RISK_COLORS["low"]

                folium.CircleMarker(
                    location=ALL_CITY_CENTRES[city_key],
                    radius=10 + (tvs_value * 30),
                    color=colour, fill=True, fill_color=colour, fill_opacity=0.7,
                    popup=f"{row['city']}: TVS {tvs_value:.3f}",
                    tooltip=f"{row['city']}: TVS {tvs_value:.3f}",
                ).add_to(risk_map)

            st_folium(risk_map, width=None, height=500, returned_objects=[])


# -------------------------------------------------------------------------------
# TAB 2 — SEASONAL & YEARLY TRENDS
# -------------------------------------------------------------------------------
with tab2:
    st.header("Seasonal and Yearly Crime Trends")

    if detailed_tvs_df is None:
        st.warning(f"Could not find `{DETAILED_TVS_PATH}`. Edit DETAILED_TVS_PATH at the top of the script.")
    else:
        selected_city = st.selectbox("Choose a city", ALL_CITIES, key="trend_city")
        city_data = detailed_tvs_df[detailed_tvs_df["city"] == selected_city]

        if city_data.empty:
            st.warning(f"No rows found for '{selected_city}' in the detailed TVS file.")
        else:
            if "year" in city_data.columns:
                yearly = city_data.groupby("year")["TVS"].mean().reset_index()
                fig_year = px.line(yearly, x="year", y="TVS", markers=True,
                                    title=f"{selected_city}: Average TVS by Year",
                                    template=PLOTLY_TEMPLATE)
                fig_year.update_traces(line_color=ACCENT_COLOR, marker=dict(size=10))
                # Force the x-axis to only show whole years (2023, 2024, ...)
                # instead of Plotly inventing decimal ticks like 2023.5.
                fig_year.update_xaxes(dtick=1, tickformat="d")
                st.plotly_chart(fig_year, use_container_width=True)

            if "season" in city_data.columns:
                season_order = ["Winter", "Spring", "Summer", "Autumn"]
                seasonal = city_data.groupby("season")["TVS"].mean()
                # .reindex() with a plain list of season names (not a
                # pd.Categorical dtype) keeps "season" as ordinary text,
                # so the chart always shows the real names - "Winter",
                # "Spring", etc. - never numeric category codes.
                seasonal = seasonal.reindex(season_order).reset_index()
                seasonal_with_data = seasonal.dropna(subset=["TVS"])  # drop seasons with no data for this city

                if seasonal_with_data.empty:
                    if selected_city in SCOTLAND_CITIES:
                        # This is an expected, documented data limitation, not a bug -
                        # Police Scotland's published data has no month/season
                        # granularity, only yearly totals per category (see Phase 2).
                        st.info(
                            f"ℹ️ Seasonal breakdown isn't available for {selected_city}. "
                            f"Police Scotland's published data only provides yearly totals "
                            f"per crime category — there's no month/season-level detail to "
                            f"break down, unlike the England/Wales cities."
                        )
                    else:
                        # For an England/Wales city, this would be genuinely unexpected -
                        # show the diagnostic so a real problem doesn't go unnoticed.
                        actual_values = city_data["season"].unique().tolist()
                        st.warning(
                            f"No seasonal data could be matched for {selected_city}. "
                            f"Expected one of {season_order}, but the actual values found "
                            f"in the data are: {actual_values}"
                        )
                else:
                    fig_season = px.bar(seasonal_with_data, x="season", y="TVS",
                                         category_orders={"season": season_order},
                                         title=f"{selected_city}: Average TVS by Season",
                                         template=PLOTLY_TEMPLATE, color="TVS",
                                         color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"])
                    fig_season.update_layout(coloraxis_showscale=False)
                    fig_season.update_xaxes(type="category")
                    st.plotly_chart(fig_season, use_container_width=True)

            # --- Seasonal Risk Calendar Heatmap (matches proposal wording exactly) ---
            # A genuine Year x Season grid, coloured by TVS - this is the "calendar
            # heatmap" your proposal specifies, distinct from the bar chart above.
            if "season" in city_data.columns and "year" in city_data.columns:
                st.subheader(f"{selected_city}: Seasonal Risk Calendar")

                season_order = ["Winter", "Spring", "Summer", "Autumn"]
                calendar_data = (
                    city_data.groupby(["year", "season"])["TVS"].mean().reset_index()
                )
                # pivot() reshapes the long list of (year, season, TVS) rows into a
                # grid: one row per year, one column per season - exactly the shape
                # a heatmap needs.
                calendar_grid = calendar_data.pivot(index="year", columns="season", values="TVS")
                calendar_grid = calendar_grid.reindex(columns=season_order)

                if calendar_grid.dropna(how="all").empty:
                    if selected_city in SCOTLAND_CITIES:
                        st.info(
                            f"ℹ️ A seasonal risk calendar isn't available for {selected_city}, "
                            f"for the same reason as above — Police Scotland's data has no "
                            f"month/season-level detail, only yearly category totals."
                        )
                    else:
                        actual_values = city_data["season"].unique().tolist()
                        st.warning(
                            f"No seasonal data could be matched for {selected_city}'s calendar. "
                            f"Expected one of {season_order}, but the actual values found "
                            f"in the data are: {actual_values}"
                        )
                else:
                    fig_calendar = px.imshow(
                        calendar_grid,
                        labels=dict(x="Season", y="Year", color="TVS"),
                        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                        aspect="auto",
                        text_auto=".2f",
                        title=f"{selected_city}: Risk Calendar (Year × Season)",
                    )
                    fig_calendar.update_layout(template=PLOTLY_TEMPLATE)
                    # Force both axes to treat their values as text/whole-category
                    # labels, not continuous numbers - this keeps years as whole
                    # numbers (2023, 2024) and seasons as their real names.
                    fig_calendar.update_xaxes(type="category")
                    fig_calendar.update_yaxes(type="category")
                    st.plotly_chart(fig_calendar, use_container_width=True)


# -------------------------------------------------------------------------------
# TAB 3 — SAFE ROUTE PLANNER
# -------------------------------------------------------------------------------
with tab3:
    st.header("Interactive Safe Route Planner")
    st.caption(
        "Only available for London, York, Liverpool, and Birmingham — Edinburgh "
        "and Glasgow have no street-level coordinates to route between."
    )

    route_city_display = st.selectbox("Choose a city", ENGLAND_CITIES, key="route_city")
    route_city_key = route_city_display.lower()

    pois_df = load_city_pois(route_city_key)
    landmark_choices = sorted(pois_df["name"].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        start_landmark = st.selectbox("Start landmark", landmark_choices, key="start_landmark")
    with col2:
        default_end_index = 1 if len(landmark_choices) > 1 else 0
        end_landmark = st.selectbox("End landmark", landmark_choices, index=default_end_index, key="end_landmark")

    if st.button("Find safe route", type="primary"):
        if start_landmark == end_landmark:
            st.warning("Please choose two different landmarks.")
            st.session_state.pop("route_result", None)
        else:
            with st.spinner(f"Loading {route_city_display}'s street network and calculating routes..."):
                graph = load_city_graph(route_city_key)
                start = find_landmark_coordinates(pois_df, start_landmark)
                end = find_landmark_coordinates(pois_df, end_landmark)
                start_lat, start_lon, start_name = start
                end_lat, end_lon, end_name = end
                safe_route = find_route(graph, start_lat, start_lon, end_lat, end_lon, "risk_weighted_cost")
                shortest_route = find_route(graph, start_lat, start_lon, end_lat, end_lon, "length")

            if safe_route is None:
                st.error("No route could be found between these two landmarks.")
                st.session_state.pop("route_result", None)
            else:
                safe_coords = route_to_coordinates(graph, safe_route)
                shortest_coords = route_to_coordinates(graph, shortest_route) if safe_route != shortest_route else None

                # Store everything needed to render the walk-through in
                # session_state, keyed so it survives later reruns (e.g. the
                # ones triggered by moving the slider below). Without this,
                # the slider - and everything else in this block - would
                # stop working the moment you touched it: st.button() only
                # reports True on the exact rerun right after the click, so
                # code nested inside "if st.button(...)" simply doesn't run
                # again on the next rerun, which is what was happening here.
                st.session_state["route_result"] = {
                    "route_city_key": route_city_key,
                    "start_lat": start_lat, "start_lon": start_lon, "start_name": start_name,
                    "end_lat": end_lat, "end_lon": end_lon, "end_name": end_name,
                    "safe_route": safe_route, "safe_coords": safe_coords,
                    "shortest_route": shortest_route, "shortest_coords": shortest_coords,
                    "safe_dist": route_distance_km(graph, safe_route),
                    "safe_risk": route_distance_weighted_risk(graph, safe_route),
                }
                if shortest_coords is not None:
                    st.session_state["route_result"]["short_dist"] = route_distance_km(graph, shortest_route)
                    st.session_state["route_result"]["short_risk"] = route_distance_weighted_risk(graph, shortest_route)
                # Reset the slider back to the start for a freshly-found route.
                st.session_state["walker_position"] = 0

    # Rendering happens here, OUTSIDE the button's if-block, reading from
    # session_state - this is what lets the slider (and the map/panel that
    # depend on it) keep working across reruns triggered by moving it.
    if "route_result" in st.session_state:
        r = st.session_state["route_result"]
        route_city_key = r["route_city_key"]
        graph = load_city_graph(route_city_key)  # cheap: already cached by @st.cache_resource

        safe_route = r["safe_route"]
        safe_coords = r["safe_coords"]
        shortest_coords = r["shortest_coords"]
        start_lat, start_lon, start_name = r["start_lat"], r["start_lon"], r["start_name"]
        end_lat, end_lon, end_name = r["end_lat"], r["end_lon"], r["end_name"]

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Safe route distance", f"{r['safe_dist']:.2f} km")
        metric_col2.metric("Safe route risk score", f"{r['safe_risk']:.4f}")

        if shortest_coords is not None:
            metric_col3, metric_col4 = st.columns(2)
            metric_col3.metric("Shortest route distance", f"{r['short_dist']:.2f} km",
                                delta=f"{r['short_dist'] - r['safe_dist']:+.2f} km vs safe route")
            metric_col4.metric("Shortest route risk score", f"{r['short_risk']:.4f}",
                                delta=f"{r['short_risk'] - r['safe_risk']:+.4f} vs safe route", delta_color="inverse")
        else:
            st.info("The safest route and the shortest route are identical here.")

        nav_data = build_navigation_data(graph, safe_route)

        st.subheader("🚶 Walk-through")
        st.caption(
            "Drag the slider to move through the route step by step. Distance, "
            "time, and direction update as you go - shown using standard page "
            "elements (not an embedded animation), so this works reliably every time."
        )
        walker_index = st.slider(
            "Position along the route", min_value=0, max_value=len(safe_coords) - 1,
            key="walker_position",
        )

        city_map = build_route_folium_map(
            graph, route_city_key, start_lat, start_lon, start_name,
            end_lat, end_lon, end_name, safe_coords, safe_route, shortest_coords,
            walker_index=walker_index,
        )
        st_folium(city_map, width=None, height=520, returned_objects=[])

        cumulative_m = nav_data["cumulative_m"]
        segment_risk_categories = nav_data["segment_risk_categories"]
        steps = nav_data["steps"]

        def find_current_step(point_index):
            for i, step in enumerate(steps):
                if point_index < step["end_index"]:
                    return i
            return len(steps) - 1

        current_step_index = find_current_step(walker_index)
        current_instruction = steps[current_step_index]["instruction"]
        next_instruction = (
            steps[current_step_index + 1]["instruction"]
            if current_step_index + 1 < len(steps) else None
        )
        distance_left_m = nav_data["total_m"] - cumulative_m[walker_index]
        walking_speed_m_per_min = 5000 / 60  # ~5 km/h assumed walking pace
        time_left_min = distance_left_m / walking_speed_m_per_min

        LOOKAHEAD_M = 100
        upcoming_high_risk = False
        here = cumulative_m[walker_index]
        for seg in range(walker_index, len(segment_risk_categories)):
            if cumulative_m[seg] - here > LOOKAHEAD_M:
                break
            if segment_risk_categories[seg] == "high":
                upcoming_high_risk = True
                break

        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if distance_left_m >= 1000:
                st.metric("Distance left", f"{distance_left_m / 1000:.2f} km")
            else:
                st.metric("Distance left", f"{distance_left_m:.0f} m")
        with nav_col2:
            st.metric("Time left (est.)", f"~{max(0, round(time_left_min))} min")

        if walker_index >= len(safe_coords) - 1:
            st.success("🏁 You have arrived!")
        else:
            info_text = f"**Direction:** {current_instruction}"
            if next_instruction:
                info_text += f"\n\n*Then:* {next_instruction}"
            st.info(info_text)

        if upcoming_high_risk:
            st.warning("⚠️ Caution: higher-risk street ahead")

        directions = build_turn_by_turn_directions(graph, safe_route)
        st.subheader(f"📋 Turn-by-turn directions ({start_name} → {end_name})")
        for step_number, step in enumerate(directions, start=1):
            st.write(f"**{step_number}.** {step['instruction']} — **{step['distance_m']:.0f} m**")
        st.write(f"**{len(directions) + 1}.** 🏁 Arrive at **{end_name}**")


# -------------------------------------------------------------------------------
# TAB 4 — LIVE RISK PREDICTION
# -------------------------------------------------------------------------------
with tab4:
    st.header("Live Risk Prediction")
    st.caption(
        "This applies only to the four England/Wales cities — Edinburgh and "
        "Glasgow are excluded, as their data has no street-level coordinates."
    )

    if prediction_assets is None:
        st.warning(
            f"Could not find the saved model files in `{SAVED_MODEL_FOLDER}/`. "
            f"Run `step35_save_final_model.py` once first."
        )
    else:
        model, scaler, target_encoder, column_order = prediction_assets

        season_options = sorted([c.replace("season_", "") for c in column_order if c.startswith("season_")])
        category_options = sorted([c.replace("most_common_category_", "") for c in column_order
                                    if c.startswith("most_common_category_")])
        city_options = sorted([c.replace("city_", "") for c in column_order if c.startswith("city_")])

        # Maps a calendar month straight to its meteorological season - this
        # replaces the old separate "Season" dropdown, which let someone
        # pick a contradictory combination (e.g. Season="Autumn" with
        # Month=6/June, which is actually summer). Deriving season directly
        # from the chosen month makes an inconsistent input impossible.
        # Matched case-insensitively against season_options so this lines up
        # with however the model's own training data actually capitalized
        # season names (e.g. "Winter" vs "winter"), rather than assuming.
        MONTH_TO_SEASON_KEY = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn",
        }
        season_lookup = {s.lower(): s for s in season_options}  # e.g. "winter" -> "Winter" or "winter"

        st.subheader("Enter scenario details")
        col1, col2, col3 = st.columns(3)
        with col1:
            input_city = st.selectbox("City", city_options)
            input_year = st.number_input("Year", min_value=2020, max_value=2030, value=2024)
        with col2:
            month_names = ["January", "February", "March", "April", "May", "June",
                            "July", "August", "September", "October", "November", "December"]
            input_month = st.selectbox("Month", options=list(range(1, 13)),
                                        format_func=lambda m: month_names[m - 1], index=5)
            input_season = season_lookup[MONTH_TO_SEASON_KEY[input_month]]
            st.caption(f"Season: **{input_season}** (derived automatically from the month)")
        with col3:
            input_category = st.selectbox("Dominant crime category nearby", category_options)

        st.subheader("Location detail")
        st.caption(
            "Pick a real place in the chosen city — the location statistics the model "
            "needs (distance to nearest attractions, how many are nearby) are calculated "
            "automatically from your actual POI data, rather than asking you to guess "
            "GIS numbers you'd have no way to know."
        )

        pois_df_predict = load_city_pois(input_city)
        landmark_choices_predict = sorted(pois_df_predict["name"].dropna().unique().tolist())
        selected_landmark = st.selectbox("Landmark / area", landmark_choices_predict, key="predict_landmark")

        # Compute the real distance/density features from the actual POI
        # data, using the chosen landmark as the reference point - this
        # replaces manual entry of avg/min distance and POI counts.
        landmark_row = pois_df_predict[pois_df_predict["name"] == selected_landmark].iloc[0]
        landmark_lat, landmark_lon = landmark_row["poi_lat"], landmark_row["poi_lon"]

        other_pois = pois_df_predict[pois_df_predict["name"] != selected_landmark]
        distances_km = haversine_km(landmark_lat, landmark_lon,
                                     other_pois["poi_lat"].values, other_pois["poi_lon"].values)

        avg_dist = float(distances_km.mean())
        min_dist = float(distances_km.min())
        poi_500m = int((distances_km <= 0.5).sum())
        poi_1km = int((distances_km <= 1.0).sum())

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Avg distance to other POIs", f"{avg_dist:.2f} km")
        metric_col2.metric("Nearest POI", f"{min_dist:.2f} km")
        metric_col3.metric("POIs within 500m", poi_500m)
        metric_col4.metric("POIs within 1km", poi_1km)

        with st.expander("🔧 Advanced: manually override these numbers"):
            st.caption("For testing hypothetical scenarios rather than a real landmark.")
            adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
            with adv_col1:
                avg_dist = st.number_input("Avg distance to POI (km)", min_value=0.0, value=avg_dist, step=0.05)
            with adv_col2:
                min_dist = st.number_input("Min distance to POI (km)", min_value=0.0, value=min_dist, step=0.05)
            with adv_col3:
                poi_500m = st.number_input("POIs within 500m", min_value=0, value=poi_500m)
            with adv_col4:
                poi_1km = st.number_input("POIs within 1km", min_value=0, value=poi_1km)

        if st.button("Predict risk level", type="primary"):
            raw_input = pd.DataFrame([{
                "avg_distance_to_poi_km": avg_dist, "min_distance_to_poi_km": min_dist,
                "poi_count_within_500m": poi_500m, "poi_count_within_1km": poi_1km,
                "year": input_year, "month_number": input_month,
                "season": input_season, "most_common_category": input_category, "city": input_city,
            }])

            encoded_input = pd.get_dummies(raw_input, columns=["season", "most_common_category", "city"],
                                            drop_first=False)
            encoded_input = encoded_input.reindex(columns=column_order, fill_value=0)

            numeric_columns = ["avg_distance_to_poi_km", "min_distance_to_poi_km",
                                "poi_count_within_500m", "poi_count_within_1km", "year", "month_number"]
            encoded_input[numeric_columns] = scaler.transform(encoded_input[numeric_columns])

            prediction_encoded = model.predict(encoded_input)[0]
            prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]
            probabilities = model.predict_proba(encoded_input)[0]

            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            st.subheader(f"{risk_emoji.get(prediction_label, '')} Predicted risk level: **{prediction_label.upper()}**")

            prob_df = pd.DataFrame({
                "risk_label": target_encoder.classes_, "probability": probabilities,
            }).sort_values("probability", ascending=False)

            fig_prob = px.bar(prob_df, x="risk_label", y="probability", color="risk_label",
                               color_discrete_map=RISK_COLORS,
                               title="Prediction confidence by class", range_y=[0, 1],
                               template=PLOTLY_TEMPLATE)
            fig_prob.update_layout(showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)


# -------------------------------------------------------------------------------
# TAB 5 — CITY RISK HEATMAP  (NEW)
# -------------------------------------------------------------------------------
with tab5:
    st.header("City Risk Heatmap")
    st.caption(
        "A density heatmap built directly from real crime coordinates — "
        "brighter/redder areas had more recorded crimes. Only available for "
        "England/Wales cities, since Police Scotland's data has no individual "
        "crime coordinates to plot."
    )

    heatmap_city_display = st.selectbox("Choose a city", ENGLAND_CITIES, key="heatmap_city")
    heatmap_city_key = heatmap_city_display.lower()

    try:
        crimes_df = load_city_crimes(heatmap_city_key)

        col1, col2 = st.columns(2)
        with col1:
            year_options = ["All years"] + sorted(crimes_df["year"].dropna().unique().tolist(), reverse=True)
            selected_year = st.selectbox("Year", year_options, key="heatmap_year")
        with col2:
            season_options_hm = ["All seasons"] + sorted(crimes_df["season"].dropna().unique().tolist())
            selected_season = st.selectbox("Season", season_options_hm, key="heatmap_season")

        # New: visual tuning controls, tucked into an expander so they don't
        # clutter the page for anyone who just wants to look at the map.
        with st.expander("🎨 Adjust heatmap appearance"):
            tune_col1, tune_col2 = st.columns(2)
            with tune_col1:
                heat_radius = st.slider("Point radius (spread of each crime)", min_value=5, max_value=30, value=14)
            with tune_col2:
                heat_blur = st.slider("Blur (smoothness)", min_value=5, max_value=30, value=18)

        filtered_crimes = crimes_df.copy()
        if selected_year != "All years":
            filtered_crimes = filtered_crimes[filtered_crimes["year"] == selected_year]
        if selected_season != "All seasons":
            filtered_crimes = filtered_crimes[filtered_crimes["season"] == selected_season]

        st.metric("Crimes shown on map", f"{len(filtered_crimes):,}")

        show_light_basemap = st.checkbox(
            "Show light base map (clearer street/area names)", value=False,
            help="The dark map style bakes its labels into the map image itself, so the heat "
                 "overlay can visually cover them in hotspot areas. This switches to a lighter "
                 "style where labels stay legible underneath the heatmap."
        )

        if filtered_crimes.empty:
            st.warning("No crimes match this filter combination.")
        else:
            @st.cache_resource
            def build_heatmap(city_key, year_key, season_key, radius, blur, light_basemap, heat_points_tuple):
                """Builds the full heatmap, cached and keyed on everything
                that actually determines its content. Without this, the
                whole heatmap (crime points, gradient, zoom-compensation
                script, marker, Fullscreen control) was being rebuilt from
                scratch on every Streamlit rerun - including reruns
                triggered by completely unrelated interactions elsewhere
                in the app, such as the Tab 3 walk-through slider."""
                base_tile = "cartodbpositron" if light_basemap else "cartodbdark_matter"
                heat_map = folium.Map(location=CITY_CENTRES[city_key], zoom_start=13, tiles=base_tile)

                heatmap_gradient = {
                    "0.2": "#313695",
                    "0.4": "#4575b4",
                    "0.6": "#74add1",
                    "0.75": "#fee090",
                    "0.9": "#f46d43",
                    "1.0": "#d73027",
                }

                heat_layer = HeatMap(
                    list(heat_points_tuple),
                    radius=radius,
                    blur=blur,
                    min_opacity=0.35,
                    gradient=heatmap_gradient,
                )
                heat_layer.add_to(heat_map)

                zoom_compensation_script = f"""
                <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    setTimeout(function() {{
                        var mapObj = {heat_map.get_name()};
                        var heatLayer = {heat_layer.get_name()};
                        var baseZoom = mapObj.getZoom();
                        var baseRadius = {radius};
                        var baseBlur = {blur};
                        mapObj.on('zoomend', function() {{
                            var zoomDiff = mapObj.getZoom() - baseZoom;
                            var scaleFactor = Math.pow(1.3, zoomDiff);
                            heatLayer.setOptions({{
                                radius: Math.max(4, baseRadius * scaleFactor),
                                blur: Math.max(4, baseBlur * scaleFactor)
                            }});
                        }});

                        setTimeout(function() {{ mapObj.invalidateSize(); }}, 100);
                        setTimeout(function() {{ mapObj.invalidateSize(); }}, 500);
                    }}, 500);
                }});
                </script>
                """
                heat_map.get_root().html.add_child(folium.Element(zoom_compensation_script))

                folium.Marker(
                    CITY_CENTRES[city_key],
                    popup=f"{city_key.title()} city centre",
                    icon=folium.Icon(color="white", icon="star", icon_color="black"),
                ).add_to(heat_map)

                Fullscreen(position="topright").add_to(heat_map)

                return heat_map

            heat_points = filtered_crimes[["Latitude", "Longitude"]].dropna().values.tolist()
            heat_map = build_heatmap(
                heatmap_city_key, str(selected_year), str(selected_season),
                heat_radius, heat_blur, show_light_basemap,
                tuple(map(tuple, heat_points)),
            )

            components.html(heat_map._repr_html_(), height=570)

            # folium's HeatMap has no built-in legend, so this small HTML/CSS
            # gradient bar recreates one - it uses the exact same colour stops
            # as heatmap_gradient above, so it's an accurate visual key.
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 10px; margin-top: -10px;">
                    <span style="font-size: 13px; color: #555;">Fewer crimes</span>
                    <div style="flex-grow: 1; height: 14px; border-radius: 4px;
                                background: linear-gradient(to right,
                                    #313695, #4575b4, #74add1, #fee090, #f46d43, #d73027);">
                    </div>
                    <span style="font-size: 13px; color: #555;">More crimes</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except FileNotFoundError:
        st.warning(
            f"Could not find `{heatmap_city_key}_crimes_cleaned.csv` in this folder. "
            f"Make sure your Phase 2 cleaned crime CSVs are in the same folder as this dashboard."
        )


# -------------------------------------------------------------------------------
# TAB 6 — RISK TREND FORECAST  (NEW)
# -------------------------------------------------------------------------------
with tab6:
    st.header("Risk Trend Forecast")
    st.caption(
        "Projects each city's yearly average TVS forward using a simple linear "
        "trend fit. A deliberately transparent, explainable method rather than "
        "an LSTM/Prophet model — with only a few years of yearly data per city, "
        "a heavier model would have too little data to learn a genuine pattern "
        "from, and would be far harder to justify or explain at viva."
    )

    if detailed_tvs_df is None:
        st.warning(f"Could not find `{DETAILED_TVS_PATH}`.")
    else:
        forecast_city = st.selectbox("Choose a city", ALL_CITIES, key="forecast_city")
        years_ahead = st.slider("Years to forecast ahead", min_value=1, max_value=5, value=3)

        city_yearly = (
            detailed_tvs_df[detailed_tvs_df["city"] == forecast_city]
            .groupby("year")["TVS"].mean().reset_index().sort_values("year")
        )

        if len(city_yearly) < 2:
            st.warning(f"Not enough yearly data points for {forecast_city} to fit a trend line.")
        else:
            # --- Fit a simple straight-line trend: TVS = slope * year + intercept ---
            # np.polyfit(x, y, 1) finds the best-fit STRAIGHT LINE through the
            # points - "1" means degree-1 (linear), the simplest and most
            # explainable choice for a small number of data points.
            years = city_yearly["year"].values
            tvs_values = city_yearly["TVS"].values

            slope, intercept = np.polyfit(years, tvs_values, 1)

            # R-squared measures how much of the actual year-to-year variation
            # the straight line explains (1.0 = perfect fit, 0.0 = no
            # explanatory power at all). With few data points and a
            # zigzagging pattern, a line can have a clear positive/negative
            # slope while still explaining very little of what's actually
            # happening - R-squared makes that visible instead of hiding it.
            correlation = np.corrcoef(years, tvs_values)[0, 1]
            r_squared = correlation ** 2

            last_year = int(years.max())
            last_actual_tvs = tvs_values[-1]
            future_years = np.arange(last_year + 1, last_year + 1 + years_ahead)
            # Anchor the forecast to the REAL last data point, then extend
            # forward using the slope - rather than using the fitted line's
            # own value for future years. A least-squares line minimizes
            # error across ALL points, so it doesn't necessarily pass
            # exactly through the last actual value; projecting from the
            # line itself can produce a next-year forecast that's LOWER
            # than this year's real value even when the slope is positive,
            # which reads as a contradiction. Anchoring to the last real
            # point guarantees the forecast always continues smoothly from
            # where the real data left off.
            future_tvs = last_actual_tvs + slope * (future_years - last_year)
            # TVS is designed to sit roughly in a 0-1 range, so clip the
            # forecast to that range - a straight line can otherwise
            # predict physically meaningless values if projected far enough.
            future_tvs = np.clip(future_tvs, 0, 1)

            fig_forecast = go.Figure()

            fig_forecast.add_trace(go.Scatter(
                x=years, y=tvs_values, mode="lines+markers", name="Historical TVS",
                line=dict(color=ACCENT_COLOR, width=3), marker=dict(size=10),
            ))
            fig_forecast.add_trace(go.Scatter(
                x=future_years, y=future_tvs, mode="lines+markers", name="Forecast TVS",
                line=dict(color="#e74c3c", width=3, dash="dash"), marker=dict(size=10, symbol="diamond"),
            ))

            fig_forecast.update_layout(
                title=f"{forecast_city}: TVS Trend and {years_ahead}-Year Forecast",
                xaxis_title="Year", yaxis_title="Tourist Vulnerability Score",
                template=PLOTLY_TEMPLATE, hovermode="x unified",
            )
            # Same fix as the Tab 2 yearly chart - force whole-year ticks only.
            fig_forecast.update_xaxes(dtick=1, tickformat="d")
            st.plotly_chart(fig_forecast, use_container_width=True)

            trend_direction = "increasing 📈" if slope > 0 else "decreasing 📉" if slope < 0 else "flat ➡️"
            st.info(
                f"**Trend direction:** {forecast_city}'s TVS is {trend_direction} over time "
                f"(slope = {slope:+.4f} per year). By {future_years[-1]}, the projected TVS "
                f"is **{future_tvs[-1]:.3f}**, compared to **{tvs_values[-1]:.3f}** in {last_year}."
            )

            # Be upfront about how well the straight line actually describes
            # the real pattern - with only a handful of years, a positive or
            # negative slope can exist even when the year-to-year data
            # zigzags rather than trending smoothly, so R-squared makes that
            # visible rather than letting the "increasing/decreasing" label
            # imply more confidence than the fit actually supports.
            if r_squared < 0.5:
                st.warning(
                    f"⚠️ **Fit quality is weak:** this line only explains **{r_squared:.0%}** of the "
                    f"year-to-year variation in {forecast_city}'s TVS (R² = {r_squared:.2f}). With only "
                    f"{len(years)} years of data, the actual pattern may zigzag up and down rather than "
                    f"follow a smooth trend — treat the direction above as a rough net average across "
                    f"the years shown, not a reliable description of what happens year to year."
                )
            else:
                st.caption(f"Fit quality: this line explains {r_squared:.0%} of the year-to-year variation (R² = {r_squared:.2f}).")

            st.caption(
                "Note: this is a straight-line projection based on a small number of "
                "yearly data points — treat it as an illustrative trend indicator, "
                "not a precise prediction."
            )