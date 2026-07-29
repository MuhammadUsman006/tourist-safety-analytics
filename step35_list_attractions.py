import pandas as pd

CITIES = ["london", "york", "liverpool", "birmingham"]

for city_name in CITIES:
    pois = pd.read_csv(f"{city_name}_poi_final.csv")

    # Only show "attraction" and "museum" types - excludes hotels,
    # guest houses, hostels, apartments, which aren't really
    # "destinations" tourists deliberately route between.
    attractions = pois[pois["poi_type"].isin(["attraction", "museum", "gallery", "viewpoint"])]
    named_attractions = attractions.dropna(subset=["name"])

    print(f"\n--- {city_name.title()}: {len(named_attractions)} named attractions/museums/galleries ---")
    for name in named_attractions["name"].unique()[:20]:
        print(f"  - {name}")
