import pandas as pd
import glob

# --- CITY CONFIGURATION ---
# Each city needs two things:
#   1. "force_filename_part" - the exact text that appears in the CSV
#      filenames data.police.uk generates for that police force
#      (e.g. "2024-03-metropolitan-street.csv" contains "metropolitan")
#   2. "bbox" - a bounding box (min_lat, min_lon, max_lat, max_lon)
#      marking out just the TOURIST-RELEVANT area of that city, not the
#      whole police force area (which is usually much bigger)
CITIES = {
    "london": {
        "force_filename_part": "metropolitan",
        "bbox": (51.47, -0.20, 51.55, -0.05),
    },
    "york": {
        "force_filename_part": "north-yorkshire",
        "bbox": (53.945, -1.11, 53.975, -1.05),
    },
    "liverpool": {
        "force_filename_part": "merseyside",
        "bbox": (53.39, -3.02, 53.42, -2.96),
    },
    "birmingham": {
        "force_filename_part": "west-midlands",
        "bbox": (52.46, -1.93, 52.51, -1.86),
    },
}


def get_season(month_number):
    """
    Takes a month number (1-12) and returns which season it falls in.
    We need this because your proposal's Objective 2 requires "season"
    as one of the model's input features, and Objective 5 needs it for
    the seasonal risk calendar later.
    """
    if month_number in [12, 1, 2]:
        return "winter"
    elif month_number in [3, 4, 5]:
        return "spring"
    elif month_number in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


# --- MAIN LOOP: process each city one at a time ---
for city_name, city_info in CITIES.items():
    print(f"\n--- Processing {city_name.title()} ---")

    pattern = f"raw_data_England/*/*-{city_info['force_filename_part']}-street.csv"
    file_list = glob.glob(pattern)
    print(f"Found {len(file_list)} files for {city_name}")

    if len(file_list) == 0:
        print(f"WARNING: no files found for {city_name} - check your download included this force.")
        continue

    all_data = [pd.read_csv(f) for f in file_list]
    crimes = pd.concat(all_data, ignore_index=True)
    print(f"Starting with {len(crimes):,} total records")

    # --- CLEANING STEP 1: remove crimes with no location ---
    crimes = crimes.dropna(subset=["Latitude", "Longitude"])

    # --- CLEANING STEP 2: keep only the tourist-relevant area ---
    min_lat, min_lon, max_lat, max_lon = city_info["bbox"]
    crimes = crimes[
        (crimes["Latitude"] >= min_lat) & (crimes["Latitude"] <= max_lat) &
        (crimes["Longitude"] >= min_lon) & (crimes["Longitude"] <= max_lon)
    ]
    print(f"After filtering to tourist-core bounding box: {len(crimes):,} records")

    # --- CLEANING STEP 3: turn the "Month" text into real date info ---
    crimes["Month"] = pd.to_datetime(crimes["Month"], format="%Y-%m")
    crimes["year"] = crimes["Month"].dt.year
    crimes["month_number"] = crimes["Month"].dt.month

    # --- CLEANING STEP 4: add the season column ---
    crimes["season"] = crimes["month_number"].apply(get_season)

    # --- CLEANING STEP 5: standardise the crime category text ---
    # Real data looks like "Violence and sexual offences" (spaces,
    # capital letters). This turns it into
    # "violence-and-sexual-offences" - consistent formatting that's
    # easier to match against our severity-weighting dictionary later.
    crimes["category"] = (
        crimes["Crime type"].str.strip().str.lower().str.replace(" ", "-")
    )

    # --- CLEANING STEP 6: tag every row with its city name ---
    # This matters later when we combine all 4 cities together for the
    # city comparison tool (Objective 5) - without this column, we'd
    # lose track of which row came from which city.
    crimes["city"] = city_name

    # --- SAVE the cleaned result ---
    output_filename = f"{city_name}_crimes_cleaned.csv"
    crimes.to_csv(output_filename, index=False)
    print(f"Saved -> {output_filename}")

print("\nAll cities processed.")