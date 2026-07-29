import pandas as pd

# --- LOAD the raw Scotland file ---
scotland = pd.read_csv("raw_data_scotland/scotland_crimes_raw.csv")

print(f"Starting with {len(scotland):,} total rows (all of Scotland, all areas, all years)")

# Quick look at what we actually have before doing anything else
print()
print("First 5 rows:")
print(scotland.head())
print()
print("Unique area names (first 20):")
print(scotland["FeatureName"].unique()[:20])
print()
print("Year range (DateCode):")
print(scotland["DateCode"].unique()[:10], "...")

# Show the FULL list of area names, not just the first 20, so we can
# find the exact spelling used for Edinburgh and Glasgow
print()
print("ALL unique area names in this file:")
for name in sorted(scotland["FeatureName"].unique()):
    print(name)

import pandas as pd

# --- LOAD the raw Scotland file ---
scotland = pd.read_csv("raw_data_scotland/scotland_crimes_raw.csv")
print(f"Starting with {len(scotland):,} total rows (all of Scotland, all areas, all years)")

# --- STEP 1: keep only Edinburgh and Glasgow ---
# Now that we've confirmed the EXACT spelling used in this file
# ("City of Edinburgh" and "Glasgow City"), we use .isin() to keep only
# rows where FeatureName exactly matches one of these two strings.
# This is more precise than a "contains" search, since we know the
# exact real values rather than guessing.
TARGET_AREAS = {
    "City of Edinburgh": "edinburgh",
    "Glasgow City": "glasgow",
}

scotland = scotland[scotland["FeatureName"].isin(TARGET_AREAS.keys())].copy()
print(f"After keeping only Edinburgh & Glasgow: {len(scotland):,} rows")

# Map the official area name to our simpler city label used throughout
# the rest of the project ("edinburgh" / "glasgow"), matching the style
# used for the English cities.
scotland["city"] = scotland["FeatureName"].map(TARGET_AREAS)

# --- STEP 2: clean up the crime category text ---
# Matches the same lowercase-with-dashes style we used for England
# ("Violence and sexual offences" -> "violence-and-sexual-offences"),
# so both nations' category labels are formatted consistently - even
# though the underlying category systems differ between the two
# countries (a genuine, worth-noting limitation - Scotland classifies
# crimes differently to England & Wales).
scotland["category"] = (
    scotland["Crime or Offence"].str.strip().str.lower().str.replace(" ", "-")
)

# --- STEP 3: extract a usable year from the financial year format ---
# DateCode looks like "2023/2024" (April 2023 - March 2024 in the UK
# government's financial year convention). We take just the FIRST year
# for simplicity, so "2023/2024" becomes 2023.
# NOTE FOR YOUR REPORT: this means Scotland's "year 2023" data covers
# April 2023-March 2024, while England's "year 2023" data covers
# January-December 2023 (calendar year) - these aren't quite the same
# 12 months. Worth one sentence in your Methodology/Limitations section.
scotland["year"] = scotland["DateCode"].str.split("/").str[0].astype(int)

# --- STEP 4: rename Value to something clearer ---
# "Value" alone doesn't explain what's being counted - "crime_count"
# is self-explanatory to anyone reading your code or report later.
scotland = scotland.rename(columns={"Value": "crime_count"})

# --- STEP 5: keep only the columns we actually need ---
scotland_clean = scotland[["city", "year", "category", "crime_count"]]

# --- STEP 6: save Edinburgh and Glasgow to their own separate files ---
# Matches how England's 4 cities were saved individually - later scripts
# (city comparison tool, generalisation test) expect one file per city.
for city_name in ["edinburgh", "glasgow"]:
    city_df = scotland_clean[scotland_clean["city"] == city_name]
    output_filename = f"{city_name}_crimes_cleaned.csv"
    city_df.to_csv(output_filename, index=False)
    print(f"{city_name.title()}: {len(city_df):,} rows saved -> {output_filename}")

# --- Quick preview of what we ended up with ---
print()
print("Sample of cleaned Edinburgh data:")
print(scotland_clean[scotland_clean["city"] == "edinburgh"].head())
print()
print("Unique crime categories found (Scotland uses different category")
print("names than England - we'll need a separate severity mapping for")
print("these later in Phase 3):")
print(scotland_clean["category"].unique())

# --- STEP 7: remove summary/subtotal rows to avoid double-counting ---
# This data is structured hierarchically: "all-crimes" is a grand total,
# "all-group-1:-..." is a subtotal within it, and "crimes:-group-1:-
# robbery" is a specific detailed crime type within THAT subtotal.
# If we kept all of these together, we'd count the same real crimes
# multiple times (once as a detail, again as part of its subtotal,
# again as part of the grand total).
#
# The fix: keep ONLY rows that start with "crimes:-" or "offences:-"
# (the detailed, most granular level) and drop anything starting with
# "all-" (the summary/subtotal rows).
is_summary_row = scotland_clean["category"].str.startswith("all-")
scotland_detailed = scotland_clean[~is_summary_row].copy()

print()
print(f"Before removing summary rows: {len(scotland_clean):,} rows")
print(f"After removing summary rows (detailed crime types only): {len(scotland_detailed):,} rows")

# Re-save the cleaned files using ONLY the detailed rows, so every
# later script (severity weighting, TVS, modelling) works with correct,
# non-duplicated counts.
for city_name in ["edinburgh", "glasgow"]:
    city_df = scotland_detailed[scotland_detailed["city"] == city_name]
    output_filename = f"{city_name}_crimes_cleaned.csv"
    city_df.to_csv(output_filename, index=False)
    print(f"{city_name.title()}: {len(city_df):,} detailed rows saved -> {output_filename}")

print()
print("Detailed crime categories we'll actually use going forward:")
for cat in sorted(scotland_detailed["category"].unique()):
    print(" -", cat)

print("\nScotland processing complete.")