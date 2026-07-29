import pandas as pd

ENGLAND_CITIES = ["london", "york", "liverpool", "birmingham"]

LOW_MEDIUM_CUTOFF = 0.33
MEDIUM_HIGH_CUTOFF = 0.66


def build_district_month_table(city_name):
    print(f"\n--- Processing {city_name.title()} ---")

    # Load the individual-crime-level file from Step 1 (has distance
    # feature, but not risk labels yet - we build those correctly here)
    crimes = pd.read_csv(f"{city_name}_crimes_with_distance.csv")
    crimes = crimes.dropna(subset=["LSOA code"])
    print(f"Loaded {len(crimes):,} individual crimes")

    # --- Build ONE ROW PER DISTRICT-MONTH, with SUMMARY features ---
    # groupby() splits all crimes into their LSOA+year+month group, then
    # .agg() calculates several different summary statistics for each
    # group all at once - this is the heart of fixing the imbalance
    # problem, since every group becomes exactly ONE row, regardless of
    # whether that district had 2 crimes or 500 crimes that month.
    district_month = (
        crimes.groupby(["LSOA code", "LSOA name", "year", "month_number", "season"])
        .agg(
            crime_count=("category", "count"),
            avg_distance_to_poi_km=("distance_to_nearest_poi_km", "mean"),
            min_distance_to_poi_km=("distance_to_nearest_poi_km", "min"),
            # .agg with a custom function: finds the MOST COMMON crime
            # category in this district-month (e.g. "mostly shoplifting"
            # vs "mostly violent-crime") - useful context for later.
            most_common_category=("category", lambda x: x.mode().iloc[0] if not x.mode().empty else None),
        )
        .reset_index()
    )

    print(f"Created {len(district_month)} district-month rows (was {len(crimes):,} individual crime rows)")

    # --- Assign risk labels at THIS level (one label per row, not per crime) ---
    # This is the fix: now that each row already represents one
    # district-month, the percentile split and the final label
    # distribution will match exactly - no more imbalance introduced
    # by busy areas contributing more individual crime rows than quiet
    # ones.
    low_cutoff = district_month["crime_count"].quantile(LOW_MEDIUM_CUTOFF)
    high_cutoff = district_month["crime_count"].quantile(MEDIUM_HIGH_CUTOFF)

    def label_risk(count):
        if count <= low_cutoff:
            return "low"
        elif count <= high_cutoff:
            return "medium"
        else:
            return "high"

    district_month["risk_label"] = district_month["crime_count"].apply(label_risk)

    print(f"Low/Medium cutoff: {low_cutoff:.2f} crimes | Medium/High cutoff: {high_cutoff:.2f} crimes")
    print("Risk label distribution (this is what the model will actually train on):")
    print(district_month["risk_label"].value_counts())
    print("As percentages:")
    print((district_month["risk_label"].value_counts(normalize=True) * 100).round(1))

    district_month["city"] = city_name

    output_filename = f"{city_name}_district_month_model_table.csv"
    district_month.to_csv(output_filename, index=False)
    print(f"Saved -> {output_filename}")


for city_name in ENGLAND_CITIES:
    build_district_month_table(city_name)

print("\nModel-ready district-month tables built for all England/Wales cities.")