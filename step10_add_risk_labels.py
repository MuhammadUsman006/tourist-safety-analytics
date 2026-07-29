import pandas as pd

ENGLAND_CITIES = ["london", "york", "liverpool", "birmingham"]

# --- Percentile cutoffs ---
# Bottom 33% of district-month crime counts = "low" risk
# Next 33% (33rd to 66th percentile) = "medium" risk
# Top 33% (above 66th percentile) = "high" risk
LOW_MEDIUM_CUTOFF = 0.33
MEDIUM_HIGH_CUTOFF = 0.66


def add_risk_labels(city_name):
    print(f"\n--- Processing {city_name.title()} ---")

    # Load the file we created in Step 1 (already has the distance
    # feature added)
    crimes = pd.read_csv(f"{city_name}_crimes_with_distance.csv")
    print(f"Loaded {len(crimes):,} crimes")

    # Some crimes might have a missing LSOA code (rare, but data.police.uk
    # occasionally doesn't assign one). We can't group these properly,
    # so we drop them here rather than let them cause errors later.
    before_drop = len(crimes)
    crimes = crimes.dropna(subset=["LSOA code"])
    print(f"Dropped {before_drop - len(crimes)} rows with missing LSOA code")

    # --- STEP 1: count how many crimes happened in each LSOA, in each
    # specific year+month ---
    # We group by LSOA code AND year AND month_number together (not just
    # LSOA + month) - this matters because otherwise "March 2023" and
    # "March 2024" would get mixed together as if they were the same
    # time period, which would be wrong.
    group_counts = (
        crimes.groupby(["LSOA code", "year", "month_number"])
        .size()
        .reset_index(name="lsoa_month_crime_count")
    )

    print(f"Created {len(group_counts)} unique district-month groups")

    # --- STEP 2: work out this city's own percentile cutoff values ---
    # .quantile(0.33) finds the crime-count value below which 33% of
    # all district-month groups fall. Anything at or below this number
    # becomes "low" risk. Same idea for the 66th percentile.
    #
    # IMPORTANT NOTE FOR YOUR REPORT: these cutoffs are calculated
    # separately for EACH city, based on that city's own distribution of
    # crime counts. This means "low risk" in London and "low risk" in
    # York are relative to each city's own crime patterns, not on one
    # single shared national scale. This is worth mentioning as a
    # methodology detail - it will matter when we get to Phase 4's
    # geographic generalisation test (train on London, test on York/
    # Liverpool), which we'll address properly when we get there.
    low_cutoff_value = group_counts["lsoa_month_crime_count"].quantile(LOW_MEDIUM_CUTOFF)
    high_cutoff_value = group_counts["lsoa_month_crime_count"].quantile(MEDIUM_HIGH_CUTOFF)

    print(f"Low/Medium cutoff (33rd percentile): {low_cutoff_value:.2f} crimes")
    print(f"Medium/High cutoff (66th percentile): {high_cutoff_value:.2f} crimes")

    # --- STEP 3: assign the actual low/medium/high label ---
    def label_risk(count):
        if count <= low_cutoff_value:
            return "low"
        elif count <= high_cutoff_value:
            return "medium"
        else:
            return "high"

    group_counts["risk_label"] = group_counts["lsoa_month_crime_count"].apply(label_risk)

    print("Risk label distribution (number of district-month groups per label):")
    print(group_counts["risk_label"].value_counts())

    # --- STEP 4: bring the risk label back onto every individual crime row ---
    # Right now, risk_label exists only at the "one row per LSOA+month"
    # level. We need it attached to every SINGLE crime record, since
    # that's what our ML model will actually be trained on later
    # (Objective 4). merge() looks up each crime's LSOA+year+month and
    # copies across the matching risk_label.
    crimes = crimes.merge(
        group_counts[["LSOA code", "year", "month_number", "lsoa_month_crime_count", "risk_label"]],
        on=["LSOA code", "year", "month_number"],
        how="left",
    )

    print("\nFinal risk label distribution across all individual crimes:")
    print(crimes["risk_label"].value_counts())

    output_filename = f"{city_name}_crimes_with_risk_labels.csv"
    crimes.to_csv(output_filename, index=False)
    print(f"Saved -> {output_filename}")


for city_name in ENGLAND_CITIES:
    add_risk_labels(city_name)

print("\nRisk labeling complete for all England/Wales cities.")