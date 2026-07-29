import pandas as pd

SCOTLAND_CITIES = ["edinburgh", "glasgow"]


def build_scotland_table(city_name):
    print(f"\n--- Processing {city_name.title()} ---")

    # Load the cleaned Scotland file from Phase 1 - remember, this is
    # ALREADY at "one row per category per year" level (yearly totals),
    # since Police Scotland doesn't publish individual crime records
    # with coordinates like England/Wales does.
    crimes = pd.read_csv(f"{city_name}_crimes_cleaned.csv")
    print(f"Loaded {len(crimes):,} category-year rows")
    print(f"Years covered: {crimes['year'].min()} to {crimes['year'].max()}")

    # --- Build ONE ROW PER YEAR, summarising across all categories ---
    # Unlike England (which we grouped down to district-month), Scotland
    # only has city-wide yearly data to begin with - so "one row per
    # year" is the finest level of detail actually available to us here.
    # This is a genuine structural difference between the two data
    # sources, not something we're choosing arbitrarily.
    yearly_totals = (
        crimes.groupby("year")
        .agg(
            total_crimes=("crime_count", "sum"),
        )
        .reset_index()
    )

    print("Total crimes per year:")
    print(yearly_totals)

    # --- Also keep the category-level breakdown, unchanged ---
    # We still need this later in Phase 3, when we apply severity
    # weights to each specific crime category (e.g. "robbery" should
    # count for more than "shoplifting" in the TVS formula) - so we
    # don't want to lose that detail by only keeping yearly totals.
    crimes["city"] = city_name

    output_filename = f"{city_name}_yearly_model_table.csv"
    crimes.to_csv(output_filename, index=False)
    print(f"Saved category-level detail -> {output_filename}")

    yearly_output_filename = f"{city_name}_yearly_totals.csv"
    yearly_totals["city"] = city_name
    yearly_totals.to_csv(yearly_output_filename, index=False)
    print(f"Saved yearly totals -> {yearly_output_filename}")


for city_name in SCOTLAND_CITIES:
    build_scotland_table(city_name)

print("\nScotland model tables built for both cities.")