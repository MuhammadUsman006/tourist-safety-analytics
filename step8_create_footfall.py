import pandas as pd

# --- Visitor footfall data, manually compiled from official sources ---
# England figures: VisitBritain "Domestic Tourism, England Top Towns"
#   report, 2-year annual average across 2023-2024 (overnight trips),
#   rounded to 1 decimal place as published in the source table.
# Scotland figures: VisitScotland regional "Key statistics" pages,
#   single year 2024 (overnight trips - both domestic and international).
#
# NOTE FOR YOUR REPORT: England's figures are a 2023-2024 average, while
# Scotland's figures are 2024 alone. This is a minor inconsistency
# worth mentioning as a limitation - it happened because each nation's
# tourism board publishes its statistics differently, not through any
# choice we made.
footfall_data = {
    "city": ["london", "york", "liverpool", "birmingham", "edinburgh", "glasgow"],
    "annual_overnight_trips_millions": [15.5, 1.2, 1.7, 3.0, 5.05, 2.64],
    "data_year": ["2023-24 avg", "2023-24 avg", "2023-24 avg", "2023-24 avg", "2024", "2024"],
    "source": [
        "VisitBritain - Domestic Tourism England Top Towns",
        "VisitBritain - Domestic Tourism England Top Towns",
        "VisitBritain - Domestic Tourism England Top Towns",
        "VisitBritain - Domestic Tourism England Top Towns",
        "VisitScotland - Edinburgh and Lothians key statistics",
        "VisitScotland - Glasgow and Clyde Valley key statistics",
    ],
}

footfall_df = pd.DataFrame(footfall_data)

print(footfall_df)

# Save this as our final footfall file - Phase 1's last piece of data
footfall_df.to_csv("footfall_final.csv", index=False)
print("\nSaved -> footfall_final.csv")
print("\nPhase 1 data collection is now complete for all 6 cities.")