import pandas as pd

# --- Severity weights for England/Wales crime categories ---
# Scale: 1 (least severe) to 10 (most severe), adapted from the general
# approach used in academic crime-risk-scoring literature (e.g. severity
# scales that weight violent/weapon-related crimes highest, and minor
# public-order offences lowest).
#
# IMPORTANT: this is a JUDGEMENT CALL, not an objective fact - you should
# briefly justify this reasoning in your Methodology section. The logic
# used here:
#   - Crimes involving violence, weapons, or direct physical threat to a
#     person score highest (8-9), since these pose the most serious risk
#     to a tourist's physical safety.
#   - Theft-from-the-person (e.g. pickpocketing, phone snatching) scores
#     high (7) specifically because it's the crime type most directly
#     and commonly experienced BY TOURISTS, even though it's less severe
#     than violence in a legal sense.
#   - Property crimes not involving a person directly present (burglary,
#     vehicle crime, criminal damage) score mid-range (4-6).
#   - Non-violent, low-harm public nuisance crimes score lowest (1-3).
ENGLAND_SEVERITY_WEIGHTS = {
    "violence-and-sexual-offences": 9,
    "possession-of-weapons": 8,
    "robbery": 8,
    "theft-from-the-person": 7,
    "burglary": 6,
    "vehicle-crime": 5,
    "criminal-damage-and-arson": 4,
    "other-theft": 4,
    "drugs": 3,
    "public-order": 3,
    "bicycle-theft": 3,
    "shoplifting": 2,
    "anti-social-behaviour": 1,
    "other-crime": 3,
}


def check_coverage(city_name):
    """
    Loads a city's model table and checks whether every crime category
    actually appearing in the real data has a severity weight defined
    for it. This matters because if even one category is missing from
    our dictionary, that row would get an empty/missing severity score
    later - silently breaking the TVS calculation without an obvious
    error message.
    """
    df = pd.read_csv(f"{city_name}_crimes_with_distance.csv")
    actual_categories = set(df["category"].unique())
    defined_categories = set(ENGLAND_SEVERITY_WEIGHTS.keys())

    missing = actual_categories - defined_categories
    unused = defined_categories - actual_categories

    print(f"\n--- {city_name.title()} ---")
    print(f"Categories in real data: {len(actual_categories)}")

    if missing:
        print(f"MISSING from our severity dictionary (needs fixing): {missing}")
    else:
        print("All categories in this city's data have a severity weight defined. Good.")

    if unused:
        print(f"Defined but not present in this city's data (harmless, just unused): {unused}")


for city_name in ["london", "york", "liverpool", "birmingham"]:
    check_coverage(city_name)

print("\nCoverage check complete.")