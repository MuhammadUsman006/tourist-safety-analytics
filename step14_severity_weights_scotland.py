import pandas as pd

# --- Severity weights for Scotland crime categories ---
# Same 1-10 scale and reasoning approach as England's dictionary, but
# mapped onto Scotland's own category names, since Police Scotland uses
# a different classification system entirely (Groups 1-8) rather than
# England's "Crime type" labels.
#
# because England and Scotland classify crimes
# differently, these two severity dictionaries are NOT directly
# equivalent category-by-category - e.g. Scotland separates "common
# assault" from "serious assault," while England groups all violence
# into one broad "violence-and-sexual-offences" category. The SAME
# underlying severity logic (violence/weapons highest, minor public
# order lowest) was applied independently to each nation's own
# categories, rather than trying to force an artificial one-to-one
# match between two genuinely different legal classification systems.
SCOTLAND_SEVERITY_WEIGHTS = {
    # Group 1: Non-sexual crimes of violence
    "crimes:-group-1:-murder-and-culpable-homicide": 10,
    "crimes:-group-1:-serious-assault-and-attempted-murder": 9,
    "crimes:-group-1:-robbery": 8,
    "crimes:-group-1:-domestic-abuse-(scotland)-act-2018": 8,
    "crimes:-group-1:-common-assault": 6,
    "crimes:-group-1:-death-by-dangerous-driving": 8,
    "crimes:-group-1:-other-non-sexual-violence": 6,

    # Group 2: Sexual crimes
    "crimes:-group-2:-rape-&-attempted-rape": 10,
    "crimes:-group-2:-sexual-assault": 9,
    "crimes:-group-2:-indecent-photos-of-children": 9,
    "crimes:-group-2:-causing-to-view-sexual-activity-or-images": 8,
    "crimes:-group-2:-communicating-indecently": 6,
    "crimes:-group-2:-threatening-to-or-disclosing-intimate-images": 7,
    "crimes:-group-2:-crimes-associated-with-prostitution": 4,
    "crimes:-group-2:-other-sexual-crimes": 7,

    # Group 3: Crimes of dishonesty (property/theft crimes)
    "crimes:-group-3:-housebreaking": 6,
    "crimes:-group-3:-theft-by-opening-lockfast-places": 5,
    "crimes:-group-3:-theft-from-a-motor-vehicle": 5,
    "crimes:-group-3:-theft-of-a-motor-vehicle": 5,
    "crimes:-group-3:-shoplifting": 2,
    "crimes:-group-3:-other-theft": 4,
    "crimes:-group-3:-fraud": 3,
    "crimes:-group-3:-other-dishonesty": 3,

    # Group 4: Fire-raising, vandalism, reckless behaviour
    "crimes:-group-4:-fire-raising": 7,
    "crimes:-group-4:-vandalism": 3,
    "crimes:-group-4:-reckless-conduct": 4,

    # Group 5: Crimes against society (drugs, weapons, public justice)
    "crimes:-group-5:-crimes-against-public-justice": 4,
    "crimes:-group-5:-drugs---supply": 5,
    "crimes:-group-5:-drugs---possession": 3,
    "crimes:-group-5:-weapons-possession-(used)": 9,
    "crimes:-group-5:-weapons-possession-(not-used)": 6,
    "crimes:-group-5:-other-crimes-against-society": 3,

    # Miscellaneous crime
    "crimes:-coronavirus-restrictions": 1,

    # Group 6: Antisocial offences
    "offences:-group-6:-threatening-and-abusive-behaviour": 5,
    "offences:-group-6:-drunkenness-and-other-disorderly-conduct": 2,
    "offences:-group-6:-urinating-etc.": 1,
    "offences:-group-6:-hate-aggravated-conduct": 6,

    # Group 7: Miscellaneous offences
    "offences:-group-7:-community-and-public-order-offences": 2,
    "offences:-group-7:-environmental-offences": 2,
    "offences:-group-7:-licensing-offences": 2,
    "offences:-group-7:-wildlife-offences": 2,
    "offences:-group-7:-other-misc.-offences": 2,

    # Group 8: Road traffic offences
    # NOTE: these score LOW for tourist-safety purposes specifically -
    # a tourist walking around a city centre is not meaningfully at risk
    # from someone else's speeding ticket or seatbelt offence the way
    # they would be from a robbery or assault. This is a deliberate
    # methodological decision, worth stating explicitly in your report.
    "offences:-group-8:-dangerous-and-careless-driving": 4,
    "offences:-group-8:-driving-under-the-influence": 4,
    "offences:-group-8:-speeding": 1,
    "offences:-group-8:-seat-belt-offences": 1,
    "offences:-group-8:-mobile-phone-offences": 1,
    "offences:-group-8:-vehicle-defect-offences": 1,
    "offences:-group-8:-unlawful-use-of-vehicle": 2,
    "offences:-group-8:-other-road-traffic-offences": 2,
}


def check_coverage(city_name):
    df = pd.read_csv(f"{city_name}_yearly_model_table.csv")
    actual_categories = set(df["category"].unique())
    defined_categories = set(SCOTLAND_SEVERITY_WEIGHTS.keys())

    missing = actual_categories - defined_categories
    unused = defined_categories - actual_categories

    print(f"\n--- {city_name.title()} ---")
    print(f"Categories in real data: {len(actual_categories)}")

    if missing:
        print(f"MISSING from our severity dictionary (needs fixing): {missing}")
    else:
        print("All categories in this city's data have a severity weight defined. Good.")

    if unused:
        print(f"Defined but not present in this city's data: {unused}")


for city_name in ["edinburgh", "glasgow"]:
    check_coverage(city_name)

print("\nCoverage check complete.")