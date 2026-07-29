# Tourist Safety Analytics — COM748 MSc Final Project

An end-to-end pipeline that predicts and visualises crime risk for tourists
across six UK cities (London, York, Liverpool, Birmingham, Edinburgh,
Glasgow), combining a custom Tourist Vulnerability Score (TVS), a machine
learning risk classifier, a risk-weighted safe route planner, and an
interactive Streamlit dashboard.

## What this project does

Four core contributions:
1. **Tourist Vulnerability Score (TVS)** — a composite score per city/area
   combining crime frequency (40%), severity-weighted crime type (40%), and
   inverse visitor footfall (20%)
2. **A trained XGBoost classifier** (76.1% accuracy, 5-fold CV: ±0.6%)
   predicting low/medium/high risk from location and time features
3. **A safe route planner** using real OpenStreetMap street networks,
   finding routes that minimise cumulative crime risk rather than just
   distance
4. **An interactive dashboard** (Streamlit) tying all of the above together:
   city comparison, seasonal/yearly trend charts, a real choropleth map, a
   live route planner with turn-by-turn directions, live risk prediction,
   a crime density heatmap, and a trend forecast

## Important data limitation (read this first)

**England/Wales cities** (London, York, Liverpool, Birmingham) use
individual crime records with coordinates from data.police.uk.
**Scotland cities** (Edinburgh, Glasgow) use yearly category totals from
statistics.gov.scot, with **no individual crime coordinates**. As a
result, Edinburgh and Glasgow are fully included in the TVS score and city
comparison, but are **excluded** from the ML classifier, the route
planner, and any street-level mapping — there is no coordinate data for
those features to work with. This is documented on-screen throughout the
dashboard itself.

Manchester was originally proposed as a fourth England city, but Greater
Manchester Police has not submitted data to data.police.uk since ~2023;
York was substituted with supervisor approval.

## Project structure

```
├── README.md
├── requirements.txt
├── dashboard.py                          (Phase 6: the interactive dashboard)
├── main.py                                (initial project entry point)
│
├── Phase 1 — Data collection & cleaning
│   ├── step2_clean_data.py                (clean England crime data)
│   ├── step3_clean_scotland.py            (clean Scotland crime data)
│   ├── step4_load_poi.py                  (load POI data from OpenStreetMap)
│   ├── step5_clean_poi.py                 (clean POI data)
│   ├── step6_check_poi_types.py           (inspect POI type tags)
│   ├── step7_filter_poi_types.py          (filter POI to tourist-relevant whitelist)
│   └── step8_create_footfall.py           (build footfall dataset)
│
├── Phase 2 — Feature engineering
│   ├── step9_add_distance_feature.py      (distance-to-nearest-POI, England)
│   ├── step10_add_risk_labels.py          (low/medium/high risk labels)
│   ├── step11_build_model_table.py        (England model feature table)
│   └── step12_build_scotland_table.py     (Scotland aggregated table)
│
├── Phase 3 — Tourist Vulnerability Score (TVS)
│   ├── step13_severity_weights_england.py
│   ├── step14_severity_weights_scotland.py
│   ├── step15_calculate_tvs.py            (per-city normalised TVS)
│   └── step16_city_level_tvs.py           (globally normalised TVS)
│
├── Phase 4 — Machine learning models
│   ├── step17_prepare_model_data.py       (encode/scale features for ML)
│   ├── step18_train_logistic_regression.py
│   ├── step19_train_random_forest.py
│   ├── step20_train_xgboost.py
│   ├── step21_add_poi_density.py          (adds POI density features)
│   ├── step22_tune_xgboost.py
│   ├── step23_tune_random_forest.py
│   ├── step24_tune_logistic_regression.py
│   ├── step25_cross_validation.py
│   ├── step26_significance_testing.py
│   ├── step27_shap_explainability.py
│   ├── step28_generalisation_test.py      (London-trained model tested on York/Liverpool)
│   └── step29_stacking_ensemble.py
│
├── Phase 5 — Safe route planner
│   ├── step30_download_street_network.py  (OSMnx street networks per city)
│   ├── step31_add_risk_to_network.py      (attach TVS-based risk per street segment)
│   ├── step32_route_planner.py            (first working route planner)
│   ├── step33_route_planner_by_name.py    (route planner using landmark names)
│   ├── step34_complete_route_planner.py   (final merged route planner + validation)
│   └── step35_list_attractions.py         (helper: lists attractions per city)
│
├── Phase 6 — Dashboard support scripts
│   ├── step35_save_final_model.py         (trains + saves the final XGBoost model)
│   └── step36_get_city_boundaries.py      (fetches real city boundary polygons, incl. Glasgow's fixed query)
│
└── Diagnostic / one-off utility scripts (not part of the core pipeline)
    ├── check_files.py
    ├── check_york.py
    └── check_york_liverpool.py
```

> **Note on numbering:** `step35_list_attractions.py` (Phase 5) and
> `step35_save_final_model.py` (Phase 6) share the same step number by
> coincidence of how the project evolved — they are unrelated scripts
> with different purposes, kept under their original filenames here.

## Data files (not included in this repository)

The following files are required to run the pipeline but are excluded
from this repo (see `.gitignore`) due to size. Follow the reproduction
steps below to regenerate them:

- `{city}_crimes_cleaned.csv` — cleaned crime records per England city
- `{city}_poi_final.csv` — points of interest per city
- `{city}_district_month_model_table_v2.csv` — engineered features for ML
- `city_tvs_summary_corrected.csv` — globally-normalised TVS per city
- `all_cities_tvs_detailed.csv` — per-city normalised TVS, LSOA/month level
- `{city}_street_network_with_risk.graphml` — OSMnx street networks with
  risk scores attached (created in Phase 5)
- `city_boundaries.geojson` — real city boundary polygons (created by
  `step36_get_city_boundaries.py`)
- `saved_model/` — the trained XGBoost model + scaler + encoders (created
  by `step35_save_final_model.py`)

## How to reproduce this project from scratch

### Prerequisites

- Python 3.10+ (developed using PyCharm on Windows)
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

### Step-by-step

Run each phase's scripts in order (each depends on files created by the
previous ones):

1. **Phase 1 — Data collection & cleaning**: `step2` through `step8`
2. **Phase 2 — Feature engineering**: `step9` through `step12`
3. **Phase 3 — Tourist Vulnerability Score**: `step13` through `step16`
4. **Phase 4 — Train the ML models**: `step17` through `step29`
5. **Phase 5 — Build the safe route planner**:
   ```
   python step30_download_street_network.py
   python step31_add_risk_to_network.py
   python step34_complete_route_planner.py
   ```
   (`step32`/`step33` were earlier working versions superseded by `step34`,
   which merges and validates everything in one script)
6. **Phase 6a — Save the final model for the dashboard**:
   ```
   python step35_save_final_model.py
   ```
7. **Phase 6b — Fetch real city boundary polygons**:
   ```
   python step36_get_city_boundaries.py
   ```
   Glasgow needed a different Nominatim query
   (`"Glasgow City, United Kingdom"`) than the other 5 cities to resolve
   to a real boundary polygon rather than a point — this fix is already
   built into the `CITY_QUERIES` list in this script.
8. **Run the dashboard**:
   ```
   streamlit run dashboard.py
   ```

## Key results

- **TVS ranking** (globally normalised, higher = riskier): Liverpool
  (0.513) > Birmingham (0.507) > London (0.503) > York (0.331) > Glasgow
  (0.198) > Edinburgh (0.098)
- **Best classifier**: XGBoost, 76.1% accuracy (5-fold CV: 76.1% ± 0.6%),
  vs. Logistic Regression (57.3%) and Random Forest (72.5%) baselines
- **SHAP analysis**: distance-to-nearest-POI is the dominant predictive
  feature; "medium" risk is consistently the hardest class to classify
  (lower SHAP magnitudes across all features)
- **Geographic generalisation**: a London-only model tested on York (54.4%)
  and Liverpool (59.8%) — both above the 33% random baseline but well
  below in-sample accuracy, suggesting per-city models generalise better
  than a single universal model

## Known limitations

- Scotland's data has no street-level coordinates (see above)
- The TVS choropleth shades each city's *entire* official administrative
  boundary (from OpenStreetMap), which is typically larger than the
  "tourist core" area the crime data itself was filtered to
- The risk trend forecast uses a simple linear fit rather than a more
  complex time-series model (e.g. LSTM/Prophet), since only a few years of
  yearly data exist per city — a heavier model would have too little data
  to learn a genuine pattern from
- The 76% classifier accuracy, while a validated, defensible ceiling for
  this feature set, should not be treated as sufficient for real
  safety-critical decisions without further validation

## Author

Muhammad Usman Shoukat — MSc Computer Science & Technology, Ulster University, London Campus. 
Supervisor: Sir Nasir Iqbal.
