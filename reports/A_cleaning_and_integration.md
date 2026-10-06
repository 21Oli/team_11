# A — Data Cleaning & Integration Report

## Qiyas Ethiopian Smallholder Crop-Yield Challenge

## A1 — Cleaning Log

| file                                                   | columns                                                                                                                                                                                                 | issue_type                      |   rows_affected |   percent_affected | fix_applied                                                             | why                                                                               |
|:-------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------|----------------:|-------------------:|:------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
| crop_yield_train.csv                                   | survey_year, altitude_m, rainfall_mm_season, farm_size_ha, fertilizer_kg_per_ha, improved_seed_used, pest_disease_flag, soil_quality_index, labor_days_per_ha, distance_to_market_km, yield_tons_per_ha | sentinel values (-999)          |             788 |              5.222 | Converted -999 to NaN before validation/imputation.                     | -999 is a missing-value sentinel, not a valid measurement.                        |
| crop_yield_train.csv + crop_yield_leaderboard_test.csv | region, crop_type, planting_month                                                                                                                                                                       | inconsistent categorical labels |             125 |              0.663 | Trimmed/lowercased labels and mapped known aliases to canonical labels. | Required identical category labels across train, test, weather, and price tables. |
| market_prices.csv                                      | region, crop_type                                                                                                                                                                                       | inconsistent categorical labels |               0 |              0     | Standardized region and crop labels.                                    | Required reliable price joins.                                                    |
| regional_weather.csv                                   | region, year, month, avg_temp_c, monthly_rainfall_mm, extreme_heat_days                                                                                                                                 | duplicate rows                  |               1 |              0.431 | Removed exact duplicate rows.                                           | Duplicate records can distort aggregation and model training.                     |
| regional_weather.csv                                   | region, year, month                                                                                                                                                                                     | duplicate weather keys          |              10 |              4.329 | Aggregated duplicate readings by mean for numeric weather variables.    | The weather join requires one region/year/month record.                           |
| regional_weather.csv                                   | region, month                                                                                                                                                                                           | inconsistent categorical labels |              19 |              8.19  | Standardized region and month labels.                                   | Required reliable region/year/month weather joins.                                |

## A2 — Key Standardization Proof

| table      | field     | unique_after                                      | allowed_set_match   |
|:-----------|:----------|:--------------------------------------------------|:--------------------|
| plot_train | region    | ['Amhara', 'Oromia', 'SNNPR', 'Somali', 'Tigray'] | True                |
| plot_test  | region    | ['Amhara', 'Oromia', 'SNNPR', 'Somali', 'Tigray'] | True                |
| weather    | region    | ['Amhara', 'Oromia', 'SNNPR', 'Somali', 'Tigray'] | True                |
| price      | region    | ['Amhara', 'Oromia', 'SNNPR', 'Somali', 'Tigray'] | True                |
| plot_train | crop_type | ['barley', 'maize', 'sorghum', 'teff', 'wheat']   | True                |
| plot_test  | crop_type | ['barley', 'maize', 'sorghum', 'teff', 'wheat']   | True                |
| price      | crop_type | ['barley', 'maize', 'sorghum', 'teff', 'wheat']   | True                |

## A3 — Join Map

Join diagram saved at `reports/join_map.png`.

Growing-season rule: planting month plus the following three months, using weather observations for the plot's survey year. Incomplete windows are recorded in the join audit.

## A4 — Join Audit

| join           |   left_rows_before |   left_rows_after |   match_rate_pct |   unmatched_plots |   incomplete_weather_seasons | cardinality                            | row_count_changed   |
|:---------------|-------------------:|------------------:|-----------------:|------------------:|-----------------------------:|:---------------------------------------|:--------------------|
| plot → weather |              15090 |             15090 |           100    |                 0 |                         6146 | many-to-one after seasonal aggregation | False               |
| plot → price   |              15090 |             15090 |            96.38 |               547 |                          nan | many-to-one                            | False               |

## A5 — Join Proof

| plot_id     | region   |   survey_year | planting_month   |   weather_months_used |   season_mean_temp_c |   season_rainfall_total_mm |   season_extreme_heat_days |
|:------------|:---------|--------------:|:-----------------|----------------------:|---------------------:|---------------------------:|---------------------------:|
| PLOT-000339 | Oromia   |          2024 | Aug              |                     3 |              17.8667 |                      205.9 |                          0 |
| PLOT-006092 | Oromia   |          2024 | Feb              |                     4 |              19.2    |                      294.9 |                          0 |
| PLOT-007457 | Amhara   |          2023 | Jun              |                     3 |              15.4667 |                      555.9 |                          0 |

## A6 — Feature Engineering

| feature_name                         | formula                                                           | source_columns                                 | source_table                | reason                                                              |
|:-------------------------------------|:------------------------------------------------------------------|:-----------------------------------------------|:----------------------------|:--------------------------------------------------------------------|
| season_mean_temp_c                   | mean(monthly avg_temp_c over planting month + next 3 months)      | regional_weather.avg_temp_c                    | regional_weather.csv        | Captures average growing-season thermal conditions.                 |
| season_rainfall_total_mm             | sum(monthly_rainfall_mm over growing season)                      | regional_weather.monthly_rainfall_mm           | regional_weather.csv        | Represents accumulated seasonal rainfall.                           |
| season_extreme_heat_days             | sum(extreme_heat_days over growing season)                        | regional_weather.extreme_heat_days             | regional_weather.csv        | Captures heat stress exposure.                                      |
| season_temp_std_c                    | std(monthly avg_temp_c over growing season)                       | regional_weather.avg_temp_c                    | regional_weather.csv        | Captures temperature variability during crop development.           |
| season_temp_deviation_c              | season_mean_temp_c - training region typical seasonal temperature | season_mean_temp_c, region                     | derived from plot + weather | Measures whether the season was unusually warm/cool for the region. |
| fertilizer_improved_seed_interaction | fertilizer_kg_per_ha × improved_seed_used                         | fertilizer_kg_per_ha, improved_seed_used       | plot survey                 | Captures the combined effect of fertilizer and improved seed.       |
| planting_month_num                   | calendar month number                                             | planting_month                                 | plot survey                 | Allows the model to represent planting-time effects.                |
| rainfall_per_fertilizer              | season_rainfall_total_mm / (fertilizer_kg_per_ha + 1)             | season_rainfall_total_mm, fertilizer_kg_per_ha | derived                     | Represents rainfall availability relative to fertilizer investment. |

## A7 — Integrity Checks

| check                              | status   | details                                                                                                                                                                                                                                   |
|:-----------------------------------|:---------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Train plot_id is unique            | PASS     | Unique IDs: 15090 / 15090                                                                                                                                                                                                                 |
| Test plot_id is unique             | PASS     | Unique IDs: 3750 / 3750                                                                                                                                                                                                                   |
| Train row count preserved          | PASS     | Original after exact-duplicate removal: 15090; master: 15090                                                                                                                                                                              |
| Test row count preserved           | PASS     | Original after exact-duplicate removal: 3750; master: 3750                                                                                                                                                                                |
| Train regions valid                | PASS     | Regions: ['Amhara', 'Oromia', 'SNNPR', 'Somali', 'Tigray']                                                                                                                                                                                |
| Test regions valid                 | PASS     | Regions: ['Amhara', 'Oromia', 'SNNPR', 'Somali', 'Tigray']                                                                                                                                                                                |
| Train crops valid                  | PASS     | Crops: ['barley', 'maize', 'sorghum', 'teff', 'wheat']                                                                                                                                                                                    |
| Test crops valid                   | PASS     | Crops: ['barley', 'maize', 'sorghum', 'teff', 'wheat']                                                                                                                                                                                    |
| No missing model features in train | PASS     | Missing cells: 0                                                                                                                                                                                                                          |
| No missing model features in test  | PASS     | Missing cells: 0                                                                                                                                                                                                                          |
| Train target is non-negative       | PASS     | Minimum yield: 0.1                                                                                                                                                                                                                        |
| Price is positive                  | PASS     | All joined price values are positive.                                                                                                                                                                                                     |
| Train/test feature schemas match   | PASS     | Test contains the same master feature columns as train excluding target.                                                                                                                                                                  |
| Weather feature exists             | PASS     | Weather-derived features: ['season_mean_temp_c', 'season_rainfall_total_mm', 'season_extreme_heat_days', 'season_temp_std_c', 'season_temp_deviation_c', 'weather_months_available', 'weather_months_expected', 'weather_months_missing'] |

## A8 — Master Tables

- `data/processed/master_train.csv`: 15,090 rows

- `data/processed/master_test.csv`: 3,750 rows

- `data/processed/data_dictionary_master.csv`


### Model hygiene
- Weather-derived features are included in the master tables.
- Market price is included for analysis/demo only and must not be used as a yield-model feature.
- Train-derived statistics are used for plot-level and weather-feature imputation.
- Raw files are never modified.
