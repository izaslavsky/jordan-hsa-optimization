# Data Flow and Reproducibility Map

This document records the current public workflow from synthetic facility/patient inputs through HSA construction, population allocation, climate extraction, weekly forecasting analyses, and the daily explanatory DLNM. All paths are relative to the repository root.

## Configuration used for the publication analysis

| Setting | Value |
|---|---|
| Public network prefix | `SYNMODINF` |
| HSA mode | `footprint` |
| Primary boundary bundle | `v7` |
| Population-coverage target | 90% |
| Daily study dates | 2022-07-15 through 2024-01-31 after lag construction |
| Daily primary cohort | All 19 v7 INF-FOOTPRINT HSAs |
| Daily sensitivity cohort | Nine HSAs with mean daily diarrheal count ≥1 |

Version 6 and version 8 are retained for boundary sensitivity work. Generated results belong under `out/` and are not committed.

## End-to-end flow

```text
Public/synthetic inputs in data/
        │
        ├── GEE facility climate extraction
        │       GEE_local_Climate_Features_by_Facilities.ipynb
        │
        ▼
HSA construction
        HSA_FINAL.ipynb
        ├── v6: greedy multi-objective solution
        ├── v7: v6 + anchor quality-control refinement [publication primary]
        └── v8: v7 + satellite-bubble geometry
        │
        ▼
Population allocation
        Population_Allocation_Probabilistic_v2.ipynb
        │
        ├──────────────────────────────┐
        ▼                              ▼
Weekly HSA climate                Daily HSA climate
GEE_local_HSA_Weekly_...          GEE_local_HSA_Daily_Climate.ipynb
        │                              │
        ▼                              ▼
Generate_Modeling_Dataset.ipynb   Generate_Daily_Modeling_Dataset.ipynb
        │                              │
        ▼                              ▼
run_climate_health_modeling.ipynb run_dlnm_primary_sensitivity.py
weekly forecasting + robustness   primary + sensitivity explanatory DLNM
```

## 1. Public and synthetic inputs

### Synthetic health-network files

For each public network, the repository includes:

```text
data/SYNMODINF_facility_coordinates.csv
data/SYNMODINF_groups_of_diagnoses.csv
data/SYNMODINF_patient_visits.csv
data/SYNMODNCD_facility_coordinates.csv
data/SYNMODNCD_groups_of_diagnoses.csv
data/SYNMODNCD_patient_visits.csv
```

Only `SYNMOD*` patient/facility files are publishable. Real files matching `data/INF_*` or `data/NCD_*` are excluded by `.gitignore`.

### General public inputs

```text
data/adm_boundaries/*.gpkg
data/jordan_boundary.gpkg
data/jordan_governorates.gpkg
data/jor_ppp_2020_UNadj.tif
data/jor_ppp_2020_constrained.tif
data/jmp_2025_jordan_governorate.csv
data/hsa_metadata.csv
data/jordan_islamic_calendar.csv
data/reporting_gaps.csv
```

The WorldPop rasters and administrative boundaries support coverage, allocation, and comparison analyses. `hsa_metadata.csv` contains derived facility/HSA sanitation values used only for the daily effect-modification analysis. Calendar and reporting-gap tables support daily temporal adjustment and missingness handling.

## 2. Climate features at facilities

**Notebook:** `GEE_local_Climate_Features_by_Facilities.ipynb`

Inputs:

- `data/{NETWORK}_facility_coordinates.csv`
- CHIRPS precipitation
- ERA5-Land temperature
- TerraClimate water-balance variables
- SRTM elevation

Output pattern:

```text
out/{NETWORK}_Facilities_Climate_Features*.csv
```

The output is required before HSA construction because climatic diversity is one component of the FOOTPRINT objective.

## 3. HSA construction

**Notebook:** `HSA_FINAL.ipynb`

**Supporting modules:** `hsa_optimization.py`, `hsa_mapping_working.py`, `hsa_objective_analysis.py`, `generate_diagnosis_counts_v2.py`, and `generate_hsa_metadata.py`.

Inputs:

- synthetic visits, diagnosis groupings, and facility coordinates;
- facility climate features from step 2;
- WorldPop population raster;
- administrative boundaries.

Processing:

1. Aggregate facility patient volume for the requested disease network.
2. Compute adaptive urban/rural service radii.
3. Select anchors using mode-specific multi-objective weights.
4. Save the v6 greedy solution.
5. Apply version 7 anchor replacement, promotion, and demotion safeguards while retaining the coverage target.
6. Construct version 8 satellite-bubble boundaries.
7. Refresh sanitation metadata from the public governorate lookup.

Principal output pattern:

```text
out/{NETWORK}_{mode}_hsas_{v6|v7|v8}.geojson
```

The paper's primary infectious-disease geography is `SYNMODINF_footprint_hsas_v7.geojson` in a synthetic reproduction run; confidential production runs use the analogous non-`SYNMOD` prefix outside the public repository.

## 4. Probabilistic population allocation

**Notebook:** `Population_Allocation_Probabilistic_v2.ipynb`

**Module:** `population_allocation.py`

Each populated raster cell in the union of HSA catchments is assigned to one admissible anchor. Attractiveness is proportional to facility volume raised to `alpha` and inversely proportional to distance raised to `beta`. The implementation limits fallback assignment by anchor service radius, prefers same-governorate alternatives for major facilities, and reports facilities that cannot be assigned plausibly.

Outputs include allocated population rasters/tables, facility assignment classifications, and HSA denominators under versioned `out/` paths.

## 5. Weekly climate-health panel and forecasting analyses

### HSA-level weekly climate

**Notebook:** `GEE_local_HSA_Weekly_Climate_Lagged.ipynb`

Use `GEE_local_HSA_Weekly_Climate_Lagged_chunked.ipynb` when Earth Engine export limits require chunking. The notebooks use versioned HSA polygons and write weekly climate CSVs beneath:

```text
out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD_{VERSION}/FINAL_HSA_CLIMATE/
```

### Weekly modeling dataset

**Notebook:** `Generate_Modeling_Dataset.ipynb`

**Supporting scripts:** `generate_weekly_disease_counts_adjusted.py`, `prepare_ml_dataset.py`, and `package_results.py`.

Principal output:

```text
out/modeling/{NETWORK}_{mode}_modeling_dataset_{version}.csv
```

### Weekly models and robustness analyses

**Notebook:** `run_climate_health_modeling.ipynb`

It invokes the weekly modeling modules, anomaly model, ML models, and analyses `08_climate_ar_decomposition.py` through `16_within_hsa_heterogeneity.py`. Results include autoregressive/seasonal baselines, climate-augmented models, HSA-month anomaly models, spatial-unit comparisons, gravity and weight sensitivity, residual spatial autocorrelation, extreme-event checks, exclusion diagnostics, and within-HSA climate-variance decomposition.

## 6. Daily climate-health panel

### HSA-level daily climate

**Notebook:** `GEE_local_HSA_Daily_Climate.ipynb`

Output root:

```text
out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD_DAILY_{VERSION}/
```

### Daily modeling dataset

**Notebook:** `Generate_Daily_Modeling_Dataset.ipynb`

**Supporting scripts:** `generate_daily_disease_counts.py`, `prepare_daily_modeling_dataset.py`, and `package_results.py`.

The preparation step constructs daily HSA counts, flags system-wide reporting gaps, merges daily climate, adds Ramadan/Eid/day-of-week indicators, and builds lags 0–14.

Principal output:

```text
out/modeling/{NETWORK}_{mode}_daily_modeling_dataset_{version}.csv
```

For the publication reproduction:

```text
out/modeling/SYNMODINF_footprint_daily_modeling_dataset_v7.csv
```

## 7. Publication daily DLNM

**Runner:** `run_dlnm_primary_sensitivity.py`

**Cross-basis module:** `dlnm/dlnm_crossbasis.py`

Model specification:

- quasi-Poisson generalized linear model with Pearson chi-square dispersion;
- HSA fixed effects;
- natural spline of study day with approximately quarterly interior knots;
- day-of-week, Ramadan, Eid al-Fitr, and Eid al-Adha indicators;
- precipitation cross-basis over lags 0–14;
- one exposure knot at the primary cohort's 80th percentile of nonzero precipitation;
- lag knots at days 3 and 7;
- joint interaction between the cross-basis and mean-centered sanitation coverage.

The primary cohort contains all 19 HSAs. The sensitivity cohort applies a mean daily count threshold of 1.0. The primary cohort fixes the exposure knots, median nonzero reference, 90th-percentile nonzero contrast, and lower/higher representative sanitation values for both models.

Run:

```bash
python run_dlnm_primary_sensitivity.py
```

Outputs:

```text
out/modeling/daily_dlnm_primary_sensitivity_v7/
├── dlnm_model_summary.csv
├── dlnm_rr_contrasts.csv
├── dlnm_temperature_screen.csv
├── dlnm_cohort_hsas.csv
├── dlnm_run_metadata.json
└── dlnm_primary_sensitivity_summary.md
```

The metadata file records the input SHA-256 digest, cohort membership, knots, model specification, and fixed contrasts.

## 8. Automated local execution

`run_pipeline.py` executes the local notebook stages but does not automate Earth Engine authentication or Drive exports. Recommended commands are shown in `README.md`. Executed notebook copies and logs are written under `_pipeline_runs/`, which is ignored by Git.

## Reproducibility boundaries

- Earth Engine source collections may be updated by their providers; retain export dates and generated metadata when archiving a run.
- Synthetic data validate the public workflow but cannot reproduce confidential outcome records exactly.
- `out/` is intentionally not version-controlled. Archive the relevant generated tables and input hashes with a manuscript release or DOI-backed repository.
