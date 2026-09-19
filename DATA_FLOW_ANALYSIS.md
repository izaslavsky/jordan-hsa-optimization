# Complete Data Flow Analysis: Synthetic Data to ML Models

**Purpose**: Document reproducible workflow from synthetic data to ML predictions
**Date**: 2026-09-17
**Status**: Reviewed and verified

> **Paths in this document.** Every `out/...` below is written relative to the
> directory for one run. A run is one network, optimization mode and boundary
> version, e.g. `out_INF_footprint_v7/`, passed as `--out-dir` or `HSA_OUT_DIR`.
> Outputs are written flat within that run directory, e.g.
> `out_INF_footprint_v7/modeling/`. One run covers one disease focus, selected
> with `--disease-focus`; to analyse a second disease, use a separate run
> directory rather than a subdirectory, so the two cannot overwrite each
> other's datasets. `out/` itself holds the delineation masters written by
> `HSA_FINAL.ipynb`.
>
> Only the v7 bundle is built by default; set `HSA_VARIANTS="v6,v7,v8"` to
> rebuild the others.

---

## Overview

This document traces the complete data flow from synthetic patient visits through climate extraction, HSA delineation, dataset preparation, to final ML modeling. Each step's inputs and outputs are explicitly defined to ensure reproducibility.

---

## Workflow Diagram

```
Step 1  GEE_local_Climate_Features_by_Facilities.ipynb   [run once]
            │
            ▼
Step 2  HSA_FINAL.ipynb                                  [run once]
            │  → out/{NETWORK}_{mode}_hsas_v7.geojson
            │    (v6: greedy | v7: +anchor QC | v8: +bubbles)
            │
            ├── (optional) compare_delineations.ipynb
            │
            ▼
    ┌── BOUNDARY_VERSION = v6 | v7 | v8 ─────────────-─┐
    │  (Steps 3–6 repeat for each boundary version)    │
    └──────────────────────────────────────────────────┘
            │
            ▼
Step 3  Population_Allocation_Probabilistic_v2.ipynb
            │
     ┌──────┴────────────────────┐
     ▼                           ▼
Step 4 (weekly)           Step 4 (daily)
GEE_local_HSA_Weekly_     GEE_local_HSA_Daily_
Climate_Lagged.ipynb      Climate.ipynb
     │                           │
     ▼                           ▼
Step 5 (weekly)           Step 5 (daily)
Generate_Modeling_        Generate_Daily_Modeling_
Dataset.ipynb             Dataset.ipynb
     │                           │
     ▼                           ▼
Step 6 (weekly)           Step 6 (daily)
run_climate_health_       run_climate_models_daily.ipynb
modeling.ipynb            Track A: DLNM (explanatory)
                          Track B: OLS horizons (predictive)
```

---

## Available Data Files ✅

### Synthetic Patient Data and Facility Coordinates

Files use `SYNMODINF_` prefix for infectious diseases and `SYNMODNCD_` prefix for non-communicable diseases:

```
data/
├── SYNMODINF_patient_visits.csv             ✅ (synthetic patient visits)
├── SYNMODNCD_patient_visits.csv             ✅
├── SYNMODINF_facility_coordinates.csv       ✅ (facility locations)
├── SYNMODNCD_facility_coordinates.csv       ✅
├── SYNMODINF_groups_of_diagnoses.csv        ✅ (diagnosis code groupings)
├── SYNMODNCD_groups_of_diagnoses.csv        ✅
├── jmp_2025_jordan_governorate.csv          ✅ (JMP 2025 sanitation/water by governorate; used only by DLNM Track A)
├── hsa_metadata.csv                         ✅ (generated from jmp_2025_jordan_governorate.csv by generate_hsa_metadata.py; used only by DLNM Track A)
├── jordan_islamic_calendar.csv              ✅ (Ramadan/Eid date ranges 2022-2024; read by prepare_daily_modeling_dataset.py; daily DLNM only)
├── reporting_gaps.csv                       ✅ (system-wide HMIS outage dates; read by generate_daily_disease_counts.py; daily DLNM only)
└── adm_boundaries/                          ✅
    ├── Jordan_governorates_simplified20m.gpkg
    ├── Jordan_districts_simplified20m.gpkg
    └── Jordan_subdistricts_simplified20m.gpkg
```

Columns in patient visits: `patientid`, `gender`, `ageatdiagnosis`, `governorate`, `diagnosisid`, `diagnosis`, `general_category`, `datetimediagnosisentered`, `healthfacility`, `healthfacilitytype`

---

## Complete Workflow Steps

### **STEP 1: Climate Feature Extraction at Facilities**
**Notebook**: `GEE_local_Climate_Features_by_Facilities.ipynb`

**Purpose**: Extract climate statistics at facility locations for HSA clustering

**Inputs**:
- ✅ `data/INF_facility_coordinates.csv` (18 facilities with lat/lon)
- GEE datasets: CHIRPS (precipitation), ERA5-Land (temperature), TerraClimate (water balance), SRTM (elevation)

**Process**:
1. Upload facility coordinates to Google Earth Engine
2. Extract climate statistics (2019-2024) for 2.5km buffer zones around each facility
3. Compute features: P_mean_mm, T_mean_C, DTR_C, PET_mm, VPD_kPa, elevation_m, etc.
4. Run k-means clustering (k=8) on climate features
5. Export facility-level climate summaries with cluster assignments

**Outputs**:
- `INF_Hospitals_Climate_Features_with_clusters.csv` (18 rows, 1 per facility)
  - Columns: `FacilityName`, `lat`, `lon`, `P_mean_mm`, `T_mean_C`, `DTR_C`, `PET_mm`, `VPD_kPa`, `elevation_m`, `climate_k`, [other climate features]

**Command**:
```
jupyter notebook GEE_local_Climate_Features_by_Facilities.ipynb
```

**GEE project ID**: Before running, set `GEE_PROJECT = "your-project-id"` in the configuration cell. See `SETUP_INSTRUCTIONS.md` for how to obtain and configure a GEE project.

**Status**: ⚠️ Requires running GEE notebook (output not in repo due to generation requirement)

---

### **STEP 2: HSA Delineation**
**Notebook**: `HSA_FINAL.ipynb`

**Purpose**: Delineate Hospital Service Areas using unified scoring system with mode-specific weight profiles

**Inputs**:
- ✅ `data/SYNMODINF_patient_visits.csv` (or `data/INF_patient_visits.csv` for real data)
- ✅ `data/SYNMODINF_facility_coordinates.csv`
- ✅ `data/SYNMODINF_groups_of_diagnoses.csv` (ICD groupings)
- From Step 1: `{NETWORK}_Facilities_Climate_Features_with_clusters.csv` ⚠️
- ✅ `data/adm_boundaries/Jordan_governorates_simplified20m.gpkg`

**Process**:
1. Load synthetic patient visits and generate diagnosis counts (integrated into notebook)
2. Load facility coordinates and merge with climate data
3. Run unified scoring optimization with 5 different modes (mode-specific weight profiles):
   - `FEWEST`: Minimize # HSAs while achieving 90% coverage
   - `FOOTPRINT`: Maximize geographic diversity across climate zones
   - `DISTANCE`: Minimize average travel distance
   - `GOVERNORATE_TAU_COVERAGE`: Achieve 90% coverage in each governorate
   - `GOVERNORATE_FEWEST`: One anchor per governorate + minimize total HSAs
4. For each mode, iteratively select facilities using composite scoring
5. Adaptive radii: Urban facilities (10km) vs rural facilities (18km)
6. Post-processing: Remove HSAs with >80% overlap
7. Export HSA boundaries as GeoJSON with circular geometries
8. Call `generate_hsa_metadata.py` — refreshes `data/hsa_metadata.csv` from facility coordinates + JMP 2025 governorate lookup (covers all facilities, not just current anchors)

**Outputs** (15 GeoJSON files, one per mode × variant, plus refreshed metadata):
- `out/{NETWORK}_{mode}_hsas_{v6|v7|v8}.geojson` (5 modes × 3 variants)
- `data/hsa_metadata.csv` — facility-level sanitation quality scores (JMP 2025); used only by the daily DLNM pipeline (Track A). The weekly pipeline does not use this file.

Each GeoJSON contains:
  - Columns: `healthfacility` (anchor name), `geometry` (circular polygon), `radius_km`, `total_patients`, `composite_score`, `climate_k`
  - **Note**: HSAs are overlapping circles - population allocation (Step 3) is required to prevent double-counting

**Command**:
```
jupyter notebook HSA_FINAL.ipynb
```

**Status**: ⚠️ Ready to run after Steps 0-1 complete (outputs not in repo due to size)

---

### **STEP 3: Patient Allocation for Overlapping HSAs** ✅
**Notebook**: `Population_Allocation_Probabilistic_v2.ipynb`
**Script**: `population_allocation.py` (called by notebook)

**Purpose**: Assign each population pixel to exactly ONE facility to prevent double/triple counting in overlapping HSA regions

**Why This Is Necessary**:
- HSAs from Step 2 are overlapping circular regions around facilities
- Same population pixel can fall within 2-3 HSAs simultaneously
- Without allocation: Same person counted multiple times when computing disease rates
- With allocation: Each pixel assigned to ONE facility based on gravity model (distance + facility size)

**Inputs**:
- From Step 2: HSA boundaries (e.g., `out/INF_footprint_hsas_v2.geojson`) ⚠️
- ✅ `data/jor_ppp_2020_UNadj.tif` (WorldPop population raster, 100m resolution)

**Process**:
1. Load HSA boundaries (17 overlapping circular polygons)
2. Load population raster (10.2M total population)
3. Extract all population pixels within any HSA
4. For each pixel, calculate gravity model attractiveness for all facilities:
   - Attractiveness = (Facility_Volume^α) / (Distance^β)
   - Default: α=1.0, β=2.0
5. Assign each pixel to the facility with highest attractiveness
6. Aggregate allocated populations by HSA (no overlap)
7. Export pixel-level and HSA-level allocations

**Outputs**:
- `out/pixel_allocations_{NETWORK}_{MODE}.csv` (~250,000 rows)
  - Columns: `pixel_id`, `x`, `y`, `lon`, `lat`, `population`, `assigned_facility`, `assigned_hsa_id`, `attractiveness`
  - Each row is one population pixel with its unique facility assignment
- `out/hsa_allocated_patients_{NETWORK}_{MODE}.csv` (1 row per HSA)
  - Columns: `anchor_id`, `anchor_name`, `network_type`, `optimization_mode`, `allocated_patients`, `num_facilities_in_hsa`, `facilities_in_hsa`
  - Summary of allocated population per HSA (no overlap)

**Command**:
```
jupyter notebook Population_Allocation_Probabilistic_v2.ipynb
```

**Status**: ✅ Notebook and script ready to run after Step 2

**Note**: This step is REQUIRED for disease modeling (Steps 5-6) to ensure accurate denominators for disease rate calculations. Skip only if doing HSA visualization or spatial analysis without disease modeling.

---

### **STEP 4: Weekly Climate Extraction by HSA**
**Notebook**: `GEE_local_HSA_Weekly_Climate_Lagged.ipynb`

**Purpose**: Extract weekly climate time series for each HSA polygon

**Inputs**:
- From Step 2: HSA boundaries (e.g., `out/INF_footprint_hsas_v2.geojson`) ⚠️
- GEE datasets: CHIRPS (precipitation), ERA5-Land (temperature/humidity), TerraClimate (water balance)

**Process**:
1. Run locally; notebook reads HSA polygon boundaries from `out/` directory
2. For each HSA, extract weekly climate aggregates (2022-06-27 to 2024-01-29 = 84 weeks)
3. Compute lagged variables for each day-lag (d-1, d-2, d-3, d-5, d-7, d-10, d-14)
4. Compute weekly aggregates (mean, sum, max)
5. Export 6 CSV files per HSA (e.g., for 18 HSAs there will be 18 * 6 = 108 files total)

**Outputs** (6 files per HSA):

**IMPORTANT**: The GEE notebook exports files to Google Drive. Users must manually download them to:
```
out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/
```

File structure:
```
out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/
├── HSA_[Hospital]_precip_lags.csv               
├── HSA_[Hospital]_tempdew_wind_lags.csv         
├── HSA_[Hospital]_evapERA5_lags.csv             
├── HSA_[Hospital]_soilmoistERA5_lags.csv       
├── HSA_[Hospital]_water_balance.csv            
└── HSA_[Hospital]_elevation_by_week.csv         
```

Each CSV structure:
- Rows: 84 (one per week, 2022-06-27 to 2024-01-29)
- Columns: `FacilityName`, `week_start`, [climate variables with lags like `P_mean_d-1`, `T_max_week`, etc.]

`elevation_by_week.csv` is the exception. SRTM is static, so the exporter writes
a single row per HSA rather than repeating one value 84 times, and it reduces
the DEM with a combined reducer instead of a mean:

| Column | Meaning |
| --- | --- |
| `elevation_m` | mean elevation inside the HSA polygon |
| `elevation_sd_m` | standard deviation, i.e. how much terrain varies within the HSA |
| `elevation_min_m`, `elevation_max_m` | range inside the polygon |
| `elevation_p25_m`, `elevation_p50_m`, `elevation_p75_m` | quartiles of the within-HSA distribution |

The spread matters because a service area spanning the Jordan Valley and the
highlands has a very different internal climate gradient than a compact urban
one, and a single mean cannot express that. `16_within_hsa_heterogeneity.py`
reads `elevation_sd_m`, `elevation_min_m` and `elevation_max_m` directly; when
they are absent it reports the gap rather than substituting a modelled
elevation surface.

**Command**:
```
jupyter notebook GEE_local_HSA_Weekly_Climate_Lagged.ipynb
```

**Status**: ⚠️ Requires running GEE notebook with Step 2 outputs, then manual download to local directory

**GEE project ID**: Set `PROJECT = "your-project-id"` in the configuration cell before running.

**Authentication**: Requires both Google Earth Engine (for computation) and Google Drive (for export detection and download). See `SETUP_INSTRUCTIONS.md` for setup.


---

### **STEP 5: Generate Modeling Dataset (Orchestration Notebook)** ✅
**Notebook**: `Generate_Modeling_Dataset.ipynb`
**Called Scripts**: `generate_weekly_disease_counts_adjusted.py`, `prepare_ml_dataset.py`

**Purpose**: Orchestrate complete dataset preparation pipeline by calling existing Python scripts in sequence

**Inputs**:
- ✅ `data/{NETWORK}_patient_visits.csv` (patient visit data)
- From Step 2: HSA boundaries (e.g., `out/{NETWORK}_{MODE}_hsas_v2.geojson`) ⚠️
- From Step 3: `out/pixel_allocations_{NETWORK}_{MODE}.csv` ⚠️
- From Step 4: `out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/*.csv` ⚠️

**Process**:
1. Verify prerequisites (HSA boundaries, population allocations, climate CSVs, patient data)
2. Call `generate_weekly_disease_counts_adjusted.py`: load allocations, apply gravity-model weights, aggregate visits by HSA × week
3. Call `prepare_ml_dataset.py`: merge climate CSVs with disease counts, add temporal features, write modeling dataset

**Outputs**:
- `out/INF_footprint_weekly_diarrheal_adjusted_{VERSION}.csv`
  - Columns: `hsa_id`, `week_start`, `diarrheal_count`
- `out/INF_footprint_weekly_infectious_adjusted_{VERSION}.csv` 
  - Columns: `hsa_id`, `week_start`, `infectious_count`
- `out/modeling/INF_footprint_modeling_dataset_{VERSION}.csv` 
  - Columns: `hsa_id`, `week_start`, [climate features], `diarrheal_count`, temporal features
- `out/modeling/INF_footprint_modeling_dataset_{VERSION}_metadata.json` (feature descriptions)

**Command**:
```bash
jupyter notebook Generate_Modeling_Dataset.ipynb
```

**Status**: ✅ Notebook ready to run after Steps 2-4 complete

**Note**: This orchestration notebook simplifies the workflow by automating the sequence of data preparation steps. It checks prerequisites, calls existing scripts, and verifies outputs - no need to run individual scripts manually.

---

### **STEP 6: ML Model Training**
**Script**: `train_improved_models.py` ✅

**Purpose**: Train ML models to predict weekly diarrheal disease counts

**Inputs**:
- From Step 5: `out/modeling/modeling_dataset_train.csv` ⚠️
- From Step 5: `out/modeling/modeling_dataset_val.csv` ⚠️
- From Step 5: `out/modeling/modeling_dataset_test.csv` ⚠️

**Process**:
1. Load datasets
2. Add autoregressive features: `diarrheal_count_adjusted_lag1`, `diarrheal_count_adjusted_lag2`
3. Test 5 feature sets:
   - AR_only (2 features)
   - AR_temporal (5 features)
   - AR_top5_climate (7 features)
   - AR_top10_climate (12 features) ← Best
   - AR_temporal_top5 (10 features)
4. Train 5 models per feature set: Ridge, Lasso, RandomForest, GradientBoosting, XGBoost
5. Evaluate on validation set (25 models total)

**Outputs**:
- `out/modeling/results_improved/{NETWORK}_{HSA_MODE}_improved_model_comparison.csv` (25 rows, 1 per model)
  - Columns: `feature_set`, `model`, `num_features`, `val_r2`, `val_rmse`, `val_mae`
  - Best model: `GradientBoosting` with `AR_top10_climate`, R² = 0.526

**Command**:
```bash
python train_improved_models.py
```

**Status**: ✅ Script exists, ⚠️ requires Step 5 outputs

---

## Summary: Workflow Status

### ✅ Complete and Ready to Use
1. **Step 3**: `Population_Allocation_Probabilistic_v2.ipynb` (calls `population_allocation.py`)
2. **Step 5 weekly**: `Generate_Modeling_Dataset.ipynb` (calls `generate_weekly_disease_counts_adjusted.py`, `prepare_ml_dataset.py`)
3. **Step 5 daily**: `Generate_Daily_Modeling_Dataset.ipynb` (calls `generate_daily_disease_counts.py`, `prepare_daily_modeling_dataset.py`)
4. **Step 6 weekly**: `run_climate_health_modeling.ipynb` (calls `climate_health_modeling_*.py`, `train_*.py`)
5. **Step 6 daily**: `run_climate_models_daily.ipynb`
6. **Optional**: `compare_delineations.ipynb`
7. **Synthetic data**: All SYNMODINF_/SYNMODNCD_ source files present

### ⚠️ Requires Running (GEE notebooks)
1. **Step 1**: `GEE_local_Climate_Features_by_Facilities.ipynb`
2. **Step 2**: `HSA_FINAL.ipynb`
3. **Step 4 weekly**: `GEE_local_HSA_Weekly_Climate_Lagged.ipynb`
4. **Step 4 daily**: `GEE_local_HSA_Daily_Climate.ipynb`

### 🔄 Dependency Chain

Steps 1 and 2 run once. Set `BOUNDARY_VERSION = "v6" | "v7" | "v8"` for Steps 3–6; run each step once per boundary version needed.

**Key dependencies**:
- Step 3 must complete before Step 5 (allocation outputs feed disease count scripts)
- Step 4 must complete before Step 5 (climate CSVs are merged with disease counts)
- `Generate_Daily_Modeling_Dataset.ipynb` and `Generate_Modeling_Dataset.ipynb` call all helper scripts internally; no need to run `.py` files directly

---

## Quick Start

### Minimal Reproducibility (Option A - Recommended)

**Goal**: Understand the HSA delineation and population allocation workflow without requiring GEE access

1. Examine synthetic data in `data/SYNMODINF_patient_visits.csv`
2. Run `GEE_local_Climate_Features_by_Facilities.ipynb` (Step 1)
3. Run `HSA_FINAL.ipynb` (Step 2) — produces v6/v7/v8 boundary bundles
4. Set `BOUNDARY_VERSION` and run `Population_Allocation_Probabilistic_v2.ipynb` (Step 3)
5. Review HSA boundaries in `out/*.geojson` and allocation outputs in `out/`
6. Optionally run `compare_delineations.ipynb` to compare the three variants

### Full Reproducibility (Option B)

**Goal**: Reproduce the complete pipeline including modeling

After completing Option A, for each `BOUNDARY_VERSION`:

1. Run `GEE_local_HSA_Weekly_Climate_Lagged.ipynb` (Step 4 weekly) — requires GEE access
2. Run `GEE_local_HSA_Daily_Climate.ipynb` (Step 4 daily) — requires GEE access
3. Run `Generate_Modeling_Dataset.ipynb` (Step 5 weekly)
4. Run `Generate_Daily_Modeling_Dataset.ipynb` (Step 5 daily)
5. Run `run_climate_health_modeling.ipynb` (Step 6 weekly)
6. Run `run_climate_models_daily.ipynb` (Step 6 daily)
7. Review results in `out/modeling/`

---

## File Reference Table

| File | Step | Type | Size | In Repo? | How to Generate |
|------|------|------|------|----------|-----------------|
| `SYNMODINF_patient_visits.csv` | Input | Data | ~6 MB | ✅ Yes | N/A (provided) |
| `SYNMODINF_facility_coordinates.csv` | Input | Data | ~13 KB | ✅ Yes | N/A (provided) |
| `jmp_2025_jordan_governorate.csv` | Input | Data | ~2 KB | ✅ Yes | N/A (public JMP 2025 + census; see source_note column) |
| `hsa_metadata.csv` | 2 | Data | ~50 KB | ✅ Yes | Auto-generated by `HSA_FINAL.ipynb` via `generate_hsa_metadata.py` |
| `jordan_islamic_calendar.csv` | Input | Data | ~1 KB | ✅ Yes | N/A (provided; daily DLNM only) |
| `reporting_gaps.csv` | Input | Data | ~2 KB | ✅ Yes | N/A (provided; daily DLNM only) |
| `jor_ppp_2020_UNadj.tif` | Input | Data | ~40 MB | ✅ Yes | N/A (provided) |
| `{NETWORK}_Facilities_Climate_Features_with_clusters.csv` | 1 | Output | ~10 KB | ⚠️ No | Run GEE notebook |
| `{NETWORK}_{MODE}_hsas_{VERSION}.geojson` | 2 | Output | ~20 KB | ⚠️ No | Run `HSA_FINAL.ipynb` |
| `pixel_allocations_{NETWORK}_{MODE}.csv` | 3 | Output | ~7 MB | ⚠️ No | Run `Population_Allocation_Probabilistic_v2.ipynb` |
| `hsa_allocated_patients_{NETWORK}_{MODE}.csv` | 3 | Output | ~5 KB | ⚠️ No | Run `Population_Allocation_Probabilistic_v2.ipynb` |
| `HSA_*_precip_lags.csv` (per HSA) | 4 | Output | varies | ⚠️ No | Run GEE notebook + download |
| `{NETWORK}_{MODE}_weekly_{DISEASE}_adjusted.csv` | 5 (weekly) | Output | ~50 KB | ⚠️ No | Run `Generate_Modeling_Dataset.ipynb` |
| `{NETWORK}_{MODE}_modeling_dataset_{VERSION}.csv` | 5 (weekly) | Output | ~100 KB | ⚠️ No | Run `Generate_Modeling_Dataset.ipynb` |
| `INF_footprint_daily_modeling_dataset_{VERSION}.csv` | 5 (daily) | Output | ~200 KB | ⚠️ No | Run `Generate_Daily_Modeling_Dataset.ipynb` |
| Model results in `out/modeling/` | 6 | Output | varies | ⚠️ No | Run `run_climate_health_modeling.ipynb` or `run_climate_models_daily.ipynb` |

---

**Last Updated**: 2026-07-07
