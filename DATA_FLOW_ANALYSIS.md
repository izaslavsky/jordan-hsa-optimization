# Complete Data Flow Analysis: Synthetic Data to ML Models

**Purpose**: Document reproducible workflow from synthetic data to ML predictions
**Date**: 2025-12-28
**Status**: Reviewed and verified

---

## Overview

This document traces the complete data flow from synthetic patient visits through climate extraction, HSA delineation, dataset preparation, to final ML modeling. Each step's inputs and outputs are explicitly defined to ensure reproducibility.

---

## Workflow Diagram

```
STEP 1: Climate Features (GEE)
  ↓
STEP 2: HSA Delineation (includes diagnosis counts)
  ↓
STEP 2b: Compare Delineations (optional)
  ↓
STEP 3: Population Allocation (prevents double counting)
  ↓
STEP 4: Weekly Climate by HSA (GEE) → Download to local
  ↓
STEP 5: Generate Modeling Dataset (orchestration notebook)
  ├── STEP 5a: Weekly Disease Counts
  └── STEP 5b: Merge Climate + Disease Data
  ↓
STEP 6: Climate-Health Modeling (run_climate_health_modeling)
```

---

## Available Data Files ✅

### Synthetic Patient Data and Facility Coordinates

Files use `SYNINF_` prefix for infectious diseases and `SYNNCD_` prefix for non-communicable diseases:

```
data/
├── SYNINF_patient_visits.csv             ✅ (synthetic patient visits)
├── SYNNCD_patient_visits.csv             ✅
├── SYNINF_facility_coordinates.csv       ✅ (facility locations)
├── SYNNCD_facility_coordinates.csv       ✅
├── SYNINF_groups_of_diagnoses.csv        ✅ (diagnosis code groupings)
├── SYNNCD_groups_of_diagnoses.csv        ✅
└── adm_boundaries/                       ✅
    ├── Jordan_governorates_simplified20m.gpkg
    ├── Jordan_districts_simplified20m.gpkg
    └── Jordan_subdistricts_simplified20m.gpkg
```

Columns in patient visits: `patientid`, `gender`, `ageatdiagnosis`, `governorate`, `diagnosisid`, `diagnosis`, `general_category`, `datetimediagnosisentered`, `healthfacility`, `healthfacilitytype`

---

## Complete Workflow Steps

### **STEP 1: Climate Feature Extraction at Facilities**
**Notebook**: `GEE_Climate_Features_by_Facilities.ipynb`

**Purpose**: Extract climate statistics at facility locations for HSA clustering

**Local GEE notebook**: Included in this repo and runnable without Colab, but requires Earth Engine authentication and (optionally) Google Drive OAuth if you use automated export/download.

**Inputs**:
- ✅ `data/SYNINF_facility_coordinates.csv` (facility locations with lat/lon)
- GEE datasets: CHIRPS (precipitation), ERA5-Land (temperature), TerraClimate (water balance), SRTM (elevation)

**Process**:
1. Upload facility coordinates to Google Earth Engine
2. Extract climate statistics (2019-2024) for 2.5km buffer zones around each facility
3. Compute features: P_mean_mm, T_mean_C, DTR_C, PET_mm, VPD_kPa, elevation_m, etc.
4. Run k-means clustering (k=8) on climate features
5. Export facility-level climate summaries with cluster assignments

**Outputs**:
- `{NETWORK}_Facilities_Climate_Features_with_clusters.csv` (1 row per facility)
  - Columns: `FacilityName`, `lat`, `lon`, `P_mean_mm`, `T_mean_C`, `DTR_C`, `PET_mm`, `VPD_kPa`, `elevation_m`, `climate_k`, [other climate features]

**Command**:
```
Upload to Google Colab and run GEE_Climate_Features_by_Facilities.ipynb
```

**Status**: ⚠️ Requires running GEE notebook (output not in repo due to generation requirement)

---

### **STEP 2: HSA Delineation**
**Notebook**: `HSA_v6_FINAL.ipynb`

**Purpose**: Delineate Hospital Service Areas using unified scoring system with mode-specific weight profiles

**Inputs**:
- ✅ `data/SYNINF_patient_visits.csv` (or `{NETWORK}_patient_visits.csv`)
- ✅ `data/SYNINF_facility_coordinates.csv`
- ✅ `data/SYNINF_groups_of_diagnoses.csv` (ICD groupings)
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

**Outputs** (per mode, two GeoJSON files + one GeoPackage):
- `out/{NETWORK}_{MODE}_hsas_circles.geojson` — service-radius circles clipped to Jordan boundary
  - Columns: `healthfacility`, `geometry` (circle polygon), `radius_km`, `circle_area_km2`, `total_patients`, `composite_score`, `climate_k`
- `out/{NETWORK}_{MODE}_hsas_v2.geojson` — circles further clipped to WorldPop constrained inhabited cells
  - Same columns plus `inhabited_area_km2` and `circle_geometry_wkt` (original circle preserved)
  - Used for GEE climate extraction, population allocation, and map display
- `out/{NETWORK}_{MODE}_map.gpkg` — GeoPackage with 5 layers: `hsa_circles`, `hsa_boundaries`, `hsa_anchors`, `all_facilities`, `country_boundary`

**Note**: HSAs are overlapping service areas — population allocation (Step 3) is required to prevent double-counting in disease rate calculations

**Command**:
```
jupyter notebook HSA_v6_FINAL.ipynb
```

**Status**: ⚠️ Ready to run after Steps 0-1 complete (outputs not in repo due to size)

---

### **STEP 3: Population Allocation for Overlapping HSAs** ✅
**Notebook**: `Patient_Allocation_Probabilistic.ipynb`
**Script**: `patient_allocation.py` (called by notebook)

**Purpose**: Probabilistically allocate population across overlapping facilities to prevent double/triple counting in overlapping HSA regions

**Why This Is Necessary**:
- HSAs from Step 2 are overlapping circular regions around facilities
- Same population pixel can fall within 2-3 HSAs simultaneously
- Without allocation: Same person counted multiple times when computing disease rates
- With probabilistic allocation: Each pixel's population is distributed proportionally across nearby facilities based on gravity model attractiveness (capacity^α / distance^β)

**Inputs**:
- From Step 2: HSA boundaries (e.g., `out/INF_footprint_hsas_v2.geojson`) ⚠️
- ✅ `data/jor_ppp_2020_UNadj.tif` (WorldPop population raster, 100m resolution)

**Process (Two-Step)**:

*Step 1 — Pixel to ALL Facilities (Probabilistic)*:
1. Load ALL facility coordinates (~188 facilities, not just HSA anchors)
2. Load population raster (10.2M total population)
3. For each population pixel, calculate gravity model attractiveness for all reachable facilities:
   - P(facility_i | pixel) = (Volume_i^α / Distance_i^β) / Σ_j (Volume_j^α / Distance_j^β)
   - Default: α=0.75, β=1.5, max distance = 100km
4. Distribute each pixel's population proportionally across facilities

*Step 2 — Facility to HSA Aggregation (Three-Case Logic)*:
5. Load HSA boundaries (overlapping circular polygons around anchor facilities)
6. For each facility, determine HSA assignment:
   - Case 1: Facility inside exactly 1 HSA → 100% to that HSA
   - Case 2: Facility outside all HSAs → nearest HSA anchor within 100km
   - Case 3: Facility inside 2+ overlapping HSAs → proportional split using gravity model
7. Aggregate facility populations to HSAs
8. Export pixel-level allocations and HSA-level population summaries

**Outputs**:
- `out/pixel_allocations_{NETWORK}_{MODE}.csv` (~250,000 rows)
  - Each row is one population pixel with its facility allocation probabilities
- `out/{NETWORK}_{MODE}_hsa_populations_probabilistic.csv` (1 row per HSA)
  - Summary of allocated population per HSA
- `out/{NETWORK}_{MODE}_facility_hsa_assignments.csv`
  - Facility-to-HSA assignments showing three-case logic

**Command**:
```
jupyter notebook Patient_Allocation_Probabilistic.ipynb
```

**Status**: ✅ Notebook and script ready to run after Step 2

**Note**: This step is REQUIRED for disease modeling (Steps 5-6) to ensure accurate denominators for disease rate calculations. Skip only if doing HSA visualization or spatial analysis without disease modeling.

---

### **STEP 4: Weekly Climate Extraction by HSA**
**Notebook**: `GEE_HSA_Weekly_Climate_Lagged.ipynb`

**Purpose**: Extract weekly climate time series for each HSA polygon

**Local GEE notebook**: Included in this repo and runnable without Colab, but requires Earth Engine authentication and (optionally) Google Drive OAuth if you use automated export/download.

**Inputs**:
- From Step 2: HSA boundaries (e.g., `out/INF_footprint_hsas_v2.geojson`) ⚠️
- GEE datasets: CHIRPS (precipitation), ERA5-Land (temperature/humidity), TerraClimate (water balance)

**Process**:
1. Upload HSA polygon boundaries to Google Earth Engine (the Google Colab notebook will ask to upload a geojson file)
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

**Command**:
```
Upload to Google Colab and run GEE_HSA_Weekly_Climate_Lagged.ipynb
Download exported files from Google Drive to local directory (see above)
```

**Status**: ⚠️ Requires running GEE notebook with Step 2 outputs, then manual download to local directory

**Authentication Note**:
STEP 4 requires both Google Earth Engine authentication (for computation)
and Google Drive authentication (for export detection and download).
See `SETUP_INSTRUCTIONS.md` for detailed setup.


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

**STEP 5a: Check Prerequisites**
1. Verify all required files exist (HSA boundaries, population allocations, climate CSVs, patient data)
2. Display file counts and sizes

**STEP 5b: Generate Weekly Disease Counts**
- Calls: `generate_weekly_disease_counts_adjusted.py` (or `generate_weekly_disease_counts.py`)
- Process:
  1. Load HSA geometries and population allocations from Step 3
  2. Calculate facility-to-HSA probabilities using gravity model
  3. Load patient visits and identify diarrheal diseases (from `general_category` field)
  4. Generate Monday-anchored weeks (84 weeks: 2022-06-27 to 2024-01-29)
  5. Apply gravity model adjustments to prevent double-counting
  6. Aggregate weighted counts by HSA × week
- Outputs:
  - `out/{NETWORK}_{MODE}_weekly_diarrheal.csv`
  - `out/{NETWORK}_{MODE}_weekly_infectious.csv`

**STEP 5c: Merge Climate + Disease Data**
- Calls: `prepare_ml_dataset.py`
- Process:
  1. Load climate CSV files (102 files: 6 variable types × 17 HSAs)
  2. Merge with weekly disease counts on `hsa_id` + `week_start`
  3. Feature engineering: temporal features (week, month, season)
  4. Feature selection: Reduce climate features to prevent overfitting
  5. Create modeling dataset ready for ML training
- Outputs:
  - `out/{NETWORK}_{MODE}_modeling_dataset.csv`
  - `out/{NETWORK}_{MODE}_modeling_dataset_metadata.json`

**STEP 5d: Verify Outputs**
1. List all generated files with row counts and sizes
2. Display dataset summary (rows, columns, HSAs, weeks, date range)
3. Show disease count statistics

**Outputs**:
- `out/INF_footprint_weekly_diarrheal_adjusted.csv`
  - Columns: `hsa_id`, `week_start`, `diarrheal_count`
- `out/INF_footprint_weekly_infectious_adjusted.csv` 
  - Columns: `hsa_id`, `week_start`, `infectious_count`
- `out/INF_footprint_modeling_dataset.csv` 
  - Columns: `hsa_id`, `week_start`, [climate features], `diarrheal_count`, temporal features
- `out/INF_footprint_modeling_dataset_metadata.json` (feature descriptions)

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
1. **STEP 2b**: Delineation comparison notebook (`compare_delineations.ipynb`)
   - Calls `compare_spatial_methods_v2.py`
2. **STEP 3**: Population allocation notebook and script (`Patient_Allocation_Probabilistic.ipynb`, `patient_allocation.py`)
3. **STEP 5**: Modeling dataset orchestration notebook (`Generate_Modeling_Dataset.ipynb`)
   - Calls `generate_weekly_disease_counts_adjusted.py` and `prepare_ml_dataset.py`
4. **STEP 6**: Climate-health modeling notebook (`run_climate_health_modeling.ipynb`)
   - Calls `climate_health_modeling_*.py` and `train_*.py` scripts
5. **Synthetic Data**: All source data files present (SYNINF_/SYNNCD_ prefix)
6. **Documentation**: README.md, CLIMATE_HEALTH_MODELING.md

### ⚠️ Requires Running (GEE or Notebooks)
1. **STEP 1**: Run GEE notebook to extract climate features at facilities
2. **STEP 2**: Run HSA optimization notebook to generate HSA boundaries
3. **STEP 4**: Run GEE notebook to extract weekly climate by HSA, then download CSVs from Google Drive

### 🔄 Dependency Chain

**To reproduce the full workflow:**

```
START → STEP 1 (GEE climate at facilities) → STEP 2 (HSA delineation, includes diagnosis counts)
  ↓
STEP 3 (population allocation - REQUIRED for disease modeling)
  ↓
STEP 4 (GEE weekly climate by HSA) → Download CSVs from Google Drive
  ↓
STEP 5 (Generate_Modeling_Dataset.ipynb - orchestrates Steps 5a & 5b)
  ├── STEP 5a: Weekly disease counts (prevent double counting)
  └── STEP 5b: Merge climate + disease data
  ↓
STEP 6 (ML modeling - future work) → RESULTS
```

**Key Dependencies**:
- STEP 3 (Population Allocation) is REQUIRED before STEP 5 to prevent double counting in overlapping HSAs
- STEP 4 outputs must be manually downloaded to `out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/`
- STEP 5 orchestrates the data preparation pipeline and calls existing Python scripts

**Critical path for reviewers to understand methodology:**
1. Read README.md and CLIMATE_HEALTH_MODELING.md (methodology)
2. Examine scripts (patient_allocation.py, generate_*.py, prepare_*.py, train_*.py)
3. See workflow diagram (this document)
4. Optionally run Steps 1-3 with synthetic data to see HSA delineation and population allocation
5. GEE steps demonstrate climate extraction (reviewers may not have GEE access)

---

## Quick Start Guide for Reviewers

### Minimal Reproducibility (Option A - Recommended)

**Goal**: Understand the HSA delineation and population allocation workflow without requiring GEE access

**Steps**:
1. **Examine synthetic data**: `data/SYNINF_patient_visits.csv`
2. **Run GEE notebook** (or use provided climate file): Extract climate features
3. **Run HSA optimization**: `jupyter notebook HSA_v6_FINAL.ipynb` (includes diagnosis counts)
4. **Run population allocation**: `jupyter notebook Patient_Allocation_Probabilistic.ipynb`
5. **Review outputs**: HSA boundaries in `out/*.geojson` and population allocations in `out/pixel_allocations_*.csv`
6. **Read methodology**: `README.md` and `CLIMATE_HEALTH_MODELING.md`

**Result**: Understand HSA delineation, population allocation to prevent double counting, and see how synthetic data flows through the system

### Full Reproducibility (Option B)

**Goal**: Reproduce entire pipeline including ML models

**Additional Steps**:
8. **Run GEE climate extraction**: Extract weekly climate by HSA (requires GEE access)
9. **Download climate CSVs**: From Google Drive to `out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/`
10. **Run modeling dataset generation**: `jupyter notebook Generate_Modeling_Dataset.ipynb`
    - Generates weekly disease counts with gravity model adjustments
    - Merges climate + disease data
11. **Run climate-health modeling**: `jupyter notebook run_climate_health_modeling.ipynb`
    - Runs comprehensive and parsimonious modeling scripts
    - Trains multiple ML model families
12. **Review results**: See `out/modeling/` for model outputs

**Result**: Complete end-to-end reproduction from synthetic data → ML predictions

---

## File Reference Table

| File | Step | Type | Size | In Repo? | How to Generate |
|------|------|------|------|----------|-----------------|
| `SYNINF_patient_visits.csv` | Input | Data | ~6 MB | ✅ Yes | N/A (provided) |
| `SYNINF_facility_coordinates.csv` | Input | Data | ~13 KB | ✅ Yes | N/A (provided) |
| `jor_ppp_2020_UNadj.tif` | Input | Data | ~40 MB | ✅ Yes | N/A (provided) |
| `{NETWORK}_Facilities_Climate_Features_with_clusters.csv` | 1 | Output | ~10 KB | ⚠️ No | Run GEE notebook |
| `{NETWORK}_{MODE}_hsas_circles.geojson` | 2 | Output | ~15 KB | ⚠️ No | Run `HSA_v6_FINAL.ipynb` |
| `{NETWORK}_{MODE}_hsas_v2.geojson` | 2 | Output | ~20 KB | ⚠️ No | Run `HSA_v6_FINAL.ipynb` |
| `{NETWORK}_{MODE}_map.gpkg` | 2 | Output | ~200 KB | ⚠️ No | Run `HSA_v6_FINAL.ipynb` |
| `pixel_allocations_{NETWORK}_{MODE}.csv` | 3 | Output | ~7 MB | ⚠️ No | Run `Patient_Allocation_Probabilistic.ipynb` |
| `{NETWORK}_{MODE}_hsa_populations_probabilistic.csv` | 3 | Output | ~5 KB | ⚠️ No | Run `Patient_Allocation_Probabilistic.ipynb` |
| `{NETWORK}_{MODE}_facility_hsa_assignments.csv` | 3 | Output | ~20 KB | ⚠️ No | Run `Patient_Allocation_Probabilistic.ipynb` |
| `HSA_*_precip_lags.csv` (per HSA) | 4 | Output | varies | ⚠️ No | Run GEE notebook + download |
| `{NETWORK}_{MODE}_weekly_{DISEASE}_adjusted.csv` | 5a | Output | ~50 KB | ⚠️ No | Run `Generate_Modeling_Dataset.ipynb` |
| `{NETWORK}_{MODE}_modeling_dataset.csv` | 5b | Output | ~100 KB | ⚠️ No | Run `Generate_Modeling_Dataset.ipynb` |
| Model results in `out/modeling/` | 6 | Output | varies | ⚠️ No | Run `run_climate_health_modeling.ipynb` |

---

**Document Status**: Complete and Updated with Population Allocation Workflow
**Last Updated**: 2026-03-29
**Next Action**:
1. Run GEE climate extraction notebook (Step 1)
2. Run `HSA_v6_FINAL.ipynb` for HSA delineation (includes diagnosis counts)
3. Run `Patient_Allocation_Probabilistic.ipynb` to prevent double counting
4. Download climate CSVs from Google Drive to `out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/`
5. Run `Generate_Modeling_Dataset.ipynb` to create complete modeling dataset
