# Complete Data Flow Analysis: Synthetic Data to ML Models

**Purpose**: Document reproducible workflow from synthetic data to ML predictions
**Date**: 2024-12-28
**Status**: Reviewed and verified

---

## Overview

This document traces the complete data flow from synthetic patient visits through climate extraction, HSA delineation, dataset preparation, to final ML modeling. Each step's inputs and outputs are explicitly defined to ensure reproducibility.

---

## Workflow Diagram

```
STEP 0: Diagnosis Counts
  ↓
STEP 1: Climate Features (GEE)
  ↓
STEP 2: HSA Delineation
  ↓
STEP 3: Weekly Climate by HSA (GEE)
  ↓
STEP 4: Weekly Disease Counts
  ↓
STEP 5: ML Dataset Prep
  ↓
STEP 6: ML Modeling
```

---

## Available Data Files ✅

### Synthetic Patient Data
```
data/synthetic/
├── INF_patient_visits_SYNTHETIC.csv  ✅ (50,000 rows)
└── NCD_patient_visits_SYNTHETIC.csv  ✅
```

Columns: `patientid`, `gender`, `ageatdiagnosis`, `governorate`, `diagnosisid`, `diagnosis`, `general_category`, `datetimediagnosisentered`, `healthfacility`, `healthfacilitytype`

### Facility & Administrative Data
```
data/
├── INF_facility_coordinates.csv          ✅ (18 facilities)
├── NCD_facility_coordinates.csv          ✅
├── INF_groups_of_diagnoses.csv          ✅
├── NCD_groups_of_diagnoses.csv          ✅
└── adm_boundaries/                       ✅
    ├── Jordan_governorates_simplified20m.gpkg
    ├── Jordan_districts_simplified20m.gpkg
    └── Jordan_subdistricts_simplified20m.gpkg
```

---

## Complete Workflow Steps

### **STEP 0: Generate Diagnosis Counts** ✅
**Script**: `generate_diagnosis_counts.py` ✅

**Purpose**: Aggregate synthetic patient visits to facility-level diagnosis counts

**Inputs**:
- ✅ `data/synthetic/INF_patient_visits_SYNTHETIC.csv` (50,000 rows)
- ✅ `data/INF_facility_coordinates.csv` (18 facilities)
- ✅ `data/INF_groups_of_diagnoses.csv`

**Process**:
1. Load synthetic patient visits
2. Assign diagnosis groups (Diarrheal Diseases, etc.)
3. Deduplicate using 3-day window rule
4. Aggregate by facility
5. Create three output tables (total, by group, pivot)

**Outputs**:
- `out/INF_diagnosis_counts_total.csv` (18 rows, 1 per facility)
  - Columns: `healthfacility`, `healthfacilitytype`, `governorate`, `total_diagnoses`, `lat`, `lon`
- `out/INF_diagnosis_counts_by_group.csv` (18×N rows, N diagnosis groups per facility)
  - Columns: `healthfacility`, `healthfacilitytype`, `governorate`, `diagnosis_group`, `diagnosis_count`, `lat`, `lon`
- `out/INF_diagnosis_counts_pivot.csv` (18 rows, wide format)
  - Columns: `healthfacility`, `healthfacilitytype`, `governorate`, `lat`, `lon`, `total_diagnoses`, [one column per diagnosis group]

**Command**:
```bash
python generate_diagnosis_counts.py
```

**Status**: ✅ Script created and ready to run

---

### **STEP 1: Climate Feature Extraction at Facilities**
**Notebook**: `GEE_Climate_Features_by_Facilities.ipynb`

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
Upload to Google Colab and run GEE_Climate_Features_by_Facilities.ipynb
```

**Status**: ⚠️ Requires running GEE notebook (output not in repo due to generation requirement)

---

### **STEP 2: HSA Delineation**
**Notebook**: `HSA_v5_FINAL_PENALTY_BASED.ipynb`

**Purpose**: Delineate Hospital Service Areas using gravity model optimization

**Inputs**:
- ✅ `data/synthetic/INF_patient_visits_SYNTHETIC.csv`
- ✅ `data/INF_facility_coordinates.csv`
- From Step 0: `out/INF_diagnosis_counts_pivot.csv` ⚠️
- From Step 1: `INF_Hospitals_Climate_Features_with_clusters.csv` ⚠️
- ✅ `data/adm_boundaries/Jordan_governorates_simplified20m.gpkg`

**Process**:
1. Load synthetic patient visits and diagnosis counts
2. Load facility coordinates and merge with climate data
3. Run penalty-based gravity model optimization with 5 different modes:
   - `fewest`: Minimize # HSAs (15-20 HSAs)
   - `footprint`: Maximize coverage (25-35 HSAs)
   - `distance`: Minimize travel distance (15-20 HSAs)
   - `governorate_tau_coverage`: 60% coverage per governorate (18-25 HSAs)
   - `governorate_fewest`: 1 HSA per governorate + FEWEST (12-20 HSAs)
4. For each mode, select anchor facilities and assign service radii
5. Export HSA boundaries as GeoJSON

**Outputs** (5 files, one per mode):
- `out/INF_fewest_hsas_v2.geojson` (~16 features)
- `out/INF_footprint_hsas_v2.geojson` (~30 features)
- `out/INF_distance_hsas_v2.geojson` (~18 features)
- `out/INF_governorate_tau_coverage_hsas_v2.geojson` (~20 features)
- `out/INF_governorate_fewest_hsas_v2.geojson` (~15 features)

Each GeoJSON contains:
  - Columns: `FacilityName`, `geometry` (Point), `service_radius_km`, `Total` (patient volume), `composite_score`, `climate_k`

**Command**:
```
jupyter notebook HSA_v5_FINAL_PENALTY_BASED.ipynb
```

**Status**: ⚠️ Ready to run after Steps 0-1 complete (outputs not in repo due to size)

---

### **STEP 3: Weekly Climate Extraction by HSA**
**Notebook**: `GEE_HSA_Weekly_Climate_Lagged.ipynb`

**Purpose**: Extract weekly climate time series for each HSA polygon

**Inputs**:
- From Step 2: HSA boundaries (e.g., `out/INF_footprint_hsas_v2.geojson`) ⚠️
- GEE datasets: CHIRPS (precipitation), ERA5-Land (temperature/humidity), TerraClimate (water balance)

**Process**:
1. Upload HSA polygon boundaries to Google Earth Engine
2. For each HSA, extract weekly climate aggregates (2022-06-27 to 2024-01-29 = 84 weeks)
3. Compute lagged variables for each day-lag (d-1, d-2, d-3, d-5, d-7, d-10, d-14)
4. Compute weekly aggregates (mean, sum, max)
5. Export 6 CSV files per HSA × 18 HSAs = 108 files total

**Outputs** (108 CSV files, ~50-100 MB total):
```
out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/
├── HSA_[Hospital]_precip_lags.csv               (18 files, 84 rows each)
├── HSA_[Hospital]_tempdew_wind_lags.csv         (18 files, 84 rows each)
├── HSA_[Hospital]_evapERA5_lags.csv             (18 files, 84 rows each)
├── HSA_[Hospital]_soilmoistERA5_lags.csv        (18 files, 84 rows each)
├── HSA_[Hospital]_water_balance.csv             (18 files, 84 rows each)
└── HSA_[Hospital]_elevation_by_week.csv         (18 files, 84 rows each)
```

Each CSV structure:
- Rows: 84 (one per week)
- Columns: `FacilityName`, `week_start`, [climate variables with lags like `P_mean_d-1`, `T_max_week`, etc.]

**Command**:
```
Upload to Google Colab and run GEE_HSA_Weekly_Climate_Lagged.ipynb
```

**Status**: ⚠️ Requires running GEE notebook with Step 2 outputs (not in repo due to size)

---

### **STEP 4: Generate Weekly Disease Counts by HSA** ✅
**Script**: `generate_weekly_disease_counts.py` ✅

**Purpose**: Aggregate synthetic patient visits to weekly disease counts by HSA

**Inputs**:
- ✅ `data/synthetic/INF_patient_visits_SYNTHETIC.csv`
- From Step 2: HSA boundaries (e.g., `out/INF_footprint_hsas_v2.geojson`) ⚠️

**Process**:
1. Load synthetic patient visits
2. Filter to diarrheal diseases (keywords: diarrhea, gastroenteritis, etc.)
3. Assign each visit to Monday-anchored week
4. Map facility to HSA (anchor facilities → own HSA)
5. Aggregate by HSA × week
6. Create complete HSA-week grid (fill zeros for weeks with no cases)

**Outputs**:
- `out/INF_footprint_weekly_diarrheal.csv` (1,512 rows = 18 HSAs × 84 weeks)
  - Columns: `hsa_id`, `week_start`, `diarrheal_count`, `diarrheal_count_adjusted`
  - Date range: 2019-01-07 to 2024-01-29 (Monday-anchored weeks)

**Command**:
```bash
python generate_weekly_disease_counts.py out/INF_footprint_hsas_v2.geojson
```

**Status**: ✅ Script created and ready to run after Step 2

---

### **STEP 5: ML Dataset Preparation**
**Script**: `prepare_ml_dataset.py` ✅

**Purpose**: Merge climate and disease data, engineer features, create train/val/test splits

**Inputs**:
- From Step 3: `out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/*.csv` (108 files) ⚠️
- From Step 4: `out/INF_footprint_weekly_diarrheal.csv` ⚠️

**Process**:
1. Load and merge 108 climate CSV files (6 types × 18 HSAs)
2. Merge with weekly disease counts on `hsa_id` + `week_start`
3. Feature engineering: temporal features (week, month, season), interactions
4. Feature selection: Reduce 144 → 33 climate features using correlation
5. Temporal train/validation/test split (weeks 1-18, 19-22, 23-84)

**Outputs**:
- `out/modeling/modeling_dataset_full.csv` (1,512 rows, 40 columns)
  - Columns: `hsa_id`, `week_start`, [33 climate features], `diarrheal_count_adjusted`, temporal features
- `out/modeling/modeling_dataset_train.csv` (288 rows = 18 HSAs × 16 weeks)
- `out/modeling/modeling_dataset_val.csv` (72 rows = 18 HSAs × 4 weeks)
- `out/modeling/modeling_dataset_test.csv` (1,152 rows = 18 HSAs × 64 weeks)
- `out/modeling/modeling_dataset_metadata.json` (feature descriptions, stats)

**Command**:
```bash
python prepare_ml_dataset.py
```

**Status**: ✅ Script exists, ⚠️ requires Steps 3-4 outputs

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
- `out/modeling/results_improved/improved_model_comparison.csv` (25 rows, 1 per model)
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
1. **STEP 0**: Diagnosis count script created (`generate_diagnosis_counts.py`)
2. **STEP 4**: Weekly disease count script created (`generate_weekly_disease_counts.py`)
3. **STEP 5**: ML dataset prep script exists (`prepare_ml_dataset.py`)
4. **STEP 6**: ML modeling script exists (`train_improved_models.py`)
5. **Synthetic Data**: All source data files present
6. **Documentation**: MODELING_METHODS.md provides methodology

### ⚠️ Requires Running (GEE or Notebooks)
1. **STEP 1**: Run GEE notebook to extract climate features at facilities
2. **STEP 2**: Run HSA optimization notebook to generate HSA boundaries
3. **STEP 3**: Run GEE notebook to extract weekly climate by HSA

### 🔄 Dependency Chain

**To reproduce the full workflow:**

```
START → STEP 0 (run script) → STEP 1 (run GEE) → STEP 2 (run notebook)
  ↓
STEP 3 (run GEE) + STEP 4 (run script)
  ↓
STEP 5 (run script) → STEP 6 (run script) → RESULTS
```

**Critical path for reviewers to understand methodology:**
1. Read MODELING_METHODS.md (methodology)
2. Examine scripts (generate_*.py, prepare_*.py, train_*.py)
3. See workflow diagram (this document)
4. Optionally run Steps 0-2 with synthetic data to see HSA delineation
5. GEE steps demonstrate climate extraction (reviewers may not have GEE access)

---

## Quick Start Guide for Reviewers

### Minimal Reproducibility (Option A - Recommended)

**Goal**: Understand the workflow without requiring GEE access

**Steps**:
1. **Examine synthetic data**: `data/synthetic/INF_patient_visits_SYNTHETIC.csv`
2. **Run diagnosis counts**: `python generate_diagnosis_counts.py`
3. **Run GEE notebook** (or use provided climate file): Extract climate features
4. **Run HSA optimization**: `jupyter notebook HSA_v5_FINAL_PENALTY_BASED.ipynb`
5. **Review outputs**: HSA boundaries in `out/*.geojson`
6. **Read methodology**: `MODELING_METHODS.md`

**Result**: Understand HSA delineation and see how synthetic data flows through the system

### Full Reproducibility (Option B)

**Goal**: Reproduce entire pipeline including ML models

**Additional Steps**:
7. **Run GEE climate extraction**: Extract weekly climate by HSA (requires GEE access)
8. **Run disease aggregation**: `python generate_weekly_disease_counts.py out/INF_footprint_hsas_v2.geojson`
9. **Prepare ML dataset**: `python prepare_ml_dataset.py`
10. **Train models**: `python train_improved_models.py`
11. **Review results**: Best model R² = 0.526 in `out/modeling/results_improved/`

**Result**: Complete end-to-end reproduction from synthetic data → ML predictions

---

## File Reference Table

| File | Step | Type | Size | In Repo? | How to Generate |
|------|------|------|------|----------|-----------------|
| `INF_patient_visits_SYNTHETIC.csv` | Input | Data | 5 MB | ✅ Yes | N/A (provided) |
| `INF_facility_coordinates.csv` | Input | Data | 1 KB | ✅ Yes | N/A (provided) |
| `INF_diagnosis_counts_pivot.csv` | 0 | Output | 5 KB | ⚠️ No | Run `generate_diagnosis_counts.py` |
| `INF_Hospitals_Climate_Features_with_clusters.csv` | 1 | Output | 10 KB | ⚠️ No | Run GEE notebook |
| `INF_footprint_hsas_v2.geojson` | 2 | Output | 50 KB | ⚠️ No | Run HSA notebook |
| `HSA_*_precip_lags.csv` (×18) | 3 | Output | 50 MB | ⚠️ No | Run GEE notebook |
| `INF_footprint_weekly_diarrheal.csv` | 4 | Output | 50 KB | ⚠️ No | Run `generate_weekly_disease_counts.py` |
| `modeling_dataset_train.csv` | 5 | Output | 20 KB | ⚠️ No | Run `prepare_ml_dataset.py` |
| `improved_model_comparison.csv` | 6 | Output | 5 KB | ⚠️ No | Run `train_improved_models.py` |

---

**Document Status**: Complete
**Last Updated**: 2024-12-28
**Next Action**: Run `python generate_diagnosis_counts.py` to begin workflow
