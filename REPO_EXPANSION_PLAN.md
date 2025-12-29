# Repository Expansion Plan: Additional Files Needed

**Date**: 2024-12-28
**Purpose**: Checklist of files to copy from `hsa_algo/` to `jordan-hsa-optimization/` for complete ML modeling workflow

---

## ✅ Already Added

- [x] `train_improved_models.py` - ML modeling script
- [x] `prepare_ml_dataset.py` - Dataset preparation script
- [x] `MODELING_METHODS.md` - Condensed modeling documentation
- [x] `requirements.txt` - Updated with xgboost and lightgbm

---

## 📊 Data Files Required

### Priority 1: Essential Data (Required for Scripts to Run)

#### Climate Data Files (108 files)
**Location**: `hsa_algo/out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/`

**Files needed**:
- `HSA_[Hospital]_precip_lags.csv` (18 files)
- `HSA_[Hospital]_tempdew_wind_lags.csv` (18 files)
- `HSA_[Hospital]_evapERA5_lags.csv` (18 files)
- `HSA_[Hospital]_soilmoistERA5_lags.csv` (18 files)
- `HSA_[Hospital]_water_balance.csv` (18 files)
- `HSA_[Hospital]_elevation_by_week.csv` (18 files)

**Size**: ~50-100 MB (need to check actual size)

**Action**:
```bash
mkdir -p jordan-hsa-optimization/out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/
cp hsa_algo/out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/*.csv \
   jordan-hsa-optimization/out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/
```

**Alternative**: If files are too large for GitHub:
- Create a sample dataset with 2-3 HSAs instead of all 18
- Document where full data can be obtained
- Add data/ to .gitignore

#### Disease Surveillance Data
**Location**: `hsa_algo/out/INF_footprint_weekly_diarrheal.csv`

**Description**: Weekly diarrheal disease counts by HSA (2019-2024)

**Privacy Note**:
- If this contains real patient counts, consider:
  - Aggregating to higher levels
  - Adding noise/perturbation
  - Using synthetic data matching statistical properties

**Action**:
```bash
mkdir -p jordan-hsa-optimization/out/
cp hsa_algo/out/INF_footprint_weekly_diarrheal.csv \
   jordan-hsa-optimization/out/
```

---

### Priority 2: Example Outputs (Good for Reviewers)

#### Prepared Datasets
**Location**: `hsa_algo/out/modeling/`

**Files**:
- `modeling_dataset_full.csv` - Complete merged dataset (1,512 rows × 40 cols)
- `modeling_dataset_train.csv` - Training set (288 rows)
- `modeling_dataset_val.csv` - Validation set (36 rows)
- `modeling_dataset_test.csv` - Test set (1,080 rows)
- `modeling_dataset_metadata.json` - Feature descriptions & stats

**Purpose**:
- Reviewers can skip Step 1 (data prep) and go straight to modeling
- Demonstrates expected data format
- Shows feature engineering results

**Action**:
```bash
mkdir -p jordan-hsa-optimization/out/modeling/
cp hsa_algo/out/modeling/modeling_dataset_*.{csv,json} \
   jordan-hsa-optimization/out/modeling/
```

#### Model Results
**Location**: `hsa_algo/out/modeling/results_improved/`

**Files**:
- `improved_model_comparison.csv` - Performance metrics for 25 models

**Purpose**: Shows expected output, validates R² = 0.526 result

**Action**:
```bash
mkdir -p jordan-hsa-optimization/out/modeling/results_improved/
cp hsa_algo/out/modeling/results_improved/improved_model_comparison.csv \
   jordan-hsa-optimization/out/modeling/results_improved/
```

---

### Priority 3: Supporting Documentation (Optional)

#### Additional Documentation Files
**Location**: `hsa_algo/`

**Files to consider**:
- `MODELING_WORKFLOW_DOCUMENTATION.md` (1579 lines - complete reference)
  - **Decision**: Already have condensed version (MODELING_METHODS.md)
  - **Action**: Skip or add as supplementary reference

- `ML_DATASET_DOCUMENTATION.md`
  - **Check**: Does this add value beyond MODELING_METHODS.md?
  - **Action**: Review and decide

- `ML_MODELING_PLAN.md`
  - **Decision**: Historical planning document, not needed for final repo
  - **Action**: Skip

**Recommendation**: Keep documentation minimal. MODELING_METHODS.md is sufficient.

---

## 🔧 Additional Files to Create

### .gitignore
**Purpose**: Prevent committing large data files or outputs

**Content**:
```gitignore
# Large data files
out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/
*.csv
!data/synthetic/*.csv

# Model outputs
out/modeling/results*/
*.pkl
*.joblib
*.h5

# Python
__pycache__/
*.py[cod]
*$py.class
.ipynb_checkpoints/

# Environment
.env
.venv
venv/
```

**Action**:
```bash
# Create or update .gitignore
nano jordan-hsa-optimization/.gitignore
```

### Example Data Download Script
**Purpose**: Document how to obtain/generate data if not in repo

**File**: `scripts/download_data.sh`

**Content**:
```bash
#!/bin/bash
# Download or generate required data files
#
# Climate data: Run GEE notebooks first
# Disease data: Contact [email] for access
```

---

## 📦 Directory Structure (Final)

```
jordan-hsa-optimization/
├── README.md ✅ (updated)
├── MODELING_METHODS.md ✅ (new)
├── SETUP_INSTRUCTIONS.md (optional - installation guide)
├── requirements.txt ✅ (updated)
├── .gitignore (create)
│
├── notebooks/
│   ├── HSA_v5_FINAL_PENALTY_BASED.ipynb ✅
│   ├── GEE_Climate_Features_by_Facilities.ipynb ✅
│   └── GEE_HSA_Weekly_Climate_Lagged.ipynb ✅
│
├── scripts/ (Python files)
│   ├── hsa_optimization.py ✅
│   ├── patient_allocation.py ✅
│   ├── prepare_ml_dataset.py ✅ (new)
│   └── train_improved_models.py ✅ (new)
│
├── data/
│   ├── adm_boundaries/ ✅
│   ├── synthetic/ ✅
│   └── README.md (data sources & licenses)
│
└── out/ (need to add)
    ├── DRIVE_CLIMATE_BY_HSA_DOWNLOAD/
    │   └── FINAL_HSA_CLIMATE/ (108 CSV files)
    ├── INF_footprint_weekly_diarrheal.csv
    └── modeling/
        ├── modeling_dataset_*.csv (example outputs)
        └── results_improved/
            └── improved_model_comparison.csv (example results)
```

---

## 🚦 Decision Points

### Question 1: Data File Size
**Check**: What's the total size of climate data (108 files)?

```bash
du -sh hsa_algo/out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/
```

**Decision**:
- If < 50 MB → Include in repo
- If 50-500 MB → Consider Git LFS or external hosting
- If > 500 MB → Definitely use external hosting (Zenodo, OSF, etc.)

### Question 2: Real vs Synthetic Disease Data
**Check**: Does `INF_footprint_weekly_diarrheal.csv` contain real patient counts?

**Decision**:
- If real → Need IRB approval or de-identification
- If already aggregated/anonymized → Can include
- If sensitive → Create synthetic version or provide access instructions

### Question 3: Repository Size Limits
**GitHub Limits**:
- File size: < 100 MB per file
- Repo size: < 1 GB recommended
- Use Git LFS for files > 50 MB

**Check total size**:
```bash
du -sh jordan-hsa-optimization/
```

---

## 📋 Recommended Next Steps

### Step 1: Check Data Sizes
```bash
# Check climate data size
du -sh hsa_algo/out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/

# Check disease data size
du -sh hsa_algo/out/INF_footprint_weekly_diarrheal.csv

# Check prepared datasets size
du -sh hsa_algo/out/modeling/
```

### Step 2: Privacy Review
- Review disease data for patient privacy
- Confirm aggregation level is sufficient
- Document data source and permissions

### Step 3: Selective Copy
Based on sizes from Step 1, copy files:

**Option A: Small datasets (< 50 MB total)**
```bash
# Copy everything
./copy_all_data.sh
```

**Option B: Medium datasets (50-500 MB)**
```bash
# Copy sample data (3 HSAs instead of 18)
# Create external data archive
# Add download instructions
```

**Option C: Large datasets (> 500 MB)**
```bash
# Copy only example outputs
# Document data generation process
# Host full data externally (Zenodo, OSF)
```

### Step 4: Create .gitignore
```bash
cat > jordan-hsa-optimization/.gitignore << 'EOF'
# Large data files (get via external source)
out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/

# Generated outputs
out/modeling/results*/
*.pkl

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
EOF
```

### Step 5: Test Workflow
```bash
cd jordan-hsa-optimization/
python prepare_ml_dataset.py  # Should work if data files present
python train_improved_models.py  # Should reproduce R² = 0.526
```

### Step 6: Document Data Access
Update README.md with data availability section:
- Where to get climate data (GEE exports or contact)
- Where to get disease data (contact info or synthetic version)
- Expected data format

---

## 🎯 Minimal Viable Addition (Quickest Path)

If you need to share with reviewers ASAP, minimum required:

1. ✅ Scripts already added:
   - `prepare_ml_dataset.py`
   - `train_improved_models.py`
   - `MODELING_METHODS.md`

2. **Add example outputs only** (small files, ~1-5 MB):
   ```bash
   mkdir -p jordan-hsa-optimization/out/modeling/results_improved/
   cp hsa_algo/out/modeling/results_improved/improved_model_comparison.csv \
      jordan-hsa-optimization/out/modeling/results_improved/
   ```

3. **Update README** ✅ (already done)

4. **Add data access note** in README:
   ```markdown
   ## Data Availability

   Due to file size constraints and privacy considerations, the full climate and
   disease datasets are not included in this repository.

   - **Climate data**: Generated using `GEE_HSA_Weekly_Climate_Lagged.ipynb`
   - **Disease data**: Contact [email] for access or use synthetic version
   - **Prepared datasets**: Available upon request or regenerate using scripts

   Example model results are provided in `out/modeling/results_improved/`
   ```

This minimal approach lets reviewers:
- Read the methodology (MODELING_METHODS.md)
- Inspect the code (train_improved_models.py)
- See the results (improved_model_comparison.csv)
- Understand the workflow (README.md)

Without requiring large data files.

---

## Summary

**Immediate Priority**: Add example results file (~1 MB)
**Short-term**: Determine data file sizes and privacy status
**Medium-term**: Add data files or external hosting instructions
**Documentation**: Already complete ✅

**Next Action**: Run the commands in Step 1 (Check Data Sizes) above.
