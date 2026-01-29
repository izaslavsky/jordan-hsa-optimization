# Hospital Service Area (HSA) Optimization and Climate-Health Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Code and data accompanying the research paper on delineating Hospital Service Areas (HSAs) using patient trajectory data and analyzing climate-health relationships in Jordan.

## 📁 Repository Contents

### Jupyter Notebooks

#### Core HSA Delineation

1. **`GEE_Climate_Features_by_Facilities.ipynb`** - Climate data extraction (Google Earth Engine from Google Colab or local environment)
   - Extracts climate variables around healthcare facility buffers
   - Sources: CHIRPS (precipitation), ERA5-Land (temperature), TerraClimate (water balance)
   - Creates facility-level climate datasets and computes climate clusters for HSA delineation and health analysis
   - **Local notebook available**: runs without Colab, but requires Earth Engine auth and (optionally) Drive API OAuth if you use the automated export/download steps

2. **`HSA_v6_FINAL.ipynb`** - Main HSA optimization workflow
   - Delineates Hospital Service Areas using unified scoring system with mode-specific weight profiles
   - Five optimization modes: FEWEST, FOOTPRINT, DISTANCE, GOVERNORATE_TAU_COVERAGE, GOVERNORATE_FEWEST
   - Pure score-driven optimization with no artificial facility count constraints
   - Generates optimized HSA boundaries with composite score tracking


#### Disease Modeling Pipeline

3. **`GEE_HSA_Weekly_Climate_Lagged.ipynb`** - Weekly climate aggregation (Google Earth Engine from Google Colab or local environment)
   - Aggregates climate data to weekly temporal resolution by HSA
   - Computes lagged climate variables (1-20 day lags)
   - Exports 6 CSV files per HSA (precipitation, temperature, soil moisture, evaporation, water balance, elevation)
   - **Output location**: If running on Google Colab, users must download exported CSVs from Google Drive to `out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/`
   - **Local notebook available**: runs without Colab, but requires Earth Engine auth and (optionally) Drive API OAuth if you use the automated export/download steps

4. **`Patient_Allocation_for_Modeling.ipynb`** - Patient allocation for overlapping HSAs
   - Eliminates double/triple counting when HSAs overlap
   - Gravity model assigns each population pixel to exactly ONE facility
   - **Required for**: Computing disease rates and weekly disease counts by HSA
   - Outputs: `pixel_allocations_{NETWORK}_{MODE}.csv` and `hsa_allocated_patients_{NETWORK}_{MODE}.csv`

5. **`Generate_Modeling_Dataset.ipynb`** - Complete modeling dataset generation
   - Orchestrates the complete dataset preparation pipeline
   - Calls existing Python scripts to: (1) generate weekly disease counts, (2) merge climate + disease data
   - Creates final dataset ready for machine learning modeling
   - Outputs: `{NETWORK}_{MODE}_modeling_dataset.csv` with feature metadata

6. **`compare_delineations.ipynb`** - Compare delineation modes
   - Runs `compare_spatial_methods_v2.py` with notebook-specified inputs
   - Generates a comparison table across methods

7. **`run_climate_health_modeling.ipynb`** - Run climate-health modeling
   - Runs `climate_health_modeling.py` with notebook-specified inputs
   - Uses `out/modeling/{NETWORK}_{MODE}_modeling_dataset.csv` as input
   - Writes outputs to `out/modeling/`

### Python Scripts

#### HSA Optimization
- **`hsa_optimization.py`** - Core HSA optimization algorithm
  - HSAOptimizer class with unified scoring system
  - Five optimization modes (FEWEST, FOOTPRINT, DISTANCE, governorate-based)
  - Population coverage calculations and HSA boundary delineation

- **`hsa_mapping_working.py`** - HSA map generation
  - Creates professional maps with circular HSA boundaries clipped to country
  - Visualization of facilities, boundaries, and population density

- **`hsa_objective_analysis.py`** - HSA results analysis
  - Statistical analysis and visualizations for all five optimization modes

- **`patient_allocation.py`** - Patient allocation for overlapping HSAs
  - **Purpose**: Eliminates double/triple counting when HSAs overlap
  - Gravity model assigns each population pixel to exactly ONE facility
  - Includes parallel processing for 3-4x speedup on multi-core systems
  - **Required for**: Computing disease rates and weekly disease counts by HSA
  - **Not required for**: HSA boundary delineation (only needed for downstream disease modeling)

#### Disease Modeling Pipeline
- **`generate_weekly_disease_counts_adjusted.py`** - Weekly disease count aggregation
  - Aggregates patient visits by HSA and week (Monday-anchored weeks)
  - Uses gravity model to avoid double-counting facilities in multiple HSAs
  - Outputs: `{NETWORK}_{MODE}_weekly_{DISEASE}_adjusted.csv`

- **`prepare_ml_dataset.py`** - Dataset preparation for ML modeling
  - Merges climate data (6 variable types per HSA) with disease surveillance data
  - Loads climate CSVs from `out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/`
  - Creates complete modeling dataset with feature metadata
  - Outputs: `{NETWORK}_{MODE}_modeling_dataset.csv` and feature descriptions JSON

#### Climate-Health Modeling
- **`climate_health_modeling.py`** - Basic climate-health EDA and preprocessing
- **`climate_health_modeling_comprehensive.py`** - Comprehensive modeling with variable pruning
- **`climate_health_modeling_parsimonious.py`** - Theory-driven parsimonious models
- **`train_ml_models.py`** - Baseline ML model training (multiple model families)
- **`train_improved_models.py`** - Improved ML models with alternative feature sets

#### Spatial Methods Comparison
- **`compare_spatial_methods_v2.py`** - Compare HSA delineation methods
  - Compares optimized HSAs vs fixed-radius buffers, Voronoi tessellation, governorate boundaries
  - Called by `compare_delineations.ipynb` with notebook-specified parameters
  - Outputs: `{NETWORK}_spatial_methods_comparison.csv`

### Data Files

All data files use **synthetic patient data** to protect privacy. Boundary and coordinate data are real.

#### Population Raster Data (GeoTIFF format)
- `data/jor_ppp_2020_UNadj.tif` - WorldPop 2020 population density (UN-adjusted, ~10.2M total)
  - Resolution: 100m × 100m
  - Used in HSA optimization for coverage calculations
  - Source: WorldPop (www.worldpop.org)
- `data/jor_ppp_2020_constrained.tif` - WorldPop 2020 population density (constrained to settlement patterns, ~7M total)
  - Resolution: 100m × 100m
  - Alternative population dataset, used as a background population density layer for mapping
  - Source: WorldPop (www.worldpop.org)

#### Administrative Boundaries (GeoPackage format)
- `data/adm_boundaries/Jordan_governorates_simplified20m.gpkg` - 12 governorates
- `data/adm_boundaries/Jordan_districts_simplified20m.gpkg` - 51 districts
- `data/adm_boundaries/Jordan_subdistricts_simplified20m.gpkg` - 89 subdistricts
- `data/jordan_boundary.gpkg` - National boundary
- `data/jordan_governorates.gpkg` - Governorate boundaries

Source: OpenStreetMap, manually aligned and verified.

#### Synthetic Patient Data and Facility Coordinates

All patient data is synthetic to protect privacy. Files use `SYNINF_` prefix for infectious diseases and `SYNNCD_` prefix for non-communicable diseases:

- `data/SYNINF_facility_coordinates.csv` - Infectious disease facility locations (lat/lon)
- `data/SYNNCD_facility_coordinates.csv` - Non-communicable disease facility locations (lat/lon)
- `data/SYNINF_groups_of_diagnoses.csv` - ICD code groupings for infectious diseases
- `data/SYNNCD_groups_of_diagnoses.csv` - ICD code groupings for non-communicable diseases
- `data/SYNINF_patient_visits.csv` - Synthetic infectious disease visits
- `data/SYNNCD_patient_visits.csv` - Synthetic non-communicable disease visits

**Note**: Synthetic data preserves statistical properties of real data (distributions, correlations) but contains no actual patient information. The `SYN` prefix indicates synthetic data.

## 🚀 Quick Start

This repository provides a reproducible workflow for HSA delineation 
and climate-health analysis implemented as sequential Jupyter notebooks:

### Prerequisites

- Python 3.8 or higher
- Google Earth Engine account (for climate extraction notebooks)
- Google Colab access (optional; local GEE notebooks are provided)
- Local Python environment (for optimization and modeling)
- GDAL/QGIS (optional, for viewing GeoPackage files)

### Installation

```bash
# Clone repository
git clone https://github.com/izaslavsky/jordan-hsa-optimization.git
cd HSA_algo_public

# Install Python dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine (for GEE notebooks)
earthengine authenticate
```

## Google Earth Engine & Google Drive Authentication (Required for Climate Extraction)

Several notebooks in this repository rely on **Google Earth Engine (GEE)** for climate data extraction and **Google Drive** for exporting large CSV outputs.

Specifically:
- `GEE_Climate_Features_by_Facilities.ipynb`
- `GEE_HSA_Weekly_Climate_Lagged.ipynb`

These notebooks:
1. Run computations on Google Earth Engine servers
2. Export results to a Google Drive folder
3. Optionally poll Google Drive and download results locally

Because of this, **both Earth Engine and Google Drive authentication are required**.

### Summary of What You Need

| Component | Purpose | Required? |
|---------|--------|-----------|
| Google Earth Engine account | Run climate computations | ✅ Yes |
| Google Drive OAuth credentials | Detect + download exported CSVs | ✅ Yes |
| `client_secrets.json` file | Local OAuth authentication | ✅ Yes |
| `.gitignore` entry | Prevent credential leaks | ✅ Yes |

---

### Earth Engine Authentication (All Environments)

You must authenticate Earth Engine **once per machine or Colab session**.

```bash
earthengine authenticate
````

This opens a browser, prompts Google login, and stores credentials locally.
After authentication, notebooks initialize Earth Engine with:

```bash
ee.Initialize(project="ee-<your-project-id>")
````

### Google Drive Authentication (Local Execution)
Climate exports are written to Google Drive.
To automatically detect when files are ready and download them, the notebooks use the **Google Drive API**.

This requires an OAuth credentials file:

```bash
client_secrets.json
````

⚠️ **This file must never be committed to Git**.
 
**Where the credentials file goes**


Place the file at the **root of the repository**:

```bash
jordan-hsa-optimization/
├── client_secrets.json   ← required, NOT committed
├── .gitignore
├── README.md
├── notebooks/
└── ...
````

The notebooks reference it as:

```bash
CLIENT_SECRETS_PATH = "client_secrets.json"
````

### Google Colab Users

When running GEE notebooks in **Google Colab**:

* Earth Engine authentication is handled interactively
* Google Drive can be mounted instead of using OAuth

Colab users **do not need** `client_secrets.json` if they rely on:
```bash
from google.colab import drive
drive.mount('/content/drive')
```
Local users **do need it**.
 
### Security Note

* client_secrets.json grants access to your Google Drive
* Treat it like a password
* Rotate it if accidentally exposed

See **SETUP_INSTRUCTIONS.md** for step-by-step instructions on creating credentials.




## Running the Notebooks

**1. HSA Optimization**

Open `HSA_v6_FINAL.ipynb` in Jupyter:

```bash
jupyter notebook HSA_v6_FINAL.ipynb
```

This notebook:
- Runs five optimization modes with different objectives (FEWEST, FOOTPRINT, DISTANCE, and governorate-based modes)
- Uses unified scoring system with mode-specific weight profiles
- Generates HSA boundaries with composite score tracking
- Outputs optimized HSA configurations to GeoJSON files

**2. Climate Data Extraction**

For facility-level climate extraction:

```upload to Google Colab and run
GEE_Climate_Features_by_Facilities.ipynb
```

For weekly lagged climate variables:

```upload to Google Colab and run
GEE_HSA_Weekly_Climate_Lagged.ipynb
```

**Important**: After running `GEE_HSA_Weekly_Climate_Lagged.ipynb`, you must download the exported climate CSV files from your Google Drive and place them in:
```
out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/
```

The notebook exports 6 CSV files per HSA (precipitation, temperature, soil moisture, evaporation, water balance, elevation).

**Note**: GEE notebooks require Google Earth Engine authentication and may take significant time to extract climate data.

**3. Patient Allocation (Required for Disease Modeling)**

HSAs created in step 2 are **overlapping circular regions** around facilities. Before computing disease rates or weekly disease counts, you must allocate each population pixel to exactly ONE facility to prevent double/triple counting.

Open and run `Patient_Allocation_for_Modeling.ipynb`:

```bash
jupyter notebook Patient_Allocation_for_Modeling.ipynb
```

This notebook:
- Loads optimized HSA boundaries (GeoJSON output from HSA_v6_FINAL.ipynb)
- Applies gravity model to assign each population pixel to exactly one facility
- Aggregates allocated populations by HSA (no overlap)
- Saves results as CSV for disease rate calculations

**Why this is necessary**:
- HSA boundaries overlap (facility service areas are circles that intersect)
- Same person lives in multiple overlapping HSAs
- Without allocation: person counted 2-3 times when computing disease rates
- With allocation: each person assigned to ONE facility based on distance + facility size
- Result: Accurate denominators for disease rate calculations

**When to skip**: Only needed if you're doing disease modeling. Not required for HSA boundary visualization or spatial analysis.

---

**4. Generate Modeling Dataset**

After completing steps 1-3 above (HSA optimization, climate extraction, patient allocation), generate the complete modeling dataset:

```bash
jupyter notebook Generate_Modeling_Dataset.ipynb
```

This notebook orchestrates the complete dataset preparation pipeline:

1. **Checks prerequisites**:
   - HSA boundaries from step 1
   - Patient allocations from step 3
   - Climate CSVs downloaded from Google Drive (step 2)
   - Patient visit data

2. **Generates weekly disease counts**:
   - Calls `generate_weekly_disease_counts_adjusted.py`
   - Aggregates patient visits by HSA and week
   - Outputs: `{NETWORK}_{MODE}_weekly_diarrheal.csv` and `{NETWORK}_{MODE}_weekly_infectious.csv`

3. **Merges climate + disease data**:
   - Calls `prepare_ml_dataset.py`
   - Loads climate CSVs (6 variable types per HSA)
   - Merges with weekly disease counts
   - Outputs: `{NETWORK}_{MODE}_modeling_dataset.csv` with feature metadata

**Outputs**:
- Weekly disease count files in `out/`
- Complete modeling dataset ready for ML training

**Prerequisites**:
- ✅ HSA boundaries generated (HSA_v6_FINAL.ipynb)
- ✅ Patient allocation completed (Patient_Allocation_for_Modeling.ipynb)
- ✅ Climate files downloaded to `out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/`

---

**5. Climate-Health Modeling**

After completing step 4 (dataset generation), run climate-health modeling:

```bash
jupyter notebook run_climate_health_modeling.ipynb
```

This notebook orchestrates the modeling pipeline by calling:
- `climate_health_modeling_comprehensive.py` - Variable pruning and model comparison
- `climate_health_modeling_parsimonious.py` - Theory-driven small models
- `train_ml_models.py` - Baseline ML models
- `train_improved_models.py` - Alternative feature sets

**See**: `CLIMATE_HEALTH_MODELING.md` and `MODELING_RESULTS.md` for methodology and results.

---

**6. Compare Delineation Methods (Optional)**

To compare the optimized HSAs against baseline spatial methods:

```bash
jupyter notebook compare_delineations.ipynb
```

This notebook calls `compare_spatial_methods_v2.py` to compare:
- Fixed-radius buffers (all facilities)
- Voronoi tessellation
- Governorate boundaries
- Optimized HSAs

**See**: `SPATIAL_METHODS_COMPARISON.md` for methodology and results.


## Synthetic Data Caveat
The included synthetic dataset enables reproduction of HSA delineation 
but is not suitable for climate-health modeling validation, as synthetic health outcomes lack temporal autocorrelation 
and climate associations present in real data.

## 📊 Workflow Overview

**Core HSA Delineation** (Steps 1-2):
```
┌─────────────────────────────────────────────────────────────┐
│ 1. Climate Extraction (GEE_Climate_Features_by_Facilities)  │
│    Input:  Facility coordinates                             │
│    Output: Climate variables by facility buffer zones       │
│    ACTION: Copy {NETWORK}_Facilities_Climate_Features_with_ │
│            clusters.csv to /data                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. HSA Optimization (HSA_v6_FINAL.ipynb)                    │
│    Input:   Facility coordinates + population raster        │
│    Output: Optimized HSA boundaries (5 modes, overlapping)  │
└─────────────────────────────────────────────────────────────┘
```

**Disease Modeling Workflow** (Steps 3-7 - requires patient allocation):
```
┌─────────────────────────────────────────────────────────────┐
│ 3. Patient Allocation (Patient_Allocation_for_Modeling)     │
│    Input:  HSA boundaries + population raster               │
│    Output: pixel_allocations_*.csv, hsa_allocated_*.csv     │
│    WHY:    HSAs overlap - prevents double counting          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Weekly Climate Aggregation (GEE_HSA_Weekly_Climate)      │
│    Input:  HSA boundaries + Google Earth Engine datasets    │
│    Output: Climate CSVs exported to Google Drive            │
│    ACTION: Download the exported CSVs to                                 │
│       out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE   │
└─────────────────────────────────────────────────────────────┘

                              │
                              ▼
┌═════════════════════════════════════════════════════════════┐
║ 5. Generate Modeling Dataset (orchestration notebook)       ║
║    Calls existing Python scripts in sequence:               ║
║                                                             ║
║    ┌───────────────────────────────────────────────────┐    ║
║    │ 5a. Weekly Disease Counts                         │    ║
║    │     Script: generate_weekly_disease_counts_       │    ║
║    │             adjusted.py                           │    ║
║    │     Input:  Patient allocations + visit data      │    ║
║    │     Output: Weekly disease counts per HSA         │    ║
║    └───────────────────────────────────────────────────┘    ║
║                            │                                ║
║                            ▼                                ║
║    ┌───────────────────────────────────────────────────┐    ║
║    │ 5b. Merge Climate + Disease Data                  │    ║
║    │     Script: prepare_ml_dataset.py                 │    ║
║    │     Input:  Climate CSVs + disease counts         │    ║
║    │     Output: {NETWORK}_{MODE}_modeling_dataset.csv │    ║
║    └───────────────────────────────────────────────────┘    ║
╚═════════════════════════════════════════════════════════════╝
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Climate-Health Modeling (run_climate_health_modeling)    │
│    Input:  Modeling dataset (climate + AR features)         │
│    Output: Trained models + metrics (out/modeling/)         │
└─────────────────────────────────────────────────────────────┘
```

## 🗺️ Administrative Boundaries

This repository uses **OpenStreetMap-derived administrative boundaries** that have been:
- Manually aligned across all three levels (governorate → district → subdistrict)
- Simplified to 20m tolerance for efficient processing (85% file size reduction)
- Verified against official Jordan government population statistics

**Why OSM?** Other commonly used boundary sources (HumData/GeoBoundaries, GADM, Stanford Earthworks) had inconsistencies between administrative levels that prevented accurate nesting of districts within governorates.

For standalone administrative boundaries with full documentation, see:
- **Repository**: [jordan-administrative-boundaries](https://github.com/izaslavsky/jordan-administrative-boundaries)

## 📖 Methodology

### HSA Delineation

The unified scoring system optimizes Hospital Service Areas using five different modes:

**Five Optimization Modes**:
1. **FEWEST**: Minimize number of HSAs while achieving 90% population coverage
2. **FOOTPRINT**: Maximize geographic diversity and spread across climate zones
3. **DISTANCE**: Minimize average travel distance from population to facilities
4. **GOVERNORATE_TAU_COVERAGE**: Achieve 90% coverage in each governorate independently
5. **GOVERNORATE_FEWEST**: One anchor per governorate + minimize total HSAs

**Key Features**:
- **Unified scoring formula**: Single formula with mode-specific weight profiles
- **Pure score-driven**: Number of facilities emerges from optimization, no artificial constraints
- **Adaptive radii**: Urban facilities (10km) vs rural facilities (18km)
- **Composite scoring**: Combines coverage gain, overlap penalty, climate diversity, and distance factors
- **Overlap removal**: Post-processing removes HSAs with >80% overlap

**Stopping Conditions**:
- Target coverage reached (90% for main modes)
- No more viable facilities (best score ≤ 0)
- All facilities selected (rare)

### Climate Feature Engineering

Climate variables extracted from Google Earth Engine:
- **CHIRPS**: Daily precipitation (mm)
- **ERA5-Land**: Temperature (2m air temp, min/max), humidity (dew point)
- **TerraClimate**: Soil moisture, water deficit, evapotranspiration

Temporal aggregation:
- Weekly means/sums computed from daily data
- Lagged features: 1-20 day lags to capture delayed health effects
- Spatial aggregation: Mean values within facility buffer zones or HSA polygons

### Climate-Health Modeling

Predictive models for weekly disease incidence using climate variables and disease history.

**Approach**:
1. **Dataset Preparation** (`prepare_ml_dataset.py`)
   - Merges climate CSV files with disease surveillance data
   - Feature selection to prevent overfitting
   - Temporal train/validation/test split (no data leakage)

2. **Model Training** (`run_climate_health_modeling.ipynb`)
   - Comprehensive modeling with variable pruning (`climate_health_modeling_comprehensive.py`)
   - Parsimonious theory-driven models (`climate_health_modeling_parsimonious.py`)
   - Multiple ML families: Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost

**See** `CLIMATE_HEALTH_MODELING.md` and `MODELING_RESULTS.md` for detailed methodology and results.

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@software{hsa_climate_health_2025,
  title = {Hospital Service Area Optimization and Climate-Health Analysis},
  author = {{Ilya Zaslavsky}},
  year = {2025},
  url = {https://github.com/izaslavsky/jordan-hsa-optimization},
  note = {GitHub repository}
}
```

**Related Publication**:
> [Citation will be added upon publication]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Data Licenses**:
- Administrative boundaries: ODbL (OpenStreetMap)
- Synthetic patient data: Public domain (no real patient information)
- Climate data: See individual source licenses (CHIRPS, ERA5-Land, TerraClimate)

## 🙏 Acknowledgments

- **Administrative boundaries**: OpenStreetMap contributors
- **Climate data**:
  - CHIRPS (UC Santa Barbara Climate Hazards Center)
  - ERA5-Land (ECMWF Copernicus)
  - TerraClimate (UC Merced)
- **Google Earth Engine**: Platform for large-scale climate data extraction

## 📧 Contact

For questions or issues:
- **GitHub Issues**: [Open an issue](https://github.com/izaslavsky/HSA_algo_public/issues)
- **Email**: [Contact information]

---

**Data Privacy Notice**: This repository contains only synthetic patient data generated to match statistical properties of real data. No actual patient information is included.
