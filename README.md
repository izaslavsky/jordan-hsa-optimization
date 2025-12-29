# Hospital Service Area (HSA) Optimization and Climate-Health Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Code and data accompanying the research paper on delineating Hospital Service Areas (HSAs) using patient trajectory data and analyzing climate-health relationships in Jordan.

## 📁 Repository Contents

### Jupyter Notebooks

1. **`HSA_v5_FINAL_PENALTY_BASED.ipynb`** - Main HSA optimization workflow
   - Delineates Hospital Service Areas using multi-objective greedy optimization
   - Allocates patient populations to healthcare facilities
   - Generates optimized HSA boundaries aligned with administrative divisions

2. **`GEE_Climate_Features_by_Facilities.ipynb`** - Climate data extraction (Google Earth Engine)
   - Extracts climate variables around healthcare facility buffers
   - Sources: CHIRPS (precipitation), ERA5-Land (temperature), TerraClimate (water balance)
   - Creates facility-level climate datasets for health analysis

3. **`GEE_HSA_Weekly_Climate_Lagged.ipynb`** - Weekly climate aggregation (Google Earth Engine)
   - Aggregates climate data to weekly temporal resolution
   - Computes lagged climate variables (1-20 day lags)
   - Prepares climate features for epidemiological modeling

### Python Scripts

#### HSA Optimization
- **`hsa_optimization.py`** - Core HSA optimization algorithm
  - HSAOptimizer class with penalty-based gravity model
  - Population allocation and boundary delineation functions

- **`patient_allocation.py`** - Patient trajectory analysis
  - Gravity model for patient allocation to facilities
  - Distance-based travel impedance calculations

#### Machine Learning Modeling
- **`prepare_ml_dataset.py`** - Dataset preparation for ML modeling
  - Merges climate data (108 files) with disease surveillance data
  - Feature engineering and selection (reduces 144 → 33 climate features)
  - Creates train/validation/test splits with temporal separation

- **`train_improved_models.py`** - ML models for disease prediction
  - Predicts weekly diarrheal disease counts from climate + disease history
  - Autoregressive features (lag-1, lag-2) + top 10 climate variables
  - Trains 5 models: Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost
  - Best performance: R² = 0.526 (Gradient Boosting)

### Data Files

All data files use **synthetic patient data** to protect privacy. Boundary and coordinate data are real.

#### Administrative Boundaries (GeoPackage format)
- `data/adm_boundaries/Jordan_governorates_simplified20m.gpkg` - 12 governorates
- `data/adm_boundaries/Jordan_districts_simplified20m.gpkg` - 51 districts
- `data/adm_boundaries/Jordan_subdistricts_simplified20m.gpkg` - 89 subdistricts
- `data/jordan_boundary.gpkg` - National boundary

Source: OpenStreetMap, manually aligned and verified

#### Healthcare Facilities
- `data/INF_facility_coordinates.csv` - Infectious disease facility locations (lat/lon)
- `data/NCD_facility_coordinates.csv` - Non-communicable disease facility locations (lat/lon)

#### Diagnosis Classification
- `data/INF_groups_of_diagnoses.csv` - ICD code groupings for infectious diseases
- `data/NCD_groups_of_diagnoses.csv` - ICD code groupings for non-communicable diseases

#### Synthetic Patient Data
- `data/synthetic/INF_patient_visits_SYNTHETIC.csv` - Synthetic infectious disease visits
- `data/synthetic/NCD_patient_visits_SYNTHETIC.csv` - Synthetic non-communicable disease visits

**Note**: Synthetic data preserves statistical properties of real data (distributions, correlations) but contains no actual patient information.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Earth Engine account (for climate extraction notebooks)
- GDAL/QGIS (optional, for viewing GeoPackage files)

### Installation

```bash
# Clone repository
git clone https://github.com/izaslavsky/HSA_algo_public.git
cd HSA_algo_public

# Install Python dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine (for GEE notebooks)
earthengine authenticate
```

### Running the Notebooks

**1. HSA Optimization**

Open `HSA_v5_FINAL_PENALTY_BASED.ipynb` in Jupyter:

```bash
jupyter notebook HSA_v5_FINAL_PENALTY_BASED.ipynb
```

This notebook:
- Loads synthetic patient visit data
- Runs penalty-based gravity model optimization
- Generates HSA boundaries aligned with administrative districts
- Outputs optimized patient allocation results

**2. Climate Data Extraction**

For facility-level climate extraction:

```upload to Google Colab and run
GEE_Climate_Features_by_Facilities.ipynb
```

For weekly lagged climate variables:

```upload to Google Colab and run
GEE_HSA_Weekly_Climate_Lagged.ipynb
```

**Note**: GEE notebooks require Google Earth Engine authentication and may take significant time to extract climate data.

**3. Machine Learning Modeling**

After completing steps 1-3 above, run the ML modeling pipeline:

```bash
# Step 1: Prepare ML dataset (merges climate + disease data)
python prepare_ml_dataset.py

# Step 2: Train models
python train_improved_models.py
```

**Outputs**:
- `out/modeling/modeling_dataset_train.csv` - Training set (288 samples)
- `out/modeling/modeling_dataset_val.csv` - Validation set (36 samples)
- `out/modeling/modeling_dataset_test.csv` - Test set (1,080 samples)
- `out/modeling/results_improved/improved_model_comparison.csv` - Model performance (25 models)

**Expected Results**: Best model achieves R² = 0.526 (Gradient Boosting with 12 features)

For detailed methodology, see **[MODELING_METHODS.md](MODELING_METHODS.md)**

## 📊 Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Climate Extraction (GEE_Climate_Features_by_Facilities)  │
│    Input:  Facility coordinates                             │
│    Output: Climate variables by facility buffer zones       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. HSA Optimization (HSA_v5_FINAL_PENALTY_BASED.ipynb)      │
│    Input:   Patient visits + facility coordinates           │
│    Output: Optimized HSA boundaries                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Weekly Climate Aggregation (GEE_HSA_Weekly_Climate)      │
│    Input:  HSA boundaries + climate data                    │
│    Output: Weekly lagged climate features by HSA            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. ML Dataset Preparation (prepare_ml_dataset.py)           │
│    Input:  Weekly climate (108 files) + disease counts      │
│    Output: Train/validation/test datasets                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. ML Modeling (train_improved_models.py)                   │
│    Input:  Prepared datasets (climate + AR features)        │
│    Output: Trained models + predictions (R² = 0.526)        │
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

The multi-objective greedy algorithm optimizes Hospital Service Areas by:

1. **Gravity model**: Patients allocated to facilities based on distance impedance
2. **Penalty constraints**: Penalizes HSAs that cross administrative boundaries
3. **Iterative optimization**: Adjusts facility catchment areas to minimize boundary violations
4. **Validation**: Compares optimized HSAs against actual patient trajectories

### Climate Feature Engineering

Climate variables extracted from Google Earth Engine:
- **CHIRPS**: Daily precipitation (mm)
- **ERA5-Land**: Temperature (2m air temp, min/max), humidity (dew point)
- **TerraClimate**: Soil moisture, water deficit, evapotranspiration

Temporal aggregation:
- Weekly means/sums computed from daily data
- Lagged features: 1-20 day lags to capture delayed health effects
- Spatial aggregation: Mean values within facility buffer zones or HSA polygons

### Machine Learning Modeling

Predictive models for weekly diarrheal disease incidence using climate variables and disease history:

**Key Innovation**: Autoregressive features (previous week's disease counts) combined with climate variables

**Approach**:
1. **Dataset Preparation** (`prepare_ml_dataset.py`)
   - Merges 108 climate CSV files with disease surveillance data
   - Feature selection: Reduces 144 → 33 climate features to prevent overfitting
   - Temporal train/validation/test split (no data leakage)

2. **Model Training** (`train_improved_models.py`)
   - **Feature Sets**: 5 combinations tested (AR-only, AR+climate, AR+temporal+climate)
   - **Models**: Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost
   - **Best Model**: Gradient Boosting with 12 features (2 AR + 10 climate)
   - **Performance**: Validation R² = 0.526, RMSE = 12.66 cases

**Feature Importance**:
- Autoregressive features (lag-1, lag-2): ~93% of predictive power
- Climate variables (top 10): ~7% additional improvement
- Top climate predictors: Water deficit, temperature range, heat stress, humidity

**Operational Use**:
- 1-week ahead forecasting of disease burden
- Requires: Previous 2 weeks disease counts + current week climate data
- Expected performance: R² ≈ 0.52, RMSE ≈ 12-13 cases

For detailed methodology, see **[MODELING_METHODS.md](MODELING_METHODS.md)**

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@software{hsa_climate_health_2024,
  title = {Hospital Service Area Optimization and Climate-Health Analysis},
  author = {{Ilya Zaslavsky}},
  year = {2025},
  url = {https://github.com/izaslavsky/HSA_algo_public},
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
