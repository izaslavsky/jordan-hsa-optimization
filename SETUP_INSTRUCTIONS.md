# Setup Instructions

This guide will help you set up your environment and run the HSA optimization and climate extraction workflows.

## Prerequisites

### Required Software
- **Python 3.8 or higher** ([Download](https://www.python.org/downloads/))
- **Git** ([Download](https://git-scm.com/downloads))
- **Jupyter Notebook** (installed via requirements.txt)

### Optional Software
- **QGIS** or **ArcGIS** - For viewing/editing GeoPackage boundary files ([QGIS Download](https://qgis.org/download/))
- **Google Earth Engine Account** - Required only for climate extraction notebooks ([Sign up](https://earthengine.google.com/signup/))

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/izaslavsky/HSA_algo_public.git
cd HSA_algo_public
```

### 2. Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected installation time**: 5-10 minutes depending on internet speed

### 4. Google Earth Engine Authentication (Optional)

Only needed if you plan to run the climate extraction notebooks (`GEE_Climate_Features_by_Facilities.ipynb` or `GEE_HSA_Weekly_Climate_Lagged.ipynb`).

```bash
earthengine authenticate
```

This will:
1. Open a browser window for Google authentication
2. Prompt you to authorize Earth Engine
3. Generate an authentication token

**Note**: You must have a Google Earth Engine account. Sign up at https://earthengine.google.com/signup/

## Running the Notebooks

### Start Jupyter Notebook Server

```bash
jupyter notebook
```

This will open Jupyter in your web browser at `http://localhost:8888`

### Notebook Execution Order

**For HSA optimization only:**
1. Open `HSA_v5_FINAL_PENALTY_BASED.ipynb`
2. Run all cells sequentially (Cell → Run All)

**For complete climate-health workflow:**
1. `HSA_v5_FINAL_PENALTY_BASED.ipynb` - Delineate HSA boundaries
2. `GEE_Climate_Features_by_Facilities.ipynb` - Extract climate data by facility
3. `GEE_HSA_Weekly_Climate_Lagged.ipynb` - Aggregate weekly climate features

### Expected Runtime

- **HSA Optimization**: 5-15 minutes (depends on dataset size and optimization parameters)
- **Climate Extraction (GEE)**: 30 minutes to 2+ hours (depends on spatial extent, temporal range, and GEE server load)

## Data Files

All necessary data files are included in the `data/` directory:

### Administrative Boundaries
- `data/adm_boundaries/*.gpkg` - Governorate, district, and subdistrict boundaries
- `data/jordan_boundary.gpkg` - National boundary

**Viewing boundaries**: Open `.gpkg` files in QGIS or use GeoPandas:
```python
import geopandas as gpd
districts = gpd.read_file('data/adm_boundaries/dis_simpl_20m.gpkg')
districts.plot()
```

### Healthcare Facilities
- `data/INF_facility_coordinates.csv` - Infectious disease facilities (lat/lon)
- `data/NCD_facility_coordinates.csv` - Non-communicable disease facilities (lat/lon)

### Synthetic Patient Data
- `data/synthetic/INF_patient_visits_SYNTHETIC.csv` - Synthetic infectious disease visits
- `data/synthetic/NCD_patient_visits_SYNTHETIC.csv` - Synthetic non-communicable disease visits

**Privacy Note**: All patient data is 100% synthetic. No real patient information is included.

## Troubleshooting

### Common Issues

**Issue 1: "ModuleNotFoundError: No module named 'geopandas'"**

**Solution**: Install geopandas and its dependencies:
```bash
pip install geopandas
```

If this fails, install GDAL first (Windows users may need pre-built wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs/)

---

**Issue 2: "earthengine.ee_exception.EEException: Please authorize access to your Earth Engine account"**

**Solution**: Run Earth Engine authentication:
```bash
earthengine authenticate
```

---

**Issue 3: Google Earth Engine tasks timeout or fail**

**Solution**:
- Check your GEE asset quota (may be full)
- Reduce spatial or temporal extent in notebook parameters
- Split large exports into smaller batches
- Check Earth Engine status page: https://status.earthengine.google.com/

---

**Issue 4: Notebook kernel dies during optimization**

**Solution**:
- Increase available RAM (close other applications)
- Reduce dataset size in notebook parameters
- Use a subset of patient data for testing

---

**Issue 5: Cannot open GeoPackage files**

**Solution**:
- Install GDAL: `pip install GDAL` (or use QGIS which includes GDAL)
- Check file path is correct (use absolute paths if relative paths fail)
- Verify file integrity: `gpd.read_file('file.gpkg')` should not raise errors

## Verifying Installation

Run this Python script to verify your environment:

```python
import sys
print(f"Python version: {sys.version}")

# Check core packages
import numpy
import pandas
import geopandas
import matplotlib
import earthengine

print(f"NumPy: {numpy.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"GeoPandas: {geopandas.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Earth Engine: {earthengine.__version__}")

# Test data loading
districts = geopandas.read_file('data/adm_boundaries/dis_simpl_20m.gpkg')
print(f"\nSuccessfully loaded {len(districts)} districts")

print("\n✅ Environment setup complete!")
```

Expected output should show no errors and display package versions.

## Performance Notes

### Memory Requirements
- **HSA Optimization**: 4-8 GB RAM recommended
- **Climate Extraction**: 8-16 GB RAM recommended (GEE handles heavy processing server-side)

### Disk Space
- **Repository**: ~50 MB
- **Generated outputs**: 100 MB - 5 GB (depends on number of climate variables and spatial resolution)

### CPU Usage
- HSA optimization is CPU-intensive (benefits from multi-core processors)
- Climate extraction relies on Google Earth Engine servers (local CPU not critical)

## Next Steps

After successful setup:
1. Read through `HSA_v5_FINAL_PENALTY_BASED.ipynb` to understand the optimization workflow
2. Modify parameters in the notebook to test different scenarios
3. Run climate extraction notebooks if you have access to Google Earth Engine
4. Refer to the main `README.md` for methodology details and citation information

## Getting Help

If you encounter issues not covered here:
- **Check documentation**: Read notebook markdown cells for parameter descriptions
- **GitHub Issues**: [Open an issue](https://github.com/izaslavsky/HSA_algo_public/issues)
- **Email**: [Contact information]

---

**Last Updated**: December 2024
