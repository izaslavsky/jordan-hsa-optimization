# Setup Instructions

These instructions configure a clean environment for the public synthetic-data workflow. Commands assume macOS or Linux and a shell opened at the repository root.

## 1. Clone and create an isolated Python environment

```bash
git clone https://github.com/izaslavsky/Jordan-hsa-optimization.git
cd Jordan-hsa-optimization
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.8 or later is supported. Python 3.10–3.12 is recommended for the geospatial stack.

On Apple Silicon, install system libraries with Homebrew if binary wheels are unavailable:

```bash
brew install gdal geos proj libomp
```

## 2. Register and authenticate Google Earth Engine

1. Request Earth Engine access at <https://earthengine.google.com/>.
2. Create or select a Google Cloud project with Earth Engine enabled.
3. Authenticate locally:

```bash
earthengine authenticate
```

4. Replace the example project value in each GEE notebook with your own project ID before execution.

The public notebooks must not contain a private project credential, API key, or service-account key. A Google Cloud project identifier is not itself a secret, but using your own project avoids coupling runs to another account.

## 3. Optional Google Drive download integration

The HSA climate notebooks can export Earth Engine tables to Google Drive and download them through the Drive API.

1. In Google Cloud Console, enable the Google Drive API for your project.
2. Configure an OAuth consent screen.
3. Create an OAuth client of type **Desktop app**.
4. Download the client configuration as `client_secrets.json` and place it in the repository root.

```text
Jordan-hsa-optimization/
├── client_secrets.json   # local only; ignored by Git
├── data/
├── out/
└── ...
```

Never commit `client_secrets.json`, OAuth tokens, service-account keys, or copied credential text. `.gitignore` already excludes the expected local credential file.

If Drive API integration is not desired, download completed Earth Engine exports manually and place them in the versioned `out/DRIVE_CLIMATE_*` directories documented in `DATA_FLOW_ANALYSIS.md`.

## 4. Verify the public data bundle

The following command should show only `SYNMOD` patient/facility datasets plus general public or derived files:

```bash
find data -maxdepth 2 -type f | sort
```

Real patient datasets must remain outside the repository. Filenames beginning with `data/INF_` or `data/NCD_` are ignored, but privacy review should never rely on filename rules alone.

## 5. Start Jupyter

```bash
jupyter notebook
```

Run the GEE notebooks interactively because authentication, export submission, and Drive polling require user participation.

## 6. Recommended execution order

### Phase 1

1. `GEE_local_Climate_Features_by_Facilities.ipynb`
2. `HSA_FINAL.ipynb`
3. `Population_Allocation_Probabilistic_v2.ipynb`

The local portions of steps 2–3 can be run reproducibly with:

```bash
python run_pipeline.py \
  --network SYNMODINF \
  --hsa-mode footprint \
  --boundary-version v7 \
  --disease-focus diarrheal \
  --only-steps 1,2
```

### Phase 2

1. `GEE_local_HSA_Weekly_Climate_Lagged.ipynb`
2. `GEE_local_HSA_Daily_Climate.ipynb`
3. `Generate_Modeling_Dataset.ipynb`
4. `run_climate_health_modeling.ipynb`
5. `Generate_Daily_Modeling_Dataset.ipynb`
6. `run_dlnm_primary_sensitivity.py`

The local notebook stages can be run with:

```bash
python run_pipeline.py \
  --network SYNMODINF \
  --hsa-mode footprint \
  --boundary-version v7 \
  --disease-focus diarrheal \
  --study-start 2022-07-01 \
  --study-end 2024-01-31 \
  --week-start 2019-01-07 \
  --week-end 2024-01-29 \
  --ml-start-date 2022-06-27 \
  --ml-end-date 2024-01-29 \
  --only-steps 3,4,5
```

Then run the manuscript-reported daily model:

```bash
python run_dlnm_primary_sensitivity.py
```

See `README.md` and `DATA_FLOW_ANALYSIS.md` for inputs and outputs.

## 7. Common checks

Confirm package imports:

```bash
python -c "import geopandas, rasterio, sklearn, statsmodels; print('imports OK')"
```

Confirm Earth Engine authentication:

```bash
earthengine asset list --project YOUR_PROJECT_ID
```

Confirm that the daily DLNM input exists before running the final model:

```bash
test -f out/modeling/SYNMODINF_footprint_daily_modeling_dataset_v7.csv
```

## 8. Output and release hygiene

- Generated files belong under `out/`; only `out/.gitkeep` is committed.
- Executed notebook copies belong under `_pipeline_runs/`; the directory is ignored.
- Clear notebook outputs before committing so local paths, transient results, and authentication messages are not published.
- Review `git status --ignored` before every public release.
