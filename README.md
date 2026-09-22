# Hospital Service Area Optimization and Climate-Health Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Code and synthetic data accompanying the research paper on delineating Hospital Service Areas (HSAs) from facility locations and patient volumes, and on analyzing climate-health relationships in Jordan. It contains the delineation algorithm and its three boundary variants, gravity-based population allocation, the weekly and daily climate-health modeling pipelines, and a DLNM cross-basis module.

---

## Reproducibility

Applying this pipeline to another facility network or another disease is a
configuration change, not a code change. Disease groups, outcome column names
and file naming are resolved at run time from the network's
`groups_of_diagnoses` table, so nothing disease-specific is written into the
scripts. Each network, optimization mode and boundary version writes to its own
directory, and results for each disease are kept separate beneath it, so two
diseases or two modes can never overwrite one another.

Four checks are executable and exit non-zero on failure, so an audit is one
command rather than an inspection:

```bash
python check_no_hardcoding.py --root .      # no hardcoded disease label or output path
python check_hsa_overlap.py --out-dir <run> --version v7   # no near-duplicate service areas
python check_pipeline_outputs.py --network INF --mode footprint --out-dir <run>
python check_pipeline_outputs.py --all      # every run present
```

`check_pipeline_outputs.py` reconciles a run against its own delineation: every
anchor has climate and no others do, every derived file is newer than the
delineation, allocation and climate it was built from, and nothing in the
directory predates the delineation. Matching file counts is not sufficient —
several defects found during development produced the right number of files
with the wrong membership.

## Delineation and modeling components

### Three HSA algorithm variants

`HSA_FINAL.ipynb` runs all three variants in a single execution and writes versioned boundary bundles:

| Bundle | Algorithm | Key additions |
|--------|-----------|---------------|
| **v6** | Greedy multi-objective optimization only | Baseline — no post-selection corrections |
| **v7** | v6 + anchor quality-control | Weak anchors replaced by stronger nearby facilities; a major facility outside every service area promoted to anchor when no admissible anchor is within the fallback distance |
| **v8** | v7 + satellite bubble boundaries | HSA polygons union the anchor catchment with smaller secondary catchments around eligible nearby facilities |

All downstream notebooks select a bundle via `BOUNDARY_VERSION = "v6" | "v7" | "v8"` near the top of each notebook.

### Daily climate-health pipeline

A second modeling track built on daily data:

- `GEE_local_HSA_Daily_Climate.ipynb` extracts daily climate from Google Earth Engine (CHIRPS + ERA5-Land) for each HSA polygon.
- `generate_daily_disease_counts.py` produces daily diarrheal case counts per HSA from patient visit records.
- `prepare_daily_modeling_dataset.py` assembles a daily panel with 14-day climate lags and infrastructure covariates.
- `Generate_Daily_Modeling_Dataset.ipynb` orchestrates these steps.
- `run_climate_models_daily.ipynb` runs two modeling tracks:
  - **Track A**: Explanatory quasi-Poisson DLNM — tests whether climate associations are modified by sanitation/infrastructure quality.
  - **Track B**: Predictive OLS at five forecast horizons.

See `DATA_FLOW_ANALYSIS.md` for end-to-end pipeline description.

### DLNM module

`dlnm/dlnm_crossbasis.py` implements distributed lag non-linear models in Python: natural cubic spline cross-basis construction, quasi-Poisson GLM fitting, and cumulative relative risk computation. Import with `from dlnm.dlnm_crossbasis import ns_basis, build_crossbasis, cumulative_rr`.

### Improved population allocation

`Population_Allocation_Probabilistic_v2.ipynb` (using updated `population_allocation.py`) replaces the previous nearest-anchor fallback with an admissibility-limited fallback: facilities outside all HSA radii are assigned only to anchors within a distance limit derived from the anchor's service radius, with same-governorate preference for major facilities. Facilities that fail the admissibility check are reported rather than silently attached to a distant anchor.

---

## Repository structure

```
jordan-hsa-optimization/
├── data/
│   ├── adm_boundaries/              Administrative boundaries (governorate/district/subdistrict)
│   ├── SYNMODINF_facility_coordinates.csv   INF facility locations
│   ├── SYNMODINF_groups_of_diagnoses.csv    ICD groupings for INF network
│   ├── SYNMODINF_patient_visits.csv         Synthetic INF patient visits (2019–2024)
│   ├── SYNMODNCD_facility_coordinates.csv   NCD facility locations
│   ├── SYNMODNCD_groups_of_diagnoses.csv    ICD groupings for NCD network
│   ├── SYNMODNCD_patient_visits.csv         Synthetic NCD patient visits
│   ├── jordan_boundary.gpkg
│   ├── jordan_governorates.gpkg
│   └── hsa_metadata.csv                     JMP sanitation quality scores per HSA
│   [WorldPop rasters not included — see Installation below]
├── dlnm/                            DLNM cross-basis module
├── out/                             Delineation masters from HSA_FINAL (gitignored)
├── out_<NETWORK>_<mode>_<version>/  One directory per run (gitignored). Holds the
│   │                                delineation, gravity allocation and per-HSA
│   │                                climate, which depend on network/mode/version
│   │                                only and are shared across diseases.
│   └── <disease_slug>/              Everything specific to one disease: counts,
│                                    modeling datasets, model results, sensitivity
├── HSA_FINAL.ipynb                  HSA delineation (produces v6, v7, v8 bundles)
├── Population_Allocation_Probabilistic_v2.ipynb
├── GEE_local_Climate_Features_by_Facilities.ipynb
├── GEE_local_HSA_Daily_Climate.ipynb
├── GEE_local_HSA_Weekly_Climate_Lagged.ipynb
├── GEE_local_HSA_Weekly_Climate_Lagged_chunked.ipynb
├── Generate_Modeling_Dataset.ipynb
├── Generate_Daily_Modeling_Dataset.ipynb
├── run_climate_health_modeling.ipynb
├── run_climate_models_daily.ipynb
├── compare_delineations.ipynb
├── dlnm/
│   └── dlnm_crossbasis.py               Natural spline cross-basis and cumulative RR
├── hsa_optimization.py                  Core algorithm (v6/v7/v8 via flags)
├── generate_hsa_metadata.py             Build data/hsa_metadata.csv from coordinates + JMP 2025 lookup
├── hsa_mapping_working.py               HSA visualization helpers
├── hsa_objective_analysis.py            Objective function diagnostics
├── population_allocation.py             Probabilistic gravity allocation
├── generate_diagnosis_counts_v2.py      Diagnosis grouping from visit records
├── generate_weekly_disease_counts_adjusted.py
├── prepare_ml_dataset.py                Weekly modeling dataset assembly
├── generate_daily_disease_counts.py
├── prepare_daily_modeling_dataset.py    Daily panel with climate lags
├── climate_health_modeling.py           Weekly climate-health models (called by notebook)
├── climate_health_modeling_comprehensive.py
├── climate_health_modeling_parsimonious.py
├── climate_health_modeling_anomalies.py
├── train_improved_models.py
├── train_ml_models.py
├── 08_climate_ar_decomposition.py       Supplementary analyses (08–16)
├── ...
├── 16_within_hsa_heterogeneity.py
├── README.md
├── DATA_FLOW_ANALYSIS.md
└── SETUP_INSTRUCTIONS.md
```

---

## Synthetic data

`SYNMOD` files preserve the statistical properties of the real data, including temporal structure, seasonal patterns, and diagnosis-category distributions. Facility coordinates and ICD groupings are identical to the real data. Patient IDs and specific visit records are synthetic.

**Caveat**: Climate associations and explanatory DLNM results using SYNMOD data should be treated as pipeline validation, not scientific findings. Real outcome data is required for substantive inference.

---

## Installation

```bash
git clone https://github.com/izaslavsky/jordan-hsa-optimization.git
cd jordan-hsa-optimization
pip install -r requirements.txt
earthengine authenticate   # required for GEE notebooks
```

### WorldPop rasters

The population rasters are not committed (too large). Download them from WorldPop and place in `data/`:

| File | URL |
|------|-----|
| `jor_ppp_2020_UNadj.tif` | https://www.worldpop.org — Jordan 2020 unconstrained |
| `jor_ppp_2020_constrained.tif` | https://www.worldpop.org — Jordan 2020 constrained |

---

## Running the pipeline

Steps 1 and 2 run once. Steps 3 onward carry a `BOUNDARY_VERSION` parameter and can be run three times (v6, v7, v8) to produce results for all boundary variants.

### Step 1 — Climate features by facility

```bash
jupyter notebook GEE_local_Climate_Features_by_Facilities.ipynb
```

Copy the output `{NETWORK}_Facilities_Climate_Features_with_clusters.csv` into `out/`.

### Step 2 — HSA delineation (all three variants)

```bash
jupyter notebook HSA_FINAL.ipynb
```

Produces 5 boundary files, one per mode: `out/{NETWORK}_{mode}_hsas_v7.geojson`.
Only v7 is built by default; set `HSA_VARIANTS="v6,v7,v8"` to rebuild the others.

### Step 3 — Population allocation

Set `BOUNDARY_VERSION = "v6" | "v7" | "v8"` in the notebook, then run:

```bash
jupyter notebook Population_Allocation_Probabilistic_v2.ipynb
```

Repeat for each boundary version.

### Step 4 — Climate aggregation by HSA

**Weekly** (set `BOUNDARY_VERSION` in the notebook):

```bash
jupyter notebook GEE_local_HSA_Weekly_Climate_Lagged.ipynb
```

**Daily** (set `BOUNDARY_VERSION` in the notebook):

```bash
jupyter notebook GEE_local_HSA_Daily_Climate.ipynb
```

### Step 5 — Generate modeling dataset

**Weekly** (disease counts + climate merge, set `BOUNDARY_VERSION`):

```bash
jupyter notebook Generate_Modeling_Dataset.ipynb
```

**Daily** (disease counts + lag assembly + dataset merge, set `BOUNDARY_VERSION`):

```bash
jupyter notebook Generate_Daily_Modeling_Dataset.ipynb
```

Both notebooks call the relevant `.py` helper scripts internally.

### Step 6 — Run models

**Weekly climate-health**:

```bash
jupyter notebook run_climate_health_modeling.ipynb
```

**Daily DLNM + predictive**:

```bash
jupyter notebook run_climate_models_daily.ipynb
```

### Optional — Compare delineation methods

```bash
jupyter notebook compare_delineations.ipynb
```

---

## Workflow diagram

```
Step 1  GEE_local_Climate_Features_by_Facilities.ipynb   [run once]
            │
            ▼
Step 2  HSA_FINAL.ipynb                                  [run once]
            │  → out/{NETWORK}_{mode}_hsas_{v6|v7|v8}.geojson
            │    (v6: greedy | v7: +anchor QC | v8: +bubbles)
            │
            ▼
    ┌── BOUNDARY_VERSION = v6 | v7 | v8 ──┐
    │  (Steps 3–6 repeat for each version) │
    └──────────────────────────────────────┘
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

## Climate data note

Weekly climate CSVs (CHIRPS + ERA5-Land + TerraClimate) and daily climate CSVs (CHIRPS + ERA5-Land) are not committed to the repository. Run the corresponding GEE notebook (Step 4) with the desired `BOUNDARY_VERSION` to generate them. The chunked variant `GEE_local_HSA_Weekly_Climate_Lagged_chunked.ipynb` is provided for runs that exceed GEE export memory limits.

The weekly export also emits per-HSA elevation statistics from SRTM: mean, standard deviation, min, max and the 25th/50th/75th percentiles, written as one row per HSA (`*_elevation_by_week.csv`). The spread, not just the mean, is what the within-HSA heterogeneity analysis needs, since an HSA spanning the Jordan Valley and the highlands has a far larger internal climate gradient than a compact urban one. Elevation is produced by the same notebook as the other climate families; there is no separate elevation notebook. To export elevation alone, set `USE_CHIRPS`, `USE_ERA5_HOURLY` and `USE_ERA5_EVP` to `False` and leave `INCLUDE_ELEVATION = True`.

---

## Documentation

| File | Contents |
|------|----------|
| `DATA_FLOW_ANALYSIS.md` | End-to-end data flow: inputs, outputs, and commands for each pipeline step |
| `SETUP_INSTRUCTIONS.md` | GEE and Google Drive credential setup |

---

## Citation

```bibtex
@software{zaslavsky_jordan_hsa_optimization_2026,
  title     = {jordan-hsa-optimization},
  author    = {Zaslavsky, Ilya and Lamont, Stephan and Hussien, Moawiah O. and
               Abudhail, Jamila and Kirkpatrick, Christine and Al-Delaimy, Wael},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  note      = {Software and synthetic data}
}
```

Replace the DOI above with the version DOI of the archived release.

---

## License

MIT License. See [LICENSE](LICENSE).

**Data licenses**: Administrative boundaries: ODbL (OpenStreetMap). Synthetic patient data: public domain. Climate data: see individual source licenses (CHIRPS, ERA5-Land, TerraClimate, WorldPop).
