"""
Merge climate time series with disease counts for climate-health modeling

Creates a single dataset combining:
- Precipitation lags (improved with cumulative windows)
- Temperature/dewpoint/wind lags (with heat stress indicators)
- Evaporation lags
- Soil moisture lags
- Water balance (E-P)
- Elevation
- Diarrheal disease counts (adjusted)
- Total infectious disease counts (adjusted)

Period: 2022-06-27 to 2024-01-29 (84 weeks)
"""

import pandas as pd
import numpy as np
import glob
import os
import argparse

def _default_disease(network):
    return "diarrheal" if network in ("INF", "SYN") else "hypertension"

def _secondary_label(network):
    return "infectious" if network in ("INF", "SYN") else "ncd"

def parse_args():
    parser = argparse.ArgumentParser(description="Merge climate and disease datasets")
    parser.add_argument("--network", default=os.environ.get("NETWORK", "INF"))
    parser.add_argument("--hsa-mode", default=os.environ.get("HSA_MODE", "footprint"))
    parser.add_argument("--disease-focus", default=os.environ.get("DISEASE_FOCUS"))
    return parser.parse_args()


args = parse_args()
NETWORK = args.network
HSA_MODE = args.hsa_mode
DISEASE_FOCUS = (args.disease_focus or _default_disease(NETWORK)).lower()
SECONDARY_LABEL = _secondary_label(NETWORK)

print("="*80)
print("MERGING CLIMATE AND DISEASE DATA FOR MODELING")
print(f"Network: {NETWORK} | HSA mode: {HSA_MODE}")
print(f"Disease focus: {DISEASE_FOCUS} | Secondary: {SECONDARY_LABEL}")
print("="*80)

# Step 1: Load disease data (adjusted counts)
print("\n[1/5] Loading adjusted disease counts...")
disease_diar = pd.read_csv(f'out/{NETWORK}_{HSA_MODE}_weekly_{DISEASE_FOCUS}_adjusted.csv')
disease_inf = pd.read_csv(f'out/{NETWORK}_{HSA_MODE}_weekly_{SECONDARY_LABEL}_adjusted.csv')

# Filter to modeling period (2022-06-27 to 2024-01-29)
disease_diar['week_start'] = pd.to_datetime(disease_diar['week_start'])
disease_inf['week_start'] = pd.to_datetime(disease_inf['week_start'])

modeling_start = pd.Timestamp('2022-06-27')
modeling_end = pd.Timestamp('2024-01-29')

disease_diar = disease_diar[
    (disease_diar['week_start'] >= modeling_start) &
    (disease_diar['week_start'] <= modeling_end)
].copy()

disease_inf = disease_inf[
    (disease_inf['week_start'] >= modeling_start) &
    (disease_inf['week_start'] <= modeling_end)
].copy()

print(f"  {DISEASE_FOCUS.title()}: {len(disease_diar)} rows ({disease_diar['hsa_id'].nunique()} HSAs)")
print(f"  {SECONDARY_LABEL.upper()}: {len(disease_inf)} rows ({disease_inf['hsa_id'].nunique()} HSAs)")

# Merge disease counts
disease = disease_diar.merge(
    disease_inf[['hsa_id', 'week_start', f'{SECONDARY_LABEL}_count_adjusted']],
    on=['hsa_id', 'week_start'],
    how='outer'
)

print(f"  Combined: {len(disease)} rows")

# Step 2: Load and merge climate data
print("\n[2/5] Loading climate data from extracted CSVs...")

os.chdir('out')

# Get list of all HSAs
hsa_list = disease['hsa_id'].unique()
print(f"  Processing {len(hsa_list)} HSAs")

# Initialize merged dataset
merged_data = disease.copy()
merged_data['week_start_str'] = merged_data['week_start'].dt.strftime('%Y-%m-%d')

# Process each HSA
file_types = {
    'precip_lags': 'precip_',
    'tempdew_wind_lags': 'temp_',
    'evapERA5_lags': 'evap_',
    'soilmoistERA5_lags': 'sm_',
    'water_balance': 'wb_',
    'elevation_by_week': 'elev_'
}

print("\n[3/5] Merging climate variables for each HSA...")

# Columns to drop (metadata, not climate variables)
cols_to_drop = ['system:index', '.geo', 'FacilityName', 'HealthFacility']

# Build one merged dataset per HSA, then concatenate
all_hsa_data = []

for hsa_name in hsa_list:
    # Construct filename prefix
    hsa_prefix = f"HSA_{hsa_name}_"

    hsa_climate = None

    for file_suffix, col_prefix in file_types.items():
        # Handle duplicate water_balance files - use the one without (1)
        pattern = f"{hsa_prefix}{file_suffix}.csv"
        files = glob.glob(pattern)

        if not files:
            print(f"  WARNING: Missing {file_suffix} for {hsa_name}")
            continue

        fname = files[0]

        try:
            df = pd.read_csv(fname)

            # Drop metadata columns
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

            # Rename columns to add prefix (except week_start)
            rename_map = {col: f"{col_prefix}{col}" for col in df.columns if col != 'week_start'}
            df = df.rename(columns=rename_map)

            if hsa_climate is None:
                hsa_climate = df
            else:
                hsa_climate = hsa_climate.merge(df, on='week_start', how='outer')

        except Exception as e:
            print(f"  ERROR loading {fname}: {e}")
            continue

    if hsa_climate is not None:
        # Add HSA identifier
        hsa_climate['hsa_id'] = hsa_name
        all_hsa_data.append(hsa_climate)

print(f"  Successfully loaded climate data for {len(all_hsa_data)}/{len(hsa_list)} HSAs")

# Concatenate all HSA climate data
print("\n  Concatenating climate data across HSAs...")
climate_all = pd.concat(all_hsa_data, ignore_index=True)

# Convert week_start to datetime for merging
climate_all['week_start'] = pd.to_datetime(climate_all['week_start'])

# Merge with disease data
print("  Merging with disease counts...")
merged_data = disease.merge(
    climate_all,
    on=['hsa_id', 'week_start'],
    how='left'
)

# Step 4: Clean up and organize columns
print("\n[4/5] Organizing columns...")

# Reorder columns: identifiers, disease counts, then climate variables
id_cols = ['hsa_id', 'week_start', 'week_start_iso']
disease_cols = ['diarrheal_count_adjusted', 'infectious_count_adjusted']

# Get all climate columns
climate_cols = [col for col in merged_data.columns if col not in id_cols + disease_cols]

# Reorder
merged_data = merged_data[id_cols + disease_cols + climate_cols]

print(f"  Total columns: {len(merged_data.columns)}")
print(f"    Identifiers: {len(id_cols)}")
print(f"    Disease counts: {len(disease_cols)}")
print(f"    Climate variables: {len(climate_cols)}")

# Step 5: Validate and save
print("\n[5/5] Validating and saving merged dataset...")

# Check for missing values
missing_summary = merged_data.isnull().sum()
missing_vars = missing_summary[missing_summary > 0]

if len(missing_vars) > 0:
    print(f"\n  WARNING: {len(missing_vars)} variables have missing values:")
    for var, count in missing_vars.head(10).items():
        print(f"    {var}: {count} missing ({count/len(merged_data)*100:.1f}%)")
else:
    print("  No missing values detected")

# Summary statistics
print(f"\n  Dataset summary:")
print(f"    Total rows: {len(merged_data):,}")
print(f"    HSAs: {merged_data['hsa_id'].nunique()}")
print(f"    Weeks: {merged_data['week_start'].nunique()}")
print(f"    Period: {merged_data['week_start'].min()} to {merged_data['week_start'].max()}")

# Disease counts summary
print(f"\n  Disease counts (total over all HSA-weeks):")
print(f"    Diarrheal: {merged_data['diarrheal_count_adjusted'].sum():,.0f}")
print(f"    Infectious: {merged_data['infectious_count_adjusted'].sum():,.0f}")

# Save merged dataset
os.chdir('..')  # Back to main directory
output_file = f'out/{NETWORK}_{HSA_MODE}_climate_disease_merged.csv'
merged_data.to_csv(output_file, index=False)

print(f"\n  Saved: {output_file}")

# Save column metadata
print("\n[BONUS] Creating column metadata file...")
metadata = []

# Disease variables
metadata.append({'category': 'disease', 'variable': 'diarrheal_count_adjusted',
                 'description': 'Weekly diarrheal disease cases (gravity model adjusted)'})
metadata.append({'category': 'disease', 'variable': 'infectious_count_adjusted',
                 'description': 'Weekly total infectious disease cases (gravity model adjusted)'})

# Climate variables by prefix
for col in climate_cols:
    if col.startswith('precip_'):
        category = 'precipitation'
    elif col.startswith('temp_'):
        category = 'temperature_humidity'
    elif col.startswith('evap_'):
        category = 'evaporation'
    elif col.startswith('sm_'):
        category = 'soil_moisture'
    elif col.startswith('wb_'):
        category = 'water_balance'
    elif col.startswith('elev_'):
        category = 'elevation'
    else:
        category = 'other'

    metadata.append({'category': category, 'variable': col, 'description': ''})

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(f'out/{NETWORK}_{HSA_MODE}_climate_disease_metadata.csv', index=False)

print(f"  Saved: out/{NETWORK}_{HSA_MODE}_climate_disease_metadata.csv")

# Print climate variable counts by category
print("\n  Climate variables by category:")
cat_counts = metadata_df[metadata_df['category'] != 'disease'].groupby('category').size().sort_values(ascending=False)
for cat, count in cat_counts.items():
    print(f"    {cat}: {count}")

print("\n" + "="*80)
print("[OK] CLIMATE-DISEASE DATASET READY FOR MODELING")
print("="*80)
print("\nNext steps:")
print(f\"  1. Load merged dataset: pd.read_csv('out/{NETWORK}_{HSA_MODE}_climate_disease_merged.csv')\")
print("  2. Check for multicollinearity among climate predictors (VIF analysis)")
print("  3. Fit distributed lag models (DLMs) or GAMs")
print("  4. Account for temporal autocorrelation (ARIMA errors, GLS)")
print("  5. Include HSA-specific random effects (mixed models)")
