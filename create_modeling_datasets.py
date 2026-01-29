"""
Create modeling-ready datasets from merged climate-disease data

Generates two versions:
1. Complete cases: 9 HSAs with all climate variables (temperature, precipitation, etc.)
2. Precipitation-only: 18 HSAs with precipitation variables only

Period: 2022-06-27 to 2024-01-29 (84 weeks)
"""

import pandas as pd
import numpy as np
import os
import argparse

def _default_disease(network):
    return "diarrheal" if network in ("INF", "SYN") else "hypertension"

def _secondary_label(network):
    return "infectious" if network in ("INF", "SYN") else "ncd"

def parse_args():
    parser = argparse.ArgumentParser(description="Create modeling-ready datasets")
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
print("CREATING MODELING-READY DATASETS")
print(f"Network: {NETWORK} | HSA mode: {HSA_MODE}")
print(f"Disease focus: {DISEASE_FOCUS} | Secondary: {SECONDARY_LABEL}")
print("="*80)

# Load merged data
print("\n[1/4] Loading merged dataset...")
df = pd.read_csv(f'out/{NETWORK}_{HSA_MODE}_climate_disease_merged.csv')
print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")

# Define HSAs with complete climate data (no missing values)
complete_hsas = [
    'Maan Hospital',
    'AL-Ramtha Hospital',
    'AL-Nadeem Hospital',
    'AL-Shuneh Hospital',
    'Jarash Hospital',
    'Princess Salma Hospital',
    'Queen Rania Hospital',
    'Tafilah Governmental Hospital',
    'Al-Yarmouk Hospital'
]

print(f"\n[2/4] Creating complete cases dataset (9 HSAs)...")

# Filter to complete HSAs
df_complete = df[df['hsa_id'].isin(complete_hsas)].copy()

# Verify no missing values
missing_count = df_complete.isnull().sum().sum()
if missing_count > 0:
    print(f"  WARNING: Found {missing_count} missing values in 'complete' HSAs")
else:
    print(f"  Verified: 0 missing values")

# Summary
print(f"\n  Complete cases dataset:")
print(f"    HSAs: {df_complete['hsa_id'].nunique()}")
print(f"    Weeks: {df_complete['week_start'].nunique()}")
print(f"    Total rows: {len(df_complete):,}")
print(f"    Variables: {len(df_complete.columns)}")
print(f"    Disease cases:")
print(f\"      {DISEASE_FOCUS.title()}: {df_complete[f'{DISEASE_FOCUS}_count_adjusted'].sum():,.0f}\")
print(f\"      {SECONDARY_LABEL.upper()}: {df_complete[f'{SECONDARY_LABEL}_count_adjusted'].sum():,.0f}\")

# Save
output_file = f'out/{NETWORK}_{HSA_MODE}_climate_disease_complete_cases.csv'
df_complete.to_csv(output_file, index=False)
print(f"\n  Saved: {output_file}")

print(f"\n[3/4] Creating precipitation-only dataset (18 HSAs)...")

# Identify precipitation columns
id_cols = ['hsa_id', 'week_start', 'week_start_iso']
disease_cols = [f'{DISEASE_FOCUS}_count_adjusted', f'{SECONDARY_LABEL}_count_adjusted']
precip_cols = [c for c in df.columns if c.startswith('precip_')]
elev_cols = [c for c in df.columns if c.startswith('elev_')]

# Select columns
selected_cols = id_cols + disease_cols + precip_cols + elev_cols
df_precip = df[selected_cols].copy()

# Check for missing values
missing_precip = df_precip.isnull().sum().sum()
if missing_precip > 0:
    print(f"  Dropping {missing_precip} rows with missing precipitation values...")
    df_precip = df_precip.dropna()

print(f"\n  Precipitation-only dataset:")
print(f"    HSAs: {df_precip['hsa_id'].nunique()}")
print(f"    Weeks: {df_precip['week_start'].nunique()}")
print(f"    Total rows: {len(df_precip):,}")
print(f"    Variables: {len(df_precip.columns)} (precipitation + disease)")
print(f"    Disease cases:")
print(f\"      {DISEASE_FOCUS.title()}: {df_precip[f'{DISEASE_FOCUS}_count_adjusted'].sum():,.0f}\")
print(f\"      {SECONDARY_LABEL.upper()}: {df_precip[f'{SECONDARY_LABEL}_count_adjusted'].sum():,.0f}\")

# Save
output_file = f'out/{NETWORK}_{HSA_MODE}_climate_disease_precipitation_only.csv'
df_precip.to_csv(output_file, index=False)
print(f"\n  Saved: {output_file}")

print(f"\n[4/4] Creating variable lists for modeling...")

# Create variable metadata for each dataset
metadata_complete = []
metadata_precip = []

# Complete cases variables
for col in df_complete.columns:
    if col in id_cols:
        continue

    if col in disease_cols:
        var_type = 'outcome'
    elif col.startswith('precip_'):
        var_type = 'predictor_precipitation'
    elif col.startswith('temp_'):
        var_type = 'predictor_temperature'
    elif col.startswith('evap_'):
        var_type = 'predictor_evaporation'
    elif col.startswith('sm_'):
        var_type = 'predictor_soil_moisture'
    elif col.startswith('wb_'):
        var_type = 'predictor_water_balance'
    elif col.startswith('elev_'):
        var_type = 'predictor_elevation'
    else:
        var_type = 'other'

    metadata_complete.append({
        'variable': col,
        'type': var_type,
        'missing_pct': df_complete[col].isnull().mean() * 100
    })

# Precipitation-only variables
for col in df_precip.columns:
    if col in id_cols:
        continue

    if col in disease_cols:
        var_type = 'outcome'
    elif col.startswith('precip_'):
        var_type = 'predictor_precipitation'
    elif col.startswith('elev_'):
        var_type = 'predictor_elevation'
    else:
        var_type = 'other'

    metadata_precip.append({
        'variable': col,
        'type': var_type,
        'missing_pct': df_precip[col].isnull().mean() * 100
    })

# Save metadata
meta_df_complete = pd.DataFrame(metadata_complete)
meta_df_precip = pd.DataFrame(metadata_precip)

meta_df_complete.to_csv(f'out/{NETWORK}_{HSA_MODE}_climate_disease_complete_cases_variables.csv', index=False)
meta_df_precip.to_csv(f'out/{NETWORK}_{HSA_MODE}_climate_disease_precipitation_only_variables.csv', index=False)

print(f"  Saved: out/{NETWORK}_{HSA_MODE}_climate_disease_complete_cases_variables.csv")
print(f"  Saved: out/{NETWORK}_{HSA_MODE}_climate_disease_precipitation_only_variables.csv")

# Summary table
print("\n" + "="*80)
print("MODELING DATASETS SUMMARY")
print("="*80)

summary_data = [
    ['Dataset', 'HSAs', 'Weeks', 'Rows', 'Climate Vars', 'Diarrheal', 'Infectious'],
    ['Complete Cases',
     df_complete['hsa_id'].nunique(),
     df_complete['week_start'].nunique(),
     len(df_complete),
     len([c for c in df_complete.columns if c.startswith(('precip_', 'temp_', 'evap_', 'sm_', 'wb_', 'elev_'))]),
     f"{df_complete['diarrheal_count_adjusted'].sum():,.0f}",
     f"{df_complete['infectious_count_adjusted'].sum():,.0f}"],
    ['Precipitation-Only',
     df_precip['hsa_id'].nunique(),
     df_precip['week_start'].nunique(),
     len(df_precip),
     len([c for c in df_precip.columns if c.startswith(('precip_', 'elev_'))]),
     f"{df_precip['diarrheal_count_adjusted'].sum():,.0f}",
     f"{df_precip['infectious_count_adjusted'].sum():,.0f}"]
]

for row in summary_data:
    if row[0] == 'Dataset':
        print(f"  {row[0]:<20} {row[1]:>6} {row[2]:>6} {row[3]:>6} {row[4]:>12} {row[5]:>12} {row[6]:>12}")
        print("  " + "-"*76)
    else:
        print(f"  {row[0]:<20} {row[1]:>6} {row[2]:>6} {row[3]:>6} {row[4]:>12} {row[5]:>12} {row[6]:>12}")

print("\n" + "="*80)
print("[OK] MODELING DATASETS READY")
print("="*80)

print("\nOutput files:")
print(f"  1. out/{NETWORK}_{HSA_MODE}_climate_disease_complete_cases.csv")
print("     - 9 HSAs with all climate variables")
print("     - Use for: Temperature-health, multi-variable models")
print("")
print(f"  2. out/{NETWORK}_{HSA_MODE}_climate_disease_precipitation_only.csv")
print("     - 18 HSAs with precipitation variables only")
print("     - Use for: Precipitation-diarrhea models, full geographic coverage")
print("")
print(f"  3. out/{NETWORK}_{HSA_MODE}_climate_disease_complete_cases_variables.csv")
print("     - Variable metadata for complete cases dataset")
print("")
print(f"  4. out/{NETWORK}_{HSA_MODE}_climate_disease_precipitation_only_variables.csv")
print("     - Variable metadata for precipitation-only dataset")

print("\nExample usage:")
print("  import pandas as pd")
print(f\"  df = pd.read_csv('out/{NETWORK}_{HSA_MODE}_climate_disease_complete_cases.csv')\")
print("  # Fit Poisson regression with lagged precipitation")
print("  from sklearn.linear_model import PoissonRegressor")
print("  X = df[[c for c in df.columns if 'precip_' in c and 'lag' in c]]")
print("  y = df['diarrheal_count_adjusted']")
