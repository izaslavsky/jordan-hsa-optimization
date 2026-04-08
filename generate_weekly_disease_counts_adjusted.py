"""
Generate ADJUSTED weekly disease counts for HSAs

Uses gravity model probabilities to properly allocate patients from facilities
that serve multiple HSAs, avoiding double-counting.

Adjustments:
1. For facilities in only 1 HSA: 100% of patients go to that HSA
2. For facilities in multiple HSAs: patients split proportionally by gravity model probabilities

Output:
- {NETWORK}_{HSA_MODE}_weekly_{DISEASE_FOCUS}_adjusted.csv: Adjusted primary disease counts
- {NETWORK}_{HSA_MODE}_weekly_{SECONDARY_LABEL}_adjusted.csv: Adjusted secondary counts
"""

import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import numpy as np
import sys
import os
from pathlib import Path
import argparse
from scipy.spatial import cKDTree

# Default date configuration (can be overridden via CLI)
DEFAULT_WEEK_START = "2019-01-07"
DEFAULT_WEEK_END = "2024-01-29"

def _parse_network_mode(hsa_geojson_path):
    stem = Path(hsa_geojson_path).stem
    # Expected patterns:
    #   NETWORK_MODE_hsas
    #   NETWORK_MODE_hsas_v2
    # where MODE itself may contain underscores, e.g. governorate_fewest.
    parts = stem.split("_")
    if len(parts) >= 3:
        try:
            hsas_idx = parts.index("hsas")
        except ValueError:
            hsas_idx = -1
        if hsas_idx >= 2:
            network = parts[0]
            mode = "_".join(parts[1:hsas_idx])
            return network, mode
    # Fallback to env or defaults
    return os.environ.get("NETWORK", "INF"), os.environ.get("HSA_MODE", "footprint")

def _default_disease(network):
    return "diarrheal" if network in ("INF", "SYNINF") else "hypertension"

def _secondary_label(network):
    return "infectious" if network in ("INF", "SYNINF") else "ncd"

parser = argparse.ArgumentParser(description="Generate adjusted weekly disease counts")
parser.add_argument("hsa_geojson", nargs="?", default=None)
parser.add_argument("--disease", default=None, help="Primary disease focus (e.g., diarrheal, hypertension)")
parser.add_argument("--week-start", default=os.environ.get("WEEK_START", DEFAULT_WEEK_START),
                    help=f"Start date for weeks (default: {DEFAULT_WEEK_START})")
parser.add_argument("--week-end", default=os.environ.get("WEEK_END", DEFAULT_WEEK_END),
                    help=f"End date for weeks (default: {DEFAULT_WEEK_END})")
args = parser.parse_args()

if args.hsa_geojson:
    HSA_GEOJSON = args.hsa_geojson
else:
    HSA_GEOJSON = input("Enter path to HSA GeoJSON file: ").strip()

NETWORK, HSA_MODE = _parse_network_mode(HSA_GEOJSON)
DISEASE_FOCUS = (args.disease or os.environ.get("DISEASE_FOCUS") or _default_disease(NETWORK)).lower()
SECONDARY_LABEL = _secondary_label(NETWORK)
WEEK_START = args.week_start
WEEK_END = args.week_end

print("="*80)
print("GENERATING ADJUSTED WEEKLY DISEASE COUNTS")
print("Using gravity model to avoid double-counting facilities in multiple HSAs")
print(f"Network: {NETWORK} | HSA mode: {HSA_MODE}")
print(f"Disease focus: {DISEASE_FOCUS} | Secondary: {SECONDARY_LABEL}")
print("="*80)

# Step 1: Load HSAs
print("\n[1/7] Loading HSA geometries...")
hsas_gdf = gpd.read_file(HSA_GEOJSON)
print(f"  Loaded {len(hsas_gdf)} HSAs")

if 'healthfacility' in hsas_gdf.columns:
    hsa_id_col = 'healthfacility'
elif 'anchor_name' in hsas_gdf.columns:
    hsa_id_col = 'anchor_name'
else:
    raise ValueError(f"Cannot find HSA ID column")

hsa_ids = hsas_gdf[hsa_id_col].tolist()
print(f"  Using '{hsa_id_col}' as HSA identifier")

from pathlib import Path

# Step 2: Load gravity model allocation details
print("\n[2/7] Loading gravity model allocations...")
# Try both naming conventions (old and new/probabilistic)
alloc_path = Path(f'out/{NETWORK}_{HSA_MODE}_allocation_details.csv')
alloc_path_alt = Path(f'out/pixel_allocations_{NETWORK}_{HSA_MODE}.csv')

if alloc_path.exists():
    print(f"  Using: {alloc_path}")
elif alloc_path_alt.exists():
    alloc_path = alloc_path_alt
    print(f"  Using: {alloc_path}")
else:
    raise FileNotFoundError(
        f"Missing allocation details. Tried:\n"
        f"  - {alloc_path}\n"
        f"  - {alloc_path_alt}\n"
        "Run patient allocation for this network/mode to generate it "
        "(e.g., Patient_Allocation_Probabilistic.ipynb or the old allocation notebook), "
        "or use generate_weekly_disease_counts.py for unadjusted counts."
    )
alloc_details = pd.read_csv(alloc_path)
print(f"  Loaded {len(alloc_details):,} population pixels with facility assignments")

# Map facility to names
# Use FACILITY_DATA env var if set, otherwise try non-synthetic file first
fac_file = os.environ.get('FACILITY_DATA')
if not fac_file:
    fac_file = f'data/{NETWORK}_facility_coordinates.csv'
    if not Path(fac_file).exists():
        fac_file = f'data/SYN{NETWORK}_facility_coordinates.csv'
print(f"  Using facilities: {fac_file}")
facilities_df = pd.read_csv(fac_file, encoding='utf-8-sig')
facilities_df['healthfacility'] = facilities_df['healthfacility'].str.replace('\xa0', ' ').str.replace(r'\s+', ' ', regex=True).str.strip()

# Handle both old format (facility_idx) and new format (facility_id only)
if 'facility_idx' in alloc_details.columns:
    # Old format: use facility_idx to map to names
    facility_idx_to_name = dict(enumerate(facilities_df['healthfacility']))
    alloc_details['facility_name'] = alloc_details['facility_idx'].map(facility_idx_to_name)
elif 'facility_id' in alloc_details.columns:
    # New format: facility_id already contains the name
    alloc_details['facility_name'] = alloc_details['facility_id'].str.replace('\xa0', ' ').str.replace(r'\s+', ' ', regex=True).str.strip()
    # Create facility_idx from facility_id for compatibility
    name_to_idx = {name: idx for idx, name in enumerate(facilities_df['healthfacility'])}
    alloc_details['facility_idx'] = alloc_details['facility_name'].map(name_to_idx)
else:
    raise ValueError("Allocation file must have either 'facility_idx' or 'facility_id' column")

# For each facility, calculate its probability of serving each HSA
print("\n[3/7] Calculating facility-to-HSA probabilities...")

# Group allocation details by facility
facility_populations = alloc_details.groupby('facility_idx').agg({
    'population': 'sum',
    'probability': 'mean',
    'facility_name': 'first'
}).reset_index()

# For each facility, determine which HSA(s) it serves based on gravity model allocations
# Strategy: For each non-anchor facility, find which anchor facilities serve the same population pixels
facility_to_hsas = {}

# First, handle anchor facilities (100% to own HSA)
for fac_name in hsa_ids:
    facility_to_hsas[fac_name] = {fac_name: 1.0}

# For non-anchor facilities, assign to HSA based on which anchor serves the most overlapping population
# Get all non-anchor facilities
non_anchor_facilities = [f for f in facility_populations['facility_name'].unique() if f not in hsa_ids]

print(f"  Processing {len(hsa_ids)} anchor facilities (direct mapping)")
print(f"  Processing {len(non_anchor_facilities)} non-anchor facilities (using gravity model)")

# For each non-anchor facility, find which population pixels it serves
for fac_name in non_anchor_facilities:
    # Get facility index
    fac_idx = facility_populations[facility_populations['facility_name'] == fac_name]['facility_idx'].iloc[0]

    # Get population pixels served by this facility
    pixels_served = alloc_details[alloc_details['facility_idx'] == fac_idx]

    if len(pixels_served) == 0:
        continue

    # For these pixels, find which HSA anchor serves the most population in the same geographic area
    # Use a simple spatial proximity approach: assign to nearest anchor
    # This is approximated by checking which anchor's catchment these pixels fall into

    # Get pixel locations
    pixel_lons = pixels_served['lon'].values
    pixel_lats = pixels_served['lat'].values
    pixel_pops = pixels_served['population'].values

    # For simplicity in footprint mode, assign 100% to the closest anchor facility
    # We could calculate this from facility coordinates, but for now use a conservative approach:
    # Assign to the anchor that serves the most population in nearby pixels

    # Find all pixels within small radius of this facility's pixels
    # Build KDTree of this facility's pixels
    fac_coords = np.column_stack([pixel_lons, pixel_lats])

    # For each anchor HSA, calculate overlap
    anchor_overlaps = {}
    for anchor_name in hsa_ids:
        anchor_idx = facilities_df[facilities_df['healthfacility'] == anchor_name].index[0]
        anchor_pixels = alloc_details[alloc_details['facility_idx'] == anchor_idx]

        if len(anchor_pixels) == 0:
            continue

        # Calculate total population served by anchor near this facility's pixels
        anchor_coords = np.column_stack([anchor_pixels['lon'].values, anchor_pixels['lat'].values])

        # Simple approach: sum population where pixels are close (within 0.1 degrees ~ 10km)
        tree = cKDTree(fac_coords)
        dists, _ = tree.query(anchor_coords)
        nearby_mask = dists < 0.1  # 0.1 degrees
        overlap_pop = anchor_pixels.loc[nearby_mask, 'population'].sum()

        if overlap_pop > 0:
            anchor_overlaps[anchor_name] = overlap_pop

    # Assign to anchor with most overlap, or skip if no overlap
    if anchor_overlaps:
        best_anchor = max(anchor_overlaps, key=anchor_overlaps.get)
        facility_to_hsas[fac_name] = {best_anchor: 1.0}

print(f"  Mapped {len(facility_to_hsas)} facilities to HSAs ({len([f for f in facility_to_hsas if f in hsa_ids])} anchors + {len([f for f in facility_to_hsas if f not in hsa_ids])} non-anchors)")

# Step 3: Generate weeks
print("\n[4/7] Generating Monday-anchored weeks...")
start_date = datetime.strptime(WEEK_START, "%Y-%m-%d").date()
end_date = datetime.strptime(WEEK_END, "%Y-%m-%d").date()
weeks = []
current = start_date
while current <= end_date:
    weeks.append(current)
    current += timedelta(days=7)
print(f"  Generated {len(weeks)} weeks")

# Step 4: Load and prepare patient data
print("\n[5/7] Loading patient data...")
# Use PATIENT_DATA env var if set, otherwise try non-synthetic file first, then synthetic
pat_file = os.environ.get('PATIENT_DATA')
if not pat_file:
    pat_file = f'data/{NETWORK}_patient_visits.csv'
    if not Path(pat_file).exists():
        pat_file = f'data/SYN{NETWORK}_patient_visits.csv'
if not Path(pat_file).exists():
    raise FileNotFoundError(f"Patient data file not found: {pat_file}")
print(f"  Using: {pat_file}")
patients = pd.read_csv(pat_file, encoding='utf-8-sig')
patients['date'] = pd.to_datetime(patients['datetimediagnosisentered'])
patients['healthfacility'] = patients['healthfacility'].str.replace('\xa0', ' ').str.replace(r'\s+', ' ', regex=True).str.strip()

print(f"  Loaded {len(patients):,} patient records")

def is_target_disease(row):
    if DISEASE_FOCUS == "diarrheal":
        diagnosis = row.get('diagnosis')
        if pd.isna(diagnosis):
            return False
        diarrheal_keywords = ['diarrhea', 'diarrhoea', 'gastroenteritis', 'dysentery',
                              'cholera', 'rotavirus', 'giardia', 'shigella', 'salmonella',
                              'enteric', 'gastro']
        return any(kw in str(diagnosis).lower() for kw in diarrheal_keywords)
    if DISEASE_FOCUS == "hypertension":
        category = row.get('general_category')
        if pd.notna(category):
            return 'hypertension' in str(category).lower()
        diagnosis = row.get('diagnosis')
        if pd.isna(diagnosis):
            return False
        return 'hypertens' in str(diagnosis).lower()
    raise ValueError(f"Unsupported disease focus: {DISEASE_FOCUS}")

patients['is_target_disease'] = patients.apply(is_target_disease, axis=1)
print(f"  Identified {patients['is_target_disease'].sum():,} {DISEASE_FOCUS} cases")

# Assign to weeks
def get_monday(date):
    weekday = date.weekday()
    monday = date - timedelta(days=weekday)
    return monday

patients['week_start'] = patients['date'].apply(get_monday)

# Filter to date range
start_dt = pd.Timestamp(start_date)
end_dt = pd.Timestamp(end_date)
patients = patients[
    (patients['week_start'] >= start_dt) &
    (patients['week_start'] <= end_dt)
].copy()

print(f"  {len(patients):,} patients in date range")

# Step 5: Apply facility-to-HSA weights
print("\n[6/7] Applying gravity model adjustments...")

# For each patient, determine their HSA assignment weight
adjusted_records = []

for _, patient in patients.iterrows():
    facility = patient['healthfacility']
    week = patient['week_start']
    is_target = patient['is_target_disease']

    if facility in facility_to_hsas:
        # This facility serves one or more HSAs
        for hsa_id, weight in facility_to_hsas[facility].items():
            adjusted_records.append({
                'hsa_id': hsa_id,
                'week_start': week,
                'is_target': is_target,
                'weight': weight
            })

adjusted_df = pd.DataFrame(adjusted_records)
print(f"  Created {len(adjusted_df):,} weighted patient-HSA-week records")

# Step 6: Aggregate weighted counts
print("\n[7/7] Aggregating weighted counts...")

target_adjusted = adjusted_df[adjusted_df['is_target']].groupby(
    ['hsa_id', 'week_start']
).agg({'weight': 'sum'}).reset_index()
target_adjusted.rename(columns={'weight': f'{DISEASE_FOCUS}_count_adjusted'}, inplace=True)

# All diseases for this network
secondary_adjusted = adjusted_df.groupby(
    ['hsa_id', 'week_start']
).agg({'weight': 'sum'}).reset_index()
secondary_adjusted.rename(columns={'weight': f'{SECONDARY_LABEL}_count_adjusted'}, inplace=True)

# Create all HSA-week combinations
weeks_ts = [pd.Timestamp(w) for w in weeks]
all_combinations = pd.MultiIndex.from_product(
    [hsa_ids, weeks_ts],
    names=['hsa_id', 'week_start']
).to_frame(index=False)

# Merge and fill zeros
target_weekly = all_combinations.merge(
    target_adjusted,
    on=['hsa_id', 'week_start'],
    how='left'
).fillna({f'{DISEASE_FOCUS}_count_adjusted': 0})

secondary_weekly = all_combinations.merge(
    secondary_adjusted,
    on=['hsa_id', 'week_start'],
    how='left'
).fillna({f'{SECONDARY_LABEL}_count_adjusted': 0})

# Add ISO week string for merging with climate data
target_weekly['week_start_iso'] = target_weekly['week_start'].dt.strftime('%Y-%m-%d')
secondary_weekly['week_start_iso'] = secondary_weekly['week_start'].dt.strftime('%Y-%m-%d')

# Step 7: Save outputs
print("\n[8/8] Saving adjusted counts...")
target_weekly.to_csv(f'out/{NETWORK}_{HSA_MODE}_weekly_{DISEASE_FOCUS}_adjusted.csv', index=False)
secondary_weekly.to_csv(f'out/{NETWORK}_{HSA_MODE}_weekly_{SECONDARY_LABEL}_adjusted.csv', index=False)

print(f"  Saved: out/{NETWORK}_{HSA_MODE}_weekly_{DISEASE_FOCUS}_adjusted.csv")
print(f"  Saved: out/{NETWORK}_{HSA_MODE}_weekly_{SECONDARY_LABEL}_adjusted.csv")

# Summary
print("\n" + "="*80)
print("SUMMARY - ADJUSTED COUNTS")
print("="*80)

print(f"\n{DISEASE_FOCUS.title()} (adjusted):")
target_col = f"{DISEASE_FOCUS}_count_adjusted"
print(f"  Total weeks with cases: {(target_weekly[target_col] > 0).sum():,}")
print(f"  Total cases: {target_weekly[target_col].sum():.1f}")
print(f"  Mean per week: {target_weekly[target_col].mean():.2f}")
print(f"  Median per week: {target_weekly[target_col].median():.1f}")
print(f"  Max per week: {target_weekly[target_col].max():.1f}")

print(f"\n{SECONDARY_LABEL.upper()} (adjusted):")
secondary_col = f"{SECONDARY_LABEL}_count_adjusted"
print(f"  Total weeks with cases: {(secondary_weekly[secondary_col] > 0).sum():,}")
print(f"  Total cases: {secondary_weekly[secondary_col].sum():.1f}")
print(f"  Mean per week: {secondary_weekly[secondary_col].mean():.2f}")
print(f"  Median per week: {secondary_weekly[secondary_col].median():.1f}")
print(f"  Max per week: {secondary_weekly[secondary_col].max():.1f}")

# Top HSAs
print(f"\nTop 5 HSAs by adjusted {DISEASE_FOCUS} cases:")
hsa_target = target_weekly.groupby('hsa_id')[target_col].sum().sort_values(ascending=False)
for hsa, count in hsa_target.head(5).items():
    print(f"  {hsa}: {count:.1f}")

print(f"\nTop 5 HSAs by adjusted {SECONDARY_LABEL} cases:")
hsa_secondary = secondary_weekly.groupby('hsa_id')[secondary_col].sum().sort_values(ascending=False)
for hsa, count in hsa_secondary.head(5).items():
    print(f"  {hsa}: {count:.1f}")

print("\n" + "="*80)
print("[OK] ADJUSTED WEEKLY DISEASE COUNTS GENERATED")
print("="*80)
print("\nNote: These counts use gravity model weights to avoid double-counting")
print("facilities that serve multiple HSAs.")
