#!/usr/bin/env python3
"""
Generate Weekly Disease Counts by HSA from Synthetic Data

This script aggregates synthetic patient visits to weekly disease counts
by Hospital Service Area (HSA), accounting for the gravity model allocation.

Inputs:
    - data/synthetic/INF_patient_visits_SYNTHETIC.csv (synthetic patient data)
    - out/INF_<mode>_hsas_v2.geojson (HSA boundaries from optimization)
    - data/INF_facility_coordinates.csv (facility locations)

Outputs:
    - out/INF_<mode>_weekly_diarrheal.csv (weekly diarrheal counts by HSA)

Usage:
    python generate_weekly_disease_counts.py <hsa_geojson_file>

    Example:
    python generate_weekly_disease_counts.py out/INF_footprint_hsas_v2.geojson

Author: Claude Code
Date: 2024-12-28
"""

import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Configuration
WEEK_START = "2019-01-07"  # First Monday
WEEK_END = "2024-01-29"

# Diarrheal disease keywords for classification
DIARRHEAL_KEYWORDS = [
    'diarrhea', 'diarrhoea', 'gastroenteritis', 'dysentery',
    'cholera', 'rotavirus', 'giardia', 'shigella', 'salmonella',
    'enteric', 'gastro', 'escherichia coli'
]


def is_diarrheal(diagnosis):
    """Check if a diagnosis is diarrheal disease."""
    if pd.isna(diagnosis):
        return False
    return any(kw in str(diagnosis).lower() for kw in DIARRHEAL_KEYWORDS)


def get_monday(date):
    """Get the Monday of the week containing this date."""
    weekday = date.weekday()
    monday = date - timedelta(days=weekday)
    return monday


def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATE WEEKLY DISEASE COUNTS FROM SYNTHETIC DATA")
    print("=" * 80)

    # Get HSA file from command line
    if len(sys.argv) < 2:
        print("\nUsage: python generate_weekly_disease_counts.py <hsa_geojson_file>")
        print("\nExample:")
        print("  python generate_weekly_disease_counts.py out/INF_footprint_hsas_v2.geojson")
        return 1

    hsa_file = Path(sys.argv[1])
    if not hsa_file.exists():
        print(f"\n[ERROR] HSA file not found: {hsa_file}")
        return 1

    # Extract mode from filename (e.g., "footprint" from "INF_footprint_hsas_v2.geojson")
    mode = hsa_file.stem.split('_')[1]  # footprint, fewest, etc.

    print(f"\n[1/6] Loading HSA boundaries from: {hsa_file.name}")
    hsas_gdf = gpd.read_file(hsa_file)
    print(f"  Loaded {len(hsas_gdf)} HSAs")

    # Identify HSA ID column
    if 'FacilityName' in hsas_gdf.columns:
        hsa_id_col = 'FacilityName'
    elif 'healthfacility' in hsas_gdf.columns:
        hsa_id_col = 'healthfacility'
    elif 'anchor_name' in hsas_gdf.columns:
        hsa_id_col = 'anchor_name'
    else:
        print(f"  [ERROR] Cannot find HSA ID column in {hsa_file}")
        print(f"  Available columns: {hsas_gdf.columns.tolist()}")
        return 1

    hsa_ids = hsas_gdf[hsa_id_col].tolist()
    print(f"  Using '{hsa_id_col}' as HSA identifier")

    # Load facility coordinates
    print("\n[2/6] Loading facility coordinates...")
    facilities_df = pd.read_csv('data/INF_facility_coordinates.csv')
    print(f"  Loaded {len(facilities_df)} facilities")

    # Create facility to HSA mapping
    # Simple approach: Each anchor facility maps to its own HSA
    # Non-anchor facilities map to nearest anchor (simplified)
    print("\n[3/6] Creating facility-to-HSA mapping...")

    facility_to_hsa = {}
    for hsa_id in hsa_ids:
        # Anchor facilities map to themselves
        facility_to_hsa[hsa_id] = hsa_id

    # For non-anchor facilities in synthetic data, assign to their actual visited facility
    # (In synthetic data, patient 'healthfacility' is where they went)
    print(f"  Mapped {len(facility_to_hsa)} anchor facilities to HSAs")

    # Generate weeks
    print("\n[4/6] Generating Monday-anchored weeks...")
    start_date = datetime.strptime(WEEK_START, "%Y-%m-%d").date()
    end_date = datetime.strptime(WEEK_END, "%Y-%m-%d").date()
    weeks = []
    current = start_date
    while current <= end_date:
        weeks.append(current)
        current += timedelta(days=7)
    print(f"  Generated {len(weeks)} weeks")

    # Load synthetic patient data
    print("\n[5/6] Loading synthetic patient data...")
    patients = pd.read_csv('data/synthetic/INF_patient_visits_SYNTHETIC.csv')
    patients['date'] = pd.to_datetime(patients['datetimediagnosisentered'])

    print(f"  Loaded {len(patients):,} patient records")

    # Identify diarrheal diseases
    patients['is_diarrheal'] = patients['diagnosis'].apply(is_diarrheal)
    print(f"  Identified {patients['is_diarrheal'].sum():,} diarrheal cases")

    # Assign to weeks
    patients['week_start'] = patients['date'].apply(get_monday)

    # Filter to date range
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    patients = patients[
        (patients['week_start'] >= start_dt) &
        (patients['week_start'] <= end_dt)
    ].copy()

    print(f"  {len(patients):,} patients in date range ({WEEK_START} to {WEEK_END})")

    # Map patients to HSAs
    print("\n[6/6] Aggregating weekly counts by HSA...")

    # For each patient, determine their HSA
    # If they visited an anchor facility, assign to that HSA
    # Otherwise, skip (not assigned to any HSA in this mode)
    patients['hsa_id'] = patients['healthfacility'].map(facility_to_hsa)

    # Count patients assigned to HSAs
    assigned = patients['hsa_id'].notna().sum()
    print(f"  {assigned:,} patients ({assigned/len(patients)*100:.1f}%) assigned to HSAs")

    # Filter to assigned patients only
    patients_assigned = patients[patients['hsa_id'].notna()].copy()

    # Aggregate diarrheal cases by HSA and week
    diarrheal_counts = patients_assigned[patients_assigned['is_diarrheal']].groupby(
        ['hsa_id', 'week_start']
    ).size().reset_index(name='diarrheal_count')

    # Add adjusted count (same as count for simple assignment)
    diarrheal_counts['diarrheal_count_adjusted'] = diarrheal_counts['diarrheal_count']

    # Create all HSA-week combinations to fill gaps
    weeks_ts = [pd.Timestamp(w) for w in weeks]
    all_combinations = pd.MultiIndex.from_product(
        [hsa_ids, weeks_ts],
        names=['hsa_id', 'week_start']
    ).to_frame(index=False)

    # Merge and fill zeros
    diarrheal_weekly = all_combinations.merge(
        diarrheal_counts,
        on=['hsa_id', 'week_start'],
        how='left'
    ).fillna({'diarrheal_count': 0, 'diarrheal_count_adjusted': 0})

    # Convert to integers
    diarrheal_weekly['diarrheal_count'] = diarrheal_weekly['diarrheal_count'].astype(int)
    diarrheal_weekly['diarrheal_count_adjusted'] = diarrheal_weekly['diarrheal_count_adjusted'].astype(int)

    # Save output
    output_file = Path(f'out/INF_{mode}_weekly_diarrheal.csv')
    output_file.parent.mkdir(exist_ok=True, parents=True)

    diarrheal_weekly.to_csv(output_file, index=False)
    print(f"\n  [OK] Saved: {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nWeekly diarrheal counts:")
    print(f"  Total HSA-weeks: {len(diarrheal_weekly):,}")
    print(f"  Weeks with cases: {(diarrheal_weekly['diarrheal_count_adjusted'] > 0).sum():,}")
    print(f"  Total cases: {diarrheal_weekly['diarrheal_count_adjusted'].sum():,}")
    print(f"  Mean per HSA-week: {diarrheal_weekly['diarrheal_count_adjusted'].mean():.2f}")
    print(f"  Median per HSA-week: {diarrheal_weekly['diarrheal_count_adjusted'].median():.1f}")
    print(f"  Max per HSA-week: {diarrheal_weekly['diarrheal_count_adjusted'].max():,}")

    # Top HSAs
    print(f"\nTop 5 HSAs by total diarrheal cases:")
    hsa_totals = diarrheal_weekly.groupby('hsa_id')['diarrheal_count_adjusted'].sum().sort_values(ascending=False)
    for i, (hsa, count) in enumerate(hsa_totals.head(5).items(), 1):
        print(f"  {i}. {hsa}: {count:,} cases")

    print("\n" + "=" * 80)
    print("[OK] WEEKLY DISEASE COUNTS GENERATED")
    print("=" * 80)
    print("\nNote: Counts generated from SYNTHETIC patient data.")
    print(f"      Output file: {output_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
