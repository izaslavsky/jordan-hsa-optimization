#!/usr/bin/env python3
"""
Generate Diagnosis Counts from Synthetic Patient Data

This script aggregates synthetic patient visit data to create facility-level
diagnosis counts needed for HSA optimization.

Inputs:
    - data/synthetic/INF_patient_visits_SYNTHETIC.csv
    - data/synthetic/NCD_patient_visits_SYNTHETIC.csv
    - data/INF_facility_coordinates.csv
    - data/NCD_facility_coordinates.csv
    - data/INF_groups_of_diagnoses.csv
    - data/NCD_groups_of_diagnoses.csv

Outputs:
    - out/INF_diagnosis_counts_total.csv
    - out/INF_diagnosis_counts_by_group.csv
    - out/INF_diagnosis_counts_pivot.csv
    - out/NCD_diagnosis_counts_total.csv
    - out/NCD_diagnosis_counts_by_group.csv
    - out/NCD_diagnosis_counts_pivot.csv

Usage:
    python generate_diagnosis_counts.py

Author: Claude Code
Date: 2024-12-28
"""

import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# Directories
DATA_DIR = Path('data')
SYNTHETIC_DIR = DATA_DIR / 'synthetic'
OUT_DIR = Path('out')

# Parameters
DUPLICATE_WINDOW_DAYS = 3

# Facility hierarchy for duplicate resolution
FACILITY_HIERARCHY = {
    'Primary Center': 1,
    'Comprehensive Center': 2,
    'Specialized Medical Center': 3,
    'Educational Hospital': 4,
    'Field Hospital': 4,
    'Hospital': 4
}


def load_facility_coordinates(network: str) -> pd.DataFrame:
    """Load facility coordinates file."""
    coord_file = DATA_DIR / f'{network}_facility_coordinates.csv'

    if not coord_file.exists():
        raise FileNotFoundError(f"Coordinate file not found: {coord_file}")

    print(f"\nLoading facility coordinates: {coord_file.name}")
    df = pd.read_csv(coord_file)

    print(f"  Loaded {len(df)} facilities")

    return df


def load_diagnosis_groups(network: str) -> dict:
    """Load diagnosis group mappings."""
    group_file = DATA_DIR / f'{network}_groups_of_diagnoses.csv'

    if not group_file.exists():
        raise FileNotFoundError(f"Diagnosis groups file not found: {group_file}")

    print(f"\nLoading diagnosis groups: {group_file.name}")
    df = pd.read_csv(group_file)

    # Determine group column name
    if 'General_Diagnosis' in df.columns:
        group_col = 'General_Diagnosis'
    elif 'General_Category' in df.columns:
        group_col = 'General_Category'
    elif 'general_category' in df.columns:
        group_col = 'general_category'
    else:
        raise ValueError(f"Could not find diagnosis group column in {group_file}")

    # Create clean mapping
    diagnosis_map = df.set_index('Diagnosis')[group_col].to_dict()

    print(f"  Loaded {len(diagnosis_map)} diagnosis mappings")
    print(f"  Unique groups: {df[group_col].nunique()}")

    return diagnosis_map


def load_patient_data(network: str) -> pd.DataFrame:
    """Load synthetic patient visit data."""
    patient_file = SYNTHETIC_DIR / f'{network}_patient_visits_SYNTHETIC.csv'

    if not patient_file.exists():
        raise FileNotFoundError(f"Synthetic patient data not found: {patient_file}")

    print(f"\nLoading synthetic patient data: {patient_file.name}")
    df = pd.read_csv(patient_file)

    # Parse dates
    df['visit_date'] = pd.to_datetime(df['datetimediagnosisentered'], errors='coerce')

    print(f"  Total records: {len(df):,}")
    print(f"  Unique patients: {df['patientid'].nunique():,}")
    print(f"  Unique facilities: {df['healthfacility'].nunique()}")

    return df


def assign_diagnosis_groups(patient_df: pd.DataFrame, diagnosis_map: dict) -> pd.DataFrame:
    """Assign diagnosis groups to patient records."""
    print(f"\nAssigning diagnosis groups...")

    patient_df['diagnosis_group'] = patient_df['diagnosis'].map(diagnosis_map)
    patient_df['diagnosis_group'] = patient_df['diagnosis_group'].fillna(patient_df['general_category'])

    with_groups = (~patient_df['diagnosis_group'].isna()).sum()
    total = len(patient_df)

    print(f"  Diagnoses with groups: {with_groups:,} ({with_groups/total*100:.1f}%)")

    return patient_df


def deduplicate_diagnoses(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate diagnoses using the 3-day window rule."""
    print(f"\nDeduplicating diagnoses...")

    # Add facility hierarchy level
    patient_df['facility_level'] = patient_df['healthfacilitytype'].map(FACILITY_HIERARCHY)
    patient_df['facility_level'] = patient_df['facility_level'].fillna(1)

    # Sort
    df_sorted = patient_df.sort_values(
        ['patientid', 'diagnosis_group', 'visit_date', 'facility_level'],
        ascending=[True, True, True, False]
    ).copy()

    # Calculate days since previous visit
    df_sorted['prev_visit_date'] = df_sorted.groupby(['patientid', 'diagnosis_group'])['visit_date'].shift(1)
    df_sorted['days_since_prev'] = (df_sorted['visit_date'] - df_sorted['prev_visit_date']).dt.days
    df_sorted['prev_facility_level'] = df_sorted.groupby(['patientid', 'diagnosis_group'])['facility_level'].shift(1)

    # Mark as duplicate
    df_sorted['is_duplicate'] = (
        (df_sorted['days_since_prev'] <= DUPLICATE_WINDOW_DAYS) &
        (df_sorted['facility_level'] <= df_sorted['prev_facility_level'])
    )

    duplicates = df_sorted['is_duplicate'].sum()
    original = len(df_sorted)

    # Remove duplicates
    df_dedup = df_sorted[~df_sorted['is_duplicate']].copy()

    print(f"  Original diagnoses: {original:,}")
    print(f"  Duplicates removed: {duplicates:,} ({duplicates/original*100:.1f}%)")
    print(f"  Unique diagnoses: {len(df_dedup):,}")

    return df_dedup


def create_diagnosis_tables(
    patient_df: pd.DataFrame,
    coord_df: pd.DataFrame,
    network: str
) -> tuple:
    """Create diagnosis count tables."""
    print(f"\nCreating diagnosis count tables...")

    # Table 1: Total counts by facility
    total_counts = patient_df.groupby('healthfacility').size().reset_index(name='total_diagnoses')

    # Merge with coordinates to get metadata
    total_counts = total_counts.merge(
        coord_df[['healthfacility', 'healthfacilitytype', 'governorate', 'lat', 'lon']],
        on='healthfacility',
        how='left'
    )

    # Reorder columns
    total_counts = total_counts[['healthfacility', 'healthfacilitytype', 'governorate', 'total_diagnoses', 'lat', 'lon']]
    total_counts = total_counts.sort_values('total_diagnoses', ascending=False)

    # Save
    total_file = OUT_DIR / f'{network}_diagnosis_counts_total.csv'
    total_counts.to_csv(total_file, index=False)
    print(f"  [OK] Saved: {total_file.name} ({len(total_counts)} facilities)")

    # Table 2: By diagnosis group
    group_counts = patient_df.groupby(['healthfacility', 'diagnosis_group']).size().reset_index(name='diagnosis_count')

    # Merge with coordinates
    group_counts = group_counts.merge(
        coord_df[['healthfacility', 'healthfacilitytype', 'governorate', 'lat', 'lon']],
        on='healthfacility',
        how='left'
    )

    # Reorder columns
    group_counts = group_counts[['healthfacility', 'healthfacilitytype', 'governorate', 'diagnosis_group', 'diagnosis_count', 'lat', 'lon']]
    group_counts = group_counts.sort_values(['healthfacility', 'diagnosis_count'], ascending=[True, False])

    # Save
    group_file = OUT_DIR / f'{network}_diagnosis_counts_by_group.csv'
    group_counts.to_csv(group_file, index=False)
    print(f"  [OK] Saved: {group_file.name} ({len(group_counts)} rows)")

    # Table 3: Pivot table
    pivot = patient_df.pivot_table(
        index='healthfacility',
        columns='diagnosis_group',
        values='patientid',
        aggfunc='count',
        fill_value=0
    ).reset_index()

    # Add total column
    diagnosis_cols = [col for col in pivot.columns if col != 'healthfacility']
    pivot['total_diagnoses'] = pivot[diagnosis_cols].sum(axis=1)

    # Merge with coordinates
    pivot = pivot.merge(
        coord_df[['healthfacility', 'healthfacilitytype', 'governorate', 'lat', 'lon']],
        on='healthfacility',
        how='left'
    )

    # Reorder columns
    first_cols = ['healthfacility', 'healthfacilitytype', 'governorate', 'lat', 'lon', 'total_diagnoses']
    other_cols = [col for col in pivot.columns if col not in first_cols]
    pivot = pivot[first_cols + sorted(other_cols)]
    pivot = pivot.sort_values('total_diagnoses', ascending=False)

    # Save
    pivot_file = OUT_DIR / f'{network}_diagnosis_counts_pivot.csv'
    pivot.to_csv(pivot_file, index=False)
    print(f"  [OK] Saved: {pivot_file.name} ({len(pivot)} facilities)")

    return total_counts, group_counts, pivot


def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATE DIAGNOSIS COUNTS FROM SYNTHETIC DATA")
    print("=" * 80)

    # Create output directory
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    try:
        # Process INF network
        print("\n" + "=" * 80)
        print("PROCESSING INF NETWORK")
        print("=" * 80)

        inf_coords = load_facility_coordinates('INF')
        inf_diag_map = load_diagnosis_groups('INF')
        inf_patients = load_patient_data('INF')

        inf_patients = assign_diagnosis_groups(inf_patients, inf_diag_map)
        inf_patients = deduplicate_diagnoses(inf_patients)

        inf_total, inf_by_group, inf_pivot = create_diagnosis_tables(inf_patients, inf_coords, 'INF')

        # Process NCD network
        print("\n" + "=" * 80)
        print("PROCESSING NCD NETWORK")
        print("=" * 80)

        ncd_coords = load_facility_coordinates('NCD')
        ncd_diag_map = load_diagnosis_groups('NCD')
        ncd_patients = load_patient_data('NCD')

        ncd_patients = assign_diagnosis_groups(ncd_patients, ncd_diag_map)
        ncd_patients = deduplicate_diagnoses(ncd_patients)

        ncd_total, ncd_by_group, ncd_pivot = create_diagnosis_tables(ncd_patients, ncd_coords, 'NCD')

        # Print summary
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)

        print(f"\nINF Network:")
        print(f"  Total diagnoses: {inf_total['total_diagnoses'].sum():,}")
        print(f"  Facilities: {len(inf_total)}")

        print(f"\nNCD Network:")
        print(f"  Total diagnoses: {ncd_total['total_diagnoses'].sum():,}")
        print(f"  Facilities: {len(ncd_total)}")

        print(f"\nOutput files in '{OUT_DIR}':")
        print(f"  1. INF_diagnosis_counts_total.csv")
        print(f"  2. INF_diagnosis_counts_by_group.csv")
        print(f"  3. INF_diagnosis_counts_pivot.csv")
        print(f"  4. NCD_diagnosis_counts_total.csv")
        print(f"  5. NCD_diagnosis_counts_by_group.csv")
        print(f"  6. NCD_diagnosis_counts_pivot.csv")

        print("\nNote: These counts are generated from SYNTHETIC patient data")
        print("      and are suitable for demonstration purposes.")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
