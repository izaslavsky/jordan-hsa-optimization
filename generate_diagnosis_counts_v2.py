#!/usr/bin/env python3
"""
Generate Diagnosis Counts Using Authoritative Facility Coordinates (Version 2)

This script treats authoritative coordinate files as the source of truth for
facility names, spellings, governorates, and coordinates.

Logic:
1. Authoritative files define correct facility-governorate pairs
2. Patient data is matched to authoritative facilities by name
3. Governorate mismatches in patient data are reassigned to authoritative governorate
4. Small-count mismatches (<50) are flagged as likely data entry errors
5. Large-count mismatches (>=50) are flagged for manual review

Usage:
    python generate_diagnosis_counts_v2.py

Date: 2025-12-07
"""

import pandas as pd
from pathlib import Path
import sys
import argparse
import os
from datetime import datetime
from difflib import get_close_matches

# Directories
DATA_DIR = Path('data')
DEFAULT_OUT_DIR = os.environ.get("HSA_OUT_DIR", os.environ.get("PIPELINE_OUT_DIR", "out"))
OUT_DIR = Path(DEFAULT_OUT_DIR)

# Parameters
DUPLICATE_WINDOW_DAYS = 3
SMALL_COUNT_THRESHOLD = 50  # Counts below this are likely data entry errors

# Facility hierarchy for duplicate resolution
FACILITY_HIERARCHY = {
    'Primary Center': 1,
    'Comprehensive Center': 2,
    'Specialized Medical Center': 3,
    'Educational Hospital': 4,
    'Field Hospital': 4,
    'Hospital': 4
}

# Known facility name corrections (from patient data to authoritative)
FACILITY_NAME_CORRECTIONS = {
    'Al-Mafraq Gynocology and Pediatrics Hospital': 'Al-Mafraq Gynecology and Pediatrics Hospital',
    ' Hashemieh Housing Comprehensive Center': 'Hashemieh Housing Comprehensive Center',
    'Hashemieh Housing Comprehensive Center': 'Hashemieh Housing Comprehensive Center',  # Remove leading space
    'Dr. jamel Al-Totanji Hospital': 'Dr. Jamel Al-Totanji Hospital',
}


def standardize_facility_name(name: str) -> str:
    """Standardize facility names for matching."""
    if pd.isna(name):
        return ''

    # Convert to string, strip whitespace
    name = str(name).strip()

    # Remove extra whitespace
    name = ' '.join(name.split())

    # Apply known corrections
    if name in FACILITY_NAME_CORRECTIONS:
        name = FACILITY_NAME_CORRECTIONS[name]

    return name


def _resolve_data_file(network: str, suffix: str) -> Path:
    """Return data/NETWORK_suffix, falling back to data/SYNMODNETWORK_suffix."""
    real = DATA_DIR / f'{network}_{suffix}'
    if real.exists():
        return real
    synmod = DATA_DIR / f'SYNMOD{network}_{suffix}'
    if synmod.exists():
        return synmod
    raise FileNotFoundError(
        f"Data file not found as '{real.name}' or '{synmod.name}' in {DATA_DIR}"
    )


def load_authoritative_facilities(network: str) -> pd.DataFrame:
    """Load authoritative facility coordinates."""
    auth_file = _resolve_data_file(network, 'facility_coordinates.csv')

    if not auth_file.exists():
        raise FileNotFoundError(f"Authoritative file not found: {auth_file}")

    print(f"\nLoading authoritative facilities: {auth_file.name}")
    df = pd.read_csv(auth_file)

    # Standardize facility names
    df['healthfacility_std'] = df['healthfacility'].apply(standardize_facility_name)

    # Create composite key
    df['auth_key'] = df['healthfacility_std'] + '|' + df['governorate']

    print(f"  Loaded {len(df)} authoritative facilities")
    print(f"  Unique governorates: {df['governorate'].nunique()}")

    return df


def load_diagnosis_groups(network: str) -> dict:
    """Load diagnosis group mappings."""
    group_file = _resolve_data_file(network, 'groups_of_diagnoses.csv')

    if not group_file.exists():
        raise FileNotFoundError(f"Diagnosis groups file not found: {group_file}")

    print(f"\nLoading diagnosis groups: {group_file.name}")
    df = pd.read_csv(group_file)

    # Determine group column name
    if 'General_Diagnosis' in df.columns:
        group_col = 'General_Diagnosis'
    elif 'General_Category' in df.columns:
        group_col = 'General_Category'
    else:
        raise ValueError(f"Could not find diagnosis group column in {group_file}")

    # Create clean mapping
    diagnosis_map = df.set_index('Diagnosis')[group_col].to_dict()

    print(f"  Loaded {len(diagnosis_map)} diagnosis mappings")
    print(f"  Unique groups: {df[group_col].nunique()}")

    return diagnosis_map


def load_patient_data(network: str) -> pd.DataFrame:
    """Load patient visit data."""
    patient_file = _resolve_data_file(network, 'patient_visits.csv')

    if not patient_file.exists():
        raise FileNotFoundError(f"Patient data not found: {patient_file}")

    print(f"\nLoading patient data: {patient_file.name}")
    df = pd.read_csv(patient_file)

    # Standardize facility names
    df['healthfacility_std'] = df['healthfacility'].apply(standardize_facility_name)

    # Create composite key
    df['patient_key'] = df['healthfacility_std'] + '|' + df['governorate']

    # Parse dates
    df['visit_date'] = pd.to_datetime(df['datetimediagnosisentered'], errors='coerce')

    print(f"  Total records: {len(df):,}")
    print(f"  Unique patients: {df['patientid'].nunique():,}")
    print(f"  Unique facility-governorate pairs: {df['patient_key'].nunique()}")

    return df


def match_patient_to_authoritative(
    patient_df: pd.DataFrame,
    auth_df: pd.DataFrame,
    network: str
) -> tuple:
    """
    Match patient data to authoritative facilities.

    Returns:
        (matched_df, discrepancy_report)
    """
    print(f"\nMatching patient data to authoritative facilities...")

    # Get unique patient facility-governorate pairs with counts
    patient_pairs = patient_df.groupby(['healthfacility_std', 'governorate']).size().reset_index(name='diagnosis_count')

    # Track discrepancies
    discrepancies = []

    for _, row in patient_pairs.iterrows():
        facility_std = row['healthfacility_std']
        patient_gov = row['governorate']
        count = row['diagnosis_count']

        # Check if exact match exists in authoritative
        exact_match = auth_df[auth_df['auth_key'] == f"{facility_std}|{patient_gov}"]

        if len(exact_match) > 0:
            # Exact match - no discrepancy
            continue

        # No exact match - check if facility exists with different governorate
        facility_in_auth = auth_df[auth_df['healthfacility_std'] == facility_std]

        if len(facility_in_auth) == 0:
            # Facility not in authoritative file at all
            discrepancies.append({
                'patient_facility': facility_std,
                'patient_governorate': patient_gov,
                'diagnosis_count': count,
                'discrepancy_type': 'facility_not_found',
                'auth_facility': None,
                'auth_governorate': None,
                'action': 'exclude' if count >= SMALL_COUNT_THRESHOLD else 'exclude_small_count'
            })
        else:
            # Facility exists but with different governorate
            auth_gov = facility_in_auth.iloc[0]['governorate']
            discrepancies.append({
                'patient_facility': facility_std,
                'patient_governorate': patient_gov,
                'diagnosis_count': count,
                'discrepancy_type': 'governorate_mismatch',
                'auth_facility': facility_std,
                'auth_governorate': auth_gov,
                'action': 'reassign_to_auth_governorate'
            })

    discrepancy_df = pd.DataFrame(discrepancies)

    # Apply corrections to patient data
    # Create mapping: patient_key -> auth_key
    correction_map = {}

    for _, auth_row in auth_df.iterrows():
        facility_std = auth_row['healthfacility_std']
        auth_gov = auth_row['governorate']
        auth_key = auth_row['auth_key']

        # Map any patient_key with this facility to the auth_key
        # This handles both exact matches and governorate corrections
        for patient_gov in patient_df['governorate'].unique():
            patient_key = f"{facility_std}|{patient_gov}"
            correction_map[patient_key] = auth_key

    # Apply corrections
    patient_df['auth_key'] = patient_df['patient_key'].map(correction_map)

    # Flag records not in authoritative file
    patient_df['in_authoritative'] = patient_df['auth_key'].notna()

    # Split corrected auth_key back into facility and governorate
    patient_df[['healthfacility_corrected', 'governorate_corrected']] = patient_df['auth_key'].str.split('|', expand=True)

    # For records not in authoritative, keep original
    patient_df['healthfacility_corrected'] = patient_df['healthfacility_corrected'].fillna(patient_df['healthfacility_std'])
    patient_df['governorate_corrected'] = patient_df['governorate_corrected'].fillna(patient_df['governorate'])

    # Summary
    total_records = len(patient_df)
    matched_records = patient_df['in_authoritative'].sum()
    print(f"\n  Records matched to authoritative: {matched_records:,} ({matched_records/total_records*100:.1f}%)")
    print(f"  Records NOT in authoritative: {total_records - matched_records:,}")
    print(f"  Discrepancy types found: {len(discrepancy_df)}")

    return patient_df, discrepancy_df


def assign_diagnosis_groups(patient_df: pd.DataFrame, diagnosis_map: dict) -> pd.DataFrame:
    """Assign diagnosis groups to patient records."""
    print(f"\nAssigning diagnosis groups...")

    patient_df['diagnosis_group'] = patient_df['diagnosis'].map(diagnosis_map)
    patient_df['diagnosis_group'] = patient_df['diagnosis_group'].fillna('Unclassified')

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
    auth_df: pd.DataFrame,
    network: str,
    hsa_mode: str
) -> tuple:
    """Create diagnosis count tables."""
    print(f"\nCreating diagnosis count tables...")

    # Filter to only records in authoritative file
    df_filtered = patient_df[patient_df['in_authoritative']].copy()

    excluded = len(patient_df) - len(df_filtered)
    if excluded > 0:
        print(f"  [WARNING] Excluding {excluded:,} records not in authoritative file")

    # Use corrected names
    df_filtered['healthfacility'] = df_filtered['healthfacility_corrected']
    df_filtered['governorate'] = df_filtered['governorate_corrected']

    # Table 1: Total counts
    total_counts = df_filtered.groupby(['healthfacility', 'governorate']).size().reset_index(name='total_diagnoses')

    # Merge with authoritative to get metadata
    total_counts = total_counts.merge(
        auth_df[['healthfacility', 'governorate', 'healthfacilitytype', 'Latitude', 'Longitude']],
        on=['healthfacility', 'governorate'],
        how='left'
    )

    # Reorder columns
    total_counts = total_counts[['healthfacility', 'healthfacilitytype', 'governorate', 'total_diagnoses', 'Latitude', 'Longitude']]
    total_counts = total_counts.sort_values('total_diagnoses', ascending=False)

    # Save
    total_file = OUT_DIR / f'{network}_{hsa_mode}_diagnosis_counts_total.csv'
    total_counts.to_csv(total_file, index=False)
    print(f"  [OK] Saved: {total_file.name} ({len(total_counts)} facilities)")

    # Table 2: By diagnosis group
    group_counts = df_filtered.groupby(['healthfacility', 'governorate', 'diagnosis_group']).size().reset_index(name='diagnosis_count')

    # Merge with authoritative
    group_counts = group_counts.merge(
        auth_df[['healthfacility', 'governorate', 'healthfacilitytype', 'Latitude', 'Longitude']],
        on=['healthfacility', 'governorate'],
        how='left'
    )

    # Reorder columns
    group_counts = group_counts[['healthfacility', 'healthfacilitytype', 'governorate', 'diagnosis_group', 'diagnosis_count', 'Latitude', 'Longitude']]
    group_counts = group_counts.sort_values(['healthfacility', 'diagnosis_count'], ascending=[True, False])

    # Save
    group_file = OUT_DIR / f'{network}_{hsa_mode}_diagnosis_counts_by_group.csv'
    group_counts.to_csv(group_file, index=False)
    print(f"  [OK] Saved: {group_file.name} ({len(group_counts)} rows)")

    # Table 3: Pivot table
    pivot = df_filtered.pivot_table(
        index=['healthfacility', 'governorate'],
        columns='diagnosis_group',
        values='patientid',
        aggfunc='count',
        fill_value=0
    ).reset_index()

    # Add total column
    diagnosis_cols = [col for col in pivot.columns if col not in ['healthfacility', 'governorate']]
    pivot['total_diagnoses'] = pivot[diagnosis_cols].sum(axis=1)

    # Merge with authoritative
    pivot = pivot.merge(
        auth_df[['healthfacility', 'governorate', 'healthfacilitytype', 'Latitude', 'Longitude']],
        on=['healthfacility', 'governorate'],
        how='left'
    )

    # Reorder columns
    first_cols = ['healthfacility', 'healthfacilitytype', 'governorate', 'Latitude', 'Longitude', 'total_diagnoses']
    other_cols = [col for col in pivot.columns if col not in first_cols]
    pivot = pivot[first_cols + sorted(other_cols)]
    pivot = pivot.sort_values('total_diagnoses', ascending=False)

    # Save
    pivot_file = OUT_DIR / f'{network}_{hsa_mode}_diagnosis_counts_pivot.csv'
    pivot.to_csv(pivot_file, index=False)
    print(f"  [OK] Saved: {pivot_file.name} ({len(pivot)} facilities)")

    return total_counts, group_counts, pivot


def generate_discrepancy_report(discrepancies_by_network: dict) -> str:
    """Generate report of patient data discrepancies."""
    report = []
    report.append("PATIENT DATA DISCREPANCY REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("\nThis report identifies facility-governorate pairs in patient data that")
    report.append("do not match the authoritative coordinate files.\n")

    for network, discrepancies in discrepancies_by_network.items():
        report.append("\n" + "=" * 80)
        report.append(f"{network} NETWORK")
        report.append("=" * 80)

        if len(discrepancies) == 0:
            report.append("\nNo discrepancies found - all patient data matches authoritative file!")
        else:
            report.append(f"\nTotal discrepancies: {len(discrepancies)}")
            report.append(f"Total affected diagnoses: {discrepancies['diagnosis_count'].sum():,}\n")

            for dtype in discrepancies['discrepancy_type'].unique():
                subset = discrepancies[discrepancies['discrepancy_type'] == dtype]
                report.append(f"\n{dtype.upper().replace('_', ' ')} ({len(subset)} cases):")
                report.append("-" * 80)

                for _, row in subset.iterrows():
                    report.append(f"\nPatient data:")
                    report.append(f"  Facility:    {row['patient_facility']}")
                    report.append(f"  Governorate: {row['patient_governorate']}")
                    report.append(f"  Diagnoses:   {row['diagnosis_count']:,}")

                    if row['auth_facility']:
                        report.append("Authoritative file:")
                        report.append(f"  Facility:    {row['auth_facility']}")
                        report.append(f"  Governorate: {row['auth_governorate']}")
                        report.append(f"Action: {row['action']}")
                    else:
                        report.append(f"Action: {row['action']} (facility not in authoritative file)")

    report.append("\n\n" + "=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    report.append(f"\nSmall count threshold: {SMALL_COUNT_THRESHOLD} diagnoses")
    report.append("\nActions taken:")
    report.append("  - reassign_to_auth_governorate: Moved diagnoses to correct governorate")
    report.append("  - exclude: Excluded from final counts (facility not in authoritative)")

    return '\n'.join(report)


def main():
    """Main execution function."""
    global OUT_DIR
    parser = argparse.ArgumentParser(description="Generate diagnosis counts from authoritative facilities (v2)")
    parser.add_argument("--networks", default="INF,NCD")
    parser.add_argument("--hsa-mode", default="footprint")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    networks = [n.strip().upper() for n in args.networks.split(',') if n.strip()]

    print("=" * 80)
    print("GENERATE DIAGNOSIS COUNTS FROM AUTHORITATIVE FACILITIES (V2)")
    print("=" * 80)
    print(f"Networks: {', '.join(networks)}")

    try:
        totals = {}
        discrepancies_by_network = {}

        for network in networks:
            print("\n" + "=" * 80)
            print(f"PROCESSING {network} NETWORK")
            print("=" * 80)

            auth = load_authoritative_facilities(network)
            diag_map = load_diagnosis_groups(network)
            patients = load_patient_data(network)

            patients, discrepancies = match_patient_to_authoritative(patients, auth, network)
            patients = assign_diagnosis_groups(patients, diag_map)
            patients = deduplicate_diagnoses(patients)

            total, by_group, pivot = create_diagnosis_tables(patients, auth, network, args.hsa_mode)
            totals[network] = total
            discrepancies_by_network[network] = discrepancies

        # Generate discrepancy report
        print("\n" + "=" * 80)
        print("GENERATING DISCREPANCY REPORT")
        print("=" * 80)

        report = generate_discrepancy_report(discrepancies_by_network)

        report_prefix = f"{'_'.join(networks)}_{args.hsa_mode}"
        report_file = OUT_DIR / f'{report_prefix}_patient_data_discrepancies.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n[OK] Saved: {report_file.name}")

        # Print summary
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)

        for network, total in totals.items():
            print(f"\n{network} Network:")
            print(f"  Total diagnoses: {total['total_diagnoses'].sum():,}")
            print(f"  Facilities: {len(total)}")
            print(f"  Discrepancies: {len(discrepancies_by_network[network])}")

        print(f"\nOutput files:")
        for network in networks:
            print(f"  - {network}_{args.hsa_mode}_diagnosis_counts_total.csv")
            print(f"  - {network}_{args.hsa_mode}_diagnosis_counts_by_group.csv")
            print(f"  - {network}_{args.hsa_mode}_diagnosis_counts_pivot.csv")
        print(f"  - {report_prefix}_patient_data_discrepancies.txt")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
