"""
ML Dataset Preparation Script
==============================

Purpose: Merge climate data (108 CSV files) with diarrheal disease data
         to create a unified dataset for machine learning modeling.

Input:
    - Climate files: out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE/*.csv
    - Diagnosis data: out/INF_footprint_weekly_diarrheal.csv

Output:
    - modeling_dataset_full.csv: Complete merged dataset
    - modeling_dataset_train.csv: Training set
    - modeling_dataset_val.csv: Validation set
    - modeling_dataset_test.csv: Test set
    - modeling_dataset_metadata.json: Feature descriptions

Author: ML Modeling Team
Date: 2024-12-13
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories
CLIMATE_DIR = Path("out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD/FINAL_HSA_CLIMATE")
OUTPUT_DIR = Path("out/modeling")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Input files
DIAGNOSIS_FILE = Path("out/INF_footprint_weekly_diarrheal.csv")

# Climate file suffixes
CLIMATE_SUFFIXES = [
    'precip_lags.csv',
    'tempdew_wind_lags.csv',
    'evapERA5_lags.csv',
    'soilmoistERA5_lags.csv',
    'water_balance.csv',
    'elevation_by_week.csv'
]

# Train/validation/test split (weeks)
TRAIN_END_WEEK = 18  # First 18 weeks for training
VAL_END_WEEK = 22    # Weeks 19-22 for validation
# Weeks 23-26 for testing

# Random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_facility_name(name):
    """Standardize facility names for matching"""
    return name.strip().replace('  ', ' ')


def load_climate_file(filepath):
    """Load a climate CSV file and clean it"""
    df = pd.read_csv(filepath)

    # Drop unnecessary columns
    cols_to_drop = [col for col in df.columns if col in ['system:index', '.geo']]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Clean facility name
    if 'FacilityName' in df.columns:
        df['FacilityName'] = df['FacilityName'].apply(clean_facility_name)

    # Convert week_start to datetime
    if 'week_start' in df.columns:
        df['week_start'] = pd.to_datetime(df['week_start'])

    return df


def merge_climate_files_for_hsa(hsa_name, climate_dir, suffixes):
    """
    Merge all 6 climate file types for a single HSA

    Args:
        hsa_name: Name of the HSA/facility
        climate_dir: Directory containing climate files
        suffixes: List of file suffixes to merge

    Returns:
        Merged DataFrame with all climate variables
    """
    dfs = []

    for suffix in suffixes:
        # Construct filename
        filename = f"HSA_{hsa_name}_{suffix}"
        filepath = climate_dir / filename

        if not filepath.exists():
            print(f"  [!]  Missing: {filename}")
            continue

        # Load file
        df = load_climate_file(filepath)

        # Keep only unique columns (avoid duplicating FacilityName, week_start)
        if len(dfs) > 0:
            # Keep only new columns (excluding FacilityName and week_start)
            existing_cols = set()
            for existing_df in dfs:
                existing_cols.update(existing_df.columns)

            cols_to_keep = ['FacilityName', 'week_start']
            cols_to_keep += [col for col in df.columns if col not in existing_cols]
            df = df[cols_to_keep]

        dfs.append(df)

    # Merge all dataframes
    if len(dfs) == 0:
        return None

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(
            df,
            on=['FacilityName', 'week_start'],
            how='outer'
        )

    return merged


def get_all_hsa_names(climate_dir):
    """Extract unique HSA names from climate filenames"""
    hsa_names = set()

    for filepath in climate_dir.glob("HSA_*_precip_lags.csv"):
        # Extract HSA name from filename
        filename = filepath.stem  # Remove .csv
        # Remove "HSA_" prefix and "_precip_lags" suffix
        hsa_name = filename.replace("HSA_", "").replace("_precip_lags", "")
        hsa_names.add(hsa_name)

    return sorted(list(hsa_names))


def create_temporal_features(df):
    """Create temporal features from week_start"""
    df = df.copy()

    df['week_of_year'] = df['week_start'].dt.isocalendar().week
    df['month'] = df['week_start'].dt.month
    df['quarter'] = df['week_start'].dt.quarter

    # Season (Northern Hemisphere)
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    })

    # Days since start of study
    min_date = df['week_start'].min()
    df['days_since_start'] = (df['week_start'] - min_date).dt.days

    # Week number (sequential, starting from 1)
    df = df.sort_values(['hsa_id', 'week_start'])
    df['week_number'] = df.groupby('hsa_id').cumcount() + 1

    return df


def create_interaction_features(df):
    """Create interaction and derived climate features"""
    df = df.copy()

    # Temperature × Precipitation interaction (if both exist)
    if 'T_mean_week_C' in df.columns and 'P_total_week' in df.columns:
        df['temp_precip_interaction'] = df['T_mean_week_C'] * df['P_total_week']

    # Extreme heat indicator
    if 'T_max_week_C' in df.columns:
        df['extreme_heat'] = (df['T_max_week_C'] > 35).astype(int)

    # Heavy rain indicator
    if 'P_total_week' in df.columns:
        df['heavy_rain'] = (df['P_total_week'] > 10).astype(int)

    # Heat-moisture stress (if heat index and wetday fraction exist)
    if 'heat_index_week_C' in df.columns and 'wetday_frac_week' in df.columns:
        df['heat_moisture_stress'] = df['heat_index_week_C'] * (1 - df['wetday_frac_week'])

    # Cumulative precipitation (last 4 weeks)
    if all(f'P_sum_lag_w-{i}' in df.columns for i in [1, 2, 3]):
        df['cumulative_precip_4weeks'] = (
            df.get('P_total_week', 0) +
            df['P_sum_lag_w-1'] +
            df['P_sum_lag_w-2'] +
            df['P_sum_lag_w-3']
        )

    return df


def feature_selection_by_importance(df, target_col='diarrheal_count_adjusted',
                                    max_features=40):
    """
    Select top features by correlation with target

    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        max_features: Maximum number of features to keep

    Returns:
        List of selected feature names
    """
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in [
        'hsa_id', 'week_start', 'week_start_iso', 'FacilityName',
        target_col, 'diarrheal_count', 'week_number', 'days_since_start',
        'week_of_year', 'month', 'quarter', 'season', '.geo', 'system:index'
    ]]

    # Compute correlations
    correlations = df[feature_cols].corrwith(df[target_col]).abs()

    # Sort by correlation
    top_features = correlations.nlargest(max_features).index.tolist()

    return top_features


def remove_highly_correlated_features(df, features, threshold=0.95):
    """
    Remove features that are highly correlated with each other

    Args:
        df: DataFrame
        features: List of feature column names
        threshold: Correlation threshold above which to remove features

    Returns:
        List of features after removing highly correlated ones
    """
    # Compute correlation matrix
    corr_matrix = df[features].corr().abs()

    # Upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation > threshold
    to_drop = [column for column in upper_triangle.columns
               if any(upper_triangle[column] > threshold)]

    # Keep features not in to_drop list
    selected_features = [f for f in features if f not in to_drop]

    return selected_features


def split_temporal_data(df, train_end_week, val_end_week):
    """
    Split data temporally into train/validation/test sets

    Args:
        df: Complete DataFrame
        train_end_week: Last week number for training
        val_end_week: Last week number for validation

    Returns:
        train_df, val_df, test_df
    """
    train_df = df[df['week_number'] <= train_end_week].copy()
    val_df = df[(df['week_number'] > train_end_week) &
                (df['week_number'] <= val_end_week)].copy()
    test_df = df[df['week_number'] > val_end_week].copy()

    return train_df, val_df, test_df


def create_metadata(df, selected_features, output_path):
    """
    Create metadata JSON file documenting the dataset

    Args:
        df: Complete DataFrame
        selected_features: List of selected feature names
        output_path: Path to save metadata JSON
    """
    metadata = {
        'dataset_info': {
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_records': int(len(df)),
            'total_features': int(len(selected_features)),
            'date_range': {
                'start': df['week_start'].min().strftime('%Y-%m-%d'),
                'end': df['week_start'].max().strftime('%Y-%m-%d'),
                'weeks': int(df['week_number'].max())
            },
            'hsas': {
                'count': int(df['hsa_id'].nunique()),
                'names': sorted(df['hsa_id'].unique().tolist())
            }
        },
        'target_variable': {
            'name': 'diarrheal_count_adjusted',
            'description': 'Weekly adjusted count of diarrheal disease cases',
            'statistics': {
                'mean': float(df['diarrheal_count_adjusted'].mean()),
                'std': float(df['diarrheal_count_adjusted'].std()),
                'min': float(df['diarrheal_count_adjusted'].min()),
                'max': float(df['diarrheal_count_adjusted'].max()),
                'median': float(df['diarrheal_count_adjusted'].median()),
                'zeros': int((df['diarrheal_count_adjusted'] == 0).sum().item()),
                'zero_percentage': float((df['diarrheal_count_adjusted'] == 0).mean() * 100)
            }
        },
        'features': {},
        'temporal_split': {
            'train_weeks': f'1-{TRAIN_END_WEEK}',
            'validation_weeks': f'{TRAIN_END_WEEK+1}-{VAL_END_WEEK}',
            'test_weeks': f'{VAL_END_WEEK+1}+',
            'random_seed': RANDOM_SEED
        }
    }

    # Add feature metadata
    for feat in selected_features:
        if feat in df.columns:
            metadata['features'][feat] = {
                'mean': float(df[feat].mean()),
                'std': float(df[feat].std()),
                'min': float(df[feat].min()),
                'max': float(df[feat].max()),
                'missing_count': int(df[feat].isna().sum().item() if hasattr(df[feat].isna().sum(), 'item') else df[feat].isna().sum()),
                'missing_percentage': float(df[feat].isna().mean() * 100)
            }

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Metadata saved to {output_path}")

    return metadata


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def main():
    """Main execution function"""

    print("="*80)
    print("ML DATASET PREPARATION")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Climate directory: {CLIMATE_DIR}")
    print(f"Diagnosis file: {DIAGNOSIS_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)

    # -------------------------------------------------------------------------
    # STEP 1: Get all HSA names
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Identifying HSAs...")
    hsa_names = get_all_hsa_names(CLIMATE_DIR)
    print(f"  Found {len(hsa_names)} HSAs")
    for i, name in enumerate(hsa_names, 1):
        print(f"    {i:2d}. {name}")

    # -------------------------------------------------------------------------
    # STEP 2: Merge climate files for each HSA
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Merging climate files for each HSA...")
    all_hsa_data = []

    for i, hsa_name in enumerate(hsa_names, 1):
        print(f"\n  [{i}/{len(hsa_names)}] Processing: {hsa_name}")

        hsa_df = merge_climate_files_for_hsa(hsa_name, CLIMATE_DIR, CLIMATE_SUFFIXES)

        if hsa_df is None:
            print(f"    [X] Failed to merge files for {hsa_name}")
            continue

        # Add HSA identifier
        hsa_df['hsa_id'] = hsa_name

        all_hsa_data.append(hsa_df)
        print(f"    [OK] Merged {len(hsa_df)} weeks, {len(hsa_df.columns)} features")

    # Concatenate all HSAs
    if len(all_hsa_data) == 0:
        print("\n[X] ERROR: No HSA data could be merged!")
        return

    climate_df = pd.concat(all_hsa_data, ignore_index=True)
    print(f"\n  [OK] Combined climate data: {len(climate_df)} records, {len(climate_df.columns)} columns")

    # -------------------------------------------------------------------------
    # STEP 3: Load diagnosis data
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Loading diagnosis data...")

    if not DIAGNOSIS_FILE.exists():
        print(f"  [X] ERROR: Diagnosis file not found: {DIAGNOSIS_FILE}")
        return

    diagnosis_df = pd.read_csv(DIAGNOSIS_FILE)
    diagnosis_df['week_start'] = pd.to_datetime(diagnosis_df['week_start'])

    # Clean HSA names for matching
    if 'hsa_id' in diagnosis_df.columns:
        diagnosis_df['hsa_id'] = diagnosis_df['hsa_id'].apply(clean_facility_name)

    print(f"  [OK] Loaded {len(diagnosis_df)} diagnosis records")
    print(f"    Date range: {diagnosis_df['week_start'].min()} to {diagnosis_df['week_start'].max()}")
    print(f"    HSAs: {diagnosis_df['hsa_id'].nunique()}")

    # -------------------------------------------------------------------------
    # STEP 4: Merge climate and diagnosis data
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Merging climate and diagnosis data...")

    # Merge on hsa_id and week_start
    merged_df = climate_df.merge(
        diagnosis_df[['hsa_id', 'week_start', 'diarrheal_count']],
        on=['hsa_id', 'week_start'],
        how='left'
    )

    # Fill missing diarrheal counts with 0 (weeks with no cases)
    merged_df['diarrheal_count'] = merged_df['diarrheal_count'].fillna(0)

    # Use diarrheal_count as the adjusted count (or load adjusted if available)
    if 'diarrheal_count_adjusted' not in merged_df.columns:
        merged_df['diarrheal_count_adjusted'] = merged_df['diarrheal_count']

    print(f"  [OK] Merged dataset: {len(merged_df)} records")
    print(f"    Features: {len(merged_df.columns)} columns")
    print(f"    Missing diagnosis data: {merged_df['diarrheal_count_adjusted'].isna().sum()} records")

    # -------------------------------------------------------------------------
    # STEP 5: Data cleaning
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Cleaning data...")

    # Remove rows with missing target
    initial_len = len(merged_df)
    merged_df = merged_df.dropna(subset=['diarrheal_count_adjusted'])
    print(f"  [OK] Removed {initial_len - len(merged_df)} rows with missing target")

    # Check for missing climate features
    climate_cols = [col for col in merged_df.columns if col not in [
        'hsa_id', 'FacilityName', 'week_start', 'week_start_iso',
        'diarrheal_count', 'diarrheal_count_adjusted', '.geo', 'system:index'
    ]]

    missing_summary = merged_df[climate_cols].isna().sum()
    missing_features = missing_summary[missing_summary > 0]

    if len(missing_features) > 0:
        print(f"\n  [!]  Features with missing values:")
        for feat, count in missing_features.items():
            pct = (count / len(merged_df)) * 100
            print(f"    {feat}: {count} ({pct:.1f}%)")

        # Option 1: Drop rows with any missing climate data
        print(f"\n  Removing rows with missing climate data...")
        merged_df = merged_df.dropna(subset=climate_cols)
        print(f"  [OK] {len(merged_df)} complete records remaining")
    else:
        print(f"  [OK] No missing climate data")

    # -------------------------------------------------------------------------
    # STEP 6: Feature engineering
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Creating engineered features...")

    # Temporal features
    merged_df = create_temporal_features(merged_df)
    print(f"  [OK] Added temporal features: week_of_year, month, season, week_number")

    # Interaction features
    merged_df = create_interaction_features(merged_df)
    print(f"  [OK] Added interaction features")

    # -------------------------------------------------------------------------
    # STEP 7: Feature selection
    # -------------------------------------------------------------------------
    print("\n[STEP 7] Selecting features...")

    # Get all potential features
    exclude_cols = ['hsa_id', 'FacilityName', 'week_start', 'week_start_iso',
                    'diarrheal_count', 'diarrheal_count_adjusted', '.geo', 'system:index']
    all_features = [col for col in merged_df.columns if col not in exclude_cols]

    print(f"  Total available features: {len(all_features)}")

    # Select top features by correlation with target
    top_features = feature_selection_by_importance(
        merged_df,
        target_col='diarrheal_count_adjusted',
        max_features=60
    )
    print(f"  [OK] Selected top 60 features by correlation with target")

    # Remove highly correlated features
    selected_features = remove_highly_correlated_features(
        merged_df,
        top_features,
        threshold=0.95
    )
    print(f"  [OK] Removed highly correlated features (r > 0.95)")
    print(f"  Final feature count: {len(selected_features)}")

    # -------------------------------------------------------------------------
    # STEP 8: Create final dataset
    # -------------------------------------------------------------------------
    print("\n[STEP 8] Creating final dataset...")

    # Select columns for final dataset
    final_cols = ['hsa_id', 'week_start', 'week_number', 'week_of_year',
                  'month', 'season', 'diarrheal_count_adjusted'] + selected_features

    final_df = merged_df[final_cols].copy()

    # Sort by HSA and week
    final_df = final_df.sort_values(['hsa_id', 'week_start'])

    print(f"  [OK] Final dataset shape: {final_df.shape}")
    print(f"    Records: {len(final_df)}")
    print(f"    Features: {len(selected_features)}")
    print(f"    HSAs: {final_df['hsa_id'].nunique()}")
    print(f"    Weeks: {final_df['week_number'].max()}")

    # -------------------------------------------------------------------------
    # STEP 9: Train/validation/test split
    # -------------------------------------------------------------------------
    print("\n[STEP 9] Splitting into train/validation/test sets...")

    train_df, val_df, test_df = split_temporal_data(
        final_df,
        TRAIN_END_WEEK,
        VAL_END_WEEK
    )

    print(f"  Training set:")
    print(f"    Weeks: 1-{TRAIN_END_WEEK}")
    print(f"    Records: {len(train_df)}")
    print(f"    Date range: {train_df['week_start'].min()} to {train_df['week_start'].max()}")

    print(f"\n  Validation set:")
    print(f"    Weeks: {TRAIN_END_WEEK+1}-{VAL_END_WEEK}")
    print(f"    Records: {len(val_df)}")
    print(f"    Date range: {val_df['week_start'].min()} to {val_df['week_start'].max()}")

    print(f"\n  Test set:")
    print(f"    Weeks: {VAL_END_WEEK+1}+")
    print(f"    Records: {len(test_df)}")
    print(f"    Date range: {test_df['week_start'].min()} to {test_df['week_start'].max()}")

    # -------------------------------------------------------------------------
    # STEP 10: Save datasets
    # -------------------------------------------------------------------------
    print("\n[STEP 10] Saving datasets...")

    # Save full dataset
    full_path = OUTPUT_DIR / "modeling_dataset_full.csv"
    final_df.to_csv(full_path, index=False)
    print(f"  [OK] Saved: {full_path}")

    # Save train set
    train_path = OUTPUT_DIR / "modeling_dataset_train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"  [OK] Saved: {train_path}")

    # Save validation set
    val_path = OUTPUT_DIR / "modeling_dataset_val.csv"
    val_df.to_csv(val_path, index=False)
    print(f"  [OK] Saved: {val_path}")

    # Save test set
    test_path = OUTPUT_DIR / "modeling_dataset_test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"  [OK] Saved: {test_path}")

    # -------------------------------------------------------------------------
    # STEP 11: Create metadata
    # -------------------------------------------------------------------------
    print("\n[STEP 11] Creating metadata...")

    metadata_path = OUTPUT_DIR / "modeling_dataset_metadata.json"
    metadata = create_metadata(final_df, selected_features, metadata_path)

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE")
    print("="*80)
    print(f"\nDataset Summary:")
    print(f"  Total records: {len(final_df)}")
    print(f"  HSAs: {final_df['hsa_id'].nunique()}")
    print(f"  Weeks: {final_df['week_number'].max()}")
    print(f"  Features: {len(selected_features)}")
    print(f"  Date range: {final_df['week_start'].min()} to {final_df['week_start'].max()}")

    print(f"\nTarget Variable (diarrheal_count_adjusted):")
    print(f"  Mean: {final_df['diarrheal_count_adjusted'].mean():.2f}")
    print(f"  Std: {final_df['diarrheal_count_adjusted'].std():.2f}")
    print(f"  Min: {final_df['diarrheal_count_adjusted'].min():.0f}")
    print(f"  Max: {final_df['diarrheal_count_adjusted'].max():.0f}")
    print(f"  Median: {final_df['diarrheal_count_adjusted'].median():.1f}")
    print(f"  Zero counts: {(final_df['diarrheal_count_adjusted'] == 0).sum()} ({(final_df['diarrheal_count_adjusted'] == 0).mean()*100:.1f}%)")

    print(f"\nData Split:")
    print(f"  Train: {len(train_df)} records ({len(train_df)/len(final_df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} records ({len(val_df)/len(final_df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} records ({len(test_df)/len(final_df)*100:.1f}%)")

    print(f"\nOutput Files:")
    print(f"  {full_path}")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {metadata_path}")

    print("\n" + "="*80)
    print("Next Steps:")
    print("  1. Review modeling_dataset_metadata.json for data summary")
    print("  2. Perform exploratory data analysis (EDA)")
    print("  3. Run baseline models")
    print("  4. Train ML models on training set")
    print("  5. Validate on validation set")
    print("  6. Final evaluation on test set (once only)")
    print("="*80)


if __name__ == "__main__":
    main()
