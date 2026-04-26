#!/usr/bin/env python3
"""
Phase 1B: Test Climate Effects in Excluded/Remote Zones

This script tests whether climate-disease relationships are stronger or
different in poorly-connected areas compared to well-connected areas.

Hypothesis: If climate effects are real but masked by allocation uncertainty,
we should see stronger climate signals in:
1. Remote areas with clearer facility assignment
2. Areas with high allocation probability (less assignment uncertainty)
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import json
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
OUTPUT_FILE_PREFIX = ""
TEXT_RESULTS_DIR = None


def out_name(filename: str) -> str:
    return f"{OUTPUT_FILE_PREFIX}_{filename}" if OUTPUT_FILE_PREFIX else filename


def md_path(filename: str) -> Path:
    return TEXT_RESULTS_DIR / out_name(filename) if TEXT_RESULTS_DIR else Path(filename)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
NETWORK = "INF"
HSA_MODE = "footprint"
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"
CLIMATE_DIR = OUT_DIR / "DRIVE_CLIMATE_BY_HSA_DOWNLOAD" / "FINAL_HSA_CLIMATE"
ANALYSIS_DIR = OUT_DIR / "analysis_climate_exclusion"

# Modeling parameters
TRAIN_RATIO = 0.75
VAL_RATIO = 0.125
TEST_RATIO = 0.125


def load_hsa_data() -> pd.DataFrame:
    """Load and merge HSA weekly data with climate."""

    print("Loading HSA weekly data...")

    # Load HSA weekly disease data
    candidates = [
        OUT_DIR / f"{NETWORK}_{HSA_MODE}_weekly_infectious_adjusted.csv",
        OUT_DIR / f"{NETWORK}_{HSA_MODE}_weekly_ncd_adjusted.csv",
        OUT_DIR / f"{NETWORK}_{HSA_MODE}_weekly_hypertension_adjusted.csv",
    ]
    hsa_weekly_path = next((p for p in candidates if p.exists()), None)
    if hsa_weekly_path is None:
        raise FileNotFoundError(f"Could not find HSA weekly data. Tried: {candidates}")

    disease_df = pd.read_csv(hsa_weekly_path)
    print(f"  Loaded {len(disease_df):,} rows from HSA weekly data")

    # Get unique HSAs
    hsas = disease_df['hsa_id'].unique()
    print(f"  Found {len(hsas)} unique HSAs")

    # Load climate data for each HSA
    climate_df = load_climate_for_hsas(hsas)

    if climate_df is not None:
        # Merge disease and climate data
        merged = disease_df.merge(
            climate_df,
            on=['hsa_id', 'week_start'],
            how='left'
        )
        print(f"  Merged dataset: {len(merged):,} rows, {len(merged.columns)} columns")
        return merged

    return disease_df


def load_climate_for_hsas(hsas: np.ndarray) -> Optional[pd.DataFrame]:
    """Load climate data for a list of HSAs."""

    print("\nLoading climate data for HSAs...")

    all_climate = []

    for hsa_name in hsas:
        # Convert HSA name to filename format
        # e.g., "Al-Basheer Hospital" -> "Al-Basheer_Hospital"
        hsa_safe = hsa_name.replace(' ', '_').replace('-', '-')

        # Try to find climate files for this HSA
        # Files are named like: NCD_HSA_Al-Basheer_Hospital_tempdew_wind_lags.csv
        patterns = [
            f"{NETWORK}_HSA_{hsa_safe}_tempdew_wind_lags.csv",
            f"NCD_HSA_{hsa_safe}_tempdew_wind_lags.csv",
            f"HSA_{hsa_safe}_tempdew_wind_lags.csv",
        ]

        climate_file = None
        for pattern in patterns:
            test_path = CLIMATE_DIR / pattern
            if test_path.exists():
                climate_file = test_path
                break

        if climate_file is None:
            # Try fuzzy matching
            for f in CLIMATE_DIR.glob("*tempdew_wind_lags.csv"):
                if hsa_safe.lower().replace('-', '').replace('_', '') in f.stem.lower().replace('-', '').replace('_', ''):
                    climate_file = f
                    break

        if climate_file:
            try:
                clim = pd.read_csv(climate_file)

                # Select key climate variables
                keep_cols = ['week_start']
                for col in clim.columns:
                    if any(x in col for x in ['T_mean', 'T_max', 'T_min', 'DTR', 'heat_index',
                                               'hours_above', 'wind_speed', 'Td_']):
                        if 'week' in col or 'd-1' in col or 'd-2' in col or 'd-3' in col:
                            keep_cols.append(col)

                clim = clim[keep_cols].copy()
                clim['hsa_id'] = hsa_name
                all_climate.append(clim)
            except Exception as e:
                pass  # Skip problematic files

    if all_climate:
        climate_df = pd.concat(all_climate, ignore_index=True)
        print(f"  Loaded climate for {len(all_climate)} HSAs")
        print(f"  Climate columns: {len(climate_df.columns) - 2}")  # minus week_start and hsa_id
        return climate_df

    print("  No climate data found")
    return None


def load_allocation_characteristics() -> pd.DataFrame:
    """Load HSA allocation characteristics (mean probability, distance)."""

    print("\nLoading allocation characteristics...")

    alloc_path = OUT_DIR / f"pixel_allocations_{NETWORK}_{HSA_MODE}.csv"
    df = pd.read_csv(alloc_path)

    # Aggregate by facility
    hsa_chars = df.groupby('facility_id').agg({
        'population': 'sum',
        'probability': 'mean',
        'lon': 'mean',
        'lat': 'mean'
    }).reset_index()

    hsa_chars.columns = ['hsa_id', 'total_population', 'mean_probability',
                          'centroid_lon', 'centroid_lat']

    print(f"  Loaded characteristics for {len(hsa_chars)} HSAs")
    print(f"  Mean probability range: {hsa_chars['mean_probability'].min():.3f} - {hsa_chars['mean_probability'].max():.3f}")

    return hsa_chars


def classify_hsas_by_connectivity(hsa_chars: pd.DataFrame) -> pd.DataFrame:
    """Classify HSAs into connectivity groups."""

    print("\nClassifying HSAs by connectivity...")

    # Use probability-based classification
    prob_median = hsa_chars['mean_probability'].median()

    hsa_chars = hsa_chars.copy()
    hsa_chars['connectivity_group'] = np.where(
        hsa_chars['mean_probability'] >= prob_median,
        'high_connectivity',
        'low_connectivity'
    )

    # Also create tertile groups for more nuanced analysis
    hsa_chars['connectivity_tertile'] = pd.qcut(
        hsa_chars['mean_probability'],
        q=3,
        labels=['low', 'medium', 'high']
    )

    print(f"  High connectivity HSAs: {(hsa_chars['connectivity_group'] == 'high_connectivity').sum()}")
    print(f"  Low connectivity HSAs: {(hsa_chars['connectivity_group'] == 'low_connectivity').sum()}")

    return hsa_chars


def merge_connectivity_with_disease(disease_df: pd.DataFrame,
                                    hsa_chars: pd.DataFrame) -> pd.DataFrame:
    """Merge connectivity classification with disease data."""

    print("\nMerging connectivity with disease data...")

    # Find the HSA column name
    hsa_col = None
    for col in ['hsa_id', 'HSA', 'healthfacility', 'HealthFacility', 'facility_id']:
        if col in disease_df.columns:
            hsa_col = col
            break

    if hsa_col is None:
        print(f"  Available columns: {disease_df.columns.tolist()}")
        raise ValueError("Could not find HSA column in disease data")

    print(f"  Using HSA column: {hsa_col}")

    # Merge
    merged = disease_df.merge(
        hsa_chars[['hsa_id', 'mean_probability', 'connectivity_group', 'connectivity_tertile', 'total_population']],
        left_on=hsa_col,
        right_on='hsa_id',
        how='left'
    )

    # Check merge success
    matched = merged['connectivity_group'].notna().sum()
    print(f"  Matched {matched:,} / {len(merged):,} rows ({100*matched/len(merged):.1f}%)")

    return merged


def identify_climate_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify different types of columns in the dataset."""

    cols = df.columns.tolist()

    # Climate columns - look for patterns like T_mean, DTR, heat_index, etc.
    climate_patterns = ['T_mean', 'T_max', 'T_min', 'DTR', 'heat_index',
                       'hours_above', 'wind_speed', 'Td_', 'temp', 'precip',
                       'rain', 'humid', 'evap', 'soil']
    climate_cols = [c for c in cols if any(p in c for p in climate_patterns)]

    ar_cols = [c for c in cols if c.startswith('cases_lag') or c.startswith('ar_') or
               ('lag' in c.lower() and 'case' in c.lower())]

    outcome_cols = [c for c in cols if any(x in c.lower() for x in
                   ['cases', 'count', 'infectious', 'disease']) and 'lag' not in c.lower()]

    return {
        'climate': climate_cols,
        'ar': ar_cols,
        'outcome': outcome_cols,
        'all': cols
    }


def create_temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create temporal train/val/test split."""

    # Find date column
    date_col = None
    for col in ['date', 'week_start', 'week', 'time']:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        df = df.sort_values(date_col)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def fit_model_by_group(data: pd.DataFrame,
                       feature_cols: List[str],
                       outcome_col: str,
                       group_col: str = 'connectivity_group') -> Dict:
    """Fit separate models for each connectivity group and compare."""

    print(f"\nFitting models by {group_col}...")

    results = {}

    # Drop rows with missing values in key columns
    valid_cols = [outcome_col] + feature_cols + [group_col]
    valid_cols = [c for c in valid_cols if c in data.columns]
    df = data[valid_cols].dropna()

    if len(df) == 0:
        print("  No valid data after dropping NAs")
        return results

    groups = df[group_col].unique()

    for group in groups:
        print(f"\n  Group: {group}")

        group_data = df[df[group_col] == group]

        if len(group_data) < 100:
            print(f"    Insufficient data ({len(group_data)} rows)")
            continue

        # Split
        train, val, test = create_temporal_split(group_data)

        # Prepare features
        X_train = train[feature_cols].fillna(0)
        y_train = train[outcome_col]
        X_test = test[feature_cols].fillna(0)
        y_test = test[outcome_col]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit model
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # Get feature importance
        coef_importance = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': model.coef_
        })
        coef_importance['abs_coef'] = np.abs(coef_importance['coefficient'])
        coef_importance = coef_importance.sort_values('abs_coef', ascending=False)

        print(f"    Train R²: {train_r2:.4f}")
        print(f"    Test R²:  {test_r2:.4f}")
        print(f"    N train:  {len(train):,}")
        print(f"    N test:   {len(test):,}")

        results[group] = {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'n_train': len(train),
            'n_test': len(test),
            'top_features': coef_importance.head(10).to_dict('records'),
            'all_coefficients': coef_importance.to_dict('records')
        }

    return results


def compare_climate_contribution_by_connectivity(data: pd.DataFrame,
                                                 col_info: Dict) -> Dict:
    """Compare climate contribution between connectivity groups."""

    print("\n" + "=" * 60)
    print("COMPARING CLIMATE CONTRIBUTION BY CONNECTIVITY")
    print("=" * 60)

    # Find outcome column
    outcome_col = None
    for candidate in ['infectious_count_adjusted', 'hypertension_count_adjusted', 'ncd_count_adjusted',
                      'infectious_cases', 'hypertension_cases', 'ncd_cases', 'cases',
                      'case_count', 'weekly_cases', 'count_adjusted']:
        if candidate in data.columns:
            outcome_col = candidate
            break

    if outcome_col is None:
        # Try to find any numeric column that could be outcome
        for col in data.columns:
            if ('case' in col.lower() or 'count' in col.lower()) and 'lag' not in col.lower():
                outcome_col = col
                break

    if outcome_col is None:
        print("Could not identify outcome column")
        print(f"  Available columns: {[c for c in data.columns if 'count' in c.lower() or 'case' in c.lower()]}")
        return {}

    print(f"\nUsing outcome column: {outcome_col}")

    # Identify AR and climate features
    ar_cols = [c for c in data.columns if 'lag' in c.lower() and 'case' in c.lower()]
    climate_cols = col_info['climate']

    print(f"  Found {len(ar_cols)} AR columns")
    print(f"  Found {len(climate_cols)} climate columns")

    if len(ar_cols) == 0:
        # Create AR features
        print("  Creating AR features...")
        for lag in [1, 2, 3, 4]:
            col_name = f'cases_lag{lag}'
            data[col_name] = data.groupby('hsa_id')[outcome_col].shift(lag)
            ar_cols.append(col_name)

    results = {
        'by_connectivity': {},
        'overall': {}
    }

    # For each connectivity group, compare:
    # 1. AR-only model
    # 2. Climate-only model
    # 3. AR + Climate model

    for group in ['high_connectivity', 'low_connectivity']:
        print(f"\n{'=' * 40}")
        print(f"Group: {group}")
        print('=' * 40)

        group_data = data[data['connectivity_group'] == group].dropna(subset=[outcome_col])

        if len(group_data) < 200:
            print(f"  Insufficient data: {len(group_data)} rows")
            continue

        train, val, test = create_temporal_split(group_data)

        y_train = train[outcome_col]
        y_test = test[outcome_col]

        scaler = StandardScaler()

        # Model 1: AR only
        if ar_cols:
            valid_ar_cols = [c for c in ar_cols if c in train.columns]
            if valid_ar_cols:
                X_train_ar = train[valid_ar_cols].fillna(0)
                X_test_ar = test[valid_ar_cols].fillna(0)

                X_train_ar_scaled = scaler.fit_transform(X_train_ar)
                X_test_ar_scaled = scaler.transform(X_test_ar)

                model_ar = Ridge(alpha=1.0)
                model_ar.fit(X_train_ar_scaled, y_train)
                ar_r2_train = r2_score(y_train, model_ar.predict(X_train_ar_scaled))
                ar_r2_test = r2_score(y_test, model_ar.predict(X_test_ar_scaled))
            else:
                ar_r2_train = ar_r2_test = None
        else:
            ar_r2_train = ar_r2_test = None

        # Model 2: Climate only
        if climate_cols:
            valid_climate_cols = [c for c in climate_cols if c in train.columns]
            if valid_climate_cols:
                X_train_clim = train[valid_climate_cols].fillna(0)
                X_test_clim = test[valid_climate_cols].fillna(0)

                scaler_clim = StandardScaler()
                X_train_clim_scaled = scaler_clim.fit_transform(X_train_clim)
                X_test_clim_scaled = scaler_clim.transform(X_test_clim)

                model_clim = Ridge(alpha=1.0)
                model_clim.fit(X_train_clim_scaled, y_train)
                clim_r2_train = r2_score(y_train, model_clim.predict(X_train_clim_scaled))
                clim_r2_test = r2_score(y_test, model_clim.predict(X_test_clim_scaled))

                # Get climate coefficients
                clim_coefs = pd.DataFrame({
                    'feature': valid_climate_cols,
                    'coefficient': model_clim.coef_
                })
                clim_coefs['abs_coef'] = np.abs(clim_coefs['coefficient'])
                clim_coefs = clim_coefs.sort_values('abs_coef', ascending=False)
            else:
                clim_r2_train = clim_r2_test = None
                clim_coefs = pd.DataFrame()
        else:
            clim_r2_train = clim_r2_test = None
            clim_coefs = pd.DataFrame()

        # Model 3: AR + Climate
        all_cols = (valid_ar_cols if ar_cols else []) + (valid_climate_cols if climate_cols else [])
        if all_cols:
            X_train_all = train[all_cols].fillna(0)
            X_test_all = test[all_cols].fillna(0)

            scaler_all = StandardScaler()
            X_train_all_scaled = scaler_all.fit_transform(X_train_all)
            X_test_all_scaled = scaler_all.transform(X_test_all)

            model_all = Ridge(alpha=1.0)
            model_all.fit(X_train_all_scaled, y_train)
            all_r2_train = r2_score(y_train, model_all.predict(X_train_all_scaled))
            all_r2_test = r2_score(y_test, model_all.predict(X_test_all_scaled))
        else:
            all_r2_train = all_r2_test = None

        # Calculate climate contribution
        if ar_r2_test is not None and all_r2_test is not None:
            climate_contribution = all_r2_test - ar_r2_test
        else:
            climate_contribution = None

        print(f"\n  Model Results:")
        ar_train_str = f"{ar_r2_train:.4f}" if ar_r2_train is not None else "N/A"
        ar_test_str = f"{ar_r2_test:.4f}" if ar_r2_test is not None else "N/A"
        clim_train_str = f"{clim_r2_train:.4f}" if clim_r2_train is not None else "N/A"
        clim_test_str = f"{clim_r2_test:.4f}" if clim_r2_test is not None else "N/A"
        all_train_str = f"{all_r2_train:.4f}" if all_r2_train is not None else "N/A"
        all_test_str = f"{all_r2_test:.4f}" if all_r2_test is not None else "N/A"
        contrib_str = f"{climate_contribution:.4f}" if climate_contribution is not None else "N/A"
        print(f"    AR only     - Train: {ar_train_str:>8}, Test: {ar_test_str:>8}")
        print(f"    Climate only- Train: {clim_train_str:>8}, Test: {clim_test_str:>8}")
        print(f"    AR + Climate- Train: {all_train_str:>8}, Test: {all_test_str:>8}")
        print(f"    Climate contribution: {contrib_str}")

        if len(clim_coefs) > 0:
            print(f"\n  Top climate features (by coefficient magnitude):")
            for _, row in clim_coefs.head(5).iterrows():
                print(f"    {row['feature']:40s} {row['coefficient']:>8.4f}")

        results['by_connectivity'][group] = {
            'n_train': len(train),
            'n_test': len(test),
            'ar_only': {
                'train_r2': float(ar_r2_train) if ar_r2_train is not None else None,
                'test_r2': float(ar_r2_test) if ar_r2_test is not None else None
            },
            'climate_only': {
                'train_r2': float(clim_r2_train) if clim_r2_train is not None else None,
                'test_r2': float(clim_r2_test) if clim_r2_test is not None else None
            },
            'ar_plus_climate': {
                'train_r2': float(all_r2_train) if all_r2_train is not None else None,
                'test_r2': float(all_r2_test) if all_r2_test is not None else None
            },
            'climate_contribution': float(climate_contribution) if climate_contribution is not None else None,
            'top_climate_features': clim_coefs.head(10).to_dict('records') if len(clim_coefs) > 0 else []
        }

    return results


def test_climate_interaction_with_probability(data: pd.DataFrame,
                                              col_info: Dict) -> Dict:
    """Test if climate effects vary with allocation probability."""

    print("\n" + "=" * 60)
    print("TESTING CLIMATE × PROBABILITY INTERACTION")
    print("=" * 60)

    # Find outcome
    outcome_col = None
    for candidate in ['infectious_count_adjusted', 'hypertension_count_adjusted', 'ncd_count_adjusted',
                      'infectious_cases', 'hypertension_cases', 'ncd_cases', 'cases',
                      'case_count', 'weekly_cases', 'count_adjusted']:
        if candidate in data.columns:
            outcome_col = candidate
            break

    if outcome_col is None:
        # Try to find any count column
        for col in data.columns:
            if ('case' in col.lower() or 'count' in col.lower()) and 'lag' not in col.lower():
                outcome_col = col
                break

    if outcome_col is None:
        print("  Could not identify outcome column")
        return {}

    climate_cols = col_info['climate'][:5]  # Top 5 climate features

    if not climate_cols or 'mean_probability' not in data.columns:
        print("  Missing required columns")
        return {}

    # Create interaction terms
    data = data.copy()
    interaction_cols = []

    for clim_col in climate_cols:
        if clim_col in data.columns:
            int_col = f'{clim_col}_x_prob'
            data[int_col] = data[clim_col] * data['mean_probability']
            interaction_cols.append(int_col)

    print(f"  Created {len(interaction_cols)} interaction terms")

    # Fit model with interactions
    train, val, test = create_temporal_split(data)

    valid_climate = [c for c in climate_cols if c in data.columns]
    all_features = valid_climate + interaction_cols + ['mean_probability']

    X_train = train[all_features].fillna(0)
    y_train = train[outcome_col]
    X_test = test[all_features].fillna(0)
    y_test = test[outcome_col]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    train_r2 = r2_score(y_train, model.predict(X_train_scaled))
    test_r2 = r2_score(y_test, model.predict(X_test_scaled))

    coefs = pd.DataFrame({
        'feature': all_features,
        'coefficient': model.coef_
    })
    coefs['abs_coef'] = np.abs(coefs['coefficient'])
    coefs = coefs.sort_values('abs_coef', ascending=False)

    print(f"\n  Model with interactions:")
    print(f"    Train R²: {train_r2:.4f}")
    print(f"    Test R²:  {test_r2:.4f}")
    print(f"\n  Top coefficients:")
    for _, row in coefs.head(10).iterrows():
        print(f"    {row['feature']:40s} {row['coefficient']:>8.4f}")

    # Check if interaction terms are significant
    interaction_coefs = coefs[coefs['feature'].str.contains('_x_prob')]
    main_coefs = coefs[~coefs['feature'].str.contains('_x_prob') & (coefs['feature'] != 'mean_probability')]

    print(f"\n  Interaction terms (climate × probability):")
    for _, row in interaction_coefs.iterrows():
        sign = "+" if row['coefficient'] > 0 else "-"
        print(f"    {row['feature']:40s} {sign} {row['abs_coef']:.4f}")

    return {
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'coefficients': coefs.to_dict('records'),
        'interaction_terms': interaction_coefs.to_dict('records')
    }


def generate_report(results: Dict, output_path: Path):
    """Generate markdown report."""

    print("\nGenerating report...")

    report = []
    report.append("# Phase 1B: Climate Effects by Connectivity\n")

    report.append("## Overview\n")
    report.append("This analysis tests whether climate-disease relationships differ")
    report.append("between well-connected and poorly-connected HSAs.\n")
    report.append("**Hypothesis**: If climate effects are real but masked by allocation")
    report.append("uncertainty, we should see stronger climate signals in areas with")
    report.append("higher allocation probability (more certain case assignment).\n")

    if 'by_connectivity' in results:
        report.append("## Results by Connectivity Group\n")

        report.append("| Group | Model | Train R² | Test R² |")
        report.append("|-------|-------|----------|---------|")

        for group, data in results['by_connectivity'].items():
            for model_name, model_key in [('AR Only', 'ar_only'),
                                          ('Climate Only', 'climate_only'),
                                          ('AR + Climate', 'ar_plus_climate')]:
                if model_key in data and data[model_key]['test_r2'] is not None:
                    train_r2 = data[model_key]['train_r2']
                    test_r2 = data[model_key]['test_r2']
                    report.append(f"| {group} | {model_name} | {train_r2:.4f} | {test_r2:.4f} |")

        report.append("")

        # Climate contribution comparison
        report.append("## Climate Contribution by Group\n")
        report.append("| Group | ΔR² (raw) | ΔR² (percentage points) | Relative change vs AR-only |")
        report.append("|-------|-----------|--------------------------|----------------------------|")

        for group, data in results['by_connectivity'].items():
            contrib = data.get('climate_contribution')
            if contrib is not None:
                ar_test_r2 = data.get('ar_only', {}).get('test_r2')
                contrib_pp = contrib * 100
                if ar_test_r2 is not None and ar_test_r2 != 0:
                    relative_pct = (contrib / ar_test_r2) * 100
                    relative_str = f"{relative_pct:+.1f}%"
                else:
                    relative_str = "N/A"
                report.append(f"| {group} | {contrib:+.4f} | {contrib_pp:+.2f} pp | {relative_str} |")

        report.append("")

    report.append("## Key Findings\n")

    # Analyze results
    if 'by_connectivity' in results:
        high = results['by_connectivity'].get('high_connectivity', {})
        low = results['by_connectivity'].get('low_connectivity', {})

        high_contrib = high.get('climate_contribution')
        low_contrib = low.get('climate_contribution')

        if high_contrib is not None and low_contrib is not None:
            if high_contrib > low_contrib:
                report.append(f"1. Climate contributes MORE in high-connectivity areas ({high_contrib:+.4f}, {high_contrib*100:+.2f} pp vs {low_contrib:+.4f}, {low_contrib*100:+.2f} pp)")
                report.append("   - This supports the hypothesis that allocation uncertainty masks climate signals")
            else:
                report.append(f"1. Climate contributes MORE in low-connectivity areas ({low_contrib:+.4f}, {low_contrib*100:+.2f} pp vs {high_contrib:+.4f}, {high_contrib*100:+.2f} pp)")
                report.append("   - This does NOT support the allocation uncertainty hypothesis")

            report.append("")

            if high_contrib <= 0 and low_contrib <= 0:
                report.append("2. Climate adds NO predictive value in either group")
                report.append("   - The weak climate signal is not due to allocation uncertainty")
            elif high_contrib > 0 or low_contrib > 0:
                report.append("2. Climate shows some positive contribution in at least one group")
                report.append("   - Further investigation may be warranted")

    report.append("")
    report.append("## Implications\n")
    report.append("If climate effects are consistently weak across connectivity groups,")
    report.append("the AR dominance is likely a real phenomenon (strong temporal autocorrelation")
    report.append("in disease incidence) rather than an artifact of allocation uncertainty.\n")

    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"  Saved report to {output_path}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Phase 1B: Climate Effects in Excluded Zones")
    parser.add_argument("--network", default="INF",
                        help='Network label, e.g. INF, NCD, SYNINF, SYNNCD, SYNMODINF, SYNMODNCD')
    parser.add_argument("--hsa-mode", default="footprint")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--climate-dir", default=None)
    parser.add_argument("--output-dir", default="out/analysis_climate_exclusion")
    parser.add_argument("--text-output-dir", default="out/textresults")
    args = parser.parse_args()

    global NETWORK, HSA_MODE, DATA_DIR, OUT_DIR, CLIMATE_DIR, ANALYSIS_DIR, OUTPUT_FILE_PREFIX, TEXT_RESULTS_DIR
    NETWORK = args.network
    HSA_MODE = args.hsa_mode
    DATA_DIR = Path(args.data_dir)
    OUT_DIR = Path(args.out_dir)
    CLIMATE_DIR = Path(args.climate_dir) if args.climate_dir else (OUT_DIR / "DRIVE_CLIMATE_BY_HSA_DOWNLOAD" / "FINAL_HSA_CLIMATE")
    ANALYSIS_DIR = Path(args.output_dir)
    OUTPUT_FILE_PREFIX = f"{NETWORK}_{HSA_MODE}"
    TEXT_RESULTS_DIR = Path(args.text_output_dir)
    TEXT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 1B: CLIMATE EFFECTS IN EXCLUDED ZONES")
    print(f"Network: {NETWORK}")
    print(f"HSA mode: {HSA_MODE}")
    print("=" * 70)

    # Load data
    disease_data = load_hsa_data()

    # Load allocation characteristics
    hsa_chars = load_allocation_characteristics()

    # Classify HSAs
    hsa_chars = classify_hsas_by_connectivity(hsa_chars)

    # Merge with disease data
    merged_data = merge_connectivity_with_disease(disease_data, hsa_chars)

    # Identify column types
    col_info = identify_climate_columns(merged_data)

    print(f"\nColumn types identified:")
    print(f"  Climate columns: {len(col_info['climate'])}")
    print(f"  AR columns: {len(col_info['ar'])}")
    print(f"  Outcome columns: {len(col_info['outcome'])}")

    # Compare climate contribution by connectivity
    results = compare_climate_contribution_by_connectivity(merged_data, col_info)

    # Test interaction effects
    interaction_results = test_climate_interaction_with_probability(merged_data, col_info)
    results['interaction_analysis'] = interaction_results

    # Save results
    results_path = ANALYSIS_DIR / out_name("climate_by_connectivity_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Generate report
    generate_report(results, md_path("climate_connectivity_report.md"))

    print("\n" + "=" * 70)
    print("PHASE 1B COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
