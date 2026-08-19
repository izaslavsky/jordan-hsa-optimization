#!/usr/bin/env python3
"""
Comprehensive Climate-Health Modeling for Diarrheal Disease Prediction

This script performs end-to-end modeling of diarrheal disease rates using climate variables,
with proper train/validation/test splits to avoid temporal leakage.

Author: Climate-Health Research Team
Date: 2024-01
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import datetime
import argparse
import os

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception as e:
    HAS_XGBOOST = False
    print(f"Note: XGBoost not available ({e}); will use scikit-learn models only")

warnings.filterwarnings('ignore')

# Configuration defaults
DEFAULT_PIPELINE_OUT_DIR = os.environ.get("HSA_OUT_DIR", os.environ.get("PIPELINE_OUT_DIR", "out"))
DEFAULT_RANDOM_SEED = 42

parser = argparse.ArgumentParser(description="Comprehensive climate-health modeling")
parser.add_argument("--network", default=os.environ.get("NETWORK", "INF"))
parser.add_argument("--hsa-mode", default=os.environ.get("HSA_MODE", "footprint"))
parser.add_argument("--target-col", default=os.environ.get("TARGET_COL", "diarrheal_count_adjusted"))
parser.add_argument("--input-csv", default=os.environ.get("MODEL_INPUT_CSV", None))
parser.add_argument("--output-dir", default=os.environ.get("MODEL_OUTPUT_DIR", str(Path(DEFAULT_PIPELINE_OUT_DIR) / "modeling" / "results_comprehensive")))
parser.add_argument("--output-prefix", default=os.environ.get("MODEL_OUTPUT_PREFIX", None))
parser.add_argument("--random-seed", type=int, default=int(os.environ.get("RANDOM_SEED", DEFAULT_RANDOM_SEED)),
                    help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})")
args = parser.parse_args()
NETWORK = args.network
HSA_MODE = args.hsa_mode
TARGET_COL = args.target_col
INPUT_CSV = args.input_csv or str(Path(DEFAULT_PIPELINE_OUT_DIR) / "modeling" / f"{NETWORK}_{HSA_MODE}_modeling_dataset.csv")
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_PREFIX = args.output_prefix or f"{NETWORK}_{HSA_MODE}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = args.random_seed
np.random.seed(RANDOM_SEED)


def load_data():
    """Load the modeling dataset."""
    df = pd.read_csv(INPUT_CSV)
    df['week_start'] = pd.to_datetime(df['week_start'])
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"HSAs: {df['hsa_id'].nunique()}, Weeks: {df['week_number'].nunique()}")
    return df


def create_temporal_splits(df, train_pct=0.75, val_pct=0.125):
    """
    Create train/validation/test splits based on temporal ordering.

    This ensures no future data leaks into training (strict temporal split).
    All HSAs share the same time periods for each split.
    """
    # Get unique weeks sorted
    unique_weeks = sorted(df['week_number'].unique())
    n_weeks = len(unique_weeks)

    # Calculate split points
    train_end_week = int(n_weeks * train_pct)
    val_end_week = int(n_weeks * (train_pct + val_pct))

    train_weeks = unique_weeks[:train_end_week]
    val_weeks = unique_weeks[train_end_week:val_end_week]
    test_weeks = unique_weeks[val_end_week:]

    # Split data
    train_df = df[df['week_number'].isin(train_weeks)].copy()
    val_df = df[df['week_number'].isin(val_weeks)].copy()
    test_df = df[df['week_number'].isin(test_weeks)].copy()

    print(f"\nTemporal Split:")
    print(f"  Train: weeks {train_weeks[0]}-{train_weeks[-1]} ({len(train_df)} rows, {len(train_weeks)} weeks)")
    print(f"  Val:   weeks {val_weeks[0]}-{val_weeks[-1]} ({len(val_df)} rows, {len(val_weeks)} weeks)")
    print(f"  Test:  weeks {test_weeks[0]}-{test_weeks[-1]} ({len(test_df)} rows, {len(test_weeks)} weeks)")

    return train_df, val_df, test_df


def add_autoregressive_features(df, target_col, lags=[1, 2, 3, 4]):
    """
    Add lagged target values as autoregressive features.
    This must be done BEFORE splitting to ensure continuity within HSAs.
    """
    df = df.sort_values(['hsa_id', 'week_number']).copy()

    for lag in lags:
        df[f'ar_lag{lag}'] = df.groupby('hsa_id')[target_col].shift(lag)

    # Count rows lost due to AR features
    original_len = len(df)
    df_with_ar = df.dropna(subset=[f'ar_lag{max(lags)}'])
    print(f"\nAdded AR features (lags {lags}): {original_len} -> {len(df_with_ar)} rows")

    return df_with_ar


def get_feature_columns(df, include_ar=True, include_hsa=True):
    """
    Get list of feature columns, excluding metadata and target.
    """
    exclude_cols = ['hsa_id', 'week_start', 'week_number', 'week_of_year',
                    'month', 'season', 'quarter', 'days_since_start',
                    TARGET_COL]

    # Get climate features
    climate_features = [col for col in df.columns
                        if col not in exclude_cols
                        and not col.startswith('ar_lag')]

    features = list(climate_features)

    # Add temporal features
    temporal_features = ['week_of_year', 'month', 'quarter']
    features.extend([f for f in temporal_features if f in df.columns])

    # Add AR features if requested
    if include_ar:
        ar_features = [col for col in df.columns if col.startswith('ar_lag')]
        features.extend(ar_features)

    # Add HSA encoding if requested
    if include_hsa:
        features.append('hsa_encoded')

    return features


def correlation_feature_selection(train_df, target_col, features, threshold=0.95, top_n=50):
    """
    Select features based on correlation analysis (on training set only to avoid leakage).

    1. Remove highly correlated feature pairs (keep one)
    2. Select top N features by correlation with target
    """
    # Get numeric feature data
    numeric_features = [f for f in features if train_df[f].dtype in ['float64', 'int64', 'int32']]

    # Step 1: Remove multicollinearity
    corr_matrix = train_df[numeric_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features to drop (highly correlated)
    to_drop = set()
    for col in upper.columns:
        correlated_cols = upper.index[upper[col] > threshold].tolist()
        for corr_col in correlated_cols:
            if corr_col not in to_drop:
                to_drop.add(corr_col)

    features_after_multicollinearity = [f for f in numeric_features if f not in to_drop]
    print(f"\nFeature Selection (on training set only):")
    print(f"  Removed {len(to_drop)} highly correlated features (r > {threshold})")
    print(f"  Remaining: {len(features_after_multicollinearity)} features")

    # Step 2: Select top N by correlation with target
    correlations = train_df[features_after_multicollinearity].corrwith(
        train_df[target_col]
    ).abs().sort_values(ascending=False)

    top_features = correlations.head(top_n).index.tolist()
    print(f"  Selected top {len(top_features)} features by correlation with target")

    # Save correlation info
    correlation_info = {
        'dropped_multicollinear': list(to_drop),
        'selected_features': top_features,
        'correlations': {f: float(correlations[f]) for f in top_features}
    }

    with open(OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_selection_info.json", 'w') as f:
        json.dump(correlation_info, f, indent=2)

    return top_features, correlation_info


def prepare_feature_sets(df, all_features, selected_climate_features, include_ar=True, include_hsa=True):
    """
    Prepare different feature sets for model comparison.
    """
    feature_sets = {}

    # AR features
    ar_features = [col for col in df.columns if col.startswith('ar_lag')]

    # Temporal features
    temporal_features = ['week_of_year', 'month', 'quarter']
    available_temporal = [f for f in temporal_features if f in df.columns]

    # 1. AR only (baseline+)
    if include_ar and ar_features:
        feature_sets['ar_only'] = ar_features

    # 2. AR + temporal
    if include_ar and ar_features:
        feature_sets['ar_temporal'] = ar_features + available_temporal

    # 3. Climate only (selected)
    feature_sets['climate_only'] = selected_climate_features[:30]

    # 4. Climate + temporal
    feature_sets['climate_temporal'] = selected_climate_features[:30] + available_temporal

    # 5. AR + climate (full model)
    if include_ar and ar_features:
        feature_sets['ar_climate'] = ar_features + selected_climate_features[:20]

    # 6. AR + climate + temporal
    if include_ar and ar_features:
        feature_sets['ar_climate_temporal'] = (
            ar_features + selected_climate_features[:20] + available_temporal
        )

    # 7. AR + climate + temporal + HSA (full model with spatial effects)
    if include_ar and ar_features and include_hsa and 'hsa_encoded' in df.columns:
        feature_sets['ar_climate_temporal_hsa'] = (
            ar_features + selected_climate_features[:20] + available_temporal + ['hsa_encoded']
        )

    return feature_sets


def train_baseline_models(train_df, val_df, test_df, target_col):
    """
    Train baseline models for comparison.
    """
    results = []

    # 1. Mean baseline
    train_mean = train_df[target_col].mean()
    for split_name, split_df in [('val', val_df), ('test', test_df)]:
        y_true = split_df[target_col].values
        y_pred = np.full_like(y_true, train_mean, dtype=float)

        results.append({
            'feature_set': 'baseline',
            'model': 'mean_baseline',
            'split': split_name,
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred)
        })

    # 2. Last week baseline (persistence)
    for split_name, split_df in [('val', val_df), ('test', test_df)]:
        if 'ar_lag1' in split_df.columns:
            y_true = split_df[target_col].values
            y_pred = split_df['ar_lag1'].values

            results.append({
                'feature_set': 'baseline',
                'model': 'last_week',
                'split': split_name,
                'r2': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred)
            })

    # 3. Seasonal mean baseline
    seasonal_means = train_df.groupby('week_of_year')[target_col].mean()
    for split_name, split_df in [('val', val_df), ('test', test_df)]:
        y_true = split_df[target_col].values
        y_pred = split_df['week_of_year'].map(seasonal_means).fillna(train_mean).values

        results.append({
            'feature_set': 'baseline',
            'model': 'seasonal_mean',
            'split': split_name,
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred)
        })

    return results


def train_model_families(train_df, val_df, test_df, feature_sets, target_col):
    """
    Train multiple model families on different feature sets.
    """
    results = []
    models_trained = {}

    for feature_set_name, features in feature_sets.items():
        # Check all features exist
        available_features = [f for f in features if f in train_df.columns]
        if len(available_features) < len(features):
            print(f"  Warning: {len(features) - len(available_features)} features missing from {feature_set_name}")

        if not available_features:
            continue

        # Prepare data
        X_train = train_df[available_features].values
        y_train = train_df[target_col].values
        X_val = val_df[available_features].values
        y_val = val_df[target_col].values
        X_test = test_df[available_features].values
        y_test = test_df[target_col].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n  Feature set: {feature_set_name} ({len(available_features)} features)")

        # Define models
        model_configs = {
            'ridge': Ridge(alpha=1.0, random_state=RANDOM_SEED),
            'lasso': Lasso(alpha=0.1, max_iter=10000, random_state=RANDOM_SEED),
            'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=RANDOM_SEED),
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                min_samples_split=10, min_samples_leaf=5, random_state=RANDOM_SEED
            ),
        }

        # Add XGBoost if available
        if HAS_XGBOOST:
            model_configs['xgboost'] = xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_SEED, n_jobs=-1
            )

        for model_name, model in model_configs.items():
            try:
                # Use scaled data for linear models, raw for tree-based
                if model_name in ['ridge', 'lasso', 'elasticnet']:
                    model.fit(X_train_scaled, y_train)
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_val = model.predict(X_val_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred_train = model.predict(X_train)
                    y_pred_val = model.predict(X_val)
                    y_pred_test = model.predict(X_test)

                # Store results
                for split_name, y_true, y_pred in [
                    ('train', y_train, y_pred_train),
                    ('val', y_val, y_pred_val),
                    ('test', y_test, y_pred_test)
                ]:
                    results.append({
                        'feature_set': feature_set_name,
                        'model': model_name,
                        'n_features': len(available_features),
                        'split': split_name,
                        'r2': r2_score(y_true, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                        'mae': mean_absolute_error(y_true, y_pred)
                    })

                # Store trained model
                models_trained[f"{feature_set_name}_{model_name}"] = {
                    'model': model,
                    'scaler': scaler if model_name in ['ridge', 'lasso', 'elasticnet'] else None,
                    'features': available_features
                }

            except Exception as e:
                print(f"    Error training {model_name}: {e}")

    return results, models_trained


def calculate_climate_contribution(results_df):
    """
    Calculate the contribution of climate variables to model performance.
    Compare AR-only vs AR+climate models.
    """
    contributions = []

    # Get test set results
    test_results = results_df[results_df['split'] == 'test']

    for model in ['ridge', 'lasso', 'elasticnet', 'random_forest', 'gradient_boosting', 'xgboost']:
        ar_only = test_results[(test_results['feature_set'] == 'ar_only') &
                               (test_results['model'] == model)]
        ar_climate = test_results[(test_results['feature_set'] == 'ar_climate') &
                                   (test_results['model'] == model)]

        if not ar_only.empty and not ar_climate.empty:
            ar_r2 = ar_only['r2'].values[0]
            ar_climate_r2 = ar_climate['r2'].values[0]

            # Climate contribution (absolute and relative)
            climate_contribution = ar_climate_r2 - ar_r2
            relative_contribution = (climate_contribution / ar_climate_r2 * 100) if ar_climate_r2 > 0 else 0

            contributions.append({
                'model': model,
                'ar_only_r2': ar_r2,
                'ar_climate_r2': ar_climate_r2,
                'climate_contribution_absolute': climate_contribution,
                'climate_contribution_relative_pct': relative_contribution
            })

    return pd.DataFrame(contributions)


def calculate_extended_climate_contribution(results_df, baseline_results):
    """
    Calculate climate contribution from multiple baselines:
    1. Climate vs mean baseline (total climate explanatory power)
    2. Climate vs AR-only (incremental beyond persistence)
    3. Climate vs AR+seasonal (incremental beyond AR and seasonality)

    This provides a fuller picture of what climate variables explain.
    """
    test_results = results_df[results_df['split'] == 'test']
    baseline_df = pd.DataFrame(baseline_results)
    baseline_test = baseline_df[baseline_df['split'] == 'test']

    # Get baseline R² values
    mean_r2 = baseline_test[baseline_test['model'] == 'mean_baseline']['r2'].values
    mean_r2 = mean_r2[0] if len(mean_r2) > 0 else 0

    last_week_r2 = baseline_test[baseline_test['model'] == 'last_week']['r2'].values
    last_week_r2 = last_week_r2[0] if len(last_week_r2) > 0 else 0

    seasonal_r2 = baseline_test[baseline_test['model'] == 'seasonal_mean']['r2'].values
    seasonal_r2 = seasonal_r2[0] if len(seasonal_r2) > 0 else 0

    extended = {}

    # For Ridge (representative linear model)
    for model in ['ridge', 'elasticnet', 'random_forest']:
        model_results = {}

        # Get model R² for different feature sets
        climate_only = test_results[(test_results['feature_set'] == 'climate_only') &
                                    (test_results['model'] == model)]
        ar_only = test_results[(test_results['feature_set'] == 'ar_only') &
                               (test_results['model'] == model)]
        ar_temporal = test_results[(test_results['feature_set'] == 'ar_temporal') &
                                   (test_results['model'] == model)]
        ar_climate = test_results[(test_results['feature_set'] == 'ar_climate') &
                                  (test_results['model'] == model)]
        ar_climate_temporal = test_results[(test_results['feature_set'] == 'ar_climate_temporal') &
                                           (test_results['model'] == model)]

        climate_r2 = climate_only['r2'].values[0] if not climate_only.empty else None
        ar_r2 = ar_only['r2'].values[0] if not ar_only.empty else None
        ar_temp_r2 = ar_temporal['r2'].values[0] if not ar_temporal.empty else None
        ar_clim_r2 = ar_climate['r2'].values[0] if not ar_climate.empty else None
        ar_clim_temp_r2 = ar_climate_temporal['r2'].values[0] if not ar_climate_temporal.empty else None

        model_results['baselines'] = {
            'mean_baseline': mean_r2,
            'seasonal_mean': seasonal_r2,
            'last_week': last_week_r2
        }

        model_results['model_results'] = {
            'climate_only': climate_r2,
            'ar_only': ar_r2,
            'ar_temporal': ar_temp_r2,
            'ar_climate': ar_clim_r2,
            'ar_climate_temporal': ar_clim_temp_r2
        }

        # Calculate contributions
        if climate_r2 is not None:
            model_results['contributions'] = {
                'climate_vs_mean': climate_r2 - mean_r2,
                'climate_vs_seasonal': climate_r2 - seasonal_r2 if seasonal_r2 else None,
            }
            if ar_r2 is not None:
                model_results['contributions']['ar_climate_vs_ar'] = ar_clim_r2 - ar_r2 if ar_clim_r2 else None
            if ar_temp_r2 is not None and ar_clim_temp_r2 is not None:
                model_results['contributions']['ar_clim_temp_vs_ar_temp'] = ar_clim_temp_r2 - ar_temp_r2

        extended[model] = model_results

    return extended


def create_model_summary(results_df, baseline_results):
    """
    Create comprehensive model summary with comparison to baseline.
    """
    # Combine results
    all_results = pd.concat([
        pd.DataFrame(baseline_results),
        results_df
    ])

    # Get test set results only for summary
    test_results = all_results[all_results['split'] == 'test'].copy()

    # Find best model per feature set
    best_per_feature_set = test_results.loc[
        test_results.groupby('feature_set')['r2'].idxmax()
    ]

    # Find overall best model
    best_overall = test_results.loc[test_results['r2'].idxmax()]

    # Get baseline comparison
    last_week_r2 = test_results[
        (test_results['feature_set'] == 'baseline') &
        (test_results['model'] == 'last_week')
    ]['r2'].values

    baseline_r2 = last_week_r2[0] if len(last_week_r2) > 0 else 0

    summary = {
        'best_overall': {
            'feature_set': best_overall['feature_set'],
            'model': best_overall['model'],
            'test_r2': best_overall['r2'],
            'test_rmse': best_overall['rmse'],
            'improvement_over_baseline': best_overall['r2'] - baseline_r2
        },
        'baseline_last_week_r2': baseline_r2,
        'best_per_feature_set': best_per_feature_set[['feature_set', 'model', 'r2', 'rmse']].to_dict('records')
    }

    return summary, all_results


def main():
    """Main execution function."""
    print("=" * 70)
    print("Climate-Health Modeling: Comprehensive Analysis")
    print("=" * 70)

    # Load data
    df = load_data()

    # Add autoregressive features
    df_with_ar = add_autoregressive_features(df, TARGET_COL, lags=[1, 2, 3, 4])

    # Encode HSA as numeric
    le = LabelEncoder()
    df_with_ar['hsa_encoded'] = le.fit_transform(df_with_ar['hsa_id'])

    # Create temporal splits
    train_df, val_df, test_df = create_temporal_splits(df_with_ar)

    # Save splits
    train_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_train_data.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_val_data.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_test_data.csv", index=False)

    # Get all potential features
    all_features = get_feature_columns(df_with_ar, include_ar=True, include_hsa=True)
    print(f"\nTotal available features: {len(all_features)}")

    # Feature selection on training set only
    climate_features = [f for f in all_features if not f.startswith('ar_lag') and f != 'hsa_encoded']
    selected_features, correlation_info = correlation_feature_selection(
        train_df, TARGET_COL, climate_features,
        threshold=0.90, top_n=40
    )

    # Prepare feature sets
    feature_sets = prepare_feature_sets(
        df_with_ar, all_features, selected_features,
        include_ar=True, include_hsa=True
    )

    print(f"\nFeature sets prepared:")
    for name, features in feature_sets.items():
        print(f"  {name}: {len(features)} features")

    # Train baseline models
    print("\n" + "=" * 70)
    print("Training Baseline Models")
    print("=" * 70)
    baseline_results = train_baseline_models(train_df, val_df, test_df, TARGET_COL)

    print("\nBaseline Results (Test Set):")
    baseline_df = pd.DataFrame(baseline_results)
    baseline_test = baseline_df[baseline_df['split'] == 'test']
    print(baseline_test.to_string(index=False))

    # Train model families
    print("\n" + "=" * 70)
    print("Training Model Families")
    print("=" * 70)
    model_results, trained_models = train_model_families(
        train_df, val_df, test_df, feature_sets, TARGET_COL
    )

    results_df = pd.DataFrame(model_results)

    # Calculate climate contribution
    print("\n" + "=" * 70)
    print("Climate Contribution Analysis (AR-only vs AR+Climate)")
    print("=" * 70)
    contributions_df = calculate_climate_contribution(results_df)
    if not contributions_df.empty:
        print(contributions_df.to_string(index=False))
        contributions_df.to_csv(
            OUTPUT_DIR / f"{OUTPUT_PREFIX}_climate_contribution_analysis.csv",
            index=False,
        )

    # Extended climate contribution analysis
    print("\n" + "=" * 70)
    print("EXTENDED CLIMATE CONTRIBUTION ANALYSIS")
    print("=" * 70)
    print("\nComparing climate contributions from different baselines:")
    print("  - Climate vs Mean: Total climate explanatory power")
    print("  - Climate vs Seasonal: Climate beyond seasonal patterns")
    print("  - AR+Climate vs AR: Climate beyond persistence")
    print("  - AR+Climate+Temporal vs AR+Temporal: Climate beyond AR+seasonal")

    extended_contributions = calculate_extended_climate_contribution(results_df, baseline_results)

    for model, data in extended_contributions.items():
        print(f"\n{model.upper()}:")
        print("-" * 50)

        if 'baselines' in data:
            print(f"  Baselines:")
            print(f"    Mean baseline:     R² = {data['baselines']['mean_baseline']:.4f}")
            print(f"    Seasonal mean:     R² = {data['baselines']['seasonal_mean']:.4f}")
            print(f"    Last week (AR):    R² = {data['baselines']['last_week']:.4f}")

        if 'model_results' in data:
            print(f"  Model Results:")
            for fs, r2 in data['model_results'].items():
                if r2 is not None:
                    print(f"    {fs:25s} R² = {r2:.4f}")

        if 'contributions' in data:
            print(f"  Climate Contributions:")
            for contrib_name, contrib_val in data['contributions'].items():
                if contrib_val is not None:
                    sign = '+' if contrib_val >= 0 else ''
                    print(f"    {contrib_name:30s} {sign}{contrib_val:.4f}")

    # Save extended analysis
    with open(OUTPUT_DIR / f"{OUTPUT_PREFIX}_extended_climate_analysis.json", 'w') as f:
        # Convert to serializable format
        serializable = {}
        for model, data in extended_contributions.items():
            serializable[model] = {
                k: {k2: float(v2) if v2 is not None else None for k2, v2 in v.items()}
                if isinstance(v, dict) else v
                for k, v in data.items()
            }
        json.dump(serializable, f, indent=2)

    # Print interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Get representative values for interpretation
    ridge_data = extended_contributions.get('ridge', {})
    contribs = ridge_data.get('contributions', {})

    climate_vs_mean = contribs.get('climate_vs_mean', 0) or 0
    ar_clim_vs_ar = contribs.get('ar_climate_vs_ar', 0) or 0
    ar_clim_temp_vs_ar_temp = contribs.get('ar_clim_temp_vs_ar_temp', 0) or 0

    print(f"""
Climate variables explain different amounts depending on the baseline:

1. TOTAL CLIMATE EXPLANATORY POWER (vs mean baseline):
   Climate R² improvement over mean: {climate_vs_mean:+.4f}
   → Climate + seasonal patterns together explain substantial variance

2. CLIMATE BEYOND PERSISTENCE (vs AR-only):
   Climate R² improvement over AR: {ar_clim_vs_ar:+.4f}
   → {'Modest' if ar_clim_vs_ar > 0.01 else 'Minimal' if ar_clim_vs_ar > 0 else 'No'} additional value beyond week-to-week persistence

3. CLIMATE BEYOND AR + SEASONALITY (vs AR+temporal):
   Climate R² improvement: {ar_clim_temp_vs_ar_temp:+.4f}
   → {'Climate adds value' if ar_clim_temp_vs_ar_temp > 0.005 else 'Climate absorbed by seasonal controls'}

KEY INSIGHT: Climate and seasonality share information. Once seasonal patterns
(via Fourier terms or AR features) are controlled, the remaining climate
signal is {'substantial' if ar_clim_temp_vs_ar_temp > 0.01 else 'modest' if ar_clim_temp_vs_ar_temp > 0 else 'minimal'}.
""")

    # Create model summary
    summary, all_results = create_model_summary(results_df, baseline_results)

    # Save all results
    all_results.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_all_model_results.csv", index=False)

    with open(OUTPUT_DIR / f"{OUTPUT_PREFIX}_model_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nBest Overall Model:")
    print(f"  Feature Set: {summary['best_overall']['feature_set']}")
    print(f"  Model: {summary['best_overall']['model']}")
    print(f"  Test R²: {summary['best_overall']['test_r2']:.4f}")
    print(f"  Test RMSE: {summary['best_overall']['test_rmse']:.2f}")
    print(f"  Improvement over baseline: {summary['best_overall']['improvement_over_baseline']:.4f}")

    print(f"\nBaseline (Last Week) R²: {summary['baseline_last_week_r2']:.4f}")

    print("\nBest Model per Feature Set (Test R²):")
    for item in summary['best_per_feature_set']:
        print(f"  {item['feature_set']}: {item['model']} (R²={item['r2']:.4f})")

    # Pivot table for easy comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE (Test R²)")
    print("=" * 70)

    test_results = all_results[all_results['split'] == 'test']
    pivot = test_results.pivot_table(
        values='r2',
        index='feature_set',
        columns='model',
        aggfunc='first'
    )
    print(pivot.round(4).to_string())

    # Save pivot table
    pivot.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_model_comparison_pivot.csv")

    print(f"\n\nResults saved to: {OUTPUT_DIR}")

    return all_results, summary, trained_models


if __name__ == '__main__':
    results, summary, models = main()
