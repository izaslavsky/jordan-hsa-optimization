#!/usr/bin/env python3
"""
Anomaly-Based Climate-Health Modeling

Tests whether climate ANOMALIES (deviations from seasonal means) predict
disease ANOMALIES, removing the shared seasonal signal that confounds
standard climate-disease analysis.

This approach answers: "Do unusual climate conditions predict unusual disease levels?"
rather than "Does seasonal climate variation explain seasonal disease variation?"

Author: Climate-Health Research Team
Date: January 2025
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

DEFAULT_PIPELINE_OUT_DIR = os.environ.get("HSA_OUT_DIR", os.environ.get("PIPELINE_OUT_DIR", "out"))
DEFAULT_RANDOM_SEED = 42

parser = argparse.ArgumentParser(description="Anomaly-based climate-health modeling")
parser.add_argument("--network", default=os.environ.get("NETWORK", "INF"))
parser.add_argument("--hsa-mode", default=os.environ.get("HSA_MODE", "footprint"))
parser.add_argument("--target-col", default=os.environ.get("TARGET_COL", "diarrheal_count_adjusted"))
parser.add_argument("--input-csv", default=os.environ.get("MODEL_INPUT_CSV", None))
parser.add_argument("--output-dir", default=os.environ.get("MODEL_OUTPUT_DIR", str(Path(DEFAULT_PIPELINE_OUT_DIR) / "modeling" / "results_anomalies")))
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


def load_and_prepare_data():
    """Load data and compute anomalies."""
    df = pd.read_csv(INPUT_CSV)
    df['week_start'] = pd.to_datetime(df['week_start'])
    df['week_of_year'] = df['week_start'].dt.isocalendar().week.astype(int)

    # Sort for proper processing
    df = df.sort_values(['hsa_id', 'week_start']).reset_index(drop=True)

    return df


def compute_anomalies(df, columns, groupby_cols=['hsa_id', 'week_of_year']):
    """
    Compute anomalies as deviations from seasonal means.

    For each HSA and week-of-year, calculate the mean value,
    then subtract to get anomalies.
    """
    df_out = df.copy()

    for col in columns:
        if col not in df.columns:
            continue
        # Calculate seasonal mean for each HSA
        seasonal_mean = df.groupby(groupby_cols)[col].transform('mean')
        # Anomaly = actual - seasonal mean
        df_out[f'{col}_anomaly'] = df[col] - seasonal_mean
        # Also store the seasonal mean for reference
        df_out[f'{col}_seasonal_mean'] = seasonal_mean

    return df_out


def create_temporal_splits(df, train_pct=0.75, val_pct=0.125):
    """Create train/val/test splits based on time."""
    weeks = sorted(df['week_start'].unique())
    n_weeks = len(weeks)

    train_end = int(n_weeks * train_pct)
    val_end = int(n_weeks * (train_pct + val_pct))

    train_weeks = weeks[:train_end]
    val_weeks = weeks[train_end:val_end]
    test_weeks = weeks[val_end:]

    train = df[df['week_start'].isin(train_weeks)].copy()
    val = df[df['week_start'].isin(val_weeks)].copy()
    test = df[df['week_start'].isin(test_weeks)].copy()

    return train, val, test


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, model_name, model):
    """Train model and return metrics."""
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    results = {
        'model': model_name,
        'val_r2': r2_score(y_val, val_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'val_mae': mean_absolute_error(y_val, val_pred),
        'test_r2': r2_score(y_test, test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_r2': r2_score(y_train, model.predict(X_train)),
    }

    return results, model


def main():
    print("="*70)
    print("ANOMALY-BASED CLIMATE-HEALTH MODELING")
    print("Testing if climate anomalies predict disease anomalies")
    print("="*70)

    # Load data
    df = load_and_prepare_data()
    print(f"\nDataset: {len(df)} rows, {df['hsa_id'].nunique()} HSAs, {df['week_start'].nunique()} weeks")

    # Define climate variables
    climate_vars = [
        'T_mean_week_C', 'T_max_week_C', 'P_total_week',
        'Td_week_C', 'E_week_mm_per_day', 'SM1_week',
        'hours_above_30C_week', 'water_deficit_mm_week',
        'P_sum_lag_w-1', 'P_sum_lag_w-2', 'P_sum_lag_w-3',
        'T_mean_d-7_C', 'T_mean_d-14_C'
    ]
    climate_vars = [c for c in climate_vars if c in df.columns]

    print(f"Climate variables: {len(climate_vars)}")

    # Compute anomalies for target and climate variables
    print("\nComputing seasonal anomalies...")
    all_vars = [TARGET_COL] + climate_vars
    df = compute_anomalies(df, all_vars)

    target_anomaly = f'{TARGET_COL}_anomaly'
    climate_anomalies = [f'{c}_anomaly' for c in climate_vars]

    # Check anomaly statistics
    print(f"\nTarget anomaly statistics:")
    print(f"  Mean: {df[target_anomaly].mean():.4f} (should be ~0)")
    print(f"  Std: {df[target_anomaly].std():.2f}")
    print(f"  Min: {df[target_anomaly].min():.1f}")
    print(f"  Max: {df[target_anomaly].max():.1f}")

    # Create AR anomaly features
    print("\nCreating AR anomaly features...")
    for lag in [1, 2]:
        df[f'ar_anomaly_lag{lag}'] = df.groupby('hsa_id')[target_anomaly].shift(lag)

    df = df.dropna(subset=['ar_anomaly_lag1', 'ar_anomaly_lag2'] + climate_anomalies)
    print(f"After dropping NaN: {len(df)} rows")

    # Create temporal splits
    train, val, test = create_temporal_splits(df)
    print(f"\nTemporal split:")
    print(f"  Train: {len(train)} rows")
    print(f"  Val: {len(val)} rows")
    print(f"  Test: {len(test)} rows")

    # Prepare feature sets
    ar_anomaly_features = ['ar_anomaly_lag1', 'ar_anomaly_lag2']

    # Scale features
    scaler_ar = StandardScaler()
    scaler_climate = StandardScaler()

    X_train_ar = scaler_ar.fit_transform(train[ar_anomaly_features])
    X_val_ar = scaler_ar.transform(val[ar_anomaly_features])
    X_test_ar = scaler_ar.transform(test[ar_anomaly_features])

    X_train_clim = scaler_climate.fit_transform(train[climate_anomalies])
    X_val_clim = scaler_climate.transform(val[climate_anomalies])
    X_test_clim = scaler_climate.transform(test[climate_anomalies])

    y_train = train[target_anomaly].values
    y_val = val[target_anomaly].values
    y_test = test[target_anomaly].values

    # Combined features
    X_train_full = np.hstack([X_train_ar, X_train_clim])
    X_val_full = np.hstack([X_val_ar, X_val_clim])
    X_test_full = np.hstack([X_test_ar, X_test_clim])

    results_list = []

    # =========================================================================
    # BASELINE: Mean prediction (always predict 0 for anomalies)
    # =========================================================================
    print("\n" + "="*70)
    print("BASELINE MODELS")
    print("="*70)

    # Zero baseline (predict 0 for all anomalies)
    zero_pred = np.zeros_like(y_test)
    r2_zero = r2_score(y_test, zero_pred)
    rmse_zero = np.sqrt(mean_squared_error(y_test, zero_pred))
    print(f"\nZero baseline (predict 0): R² = {r2_zero:.4f}, RMSE = {rmse_zero:.2f}")
    results_list.append({
        'feature_set': 'baseline', 'model': 'zero_baseline',
        'test_r2': r2_zero, 'test_rmse': rmse_zero, 'test_mae': mean_absolute_error(y_test, zero_pred),
        'val_r2': r2_score(y_val, np.zeros_like(y_val)), 'n_features': 0
    })

    # =========================================================================
    # MODEL SET 1: AR Anomalies Only
    # =========================================================================
    print("\n" + "="*70)
    print("AR ANOMALY MODELS (lagged disease anomalies only)")
    print("="*70)

    models = {
        'ridge': Ridge(alpha=1.0, random_state=RANDOM_SEED),
        'lasso': Lasso(alpha=0.1, random_state=RANDOM_SEED),
        'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_SEED),
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_SEED),
    }

    ar_results = {}
    for name, model in models.items():
        res, _ = train_and_evaluate(X_train_ar, y_train, X_val_ar, y_val, X_test_ar, y_test, name, model)
        ar_results[name] = res
        print(f"  {name:20s} Test R² = {res['test_r2']:.4f}, RMSE = {res['test_rmse']:.2f}")
        results_list.append({
            'feature_set': 'ar_anomaly', 'model': name,
            'test_r2': res['test_r2'], 'test_rmse': res['test_rmse'], 'test_mae': res['test_mae'],
            'val_r2': res['val_r2'], 'train_r2': res['train_r2'], 'n_features': 2
        })

    # =========================================================================
    # MODEL SET 2: Climate Anomalies Only
    # =========================================================================
    print("\n" + "="*70)
    print("CLIMATE ANOMALY MODELS (climate anomalies only, no AR)")
    print("="*70)

    climate_results = {}
    for name, model_class in [
        ('ridge', Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ('lasso', Lasso(alpha=0.1, random_state=RANDOM_SEED)),
        ('elasticnet', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_SEED)),
        ('random_forest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)),
        ('gradient_boosting', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)),
    ]:
        res, _ = train_and_evaluate(X_train_clim, y_train, X_val_clim, y_val, X_test_clim, y_test, name, model_class)
        climate_results[name] = res
        print(f"  {name:20s} Test R² = {res['test_r2']:.4f}, RMSE = {res['test_rmse']:.2f}")
        results_list.append({
            'feature_set': 'climate_anomaly', 'model': name,
            'test_r2': res['test_r2'], 'test_rmse': res['test_rmse'], 'test_mae': res['test_mae'],
            'val_r2': res['val_r2'], 'train_r2': res['train_r2'], 'n_features': len(climate_anomalies)
        })

    # =========================================================================
    # MODEL SET 3: AR + Climate Anomalies
    # =========================================================================
    print("\n" + "="*70)
    print("COMBINED MODELS (AR + Climate anomalies)")
    print("="*70)

    combined_results = {}
    feature_importance_dict = {}
    for name, model_class in [
        ('ridge', Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ('lasso', Lasso(alpha=0.1, random_state=RANDOM_SEED)),
        ('elasticnet', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_SEED)),
        ('random_forest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)),
        ('gradient_boosting', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)),
    ]:
        res, fitted_model = train_and_evaluate(X_train_full, y_train, X_val_full, y_val, X_test_full, y_test, name, model_class)
        combined_results[name] = res
        print(f"  {name:20s} Test R² = {res['test_r2']:.4f}, RMSE = {res['test_rmse']:.2f}")
        results_list.append({
            'feature_set': 'ar_climate_anomaly', 'model': name,
            'test_r2': res['test_r2'], 'test_rmse': res['test_rmse'], 'test_mae': res['test_mae'],
            'val_r2': res['val_r2'], 'train_r2': res['train_r2'],
            'n_features': len(ar_anomaly_features) + len(climate_anomalies)
        })

        # Extract feature importance
        all_features = ar_anomaly_features + climate_anomalies
        if hasattr(fitted_model, 'feature_importances_'):
            feature_importance_dict[name] = dict(zip(all_features, fitted_model.feature_importances_))
        elif hasattr(fitted_model, 'coef_'):
            feature_importance_dict[name] = dict(zip(all_features, np.abs(fitted_model.coef_)))

    # =========================================================================
    # CLIMATE CONTRIBUTION ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("CLIMATE ANOMALY CONTRIBUTION ANALYSIS")
    print("="*70)
    print("\nDoes adding climate anomalies improve over AR anomalies alone?")
    print("-"*70)
    print(f"{'Model':<20} {'AR-only R²':>12} {'AR+Climate R²':>14} {'Climate Δ':>12} {'Relative %':>12}")
    print("-"*70)

    climate_contributions = []
    for name in models.keys():
        ar_r2 = ar_results[name]['test_r2']
        combined_r2 = combined_results[name]['test_r2']
        delta = combined_r2 - ar_r2
        relative = (delta / max(ar_r2, 0.001)) * 100 if ar_r2 > 0 else 0
        print(f"{name:<20} {ar_r2:>12.4f} {combined_r2:>14.4f} {delta:>+12.4f} {relative:>+11.1f}%")
        climate_contributions.append({
            'model': name, 'ar_only_r2': ar_r2, 'ar_climate_r2': combined_r2,
            'climate_contribution': delta, 'relative_pct': relative
        })

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (from Random Forest)")
    print("="*70)

    if 'random_forest' in feature_importance_dict:
        imp = feature_importance_dict['random_forest']
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)

        # Separate AR and climate features
        ar_importance = sum(v for k, v in sorted_imp if 'ar_anomaly' in k)
        climate_importance = sum(v for k, v in sorted_imp if 'ar_anomaly' not in k)

        print(f"\nAggregate importance:")
        print(f"  AR anomaly features: {ar_importance:.3f} ({ar_importance*100:.1f}%)")
        print(f"  Climate anomaly features: {climate_importance:.3f} ({climate_importance*100:.1f}%)")

        print(f"\nTop 10 individual features:")
        for feat, importance in sorted_imp[:10]:
            bar = "█" * int(importance * 50)
            print(f"  {feat:<35} {importance:.4f} {bar}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY: ANOMALY-BASED ANALYSIS")
    print("="*70)

    # Best models
    best_ar = max(ar_results.items(), key=lambda x: x[1]['test_r2'])
    best_climate = max(climate_results.items(), key=lambda x: x[1]['test_r2'])
    best_combined = max(combined_results.items(), key=lambda x: x[1]['test_r2'])

    print(f"\nBest AR-anomaly model: {best_ar[0]} (R² = {best_ar[1]['test_r2']:.4f})")
    print(f"Best Climate-anomaly model: {best_climate[0]} (R² = {best_climate[1]['test_r2']:.4f})")
    print(f"Best Combined model: {best_combined[0]} (R² = {best_combined[1]['test_r2']:.4f})")

    # Average climate contribution
    avg_contribution = np.mean([c['climate_contribution'] for c in climate_contributions])
    print(f"\nAverage climate contribution: {avg_contribution:+.4f} R²")

    if avg_contribution > 0.01:
        print("\n→ Climate anomalies DO add predictive value beyond AR anomalies.")
        print("  Unusual weather conditions help predict unusual disease levels.")
    elif avg_contribution > 0:
        print("\n→ Climate anomalies add MODEST predictive value beyond AR anomalies.")
        print("  The effect is small but positive.")
    else:
        print("\n→ Climate anomalies do NOT add predictive value beyond AR anomalies.")
        print("  Disease anomalies are driven primarily by persistence, not weather anomalies.")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results_df = pd.DataFrame(results_list)
    results_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_anomaly_model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    contributions_df = pd.DataFrame(climate_contributions)
    contributions_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_anomaly_climate_contribution.csv"
    contributions_df.to_csv(contributions_path, index=False)

    print(f"\nResults saved to: {OUTPUT_DIR}")

    return results_df, contributions_df


if __name__ == "__main__":
    results, contributions = main()
