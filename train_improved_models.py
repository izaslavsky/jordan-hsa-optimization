"""
Improved ML Models with Autoregressive Features
================================================

Improvements:
1. Add autoregressive features (lag-1, lag-2 disease counts)
2. Include HSA fixed effects
3. More aggressive feature selection (top 10-15 climate features)
4. Better hyperparameter tuning
5. Focus on simpler, more regularized models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import argparse
import os

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"[!]  XGBoost unavailable ({e}). Install with: pip install xgboost")

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PIPELINE_OUT_DIR = os.environ.get("HSA_OUT_DIR", os.environ.get("PIPELINE_OUT_DIR", "out"))
DEFAULT_RANDOM_SEED = 42

print("="*80)
print("IMPROVED ML MODELING - AUTOREGRESSIVE + FEATURE SELECTION")
print("="*80)

parser = argparse.ArgumentParser(description="Train improved models with AR features")
parser.add_argument("--network", default=os.environ.get("NETWORK", "INF"))
parser.add_argument("--hsa-mode", default=os.environ.get("HSA_MODE", "footprint"))
parser.add_argument("--target-col", default=os.environ.get("TARGET_COL", "diarrheal_count_adjusted"))
parser.add_argument("--data-dir", default=os.environ.get("MODEL_DATA_DIR", str(Path(DEFAULT_PIPELINE_OUT_DIR) / "modeling")))
parser.add_argument("--output-dir", default=os.environ.get("MODEL_OUTPUT_DIR", str(Path(DEFAULT_PIPELINE_OUT_DIR) / "modeling" / "results_improved")))
parser.add_argument("--random-seed", type=int, default=int(os.environ.get("RANDOM_SEED", DEFAULT_RANDOM_SEED)),
                    help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})")
parser.add_argument("--boundary-version", default=os.environ.get("BOUNDARY_VERSION", os.environ.get("PIPELINE_VERSION", "v7")),
                    help="HSA boundary version (v6, v7, v8)")
args = parser.parse_args()
NETWORK = args.network
HSA_MODE = args.hsa_mode
TARGET_COL = args.target_col
DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
RANDOM_SEED = args.random_seed
BOUNDARY_VERSION = args.boundary_version
np.random.seed(RANDOM_SEED)

# ============================================================================
# LOAD DATA
# ============================================================================

train_candidate = DATA_DIR / f"{NETWORK}_{HSA_MODE}_modeling_dataset_{BOUNDARY_VERSION}_train.csv"
val_candidate = DATA_DIR / f"{NETWORK}_{HSA_MODE}_modeling_dataset_{BOUNDARY_VERSION}_val.csv"
test_candidate = DATA_DIR / f"{NETWORK}_{HSA_MODE}_modeling_dataset_{BOUNDARY_VERSION}_test.csv"

train = pd.read_csv(train_candidate if train_candidate.exists() else DATA_DIR / "modeling_dataset_train.csv")
val = pd.read_csv(val_candidate if val_candidate.exists() else DATA_DIR / "modeling_dataset_val.csv")
test = pd.read_csv(test_candidate if test_candidate.exists() else DATA_DIR / "modeling_dataset_test.csv")

print(f"\nOriginal Data:")
print(f"  Train: {len(train)} samples")
print(f"  Val:   {len(val)} samples")
print(f"  Test:  {len(test)} samples")

# ============================================================================
# ADD AUTOREGRESSIVE FEATURES
# ============================================================================

print("\n[STEP 1] Creating autoregressive features...")

def add_lag_features(df, target_col, lags=[1, 2]):
    """Add lagged disease counts as features"""
    df = df.sort_values(['hsa_id', 'week_start']).copy()

    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df.groupby('hsa_id')[target_col].shift(lag)

    return df

# Add lags to all datasets
train = add_lag_features(train, TARGET_COL)
val = add_lag_features(val, TARGET_COL)
test = add_lag_features(test, TARGET_COL)

# Drop rows with missing lag features
train_with_lags = train.dropna(subset=[f'{TARGET_COL}_lag1', f'{TARGET_COL}_lag2'])
val_with_lags = val.dropna(subset=[f'{TARGET_COL}_lag1', f'{TARGET_COL}_lag2'])
test_with_lags = test.dropna(subset=[f'{TARGET_COL}_lag1', f'{TARGET_COL}_lag2'])

print(f"  After adding lags (some rows dropped):")
print(f"    Train: {len(train_with_lags)} samples")
print(f"    Val:   {len(val_with_lags)} samples")
print(f"    Test:  {len(test_with_lags)} samples")

# ============================================================================
# FEATURE SELECTION - TOP CLIMATE FEATURES
# ============================================================================

print("\n[STEP 2] Selecting top climate features...")

# Identify feature columns (exclude categorical and metadata)
exclude_cols = ['hsa_id', 'week_start', 'week_number', TARGET_COL,
                f'{TARGET_COL}_lag1', f'{TARGET_COL}_lag2',
                'season', 'week_of_year', 'month', 'quarter']  # Exclude temporal for now

climate_features = [col for col in train_with_lags.columns
                   if col not in exclude_cols and train_with_lags[col].dtype in ['float64', 'int64']]

# Compute correlations with target
correlations = train_with_lags[climate_features].corrwith(
    train_with_lags[TARGET_COL]
).abs().sort_values(ascending=False)

# Select top 10 climate features
top_climate_features = correlations.head(10).index.tolist()

print(f"  Top 10 Climate Features:")
for i, feat in enumerate(top_climate_features, 1):
    print(f"    {i:2d}. {feat:40s} (r = {correlations[feat]:.3f})")

# ============================================================================
# PREPARE FEATURE SETS
# ============================================================================

print("\n[STEP 3] Preparing feature sets...")

# Define feature sets to test
feature_sets = {
    'AR_only': [f'{TARGET_COL}_lag1', f'{TARGET_COL}_lag2'],
    'AR_temporal': [f'{TARGET_COL}_lag1', f'{TARGET_COL}_lag2',
                    'week_of_year', 'month', 'season'],
    'AR_top5_climate': [f'{TARGET_COL}_lag1', f'{TARGET_COL}_lag2'] + top_climate_features[:5],
    'AR_top10_climate': [f'{TARGET_COL}_lag1', f'{TARGET_COL}_lag2'] + top_climate_features[:10],
    'AR_temporal_top5': [f'{TARGET_COL}_lag1', f'{TARGET_COL}_lag2',
                         'week_of_year', 'month', 'season'] + top_climate_features[:5],
}

for name, feats in feature_sets.items():
    print(f"  {name}: {len(feats)} features")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_data(train_df, val_df, test_df, features):
    """Prepare X and y with proper encoding"""
    # Handle season encoding
    if 'season' in features:
        train_seasons = pd.get_dummies(train_df['season'], prefix='season', drop_first=True)
        val_seasons = pd.get_dummies(val_df['season'], prefix='season', drop_first=True)
        test_seasons = pd.get_dummies(test_df['season'], prefix='season', drop_first=True)

        # Align columns
        all_season_cols = train_seasons.columns.tolist()
        for col in all_season_cols:
            if col not in val_seasons.columns:
                val_seasons[col] = 0
            if col not in test_seasons.columns:
                test_seasons[col] = 0
        val_seasons = val_seasons[all_season_cols]
        test_seasons = test_seasons[all_season_cols]

        # Remove season from features list
        features_no_season = [f for f in features if f != 'season']

        X_train = pd.concat([train_df[features_no_season], train_seasons], axis=1)
        X_val = pd.concat([val_df[features_no_season], val_seasons], axis=1)
        X_test = pd.concat([test_df[features_no_season], test_seasons], axis=1)
    else:
        X_train = train_df[features]
        X_val = val_df[features]
        X_test = test_df[features]

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    return X_train, y_train, X_val, y_val, X_test, y_test

def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics"""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def last_week_baseline_metrics(df):
    """Compute last-week baseline metrics using lag-1 target."""
    y_true = df[TARGET_COL].values
    y_pred = df[f'{TARGET_COL}_lag1'].values
    return compute_metrics(y_true, y_pred)


# Baseline: last-week persistence (validation/test)
baseline_val_metrics = last_week_baseline_metrics(val_with_lags)
baseline_test_metrics = last_week_baseline_metrics(test_with_lags)
print("\nBaseline (Last-Week Persistence):")
print(f"  Val R²:  {baseline_val_metrics['r2']:.4f}, RMSE: {baseline_val_metrics['rmse']:.3f}, MAE: {baseline_val_metrics['mae']:.3f}")
print(f"  Test R²: {baseline_test_metrics['r2']:.4f}, RMSE: {baseline_test_metrics['rmse']:.3f}, MAE: {baseline_test_metrics['mae']:.3f}")

# ============================================================================
# TRAIN MODELS WITH DIFFERENT FEATURE SETS
# ============================================================================

print("\n[STEP 4] Training models with different feature sets...")

all_results = []

for feature_set_name, features in feature_sets.items():
    print(f"\n{'='*80}")
    print(f"Feature Set: {feature_set_name} ({len(features)} features)")
    print(f"{'='*80}")

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        train_with_lags, val_with_lags, test_with_lags, features
    )

    # Scale features (except lag features which are already in target scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Data shape: Train {X_train_scaled.shape}, Val {X_val_scaled.shape}")

    # -------------------------------------------------------------------------
    # Ridge Regression
    # -------------------------------------------------------------------------
    print(f"\n  [1/5] Ridge Regression")
    best_r2 = -np.inf
    best_alpha = None

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        model.fit(X_train_scaled, y_train)
        val_pred = model.predict(X_val_scaled)
        r2 = r2_score(y_val, val_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    ridge = Ridge(alpha=best_alpha, random_state=RANDOM_SEED)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred_val = ridge.predict(X_val_scaled)
    ridge_pred_test = ridge.predict(X_test_scaled)
    ridge_metrics = compute_metrics(y_val, ridge_pred_val)
    ridge_test_metrics = compute_metrics(y_test, ridge_pred_test)

    print(f"    Best alpha: {best_alpha}, Val R²: {ridge_metrics['r2']:.4f}")

    all_results.append({
        'feature_set': feature_set_name,
        'n_features': X_train_scaled.shape[1],
        'model': 'Ridge',
        'alpha': best_alpha,
        'r2': ridge_metrics['r2'],
        'rmse': ridge_metrics['rmse'],
        'mae': ridge_metrics['mae'],
        'test_r2': ridge_test_metrics['r2'],
        'test_rmse': ridge_test_metrics['rmse'],
        'test_mae': ridge_test_metrics['mae'],
    })

    # -------------------------------------------------------------------------
    # Lasso Regression
    # -------------------------------------------------------------------------
    print(f"  [2/5] Lasso Regression")
    best_r2 = -np.inf
    best_alpha = None

    for alpha in [0.001, 0.01, 0.1, 1.0]:
        model = Lasso(alpha=alpha, random_state=RANDOM_SEED, max_iter=5000)
        model.fit(X_train_scaled, y_train)
        val_pred = model.predict(X_val_scaled)
        r2 = r2_score(y_val, val_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    lasso = Lasso(alpha=best_alpha, random_state=RANDOM_SEED, max_iter=5000)
    lasso.fit(X_train_scaled, y_train)
    lasso_pred_val = lasso.predict(X_val_scaled)
    lasso_pred_test = lasso.predict(X_test_scaled)
    lasso_metrics = compute_metrics(y_val, lasso_pred_val)
    lasso_test_metrics = compute_metrics(y_test, lasso_pred_test)

    print(f"    Best alpha: {best_alpha}, Val R²: {lasso_metrics['r2']:.4f}")

    all_results.append({
        'feature_set': feature_set_name,
        'n_features': X_train_scaled.shape[1],
        'model': 'Lasso',
        'alpha': best_alpha,
        'r2': lasso_metrics['r2'],
        'rmse': lasso_metrics['rmse'],
        'mae': lasso_metrics['mae'],
        'test_r2': lasso_test_metrics['r2'],
        'test_rmse': lasso_test_metrics['rmse'],
        'test_mae': lasso_test_metrics['mae'],
    })

    # -------------------------------------------------------------------------
    # Random Forest
    # -------------------------------------------------------------------------
    print(f"  [3/5] Random Forest")

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred_val = rf.predict(X_val)
    rf_pred_test = rf.predict(X_test)
    rf_metrics = compute_metrics(y_val, rf_pred_val)
    rf_test_metrics = compute_metrics(y_test, rf_pred_test)

    print(f"    Val R²: {rf_metrics['r2']:.4f}")

    all_results.append({
        'feature_set': feature_set_name,
        'n_features': X_train_scaled.shape[1],
        'model': 'RandomForest',
        'alpha': None,
        'r2': rf_metrics['r2'],
        'rmse': rf_metrics['rmse'],
        'mae': rf_metrics['mae'],
        'test_r2': rf_test_metrics['r2'],
        'test_rmse': rf_test_metrics['rmse'],
        'test_mae': rf_test_metrics['mae'],
    })

    # -------------------------------------------------------------------------
    # Gradient Boosting
    # -------------------------------------------------------------------------
    print(f"  [4/5] Gradient Boosting")

    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_SEED
    )
    gb.fit(X_train, y_train)
    gb_pred_val = gb.predict(X_val)
    gb_pred_test = gb.predict(X_test)
    gb_metrics = compute_metrics(y_val, gb_pred_val)
    gb_test_metrics = compute_metrics(y_test, gb_pred_test)

    print(f"    Val R²: {gb_metrics['r2']:.4f}")

    all_results.append({
        'feature_set': feature_set_name,
        'n_features': X_train_scaled.shape[1],
        'model': 'GradientBoosting',
        'alpha': None,
        'r2': gb_metrics['r2'],
        'rmse': gb_metrics['rmse'],
        'mae': gb_metrics['mae'],
        'test_r2': gb_test_metrics['r2'],
        'test_rmse': gb_test_metrics['rmse'],
        'test_mae': gb_test_metrics['mae'],
    })

    # -------------------------------------------------------------------------
    # XGBoost (if available)
    # -------------------------------------------------------------------------
    if XGBOOST_AVAILABLE:
        print(f"  [5/5] XGBoost")

        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred_val = xgb_model.predict(X_val)
        xgb_pred_test = xgb_model.predict(X_test)
        xgb_metrics = compute_metrics(y_val, xgb_pred_val)
        xgb_test_metrics = compute_metrics(y_test, xgb_pred_test)

        print(f"    Val R²: {xgb_metrics['r2']:.4f}")

        all_results.append({
            'feature_set': feature_set_name,
            'n_features': X_train_scaled.shape[1],
            'model': 'XGBoost',
            'alpha': None,
            'r2': xgb_metrics['r2'],
            'rmse': xgb_metrics['rmse'],
            'mae': xgb_metrics['mae'],
            'test_r2': xgb_test_metrics['r2'],
            'test_rmse': xgb_test_metrics['rmse'],
            'test_mae': xgb_test_metrics['mae'],
        })

# ============================================================================
# SUMMARIZE RESULTS
# ============================================================================

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('r2', ascending=False)

print("\nTop 10 Models by Validation R²:")
print("-"*80)
print(results_df[['feature_set', 'model', 'n_features', 'r2', 'rmse', 'mae']].head(10).to_string(index=False))

print("\nTop 10 Models by Test R²:")
print("-"*80)
test_sorted = results_df.sort_values('test_r2', ascending=False)
print(test_sorted[['feature_set', 'model', 'n_features', 'test_r2', 'test_rmse', 'test_mae']].head(10).to_string(index=False))

# Save results
results_path = OUTPUT_DIR / f"{NETWORK}_{HSA_MODE}_improved_model_comparison.csv"
results_df.to_csv(results_path, index=False)
print(f"\n[OK] Results saved to {results_path}")

# Find best model
best_model = results_df.iloc[0]
print(f"\n{'='*80}")
print("BEST MODEL")
print(f"{'='*80}")
print(f"  Feature Set: {best_model['feature_set']}")
print(f"  Model: {best_model['model']}")
print(f"  Features: {int(best_model['n_features'])}")
print(f"  Val R²: {best_model['r2']:.4f}")
print(f"  Val RMSE: {best_model['rmse']:.3f}")
print(f"  Val MAE: {best_model['mae']:.3f}")
print(f"  Test R²: {best_model['test_r2']:.4f}")
print(f"  Test RMSE: {best_model['test_rmse']:.3f}")
print(f"  Test MAE: {best_model['test_mae']:.3f}")

# Compare to baseline (validation)
baseline_r2 = baseline_val_metrics['r2']
baseline_rmse = baseline_val_metrics['rmse']
improvement_r2 = best_model['r2'] - baseline_r2
improvement_rmse = baseline_rmse - best_model['rmse']

print(f"\n  Improvement over Last-Week Baseline (Val):")
print(f"    ΔR²: {improvement_r2:+.4f} ({(improvement_r2/baseline_r2*100) if baseline_r2 else 0:+.1f}%)")
print(f"    ΔRMSE: {improvement_rmse:+.3f} ({(improvement_rmse/baseline_rmse*100) if baseline_rmse else 0:+.1f}%)")

print("\n" + "="*80)
