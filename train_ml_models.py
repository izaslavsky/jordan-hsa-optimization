"""
ML Model Training & Evaluation Script
======================================

Purpose: Train and evaluate machine learning models to predict diarrheal disease
         outcomes from climate variables. Implements proper temporal validation
         and extracts feature importance.

Models Implemented:
    1. Baseline models (mean, seasonal, last-week)
    2. Linear models (Ridge, Lasso, ElasticNet)
    3. GLM models (Poisson, Negative Binomial)
    4. Tree-based models (Random Forest, XGBoost, LightGBM)
    5. Ensemble models (Voting, Stacking)

Outputs:
    - Model performance metrics (RMSE, MAE, R², Poisson deviance)
    - Feature importance rankings
    - Model predictions
    - Diagnostic plots
    - SHAP analysis

Author: ML Modeling Team
Date: 2024-12-13
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# ML libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# GLM models
from sklearn.linear_model import PoissonRegressor
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, NegativeBinomial

# Tree-based models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"[!]  XGBoost unavailable ({e}). Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[!]  LightGBM not installed. Install with: pip install lightgbm")

# SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[!]  SHAP not installed. Install with: pip install shap")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories

# Input files (network/mode aware)
DEFAULT_PIPELINE_OUT_DIR = os.environ.get("HSA_OUT_DIR", os.environ.get("PIPELINE_OUT_DIR", "out"))
DEFAULT_RANDOM_SEED = 42

parser = argparse.ArgumentParser(description="Train ML models for climate-health modeling")
parser.add_argument("--network", default=os.environ.get("NETWORK", "INF"))
parser.add_argument("--hsa-mode", default=os.environ.get("HSA_MODE", "footprint"))
parser.add_argument("--target-col", default=os.environ.get("TARGET_COL", "diarrheal_count_adjusted"))
parser.add_argument("--data-dir", default=os.environ.get("MODEL_DATA_DIR", str(Path(DEFAULT_PIPELINE_OUT_DIR) / "modeling")))
parser.add_argument("--output-dir", default=os.environ.get("MODEL_OUTPUT_DIR", str(Path(DEFAULT_PIPELINE_OUT_DIR) / "modeling" / "results")))
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

train_candidate = DATA_DIR / f"{NETWORK}_{HSA_MODE}_modeling_dataset_{BOUNDARY_VERSION}_train.csv"
val_candidate = DATA_DIR / f"{NETWORK}_{HSA_MODE}_modeling_dataset_{BOUNDARY_VERSION}_val.csv"
test_candidate = DATA_DIR / f"{NETWORK}_{HSA_MODE}_modeling_dataset_{BOUNDARY_VERSION}_test.csv"

TRAIN_FILE = train_candidate if train_candidate.exists() else DATA_DIR / "modeling_dataset_train.csv"
VAL_FILE = val_candidate if val_candidate.exists() else DATA_DIR / "modeling_dataset_val.csv"
TEST_FILE = test_candidate if test_candidate.exists() else DATA_DIR / "modeling_dataset_test.csv"
metadata_candidate = DATA_DIR / f"{NETWORK}_{HSA_MODE}_modeling_dataset_{BOUNDARY_VERSION}_metadata.json"
METADATA_FILE = metadata_candidate if metadata_candidate.exists() else DATA_DIR / "modeling_dataset_metadata.json"
np.random.seed(RANDOM_SEED)

# Model configuration
IDENTIFIER_COLS = ['hsa_id', 'week_start', 'week_number']
TEMPORAL_COLS = ['week_of_year', 'month', 'quarter', 'season', 'days_since_start']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_temporal_splits(df, train_pct=0.75, val_pct=0.125):
    """Create strict temporal train/val/test splits."""
    time_col = 'week_number' if 'week_number' in df.columns else 'week_start'
    if time_col == 'week_start':
        df = df.copy()
        df['week_start'] = pd.to_datetime(df['week_start'])
        unique_times = sorted(df['week_start'].unique())
    else:
        unique_times = sorted(df['week_number'].unique())

    n_times = len(unique_times)
    train_end = int(n_times * train_pct)
    val_end = int(n_times * (train_pct + val_pct))

    train_times = unique_times[:train_end]
    val_times = unique_times[train_end:val_end]
    test_times = unique_times[val_end:]

    train_df = df[df[time_col].isin(train_times)].copy()
    val_df = df[df[time_col].isin(val_times)].copy()
    test_df = df[df[time_col].isin(test_times)].copy()

    print(f"Temporal split using '{time_col}':")
    print(f"  Train: {len(train_df)} rows, {len(train_times)} periods")
    print(f"  Val:   {len(val_df)} rows, {len(val_times)} periods")
    print(f"  Test:  {len(test_df)} rows, {len(test_times)} periods")

    return train_df, val_df, test_df


def load_datasets():
    """Load train/val/test datasets"""
    print("Loading datasets...")

    if TRAIN_FILE.exists() and VAL_FILE.exists() and TEST_FILE.exists():
        train = pd.read_csv(TRAIN_FILE)
        val = pd.read_csv(VAL_FILE)
        test = pd.read_csv(TEST_FILE)
    else:
        full_candidate = DATA_DIR / f"{NETWORK}_{HSA_MODE}_modeling_dataset_{BOUNDARY_VERSION}.csv"
        full_fallback = DATA_DIR / "modeling_dataset.csv"
        source_path = full_candidate if full_candidate.exists() else full_fallback

        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing train/val/test splits and no full dataset found in {DATA_DIR}"
            )

        print(f"  Split files missing. Creating temporal splits from: {source_path}")
        full_df = pd.read_csv(source_path)
        train, val, test = create_temporal_splits(full_df)

        # Save versioned splits for reuse
        train.to_csv(train_candidate, index=False)
        val.to_csv(val_candidate, index=False)
        test.to_csv(test_candidate, index=False)
        print(f"  Saved splits to: {train_candidate}, {val_candidate}, {test_candidate}")

    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")

    return train, val, test


def prepare_features(train, val, test, include_temporal=True, include_hsa_dummies=False):
    """
    Prepare X and y matrices from dataframes

    Args:
        train, val, test: DataFrames
        include_temporal: Whether to include temporal features
        include_hsa_dummies: Whether to include HSA fixed effects

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    """
    # Identify feature columns
    exclude_cols = IDENTIFIER_COLS + [TARGET_COL]
    if not include_temporal:
        exclude_cols += TEMPORAL_COLS

    # Climate features
    climate_features = [col for col in train.columns if col not in exclude_cols]

    # Handle categorical features (season)
    if 'season' in climate_features:
        # Combine all data to get all categories
        all_seasons = pd.concat([train['season'], val['season'], test['season']])

        # One-hot encode season with all categories
        train_seasons = pd.get_dummies(train['season'], prefix='season', drop_first=True)
        val_seasons = pd.get_dummies(val['season'], prefix='season', drop_first=True)
        test_seasons = pd.get_dummies(test['season'], prefix='season', drop_first=True)

        # Get all possible season columns (from training)
        all_season_cols = train_seasons.columns.tolist()

        # Add missing columns to val and test (with zeros)
        for col in all_season_cols:
            if col not in val_seasons.columns:
                val_seasons[col] = 0
            if col not in test_seasons.columns:
                test_seasons[col] = 0

        # Ensure same column order
        val_seasons = val_seasons[all_season_cols]
        test_seasons = test_seasons[all_season_cols]

        # Remove original season column
        climate_features = [col for col in climate_features if col != 'season']

        # Add encoded columns
        X_train = pd.concat([train[climate_features], train_seasons], axis=1)
        X_val = pd.concat([val[climate_features], val_seasons], axis=1)
        X_test = pd.concat([test[climate_features], test_seasons], axis=1)
    else:
        X_train = train[climate_features]
        X_val = val[climate_features]
        X_test = test[climate_features]

    # Ensure all features are numeric (drop any remaining non-numeric columns)
    non_numeric = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    if non_numeric:
        print(f"  Warning: dropping non-numeric features: {non_numeric}")
        X_train = X_train.drop(columns=non_numeric)
        X_val = X_val.drop(columns=[c for c in non_numeric if c in X_val.columns])
        X_test = X_test.drop(columns=[c for c in non_numeric if c in X_test.columns])

    # Add HSA dummy variables if requested
    if include_hsa_dummies:
        train_hsa = pd.get_dummies(train['hsa_id'], prefix='hsa', drop_first=True)
        val_hsa = pd.get_dummies(val['hsa_id'], prefix='hsa', drop_first=True)
        test_hsa = pd.get_dummies(test['hsa_id'], prefix='hsa', drop_first=True)

        # Get all possible HSA columns (from training)
        all_hsa_cols = train_hsa.columns.tolist()

        # Add missing columns to val and test (with zeros)
        for col in all_hsa_cols:
            if col not in val_hsa.columns:
                val_hsa[col] = 0
            if col not in test_hsa.columns:
                test_hsa[col] = 0

        # Ensure same column order
        val_hsa = val_hsa[all_hsa_cols]
        test_hsa = test_hsa[all_hsa_cols]

        X_train = pd.concat([X_train, train_hsa], axis=1)
        X_val = pd.concat([X_val, val_hsa], axis=1)
        X_test = pd.concat([X_test, test_hsa], axis=1)

    # Extract target
    y_train = train[TARGET_COL].values
    y_val = val[TARGET_COL].values
    y_test = test[TARGET_COL].values

    # Get feature names
    feature_names = X_train.columns.tolist()

    # Convert to numpy arrays
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def scale_features(X_train, X_val, X_test):
    """Standardize features using training set statistics"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def compute_metrics(y_true, y_pred, model_name="Model"):
    """Compute comprehensive evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)

    # MAPE (handle zeros)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    # Poisson deviance (for count data)
    # D = 2 * sum(y * log(y/y_pred) - (y - y_pred))
    # Handle zeros and negatives in predictions
    y_pred_safe = np.maximum(y_pred, 1e-10)
    with np.errstate(divide='ignore', invalid='ignore'):
        deviance = 2 * np.sum(y_true * np.log(y_true / y_pred_safe + 1e-10) - (y_true - y_pred_safe))

    metrics = {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'median_ae': median_ae,
        'mape': mape,
        'poisson_deviance': deviance,
        'n_samples': len(y_true)
    }

    return metrics


def print_metrics(metrics, dataset="Validation"):
    """Pretty print metrics"""
    print(f"\n  {dataset} Metrics:")
    print(f"    RMSE:              {metrics['rmse']:.3f}")
    print(f"    MAE:               {metrics['mae']:.3f}")
    print(f"    R²:                {metrics['r2']:.3f}")
    print(f"    Median AE:         {metrics['median_ae']:.3f}")
    if not np.isnan(metrics['mape']):
        print(f"    MAPE:              {metrics['mape']:.1f}%")
    print(f"    Poisson Deviance:  {metrics['poisson_deviance']:.1f}")


# ============================================================================
# BASELINE MODELS
# ============================================================================

def train_baseline_models(y_train, y_val, y_test, train_df, val_df, test_df):
    """
    Train simple baseline models

    Returns:
        Dictionary of baseline predictions and metrics
    """
    print("\n" + "="*80)
    print("BASELINE MODELS")
    print("="*80)

    results = {}

    # -------------------------------------------------------------------------
    # Model 1: Mean Baseline
    # -------------------------------------------------------------------------
    print("\n[1/3] Mean Baseline")
    mean_pred = np.full_like(y_val, y_train.mean())

    metrics_val = compute_metrics(y_val, mean_pred, "Mean Baseline")
    print_metrics(metrics_val, "Validation")

    mean_pred_test = np.full_like(y_test, y_train.mean())
    metrics_test = compute_metrics(y_test, mean_pred_test, "Mean Baseline")

    results['mean_baseline'] = {
        'val_metrics': metrics_val,
        'test_metrics': metrics_test,
        'val_predictions': mean_pred,
        'test_predictions': mean_pred_test
    }

    # -------------------------------------------------------------------------
    # Model 2: Seasonal Baseline (by week_of_year)
    # -------------------------------------------------------------------------
    print("\n[2/3] Seasonal Baseline")

    # Compute mean by week_of_year from training data
    seasonal_means = train_df.groupby('week_of_year')[TARGET_COL].mean()

    # Predict using seasonal means
    seasonal_pred_val = val_df['week_of_year'].map(seasonal_means).fillna(y_train.mean()).values
    seasonal_pred_test = test_df['week_of_year'].map(seasonal_means).fillna(y_train.mean()).values

    metrics_val = compute_metrics(y_val, seasonal_pred_val, "Seasonal Baseline")
    print_metrics(metrics_val, "Validation")

    metrics_test = compute_metrics(y_test, seasonal_pred_test, "Seasonal Baseline")

    results['seasonal_baseline'] = {
        'val_metrics': metrics_val,
        'test_metrics': metrics_test,
        'val_predictions': seasonal_pred_val,
        'test_predictions': seasonal_pred_test
    }

    # -------------------------------------------------------------------------
    # Model 3: Last-Week Baseline (Persistence)
    # -------------------------------------------------------------------------
    print("\n[3/3] Last-Week Baseline (Persistence)")

    # For each HSA, predict using previous week's value
    # This requires chronological data
    val_pred_list = []
    test_pred_list = []

    for hsa in val_df['hsa_id'].unique():
        # Get train data for this HSA
        hsa_train = train_df[train_df['hsa_id'] == hsa].sort_values('week_number')

        # Validation predictions
        hsa_val = val_df[val_df['hsa_id'] == hsa].sort_values('week_number')
        if len(hsa_train) > 0:
            # Last week of training is the prediction for first week of validation
            last_train_value = hsa_train[TARGET_COL].iloc[-1]
            hsa_val_pred = [last_train_value]

            # Subsequent weeks use previous week from validation
            for i in range(1, len(hsa_val)):
                hsa_val_pred.append(hsa_val[TARGET_COL].iloc[i-1])

            val_pred_list.extend(hsa_val_pred)
        else:
            # If no training data for this HSA, use overall mean
            val_pred_list.extend([y_train.mean()] * len(hsa_val))

        # Test predictions (use last week of validation)
        hsa_test = test_df[test_df['hsa_id'] == hsa].sort_values('week_number')
        if len(hsa_val) > 0:
            last_val_value = hsa_val[TARGET_COL].iloc[-1]
            hsa_test_pred = [last_val_value]

            for i in range(1, len(hsa_test)):
                hsa_test_pred.append(hsa_test[TARGET_COL].iloc[i-1])

            test_pred_list.extend(hsa_test_pred)
        else:
            test_pred_list.extend([y_train.mean()] * len(hsa_test))

    lastweek_pred_val = np.array(val_pred_list)
    lastweek_pred_test = np.array(test_pred_list)

    metrics_val = compute_metrics(y_val, lastweek_pred_val, "Last-Week Baseline")
    print_metrics(metrics_val, "Validation")

    metrics_test = compute_metrics(y_test, lastweek_pred_test, "Last-Week Baseline")

    results['lastweek_baseline'] = {
        'val_metrics': metrics_val,
        'test_metrics': metrics_test,
        'val_predictions': lastweek_pred_val,
        'test_predictions': lastweek_pred_test
    }

    return results


# ============================================================================
# LINEAR MODELS
# ============================================================================

def train_linear_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """Train regularized linear regression models"""

    print("\n" + "="*80)
    print("LINEAR MODELS")
    print("="*80)

    results = {}

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)

    # -------------------------------------------------------------------------
    # Model 1: Ridge Regression
    # -------------------------------------------------------------------------
    print("\n[1/3] Ridge Regression")

    # Try different alpha values
    alphas = [0.1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_score = -np.inf

    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        model.fit(X_train_scaled, y_train)
        score = model.score(X_val_scaled, y_val)
        if score > best_score:
            best_score = score
            best_alpha = alpha

    print(f"  Best alpha: {best_alpha} (R² = {best_score:.3f})")

    # Train final model
    ridge = Ridge(alpha=best_alpha, random_state=RANDOM_SEED)
    ridge.fit(X_train_scaled, y_train)

    ridge_pred_val = ridge.predict(X_val_scaled)
    ridge_pred_test = ridge.predict(X_test_scaled)

    metrics_val = compute_metrics(y_val, ridge_pred_val, "Ridge")
    print_metrics(metrics_val, "Validation")

    metrics_test = compute_metrics(y_test, ridge_pred_test, "Ridge")

    results['ridge'] = {
        'model': ridge,
        'val_metrics': metrics_val,
        'test_metrics': metrics_test,
        'val_predictions': ridge_pred_val,
        'test_predictions': ridge_pred_test,
        'feature_importance': np.abs(ridge.coef_),
        'feature_names': feature_names
    }

    # -------------------------------------------------------------------------
    # Model 2: Lasso Regression
    # -------------------------------------------------------------------------
    print("\n[2/3] Lasso Regression")

    alphas = [0.01, 0.1, 1.0, 10.0]
    best_alpha = None
    best_score = -np.inf

    for alpha in alphas:
        model = Lasso(alpha=alpha, random_state=RANDOM_SEED, max_iter=5000)
        model.fit(X_train_scaled, y_train)
        score = model.score(X_val_scaled, y_val)
        if score > best_score:
            best_score = score
            best_alpha = alpha

    print(f"  Best alpha: {best_alpha} (R² = {best_score:.3f})")

    lasso = Lasso(alpha=best_alpha, random_state=RANDOM_SEED, max_iter=5000)
    lasso.fit(X_train_scaled, y_train)

    lasso_pred_val = lasso.predict(X_val_scaled)
    lasso_pred_test = lasso.predict(X_test_scaled)

    metrics_val = compute_metrics(y_val, lasso_pred_val, "Lasso")
    print_metrics(metrics_val, "Validation")

    metrics_test = compute_metrics(y_test, lasso_pred_test, "Lasso")

    # Count non-zero coefficients
    n_nonzero = np.sum(lasso.coef_ != 0)
    print(f"  Features selected: {n_nonzero}/{len(feature_names)}")

    results['lasso'] = {
        'model': lasso,
        'val_metrics': metrics_val,
        'test_metrics': metrics_test,
        'val_predictions': lasso_pred_val,
        'test_predictions': lasso_pred_test,
        'feature_importance': np.abs(lasso.coef_),
        'feature_names': feature_names
    }

    # -------------------------------------------------------------------------
    # Model 3: ElasticNet
    # -------------------------------------------------------------------------
    print("\n[3/3] ElasticNet")

    alphas = [0.01, 0.1, 1.0]
    l1_ratios = [0.3, 0.5, 0.7]
    best_alpha = None
    best_l1 = None
    best_score = -np.inf

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                             random_state=RANDOM_SEED, max_iter=5000)
            model.fit(X_train_scaled, y_train)
            score = model.score(X_val_scaled, y_val)
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_l1 = l1_ratio

    print(f"  Best alpha: {best_alpha}, l1_ratio: {best_l1} (R² = {best_score:.3f})")

    elasticnet = ElasticNet(alpha=best_alpha, l1_ratio=best_l1,
                          random_state=RANDOM_SEED, max_iter=5000)
    elasticnet.fit(X_train_scaled, y_train)

    en_pred_val = elasticnet.predict(X_val_scaled)
    en_pred_test = elasticnet.predict(X_test_scaled)

    metrics_val = compute_metrics(y_val, en_pred_val, "ElasticNet")
    print_metrics(metrics_val, "Validation")

    metrics_test = compute_metrics(y_test, en_pred_test, "ElasticNet")

    results['elasticnet'] = {
        'model': elasticnet,
        'val_metrics': metrics_val,
        'test_metrics': metrics_test,
        'val_predictions': en_pred_val,
        'test_predictions': en_pred_test,
        'feature_importance': np.abs(elasticnet.coef_),
        'feature_names': feature_names
    }

    return results


# ============================================================================
# GLM MODELS
# ============================================================================

def train_glm_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """Train Generalized Linear Models for count data"""

    print("\n" + "="*80)
    print("GENERALIZED LINEAR MODELS (GLM)")
    print("="*80)

    results = {}

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)

    # -------------------------------------------------------------------------
    # Model 1: Poisson Regression (sklearn)
    # -------------------------------------------------------------------------
    print("\n[1/2] Poisson Regression (sklearn)")

    poisson_sk = PoissonRegressor(alpha=1.0, max_iter=1000)
    poisson_sk.fit(X_train_scaled, y_train)

    poisson_pred_val = poisson_sk.predict(X_val_scaled)
    poisson_pred_test = poisson_sk.predict(X_test_scaled)

    metrics_val = compute_metrics(y_val, poisson_pred_val, "Poisson (sklearn)")
    print_metrics(metrics_val, "Validation")

    metrics_test = compute_metrics(y_test, poisson_pred_test, "Poisson (sklearn)")

    results['poisson_sklearn'] = {
        'model': poisson_sk,
        'val_metrics': metrics_val,
        'test_metrics': metrics_test,
        'val_predictions': poisson_pred_val,
        'test_predictions': poisson_pred_test,
        'feature_importance': np.abs(poisson_sk.coef_),
        'feature_names': feature_names
    }

    # -------------------------------------------------------------------------
    # Model 2: Negative Binomial (statsmodels)
    # -------------------------------------------------------------------------
    print("\n[2/2] Negative Binomial Regression (statsmodels)")

    try:
        # Add intercept
        X_train_sm = sm.add_constant(X_train_scaled, has_constant='add')
        X_val_sm = sm.add_constant(X_val_scaled, has_constant='add')
        X_test_sm = sm.add_constant(X_test_scaled, has_constant='add')

        # Fit NB model
        nb_model = sm.GLM(y_train, X_train_sm, family=sm.families.NegativeBinomial())
        nb_results = nb_model.fit()

        nb_pred_val = nb_results.predict(X_val_sm)
        nb_pred_test = nb_results.predict(X_test_sm)

        metrics_val = compute_metrics(y_val, nb_pred_val, "Negative Binomial")
        print_metrics(metrics_val, "Validation")

        metrics_test = compute_metrics(y_test, nb_pred_test, "Negative Binomial")

        results['negative_binomial'] = {
            'model': nb_results,
            'val_metrics': metrics_val,
            'test_metrics': metrics_test,
            'val_predictions': nb_pred_val,
            'test_predictions': nb_pred_test,
            'feature_importance': np.abs(nb_results.params[1:]),  # Exclude intercept
            'feature_names': feature_names
        }

    except Exception as e:
        print(f"  [X] Negative Binomial failed: {e}")
        results['negative_binomial'] = None

    return results


# ============================================================================
# TREE-BASED MODELS
# ============================================================================

def train_tree_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """Train tree-based ensemble models"""

    print("\n" + "="*80)
    print("TREE-BASED MODELS")
    print("="*80)

    results = {}

    # -------------------------------------------------------------------------
    # Model 1: Random Forest
    # -------------------------------------------------------------------------
    print("\n[1/4] Random Forest")

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    rf_pred_val = rf.predict(X_val)
    rf_pred_test = rf.predict(X_test)

    metrics_val = compute_metrics(y_val, rf_pred_val, "Random Forest")
    print_metrics(metrics_val, "Validation")

    metrics_test = compute_metrics(y_test, rf_pred_test, "Random Forest")

    results['random_forest'] = {
        'model': rf,
        'val_metrics': metrics_val,
        'test_metrics': metrics_test,
        'val_predictions': rf_pred_val,
        'test_predictions': rf_pred_test,
        'feature_importance': rf.feature_importances_,
        'feature_names': feature_names
    }

    # -------------------------------------------------------------------------
    # Model 2: Gradient Boosting
    # -------------------------------------------------------------------------
    print("\n[2/4] Gradient Boosting")

    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_SEED
    )
    gb.fit(X_train, y_train)

    gb_pred_val = gb.predict(X_val)
    gb_pred_test = gb.predict(X_test)

    metrics_val = compute_metrics(y_val, gb_pred_val, "Gradient Boosting")
    print_metrics(metrics_val, "Validation")

    metrics_test = compute_metrics(y_test, gb_pred_test, "Gradient Boosting")

    results['gradient_boosting'] = {
        'model': gb,
        'val_metrics': metrics_val,
        'test_metrics': metrics_test,
        'val_predictions': gb_pred_val,
        'test_predictions': gb_pred_test,
        'feature_importance': gb.feature_importances_,
        'feature_names': feature_names
    }

    # -------------------------------------------------------------------------
    # Model 3: XGBoost (if available)
    # -------------------------------------------------------------------------
    if XGBOOST_AVAILABLE:
        print("\n[3/4] XGBoost")

        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)

        xgb_pred_val = xgb_model.predict(X_val)
        xgb_pred_test = xgb_model.predict(X_test)

        metrics_val = compute_metrics(y_val, xgb_pred_val, "XGBoost")
        print_metrics(metrics_val, "Validation")

        metrics_test = compute_metrics(y_test, xgb_pred_test, "XGBoost")

        results['xgboost'] = {
            'model': xgb_model,
            'val_metrics': metrics_val,
            'test_metrics': metrics_test,
            'val_predictions': xgb_pred_val,
            'test_predictions': xgb_pred_test,
            'feature_importance': xgb_model.feature_importances_,
            'feature_names': feature_names
        }
    else:
        print("\n[3/4] XGBoost - SKIPPED (not installed)")

    # -------------------------------------------------------------------------
    # Model 4: LightGBM (if available)
    # -------------------------------------------------------------------------
    if LIGHTGBM_AVAILABLE:
        print("\n[4/4] LightGBM")

        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)

        lgb_pred_val = lgb_model.predict(X_val)
        lgb_pred_test = lgb_model.predict(X_test)

        metrics_val = compute_metrics(y_val, lgb_pred_val, "LightGBM")
        print_metrics(metrics_val, "Validation")

        metrics_test = compute_metrics(y_test, lgb_pred_test, "LightGBM")

        results['lightgbm'] = {
            'model': lgb_model,
            'val_metrics': metrics_val,
            'test_metrics': metrics_test,
            'val_predictions': lgb_pred_val,
            'test_predictions': lgb_pred_test,
            'feature_importance': lgb_model.feature_importances_,
            'feature_names': feature_names
        }
    else:
        print("\n[4/4] LightGBM - SKIPPED (not installed)")

    return results


# ============================================================================
# ENSEMBLE MODELS
# ============================================================================

def train_ensemble_models(linear_results, tree_results,
                         X_train, y_train, X_val, y_val, X_test, y_test):
    """Create ensemble models from best individual models"""

    print("\n" + "="*80)
    print("ENSEMBLE MODELS")
    print("="*80)

    results = {}

    # -------------------------------------------------------------------------
    # Simple Voting Ensemble
    # -------------------------------------------------------------------------
    print("\n[1/1] Voting Ensemble (Average Predictions)")

    # Collect predictions from all models
    val_predictions = []
    test_predictions = []
    model_names = []

    # Add linear models
    for name, res in linear_results.items():
        if res is not None:
            val_predictions.append(res['val_predictions'])
            test_predictions.append(res['test_predictions'])
            model_names.append(name)

    # Add tree models
    for name, res in tree_results.items():
        if res is not None:
            val_predictions.append(res['val_predictions'])
            test_predictions.append(res['test_predictions'])
            model_names.append(name)

    # Average predictions
    ensemble_pred_val = np.mean(val_predictions, axis=0)
    ensemble_pred_test = np.mean(test_predictions, axis=0)

    metrics_val = compute_metrics(y_val, ensemble_pred_val, "Voting Ensemble")
    print_metrics(metrics_val, "Validation")
    print(f"  Models included: {', '.join(model_names)}")

    metrics_test = compute_metrics(y_test, ensemble_pred_test, "Voting Ensemble")

    results['voting_ensemble'] = {
        'val_metrics': metrics_val,
        'test_metrics': metrics_test,
        'val_predictions': ensemble_pred_val,
        'test_predictions': ensemble_pred_test,
        'model_names': model_names
    }

    return results


# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def extract_feature_importance(all_results, top_n=20):
    """Extract and rank feature importance across all models"""

    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    importance_dict = {}

    # Collect feature importances from all models
    for model_category, models in all_results.items():
        if model_category == 'baselines' or model_category == 'ensemble':
            continue

        for model_name, res in models.items():
            if res is None or 'feature_importance' not in res:
                continue

            importances = res['feature_importance']
            feature_names = res['feature_names']

            for feat, imp in zip(feature_names, importances):
                if feat not in importance_dict:
                    importance_dict[feat] = []
                importance_dict[feat].append((model_name, imp))

    # Aggregate importance scores
    feature_rankings = []
    for feat, scores in importance_dict.items():
        avg_importance = np.mean([score for _, score in scores])
        feature_rankings.append({
            'feature': feat,
            'avg_importance': avg_importance,
            'n_models': len(scores),
            'scores': scores
        })

    # Sort by average importance
    feature_rankings.sort(key=lambda x: x['avg_importance'], reverse=True)

    # Print top features
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Feature':<40} {'Avg Importance':<15} {'Models':<10}")
    print("-" * 80)

    for i, feat_info in enumerate(feature_rankings[:top_n], 1):
        print(f"{i:<6} {feat_info['feature']:<40} {feat_info['avg_importance']:<15.4f} {feat_info['n_models']:<10}")

    # Save to CSV
    rankings_df = pd.DataFrame(feature_rankings)
    rankings_path = OUTPUT_DIR / f"{NETWORK}_{HSA_MODE}_feature_importance_rankings.csv"
    rankings_df.to_csv(rankings_path, index=False)
    print(f"\n[OK] Feature rankings saved to {rankings_path}")

    return feature_rankings


# ============================================================================
# RESULTS SUMMARY
# ============================================================================

def save_results_summary(all_results, output_dir):
    """Save comprehensive results summary"""

    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Collect all metrics
    summary_rows = []

    for category, models in all_results.items():
        for model_name, res in models.items():
            if res is None:
                continue

            # Validation metrics
            val_metrics = res.get('val_metrics', {})
            test_metrics = res.get('test_metrics', {})

            row = {
                'category': category,
                'model': model_name,
                'val_rmse': val_metrics.get('rmse', np.nan),
                'val_mae': val_metrics.get('mae', np.nan),
                'val_r2': val_metrics.get('r2', np.nan),
                'val_median_ae': val_metrics.get('median_ae', np.nan),
                'val_mape': val_metrics.get('mape', np.nan),
                'val_poisson_dev': val_metrics.get('poisson_deviance', np.nan),
                'test_rmse': test_metrics.get('rmse', np.nan),
                'test_mae': test_metrics.get('mae', np.nan),
                'test_r2': test_metrics.get('r2', np.nan),
                'test_median_ae': test_metrics.get('median_ae', np.nan),
                'test_mape': test_metrics.get('mape', np.nan),
                'test_poisson_dev': test_metrics.get('poisson_deviance', np.nan)
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Sort by validation R² (default view)
    summary_df = summary_df.sort_values('val_r2', ascending=False)

    # Save to CSV
    summary_path = output_dir / f"{NETWORK}_{HSA_MODE}_model_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[OK] Model comparison saved to {summary_path}")

    # Print top 5 models
    print("\nTop 5 Models by Validation R²:")
    print("-" * 80)
    print(summary_df[['model', 'val_r2', 'val_rmse', 'val_mae']].head(5).to_string(index=False))

    # Also show top 5 by test R² when available
    test_sorted = summary_df.dropna(subset=['test_r2']).sort_values('test_r2', ascending=False)
    if not test_sorted.empty:
        print("\nTop 5 Models by Test R²:")
        print("-" * 80)
        print(test_sorted[['model', 'test_r2', 'test_rmse', 'test_mae']].head(5).to_string(index=False))

    return summary_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("="*80)
    print("ML MODEL TRAINING & EVALUATION")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random Seed: {RANDOM_SEED}")
    print("="*80)

    # Load data
    train_df, val_df, test_df = load_datasets()

    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_features(
        train_df, val_df, test_df,
        include_temporal=True,
        include_hsa_dummies=False
    )
    print(f"  Features: {len(feature_names)}")
    print(f"  Train samples: {len(y_train)}")
    print(f"  Val samples: {len(y_val)}")
    print(f"  Test samples: {len(y_test)}")

    # Store all results
    all_results = {}

    # Train baseline models
    baseline_results = train_baseline_models(y_train, y_val, y_test, train_df, val_df, test_df)
    all_results['baselines'] = baseline_results

    # Train linear models
    linear_results = train_linear_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
    all_results['linear'] = linear_results

    # Train GLM models
    glm_results = train_glm_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
    all_results['glm'] = glm_results

    # Train tree-based models
    tree_results = train_tree_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
    all_results['tree'] = tree_results

    # Train ensemble models
    ensemble_results = train_ensemble_models(linear_results, tree_results,
                                            X_train, y_train, X_val, y_val, X_test, y_test)
    all_results['ensemble'] = ensemble_results

    # Feature importance analysis
    feature_rankings = extract_feature_importance(all_results, top_n=20)

    # Save results summary
    summary_df = save_results_summary(all_results, OUTPUT_DIR)

    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest Model (by Validation R²):")
    best_model = summary_df.iloc[0]
    print(f"  Model: {best_model['model']}")
    print(f"  Validation R²: {best_model['val_r2']:.3f}")
    print(f"  Validation RMSE: {best_model['val_rmse']:.3f}")
    print(f"  Validation MAE: {best_model['val_mae']:.3f}")

    test_sorted = summary_df.dropna(subset=['test_r2']).sort_values('test_r2', ascending=False)
    if not test_sorted.empty:
        best_test = test_sorted.iloc[0]
        print(f"\nBest Model (by Test R²):")
        print(f"  Model: {best_test['model']}")
        print(f"  Test R²: {best_test['test_r2']:.3f}")
        print(f"  Test RMSE: {best_test['test_rmse']:.3f}")
        print(f"  Test MAE: {best_test['test_mae']:.3f}")

    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print("  - model_comparison_summary.csv")
    print("  - feature_importance_rankings.csv")


if __name__ == "__main__":
    main()
