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

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("out/modeling")
OUTPUT_DIR = Path("out/modeling/results_improved")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print("IMPROVED ML MODELING - AUTOREGRESSIVE + FEATURE SELECTION")
print("="*80)

train = pd.read_csv(DATA_DIR / "modeling_dataset_train.csv")
val = pd.read_csv(DATA_DIR / "modeling_dataset_val.csv")
test = pd.read_csv(DATA_DIR / "modeling_dataset_test.csv")

print(f"\nOriginal Data:")
print(f"  Train: {len(train)} samples")
print(f"  Val:   {len(val)} samples")
print(f"  Test:  {len(test)} samples")

# ============================================================================
# ADD AUTOREGRESSIVE FEATURES
# ============================================================================

print("\n[STEP 1] Creating autoregressive features...")

def add_lag_features(df, target_col='diarrheal_count_adjusted', lags=[1, 2]):
    """Add lagged disease counts as features"""
    df = df.sort_values(['hsa_id', 'week_start']).copy()

    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df.groupby('hsa_id')[target_col].shift(lag)

    return df

# Add lags to all datasets
train = add_lag_features(train)
val = add_lag_features(val)
test = add_lag_features(test)

# Drop rows with missing lag features
train_with_lags = train.dropna(subset=['diarrheal_count_adjusted_lag1', 'diarrheal_count_adjusted_lag2'])
val_with_lags = val.dropna(subset=['diarrheal_count_adjusted_lag1', 'diarrheal_count_adjusted_lag2'])
test_with_lags = test.dropna(subset=['diarrheal_count_adjusted_lag1', 'diarrheal_count_adjusted_lag2'])

print(f"  After adding lags (some rows dropped):")
print(f"    Train: {len(train_with_lags)} samples")
print(f"    Val:   {len(val_with_lags)} samples")
print(f"    Test:  {len(test_with_lags)} samples")

# ============================================================================
# FEATURE SELECTION - TOP CLIMATE FEATURES
# ============================================================================

print("\n[STEP 2] Selecting top climate features...")

# Identify feature columns (exclude categorical and metadata)
exclude_cols = ['hsa_id', 'week_start', 'week_number', 'diarrheal_count_adjusted',
                'diarrheal_count_adjusted_lag1', 'diarrheal_count_adjusted_lag2',
                'season', 'week_of_year', 'month', 'quarter']  # Exclude temporal for now

climate_features = [col for col in train_with_lags.columns
                   if col not in exclude_cols and train_with_lags[col].dtype in ['float64', 'int64']]

# Compute correlations with target
correlations = train_with_lags[climate_features].corrwith(
    train_with_lags['diarrheal_count_adjusted']
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
    'AR_only': ['diarrheal_count_adjusted_lag1', 'diarrheal_count_adjusted_lag2'],
    'AR_temporal': ['diarrheal_count_adjusted_lag1', 'diarrheal_count_adjusted_lag2',
                    'week_of_year', 'month', 'season'],
    'AR_top5_climate': ['diarrheal_count_adjusted_lag1', 'diarrheal_count_adjusted_lag2'] + top_climate_features[:5],
    'AR_top10_climate': ['diarrheal_count_adjusted_lag1', 'diarrheal_count_adjusted_lag2'] + top_climate_features[:10],
    'AR_temporal_top5': ['diarrheal_count_adjusted_lag1', 'diarrheal_count_adjusted_lag2',
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

    y_train = train_df['diarrheal_count_adjusted'].values
    y_val = val_df['diarrheal_count_adjusted'].values
    y_test = test_df['diarrheal_count_adjusted'].values

    return X_train, y_train, X_val, y_val, X_test, y_test

def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics"""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

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
    ridge_pred = ridge.predict(X_val_scaled)
    ridge_metrics = compute_metrics(y_val, ridge_pred)

    print(f"    Best alpha: {best_alpha}, Val R²: {ridge_metrics['r2']:.4f}")

    all_results.append({
        'feature_set': feature_set_name,
        'n_features': X_train_scaled.shape[1],
        'model': 'Ridge',
        'alpha': best_alpha,
        **ridge_metrics
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
    lasso_pred = lasso.predict(X_val_scaled)
    lasso_metrics = compute_metrics(y_val, lasso_pred)

    print(f"    Best alpha: {best_alpha}, Val R²: {lasso_metrics['r2']:.4f}")

    all_results.append({
        'feature_set': feature_set_name,
        'n_features': X_train_scaled.shape[1],
        'model': 'Lasso',
        'alpha': best_alpha,
        **lasso_metrics
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
    rf_pred = rf.predict(X_val)
    rf_metrics = compute_metrics(y_val, rf_pred)

    print(f"    Val R²: {rf_metrics['r2']:.4f}")

    all_results.append({
        'feature_set': feature_set_name,
        'n_features': X_train_scaled.shape[1],
        'model': 'RandomForest',
        'alpha': None,
        **rf_metrics
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
    gb_pred = gb.predict(X_val)
    gb_metrics = compute_metrics(y_val, gb_pred)

    print(f"    Val R²: {gb_metrics['r2']:.4f}")

    all_results.append({
        'feature_set': feature_set_name,
        'n_features': X_train_scaled.shape[1],
        'model': 'GradientBoosting',
        'alpha': None,
        **gb_metrics
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
        xgb_pred = xgb_model.predict(X_val)
        xgb_metrics = compute_metrics(y_val, xgb_pred)

        print(f"    Val R²: {xgb_metrics['r2']:.4f}")

        all_results.append({
            'feature_set': feature_set_name,
            'n_features': X_train_scaled.shape[1],
            'model': 'XGBoost',
            'alpha': None,
            **xgb_metrics
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

# Save results
results_path = OUTPUT_DIR / "improved_model_comparison.csv"
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

# Compare to baseline
baseline_r2 = 0.478
baseline_rmse = 12.118
improvement_r2 = best_model['r2'] - baseline_r2
improvement_rmse = baseline_rmse - best_model['rmse']

print(f"\n  Improvement over Last-Week Baseline:")
print(f"    ΔR²: {improvement_r2:+.4f} ({improvement_r2/baseline_r2*100:+.1f}%)")
print(f"    ΔRMSE: {improvement_rmse:+.3f} ({improvement_rmse/baseline_rmse*100:+.1f}%)")

print("\n" + "="*80)
