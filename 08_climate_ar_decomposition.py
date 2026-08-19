#!/usr/bin/env python3
"""
Task 1.1: Variance Decomposition - Disentangle AR vs Climate Contributions
==========================================================================

This script addresses Issue #1: Circular reasoning in feature importance

Objectives:
- Test if climate predicts disease WITHOUT autoregressive terms
- Quantify shared variance between AR structure and climate
- Provide evidence for whether AR "masks" climate effects

Author: HSA Research Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import argparse
import os
from pathlib import Path
import json

warnings.filterwarnings('ignore')
DEFAULT_PIPELINE_OUT_DIR = os.environ.get("HSA_OUT_DIR", os.environ.get("PIPELINE_OUT_DIR", "out"))
OUTPUT_FILE_PREFIX = ""
TEXT_RESULTS_DIR = None


def out_name(filename: str) -> str:
    return f"{OUTPUT_FILE_PREFIX}_{filename}" if OUTPUT_FILE_PREFIX else filename


def md_path(filename: str) -> Path:
    return TEXT_RESULTS_DIR / out_name(filename) if TEXT_RESULTS_DIR else Path(filename)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Feature groups for decomposition
FEATURE_GROUPS = {
    'climate_precip': [
        'P_d-1', 'P_d-2', 'P_d-3', 'P_d-5', 'P_d-7', 'P_d-10', 'P_d-14',
        'P_total_week', 'P_mean_week', 'P_max_day_week', 'heavy_days_week'
    ],
    'climate_temp': [
        'T_mean_week_C', 'T_max_week_C', 'T_min_week_C', 'DTR_week_C',
        'T_mean_d-1_C', 'T_mean_d-2_C', 'T_mean_d-3_C', 'T_mean_d-5_C', 'T_mean_d-7_C'
    ],
    'climate_humidity': [
        'Td_week_C', 'Td_d-1_C', 'Td_d-2_C', 'Td_d-3_C', 'Td_d-7_C'
    ],
    'climate_heat': [
        'hours_above_30C_week', 'hours_above_35C_week', 'heat_index_week_C'
    ],
    'climate_soil': [
        'SM1_week', 'SM2_week', 'SM1_d-1', 'SM1_d-7'
    ],
    'climate_evap': [
        'E_week_mm_per_day', 'E_d-1_mm_per_day', 'E_d-7_mm_per_day', 'water_deficit_mm_week'
    ],
}

# Combine all climate features
ALL_CLIMATE_FEATURES = []
for group in FEATURE_GROUPS.values():
    ALL_CLIMATE_FEATURES.extend(group)


def load_data(input_csv, target_col):
    """Load and prepare modeling dataset."""
    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)

    # Create AR features
    df = df.sort_values(['hsa_id', 'week_number'])
    for lag in [1, 2, 3, 4]:
        df[f'ar_lag{lag}'] = df.groupby('hsa_id')[target_col].shift(lag)

    # Drop rows with NaN from AR lags
    df = df.dropna(subset=[f'ar_lag{i}' for i in [1, 2, 3, 4]])

    # Create seasonal features (Fourier terms)
    df['sin_annual'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['cos_annual'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['sin_semiannual'] = np.sin(4 * np.pi * df['week_of_year'] / 52)
    df['cos_semiannual'] = np.cos(4 * np.pi * df['week_of_year'] / 52)

    print(f"  Loaded {len(df)} rows after AR lag creation")
    print(f"  HSAs: {df['hsa_id'].nunique()}")
    print(f"  Weeks: {df['week_number'].nunique()}")

    return df


def temporal_train_test_split(df, train_frac=0.75, val_frac=0.125):
    """Split data temporally to avoid leakage."""
    weeks = sorted(df['week_number'].unique())
    n_weeks = len(weeks)

    train_end = int(n_weeks * train_frac)
    val_end = int(n_weeks * (train_frac + val_frac))

    train_weeks = weeks[:train_end]
    val_weeks = weeks[train_end:val_end]
    test_weeks = weeks[val_end:]

    train_df = df[df['week_number'].isin(train_weeks)]
    val_df = df[df['week_number'].isin(val_weeks)]
    test_df = df[df['week_number'].isin(test_weeks)]

    print(f"  Train: {len(train_df)} rows (weeks {min(train_weeks)}-{max(train_weeks)})")
    print(f"  Val: {len(val_df)} rows (weeks {min(val_weeks)}-{max(val_weeks)})")
    print(f"  Test: {len(test_df)} rows (weeks {min(test_weeks)}-{max(test_weeks)})")

    return train_df, val_df, test_df


def get_available_features(df, feature_list):
    """Get features that exist in the dataframe."""
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"  Warning: Missing features: {missing[:5]}...")
    return available


def fit_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='elasticnet'):
    """Fit model and return metrics."""
    if model_type == 'elasticnet':
        model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.01, max_iter=5000, random_state=42)
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    elif model_type == 'gbm':
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return {
        'model': model,
        'scaler': scaler,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'overfit_gap': train_r2 - test_r2,
        'y_pred_test': y_pred_test
    }


def decompose_variance_contributions(df, target_col, output_dir):
    """
    Fit multiple model specifications to decompose variance contributions.

    Model specifications:
    A. Climate-only (NO AR terms)
    B. AR-only (NO climate)
    C. Seasonal-only
    D. AR + Seasonal
    E. Climate + Seasonal
    F. AR + Climate
    G. AR + Climate + Seasonal (full model)
    """
    print("\n" + "="*80)
    print("VARIANCE DECOMPOSITION ANALYSIS")
    print("="*80)

    # Split data
    train_df, val_df, test_df = temporal_train_test_split(df)

    # Define feature sets
    ar_features = ['ar_lag1', 'ar_lag2']
    seasonal_features = ['sin_annual', 'cos_annual', 'sin_semiannual', 'cos_semiannual']

    # Get available climate features
    climate_features = get_available_features(df, ALL_CLIMATE_FEATURES)
    # Use top climate features to avoid overfitting
    climate_features_reduced = [
        'T_mean_week_C', 'T_max_week_C', 'P_total_week', 'Td_week_C',
        'hours_above_30C_week', 'E_week_mm_per_day', 'SM1_week', 'water_deficit_mm_week'
    ]
    climate_features_reduced = get_available_features(df, climate_features_reduced)

    print(f"\nFeature sets:")
    print(f"  AR features: {ar_features}")
    print(f"  Seasonal features: {seasonal_features}")
    print(f"  Climate features (reduced): {len(climate_features_reduced)}")

    # Model specifications
    model_specs = {
        'A_climate_only': climate_features_reduced,
        'B_ar_only': ar_features,
        'C_seasonal_only': seasonal_features,
        'D_ar_seasonal': ar_features + seasonal_features,
        'E_climate_seasonal': climate_features_reduced + seasonal_features,
        'F_ar_climate': ar_features + climate_features_reduced,
        'G_full_model': ar_features + seasonal_features + climate_features_reduced,
    }

    # Prepare target
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Fit models
    results = []
    model_objects = {}

    for model_name, features in model_specs.items():
        print(f"\n--- {model_name} ({len(features)} features) ---")

        X_train = train_df[features].values
        X_test = test_df[features].values

        # Try multiple model types
        for model_type in ['elasticnet', 'ridge', 'rf']:
            result = fit_and_evaluate_model(X_train, y_train, X_test, y_test, model_type)

            results.append({
                'specification': model_name,
                'model_type': model_type,
                'n_features': len(features),
                'features': ', '.join(features[:5]) + ('...' if len(features) > 5 else ''),
                'train_r2': result['train_r2'],
                'test_r2': result['test_r2'],
                'test_mae': result['test_mae'],
                'test_rmse': result['test_rmse'],
                'overfit_gap': result['overfit_gap']
            })

            if model_type == 'elasticnet':
                model_objects[model_name] = result
                print(f"  {model_type}: Train R²={result['train_r2']:.4f}, Test R²={result['test_r2']:.4f}")

    results_df = pd.DataFrame(results)

    # Calculate variance decomposition
    print("\n" + "="*80)
    print("VARIANCE DECOMPOSITION SUMMARY (ElasticNet)")
    print("="*80)

    decomposition = calculate_variance_decomposition(results_df, model_objects)

    # Save results
    results_df.to_csv(output_dir / out_name('variance_decomposition_results.csv'), index=False)

    with open(output_dir / out_name('variance_decomposition_summary.json'), 'w') as f:
        json.dump(decomposition, f, indent=2)

    # Create plots
    plot_variance_decomposition(results_df, decomposition, output_dir)

    return results_df, decomposition


def calculate_variance_decomposition(results_df, model_objects):
    """Calculate variance contributions and overlaps."""

    # Get ElasticNet results
    en_results = results_df[results_df['model_type'] == 'elasticnet'].set_index('specification')

    # Extract R² values (convert to native Python float)
    r2_climate = float(en_results.loc['A_climate_only', 'test_r2'])
    r2_ar = float(en_results.loc['B_ar_only', 'test_r2'])
    r2_seasonal = float(en_results.loc['C_seasonal_only', 'test_r2'])
    r2_ar_seasonal = float(en_results.loc['D_ar_seasonal', 'test_r2'])
    r2_climate_seasonal = float(en_results.loc['E_climate_seasonal', 'test_r2'])
    r2_ar_climate = float(en_results.loc['F_ar_climate', 'test_r2'])
    r2_full = float(en_results.loc['G_full_model', 'test_r2'])

    # Calculate incremental contributions
    decomposition = {
        'individual_contributions': {
            'climate_only': r2_climate,
            'ar_only': r2_ar,
            'seasonal_only': r2_seasonal,
        },
        'combined_contributions': {
            'ar_seasonal': r2_ar_seasonal,
            'climate_seasonal': r2_climate_seasonal,
            'ar_climate': r2_ar_climate,
            'full_model': r2_full,
        },
        'incremental_contributions': {
            'climate_over_ar': r2_ar_climate - r2_ar,
            'climate_over_ar_seasonal': r2_full - r2_ar_seasonal,
            'ar_over_climate': r2_ar_climate - r2_climate,
            'ar_over_climate_seasonal': r2_full - r2_climate_seasonal,
            'seasonal_over_ar': r2_ar_seasonal - r2_ar,
            'seasonal_over_climate': r2_climate_seasonal - r2_climate,
        },
        'shared_variance': {
            'ar_climate_overlap': r2_ar + r2_climate - r2_ar_climate,
            'ar_seasonal_overlap': r2_ar + r2_seasonal - r2_ar_seasonal,
            'climate_seasonal_overlap': r2_climate + r2_seasonal - r2_climate_seasonal,
        },
        'interpretation': {
            'ar_dominance_ratio': r2_ar / max(r2_full, 0.001),
            'climate_unique_contribution': r2_full - r2_ar_seasonal,
            'seasonal_absorbed_by_ar': bool(r2_ar_seasonal - r2_ar < 0.01),
        }
    }

    # Print summary
    print(f"\nIndividual Contributions (Test R²):")
    print(f"  Climate-only:     {r2_climate:.4f} ({r2_climate*100:.1f}%)")
    print(f"  AR-only:          {r2_ar:.4f} ({r2_ar*100:.1f}%)")
    print(f"  Seasonal-only:    {r2_seasonal:.4f} ({r2_seasonal*100:.1f}%)")

    print(f"\nCombined Models (Test R²):")
    print(f"  AR + Seasonal:        {r2_ar_seasonal:.4f}")
    print(f"  Climate + Seasonal:   {r2_climate_seasonal:.4f}")
    print(f"  AR + Climate:         {r2_ar_climate:.4f}")
    print(f"  Full (AR+Clim+Seas):  {r2_full:.4f}")

    print(f"\nIncremental Contributions:")
    print(f"  Climate beyond AR:            {decomposition['incremental_contributions']['climate_over_ar']:.4f}")
    print(f"  Climate beyond AR+Seasonal:   {decomposition['incremental_contributions']['climate_over_ar_seasonal']:.4f}")
    print(f"  AR beyond Climate:            {decomposition['incremental_contributions']['ar_over_climate']:.4f}")
    print(f"  Seasonal beyond AR:           {decomposition['incremental_contributions']['seasonal_over_ar']:.4f}")

    print(f"\nShared Variance (Overlap):")
    print(f"  AR-Climate overlap:    {decomposition['shared_variance']['ar_climate_overlap']:.4f}")
    print(f"  AR-Seasonal overlap:   {decomposition['shared_variance']['ar_seasonal_overlap']:.4f}")
    print(f"  Climate-Seasonal overlap: {decomposition['shared_variance']['climate_seasonal_overlap']:.4f}")

    return decomposition


def plot_variance_decomposition(results_df, decomposition, output_dir):
    """Create visualization of variance decomposition."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Model comparison bar chart
    ax1 = axes[0, 0]
    en_results = results_df[results_df['model_type'] == 'elasticnet'].copy()
    en_results = en_results.sort_values('test_r2', ascending=True)

    colors = ['#e74c3c' if 'climate' in s and 'ar' not in s else
              '#3498db' if 'ar' in s and 'climate' not in s else
              '#2ecc71' if 'seasonal' in s and 'ar' not in s and 'climate' not in s else
              '#9b59b6' for s in en_results['specification']]

    bars = ax1.barh(en_results['specification'], en_results['test_r2'], color=colors)
    ax1.set_xlabel('Test R²')
    ax1.set_title('Model Comparison: Variance Explained')
    ax1.axvline(x=0, color='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, en_results['test_r2']):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    # 2. Incremental contributions
    ax2 = axes[0, 1]
    incremental = decomposition['incremental_contributions']
    inc_df = pd.DataFrame({
        'Contribution': list(incremental.keys()),
        'R² Change': list(incremental.values())
    })
    inc_df = inc_df.sort_values('R² Change')

    colors2 = ['#e74c3c' if v < 0 else '#2ecc71' for v in inc_df['R² Change']]
    ax2.barh(inc_df['Contribution'], inc_df['R² Change'], color=colors2)
    ax2.set_xlabel('Incremental R² Change')
    ax2.set_title('Incremental Variance Contributions')
    ax2.axvline(x=0, color='black', linewidth=0.5)

    # 3. Stacked variance decomposition
    ax3 = axes[1, 0]

    # Calculate unique and shared contributions
    r2_full = decomposition['combined_contributions']['full_model']
    r2_ar = decomposition['individual_contributions']['ar_only']
    r2_climate_unique = decomposition['incremental_contributions']['climate_over_ar_seasonal']
    r2_seasonal_unique = decomposition['incremental_contributions']['seasonal_over_ar']

    # For stacked bar
    components = ['AR (unique)', 'Seasonal (add\'l)', 'Climate (add\'l)', 'Unexplained']
    values = [
        r2_ar,
        max(0, r2_seasonal_unique),
        max(0, r2_climate_unique),
        max(0, 1 - r2_full)
    ]
    colors3 = ['#3498db', '#f39c12', '#e74c3c', '#95a5a6']

    bottom = 0
    for comp, val, col in zip(components, values, colors3):
        ax3.bar(['Variance\nDecomposition'], [val], bottom=[bottom], label=comp, color=col)
        if val > 0.02:
            ax3.text(0, bottom + val/2, f'{val:.1%}', ha='center', va='center', fontsize=10, fontweight='bold')
        bottom += val

    ax3.set_ylabel('Proportion of Variance')
    ax3.set_title('Hierarchical Variance Decomposition')
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 1.1)

    # 4. Model type comparison
    ax4 = axes[1, 1]
    pivot_df = results_df.pivot(index='specification', columns='model_type', values='test_r2')
    pivot_df = pivot_df.reindex(['A_climate_only', 'B_ar_only', 'C_seasonal_only',
                                  'D_ar_seasonal', 'E_climate_seasonal', 'F_ar_climate', 'G_full_model'])

    x = np.arange(len(pivot_df))
    width = 0.25

    for i, model_type in enumerate(['elasticnet', 'ridge', 'rf']):
        if model_type in pivot_df.columns:
            ax4.bar(x + i*width, pivot_df[model_type], width, label=model_type)

    ax4.set_xlabel('Model Specification')
    ax4.set_ylabel('Test R²')
    ax4.set_title('Model Type Comparison Across Specifications')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels([s.split('_')[0] for s in pivot_df.index], rotation=45, ha='right')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / out_name('variance_decomposition_figure.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / out_name('variance_decomposition_figure.pdf'), bbox_inches='tight')
    plt.close()

    print(f"\nFigures saved to {output_dir}")


def test_climate_on_residuals(df, target_col, output_dir):
    """
    Alternative approach: Test if climate predicts AR-model residuals.
    This isolates climate effect independent of persistence.
    """
    print("\n" + "="*80)
    print("RESIDUAL ANALYSIS: Climate Prediction of AR Residuals")
    print("="*80)

    # Split data
    train_df, val_df, test_df = temporal_train_test_split(df)

    # AR features
    ar_features = ['ar_lag1', 'ar_lag2']
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Fit AR-only model
    X_train_ar = train_df[ar_features].values
    X_test_ar = test_df[ar_features].values

    scaler_ar = StandardScaler()
    X_train_ar_scaled = scaler_ar.fit_transform(X_train_ar)
    X_test_ar_scaled = scaler_ar.transform(X_test_ar)

    ar_model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)
    ar_model.fit(X_train_ar_scaled, y_train)

    # Get residuals
    y_pred_ar_test = ar_model.predict(X_test_ar_scaled)
    residuals_test = y_test - y_pred_ar_test

    ar_r2 = r2_score(y_test, y_pred_ar_test)
    print(f"\nAR-only model Test R²: {ar_r2:.4f}")
    print(f"Residual variance: {np.var(residuals_test):.2f}")
    print(f"Residual std: {np.std(residuals_test):.2f}")

    # Get climate features
    climate_features = get_available_features(df, [
        'T_mean_week_C', 'T_max_week_C', 'P_total_week', 'Td_week_C',
        'hours_above_30C_week', 'E_week_mm_per_day', 'SM1_week', 'water_deficit_mm_week'
    ])

    # Test if climate predicts residuals
    X_train_clim = train_df[climate_features].values
    X_test_clim = test_df[climate_features].values

    scaler_clim = StandardScaler()
    X_train_clim_scaled = scaler_clim.fit_transform(X_train_clim)
    X_test_clim_scaled = scaler_clim.transform(X_test_clim)

    # Get training residuals for fitting
    y_pred_ar_train = ar_model.predict(X_train_ar_scaled)
    residuals_train = y_train - y_pred_ar_train

    # Fit climate model on residuals
    clim_on_resid_model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)
    clim_on_resid_model.fit(X_train_clim_scaled, residuals_train)

    # Predict residuals
    resid_pred = clim_on_resid_model.predict(X_test_clim_scaled)

    # Calculate R² of climate on residuals
    # Note: R² can be negative if model is worse than predicting mean
    ss_res = np.sum((residuals_test - resid_pred) ** 2)
    ss_tot = np.sum((residuals_test - np.mean(residuals_test)) ** 2)
    r2_climate_on_residuals = 1 - (ss_res / ss_tot)

    # Correlation test
    corr, p_value = stats.pearsonr(residuals_test, resid_pred)

    print(f"\nClimate prediction of AR residuals:")
    print(f"  R² (climate on residuals): {r2_climate_on_residuals:.4f}")
    print(f"  Correlation: {corr:.4f} (p={p_value:.4e})")

    # F-test for significance
    n = len(residuals_test)
    k_climate = len(climate_features)

    # Partial F-test
    r2_full = ar_r2 + max(0, r2_climate_on_residuals) * (1 - ar_r2)  # Approximate full R²
    f_stat = ((r2_full - ar_r2) / k_climate) / ((1 - r2_full) / (n - k_climate - 2 - 1))
    f_pvalue = 1 - stats.f.cdf(f_stat, k_climate, n - k_climate - 2 - 1)

    print(f"\nNested model F-test:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  P-value: {f_pvalue:.4e}")
    print(f"  Significant at α=0.05: {'Yes' if f_pvalue < 0.05 else 'No'}")

    # Save results
    residual_results = {
        'ar_only_r2': ar_r2,
        'climate_on_residuals_r2': r2_climate_on_residuals,
        'correlation': corr,
        'correlation_pvalue': p_value,
        'f_statistic': f_stat,
        'f_pvalue': f_pvalue,
        'n_observations': n,
        'n_climate_features': k_climate,
        'conclusion': 'Climate explains additional variance' if f_pvalue < 0.05 else 'Climate does not add significant variance'
    }

    with open(output_dir / out_name('residual_analysis_results.json'), 'w') as f:
        json.dump(residual_results, f, indent=2)

    # Plot residuals vs climate predictions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.scatter(residuals_test, resid_pred, alpha=0.5, s=20)
    ax1.plot([residuals_test.min(), residuals_test.max()],
             [residuals_test.min(), residuals_test.max()], 'r--', label='Perfect prediction')
    ax1.set_xlabel('Actual AR Residuals')
    ax1.set_ylabel('Climate-Predicted Residuals')
    ax1.set_title(f'Climate Prediction of AR Residuals\nR²={r2_climate_on_residuals:.4f}, r={corr:.4f}')
    ax1.legend()

    ax2 = axes[1]
    ax2.hist(residuals_test, bins=30, alpha=0.7, label='AR residuals', density=True)
    ax2.axvline(x=0, color='red', linestyle='--', label='Zero')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of AR Model Residuals')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / out_name('residual_analysis_figure.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return residual_results


def create_supplement_table(results_df, decomposition, output_dir):
    """Create formatted table for supplement."""

    # Filter to ElasticNet results
    en_results = results_df[results_df['model_type'] == 'elasticnet'].copy()

    # Rename for clarity
    en_results['Model Specification'] = en_results['specification'].map({
        'A_climate_only': 'Climate only',
        'B_ar_only': 'Autoregressive only',
        'C_seasonal_only': 'Seasonal only',
        'D_ar_seasonal': 'AR + Seasonal',
        'E_climate_seasonal': 'Climate + Seasonal',
        'F_ar_climate': 'AR + Climate',
        'G_full_model': 'AR + Climate + Seasonal'
    })

    # Format table
    table = en_results[['Model Specification', 'n_features', 'train_r2', 'test_r2', 'overfit_gap']].copy()
    table.columns = ['Model Specification', 'N Features', 'Train R²', 'Test R²', 'Overfit Gap']

    # Round values
    for col in ['Train R²', 'Test R²', 'Overfit Gap']:
        table[col] = table[col].apply(lambda x: f'{x:.4f}')

    # Save as CSV
    table.to_csv(output_dir / out_name('supplement_table_variance_decomposition.csv'), index=False)

    # Create markdown version
    try:
        md_table = table.to_markdown(index=False)
    except Exception:
        # Fallback when optional 'tabulate' dependency is not installed.
        md_table = table.to_string(index=False)

    with open(md_path('supplement_table_variance_decomposition.md'), 'w') as f:
        f.write("# Table S_X: Variance Decomposition Across Model Specifications\n\n")
        f.write(md_table)
        f.write("\n\n")
        f.write("## Key Findings\n\n")
        f.write(f"- AR-only model explains {decomposition['individual_contributions']['ar_only']*100:.1f}% of variance\n")
        f.write(f"- Climate-only model explains {decomposition['individual_contributions']['climate_only']*100:.1f}% of variance\n")
        f.write(f"- Climate adds {decomposition['incremental_contributions']['climate_over_ar_seasonal']*100:.1f}% beyond AR + Seasonal\n")
        f.write(f"- AR-Climate overlap: {decomposition['shared_variance']['ar_climate_overlap']*100:.1f}%\n")

    print(f"\nSupplement table saved to {output_dir}")
    return table


def main():
    parser = argparse.ArgumentParser(description='Variance Decomposition Analysis')
    parser.add_argument('--network', default='INF', )
    parser.add_argument('--hsa-mode', default='footprint')
    parser.add_argument('--input-csv', default=None, help='Path to modeling dataset')
    parser.add_argument('--output-dir', default=str(Path(DEFAULT_PIPELINE_OUT_DIR) / 'analysis_variance_decomposition'))
    parser.add_argument('--text-output-dir', default=str(Path(DEFAULT_PIPELINE_OUT_DIR) / 'textresults'))
    parser.add_argument('--target-col', default=None, help='Target column name')

    args = parser.parse_args()

    # Set defaults based on network
    if args.target_col is None:
        args.target_col = 'diarrheal_count_adjusted' if args.network == 'INF' else 'hypertension_count_adjusted'

    if args.input_csv is None:
        args.input_csv = str(Path(DEFAULT_PIPELINE_OUT_DIR) / 'modeling' / f'{args.network}_{args.hsa_mode}_modeling_dataset.csv')

    global OUTPUT_FILE_PREFIX, TEXT_RESULTS_DIR
    OUTPUT_FILE_PREFIX = f"{args.network}_{args.hsa_mode}"
    TEXT_RESULTS_DIR = Path(args.text_output_dir)
    TEXT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir) / args.network
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"VARIANCE DECOMPOSITION ANALYSIS")
    print(f"Network: {args.network}")
    print(f"HSA Mode: {args.hsa_mode}")
    print(f"Target: {args.target_col}")
    print("="*80)

    # Load data
    df = load_data(args.input_csv, args.target_col)

    # Run variance decomposition
    results_df, decomposition = decompose_variance_contributions(df, args.target_col, output_dir)

    # Run residual analysis
    residual_results = test_climate_on_residuals(df, args.target_col, output_dir)

    # Create supplement table
    table = create_supplement_table(results_df, decomposition, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
