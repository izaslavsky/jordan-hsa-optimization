#!/usr/bin/env python3
"""
Extreme Event Analysis - Testing Functional Form Hypothesis
============================================================

Tests whether climate EXTREMES predict disease better than climate MEANS.

Hypothesis: "Maybe climate matters, but we tested means instead of extremes"

This script:
1. Creates extreme event indicators from existing climate data
2. Compares models: AR+Means vs AR+Extremes vs AR+Both
3. Identifies which extremes have strongest associations
4. Performs event-based case studies

Author: HSA Research Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
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

# Climate MEAN features (current approach)
CLIMATE_MEANS = [
    'T_mean_week_C', 'T_max_week_C', 'T_min_week_C',
    'P_total_week', 'P_mean_week',
    'Td_week_C',  # Dewpoint (humidity proxy)
    'SM1_week', 'SM2_week',  # Soil moisture
    'E_week_mm_per_day',  # Evapotranspiration
    'water_deficit_mm_week',
]

# Existing extreme indicators in the dataset
EXISTING_EXTREMES = [
    'extreme_heat',  # Binary: extreme heat day
    'heavy_rain',  # Binary: heavy rain
    'heavy_days_week',  # Count of heavy precip days
    'hours_above_30C_week',  # Hours >30°C
    'hours_above_35C_week',  # Hours >35°C
    'P_max_day_week',  # Max daily precip
    'heat_moisture_stress',  # Compound indicator
]


def load_modeling_data(out_dir, network, hsa_mode, boundary_version="v7"):
    """Load the modeling dataset."""
    file_path = out_dir / 'modeling' / f'{network}_{hsa_mode}_modeling_dataset_{boundary_version}.csv'
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"HSAs: {df['hsa_id'].nunique()}, Weeks: {df['week_number'].nunique()}")
    return df


def create_additional_extreme_indicators(df):
    """
    Create additional extreme event indicators from existing columns.
    These supplement the existing extreme indicators in the dataset.
    """
    df = df.copy()

    # ============================================
    # TEMPERATURE EXTREMES
    # ============================================

    # 1. Severe heat: T_max > 40°C (using weekly max)
    df['severe_heat_week'] = (df['T_max_week_C'] > 40).astype(int)

    # 2. Very hot week: Mean temp > 30°C
    df['very_hot_week'] = (df['T_mean_week_C'] > 30).astype(int)

    # 3. Large diurnal temperature range (thermal stress)
    if 'DTR_week_C' in df.columns:
        df['high_dtr'] = (df['DTR_week_C'] > 15).astype(int)
    else:
        df['high_dtr'] = 0

    # 4. Hot nights: T_min > 25°C
    if 'T_min_week_C' in df.columns:
        df['hot_nights_week'] = (df['T_min_week_C'] > 25).astype(int)
    else:
        df['hot_nights_week'] = 0

    # 5. Temperature spike: Week-over-week temp increase > 5°C
    df = df.sort_values(['hsa_id', 'week_number'])
    df['temp_change'] = df.groupby('hsa_id')['T_mean_week_C'].diff()
    df['temp_spike'] = (df['temp_change'] > 5).astype(int)
    df['temp_drop'] = (df['temp_change'] < -5).astype(int)

    # ============================================
    # PRECIPITATION EXTREMES
    # ============================================

    # 6. Extreme precipitation week: Total > 50mm
    df['extreme_precip_week'] = (df['P_total_week'] > 50).astype(int)

    # 7. Very dry week: Total < 1mm
    df['dry_week'] = (df['P_total_week'] < 1).astype(int)

    # 8. Consecutive dry weeks (drought proxy)
    df['prev_dry'] = df.groupby('hsa_id')['dry_week'].shift(1).fillna(0)
    df['dry_spell_2wk'] = ((df['dry_week'] == 1) & (df['prev_dry'] == 1)).astype(int)

    # 9. Drought breaking rain: Heavy rain after dry period
    df['drought_break'] = ((df['prev_dry'] == 1) & (df['P_total_week'] > 25)).astype(int)

    # ============================================
    # COMPOUND EVENTS
    # ============================================

    # 10. Hot and dry: T_max > 35°C AND P < 1mm
    df['hot_dry_week'] = ((df['T_max_week_C'] > 35) & (df['P_total_week'] < 1)).astype(int)

    # 11. Hot and humid: Heat index > 35°C
    if 'heat_index_week_C' in df.columns:
        df['high_heat_index'] = (df['heat_index_week_C'] > 35).astype(int)
    else:
        df['high_heat_index'] = 0

    # 12. Wet and warm: P > 25mm AND T > 25°C (pathogen growth conditions)
    df['wet_warm_week'] = ((df['P_total_week'] > 25) & (df['T_mean_week_C'] > 25)).astype(int)

    # ============================================
    # SOIL MOISTURE EXTREMES
    # ============================================

    # 13. Very dry soil (drought conditions)
    if 'SM1_week' in df.columns:
        sm_threshold = df['SM1_week'].quantile(0.1)
        df['very_dry_soil'] = (df['SM1_week'] < sm_threshold).astype(int)

        # 14. Very wet soil (flooding conditions)
        sm_wet_threshold = df['SM1_week'].quantile(0.9)
        df['very_wet_soil'] = (df['SM1_week'] > sm_wet_threshold).astype(int)
    else:
        df['very_dry_soil'] = 0
        df['very_wet_soil'] = 0

    print(f"Created {13} additional extreme indicators")

    return df


def get_feature_lists(df):
    """Get lists of available features by category."""

    # Climate means (available)
    climate_means = [c for c in CLIMATE_MEANS if c in df.columns]

    # Existing extremes (available)
    existing_extremes = [c for c in EXISTING_EXTREMES if c in df.columns]

    # Created extremes
    created_extremes = [
        'severe_heat_week', 'very_hot_week', 'high_dtr', 'hot_nights_week',
        'temp_spike', 'temp_drop', 'extreme_precip_week', 'dry_week',
        'dry_spell_2wk', 'drought_break', 'hot_dry_week', 'high_heat_index',
        'wet_warm_week', 'very_dry_soil', 'very_wet_soil'
    ]
    created_extremes = [c for c in created_extremes if c in df.columns]

    # All extremes
    all_extremes = list(set(existing_extremes + created_extremes))

    return climate_means, all_extremes


def prepare_model_data(df, target_col, group_col='hsa_id'):
    """Prepare features and target with AR terms."""
    df = df.copy()

    # Create AR features per group
    df = df.sort_values([group_col, 'week_number'])
    for lag in [1, 2]:
        df[f'ar_lag{lag}'] = df.groupby(group_col)[target_col].shift(lag)

    # Create seasonal features
    df['sin_annual'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['cos_annual'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # Drop NA from AR lags
    df = df.dropna(subset=['ar_lag1', 'ar_lag2'])

    return df


def temporal_train_test_split(df, train_frac=0.75, val_frac=0.125):
    """Split data temporally."""
    weeks = sorted(df['week_number'].unique())
    n_weeks = len(weeks)

    train_end = int(n_weeks * train_frac)
    val_end = int(n_weeks * (train_frac + val_frac))

    train_weeks = weeks[:train_end]
    test_weeks = weeks[val_end:]

    train_df = df[df['week_number'].isin(train_weeks)]
    test_df = df[df['week_number'].isin(test_weeks)]

    return train_df, test_df


def fit_and_evaluate(X_train, y_train, X_test, y_test, model_type='elasticnet'):
    """Fit model and return metrics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == 'elasticnet':
        model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    return {
        'model': model,
        'scaler': scaler,
        'train_r2': r2_score(y_train, model.predict(X_train_scaled)),
        'test_r2': r2_score(y_test, y_pred),
        'test_mae': mean_absolute_error(y_test, y_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'y_pred': y_pred
    }


def compare_means_vs_extremes(df, target_col, climate_means, all_extremes, output_dir):
    """
    Compare model specifications:
    A. AR + Seasonal only (baseline)
    B. AR + Seasonal + Climate Means
    C. AR + Seasonal + Extreme Indicators
    D. AR + Seasonal + Means + Extremes (full)
    """
    print("\n" + "="*80)
    print("COMPARING CLIMATE MEANS vs. EXTREME INDICATORS")
    print("="*80)

    # Prepare data
    df = prepare_model_data(df, target_col)
    train_df, test_df = temporal_train_test_split(df)

    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")
    print(f"Climate means: {len(climate_means)}")
    print(f"Extreme indicators: {len(all_extremes)}")

    # Feature sets
    ar_features = ['ar_lag1', 'ar_lag2']
    seasonal_features = ['sin_annual', 'cos_annual']

    model_specs = {
        'A_AR_Seasonal_Only': ar_features + seasonal_features,
        'B_AR_Seasonal_Means': ar_features + seasonal_features + climate_means,
        'C_AR_Seasonal_Extremes': ar_features + seasonal_features + all_extremes,
        'D_AR_Seasonal_Both': ar_features + seasonal_features + climate_means + all_extremes,
    }

    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    results = []
    model_objects = {}

    for spec_name, features in model_specs.items():
        # Filter to available features
        features = [f for f in features if f in df.columns]

        print(f"\n--- {spec_name} ({len(features)} features) ---")

        X_train = train_df[features].values
        X_test = test_df[features].values

        # Fit ElasticNet
        result = fit_and_evaluate(X_train, y_train, X_test, y_test, 'elasticnet')

        results.append({
            'specification': spec_name,
            'n_features': len(features),
            'features_used': ', '.join(features[:5]) + ('...' if len(features) > 5 else ''),
            'train_r2': result['train_r2'],
            'test_r2': result['test_r2'],
            'test_mae': result['test_mae'],
            'overfit_gap': result['train_r2'] - result['test_r2']
        })

        model_objects[spec_name] = {'result': result, 'features': features}

        print(f"  Train R²: {result['train_r2']:.4f}")
        print(f"  Test R²: {result['test_r2']:.4f}")
        print(f"  Test MAE: {result['test_mae']:.2f}")

    results_df = pd.DataFrame(results)

    # Calculate contributions
    baseline_r2 = results_df[results_df['specification'] == 'A_AR_Seasonal_Only']['test_r2'].values[0]

    results_df['contribution_over_baseline'] = results_df['test_r2'] - baseline_r2

    print("\n" + "="*80)
    print("SUMMARY: Climate Contribution Over AR+Seasonal Baseline")
    print("="*80)

    for _, row in results_df.iterrows():
        contrib = row['contribution_over_baseline']
        sign = '+' if contrib >= 0 else ''
        print(f"  {row['specification']}: {sign}{contrib:.4f} ({sign}{contrib*100:.2f}%)")

    # Save results
    results_df.to_csv(output_dir / out_name('means_vs_extremes_comparison.csv'), index=False)

    return results_df, model_objects


def analyze_extreme_importance(model_objects, all_extremes, df, output_dir):
    """Analyze which extreme indicators have strongest effects."""
    print("\n" + "="*80)
    print("EXTREME INDICATOR IMPORTANCE ANALYSIS")
    print("="*80)

    # Get extremes-only model
    extremes_model = model_objects['C_AR_Seasonal_Extremes']
    model = extremes_model['result']['model']
    features = extremes_model['features']

    # Extract coefficients
    coef_data = []
    for i, feat in enumerate(features):
        coef_data.append({
            'feature': feat,
            'coefficient': model.coef_[i],
            'abs_coef': abs(model.coef_[i])
        })

    coef_df = pd.DataFrame(coef_data).sort_values('abs_coef', ascending=False)

    # Separate AR/seasonal from extremes
    ar_seasonal = ['ar_lag1', 'ar_lag2', 'sin_annual', 'cos_annual']
    extreme_coefs = coef_df[~coef_df['feature'].isin(ar_seasonal)].copy()

    # Calculate prevalence of each extreme
    for feat in extreme_coefs['feature']:
        if feat in df.columns:
            extreme_coefs.loc[extreme_coefs['feature'] == feat, 'pct_weeks_active'] = (df[feat] > 0).mean() * 100
            extreme_coefs.loc[extreme_coefs['feature'] == feat, 'mean_value'] = df[feat].mean()

    print("\nTop 10 Extreme Indicators by Effect Size:")
    print("-" * 60)
    for _, row in extreme_coefs.head(10).iterrows():
        pct = row.get('pct_weeks_active', 0)
        print(f"  {row['feature']:30s}: coef={row['coefficient']:+.4f}, active={pct:.1f}% of weeks")

    # Save
    extreme_coefs.to_csv(output_dir / out_name('extreme_indicator_importance.csv'), index=False)

    return extreme_coefs


def event_case_study(df, target_col, extreme_col, output_dir):
    """
    Analyze disease response to specific extreme events.
    Track excess cases in weeks following extreme events.
    """
    print(f"\n--- Event Case Study: {extreme_col} ---")

    df = df.copy()

    # Calculate baseline expected cases per HSA per week-of-year
    df['expected_cases'] = df.groupby(['hsa_id', 'week_of_year'])[target_col].transform('mean')
    df['excess_cases'] = df[target_col] - df['expected_cases']

    # Find event weeks
    event_weeks = df[df[extreme_col] > 0].copy()
    print(f"  Found {len(event_weeks)} weeks with {extreme_col} > 0")

    if len(event_weeks) < 5:
        print(f"  Too few events for analysis")
        return None

    # For each event, look at excess cases at lag 0, 1, 2, 3 weeks
    event_analysis = []

    for _, event_row in event_weeks.iterrows():
        hsa_id = event_row['hsa_id']
        event_week = event_row['week_number']

        # Get following weeks
        following = df[(df['hsa_id'] == hsa_id) &
                       (df['week_number'] >= event_week) &
                       (df['week_number'] <= event_week + 3)]

        if len(following) >= 4:
            following = following.sort_values('week_number')
            event_analysis.append({
                'hsa_id': hsa_id,
                'event_week': event_week,
                'event_intensity': event_row[extreme_col],
                'excess_lag0': following.iloc[0]['excess_cases'],
                'excess_lag1': following.iloc[1]['excess_cases'] if len(following) > 1 else np.nan,
                'excess_lag2': following.iloc[2]['excess_cases'] if len(following) > 2 else np.nan,
                'excess_lag3': following.iloc[3]['excess_cases'] if len(following) > 3 else np.nan,
            })

    if len(event_analysis) < 5:
        print(f"  Not enough complete event sequences")
        return None

    events_df = pd.DataFrame(event_analysis)

    # Statistical tests
    tests = {}
    for lag in ['lag0', 'lag1', 'lag2', 'lag3']:
        col = f'excess_{lag}'
        values = events_df[col].dropna()
        if len(values) > 3:
            t_stat, p_val = stats.ttest_1samp(values, 0)
            tests[lag] = {
                'mean_excess': values.mean(),
                'std_excess': values.std(),
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'n_events': len(values)
            }

    print(f"  Disease response by lag (mean excess cases):")
    for lag, res in tests.items():
        sig = '*' if res['significant'] else ''
        print(f"    {lag}: {res['mean_excess']:+.2f} ± {res['std_excess']:.2f} (p={res['p_value']:.3f}){sig}")

    return {'events': events_df, 'tests': tests}


def run_event_case_studies(df, target_col, all_extremes, output_dir):
    """Run case studies for each extreme type."""
    print("\n" + "="*80)
    print("EVENT-BASED CASE STUDIES")
    print("="*80)

    # Prepare data
    df = prepare_model_data(df, target_col)

    # Test each extreme type
    case_study_results = {}

    for extreme_col in all_extremes:
        if extreme_col in df.columns and df[extreme_col].sum() > 10:
            result = event_case_study(df, target_col, extreme_col, output_dir)
            if result:
                case_study_results[extreme_col] = result

    # Summary
    print("\n" + "-"*60)
    print("Summary: Significant Disease Responses to Extreme Events")
    print("-"*60)

    significant_effects = []
    for extreme, res in case_study_results.items():
        for lag, test in res['tests'].items():
            if test['significant']:
                significant_effects.append({
                    'extreme_type': extreme,
                    'lag': lag,
                    'mean_excess': test['mean_excess'],
                    'p_value': test['p_value'],
                    'n_events': test['n_events']
                })

    if significant_effects:
        sig_df = pd.DataFrame(significant_effects).sort_values('p_value')
        print(sig_df.to_string(index=False))
        sig_df.to_csv(output_dir / out_name('significant_extreme_effects.csv'), index=False)
    else:
        print("No statistically significant effects found")

    return case_study_results


def create_visualizations(results_df, extreme_coefs, output_dir):
    """Create summary visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Model comparison
    ax1 = axes[0, 0]
    specs = results_df['specification'].str.replace('_', '\n')
    bars = ax1.bar(specs, results_df['test_r2'], color=['gray', 'blue', 'red', 'purple'])
    ax1.axhline(y=results_df.iloc[0]['test_r2'], color='black', linestyle='--',
                label=f"Baseline: {results_df.iloc[0]['test_r2']:.4f}")
    ax1.set_ylabel('Test R²')
    ax1.set_title('Model Comparison: Means vs Extremes')
    ax1.legend()

    # Add value labels
    for bar, val in zip(bars, results_df['test_r2']):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}',
                ha='center', fontsize=10)

    # 2. Contribution over baseline
    ax2 = axes[0, 1]
    contrib = results_df['contribution_over_baseline']
    colors = ['green' if c > 0 else 'red' for c in contrib]
    ax2.bar(specs, contrib * 100, color=colors)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Contribution over Baseline (%)')
    ax2.set_title('Added Predictive Value Over AR+Seasonal')

    # 3. Top extreme indicator coefficients
    ax3 = axes[1, 0]
    top_extremes = extreme_coefs.head(10)
    colors = ['green' if c > 0 else 'red' for c in top_extremes['coefficient']]
    ax3.barh(top_extremes['feature'], top_extremes['coefficient'], color=colors)
    ax3.axvline(x=0, color='black', linewidth=0.5)
    ax3.set_xlabel('Model Coefficient')
    ax3.set_title('Top 10 Extreme Indicators by Effect Size')

    # 4. Prevalence vs effect size
    ax4 = axes[1, 1]
    if 'pct_weeks_active' in extreme_coefs.columns:
        scatter = ax4.scatter(extreme_coefs['pct_weeks_active'],
                             extreme_coefs['abs_coef'],
                             alpha=0.6, s=60)
        ax4.set_xlabel('% of Weeks with Event')
        ax4.set_ylabel('Absolute Coefficient')
        ax4.set_title('Event Frequency vs. Effect Strength')

        # Label top points
        for _, row in extreme_coefs.head(5).iterrows():
            if pd.notna(row.get('pct_weeks_active')):
                ax4.annotate(row['feature'][:15],
                            (row['pct_weeks_active'], row['abs_coef']),
                            fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / out_name('extreme_event_analysis_figure.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / out_name('extreme_event_analysis_figure.pdf'), bbox_inches='tight')
    plt.close()

    print(f"\nFigures saved to {output_dir}")


def create_summary_table(results_df, output_dir):
    """Create markdown summary table."""
    table = results_df[['specification', 'n_features', 'train_r2', 'test_r2',
                        'contribution_over_baseline']].copy()
    table.columns = ['Model', 'N Features', 'Train R²', 'Test R²', 'Contribution']

    for col in ['Train R²', 'Test R²', 'Contribution']:
        table[col] = table[col].apply(lambda x: f'{x:.4f}')

    try:
        md_table = table.to_markdown(index=False)
    except Exception:
        md_table = table.to_string(index=False)

    with open(md_path('extreme_event_summary.md'), 'w') as f:
        f.write("# Extreme Event Analysis: Means vs Extremes\n\n")
        f.write("## Model Comparison\n\n")
        f.write(md_table)
        f.write("\n\n## Key Findings\n\n")

        means_r2 = results_df[results_df['specification'] == 'B_AR_Seasonal_Means']['test_r2'].values[0]
        extremes_r2 = results_df[results_df['specification'] == 'C_AR_Seasonal_Extremes']['test_r2'].values[0]
        baseline_r2 = results_df[results_df['specification'] == 'A_AR_Seasonal_Only']['test_r2'].values[0]

        f.write(f"- Baseline (AR+Seasonal): R² = {baseline_r2:.4f}\n")
        f.write(f"- Adding climate means: R² = {means_r2:.4f} (Δ = {(means_r2-baseline_r2)*100:+.2f}%)\n")
        f.write(f"- Adding extreme indicators: R² = {extremes_r2:.4f} (Δ = {(extremes_r2-baseline_r2)*100:+.2f}%)\n")

        if extremes_r2 > means_r2:
            f.write("\n**Conclusion:** Extreme indicators outperform climate means.\n")
        elif means_r2 > extremes_r2:
            f.write("\n**Conclusion:** Climate means outperform extreme indicators.\n")
        else:
            f.write("\n**Conclusion:** Neither means nor extremes add substantial predictive value.\n")


def main():
    parser = argparse.ArgumentParser(description='Extreme Event Analysis')
    parser.add_argument('--network', default='INF', )
    parser.add_argument('--hsa-mode', default='footprint')
    parser.add_argument('--out-dir', default=DEFAULT_PIPELINE_OUT_DIR)
    parser.add_argument('--output-dir', default=str(Path(DEFAULT_PIPELINE_OUT_DIR) / 'analysis_extreme_events'))
    parser.add_argument('--text-output-dir', default=str(Path(DEFAULT_PIPELINE_OUT_DIR) / 'textresults'))
    parser.add_argument('--target-col', default=None)
    parser.add_argument('--boundary-version', default=os.environ.get("BOUNDARY_VERSION", os.environ.get("PIPELINE_VERSION", "v7")),
                        help="HSA boundary version (v6, v7, v8)")

    args = parser.parse_args()

    global OUTPUT_FILE_PREFIX, TEXT_RESULTS_DIR
    OUTPUT_FILE_PREFIX = f"{args.network}_{args.hsa_mode}"
    TEXT_RESULTS_DIR = Path(args.text_output_dir)
    TEXT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out_dir)
    output_dir = Path(args.output_dir) / args.network
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.target_col is None:
        args.target_col = 'diarrheal_count_adjusted' if args.network == 'INF' else 'hypertension_count_adjusted'

    print("="*80)
    print("EXTREME EVENT ANALYSIS")
    print(f"Network: {args.network}")
    print(f"Target: {args.target_col}")
    print("="*80)

    # Load data
    df = load_modeling_data(out_dir, args.network, args.hsa_mode, args.boundary_version)

    # Create additional extreme indicators
    df = create_additional_extreme_indicators(df)

    # Get feature lists
    climate_means, all_extremes = get_feature_lists(df)
    print(f"\nClimate means available: {len(climate_means)}")
    print(f"Extreme indicators available: {len(all_extremes)}")

    # Compare means vs extremes
    results_df, model_objects = compare_means_vs_extremes(
        df, args.target_col, climate_means, all_extremes, output_dir
    )

    # Analyze extreme importance
    extreme_coefs = analyze_extreme_importance(model_objects, all_extremes, df, output_dir)

    # Event case studies
    case_studies = run_event_case_studies(df, args.target_col, all_extremes, output_dir)

    # Create visualizations
    create_visualizations(results_df, extreme_coefs, output_dir)

    # Create summary
    create_summary_table(results_df, output_dir)

    # Save full results
    with open(output_dir / out_name('analysis_summary.json'), 'w') as f:
        summary = {
            'network': args.network,
            'target': args.target_col,
            'n_climate_means': len(climate_means),
            'n_extreme_indicators': len(all_extremes),
            'model_comparison': results_df.to_dict('records')
        }
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
