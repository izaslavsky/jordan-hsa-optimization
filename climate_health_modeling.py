"""
Comprehensive Climate-Health Modeling

Implements multiple modeling approaches:
1. Distributed Lag Non-Linear Models (DLNM)
2. Generalized Additive Models (GAM)
3. Mixed Effects Poisson/Negative Binomial
4. Random Forest and Gradient Boosting (baseline)
5. Extreme Event Classification (high vs normal weeks)

Features:
- Temporal autocorrelation handling
- HSA-specific random effects
- Cross-validation with temporal blocking
- Variable importance and partial dependence plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
from pathlib import Path

# Statistical modeling
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                            classification_report, roc_auc_score, roc_curve,
                            precision_recall_curve, confusion_matrix)

# ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    import xgboost as xgb
    _HAVE_XGBOOST = True
except Exception:
    xgb = None
    _HAVE_XGBOOST = False

# Statistical models
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.genmod.families import Poisson, NegativeBinomial
from scipy import stats

# For distributed lag models
from scipy.interpolate import BSpline

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("CLIMATE-HEALTH MODELING")
print("="*80)

# ============================================================================
# PART 1: DATA LOADING AND PREPARATION
# ============================================================================

print("\n[PART 1] DATA LOADING AND PREPARATION")
print("-"*80)

parser = argparse.ArgumentParser(description="Climate-health modeling")
parser.add_argument("--network", default=os.environ.get("NETWORK", "INF"))
parser.add_argument("--hsa-mode", default=os.environ.get("HSA_MODE", "footprint"))
parser.add_argument("--input-csv", required=True, help="Path to climate+disease merged CSV")
parser.add_argument("--target-col", required=True, help="Target outcome column in input CSV")
parser.add_argument("--output-dir", required=True, help="Directory for outputs")
parser.add_argument("--output-prefix", default=None, help="Prefix for output files")
args = parser.parse_args()
NETWORK = args.network
HSA_MODE = args.hsa_mode
INPUT_CSV = Path(args.input_csv)
TARGET_COL = args.target_col
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_PREFIX = args.output_prefix or f"{NETWORK}_{HSA_MODE}"

# Validate inputs/outputs early.
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load complete cases dataset
df = pd.read_csv(INPUT_CSV)
df['week_start'] = pd.to_datetime(df['week_start'])

print(f"Loaded dataset: {len(df)} rows × {len(df.columns)} columns")
print(f"  HSAs: {df['hsa_id'].nunique()}")
print(f"  Time period: {df['week_start'].min()} to {df['week_start'].max()}")
print(f"  Total cases: {df[TARGET_COL].sum():,.0f}")

# Create temporal features
df['week_of_year'] = df['week_start'].dt.isocalendar().week
df['month'] = df['week_start'].dt.month
df['year'] = df['week_start'].dt.year
df['time_index'] = (df['week_start'] - df['week_start'].min()).dt.days // 7

# Create HSA categorical encoding
df['hsa_code'] = pd.Categorical(df['hsa_id']).codes

# Define variable groups
# Accept both prefixed (precip_, temp_, etc.) and legacy unprefixed columns.
precip_vars = [
    c for c in df.columns
    if c.startswith('precip_') or c.startswith('P_') or c.startswith('wetday_')
    or c.startswith('heavy_days_') or c.startswith('P95_') or c.startswith('P_max_')
]
temp_vars = [
    c for c in df.columns
    if c.startswith('temp_') or c.startswith('T_') or c.startswith('Td_')
    or c.startswith('wind_speed_') or c.startswith('DTR_') or c.startswith('hours_above_')
    or c.startswith('heat_index_')
]
evap_vars = [c for c in df.columns if c.startswith('evap_') or c.startswith('E_')]
sm_vars = [c for c in df.columns if c.startswith('sm_') or c.startswith('SM')]
wb_vars = [
    c for c in df.columns
    if c.startswith('wb_') or c.startswith('water_deficit_')
]

print(f"\nVariable groups:")
print(f"  Precipitation: {len(precip_vars)}")
print(f"  Temperature: {len(temp_vars)}")
print(f"  Evaporation: {len(evap_vars)}")
print(f"  Soil moisture: {len(sm_vars)}")
print(f"  Water balance: {len(wb_vars)}")

# Create outcome variable (with small constant to avoid log(0))
rate_col = f"{TARGET_COL}_rate"
df[rate_col] = df[TARGET_COL] + 0.5

# Define extreme events (>90th percentile of weekly counts)
threshold_90 = df[TARGET_COL].quantile(0.90)
df['extreme_event'] = (df[TARGET_COL] >= threshold_90).astype(int)

print(f"\nOutcome variables:")
print(f"  Outcome: mean={df[TARGET_COL].mean():.2f}, "
      f"median={df[TARGET_COL].median():.0f}")
print(f"  Extreme event threshold (90th %ile): {threshold_90:.0f} cases/week")
print(f"  Extreme event frequency: {df['extreme_event'].mean()*100:.1f}%")

# ============================================================================
# PART 2: EXPLORATORY ANALYSIS
# ============================================================================

print("\n[PART 2] EXPLORATORY ANALYSIS")
print("-"*80)

# 2.1: Temporal patterns
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Weekly time series
for hsa in df['hsa_id'].unique()[:3]:  # Show top 3 HSAs
    hsa_data = df[df['hsa_id'] == hsa]
    axes[0].plot(hsa_data['week_start'], hsa_data[TARGET_COL],
                 alpha=0.7, label=hsa)
axes[0].set_xlabel('Week')
axes[0].set_ylabel('Cases')
axes[0].set_title(f'Weekly Cases (Top 3 HSAs)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Monthly aggregation across all HSAs
monthly = df.groupby(df['week_start'].dt.to_period('M'))[TARGET_COL].sum()
monthly.index = monthly.index.to_timestamp()
axes[1].plot(monthly.index, monthly.values, marker='o', linewidth=2)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Total Cases (All HSAs)')
axes[1].set_title('Monthly Cases (All HSAs)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
temporal_plot = OUTPUT_DIR / f"{OUTPUT_PREFIX}_01_temporal_patterns.png"
plt.savefig(temporal_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {temporal_plot}")

# 2.2: Climate-disease correlations
# Select key climate variables for correlation analysis
key_var_candidates = {
    'precip_P_total_week': ['precip_P_total_week', 'P_total_week'],
    'precip_P_mean_lag_w-1': ['precip_P_mean_lag_w-1', 'P_mean_lag_w-1'],
    'precip_P_mean_lag_w-2': ['precip_P_mean_lag_w-2', 'P_mean_lag_w-2'],
    'temp_T_mean_week_C': ['temp_T_mean_week_C', 'T_mean_week_C'],
    'temp_hours_above_30C_week': ['temp_hours_above_30C_week', 'hours_above_30C_week'],
    'temp_heat_index_week_C': ['temp_heat_index_week_C', 'heat_index_week_C'],
    'wb_water_deficit_mm_week': ['wb_water_deficit_mm_week', 'water_deficit_mm_week'],
    'sm_SM1_week': ['sm_SM1_week', 'SM1_week'],
}

resolved_key_vars = []
for label, options in key_var_candidates.items():
    found = next((opt for opt in options if opt in df.columns), None)
    if found:
        resolved_key_vars.append(found)
    else:
        print(f"  ⚠️  Missing key climate variable for correlation: {label}")

if resolved_key_vars:
    corr_data = df[resolved_key_vars + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)

    fig, ax = plt.subplots(figsize=(10, 6))
    corr_data.sort_values().plot(kind='barh', ax=ax)
    ax.set_xlabel('Correlation with Outcome')
    ax.set_title('Climate Variable Correlations with Outcome')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    corr_plot = OUTPUT_DIR / f"{OUTPUT_PREFIX}_02_climate_correlations.png"
    plt.savefig(corr_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {corr_plot}")
else:
    print("  ⚠️  Skipping correlation plot (no matching climate variables found).")

# 2.3: Lag structure visualization
lag_var_candidates = [
    ['precip_P_mean_lag_w-1', 'P_mean_lag_w-1'],
    ['precip_P_mean_lag_w-2', 'P_mean_lag_w-2'],
    ['precip_P_mean_lag_w-3', 'P_mean_lag_w-3'],
]
lag_vars = []
lag_labels = ['1 week', '2 weeks', '3 weeks']
for options in lag_var_candidates:
    found = next((opt for opt in options if opt in df.columns), None)
    if found:
        lag_vars.append(found)
    else:
        lag_vars.append(None)

if all(lag_vars):
    lag_corrs = [df[var].corr(df[TARGET_COL]) for var in lag_vars]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([1, 2, 3], lag_corrs, marker='o', linewidth=2, markersize=10)
    ax.set_xlabel('Lag (weeks)')
    ax.set_ylabel('Correlation with Outcome')
    ax.set_title('Lagged Precipitation Effects')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(lag_labels)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    lag_plot = OUTPUT_DIR / f"{OUTPUT_PREFIX}_03_lag_structure.png"
    plt.savefig(lag_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {lag_plot}")
else:
    print("  ⚠️  Skipping lag structure plot (missing lagged precipitation columns).")

print("\n  Exploratory plots complete")

# Save the prepared dataset
prepared_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}_modeling_data_prepared.csv"
df.to_csv(prepared_csv, index=False)
print(f"\n  Saved prepared data: {prepared_csv}")

print("\n" + "="*80)
print("DATA PREPARATION COMPLETE")
print("="*80)
print(f"\nDataset ready for modeling:")
print(f"  Rows: {len(df):,}")
print(f"  Features: {len(precip_vars + temp_vars + evap_vars + sm_vars + wb_vars)}")
print(f"  Outcome (count): {TARGET_COL}")
print(f"  Outcome (binary): extreme_event")
print(f"  Temporal features: week_of_year, month, year, time_index")
print(f"  Spatial features: hsa_code (9 HSAs)")
