#!/usr/bin/env python3
"""
Task 1.3: Spatial Autocorrelation Analysis
==========================================

This script analyzes spatial autocorrelation in model residuals using Moran's I.
Tests whether there's unexplained spatial clustering in prediction errors.

Key questions:
- Do model residuals cluster spatially?
- Does HSA delineation capture spatial dependencies?
- Are there systematic spatial patterns the model misses?

Author: HSA Research Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
import argparse
from pathlib import Path
import json

warnings.filterwarnings('ignore')
OUTPUT_FILE_PREFIX = ""


def out_name(filename: str) -> str:
    return f"{OUTPUT_FILE_PREFIX}_{filename}" if OUTPUT_FILE_PREFIX else filename

# =============================================================================
# CONFIGURATION
# =============================================================================

CLIMATE_FEATURES = [
    'T_mean_week_C', 'T_max_week_C', 'P_total_week', 'Td_week_C',
    'hours_above_30C_week', 'E_week_mm_per_day', 'SM1_week', 'water_deficit_mm_week'
]


def load_facility_coordinates(data_dir, network):
    """Load facility coordinates."""
    fac_file = data_dir / f'{network}_facility_coordinates.csv'
    fac_df = pd.read_csv(fac_file)

    # Standardize column names (case-insensitive)
    col_map = {c.lower(): c for c in fac_df.columns}

    if 'lat' not in fac_df.columns:
        if 'latitude' in col_map:
            fac_df['lat'] = fac_df[col_map['latitude']]
            fac_df['lon'] = fac_df[col_map['longitude']]

    return fac_df


def load_hsa_data(out_dir, network, hsa_mode):
    """Load HSA modeling dataset."""
    hsa_file = out_dir / 'modeling' / f'{network}_{hsa_mode}_modeling_dataset.csv'
    return pd.read_csv(hsa_file)


def create_spatial_weights_matrix(coords, method='inverse_distance', k=5, bandwidth=None):
    """
    Create spatial weights matrix from coordinates.

    Parameters:
    -----------
    coords : array-like
        Nx2 array of (lon, lat) coordinates
    method : str
        'inverse_distance', 'k_nearest', or 'binary'
    k : int
        Number of nearest neighbors (for k_nearest)
    bandwidth : float
        Distance threshold (for binary, in degrees)

    Returns:
    --------
    W : ndarray
        NxN spatial weights matrix (row-normalized)
    """
    n = len(coords)
    coords = np.array(coords)

    # Calculate pairwise distances (Euclidean in degree space - approximation)
    dist_matrix = squareform(pdist(coords))

    if method == 'inverse_distance':
        # Inverse distance weights
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    W[i, j] = 1.0 / (dist_matrix[i, j] + 1e-6)

    elif method == 'k_nearest':
        # K nearest neighbors
        W = np.zeros((n, n))
        for i in range(n):
            # Get k nearest (excluding self)
            distances = dist_matrix[i].copy()
            distances[i] = np.inf
            nearest = np.argsort(distances)[:k]
            W[i, nearest] = 1.0

    elif method == 'binary':
        # Binary weights within bandwidth
        if bandwidth is None:
            bandwidth = np.median(dist_matrix[dist_matrix > 0])
        W = (dist_matrix <= bandwidth) & (dist_matrix > 0)
        W = W.astype(float)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Row normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums

    return W


def calculate_morans_i(residuals, W, permutations=999):
    """
    Calculate Moran's I statistic for spatial autocorrelation.

    Parameters:
    -----------
    residuals : array-like
        Model residuals for each spatial unit
    W : ndarray
        Row-normalized spatial weights matrix
    permutations : int
        Number of permutations for significance testing

    Returns:
    --------
    dict : Moran's I, expected value, variance, z-score, p-value
    """
    n = len(residuals)
    residuals = np.array(residuals)

    # Center residuals
    z = residuals - residuals.mean()

    # Calculate Moran's I
    numerator = n * np.sum(W * np.outer(z, z))
    denominator = W.sum() * np.sum(z ** 2)

    if denominator == 0:
        return {'I': 0, 'EI': 0, 'VI': 0, 'z': 0, 'p_value': 1.0}

    I = numerator / denominator

    # Expected value under null hypothesis
    EI = -1.0 / (n - 1)

    # Variance under null (normality assumption)
    S0 = W.sum()
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2)

    # Normality assumption variance
    n2 = n * n
    VI = (n2 * S1 - n * S2 + 3 * S0 * S0) / ((n2 - 1) * S0 * S0) - EI * EI

    # Z-score
    z_score = (I - EI) / np.sqrt(max(VI, 1e-10))

    # P-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Permutation test for more robust p-value
    if permutations > 0:
        I_perm = []
        for _ in range(permutations):
            perm_residuals = np.random.permutation(residuals)
            z_perm = perm_residuals - perm_residuals.mean()
            num_perm = n * np.sum(W * np.outer(z_perm, z_perm))
            denom_perm = W.sum() * np.sum(z_perm ** 2)
            if denom_perm > 0:
                I_perm.append(num_perm / denom_perm)

        I_perm = np.array(I_perm)
        p_perm = (np.sum(np.abs(I_perm) >= np.abs(I)) + 1) / (permutations + 1)
    else:
        p_perm = p_value

    return {
        'I': float(I),
        'EI': float(EI),
        'VI': float(VI),
        'z_score': float(z_score),
        'p_value_normal': float(p_value),
        'p_value_perm': float(p_perm)
    }


def prepare_model_data(df, target_col, group_col='hsa_id'):
    """Prepare features and target for modeling."""
    df = df.copy()

    climate_cols = [c for c in CLIMATE_FEATURES if c in df.columns]

    # Create AR features per group
    df = df.sort_values([group_col, 'week_number'])
    for lag in [1, 2]:
        df[f'ar_lag{lag}'] = df.groupby(group_col)[target_col].shift(lag)

    # Seasonal features
    df['sin_annual'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['cos_annual'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    df = df.dropna(subset=['ar_lag1', 'ar_lag2'])

    ar_features = ['ar_lag1', 'ar_lag2']
    seasonal_features = ['sin_annual', 'cos_annual']
    features = ar_features + seasonal_features + climate_cols

    return df, features


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


def run_spatial_autocorrelation_analysis(data_dir, out_dir, network, hsa_mode, target_col, output_dir):
    """Run comprehensive spatial autocorrelation analysis."""
    print("="*80)
    print("SPATIAL AUTOCORRELATION ANALYSIS")
    print(f"Network: {network}, Target: {target_col}")
    print("="*80)

    # Load data
    hsa_df = load_hsa_data(out_dir, network, hsa_mode)
    fac_df = load_facility_coordinates(data_dir, network)

    # Get HSA names/ids and their coordinates
    hsa_ids = hsa_df['hsa_id'].unique()
    print(f"\nNumber of HSAs: {len(hsa_ids)}")

    # Match coordinates to HSAs
    # HSA names are like "AL-Ramtha_Hospital" in hsa_id
    # Find the facility name column (case-insensitive)
    fac_name_col = None
    for col in fac_df.columns:
        if col.lower() == 'healthfacility':
            fac_name_col = col
            break

    if fac_name_col is None:
        print("Warning: Could not find HealthFacility column")
        fac_name_col = fac_df.columns[0]

    coords = []
    matched_hsas = []
    for hsa_id in hsa_ids:
        # Try to match facility name
        hsa_name = hsa_id.replace('_', ' ')
        match = fac_df[fac_df[fac_name_col].str.contains(hsa_name.split()[0], case=False, na=False)]
        if len(match) > 0:
            coords.append([match.iloc[0]['lon'], match.iloc[0]['lat']])
            matched_hsas.append(hsa_id)

    if len(coords) < 3:
        print("Warning: Could not match enough HSAs to coordinates. Using all facilities.")
        coords = [[row['lon'], row['lat']] for _, row in fac_df.iterrows()][:len(hsa_ids)]
        matched_hsas = list(hsa_ids)[:len(coords)]

    coords = np.array(coords)
    print(f"Matched {len(coords)} HSAs to coordinates")

    # Create spatial weights
    print("\nCreating spatial weights matrices...")
    W_inv = create_spatial_weights_matrix(coords, method='inverse_distance')
    W_knn = create_spatial_weights_matrix(coords, method='k_nearest', k=min(5, len(coords)-1))

    # Prepare model data
    hsa_df, features = prepare_model_data(hsa_df, target_col)
    train_df, test_df = temporal_train_test_split(hsa_df)

    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")
    print(f"Features: {features}")

    # Fit model and get residuals
    X_train = train_df[features].values
    y_train = train_df[target_col].values
    X_test = test_df[features].values
    y_test = test_df[target_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred_test = model.predict(X_test_scaled)
    residuals_test = y_test - y_pred_test

    test_r2 = r2_score(y_test, y_pred_test)
    print(f"\nModel Test R²: {test_r2:.4f}")

    # Add residuals to test dataframe
    test_df = test_df.copy()
    test_df['residual'] = residuals_test

    # Calculate Moran's I for each week's residuals
    print("\n" + "="*80)
    print("MORAN'S I ANALYSIS BY WEEK")
    print("="*80)

    weekly_results = []
    test_weeks = sorted(test_df['week_number'].unique())

    for week in test_weeks:
        week_data = test_df[test_df['week_number'] == week]

        # Get residuals for matched HSAs in this week
        week_residuals = []
        for hsa_id in matched_hsas:
            hsa_week = week_data[week_data['hsa_id'] == hsa_id]
            if len(hsa_week) > 0:
                week_residuals.append(hsa_week['residual'].values[0])
            else:
                week_residuals.append(0)  # Missing data

        week_residuals = np.array(week_residuals)

        if np.std(week_residuals) > 0:
            morans = calculate_morans_i(week_residuals, W_inv, permutations=99)
            weekly_results.append({
                'week': week,
                'morans_I': morans['I'],
                'z_score': morans['z_score'],
                'p_value': morans['p_value_perm'],
                'n_hsas': len([r for r in week_residuals if r != 0])
            })

    weekly_df = pd.DataFrame(weekly_results)

    print(f"\nWeekly Moran's I Summary:")
    print(f"  Mean I: {weekly_df['morans_I'].mean():.4f}")
    print(f"  Std I: {weekly_df['morans_I'].std():.4f}")
    print(f"  Weeks with significant clustering (p<0.05): {(weekly_df['p_value'] < 0.05).sum()} / {len(weekly_df)}")

    # Aggregate analysis: mean residuals per HSA
    print("\n" + "="*80)
    print("AGGREGATE SPATIAL AUTOCORRELATION")
    print("="*80)

    mean_residuals = []
    for hsa_id in matched_hsas:
        hsa_data = test_df[test_df['hsa_id'] == hsa_id]
        if len(hsa_data) > 0:
            mean_residuals.append(hsa_data['residual'].mean())
        else:
            mean_residuals.append(0)

    mean_residuals = np.array(mean_residuals)

    morans_inv = calculate_morans_i(mean_residuals, W_inv, permutations=999)
    morans_knn = calculate_morans_i(mean_residuals, W_knn, permutations=999)

    print(f"\nInverse Distance Weights:")
    print(f"  Moran's I: {morans_inv['I']:.4f}")
    print(f"  Expected I: {morans_inv['EI']:.4f}")
    print(f"  Z-score: {morans_inv['z_score']:.4f}")
    print(f"  P-value (permutation): {morans_inv['p_value_perm']:.4f}")
    print(f"  Significant at α=0.05: {'Yes' if morans_inv['p_value_perm'] < 0.05 else 'No'}")

    print(f"\nK-Nearest Neighbors (k={min(5, len(coords)-1)}):")
    print(f"  Moran's I: {morans_knn['I']:.4f}")
    print(f"  Expected I: {morans_knn['EI']:.4f}")
    print(f"  Z-score: {morans_knn['z_score']:.4f}")
    print(f"  P-value (permutation): {morans_knn['p_value_perm']:.4f}")
    print(f"  Significant at α=0.05: {'Yes' if morans_knn['p_value_perm'] < 0.05 else 'No'}")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    if morans_inv['p_value_perm'] < 0.05 and morans_inv['I'] > 0:
        print("Significant POSITIVE spatial autocorrelation detected.")
        print("This suggests neighboring HSAs have similar residuals,")
        print("indicating unexplained spatial clustering.")
    elif morans_inv['p_value_perm'] < 0.05 and morans_inv['I'] < 0:
        print("Significant NEGATIVE spatial autocorrelation detected.")
        print("This suggests neighboring HSAs have dissimilar residuals.")
    else:
        print("No significant spatial autocorrelation detected.")
        print("Residuals appear spatially random, suggesting the model")
        print("adequately captures spatial dependencies.")

    # Save results
    results = {
        'model_test_r2': float(test_r2),
        'n_hsas': len(matched_hsas),
        'n_test_weeks': len(test_weeks),
        'aggregate_morans': {
            'inverse_distance': morans_inv,
            'k_nearest': morans_knn
        },
        'weekly_summary': {
            'mean_I': float(weekly_df['morans_I'].mean()),
            'std_I': float(weekly_df['morans_I'].std()),
            'n_significant': int((weekly_df['p_value'] < 0.05).sum()),
            'proportion_significant': float((weekly_df['p_value'] < 0.05).mean())
        }
    }

    with open(output_dir / out_name('spatial_autocorrelation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    weekly_df.to_csv(output_dir / out_name('weekly_morans_i.csv'), index=False)

    # Create visualizations
    plot_spatial_autocorrelation(weekly_df, morans_inv, morans_knn, mean_residuals, coords, output_dir)

    return results


def plot_spatial_autocorrelation(weekly_df, morans_inv, morans_knn, mean_residuals, coords, output_dir):
    """Create visualizations for spatial autocorrelation analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Weekly Moran's I time series
    ax1 = axes[0, 0]
    ax1.plot(weekly_df['week'], weekly_df['morans_I'], 'b-', linewidth=1, label="Moran's I")
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=-1/(len(coords)-1), color='red', linestyle='--', label='Expected (no autocorr.)')
    ax1.fill_between(weekly_df['week'],
                      weekly_df['morans_I'] - 1.96*weekly_df['morans_I'].std(),
                      weekly_df['morans_I'] + 1.96*weekly_df['morans_I'].std(),
                      alpha=0.2)
    ax1.set_xlabel('Week')
    ax1.set_ylabel("Moran's I")
    ax1.set_title("Weekly Moran's I of Model Residuals")
    ax1.legend()

    # 2. Distribution of weekly Moran's I
    ax2 = axes[0, 1]
    ax2.hist(weekly_df['morans_I'], bins=15, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', label='Zero (random)')
    ax2.axvline(x=weekly_df['morans_I'].mean(), color='blue', linestyle='-', label='Mean')
    ax2.set_xlabel("Moran's I")
    ax2.set_ylabel('Frequency')
    ax2.set_title("Distribution of Weekly Moran's I")
    ax2.legend()

    # 3. Spatial plot of mean residuals
    ax3 = axes[1, 0]
    sc = ax3.scatter(coords[:, 0], coords[:, 1], c=mean_residuals, cmap='RdBu_r',
                     s=100, edgecolors='black', linewidth=0.5)
    plt.colorbar(sc, ax=ax3, label='Mean Residual')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title(f"Spatial Distribution of Mean Residuals\nMoran's I={morans_inv['I']:.3f}, p={morans_inv['p_value_perm']:.3f}")

    # 4. P-value distribution
    ax4 = axes[1, 1]
    ax4.hist(weekly_df['p_value'], bins=20, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0.05, color='red', linestyle='--', label='α=0.05')
    ax4.set_xlabel('P-value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Weekly P-values')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / out_name('spatial_autocorrelation_figure.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / out_name('spatial_autocorrelation_figure.pdf'), bbox_inches='tight')
    plt.close()

    print(f"\nFigures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Spatial Autocorrelation Analysis')
    parser.add_argument('--network', default='INF', choices=['INF', 'NCD'])
    parser.add_argument('--hsa-mode', default='footprint')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out-dir', default='out')
    parser.add_argument('--output-dir', default='out/analysis_spatial_autocorrelation')
    parser.add_argument('--target-col', default=None)

    args = parser.parse_args()

    global OUTPUT_FILE_PREFIX
    OUTPUT_FILE_PREFIX = f"{args.network}_{args.hsa_mode}"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    output_dir = Path(args.output_dir) / args.network
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.target_col is None:
        args.target_col = 'diarrheal_count_adjusted' if args.network == 'INF' else 'hypertension_count_adjusted'

    results = run_spatial_autocorrelation_analysis(
        data_dir, out_dir, args.network, args.hsa_mode, args.target_col, output_dir
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
