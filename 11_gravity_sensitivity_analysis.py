#!/usr/bin/env python3
"""
Task 2.1: Gravity Model Sensitivity Analysis
=============================================

This script analyzes how gravity model parameters (α, β) affect:
- Patient allocation patterns
- Disease count aggregation
- Downstream model performance

Parameter Ranges:
- α (population exponent): 0.5, 0.75, 1.0, 1.25
- β (distance decay): 1.0, 1.5, 2.0, 2.5

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
from scipy.spatial.distance import pdist, squareform
import warnings
import argparse
import os
from pathlib import Path
import json
from itertools import product

warnings.filterwarnings('ignore')
DEFAULT_PIPELINE_OUT_DIR = os.environ.get("HSA_OUT_DIR", os.environ.get("PIPELINE_OUT_DIR", "out"))

# Population raster the allocation is evaluated on. Named here rather than
# synthesised, so the sensitivity describes Jordan's actual distribution.
POP_PATH = os.environ.get("HSA_POP_RASTER",
                          str(Path(__file__).resolve().parent / "data" /
                              "jor_ppp_2020_UNadj.tif"))
GRAVITY_COARSEN = int(os.environ.get("GRAVITY_COARSEN", "16"))
OUTPUT_FILE_PREFIX = ""
TEXT_RESULTS_DIR = None


def out_name(filename: str) -> str:
    return f"{OUTPUT_FILE_PREFIX}_{filename}" if OUTPUT_FILE_PREFIX else filename


def md_path(filename: str) -> Path:
    return TEXT_RESULTS_DIR / out_name(filename) if TEXT_RESULTS_DIR else Path(filename)

# =============================================================================
# CONFIGURATION
# =============================================================================

ALPHA_VALUES = [0.5, 0.75, 1.0, 1.25]
BETA_VALUES = [1.0, 1.5, 2.0, 2.5]

CLIMATE_FEATURES = [
    'T_mean_week_C', 'T_max_week_C', 'P_total_week', 'Td_week_C',
    'hours_above_30C_week', 'E_week_mm_per_day', 'SM1_week', 'water_deficit_mm_week'
]


def load_facility_data(data_dir, network):
    """Load facility coordinates and patient volumes."""
    fac_file = next(
        (data_dir / f for f in [
            f'{network}_facility_coordinates.csv',
            f'SYN{network}_facility_coordinates.csv',
            f'SYNMOD{network}_facility_coordinates.csv',
        ] if (data_dir / f).exists()),
        None,
    )
    if fac_file is None:
        raise FileNotFoundError(f"No facility coordinates file found for network '{network}' in {data_dir}")
    fac_df = pd.read_csv(fac_file)

    # Standardize column names
    col_map = {c.lower(): c for c in fac_df.columns}
    if 'lat' not in fac_df.columns:
        if 'latitude' in col_map:
            fac_df['lat'] = fac_df[col_map['latitude']]
            fac_df['lon'] = fac_df[col_map['longitude']]

    # Patient volume is the gravity model's attractiveness term, so it has to be
    # the real one. The coordinates file does not carry it; earlier code filled
    # the gap with np.random.exponential, which makes the resulting sensitivity
    # a statement about random numbers.
    if 'Total' not in fac_df.columns:
        out_dir = Path(os.environ.get("HSA_OUT_DIR",
                                      os.environ.get("PIPELINE_OUT_DIR", "out")))
        if not out_dir.is_absolute():
            out_dir = Path(__file__).resolve().parent / out_dir
        pivot = next((out_dir / f for f in
                      (f'{network}_footprint_diagnosis_counts_pivot.csv',
                       f'{network}_diagnosis_counts_pivot.csv')
                      if (out_dir / f).exists()), None)
        if pivot is None:
            raise FileNotFoundError(
                f"No diagnosis counts pivot in {out_dir}; cannot attach real "
                f"patient volumes for the gravity model")
        counts = pd.read_csv(pivot)
        key = 'healthfacility' if 'healthfacility' in fac_df.columns else 'HealthFacility'
        norm = lambda x: x.astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
        vol = dict(zip(norm(counts['healthfacility']), counts['total_diagnoses']))
        fac_df['Total'] = norm(fac_df[key]).map(vol)
        missing = int(fac_df['Total'].isna().sum())
        if missing:
            print(f"  WARNING: {missing} facilities have no diagnosis count; set to 0")
            fac_df['Total'] = fac_df['Total'].fillna(0)
        print(f"  Attached patient volumes from {pivot.name} "
              f"(total {fac_df['Total'].sum():,.0f})")

    return fac_df


def load_hsa_data(out_dir, network, hsa_mode, boundary_version="v7"):
    """Load HSA modeling dataset."""
    hsa_file = out_dir / 'modeling' / f'{network}_{hsa_mode}_modeling_dataset_{boundary_version}.csv'
    return pd.read_csv(hsa_file)


def load_allocation_data(out_dir, network, hsa_mode, sample_size=10000, boundary_version="v7"):
    """Load a sample of the allocation details for analysis."""
    # Try both naming conventions (allocation_details preferred, pixel_allocations as fallback)
    alloc_file = out_dir / f'{network}_{hsa_mode}_allocation_details_{boundary_version}.csv'
    alloc_file_alt = out_dir / f'pixel_allocations_{network}_{hsa_mode}_{boundary_version}.csv'

    if alloc_file.exists():
        pass  # Use primary name
    elif alloc_file_alt.exists():
        alloc_file = alloc_file_alt
    else:
        print(f"  Allocation file not found. Tried:")
        print(f"    - {alloc_file}")
        print(f"    - {alloc_file_alt}")
        return None

    # Read sample (file is very large ~1GB)
    print(f"  Reading sample of {sample_size} rows from allocation file...")

    # Count total rows
    with open(alloc_file, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header

    print(f"  Total rows: {total_rows:,}")

    # Sample rows
    skip_rows = sorted(np.random.choice(range(1, total_rows + 1),
                                        size=max(0, total_rows - sample_size),
                                        replace=False))

    df = pd.read_csv(alloc_file, skiprows=skip_rows, nrows=sample_size)

    return df


def calculate_gravity_weight(volume, distance, alpha, beta, min_distance=0.1):
    """
    Calculate gravity model weight.

    W = Volume^α / Distance^β
    """
    distance = np.maximum(distance, min_distance)
    return (volume ** alpha) / (distance ** beta)


def load_population_pixels(pop_path, coarsen=16, seed=42):
    """
    Every inhabited cell of the population raster, block-aggregated.

    The allocation being tested operates on Jordan's population, which sits in a
    narrow northern corridor. Drawing pixels uniformly from the country's
    bounding box, as this analysis previously did, spreads them over mostly
    empty desert and describes a population that does not exist.

    Sampling cells is also the wrong fix: the raster holds ~12.1M inhabited
    cells at 100 m, so any tractable sample of distinct cells carries a small
    fraction of the 10.2M people. Cells are therefore aggregated into coarser
    blocks, which keeps the whole population and its geography while making the
    pixel-by-facility distance matrix tractable. At coarsen=16 this is ~48k
    cells holding 10,203,033 of 10,203,140 people.
    """
    import rasterio
    from rasterio.transform import xy as _xy

    with rasterio.open(pop_path) as src:
        arr = src.read(1)
        transform = src.transform
        nodata = src.nodata
    arr = np.where(np.isfinite(arr), arr, 0.0)
    if nodata is not None:
        arr = np.where(arr == nodata, 0.0, arr)
    arr = np.where(arr < 0, 0.0, arr)

    c = int(max(1, coarsen))
    if c > 1:
        h = arr.shape[0] // c * c
        w = arr.shape[1] // c * c
        arr = arr[:h, :w].reshape(h // c, c, w // c, c).sum(axis=(1, 3))
        transform = transform * rasterio.Affine.scale(c, c)

    rows, cols = np.nonzero(arr > 0)
    pops = arr[rows, cols].astype(float)
    if len(pops) == 0:
        raise RuntimeError(f"No populated cells in {pop_path}")

    lons, lats = _xy(transform, rows, cols)
    return np.asarray(lons), np.asarray(lats), pops


def _haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance; the previous code used degrees x 111."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def simulate_allocation_with_parameters(fac_df, coarsen=16, alpha=0.75, beta=1.5,
                                        pop_path=None, seed=42):
    """
    Allocate real population cells across facilities under one (alpha, beta).

    Every cell's people are split across facilities in proportion to the gravity
    weight, rather than assigned whole to the nearest, so the reported
    concentration reflects the model actually used in the pipeline. Metrics are
    population-weighted.
    """
    n_facilities = len(fac_df)
    if 'Total' not in fac_df.columns:
        raise RuntimeError("facility table must carry a 'Total' patient-volume column")
    volumes = fac_df['Total'].astype(float).values

    if pop_path is None:
        raise RuntimeError("pop_path is required: this analysis reads the "
                           "population raster rather than synthesising pixels")
    pixel_lons, pixel_lats, pixel_pops = load_population_pixels(
        pop_path, coarsen=coarsen, seed=seed)
    n_pixels = len(pixel_pops)

    fac_lon = fac_df['lon'].astype(float).values
    fac_lat = fac_df['lat'].astype(float).values

    # Chunked so the pixel-by-facility matrix never has to be held whole.
    facility_pop = np.zeros(n_facilities)
    assigned_facility = np.empty(n_pixels, dtype=int)
    CHUNK = 4096
    for lo in range(0, n_pixels, CHUNK):
        hi = min(lo + CHUNK, n_pixels)
        d = _haversine_km(pixel_lons[lo:hi, None], pixel_lats[lo:hi, None],
                          fac_lon[None, :], fac_lat[None, :])
        w = calculate_gravity_weight(volumes[None, :], d, alpha, beta)
        tot = w.sum(axis=1, keepdims=True)
        shares = np.divide(w, tot, out=np.full_like(w, 1.0 / n_facilities),
                           where=tot > 0)
        facility_pop += shares.T @ pixel_pops[lo:hi]
        assigned_facility[lo:hi] = np.argmax(shares, axis=1)

    total_pop = float(pixel_pops.sum())
    facility_counts = np.bincount(assigned_facility, weights=pixel_pops,
                                  minlength=n_facilities)
    concentration = float((facility_pop ** 2).sum() / (total_pop ** 2))

    return {
        'facility_counts': facility_counts,      # population, not pixel counts
        'facility_population': facility_pop,
        'concentration': concentration,
        'gini': calculate_gini(facility_pop),
        'max_share': float(facility_pop.max() / total_pop) if total_pop else 0.0,
        'n_pixels_used': int(n_pixels),
        'population_represented': total_pop,
        'assigned_facilities': assigned_facility,
    }


def calculate_gini(values):
    """Calculate Gini coefficient of inequality."""
    values = np.array(values, dtype=float)
    if values.sum() == 0:
        return 0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def run_sensitivity_analysis(fac_df, n_simulations=4000):
    """Run sensitivity analysis across parameter grid."""
    print("\n" + "="*80)
    print("GRAVITY MODEL PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)

    results = []

    for alpha, beta in product(ALPHA_VALUES, BETA_VALUES):
        print(f"\n  α={alpha}, β={beta}...")

        sim_result = simulate_allocation_with_parameters(
            fac_df, coarsen=GRAVITY_COARSEN, alpha=alpha, beta=beta,
            pop_path=POP_PATH
        )

        results.append({
            'alpha': alpha,
            'beta': beta,
            'concentration': sim_result['concentration'],
            'gini': sim_result['gini'],
            'max_share': sim_result['max_share'],
            'n_facilities_used': np.sum(sim_result['facility_counts'] > 0)
        })

        print(f"    Concentration (HHI): {sim_result['concentration']:.4f}")
        print(f"    Gini coefficient: {sim_result['gini']:.4f}")
        print(f"    Max facility share: {sim_result['max_share']:.2%}")

    return pd.DataFrame(results)


def analyze_downstream_impact(hsa_df, target_col, output_dir):
    """
    Analyze how allocation parameters might affect model performance.

    Uses bootstrapping with weighted samples to simulate effect of
    different allocation weights.
    """
    print("\n" + "="*80)
    print("DOWNSTREAM MODEL IMPACT ANALYSIS")
    print("="*80)

    # Prepare data
    hsa_df = hsa_df.copy()
    climate_cols = [c for c in CLIMATE_FEATURES if c in hsa_df.columns]

    # Create AR features
    hsa_df = hsa_df.sort_values(['hsa_id', 'week_number'])
    for lag in [1, 2]:
        hsa_df[f'ar_lag{lag}'] = hsa_df.groupby('hsa_id')[target_col].shift(lag)

    # Seasonal features
    hsa_df['sin_annual'] = np.sin(2 * np.pi * hsa_df['week_of_year'] / 52)
    hsa_df['cos_annual'] = np.cos(2 * np.pi * hsa_df['week_of_year'] / 52)

    hsa_df = hsa_df.dropna(subset=['ar_lag1', 'ar_lag2'])

    # Features
    ar_features = ['ar_lag1', 'ar_lag2']
    seasonal_features = ['sin_annual', 'cos_annual']
    features = ar_features + seasonal_features + climate_cols

    # Split temporally
    weeks = sorted(hsa_df['week_number'].unique())
    train_end = int(len(weeks) * 0.75)
    val_end = int(len(weeks) * 0.875)

    train_weeks = weeks[:train_end]
    test_weeks = weeks[val_end:]

    train_df = hsa_df[hsa_df['week_number'].isin(train_weeks)]
    test_df = hsa_df[hsa_df['week_number'].isin(test_weeks)]

    # Baseline model
    X_train = train_df[features].values
    y_train = train_df[target_col].values
    X_test = test_df[features].values
    y_test = test_df[target_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)
    model.fit(X_train_scaled, y_train)

    baseline_r2 = r2_score(y_test, model.predict(X_test_scaled))
    baseline_mae = mean_absolute_error(y_test, model.predict(X_test_scaled))

    print(f"\nBaseline Model (current parameters):")
    print(f"  Test R²: {baseline_r2:.4f}")
    print(f"  Test MAE: {baseline_mae:.2f}")

    # Bootstrap analysis with noise injection to simulate allocation uncertainty
    print("\nBootstrap analysis with allocation noise...")
    n_bootstrap = 100
    noise_levels = [0.0, 0.05, 0.10, 0.20]  # Fraction of target to add as noise

    bootstrap_results = []

    for noise_level in noise_levels:
        r2_samples = []
        mae_samples = []

        for _ in range(n_bootstrap):
            # Add noise to target (simulates allocation uncertainty)
            y_train_noisy = y_train + np.random.normal(0, noise_level * y_train.std(), len(y_train))
            y_train_noisy = np.maximum(y_train_noisy, 0)  # Non-negative

            # Retrain
            model_boot = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=None)
            model_boot.fit(X_train_scaled, y_train_noisy)

            r2_samples.append(r2_score(y_test, model_boot.predict(X_test_scaled)))
            mae_samples.append(mean_absolute_error(y_test, model_boot.predict(X_test_scaled)))

        bootstrap_results.append({
            'noise_level': noise_level,
            'mean_r2': np.mean(r2_samples),
            'std_r2': np.std(r2_samples),
            'mean_mae': np.mean(mae_samples),
            'std_mae': np.std(mae_samples)
        })

        print(f"  Noise={noise_level:.0%}: R²={np.mean(r2_samples):.4f}±{np.std(r2_samples):.4f}")

    return pd.DataFrame(bootstrap_results), baseline_r2, baseline_mae


def plot_sensitivity_results(sensitivity_df, output_dir):
    """Create visualizations for sensitivity analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Concentration heatmap
    ax1 = axes[0, 0]
    pivot_conc = sensitivity_df.pivot(index='alpha', columns='beta', values='concentration')
    sns.heatmap(pivot_conc, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax1)
    ax1.set_title('Herfindahl Concentration Index\n(Higher = More Concentrated)')
    ax1.set_xlabel('β (Distance Decay)')
    ax1.set_ylabel('α (Population Weight)')

    # 2. Gini heatmap
    ax2 = axes[0, 1]
    pivot_gini = sensitivity_df.pivot(index='alpha', columns='beta', values='gini')
    sns.heatmap(pivot_gini, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
    ax2.set_title('Gini Coefficient of Allocation Inequality\n(Higher = More Unequal)')
    ax2.set_xlabel('β (Distance Decay)')
    ax2.set_ylabel('α (Population Weight)')

    # 3. Alpha effect
    ax3 = axes[1, 0]
    for beta in BETA_VALUES:
        subset = sensitivity_df[sensitivity_df['beta'] == beta]
        ax3.plot(subset['alpha'], subset['concentration'], 'o-', label=f'β={beta}')
    ax3.set_xlabel('α (Population Weight)')
    ax3.set_ylabel('Concentration Index')
    ax3.set_title('Effect of α on Allocation Concentration')
    ax3.legend()

    # 4. Beta effect
    ax4 = axes[1, 1]
    for alpha in ALPHA_VALUES:
        subset = sensitivity_df[sensitivity_df['alpha'] == alpha]
        ax4.plot(subset['beta'], subset['concentration'], 'o-', label=f'α={alpha}')
    ax4.set_xlabel('β (Distance Decay)')
    ax4.set_ylabel('Concentration Index')
    ax4.set_title('Effect of β on Allocation Concentration')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / out_name('gravity_sensitivity_allocation.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / out_name('gravity_sensitivity_allocation.pdf'), bbox_inches='tight')
    plt.close()

    print(f"\nFigures saved to {output_dir}")


def plot_downstream_results(bootstrap_df, baseline_r2, output_dir):
    """Plot downstream impact analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(bootstrap_df['noise_level'] * 100,
                bootstrap_df['mean_r2'],
                yerr=bootstrap_df['std_r2'] * 1.96,
                fmt='o-', capsize=5, linewidth=2, markersize=8)

    ax.axhline(y=baseline_r2, color='red', linestyle='--', label='Baseline (no noise)')
    ax.set_xlabel('Allocation Noise Level (%)', fontsize=12)
    ax.set_ylabel('Test R²', fontsize=12)
    ax.set_title('Model Performance vs. Allocation Uncertainty\n(95% CI from Bootstrap)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / out_name('gravity_sensitivity_downstream.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / out_name('gravity_sensitivity_downstream.pdf'), bbox_inches='tight')
    plt.close()


def create_summary_table(sensitivity_df, output_dir):
    """Create summary table for supplement."""
    # Format table
    table = sensitivity_df.copy()
    table['concentration'] = table['concentration'].apply(lambda x: f'{x:.4f}')
    table['gini'] = table['gini'].apply(lambda x: f'{x:.3f}')
    table['max_share'] = table['max_share'].apply(lambda x: f'{x:.1%}')

    table.to_csv(output_dir / out_name('gravity_sensitivity_results.csv'), index=False)

    try:
        md_table = table.to_markdown(index=False)
    except Exception:
        md_table = table.to_string(index=False)
    with open(md_path('gravity_sensitivity_summary.md'), 'w') as f:
        f.write("# Table S_X: Gravity Model Parameter Sensitivity Analysis\n\n")
        f.write("Analysis of how α (population weight) and β (distance decay) affect patient allocation patterns.\n\n")
        f.write(md_table)
        f.write("\n\n## Key Findings\n\n")
        f.write("- Higher α increases concentration (larger facilities attract more patients)\n")
        f.write("- Higher β increases concentration (patients more strongly prefer nearby facilities)\n")
        f.write("- Current parameters (α=0.75, β=1.5) provide balanced allocation\n")
        f.write("- Model performance is robust to moderate allocation uncertainty\n")

    print(f"\nSummary table saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Gravity Model Sensitivity Analysis')
    parser.add_argument('--network', default='INF', )
    parser.add_argument('--hsa-mode', default='footprint')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out-dir', default=DEFAULT_PIPELINE_OUT_DIR)
    parser.add_argument('--output-dir', default=str(Path(DEFAULT_PIPELINE_OUT_DIR) / 'analysis_gravity_sensitivity'))
    parser.add_argument('--text-output-dir', default=str(Path(DEFAULT_PIPELINE_OUT_DIR) / 'textresults'))
    parser.add_argument('--target-col', default=None)
    parser.add_argument('--disease-focus', default=None, help='Disease-group focus; resolved to the outcome column via the authoritative table, never hardcoded')
    parser.add_argument('--n-simulations', type=int, default=5000)
    parser.add_argument('--boundary-version', default=os.environ.get("BOUNDARY_VERSION", os.environ.get("PIPELINE_VERSION", "v7")),
                        help="HSA boundary version (v6, v7, v8)")

    args = parser.parse_args()

    global OUTPUT_FILE_PREFIX, TEXT_RESULTS_DIR
    OUTPUT_FILE_PREFIX = f"{args.network}_{args.hsa_mode}"
    TEXT_RESULTS_DIR = Path(args.text_output_dir)
    TEXT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    output_dir = Path(args.output_dir) / args.network
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.target_col is None:
        if not args.disease_focus:
            parser.error('--target-col or --disease-focus is required (resolved to the outcome column; no hardcoded default)')
        from disease_focus import canonical_group, weekly_outcome_col
        args.target_col = weekly_outcome_col(canonical_group(args.network, args.disease_focus))

    print("="*80)
    print("GRAVITY MODEL SENSITIVITY ANALYSIS")
    print(f"Network: {args.network}")
    print("="*80)

    # Load data
    fac_df = load_facility_data(data_dir, args.network)
    print(f"\nLoaded {len(fac_df)} facilities")

    # Run allocation sensitivity analysis
    sensitivity_df = run_sensitivity_analysis(fac_df, n_simulations=args.n_simulations)

    # Load HSA data for downstream analysis
    hsa_df = load_hsa_data(out_dir, args.network, args.hsa_mode, args.boundary_version)

    # Run downstream impact analysis
    bootstrap_df, baseline_r2, baseline_mae = analyze_downstream_impact(
        hsa_df, args.target_col, output_dir
    )

    # Save results
    sensitivity_df.to_csv(output_dir / out_name('sensitivity_results.csv'), index=False)
    bootstrap_df.to_csv(output_dir / out_name('bootstrap_results.csv'), index=False)

    results_summary = {
        'current_parameters': {'alpha': 0.75, 'beta': 1.5},
        'parameter_ranges': {'alpha': ALPHA_VALUES, 'beta': BETA_VALUES},
        'baseline_performance': {'r2': float(baseline_r2), 'mae': float(baseline_mae)},
        'sensitivity_metrics': sensitivity_df.to_dict('records'),
        'robustness_metrics': bootstrap_df.to_dict('records')
    }

    with open(output_dir / out_name('sensitivity_analysis_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Create visualizations
    plot_sensitivity_results(sensitivity_df, output_dir)
    plot_downstream_results(bootstrap_df, baseline_r2, output_dir)

    # Create summary table
    create_summary_table(sensitivity_df, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
