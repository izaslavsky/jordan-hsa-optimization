#!/usr/bin/env python3
"""
Within-HSA Climate Heterogeneity Analysis
==========================================

This script analyzes climate variance within vs between HSAs to validate
that HSAs are climatically coherent units for modeling.

Key Questions:
1. How much climate variance exists WITHIN each HSA?
2. How does within-HSA variance compare to between-HSA variance?
3. Should we switch to population-weighted climate aggregation?

Approach (without additional GEE queries):
- Use pixel allocation data (lon, lat, population) for each HSA
- Estimate climate at each pixel using:
  * Elevation-based temperature lapse rate
  * Latitude gradient
  * Existing climate data from HSA centroids
- Compute within-HSA variance metrics

Author: HSA Research Team
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
from pathlib import Path
import json
from scipy import stats

warnings.filterwarnings('ignore')
OUTPUT_FILE_PREFIX = ""
TEXT_RESULTS_DIR = None


def out_name(filename: str) -> str:
    return f"{OUTPUT_FILE_PREFIX}_{filename}" if OUTPUT_FILE_PREFIX else filename


def md_path(filename: str) -> Path:
    return TEXT_RESULTS_DIR / out_name(filename) if TEXT_RESULTS_DIR else Path(filename)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Temperature lapse rate (°C per 100m elevation gain)
LAPSE_RATE = 0.65  # Standard environmental lapse rate

# Latitude temperature gradient (°C per degree latitude, decreases going north)
LATITUDE_GRADIENT = 0.5  # Approximate for Jordan

# Reference values for Jordan
REFERENCE_ELEVATION_M = 800  # Approximate mean elevation
REFERENCE_TEMP_C = 18  # Approximate mean annual temperature


def load_allocation_data(out_dir, network, hsa_mode, sample_size=500000):
    """Load pixel allocation data with coordinates."""
    alloc_file = out_dir / f'pixel_allocations_{network}_{hsa_mode}.csv'

    if not alloc_file.exists():
        print(f"  Allocation file not found: {alloc_file}")
        return None

    print(f"  Loading pixel allocations...")

    # Sample if file is large
    df = pd.read_csv(alloc_file, nrows=sample_size)
    print(f"    Loaded {len(df):,} pixels")

    return df


def load_elevation_data(data_dir):
    """Load elevation data if available."""
    # Try to find elevation raster
    elev_files = list(data_dir.glob('*elevation*.tif')) + list(data_dir.glob('*dem*.tif'))

    if elev_files:
        print(f"  Found elevation file: {elev_files[0]}")
        # Would use rasterio to read elevation
        return None  # For now, return None and use estimation

    return None


def estimate_elevation(lat, lon):
    """
    Estimate elevation based on location in Jordan.

    Jordan's topography:
    - Jordan Valley: ~-400m (Dead Sea area around lon=35.5, lat=31.5)
    - Central highlands: ~900-1200m (lon=35.8-36.2, lat=30-32)
    - Eastern desert: ~600-800m (lon>36.5)
    """
    # Simple elevation model based on longitude and latitude

    # Distance from Dead Sea (lowest point)
    dead_sea_lon = 35.55
    dead_sea_lat = 31.5

    # Base elevation increases with distance from Dead Sea
    dist_from_dead_sea = np.sqrt((lon - dead_sea_lon)**2 + (lat - dead_sea_lat)**2)

    # Elevation gradient: ~500m per degree away from Dead Sea
    elevation = -400 + dist_from_dead_sea * 500

    # Cap at realistic values for Jordan
    elevation = np.clip(elevation, -400, 1200)

    # Add some east-west gradient (higher in central highlands)
    if lon < 36.5:
        elevation += 200 * (1 - abs(lon - 35.9) / 1.0)

    return elevation


def estimate_temperature(lat, lon, elevation=None):
    """
    Estimate mean annual temperature based on location.

    Factors:
    1. Elevation (lapse rate)
    2. Latitude (cooler in north)
    3. Distance from sea (continentality)
    """
    if elevation is None:
        elevation = estimate_elevation(lat, lon)

    # Base temperature at reference point
    base_temp = REFERENCE_TEMP_C

    # Elevation adjustment (lapse rate)
    elevation_diff = elevation - REFERENCE_ELEVATION_M
    temp_elev_adj = -LAPSE_RATE * (elevation_diff / 100)

    # Latitude adjustment (cooler in north)
    # Reference latitude: ~31°N (center of Jordan)
    lat_diff = lat - 31.0
    temp_lat_adj = -LATITUDE_GRADIENT * lat_diff

    # Final temperature
    temp = base_temp + temp_elev_adj + temp_lat_adj

    return temp


def calculate_within_hsa_variance(allocations):
    """Calculate climate variance within each HSA."""

    print("\nCalculating within-HSA climate variance...")

    # Estimate elevation and temperature for each pixel
    allocations = allocations.copy()
    allocations['estimated_elevation'] = allocations.apply(
        lambda row: estimate_elevation(row['lat'], row['lon']), axis=1
    )
    allocations['estimated_temp'] = allocations.apply(
        lambda row: estimate_temperature(row['lat'], row['lon'], row['estimated_elevation']),
        axis=1
    )

    # Calculate within-HSA statistics
    within_stats = allocations.groupby('facility_id').agg({
        'estimated_temp': ['mean', 'std', 'min', 'max', 'count'],
        'estimated_elevation': ['mean', 'std', 'min', 'max'],
        'population': 'sum',
        'lat': ['min', 'max'],
        'lon': ['min', 'max']
    }).reset_index()

    # Flatten column names
    within_stats.columns = ['_'.join(col).strip('_') for col in within_stats.columns.values]

    # Rename for clarity
    within_stats = within_stats.rename(columns={
        'facility_id': 'hsa_id',
        'estimated_temp_mean': 'temp_mean',
        'estimated_temp_std': 'temp_within_std',
        'estimated_temp_min': 'temp_min',
        'estimated_temp_max': 'temp_max',
        'estimated_temp_count': 'n_pixels',
        'estimated_elevation_mean': 'elev_mean',
        'estimated_elevation_std': 'elev_within_std',
        'estimated_elevation_min': 'elev_min',
        'estimated_elevation_max': 'elev_max',
        'population_sum': 'total_population',
        'lat_min': 'lat_south',
        'lat_max': 'lat_north',
        'lon_min': 'lon_west',
        'lon_max': 'lon_east'
    })

    # Calculate spatial extent
    within_stats['lat_range'] = within_stats['lat_north'] - within_stats['lat_south']
    within_stats['lon_range'] = within_stats['lon_east'] - within_stats['lon_west']
    within_stats['temp_range'] = within_stats['temp_max'] - within_stats['temp_min']
    within_stats['elev_range'] = within_stats['elev_max'] - within_stats['elev_min']

    return within_stats, allocations


def calculate_between_hsa_variance(within_stats):
    """Calculate variance between HSAs."""

    print("\nCalculating between-HSA climate variance...")

    # Between-HSA variance is the variance of HSA means
    between_stats = {
        'temp_between_std': within_stats['temp_mean'].std(),
        'temp_between_range': within_stats['temp_mean'].max() - within_stats['temp_mean'].min(),
        'elev_between_std': within_stats['elev_mean'].std(),
        'elev_between_range': within_stats['elev_mean'].max() - within_stats['elev_mean'].min(),
        'n_hsas': len(within_stats)
    }

    return between_stats


def calculate_variance_decomposition(within_stats, between_stats):
    """
    Decompose total variance into within and between components.

    Total Variance = Within-Group Variance + Between-Group Variance

    ICC (Intraclass Correlation) = Between / Total
    - ICC close to 1: HSAs are climatically distinct
    - ICC close to 0: HSAs have similar climate (within >> between)
    """

    # Mean within-HSA variance (weighted by n_pixels)
    weights = within_stats['n_pixels'].values
    within_var_temp = np.average(within_stats['temp_within_std'].fillna(0).values ** 2, weights=weights)
    within_var_elev = np.average(within_stats['elev_within_std'].fillna(0).values ** 2, weights=weights)

    # Between-HSA variance
    between_var_temp = between_stats['temp_between_std'] ** 2
    between_var_elev = between_stats['elev_between_std'] ** 2

    # Total variance
    total_var_temp = within_var_temp + between_var_temp
    total_var_elev = within_var_elev + between_var_elev

    # ICC (proportion of variance between HSAs)
    icc_temp = between_var_temp / total_var_temp if total_var_temp > 0 else 0
    icc_elev = between_var_elev / total_var_elev if total_var_elev > 0 else 0

    decomposition = {
        'temperature': {
            'within_variance': float(within_var_temp),
            'between_variance': float(between_var_temp),
            'total_variance': float(total_var_temp),
            'within_std': float(np.sqrt(within_var_temp)),
            'between_std': float(np.sqrt(between_var_temp)),
            'icc': float(icc_temp),
            'pct_within': float(100 * within_var_temp / total_var_temp) if total_var_temp > 0 else 0,
            'pct_between': float(100 * between_var_temp / total_var_temp) if total_var_temp > 0 else 0,
        },
        'elevation': {
            'within_variance': float(within_var_elev),
            'between_variance': float(between_var_elev),
            'total_variance': float(total_var_elev),
            'within_std': float(np.sqrt(within_var_elev)),
            'between_std': float(np.sqrt(between_var_elev)),
            'icc': float(icc_elev),
            'pct_within': float(100 * within_var_elev / total_var_elev) if total_var_elev > 0 else 0,
            'pct_between': float(100 * between_var_elev / total_var_elev) if total_var_elev > 0 else 0,
        }
    }

    return decomposition


def compare_weighting_schemes(allocations):
    """
    Compare simple mean vs population-weighted mean for climate aggregation.

    If population-weighted mean differs significantly from simple mean,
    it suggests exposure misclassification risk.
    """

    print("\nComparing weighting schemes...")

    # Calculate both means for each HSA
    comparison = []

    for hsa_id in allocations['facility_id'].unique():
        hsa_data = allocations[allocations['facility_id'] == hsa_id]

        simple_mean_temp = hsa_data['estimated_temp'].mean()
        pop_weighted_temp = np.average(hsa_data['estimated_temp'],
                                       weights=hsa_data['population'])

        simple_mean_elev = hsa_data['estimated_elevation'].mean()
        pop_weighted_elev = np.average(hsa_data['estimated_elevation'],
                                       weights=hsa_data['population'])

        comparison.append({
            'hsa_id': hsa_id,
            'temp_simple_mean': simple_mean_temp,
            'temp_pop_weighted': pop_weighted_temp,
            'temp_difference': pop_weighted_temp - simple_mean_temp,
            'elev_simple_mean': simple_mean_elev,
            'elev_pop_weighted': pop_weighted_elev,
            'elev_difference': pop_weighted_elev - simple_mean_elev,
            'total_population': hsa_data['population'].sum()
        })

    comparison_df = pd.DataFrame(comparison)

    # Statistics
    stats_dict = {
        'temp_mean_abs_difference': float(comparison_df['temp_difference'].abs().mean()),
        'temp_max_abs_difference': float(comparison_df['temp_difference'].abs().max()),
        'temp_correlation': float(comparison_df['temp_simple_mean'].corr(comparison_df['temp_pop_weighted'])),
        'elev_mean_abs_difference': float(comparison_df['elev_difference'].abs().mean()),
        'elev_max_abs_difference': float(comparison_df['elev_difference'].abs().max()),
        'elev_correlation': float(comparison_df['elev_simple_mean'].corr(comparison_df['elev_pop_weighted'])),
    }

    return comparison_df, stats_dict


def create_heterogeneity_plots(within_stats, allocations, decomposition, comparison_df, output_dir):
    """Create visualization plots."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Within-HSA temperature standard deviation
    ax1 = axes[0, 0]
    ax1.bar(range(len(within_stats)), within_stats['temp_within_std'].fillna(0).sort_values())
    ax1.set_xlabel('HSA (sorted)')
    ax1.set_ylabel('Within-HSA Temp Std (°C)')
    ax1.set_title('Temperature Heterogeneity Within Each HSA')
    ax1.axhline(within_stats['temp_within_std'].mean(), color='red', linestyle='--',
                label=f'Mean: {within_stats["temp_within_std"].mean():.2f}°C')
    ax1.legend()

    # 2. Variance decomposition
    ax2 = axes[0, 1]
    temp_decomp = decomposition['temperature']
    bars = ax2.bar(['Within HSA', 'Between HSA'],
                   [temp_decomp['pct_within'], temp_decomp['pct_between']],
                   color=['steelblue', 'coral'])
    ax2.set_ylabel('% of Total Variance')
    ax2.set_title(f'Temperature Variance Decomposition\nICC = {temp_decomp["icc"]:.3f}')
    ax2.set_ylim(0, 100)
    for bar, val in zip(bars, [temp_decomp['pct_within'], temp_decomp['pct_between']]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%',
                ha='center', fontsize=12)

    # 3. HSA mean temperature vs within-HSA variance
    ax3 = axes[0, 2]
    scatter = ax3.scatter(within_stats['temp_mean'], within_stats['temp_within_std'].fillna(0),
                         s=within_stats['total_population']/1000, alpha=0.6)
    ax3.set_xlabel('HSA Mean Temperature (°C)')
    ax3.set_ylabel('Within-HSA Temperature Std (°C)')
    ax3.set_title('HSA Climate: Mean vs Heterogeneity\n(size = population)')

    # 4. Simple vs population-weighted means
    ax4 = axes[1, 0]
    ax4.scatter(comparison_df['temp_simple_mean'], comparison_df['temp_pop_weighted'], alpha=0.7)
    ax4.plot([comparison_df['temp_simple_mean'].min(), comparison_df['temp_simple_mean'].max()],
             [comparison_df['temp_simple_mean'].min(), comparison_df['temp_simple_mean'].max()],
             'r--', label='1:1 line')
    ax4.set_xlabel('Simple Mean Temperature (°C)')
    ax4.set_ylabel('Population-Weighted Mean (°C)')
    ax4.set_title('Simple vs Population-Weighted Aggregation')
    ax4.legend()

    # 5. Distribution of weighting differences
    ax5 = axes[1, 1]
    ax5.hist(comparison_df['temp_difference'], bins=15, edgecolor='black', alpha=0.7)
    ax5.axvline(0, color='red', linestyle='--')
    ax5.set_xlabel('Temperature Difference (Pop-Weighted - Simple)')
    ax5.set_ylabel('Number of HSAs')
    ax5.set_title(f'Impact of Population Weighting\nMean Abs Diff: {comparison_df["temp_difference"].abs().mean():.2f}°C')

    # 6. Elevation heterogeneity
    ax6 = axes[1, 2]
    elev_decomp = decomposition['elevation']
    bars = ax6.bar(['Within HSA', 'Between HSA'],
                   [elev_decomp['pct_within'], elev_decomp['pct_between']],
                   color=['steelblue', 'coral'])
    ax6.set_ylabel('% of Total Variance')
    ax6.set_title(f'Elevation Variance Decomposition\nICC = {elev_decomp["icc"]:.3f}')
    ax6.set_ylim(0, 100)
    for bar, val in zip(bars, [elev_decomp['pct_within'], elev_decomp['pct_between']]):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%',
                ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / out_name('within_hsa_heterogeneity.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / out_name('within_hsa_heterogeneity.pdf'), bbox_inches='tight')
    plt.close()

    print(f"  Figures saved to {output_dir}")


def generate_report(within_stats, between_stats, decomposition, comparison_stats, output_dir):
    """Generate markdown report."""

    report = []
    report.append("# Within-HSA Climate Heterogeneity Analysis\n")

    report.append("## Summary\n")
    report.append("This analysis examines climate variance within vs between Hospital Service Areas (HSAs)")
    report.append("to validate that HSAs are climatically coherent units for modeling.\n")

    report.append("## Variance Decomposition\n")
    report.append("### Temperature\n")
    temp = decomposition['temperature']
    report.append(f"| Component | Variance | Std Dev | % of Total |")
    report.append(f"|-----------|----------|---------|------------|")
    report.append(f"| Within HSA | {temp['within_variance']:.4f} | {temp['within_std']:.2f}°C | {temp['pct_within']:.1f}% |")
    report.append(f"| Between HSA | {temp['between_variance']:.4f} | {temp['between_std']:.2f}°C | {temp['pct_between']:.1f}% |")
    report.append(f"| **ICC** | | | **{temp['icc']:.3f}** |")
    report.append("")

    report.append("### Elevation\n")
    elev = decomposition['elevation']
    report.append(f"| Component | Variance | Std Dev | % of Total |")
    report.append(f"|-----------|----------|---------|------------|")
    report.append(f"| Within HSA | {elev['within_variance']:.1f} | {elev['within_std']:.0f}m | {elev['pct_within']:.1f}% |")
    report.append(f"| Between HSA | {elev['between_variance']:.1f} | {elev['between_std']:.0f}m | {elev['pct_between']:.1f}% |")
    report.append(f"| **ICC** | | | **{elev['icc']:.3f}** |")
    report.append("")

    report.append("## Population Weighting Impact\n")
    report.append(f"- Mean absolute temperature difference: {comparison_stats['temp_mean_abs_difference']:.2f}°C")
    report.append(f"- Max absolute temperature difference: {comparison_stats['temp_max_abs_difference']:.2f}°C")
    report.append(f"- Correlation (simple vs weighted): {comparison_stats['temp_correlation']:.4f}")
    report.append("")

    report.append("## Interpretation\n")

    # ICC interpretation
    temp_icc = temp['icc']
    if temp_icc > 0.5:
        report.append(f"- **Temperature ICC = {temp_icc:.3f}**: HSAs are climatically DISTINCT")
        report.append("  - More than half the temperature variance is between HSAs")
        report.append("  - HSAs capture real climate differences")
    elif temp_icc > 0.2:
        report.append(f"- **Temperature ICC = {temp_icc:.3f}**: HSAs show MODERATE climate distinction")
        report.append("  - Some climate variance between HSAs, but substantial within-HSA heterogeneity")
    else:
        report.append(f"- **Temperature ICC = {temp_icc:.3f}**: HSAs are climatically SIMILAR")
        report.append("  - Most variance is within HSAs, not between them")
        report.append("  - Climate effects may be diluted by spatial averaging")

    # Weighting recommendation
    if comparison_stats['temp_mean_abs_difference'] > 0.5:
        report.append("\n- **Recommendation**: Use population-weighted climate aggregation")
        report.append(f"  - Differences of up to {comparison_stats['temp_max_abs_difference']:.1f}°C observed")
    else:
        report.append("\n- **Finding**: Population weighting has minimal impact")
        report.append("  - Simple averaging is sufficient for this spatial resolution")

    report.append("")

    with open(md_path('heterogeneity_report.md'), 'w') as f:
        f.write('\n'.join(report))

    print(f"  Report saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Within-HSA Climate Heterogeneity Analysis')
    parser.add_argument('--network', default='INF', choices=['INF', 'NCD'])
    parser.add_argument('--hsa-mode', default='footprint')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out-dir', default='out')
    parser.add_argument('--output-dir', default='out/analysis_climate_heterogeneity')
    parser.add_argument('--text-output-dir', default='out/textresults')
    parser.add_argument('--sample-size', type=int, default=500000)

    args = parser.parse_args()

    global OUTPUT_FILE_PREFIX, TEXT_RESULTS_DIR
    OUTPUT_FILE_PREFIX = f"{args.network}_{args.hsa_mode}"
    TEXT_RESULTS_DIR = Path(args.text_output_dir)
    TEXT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    output_dir = Path(args.output_dir) / args.network
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WITHIN-HSA CLIMATE HETEROGENEITY ANALYSIS")
    print("=" * 70)

    # Load allocation data
    allocations = load_allocation_data(out_dir, args.network, args.hsa_mode, args.sample_size)

    if allocations is None:
        print("Error: Could not load allocation data")
        return

    # Calculate within-HSA variance
    within_stats, allocations = calculate_within_hsa_variance(allocations)

    # Calculate between-HSA variance
    between_stats = calculate_between_hsa_variance(within_stats)

    # Decompose variance
    decomposition = calculate_variance_decomposition(within_stats, between_stats)

    print("\nVariance Decomposition:")
    print(f"  Temperature:")
    print(f"    Within-HSA std:  {decomposition['temperature']['within_std']:.2f}°C ({decomposition['temperature']['pct_within']:.1f}%)")
    print(f"    Between-HSA std: {decomposition['temperature']['between_std']:.2f}°C ({decomposition['temperature']['pct_between']:.1f}%)")
    print(f"    ICC: {decomposition['temperature']['icc']:.3f}")

    print(f"  Elevation:")
    print(f"    Within-HSA std:  {decomposition['elevation']['within_std']:.0f}m ({decomposition['elevation']['pct_within']:.1f}%)")
    print(f"    Between-HSA std: {decomposition['elevation']['between_std']:.0f}m ({decomposition['elevation']['pct_between']:.1f}%)")
    print(f"    ICC: {decomposition['elevation']['icc']:.3f}")

    # Compare weighting schemes
    comparison_df, comparison_stats = compare_weighting_schemes(allocations)

    print(f"\nPopulation Weighting Impact:")
    print(f"  Mean temperature difference: {comparison_stats['temp_mean_abs_difference']:.2f}°C")
    print(f"  Max temperature difference: {comparison_stats['temp_max_abs_difference']:.2f}°C")

    # Save results
    within_stats.to_csv(output_dir / out_name('within_hsa_statistics.csv'), index=False)
    comparison_df.to_csv(output_dir / out_name('weighting_comparison.csv'), index=False)

    results = {
        'n_hsas': int(len(within_stats)),
        'n_pixels_analyzed': int(len(allocations)),
        'between_hsa_stats': between_stats,
        'variance_decomposition': decomposition,
        'weighting_comparison': comparison_stats
    }

    with open(output_dir / out_name('heterogeneity_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Create plots
    create_heterogeneity_plots(within_stats, allocations, decomposition, comparison_df, output_dir)

    # Generate report
    generate_report(within_stats, between_stats, decomposition, comparison_stats, output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
