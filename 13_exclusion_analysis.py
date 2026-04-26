#!/usr/bin/env python3
"""
Phase 1A: Characterize Excluded Populations

This script analyzes which populations are poorly connected to health facilities
by examining allocation probability and distance to nearest facility.

Key questions:
1. WHO was excluded/weakly allocated?
2. WHERE are they located?
3. What are their climate characteristics?
4. How do they compare to well-connected populations?
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import argparse
from pathlib import Path
import json
import warnings
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')
OUTPUT_FILE_PREFIX = ""
TEXT_RESULTS_DIR = None


def out_name(filename: str) -> str:
    return f"{OUTPUT_FILE_PREFIX}_{filename}" if OUTPUT_FILE_PREFIX else filename


def md_path(filename: str) -> Path:
    return TEXT_RESULTS_DIR / out_name(filename) if TEXT_RESULTS_DIR else Path(filename)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
NETWORK = "INF"
HSA_MODE = "footprint"
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"
ANALYSIS_DIR = OUT_DIR / "analysis_exclusion"

# Analysis parameters
PROBABILITY_THRESHOLDS = {
    'strongly_connected': 0.5,    # High confidence allocation
    'moderately_connected': 0.2,  # Moderate allocation
    'weakly_connected': 0.1,      # Weak allocation
    'very_weakly_connected': 0.05 # Very weak - effectively excluded
}

DISTANCE_THRESHOLDS_KM = {
    'urban': 5,
    'peri_urban': 15,
    'rural': 30,
    'remote': 50,
    'very_remote': 100
}


class GeoUtils:
    """Geographic utility functions."""

    @staticmethod
    def haversine_km(lon1, lat1, lon2, lat2):
        """Calculate haversine distance in km (vectorized)."""
        R = 6371  # Earth radius in km

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    """Load allocation data, facility data, and boundary."""

    print("Loading data...")

    # Load pixel allocations
    alloc_path = OUT_DIR / f"pixel_allocations_{NETWORK}_{HSA_MODE}.csv"
    print(f"  Loading pixel allocations from {alloc_path}")
    allocations = pd.read_csv(alloc_path)
    print(f"    Loaded {len(allocations):,} allocated pixels")
    print(f"    Total population: {allocations['population'].sum():,.0f}")

    # Load facility coordinates
    fac_path = DATA_DIR / f"{NETWORK}_facility_coordinates.csv"
    print(f"  Loading facilities from {fac_path}")
    facilities = pd.read_csv(fac_path)

    # Normalize column names
    col_map = {c: c.lower() for c in facilities.columns}
    facilities = facilities.rename(columns=col_map)

    # Ensure lat/lon columns exist
    if 'latitude' in facilities.columns and 'longitude' in facilities.columns:
        facilities = facilities.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

    print(f"    Loaded {len(facilities)} facilities")

    # Load Jordan boundary
    boundary_path = DATA_DIR / "jordan_boundary.gpkg"
    boundary = gpd.read_file(boundary_path)
    print(f"  Loaded boundary with {len(boundary)} features")

    # Try to load governorates
    gov_path = DATA_DIR / "jordan_governorates.gpkg"
    if gov_path.exists():
        governorates = gpd.read_file(gov_path)
        print(f"  Loaded {len(governorates)} governorates")
    else:
        governorates = None

    return allocations, facilities, boundary, governorates


def calculate_distance_to_nearest_facility(allocations: pd.DataFrame,
                                           facilities: pd.DataFrame) -> pd.DataFrame:
    """Calculate distance from each pixel to the nearest facility."""

    print("\nCalculating distance to nearest facility...")

    # Build KD-tree for facilities
    fac_coords = facilities[['lon', 'lat']].values
    kdtree = cKDTree(fac_coords)

    # Query nearest facility for each pixel
    pixel_coords = allocations[['lon', 'lat']].values

    # Get nearest facility index (approximate using KD-tree, then calculate exact distance)
    _, nearest_idx = kdtree.query(pixel_coords, k=1)

    # Calculate exact haversine distance
    nearest_fac = facilities.iloc[nearest_idx]

    distances = GeoUtils.haversine_km(
        allocations['lon'].values,
        allocations['lat'].values,
        nearest_fac['lon'].values,
        nearest_fac['lat'].values
    )

    allocations = allocations.copy()
    allocations['distance_to_nearest_km'] = distances

    print(f"  Distance range: {distances.min():.1f} - {distances.max():.1f} km")
    print(f"  Mean distance: {distances.mean():.1f} km")
    print(f"  Median distance: {np.median(distances):.1f} km")

    return allocations


def classify_connectivity(allocations: pd.DataFrame) -> pd.DataFrame:
    """Classify pixels by connectivity level based on probability and distance."""

    print("\nClassifying connectivity levels...")

    allocations = allocations.copy()

    # Probability-based classification
    def prob_class(p):
        if p >= PROBABILITY_THRESHOLDS['strongly_connected']:
            return 'strong'
        elif p >= PROBABILITY_THRESHOLDS['moderately_connected']:
            return 'moderate'
        elif p >= PROBABILITY_THRESHOLDS['weakly_connected']:
            return 'weak'
        else:
            return 'very_weak'

    allocations['prob_class'] = allocations['probability'].apply(prob_class)

    # Distance-based classification
    def dist_class(d):
        if d <= DISTANCE_THRESHOLDS_KM['urban']:
            return 'urban'
        elif d <= DISTANCE_THRESHOLDS_KM['peri_urban']:
            return 'peri_urban'
        elif d <= DISTANCE_THRESHOLDS_KM['rural']:
            return 'rural'
        elif d <= DISTANCE_THRESHOLDS_KM['remote']:
            return 'remote'
        else:
            return 'very_remote'

    allocations['dist_class'] = allocations['distance_to_nearest_km'].apply(dist_class)

    # Combined classification
    allocations['connectivity'] = allocations.apply(
        lambda x: f"{x['prob_class']}_{x['dist_class']}", axis=1
    )

    # Print summary
    print("\n  By probability class:")
    for cls in ['strong', 'moderate', 'weak', 'very_weak']:
        mask = allocations['prob_class'] == cls
        pop = allocations.loc[mask, 'population'].sum()
        pct = 100 * pop / allocations['population'].sum()
        print(f"    {cls:12s}: {pop:>12,.0f} ({pct:5.1f}%)")

    print("\n  By distance class:")
    for cls in ['urban', 'peri_urban', 'rural', 'remote', 'very_remote']:
        mask = allocations['dist_class'] == cls
        pop = allocations.loc[mask, 'population'].sum()
        pct = 100 * pop / allocations['population'].sum()
        print(f"    {cls:12s}: {pop:>12,.0f} ({pct:5.1f}%)")

    return allocations


def create_exclusion_summary(allocations: pd.DataFrame) -> Dict:
    """Create comprehensive summary statistics of exclusion patterns."""

    print("\nCreating exclusion summary...")

    total_pop = allocations['population'].sum()

    summary = {
        'total_pixels': len(allocations),
        'total_population': float(total_pop),
        'probability_distribution': {
            'mean': float(allocations['probability'].mean()),
            'median': float(allocations['probability'].median()),
            'std': float(allocations['probability'].std()),
            'p10': float(allocations['probability'].quantile(0.1)),
            'p25': float(allocations['probability'].quantile(0.25)),
            'p75': float(allocations['probability'].quantile(0.75)),
            'p90': float(allocations['probability'].quantile(0.9)),
        },
        'distance_distribution': {
            'mean': float(allocations['distance_to_nearest_km'].mean()),
            'median': float(allocations['distance_to_nearest_km'].median()),
            'std': float(allocations['distance_to_nearest_km'].std()),
            'p10': float(allocations['distance_to_nearest_km'].quantile(0.1)),
            'p25': float(allocations['distance_to_nearest_km'].quantile(0.25)),
            'p75': float(allocations['distance_to_nearest_km'].quantile(0.75)),
            'p90': float(allocations['distance_to_nearest_km'].quantile(0.9)),
        },
        'by_probability_class': {},
        'by_distance_class': {},
        'thresholds_used': {
            'probability': PROBABILITY_THRESHOLDS,
            'distance_km': DISTANCE_THRESHOLDS_KM
        }
    }

    # Summarize by probability class
    for cls in ['strong', 'moderate', 'weak', 'very_weak']:
        mask = allocations['prob_class'] == cls
        subset = allocations[mask]
        summary['by_probability_class'][cls] = {
            'pixel_count': int(len(subset)),
            'population': float(subset['population'].sum()),
            'population_pct': float(100 * subset['population'].sum() / total_pop),
            'mean_distance_km': float(subset['distance_to_nearest_km'].mean()) if len(subset) > 0 else None,
            'mean_probability': float(subset['probability'].mean()) if len(subset) > 0 else None,
        }

    # Summarize by distance class
    for cls in ['urban', 'peri_urban', 'rural', 'remote', 'very_remote']:
        mask = allocations['dist_class'] == cls
        subset = allocations[mask]
        summary['by_distance_class'][cls] = {
            'pixel_count': int(len(subset)),
            'population': float(subset['population'].sum()),
            'population_pct': float(100 * subset['population'].sum() / total_pop),
            'mean_distance_km': float(subset['distance_to_nearest_km'].mean()) if len(subset) > 0 else None,
            'mean_probability': float(subset['probability'].mean()) if len(subset) > 0 else None,
        }

    return summary


def identify_exclusion_hotspots(allocations: pd.DataFrame,
                                governorates: Optional[gpd.GeoDataFrame]) -> pd.DataFrame:
    """Identify geographic hotspots of exclusion."""

    print("\nIdentifying exclusion hotspots...")

    # Define "excluded" as very weak probability AND remote distance
    excluded_mask = (allocations['prob_class'].isin(['weak', 'very_weak'])) & \
                    (allocations['dist_class'].isin(['rural', 'remote', 'very_remote']))

    excluded = allocations[excluded_mask].copy()
    excluded_pop = excluded['population'].sum()
    total_pop = allocations['population'].sum()

    print(f"  Functionally excluded: {excluded_pop:,.0f} ({100*excluded_pop/total_pop:.1f}%)")

    if governorates is not None:
        # Create GeoDataFrame of excluded pixels
        excluded_gdf = gpd.GeoDataFrame(
            excluded,
            geometry=gpd.points_from_xy(excluded.lon, excluded.lat),
            crs="EPSG:4326"
        )

        # Spatial join to get governorate
        excluded_with_gov = gpd.sjoin(excluded_gdf, governorates, how='left', predicate='within')

        # Check for governorate column names
        gov_col = None
        for col in ['gov_name', 'name', 'NAME', 'Governorate', 'governorate', 'NAME_1']:
            if col in excluded_with_gov.columns:
                gov_col = col
                break

        if gov_col:
            hotspots = excluded_with_gov.groupby(gov_col).agg({
                'population': 'sum',
                'distance_to_nearest_km': 'mean',
                'probability': 'mean'
            }).reset_index()
            hotspots.columns = ['governorate', 'excluded_population', 'mean_distance', 'mean_probability']
            hotspots = hotspots.sort_values('excluded_population', ascending=False)

            print("\n  Top exclusion hotspots by governorate:")
            for _, row in hotspots.head(5).iterrows():
                print(f"    {row['governorate']}: {row['excluded_population']:,.0f} people, "
                      f"mean dist: {row['mean_distance']:.1f} km")

            return hotspots

    # Fallback: grid-based hotspots
    # Bin into 0.1 degree cells
    excluded['lat_bin'] = (excluded['lat'] / 0.1).round() * 0.1
    excluded['lon_bin'] = (excluded['lon'] / 0.1).round() * 0.1

    hotspots = excluded.groupby(['lat_bin', 'lon_bin']).agg({
        'population': 'sum',
        'distance_to_nearest_km': 'mean',
        'probability': 'mean'
    }).reset_index()
    hotspots.columns = ['lat', 'lon', 'excluded_population', 'mean_distance', 'mean_probability']
    hotspots = hotspots.sort_values('excluded_population', ascending=False)

    print("\n  Top exclusion hotspots by grid cell:")
    for _, row in hotspots.head(5).iterrows():
        print(f"    ({row['lat']:.1f}, {row['lon']:.1f}): {row['excluded_population']:,.0f} people")

    return hotspots


def create_comparison_table(allocations: pd.DataFrame) -> pd.DataFrame:
    """Create comparison table between connected and excluded populations."""

    print("\nCreating comparison table...")

    # Define groups
    well_connected = allocations[
        (allocations['prob_class'].isin(['strong', 'moderate'])) |
        (allocations['dist_class'].isin(['urban', 'peri_urban']))
    ]

    poorly_connected = allocations[
        (allocations['prob_class'].isin(['weak', 'very_weak'])) &
        (allocations['dist_class'].isin(['rural', 'remote', 'very_remote']))
    ]

    def calc_stats(df, label):
        return {
            'Group': label,
            'Population': df['population'].sum(),
            'Pixels': len(df),
            'Mean Probability': df['probability'].mean(),
            'Mean Distance (km)': df['distance_to_nearest_km'].mean(),
            'Median Distance (km)': df['distance_to_nearest_km'].median(),
            'P90 Distance (km)': df['distance_to_nearest_km'].quantile(0.9),
        }

    comparison = pd.DataFrame([
        calc_stats(well_connected, 'Well Connected'),
        calc_stats(poorly_connected, 'Poorly Connected'),
        calc_stats(allocations, 'All Population'),
    ])

    # Calculate percentages
    total_pop = allocations['population'].sum()
    comparison['Population %'] = 100 * comparison['Population'] / total_pop

    return comparison


def create_exclusion_map(allocations: pd.DataFrame,
                         boundary: gpd.GeoDataFrame,
                         governorates: Optional[gpd.GeoDataFrame],
                         output_path: Path):
    """Create map showing exclusion patterns."""

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  Skipping map (matplotlib not available)")
        return

    print("\nCreating exclusion map...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Map 1: Probability
    ax1 = axes[0]
    ax1.set_title('Allocation Probability', fontsize=12)

    # Plot boundary
    boundary.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=1)

    if governorates is not None:
        governorates.plot(ax=ax1, facecolor='none', edgecolor='gray', linewidth=0.5, alpha=0.5)

    # Sample pixels for visualization (too many to plot all)
    sample = allocations.sample(min(50000, len(allocations)), random_state=42)

    scatter1 = ax1.scatter(
        sample['lon'], sample['lat'],
        c=sample['probability'],
        s=0.5, alpha=0.5,
        cmap='RdYlGn',
        vmin=0, vmax=0.5
    )
    plt.colorbar(scatter1, ax=ax1, label='Probability')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    # Map 2: Distance to nearest facility
    ax2 = axes[1]
    ax2.set_title('Distance to Nearest Facility (km)', fontsize=12)

    boundary.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=1)

    if governorates is not None:
        governorates.plot(ax=ax2, facecolor='none', edgecolor='gray', linewidth=0.5, alpha=0.5)

    scatter2 = ax2.scatter(
        sample['lon'], sample['lat'],
        c=sample['distance_to_nearest_km'],
        s=0.5, alpha=0.5,
        cmap='YlOrRd',
        vmin=0, vmax=50
    )
    plt.colorbar(scatter2, ax=ax2, label='Distance (km)')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved exclusion map to {output_path}")


def generate_markdown_report(summary: Dict,
                            comparison: pd.DataFrame,
                            hotspots: pd.DataFrame,
                            output_path: Path):
    """Generate markdown report of findings."""

    print("\nGenerating markdown report...")

    report = []
    report.append("# Phase 1A: Excluded Population Analysis\n")
    report.append("## Overview\n")
    report.append(f"- **Total pixels analyzed**: {summary['total_pixels']:,}")
    report.append(f"- **Total population**: {summary['total_population']:,.0f}")
    report.append("")

    report.append("## Probability Distribution\n")
    prob = summary['probability_distribution']
    report.append(f"| Statistic | Value |")
    report.append(f"|-----------|-------|")
    report.append(f"| Mean | {prob['mean']:.3f} |")
    report.append(f"| Median | {prob['median']:.3f} |")
    report.append(f"| Std Dev | {prob['std']:.3f} |")
    report.append(f"| P10 | {prob['p10']:.3f} |")
    report.append(f"| P90 | {prob['p90']:.3f} |")
    report.append("")

    report.append("## Distance Distribution (km)\n")
    dist = summary['distance_distribution']
    report.append(f"| Statistic | Value |")
    report.append(f"|-----------|-------|")
    report.append(f"| Mean | {dist['mean']:.1f} |")
    report.append(f"| Median | {dist['median']:.1f} |")
    report.append(f"| Std Dev | {dist['std']:.1f} |")
    report.append(f"| P10 | {dist['p10']:.1f} |")
    report.append(f"| P90 | {dist['p90']:.1f} |")
    report.append("")

    report.append("## Population by Probability Class\n")
    report.append("| Class | Population | % Total | Mean Distance (km) |")
    report.append("|-------|------------|---------|-------------------|")
    for cls in ['strong', 'moderate', 'weak', 'very_weak']:
        data = summary['by_probability_class'][cls]
        mean_dist = data['mean_distance_km']
        mean_dist_str = f"{mean_dist:.1f}" if mean_dist else "N/A"
        report.append(f"| {cls} | {data['population']:,.0f} | {data['population_pct']:.1f}% | {mean_dist_str} |")
    report.append("")

    report.append("## Population by Distance Class\n")
    report.append("| Class | Population | % Total | Mean Probability |")
    report.append("|-------|------------|---------|-----------------|")
    for cls in ['urban', 'peri_urban', 'rural', 'remote', 'very_remote']:
        data = summary['by_distance_class'][cls]
        mean_prob = data['mean_probability']
        mean_prob_str = f"{mean_prob:.3f}" if mean_prob else "N/A"
        report.append(f"| {cls} | {data['population']:,.0f} | {data['population_pct']:.1f}% | {mean_prob_str} |")
    report.append("")

    report.append("## Comparison: Connected vs Poorly Connected\n")
    try:
        report.append(comparison.to_markdown(index=False))
    except Exception:
        report.append(comparison.to_string(index=False))
    report.append("")

    report.append("## Key Findings\n")

    # Calculate key metrics
    well_connected_pct = summary['by_probability_class']['strong']['population_pct'] + \
                        summary['by_probability_class']['moderate']['population_pct']
    poorly_connected_pct = summary['by_probability_class']['weak']['population_pct'] + \
                          summary['by_probability_class']['very_weak']['population_pct']

    remote_pct = summary['by_distance_class'].get('remote', {}).get('population_pct', 0) + \
                summary['by_distance_class'].get('very_remote', {}).get('population_pct', 0)

    report.append(f"1. **{poorly_connected_pct:.1f}%** of population has weak/very weak probability allocation")
    report.append(f"2. **{remote_pct:.1f}%** of population is in remote/very remote areas")
    report.append(f"3. Only **{well_connected_pct:.1f}%** of population is strongly or moderately connected")
    report.append(f"4. Mean allocation probability is only **{prob['mean']:.1%}**")
    report.append(f"5. Mean distance to nearest facility is **{dist['mean']:.1f} km**")
    report.append("")

    report.append("## Implications for Climate-Health Modeling\n")
    report.append("The weak allocation probabilities suggest that the gravity model assigns most populations")
    report.append("to facilities with low confidence. This has several implications:\n")
    report.append("1. **Disease case assignment uncertainty**: Most cases are allocated with <50% confidence")
    report.append("2. **Spatial averaging of climate**: If assignment is uncertain, climate signals get diluted")
    report.append("3. **Need for alternative approaches**: Consider testing climate effects in remote areas")
    report.append("   where disease surveillance might use different methods (community health workers, etc.)")
    report.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"  Saved report to {output_path}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Phase 1A: Excluded Population Analysis")
    parser.add_argument("--network", default="INF",
                        help='Network label, e.g. INF, NCD, SYNINF, SYNNCD, SYNMODINF, SYNMODNCD')
    parser.add_argument("--hsa-mode", default="footprint")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--output-dir", default="out/analysis_exclusion")
    parser.add_argument("--text-output-dir", default="out/textresults")
    args = parser.parse_args()

    global NETWORK, HSA_MODE, DATA_DIR, OUT_DIR, ANALYSIS_DIR, OUTPUT_FILE_PREFIX, TEXT_RESULTS_DIR
    NETWORK = args.network
    HSA_MODE = args.hsa_mode
    DATA_DIR = Path(args.data_dir)
    OUT_DIR = Path(args.out_dir)
    ANALYSIS_DIR = Path(args.output_dir)
    OUTPUT_FILE_PREFIX = f"{NETWORK}_{HSA_MODE}"
    TEXT_RESULTS_DIR = Path(args.text_output_dir)
    TEXT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 1A: EXCLUDED POPULATION ANALYSIS")
    print(f"Network: {NETWORK}")
    print(f"HSA mode: {HSA_MODE}")
    print("=" * 70)

    # Load data
    allocations, facilities, boundary, governorates = load_data()

    # Calculate distance to nearest facility
    allocations = calculate_distance_to_nearest_facility(allocations, facilities)

    # Classify connectivity
    allocations = classify_connectivity(allocations)

    # Create summary statistics
    summary = create_exclusion_summary(allocations)

    # Save summary JSON
    summary_path = ANALYSIS_DIR / out_name("exclusion_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Identify hotspots
    hotspots = identify_exclusion_hotspots(allocations, governorates)
    hotspots.to_csv(ANALYSIS_DIR / out_name("exclusion_hotspots.csv"), index=False)

    # Create comparison table
    comparison = create_comparison_table(allocations)
    comparison.to_csv(ANALYSIS_DIR / out_name("connectivity_comparison.csv"), index=False)
    print("\n  Comparison table:")
    print(comparison.to_string(index=False))

    # Create map
    create_exclusion_map(
        allocations, boundary, governorates,
        ANALYSIS_DIR / out_name("exclusion_patterns_map.png")
    )

    # Generate report
    generate_markdown_report(
        summary, comparison, hotspots,
        md_path("exclusion_analysis_report.md")
    )

    # Save classified allocations for Phase 1B
    # (Save only a subset to reduce file size)
    classified_path = ANALYSIS_DIR / out_name("allocations_classified.parquet")
    try:
        allocations.to_parquet(classified_path)
        print(f"\nSaved classified allocations to {classified_path}")
    except Exception as e:
        # Fallback to CSV if parquet not available
        classified_path = ANALYSIS_DIR / out_name("allocations_classified.csv")
        # Sample to reduce size
        sample_size = min(500000, len(allocations))
        allocations.sample(sample_size, random_state=42).to_csv(classified_path, index=False)
        print(f"\nSaved sampled classified allocations to {classified_path} ({sample_size:,} rows)")

    print("\n" + "=" * 70)
    print("PHASE 1A COMPLETE")
    print("=" * 70)

    return summary, allocations


if __name__ == "__main__":
    summary, allocations = main()
