"""
HSA Objective Analysis Module
==============================

Functions for analyzing HSA optimization results across different objective modes.
Each section (3-8) is encapsulated as a separate function for clean, reusable analysis.

Usage:
    from hsa_objective_analysis import analyze_all_modes

    analyze_all_modes(all_results)

Author: HSA Research Team
Date: 2025-11-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# SECTION 3: Selected Facilities Summary
# ============================================================================
def section_3_facilities_summary(mode, result):
    """
    Display ranked list of all selected facilities with volume and score statistics.

    Parameters:
    -----------
    mode : str
        Optimization mode name (e.g., 'fewest', 'footprint')
    result : dict
        Optimization result dictionary with 'facilities' and 'coverage' keys

    Returns:
    --------
    pandas.DataFrame
        Sorted facilities dataframe with volume_numeric column added
    """
    selected_facilities = result['facilities']
    coverage_pct = result['coverage']

    print(f"\n{'='*80}")
    print(f"{mode.upper()} MODE - ALL SELECTED FACILITIES")
    print('='*80)
    print(f"Number of HSAs: {len(selected_facilities)}")
    print(f"Coverage: {coverage_pct:.2f}%")

    # Sort by composite score (highest first)
    sorted_facilities = selected_facilities.copy()
    sorted_facilities['volume_numeric'] = pd.to_numeric(sorted_facilities['Total'], errors='coerce')

    if 'composite_score' in sorted_facilities.columns:
        sorted_facilities = sorted_facilities.sort_values('composite_score', ascending=False)

        # Display ALL facilities
        print(f"\n{'Rank':<5}{'Facility':<45}{'Volume':>8}{'Score':>12}{'Radius_km':>10}")
        print('-'*80)
        for rank, (idx, row) in enumerate(sorted_facilities.iterrows(), 1):
            vol = row['volume_numeric']
            score = row.get('composite_score', 0)
            radius = row.get('service_radius_km', row.get('initial_radius_km', 0))
            print(f"{rank:<5}{row['HealthFacility'][:44]:<45}{vol:>8.0f}{score:>12.2e}{radius:>10.1f}")
    else:
        sorted_facilities = sorted_facilities.sort_values('volume_numeric', ascending=False)

        # Display ALL facilities
        print(f"\n{'Rank':<5}{'Facility':<45}{'Volume':>8}{'Radius_km':>10}")
        print('-'*80)
        for rank, (idx, row) in enumerate(sorted_facilities.iterrows(), 1):
            vol = row['volume_numeric']
            radius = row.get('service_radius_km', row.get('initial_radius_km', 0))
            print(f"{rank:<5}{row['HealthFacility'][:44]:<45}{vol:>8.0f}{radius:>10.1f}")

    # Summary statistics
    print(f"\nVolume Statistics:")
    print(f"  Min: {sorted_facilities['volume_numeric'].min():.0f}")
    print(f"  Median: {sorted_facilities['volume_numeric'].median():.0f}")
    print(f"  Max: {sorted_facilities['volume_numeric'].max():.0f}")
    print(f"  Total: {sorted_facilities['volume_numeric'].sum():.0f}")

    return sorted_facilities


# ============================================================================
# SECTION 4: Volume Distribution Visualization
# ============================================================================
def section_4_volume_distribution(mode, sorted_facilities):
    """
    Visualize patient volume distribution across selected HSAs.
    Creates histogram and box plot showing volume statistics.

    Parameters:
    -----------
    mode : str
        Optimization mode name
    sorted_facilities : pandas.DataFrame
        Facilities dataframe with 'volume_numeric' column
    """
    volumes = sorted_facilities['volume_numeric'].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(volumes, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.median(volumes), color='red', linestyle='--', linewidth=2,
                label=f'Median: {np.median(volumes):.0f}')
    ax1.axvline(np.mean(volumes), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(volumes):.0f}')
    ax1.set_xlabel('Patient Volume', fontsize=12)
    ax1.set_ylabel('Number of HSAs', fontsize=12)
    ax1.set_title(f'{mode.upper()} - Volume Distribution', fontsize=14, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(volumes, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))
    ax2.set_ylabel('Patient Volume', fontsize=12)
    ax2.set_title(f'{mode.upper()} - Volume Box Plot', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Print quartile statistics
    print(f"\n{'='*80}")
    print(f"{mode.upper()} - VOLUME QUARTILES")
    print('='*80)
    print(f"  Minimum:         {np.min(volumes):>10.0f}")
    print(f"  25th Percentile: {np.percentile(volumes, 25):>10.0f}")
    print(f"  Median (50th):   {np.percentile(volumes, 50):>10.0f}")
    print(f"  75th Percentile: {np.percentile(volumes, 75):>10.0f}")
    print(f"  Maximum:         {np.max(volumes):>10.0f}")
    print(f"  Mean:            {np.mean(volumes):>10.0f}")
    print(f"  Std Dev:         {np.std(volumes):>10.0f}")


# ============================================================================
# SECTION 5: Service Radius Analysis
# ============================================================================
def section_5_radius_analysis(mode, sorted_facilities):
    """
    Analyze service radius distribution across selected HSAs.
    Shows radius statistics and distribution plot.

    Parameters:
    -----------
    mode : str
        Optimization mode name
    sorted_facilities : pandas.DataFrame
        Facilities dataframe with radius information
    """
    radii = sorted_facilities.apply(
        lambda row: row.get('service_radius_km', row.get('initial_radius_km', 0)),
        axis=1
    ).values

    print(f"\n{'='*80}")
    print(f"{mode.upper()} - SERVICE RADIUS ANALYSIS")
    print('='*80)
    print(f"  Minimum radius:  {np.min(radii):>6.1f} km")
    print(f"  Median radius:   {np.median(radii):>6.1f} km")
    print(f"  Maximum radius:  {np.max(radii):>6.1f} km")
    print(f"  Mean radius:     {np.mean(radii):>6.1f} km")
    print(f"  Unique radii:    {len(np.unique(radii)):>6d}")

    # Radius distribution plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(radii, bins=15, edgecolor='black', alpha=0.7, color='forestgreen')
    ax.axvline(np.median(radii), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(radii):.1f} km')
    ax.set_xlabel('Service Radius (km)', fontsize=12)
    ax.set_ylabel('Number of HSAs', fontsize=12)
    ax.set_title(f'{mode.upper()} - Service Radius Distribution', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 6: Geographic Coverage Map (Simple)
# ============================================================================
def section_6_simple_coverage_map(mode, sorted_facilities):
    """
    Create simple scatter plot of HSA anchor locations.
    Points are sized by patient volume and colored by volume.

    Parameters:
    -----------
    mode : str
        Optimization mode name
    sorted_facilities : pandas.DataFrame
        Facilities dataframe with geometry and volume_numeric
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot anchor locations
    lons = [geom.x for geom in sorted_facilities.geometry]
    lats = [geom.y for geom in sorted_facilities.geometry]
    volumes = sorted_facilities['volume_numeric'].values

    scatter = ax.scatter(lons, lats, c=volumes, s=volumes/10,
                        cmap='YlOrRd', alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add facility names for top 5
    top_5 = sorted_facilities.head(5)
    for idx, row in top_5.iterrows():
        ax.annotate(row['HealthFacility'][:20],
                   xy=(row.geometry.x, row.geometry.y),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, color='darkred', weight='bold')

    plt.colorbar(scatter, ax=ax, label='Patient Volume')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'{mode.upper()} - HSA Anchor Locations (sized by volume)',
                fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 7: Composite Score Analysis (if available)
# ============================================================================
def section_7_score_analysis(mode, sorted_facilities):
    """
    Analyze composite scores if available in the results.
    Shows score distribution and statistics.

    Parameters:
    -----------
    mode : str
        Optimization mode name
    sorted_facilities : pandas.DataFrame
        Facilities dataframe (may contain 'composite_score' column)
    """
    if 'composite_score' not in sorted_facilities.columns:
        print(f"\n{mode.upper()}: No composite scores available (skipping score analysis)")
        return

    scores = sorted_facilities['composite_score'].values

    print(f"\n{'='*80}")
    print(f"{mode.upper()} - COMPOSITE SCORE ANALYSIS")
    print('='*80)
    print(f"  Score range: {np.min(scores):.2e} to {np.max(scores):.2e}")
    print(f"  Median score: {np.median(scores):.2e}")
    print(f"  Mean score: {np.mean(scores):.2e}")

    # Score distribution plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(scores, bins=20, edgecolor='black', alpha=0.7, color='mediumpurple')
    ax.axvline(np.median(scores), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(scores):.2e}')
    ax.set_xlabel('Composite Score', fontsize=12)
    ax.set_ylabel('Number of HSAs', fontsize=12)
    ax.set_title(f'{mode.upper()} - Composite Score Distribution', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 8: Selection Order Analysis (Score Progression)
# ============================================================================
def section_8_selection_order_analysis(mode, sorted_facilities):
    """
    Show how scores progress in the order facilities were selected.
    This reveals the greedy selection strategy.

    Parameters:
    -----------
    mode : str
        Optimization mode name
    sorted_facilities : pandas.DataFrame
        Facilities dataframe (may contain 'composite_score' and 'selection_order')
    """
    if 'composite_score' not in sorted_facilities.columns:
        print(f"\n{mode.upper()}: No composite scores available (skipping selection order analysis)")
        return

    print(f"\n{'='*80}")
    print(f"{mode.upper()} - SELECTION ORDER ANALYSIS")
    print('='*80)

    # Sort by selection order if available, otherwise by score
    if 'selection_order' in sorted_facilities.columns:
        ordered = sorted_facilities.sort_values('selection_order')
        xlabel = 'Selection Order'
        x_vals = ordered['selection_order'].values
    else:
        # Assume current index order represents selection order
        ordered = sorted_facilities.copy()
        ordered['order'] = range(1, len(ordered) + 1)
        xlabel = 'Selection Rank (by score)'
        x_vals = ordered['order'].values

    scores = ordered['composite_score'].values

    # Plot selection progression
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Line plot with markers
    ax.plot(x_vals, scores, 'o-', color='steelblue', markersize=8,
           linewidth=2, markeredgecolor='darkblue', markeredgewidth=1)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Composite Score', fontsize=12)
    ax.set_title(f'{mode.upper()} - Score Progression During Selection',
                fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)

    # Add horizontal line for mean score
    mean_score = np.mean(scores)
    ax.axhline(mean_score, color='red', linestyle='--', linewidth=2,
              alpha=0.7, label=f'Mean: {mean_score:.2e}')

    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"Score range: {np.min(scores):.2e} to {np.max(scores):.2e}")
    print(f"Score decay ratio (last/first): {scores[-1]/scores[0]:.3f}")


# ============================================================================
# SECTION 9: Correlation Analysis (Volume vs Radius)
# ============================================================================
def section_9_correlation_analysis(mode, sorted_facilities):
    """
    Analyze correlation between patient volume and service radius.
    Creates scatter plot with trend line and reports correlation coefficient.

    Parameters:
    -----------
    mode : str
        Optimization mode name
    sorted_facilities : pandas.DataFrame
        Facilities dataframe with volume_numeric and radius information
    """
    volumes = sorted_facilities['volume_numeric'].values
    radii = sorted_facilities.apply(
        lambda row: row.get('service_radius_km', row.get('initial_radius_km', 0)),
        axis=1
    ).values

    # Calculate correlation
    correlation = np.corrcoef(volumes, radii)[0, 1]

    print(f"\n{'='*80}")
    print(f"{mode.upper()} - VOLUME vs RADIUS CORRELATION")
    print('='*80)
    print(f"  Correlation coefficient: {correlation:.3f}")

    # Scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(volumes, radii, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

    # Add trend line
    z = np.polyfit(volumes, radii, 1)
    p = np.poly1d(z)
    ax.plot(volumes, p(volumes), "r--", alpha=0.8, linewidth=2,
            label=f'Trend (r={correlation:.3f})')

    ax.set_xlabel('Patient Volume', fontsize=12)
    ax.set_ylabel('Service Radius (km)', fontsize=12)
    ax.set_title(f'{mode.upper()} - Volume vs Radius Correlation', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN FUNCTION: Analyze Single Mode
# ============================================================================
def analyze_single_mode(mode, result):
    """
    Execute all analysis sections (3-8) for a single optimization mode.

    Parameters:
    -----------
    mode : str
        Optimization mode name (e.g., 'fewest', 'footprint')
    result : dict
        Optimization result dictionary with 'facilities' and 'coverage' keys
    """
    print(f"\n\n{'#'*80}")
    print(f"# ANALYZING MODE: {mode.upper()}")
    print(f"{'#'*80}\n")

    # Section 3: Facilities Summary
    sorted_facilities = section_3_facilities_summary(mode, result)

    # Section 4: Volume Distribution
    section_4_volume_distribution(mode, sorted_facilities)

    # Section 5: Radius Analysis
    section_5_radius_analysis(mode, sorted_facilities)

    # Section 6: Simple Coverage Map
    section_6_simple_coverage_map(mode, sorted_facilities)

    # Section 7: Score Analysis
    section_7_score_analysis(mode, sorted_facilities)

    # Section 8: Selection Order Analysis
    section_8_selection_order_analysis(mode, sorted_facilities)

    # Section 9: Correlation Analysis
    section_9_correlation_analysis(mode, sorted_facilities)

    print(f"\n{'='*80}")
    print(f"COMPLETED ANALYSIS FOR {mode.upper()}")
    print(f"{'='*80}\n")


# ============================================================================
# MAIN FUNCTION: Analyze All Modes
# ============================================================================
def analyze_all_modes(all_results, modes=None):
    """
    Execute complete analysis (sections 3-8) for all optimization modes.

    Parameters:
    -----------
    all_results : dict
        Dictionary mapping mode names to optimization results
        Format: {'fewest': {'facilities': GeoDataFrame, 'coverage': float}, ...}
    modes : list, optional
        List of mode names to analyze. If None, analyzes all standard modes:
        ['fewest', 'footprint', 'distance', 'governorate_tau_coverage', 'governorate_fewest']

    Example:
    --------
    >>> from hsa_objective_analysis import analyze_all_modes
    >>> analyze_all_modes(all_results)
    """
    if modes is None:
        modes = ['fewest', 'footprint', 'distance',
                'governorate_tau_coverage', 'governorate_fewest']

    for mode in modes:
        if mode not in all_results:
            print(f"\nSkipping {mode.upper()} - not in results")
            continue

        result = all_results[mode]
        analyze_single_mode(mode, result)

    print("\n" + "#"*80)
    print("# ALL MODES ANALYZED")
    print("#"*80)
