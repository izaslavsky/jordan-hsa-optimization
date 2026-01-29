#!/usr/bin/env python3
"""
Compare Spatial Unit Construction Methods for HSA Delineation - V2

FAIR COMPARISON: Uses ALL candidate facilities for baseline methods,
not just the optimized anchors.

Methods:
1. Fixed-radius buffers (ALL 184 facilities)
2. Voronoi tessellation (ALL 184 facilities)
3. Governorate-based administrative units (12 units)
4. Optimized HSAs (17 selected facilities with adaptive radii)

Metrics:
- Population coverage
- Overlap / double-counting
- Number of spatial units
- Shape metrics (compactness)
- Spatial concordance with optimized HSAs

Author: Climate-Health Research Team
Date: January 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union, voronoi_diagram
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
from pathlib import Path

def load_population_raster(pop_raster_path):
    """Load population raster."""
    with rasterio.open(pop_raster_path) as src:
        pop_arr = src.read(1)
        pop_arr = np.where(pop_arr < 0, 0, pop_arr)
        transform = src.transform
    return pop_arr, transform

def load_optimized_hsas(hsas_path):
    """Load optimized HSA boundaries."""
    gdf = gpd.read_file(hsas_path)
    return gdf

def load_all_facilities(facilities_csv):
    """Load ALL candidate facilities and ensure WGS84 CRS."""
    df = pd.read_csv(facilities_csv, encoding='utf-8-sig')
    lat_col = 'lat' if 'lat' in df.columns else 'Latitude'
    lon_col = 'lon' if 'lon' in df.columns else 'Longitude'
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Facility CSV missing lat/lon columns: {facilities_csv}")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs='EPSG:4326'
    )
    gdf['lon'] = df[lon_col]
    gdf['lat'] = df[lat_col]
    return gdf

def load_governorates(governorates_path):
    """Load governorate boundaries."""
    return gpd.read_file(governorates_path)

def load_jordan_boundary(boundary_path):
    """Load Jordan boundary."""
    gdf = gpd.read_file(boundary_path)
    return gdf.union_all()

def compute_population_in_geometry(geometry, pop_arr, transform, pop_raster_path):
    """Compute total population within a geometry."""
    if geometry is None or geometry.is_empty:
        return 0
    try:
        with rasterio.open(pop_raster_path) as src:
            out_image, _ = mask(src, [geometry], crop=True, nodata=0)
            pop = out_image[0]
            pop = np.where(pop < 0, 0, pop)
            return pop.sum()
    except:
        return 0

def compute_overlap_metrics(geometries, pop_arr, transform, pop_raster_path):
    """Compute coverage and overlap metrics."""
    valid_geoms = [g for g in geometries if g is not None and not g.is_empty]
    if not valid_geoms:
        return 0, 0, 0, 0

    union_geom = unary_union(valid_geoms)
    unique_pop = compute_population_in_geometry(union_geom, pop_arr, transform, pop_raster_path)
    total_counted = sum(compute_population_in_geometry(g, pop_arr, transform, pop_raster_path) for g in valid_geoms)

    overlap_pop = total_counted - unique_pop
    multiplier = total_counted / unique_pop if unique_pop > 0 else 0

    return unique_pop, total_counted, overlap_pop, multiplier

def compute_shape_metrics(geometries):
    """Compute shape metrics for polygons."""
    compactness_values = []
    areas_km2 = []

    for geom in geometries:
        if geom is None or geom.is_empty:
            continue

        # Project to UTM for accurate area/perimeter
        gdf_temp = gpd.GeoDataFrame(geometry=[geom], crs='EPSG:4326').to_crs('EPSG:32636')
        geom_utm = gdf_temp.geometry.iloc[0]

        area = geom_utm.area / 1e6  # km²
        perimeter = geom_utm.length / 1000  # km

        # Compactness: 4π*area / perimeter² (1.0 = perfect circle)
        if perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            compactness_values.append(compactness)

        areas_km2.append(area)

    return {
        'mean_compactness': np.mean(compactness_values) if compactness_values else 0,
        'mean_area_km2': np.mean(areas_km2) if areas_km2 else 0,
        'std_area_km2': np.std(areas_km2) if areas_km2 else 0,
        'min_area_km2': np.min(areas_km2) if areas_km2 else 0,
        'max_area_km2': np.max(areas_km2) if areas_km2 else 0
    }

def compute_concordance_with_hsas(test_geometries, hsa_geometries):
    """
    Compute what fraction of test geometries' area overlaps with optimized HSAs.
    """
    hsa_union = unary_union([g for g in hsa_geometries if g is not None and not g.is_empty])

    total_area = 0
    overlap_area = 0

    for geom in test_geometries:
        if geom is None or geom.is_empty:
            continue

        # Project for accurate area
        gdf_temp = gpd.GeoDataFrame(geometry=[geom], crs='EPSG:4326').to_crs('EPSG:32636')
        geom_utm = gdf_temp.geometry.iloc[0]
        total_area += geom_utm.area

        # Compute overlap
        intersection = geom.intersection(hsa_union)
        if not intersection.is_empty:
            gdf_int = gpd.GeoDataFrame(geometry=[intersection], crs='EPSG:4326').to_crs('EPSG:32636')
            overlap_area += gdf_int.geometry.iloc[0].area

    concordance = (overlap_area / total_area * 100) if total_area > 0 else 0
    return concordance

def create_fixed_radius_buffers(facilities_gdf, radius_km, jordan_boundary):
    """Create fixed-radius buffers around ALL facilities."""
    facilities_utm = facilities_gdf.to_crs('EPSG:32636')
    radius_m = radius_km * 1000

    buffers = facilities_utm.geometry.buffer(radius_m)
    buffers_gdf = gpd.GeoDataFrame(geometry=buffers, crs='EPSG:32636').to_crs('EPSG:4326')
    buffers_gdf['geometry'] = buffers_gdf.geometry.intersection(jordan_boundary)

    return buffers_gdf

def create_voronoi_tessellation(facilities_gdf, jordan_boundary):
    """Create Voronoi tessellation from ALL facility points."""
    from shapely.geometry import MultiPoint

    points = MultiPoint([p for p in facilities_gdf.geometry])

    try:
        regions = voronoi_diagram(points, envelope=jordan_boundary)

        polygons = []
        for geom in list(regions.geoms):
            if geom.geom_type == 'Polygon':
                clipped = geom.intersection(jordan_boundary)
                if not clipped.is_empty and clipped.area > 0:
                    if clipped.geom_type == 'MultiPolygon':
                        for p in clipped.geoms:
                            if p.area > 0:
                                polygons.append(p)
                    elif clipped.geom_type == 'Polygon':
                        polygons.append(clipped)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    clipped = poly.intersection(jordan_boundary)
                    if not clipped.is_empty and clipped.area > 0:
                        polygons.append(clipped)

        print(f"    Created {len(polygons)} Voronoi cells from {len(facilities_gdf)} points")
        return gpd.GeoDataFrame(geometry=polygons, crs='EPSG:4326')
    except Exception as e:
        print(f"  Voronoi failed: {e}")
        import traceback
        traceback.print_exc()
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')

def main():
    parser = argparse.ArgumentParser(description="Compare spatial unit construction methods (v2)")
    parser.add_argument("--network", default=os.environ.get("NETWORK", "INF"))
    parser.add_argument("--hsa-mode", default=os.environ.get("HSA_MODE", "footprint"))
    parser.add_argument("--population-raster", required=True, help="Path to population raster")
    parser.add_argument("--hsas-path", required=True, help="Path to optimized HSA GeoJSON")
    parser.add_argument("--facilities-csv", required=True, help="Path to facility coordinates CSV")
    parser.add_argument("--governorates-path", required=True, help="Path to governorates GeoPackage")
    parser.add_argument("--boundary-path", required=True, help="Path to Jordan boundary GeoPackage")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path for comparison output CSV (defaults to out/{network}_spatial_methods_comparison.csv)",
    )
    args = parser.parse_args()
    network = args.network
    mode = args.hsa_mode

    print("="*80)
    print("SPATIAL UNIT CONSTRUCTION METHOD COMPARISON - FAIR COMPARISON")
    print(f"Network: {network} | HSA mode: {mode}")
    print("="*80)
    print("\nUsing ALL candidate facilities for baseline methods")

    # Load data
    print("\nLoading data...")
    pop_arr, transform = load_population_raster(args.population_raster)
    total_pop = pop_arr.sum()
    print(f"  Total population: {total_pop:,.0f}")

    jordan_boundary = load_jordan_boundary(args.boundary_path)
    governorates = load_governorates(args.governorates_path)
    optimized_hsas = load_optimized_hsas(args.hsas_path)
    all_facilities = load_all_facilities(args.facilities_csv)

    n_facilities = len(all_facilities)
    n_hsas = len(optimized_hsas)
    print(f"  ALL candidate facilities: {n_facilities}")
    print(f"  Optimized HSA anchors: {n_hsas}")

    results = []

    # ==========================================
    # 1. OPTIMIZED HSAs (17 selected facilities)
    # ==========================================
    print("\n" + "-"*60)
    print("1. OPTIMIZED HSAs (17 anchors, adaptive radii)")
    print("-"*60)

    geoms = list(optimized_hsas.geometry)
    unique_pop, total_counted, overlap_pop, multiplier = compute_overlap_metrics(
        geoms, pop_arr, transform, args.population_raster
    )
    shape = compute_shape_metrics(geoms)

    results.append({
        'method': 'Optimized HSA',
        'n_units': n_hsas,
        'n_facilities_used': n_hsas,
        'coverage_pct': unique_pop / total_pop * 100,
        'overlap_pct': overlap_pop / unique_pop * 100 if unique_pop > 0 else 0,
        'coverage_multiplier': multiplier,
        'mean_compactness': shape['mean_compactness'],
        'mean_area_km2': shape['mean_area_km2'],
        'std_area_km2': shape['std_area_km2'],
        'concordance_pct': 100.0  # By definition
    })
    print(f"  Units: {n_hsas}, Coverage: {unique_pop/total_pop*100:.1f}%, Overlap: {overlap_pop/unique_pop*100:.1f}%")

    # ==========================================
    # 2. FIXED-RADIUS BUFFERS (ALL facilities)
    # ==========================================
    for radius in [10, 15, 18, 20]:
        print("\n" + "-"*60)
        print(f"2. FIXED-RADIUS {radius}km (ALL {n_facilities} facilities)")
        print("-"*60)

        buffers = create_fixed_radius_buffers(all_facilities, radius, jordan_boundary)
        geoms = list(buffers.geometry)

        unique_pop, total_counted, overlap_pop, multiplier = compute_overlap_metrics(
            geoms, pop_arr, transform, args.population_raster
        )
        shape = compute_shape_metrics(geoms)
        concordance = compute_concordance_with_hsas(geoms, list(optimized_hsas.geometry))

        results.append({
            'method': f'Fixed {radius}km (all)',
            'n_units': n_facilities,
            'n_facilities_used': n_facilities,
            'coverage_pct': unique_pop / total_pop * 100,
            'overlap_pct': overlap_pop / unique_pop * 100 if unique_pop > 0 else 0,
            'coverage_multiplier': multiplier,
            'mean_compactness': shape['mean_compactness'],
            'mean_area_km2': shape['mean_area_km2'],
            'std_area_km2': shape['std_area_km2'],
            'concordance_pct': concordance
        })
        print(f"  Units: {n_facilities}, Coverage: {unique_pop/total_pop*100:.1f}%, Overlap: {overlap_pop/unique_pop*100:.1f}% ({multiplier:.1f}x)")
        print(f"  Concordance with HSAs: {concordance:.1f}%")

    # ==========================================
    # 3. VORONOI TESSELLATION (ALL facilities)
    # ==========================================
    print("\n" + "-"*60)
    print(f"3. VORONOI TESSELLATION (ALL {n_facilities} facilities)")
    print("-"*60)

    voronoi = create_voronoi_tessellation(all_facilities, jordan_boundary)
    geoms = list(voronoi.geometry)

    unique_pop, total_counted, overlap_pop, multiplier = compute_overlap_metrics(
        geoms, pop_arr, transform, args.population_raster
    )
    shape = compute_shape_metrics(geoms)
    concordance = compute_concordance_with_hsas(geoms, list(optimized_hsas.geometry))

    results.append({
        'method': 'Voronoi (all)',
        'n_units': len(voronoi),
        'n_facilities_used': n_facilities,
        'coverage_pct': unique_pop / total_pop * 100,
        'overlap_pct': overlap_pop / unique_pop * 100 if unique_pop > 0 else 0,
        'coverage_multiplier': multiplier,
        'mean_compactness': shape['mean_compactness'],
        'mean_area_km2': shape['mean_area_km2'],
        'std_area_km2': shape['std_area_km2'],
        'concordance_pct': concordance
    })
    print(f"  Units: {len(voronoi)}, Coverage: {unique_pop/total_pop*100:.1f}%, Overlap: {overlap_pop/unique_pop*100:.1f}%")
    print(f"  Mean compactness: {shape['mean_compactness']:.3f}, Mean area: {shape['mean_area_km2']:.1f} km²")
    print(f"  Concordance with HSAs: {concordance:.1f}%")

    # ==========================================
    # 4. GOVERNORATE BOUNDARIES
    # ==========================================
    print("\n" + "-"*60)
    print("4. GOVERNORATE BOUNDARIES (12 administrative units)")
    print("-"*60)

    geoms = list(governorates.geometry)

    unique_pop, total_counted, overlap_pop, multiplier = compute_overlap_metrics(
        geoms, pop_arr, transform, args.population_raster
    )
    shape = compute_shape_metrics(geoms)
    concordance = compute_concordance_with_hsas(geoms, list(optimized_hsas.geometry))

    # Count facilities per governorate
    facilities_with_gov = gpd.sjoin(all_facilities, governorates, how='left', predicate='within')
    facilities_per_gov = facilities_with_gov.groupby('index_right').size()

    results.append({
        'method': 'Governorate',
        'n_units': len(governorates),
        'n_facilities_used': 'N/A',
        'coverage_pct': unique_pop / total_pop * 100,
        'overlap_pct': 0,
        'coverage_multiplier': 1.0,
        'mean_compactness': shape['mean_compactness'],
        'mean_area_km2': shape['mean_area_km2'],
        'std_area_km2': shape['std_area_km2'],
        'concordance_pct': concordance
    })
    print(f"  Units: {len(governorates)}, Coverage: {unique_pop/total_pop*100:.1f}%, Overlap: 0%")
    print(f"  Mean facilities per governorate: {facilities_per_gov.mean():.1f}")
    print(f"  Mean compactness: {shape['mean_compactness']:.3f}, Mean area: {shape['mean_area_km2']:.1f} km²")
    print(f"  Concordance with HSAs: {concordance:.1f}%")

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))

    # Save
    output_csv = args.output_csv or f"out/{network}_spatial_methods_comparison.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")

    # Markdown table
    print("\n" + "="*80)
    print("MARKDOWN TABLE FOR PAPER")
    print("="*80)

    print("\n| Method | Units | Coverage (%) | Overlap (%) | Multiplier | Compactness | Concordance (%) |")
    print("|--------|-------|--------------|-------------|------------|-------------|-----------------|")
    for _, row in df.iterrows():
        print(f"| {row['method']} | {row['n_units']} | {row['coverage_pct']:.1f} | "
              f"{row['overlap_pct']:.1f} | {row['coverage_multiplier']:.2f}x | "
              f"{row['mean_compactness']:.3f} | {row['concordance_pct']:.1f} |")

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    opt = df[df['method'] == 'Optimized HSA'].iloc[0]
    vor = df[df['method'] == 'Voronoi (all)'].iloc[0]
    gov = df[df['method'] == 'Governorate'].iloc[0]
    fix18 = df[df['method'] == 'Fixed 18km (all)'].iloc[0]

    print(f"""
1. DOUBLE-COUNTING PROBLEM:
   - Fixed 18km (all facilities): {fix18['overlap_pct']:.0f}% overlap ({fix18['coverage_multiplier']:.1f}x multiplier)
   - Each person counted {fix18['coverage_multiplier']:.1f} times on average
   - Makes disease rate calculation impossible without de-duplication

2. GRANULARITY PROBLEM:
   - Voronoi (all facilities): {vor['n_units']} cells - too fine for surveillance
   - Governorate: {gov['n_units']} units - too coarse for local patterns
   - Optimized HSA: {opt['n_units']} units - appropriate granularity

3. SHAPE QUALITY:
   - Optimized HSA compactness: {opt['mean_compactness']:.3f} (circular = 1.0)
   - Voronoi compactness: {vor['mean_compactness']:.3f} (irregular shapes)
   - Governorate compactness: {gov['mean_compactness']:.3f}

4. SPATIAL CONCORDANCE WITH OPTIMIZED HSAs:
   - What % of alternative boundaries overlap with optimized HSA coverage?
   - Fixed 18km: {fix18['concordance_pct']:.1f}%
   - Voronoi: {vor['concordance_pct']:.1f}%
   - Governorate: {gov['concordance_pct']:.1f}%
""")

    return df

if __name__ == '__main__':
    results = main()
