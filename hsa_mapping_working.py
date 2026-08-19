"""
HSA Mapping Module - WORKING VERSION
====================================

Extracted from create_geopandas_maps.py which generates correct maps.

Functions for creating high-quality maps of HSA optimization results.

Usage:
    from hsa_mapping_working import create_hsa_map, create_all_hsa_maps

    create_all_hsa_maps(all_results, network='INF', data_dir=DATA_DIR, out_dir=OUT_DIR)

Date: 2025-11-29
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import rasterio
from adjustText import adjust_text


def create_hsa_circles_clipped(facilities_gdf, country_boundary):
    """
    Create circular HSA polygons and clip them to country boundary.

    Parameters:
    -----------
    facilities_gdf : GeoDataFrame
        Facilities with geometry (points) and radius information
    country_boundary : GeoDataFrame or Polygon
        Country boundary polygon(s)

    Returns:
    --------
    GeoDataFrame
        Facilities with geometry replaced by clipped circles
    """
    # Get country polygon
    if hasattr(country_boundary, 'unary_union'):
        country_polygon = country_boundary.unary_union
    else:
        country_polygon = country_boundary

    circles = []
    for idx, row in facilities_gdf.iterrows():
        point = row.geometry
        radius_km = row.get('service_radius_km', row.get('initial_radius_km', 15.0))

        # Convert radius from km to degrees at this latitude
        lat = point.y
        radius_lat_deg = radius_km / 111.32
        radius_lon_deg = radius_km / (111.32 * np.cos(np.radians(lat)))
        radius_deg = (radius_lat_deg + radius_lon_deg) / 2

        # Create circle
        circle = point.buffer(radius_deg)

        # Clip to country boundary
        clipped = circle.intersection(country_polygon)
        circles.append(clipped)

    gdf = facilities_gdf.copy()
    gdf.geometry = circles
    return gdf


def create_hsa_map(
    mode_name,
    mode_title,
    facilities_hsa,
    all_facilities,
    country_boundary,
    governorates,
    pop_raster_path,
    output_path=None,
    network='INF',
    target_crs='EPSG:4326',
    hsa_polygons=None,
):
    """
    Create a single HSA map with all layers properly styled.

    THIS IS THE WORKING VERSION extracted from create_geopandas_maps.py

    Parameters:
    -----------
    mode_name : str
        Mode identifier (e.g., 'fewest')
    mode_title : str
        Mode display title (e.g., 'FEWEST HSAs')
    facilities_hsa : GeoDataFrame
        HSA anchor facilities (points)
    all_facilities : GeoDataFrame
        All facilities in the network (points)
    country_boundary : GeoDataFrame
        Country boundary polygon
    governorates : GeoDataFrame
        Governorate boundaries
    pop_raster_path : Path or str
        Path to population raster file
    output_path : Path or str, optional
        Path to save map. If None, map is shown but not saved.
    network : str
        Network type (e.g., 'INF', 'NCD')
    target_crs : str
        Target CRS for all layers (default: EPSG:4326)

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Ensure all layers are in the same CRS
    if facilities_hsa.crs != target_crs:
        facilities_hsa = facilities_hsa.to_crs(target_crs)
    if all_facilities.crs != target_crs:
        all_facilities = all_facilities.to_crs(target_crs)
    if country_boundary.crs != target_crs:
        country_boundary = country_boundary.to_crs(target_crs)
    if governorates.crs != target_crs:
        governorates = governorates.to_crs(target_crs)

    # Always build degree-based circles for the outline (appear circular on screen)
    hsa_circles_outline = create_hsa_circles_clipped(facilities_hsa, country_boundary)

    # Use pre-built population-clipped polygons for the fill if provided,
    # otherwise fall back to the degree circles
    if hsa_polygons is not None:
        hsa_circles = hsa_polygons.to_crs(target_crs)
    else:
        hsa_circles = hsa_circles_outline

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # 1. Plot population raster (black-gray background) - WORKING VERSION
    with rasterio.open(pop_raster_path) as src:
        pop_data = src.read(1)
        pop_data_masked = np.ma.masked_where(pop_data <= 0, pop_data)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

        # Plot with reversed grayscale - white for no population, dark for high population
        im = ax.imshow(np.log10(pop_data_masked + 1), extent=extent,
                      cmap='gray_r', alpha=0.5, zorder=1, aspect='equal',
                      interpolation='bilinear', vmin=0, vmax=4)

    # 2. Plot country boundary
    country_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=2.0, zorder=2)

    # 3. Plot governorate boundaries
    governorates.boundary.plot(ax=ax, edgecolor='gray', linewidth=0.8, zorder=3, alpha=0.6)

    # 4a. Plot full service radius as dashed outline (circular on screen, shows intended radius)
    hsa_circles_outline.plot(ax=ax, color='none', edgecolor='#FF0000',
                             linewidth=1.8, linestyle='dashed', zorder=4)

    # 4b. Plot population-clipped fill (solid green = actual populated service area)
    #     The gap between dashed outline and solid fill shows excluded desert areas
    hsa_circles.plot(ax=ax, color='#00AA00', alpha=0.25, edgecolor='black',
                    linewidth=0.5, zorder=4)

    # 5. Plot all facilities (small blue dots)
    all_facilities.plot(ax=ax, color='#0000FF', markersize=8, alpha=0.5,
                       zorder=5, marker='o')

    # 6. Plot HSA anchors (red dots, slightly bigger)
    facilities_hsa.plot(ax=ax, color='#FF0000', markersize=25,
                       edgecolor='white', linewidth=0.8, zorder=6, marker='o')

    # 7. Add labels for HSA anchors (non-overlapping)
    texts = []
    for idx, row in facilities_hsa.iterrows():
        point = row.geometry
        label = row['HealthFacility']
        # Create text object
        txt = ax.text(point.x, point.y, label, fontsize=7, ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'),
                     zorder=7)
        texts.append(txt)

    # Adjust text positions to avoid overlap
    if len(texts) > 0:
        try:
            adjust_text(texts, ax=ax,
                       arrowprops=dict(arrowstyle='->', color='red', lw=0.5, alpha=0.5))
        except Exception as e:
            print(f"Warning: Could not adjust text labels: {e}")

    # Set bounds to Jordan extent - DO THIS BEFORE ASPECT RATIO
    bounds = country_boundary.total_bounds
    ax.set_xlim(bounds[0] - 0.1, bounds[2] + 0.1)
    ax.set_ylim(bounds[1] - 0.1, bounds[3] + 0.1)

    # Set aspect ratio to equal - DO THIS AFTER BOUNDS
    ax.set_aspect('equal', adjustable='box')

    # Labels and title
    ax.set_xlabel('Longitude (°E)', fontsize=11)
    ax.set_ylabel('Latitude (°N)', fontsize=11)

    title_text = f"{mode_title} - {network} Network ({len(facilities_hsa)} HSAs)"
    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='gray', zorder=0)

    # Add coordinate labels
    ax.tick_params(labelsize=9)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#00AA00', alpha=0.25, edgecolor='black', linewidth=0.5, label='HSA populated area (clipped to WorldPop)'),
        mpatches.Patch(facecolor='none', edgecolor='#FF0000', linewidth=1.8, linestyle='dashed', label='Full service radius (reference)'),
        mpatches.Circle((0, 0), 0.1, facecolor='#FF0000', edgecolor='white', linewidth=0.8, label='HSA Anchor Facility'),
        mpatches.Circle((0, 0), 0.1, facecolor='#0000FF', alpha=0.5, label='All Facilities'),
        mpatches.Patch(facecolor='none', edgecolor='black', linewidth=2, label='Country Boundary'),
        mpatches.Patch(facecolor='none', edgecolor='gray', linewidth=0.8, label='Governorate Boundary'),
        mpatches.Patch(facecolor='gray', alpha=0.5, label='Population Density'),
    ]

    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                      frameon=True, framealpha=1.0, edgecolor='black',
                      facecolor='white', shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {Path(output_path).name}")

    return fig


def create_all_hsa_maps(
    all_results,
    network='INF',
    data_dir=None,
    out_dir=None,
    target_crs='EPSG:4326',
    algo_version='v7',
):
    """
    Create maps for all optimization modes.

    Parameters:
    -----------
    all_results : dict
        Dictionary mapping mode names to optimization results
    network : str
        Network type (e.g., 'INF', 'NCD')
    data_dir : Path or str
        Directory containing input data files
    out_dir : Path or str
        Directory to save output maps
    target_crs : str
        Target CRS for all layers

    Returns:
    --------
    dict
        Dictionary mapping mode names to figure objects
    """
    from pathlib import Path

    if data_dir is None:
        data_dir = Path.cwd() / 'data'
    else:
        data_dir = Path(data_dir)

    if out_dir is None:
        out_dir = Path.cwd() / 'out'
    else:
        out_dir = Path(out_dir)

    # Load base layers
    print("Loading base layers...")

    # Country boundary
    country_file = data_dir / 'jordan_boundary.gpkg'
    country_boundary = gpd.read_file(country_file)

    # Governorates
    gov_file = data_dir / 'jordan_governorates.gpkg'
    governorates = gpd.read_file(gov_file)

    # Population raster
    pop_raster = data_dir / 'jor_ppp_2020_constrained.tif'

    import pandas as pd
    # All facilities
    fac_gpkg = data_dir / f'{network}_hospitals_projected_total_WITH_CLIMATE.gpkg'
    if fac_gpkg.exists():
        all_facilities = gpd.read_file(fac_gpkg)
    else:
        # Fallback to climate features CSV (or facility coordinates) if GPKG is missing.
        # Climate CSV is written to out_dir by GEE_local_Climate_Features_by_Facilities.ipynb.
        climate_csv = out_dir / f'{network}_Facilities_Climate_Features_with_clusters.csv'
        if not climate_csv.exists():
            climate_csv = data_dir / f'{network}_Facilities_Climate_Features_with_clusters.csv'
        # Try SYNMOD-prefixed coords first (synthetic data), then real, then legacy SYN prefix
        for _prefix in (f'SYNMOD{network}', f'SYN{network}', network):
            coords_csv = data_dir / f'{_prefix}_facility_coordinates.csv'
            if coords_csv.exists():
                break
        if climate_csv.exists():
            fac_df = pd.read_csv(climate_csv)
            if 'FacilityName' in fac_df.columns:
                fac_df['HealthFacility'] = fac_df['FacilityName']
        elif coords_csv.exists():
            fac_df = pd.read_csv(coords_csv)
        else:
            raise FileNotFoundError(
                f"Missing facility inputs: {fac_gpkg}, {climate_csv}, {coords_csv}"
            )
        lat_col = 'lat' if 'lat' in fac_df.columns else 'Latitude'
        lon_col = 'lon' if 'lon' in fac_df.columns else 'Longitude'
        if lat_col not in fac_df.columns or lon_col not in fac_df.columns:
            raise ValueError("Facility input missing lat/lon columns for mapping.")
        all_facilities = gpd.GeoDataFrame(
            fac_df,
            geometry=gpd.points_from_xy(fac_df[lon_col], fac_df[lat_col]),
            crs='EPSG:4326'
        )

    print(f"Loaded base layers (CRS: {country_boundary.crs})")

    # Mode titles
    modes_to_map = {
        'fewest': 'FEWEST HSAs',
        'footprint': 'FOOTPRINT HSAs',
        'distance': 'DISTANCE HSAs',
        'governorate_tau_coverage': 'GOVERNORATE TAU COVERAGE HSAs',
        'governorate_fewest': 'GOVERNORATE FEWEST HSAs'
    }

    figures = {}

    for mode_key, mode_title in modes_to_map.items():
        if mode_key not in all_results:
            print(f"Skipping {mode_title} - not in results")
            continue

        print(f"\nCreating map for {mode_title}...")

        # Get facilities for this mode
        facilities_hsa = all_results[mode_key]['facilities']

        # Load population-clipped HSA polygons from saved GeoJSON if available
        geojson_path = out_dir / f'{network}_{mode_key}_hsas_{algo_version}.geojson'
        if geojson_path.exists():
            hsa_polygons = gpd.read_file(geojson_path)
            print(f"  Using population-clipped polygons from {geojson_path.name}")
        else:
            hsa_polygons = None
            print(f"  GeoJSON not found ({geojson_path.name}), falling back to circles")

        # Create output path
        output_path = out_dir / f'{network}_{mode_key}_map_{algo_version}.png'

        # Create map
        fig = create_hsa_map(
            mode_name=mode_key,
            mode_title=mode_title,
            facilities_hsa=facilities_hsa,
            all_facilities=all_facilities,
            country_boundary=country_boundary,
            governorates=governorates,
            pop_raster_path=pop_raster,
            output_path=output_path,
            network=network,
            target_crs=target_crs,
            hsa_polygons=hsa_polygons,
        )

        figures[mode_key] = fig
        plt.show()

    print("\n" + "="*80)
    print("ALL MAPS CREATED SUCCESSFULLY")
    print("="*80)

    return figures


def save_hsa_geopackages(
    all_results,
    network='INF',
    data_dir=None,
    out_dir=None,
    target_crs='EPSG:4326',
    algo_version='v7',
):
    """
    Save one GeoPackage per optimization mode containing all map layers.

    Layers per GPKG:
        hsa_boundaries  — population-clipped HSA polygons (from GeoJSON) or
                          degree-circles clipped to country if GeoJSON not found
        hsa_circles     — full service-radius circles clipped to country boundary
        hsa_anchors     — selected HSA anchor facility points
        all_facilities  — all facilities in the network (points)
        country_boundary
        governorates

    All layers are reprojected to `target_crs` (default EPSG:4326).

    Parameters
    ----------
    all_results : dict
        Dictionary mapping mode names to optimization results (each entry has
        a 'facilities' GeoDataFrame).
    network : str
        Network type prefix, e.g. 'INF' or 'NCD'.
    data_dir : Path or str
        Directory containing input data files.
    out_dir : Path or str
        Directory where GeoPackages and GeoJSON files are located / will be
        saved.
    target_crs : str
        Output CRS for all layers.

    Returns
    -------
    list of Path
        Paths of the written GeoPackage files.
    """
    import pandas as pd

    if data_dir is None:
        data_dir = Path.cwd() / 'data'
    else:
        data_dir = Path(data_dir)

    if out_dir is None:
        out_dir = Path.cwd() / 'out'
    else:
        out_dir = Path(out_dir)

    # ------------------------------------------------------------------
    # Load shared base layers once
    # ------------------------------------------------------------------
    country_boundary = gpd.read_file(data_dir / 'jordan_boundary.gpkg').to_crs(target_crs)
    governorates = gpd.read_file(data_dir / 'jordan_governorates.gpkg').to_crs(target_crs)
    country_polygon = country_boundary.unary_union

    fac_gpkg = data_dir / f'{network}_hospitals_projected_total_WITH_CLIMATE.gpkg'
    if fac_gpkg.exists():
        all_facilities = gpd.read_file(fac_gpkg).to_crs(target_crs)
    else:
        climate_csv = out_dir / f'{network}_Facilities_Climate_Features_with_clusters.csv'
        if not climate_csv.exists():
            climate_csv = data_dir / f'{network}_Facilities_Climate_Features_with_clusters.csv'
        for _prefix in (f'SYNMOD{network}', f'SYN{network}', network):
            coords_csv = data_dir / f'{_prefix}_facility_coordinates.csv'
            if coords_csv.exists():
                break
        if climate_csv.exists():
            fac_df = pd.read_csv(climate_csv)
            if 'FacilityName' in fac_df.columns:
                fac_df['HealthFacility'] = fac_df['FacilityName']
        elif coords_csv.exists():
            fac_df = pd.read_csv(coords_csv)
        else:
            raise FileNotFoundError(
                f"Missing facility inputs: {fac_gpkg}, {climate_csv}, {coords_csv}"
            )
        lat_col = 'lat' if 'lat' in fac_df.columns else 'Latitude'
        lon_col = 'lon' if 'lon' in fac_df.columns else 'Longitude'
        all_facilities = gpd.GeoDataFrame(
            fac_df,
            geometry=gpd.points_from_xy(fac_df[lon_col], fac_df[lat_col]),
            crs='EPSG:4326',
        ).to_crs(target_crs)

    written = []

    modes_to_save = {
        'fewest': 'FEWEST HSAs',
        'footprint': 'FOOTPRINT HSAs',
        'distance': 'DISTANCE HSAs',
        'governorate_tau_coverage': 'GOVERNORATE TAU COVERAGE HSAs',
        'governorate_fewest': 'GOVERNORATE FEWEST HSAs',
    }

    for mode_key in modes_to_save:
        if mode_key not in all_results:
            print(f"Skipping {mode_key} — not in results")
            continue

        gpkg_path = out_dir / f'{network}_{mode_key}_map_{algo_version}.gpkg'
        print(f"\nSaving GeoPackage: {gpkg_path.name}")

        facilities_hsa = all_results[mode_key]['facilities'].copy()
        facilities_hsa = facilities_hsa.to_crs(target_crs)

        # --- hsa_circles: degree-based circles clipped to country boundary ---
        circles_gdf = create_hsa_circles_clipped(facilities_hsa, country_polygon)

        # --- hsa_boundaries: population-clipped polygons from saved GeoJSON ---
        geojson_path = out_dir / f'{network}_{mode_key}_hsas_{algo_version}.geojson'
        if geojson_path.exists():
            hsa_boundaries = gpd.read_file(geojson_path).to_crs(target_crs)
            print(f"  hsa_boundaries: population-clipped ({geojson_path.name})")
        else:
            hsa_boundaries = circles_gdf.copy()
            print(f"  hsa_boundaries: GeoJSON not found, using circles as fallback")

        # Write layers: first write creates the file ('w'), subsequent writes
        # append new layers ('a').  Without mode='a' every call overwrites the
        # whole file and only the last layer would survive.
        def _write_layer(gdf, path, layer, mode):
            gdf = gdf.copy()
            # Drop circle_geometry_wkt (very long WKT strings; not needed in GPKG)
            if 'circle_geometry_wkt' in gdf.columns:
                gdf = gdf.drop(columns=['circle_geometry_wkt'])
            # GPKG/SQLite column names are case-insensitive.
            # Drop duplicate columns that differ only by case (keep first occurrence).
            seen = {}
            drop_cols = []
            for col in gdf.columns:
                key = col.lower()
                if key in seen:
                    drop_cols.append(col)
                else:
                    seen[key] = col
            if drop_cols:
                gdf = gdf.drop(columns=drop_cols)
            gdf.to_file(path, layer=layer, driver='GPKG', mode=mode)

        _write_layer(circles_gdf,     gpkg_path, 'hsa_circles',     'w')
        _write_layer(hsa_boundaries,  gpkg_path, 'hsa_boundaries',  'a')
        _write_layer(facilities_hsa,  gpkg_path, 'hsa_anchors',     'a')
        _write_layer(all_facilities,  gpkg_path, 'all_facilities',  'a')
        _write_layer(country_boundary, gpkg_path, 'country_boundary', 'a')
        _write_layer(governorates,    gpkg_path, 'governorates',    'a')

        print(f"  Layers: hsa_circles, hsa_boundaries, hsa_anchors, "
              f"all_facilities, country_boundary, governorates")
        print(f"  Saved: {gpkg_path}")
        written.append(gpkg_path)

    print("\n" + "="*80)
    print(f"GEOPACKAGES SAVED: {len(written)}")
    print("="*80)
    return written
