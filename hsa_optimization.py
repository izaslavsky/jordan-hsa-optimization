#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HSA Optimization Script
Converted from hsa_v3_9_nooutput.ipynb for easier testing
"""

# ============ Cell 3 ============
# Standard imports

import os

import sys

import warnings

warnings.filterwarnings('ignore')



# Core libraries

import numpy as np

import pandas as pd

import geopandas as gpd

from shapely.geometry import Point, Polygon

from shapely.ops import unary_union

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns

from datetime import datetime

import rasterio

from rasterio import open as rasterio_open

from rasterio.mask import mask as rasterio_mask

from rasterio.windows import Window

from typing import List, Tuple, Dict, Optional, Union, Any, Set

from collections import defaultdict

from tqdm import tqdm

import pickle



# Set style

plt.style.use('seaborn-v0_8-darkgrid')

sns.set_palette('husl')

# ============ Cell 5 ============
# ============================================================================

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
# NOTE: All parameters are now defined in the notebook (Cell 1) and
# imported into this module's global namespace when the notebook runs.
# The notebook sets these variables before importing this module:
#
# File paths: DATA_DIR, OUT_DIR, MAP_DIR, FACILITIES_*_PATH, etc.
# Core algorithm: TAU_COVERAGE, MAX_RADIUS_KM, RADIUS_STEP_KM, RNG_SEED
# Efficiency: BUFFER_KM, COARSEN_FACTOR
# Density classification: DENSITY_RADIUS_KM, URBAN_DENSITY_THRESH
# Base radii: URBAN_BASE_RADIUS_KM, RURAL_BASE_RADIUS_KM
# HSA targets: TARGET_HSA_MIN, TARGET_HSA_MAX
# Volume scoring: VOLUME_WEIGHT, PATIENT_RADIUS_ALPHA
# Network multipliers: NETWORK_RADIUS_MULTIPLIERS
# Climate diversity: CLIMATE_DIVERSITY_ON, CLIMATE_K, CLIMATE_MIN_PER_CLUSTER,
#                    CLIMATE_CLUSTER_COL, CLIMATE_CSV
# Scoring weights: WEIGHT_COVERAGE, WEIGHT_OVERLAP_PENALTY, WEIGHT_CLIMATE,
#                  WEIGHT_PATIENT_VOLUME, WEIGHT_COVERAGE_PROGRESS,
#                  WEIGHT_DISTANCE_PENALTY, MODE_WEIGHT_PROFILES
#
# To modify parameters, edit Cell 1 in the notebook, not this file.
# ============================================================================

class GeoUtils:

    """Geographic utility functions."""

    

    @staticmethod

    def haversine_km(lon1, lat1, lon2, lat2):

        """Calculate great-circle distance in km."""

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1

        dlat = lat2 - lat1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2

        c = 2 * np.arcsin(np.sqrt(a))

        return 6371 * c

    

    @staticmethod

    def buffer_polygon_km(polygon, km):

        """Buffer a polygon by kilometers (approximate)."""

        # Rough conversion: 1 degree ≈ 111 km

        buffer_deg = km / 111.0

        return polygon.buffer(buffer_deg)





# --- imports (module-level, not inside the class) ---

from affine import Affine

import rasterio.windows as rwin

from rasterio.warp import transform_bounds



class PopulationRaster:

    """Handle population raster operations."""



    def __init__(self, pop_path, coarsen=1):

        self.path = pop_path

        self.coarsen = coarsen

        with rasterio_open(self.path) as src:

            self.crs = src.crs

            self.ds_bounds = src.bounds  # (left, bottom, right, top)



    def get_cropped(self, bounds_wgs84):

        """

        bounds_wgs84 = [minx, miny, maxx, maxy] in EPSG:4326.

        Returns (arr, transform).

        """

        with rasterio_open(self.path) as src:

            # Reproject WGS84 bounds into the raster CRS if needed

            if (self.crs is not None) and (str(self.crs).upper() not in ("EPSG:4326", "WGS84")):

                minx, miny, maxx, maxy = transform_bounds("EPSG:4326", self.crs, *bounds_wgs84, densify_pts=21)

            else:

                minx, miny, maxx, maxy = bounds_wgs84



            # Intersect with dataset bounds (both in raster CRS here)

            ds_left, ds_bottom, ds_right, ds_top = self.ds_bounds

            ix_minx = max(minx, ds_left)

            ix_miny = max(miny, ds_bottom)

            ix_maxx = min(maxx, ds_right)

            ix_maxy = min(maxy, ds_top)



            # Debug

            print(f"  Raster CRS: {self.crs}")

            print(f"  Reprojected bounds (raster CRS): {(minx, miny, maxx, maxy)}")

            print(f"  Dataset bounds (raster CRS): {(ds_left, ds_bottom, ds_right, ds_top)}")

            print(f"  Intersection bounds: {(ix_minx, ix_miny, ix_maxx, ix_maxy)}")



            # No overlap -> empty

            if (ix_minx >= ix_maxx) or (ix_miny >= ix_maxy):

                print("  [WARNING] Crop bounds do not intersect raster. Returning empty array.")

                return np.zeros((0, 0), dtype="int64"), Affine(0, 0, 0, 0, 0, 0)



            # Build window; guard against zero-sized due to edge rounding

            window = rwin.from_bounds(ix_minx, ix_miny, ix_maxx, ix_maxy, transform=src.transform)

            if (window.width == 0) or (window.height == 0):

                # Convert to indices and pad one pixel

                col_min, row_max = src.index(ix_minx, ix_miny)

                col_max, row_min = src.index(ix_maxx, ix_maxy)

                c0, c1 = sorted([col_min, col_max])

                r0, r1 = sorted([row_min, row_max])

                c0 = max(c0 - 1, 0); r0 = max(r0 - 1, 0)

                c1 = min(c1 + 1, src.width); r1 = min(r1 + 1, src.height)

                if c0 >= c1 or r0 >= r1:

                    print("  [WARNING] Computed pixel window is empty after padding. Returning empty array.")

                    return np.zeros((0, 0), dtype="int64"), Affine(0, 0, 0, 0, 0, 0)

                window = rwin.Window.from_slices((r0, r1), (c0, c1))



            arr = src.read(1, window=window, masked=True).filled(0).astype("int64")

            transform = src.window_transform(window)

            

            # ↓↓↓ Apply grid coarsening if requested

            factor = getattr(self, "coarsen", 1)

            if factor and factor > 1:

                # sum population in blocks

                arr = PopulationRaster._coarsen_array(arr, factor)

                # scale the affine (pixel size ↑ by factor, origin unchanged)

                transform = Affine(

                    transform.a * factor, transform.b, transform.c,

                    transform.d, transform.e * factor, transform.f

                )

            

            return arr, transform





    def get_full(self):

        """Get full population raster."""

        with rasterio_open(self.path) as src:

            pop_arr = src.read(1, masked=True).filled(0).astype('int64')

            transform = src.transform

        return pop_arr, transform



    @staticmethod

    def _coarsen_array(arr, factor):

        """Coarsen array by summing blocks."""

        h, w = arr.shape

        if factor <= 1 or h < factor or w < factor:

            return arr

        h_new = h // factor

        w_new = w // factor

        result = np.zeros((h_new, w_new), dtype='int64')

        for i in range(h_new):

            for j in range(w_new):

                result[i, j] = np.sum(arr[i*factor:(i+1)*factor, j*factor:(j+1)*factor])

        return result


# ============ Cell 8 ============
def run_climate_ablation(optimizer, facilities, objective: str, network_type: str, governorates_gdf=None):

    """

    Run the same optimization twice: with and without climate reward.

    Returns a dict with side-by-side metrics to quantify climate's role.

    """

    global CLIMATE_DIVERSITY_ON, LAMBDA_CLIMATE



    # Save originals

    _orig_on = CLIMATE_DIVERSITY_ON

    _orig_lambda = LAMBDA_CLIMATE



    # With climate

    CLIMATE_DIVERSITY_ON = True

    LAMBDA_CLIMATE = _orig_lambda

    if objective == 'governorate':

        res_with = optimizer.optimize(facilities.copy(), objective=objective, network_type=network_type, governorates_gdf=governorates_gdf)

    else:

        res_with = optimizer.optimize(facilities.copy(), objective=objective, network_type=network_type)



    # Without climate

    CLIMATE_DIVERSITY_ON = False

    LAMBDA_CLIMATE = 0.0

    if objective == 'governorate':

        res_no = optimizer.optimize(facilities.copy(), objective=objective, network_type=network_type, governorates_gdf=governorates_gdf)

    else:

        res_no = optimizer.optimize(facilities.copy(), objective=objective, network_type=network_type)



    # Restore

    CLIMATE_DIVERSITY_ON = _orig_on

    LAMBDA_CLIMATE = _orig_lambda



    return {

        "with_climate": res_with,

        "no_climate": res_no,

    }

# ============ Cell 10 ============
def compute_local_density(fac: gpd.GeoDataFrame, pop_raster_path: str, radius_km: float = 10.0):

    """

    Compute population density within fixed radius of each facility.

    

    Args:

        fac: Facilities GeoDataFrame

        pop_raster_path: Path to population raster

        radius_km: Fixed radius for density calculation (10km)

    

    Returns:

        Array of population densities (people/km²)

    """

    print(f"  Computing local density within {radius_km}km radius...")

    

    densities = np.zeros(len(fac))

    

    # Ensure facilities are in WGS84 for haversine

    if fac.crs is None:

        print("    [WARNING] Facilities GeoDataFrame has no CRS; assuming EPSG:4326.")

    else:

        if str(fac.crs).upper() not in ("EPSG:4326", "WGS84"):

            fac = fac.to_crs("EPSG:4326")

    

    with rasterio_open(pop_raster_path) as src:

        # print(f"    Raster CRS: {raster_crs}")

        # Get population data

        pop_arr = src.read(1, masked=True).filled(0).astype('int64')

        transform = src.transform

        raster_crs = src.crs

            

        # Get coordinates of populated cells

        rows, cols = np.where(pop_arr > 0)

        if len(rows) == 0:

            print("    Warning: No population data found (all zeros in raster band).")

            return densities

        

        # Cell-center coordinates in raster CRS

        xs = transform.c + (cols + 0.5) * transform.a

        ys = transform.f + (rows + 0.5) * transform.e

        

        # Reproject raster cell centers to WGS84 if needed

        try:

            crs_is_wgs84 = (raster_crs is not None) and (

                str(raster_crs).upper() in ("EPSG:4326", "WGS84") or getattr(raster_crs, "is_geographic", False)

            )

        except Exception:

            crs_is_wgs84 = False

        

        if not crs_is_wgs84:

            from pyproj import Transformer

            transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)

            xs, ys = transformer.transform(xs, ys)

        

        pop_vals = pop_arr[rows, cols]

        

        # Compute area of circle

        area_km2 = np.pi * radius_km ** 2

        

        # For each facility

        for i, (idx, row) in enumerate(fac.iterrows()):

            fac_lon = row.geometry.x

            fac_lat = row.geometry.y

            

            # Find population within radius (inputs now guaranteed lon/lat degrees)

            dist = GeoUtils.haversine_km(fac_lon, fac_lat, xs, ys)

            within = np.isfinite(dist) & (dist <= radius_km)

            

            if within.any():

                total_pop = np.sum(pop_vals[within])

                densities[i] = total_pop / area_km2

            else:

                densities[i] = 0.0

    

    print(f"    Density range: {densities.min():.1f} - {densities.max():.1f} people/km²")

    return densities





def choose_radius_km_for_facilities(fac: gpd.GeoDataFrame, 

                                   pop_raster_path: str,

                                   network_type: str = 'NCD'):

    """

    Determine initial service radius for each facility.

    

    Classification based ONLY on population density (not patient volume).

    

    Args:

        fac: Facilities with geometry and Patients columns

        pop_raster_path: Path to population raster

        network_type: 'INF' or 'NCD' for network-specific multiplier

    

    Returns:

        Tuple of (radii, urban_rural):

            - radii: Array of radii in km

            - urban_rural: Array of 'Urban' or 'Rural' classifications

    """

    print(f"\nAssigning radii for {len(fac)} {network_type} facilities...")

    

    # Step 1: Compute local density within FIXED 10km radius

    local_density = compute_local_density(fac, pop_raster_path, radius_km=DENSITY_RADIUS_KM)

    

    # Step 2: Classify as urban/rural based on density ONLY

    is_urban = local_density > URBAN_DENSITY_THRESH

    n_urban = is_urban.sum()

    n_rural = (~is_urban).sum()

    

    print(f"  Classification (density-based only):")

    print(f"    Urban (density > {URBAN_DENSITY_THRESH}): {n_urban} facilities")

    print(f"    Rural: {n_rural} facilities")

    

    # Step 3: Assign base radius

    radii = np.where(is_urban, URBAN_BASE_RADIUS_KM, RURAL_BASE_RADIUS_KM)



    # Step 4: Apply patient volume scaling

    # Scale radius based on patient volume (larger hospitals get larger radii)

    if 'Total' in fac.columns:

        patient_volumes = pd.to_numeric(fac['Total'], errors='coerce').fillna(0).values

        if patient_volumes.max() > 0:

            # Normalize volumes to 0-1 range

            volume_norm = patient_volumes / patient_volumes.max()



            # Apply scaling: 0.8x to 1.2x based on volume

            # Small hospitals (low volume): 0.8x multiplier (smaller radius)

            # Large hospitals (high volume): 1.2x multiplier (larger radius)

            volume_scale = 0.8 + (0.4 * volume_norm)  # Range: 0.8 to 1.2



            radii = radii * volume_scale



            print(f"  Applied patient volume scaling:")

            print(f"    Volume range: {patient_volumes.min():.0f} - {patient_volumes.max():.0f} patients")

            print(f"    Radius multiplier range: 0.80x - 1.20x")



    # Step 5: Apply network-specific multiplier

    network_mult = NETWORK_RADIUS_MULTIPLIERS.get(network_type, 1.0)

    radii = radii * network_mult

    print(f"  Applied {network_type} network multiplier: {network_mult}x")

    

    # Step 6: Cap at maximum

    radii = np.minimum(radii, MAX_RADIUS_KM)

    

    print(f"  Final radius range: {radii.min():.1f} - {radii.max():.1f} km")

    print(f"  Mean radius: {radii.mean():.1f} km")



    # Step 7: Create urban/rural classification labels

    urban_rural = np.where(is_urban, 'Urban', 'Rural')



    return radii, urban_rural

# ============ Cell 13 ============
def _km_per_pixel(transform, lat_deg: float):

    # longitude scale shrinks by cos(lat)

    from math import cos, pi

    # meters per degree ≈ 111_320 * cos(lat) for lon, 110_574 for lat (rough)

    lat_m_per_deg = 110_574.0

    lon_m_per_deg = 111_320.0 * max(0.1, cos(lat_deg * pi / 180.0))



    # transform.a ~ degrees per pixel in x; transform.e ~ degrees per pixel in y (negative)

    km_x = abs(transform.a) * lon_m_per_deg / 1000.0

    km_y = abs(transform.e) * lat_m_per_deg / 1000.0

    return max(km_x, km_y)

def _pairwise_km(pts):

    # pts: Nx2 array of lon,lat in degrees

    N = len(pts)

    out = np.full((N, N), np.inf, dtype=float)

    for i in range(N):

        lon1, lat1 = pts[i]

        # broadcast haversine against all j

        d = GeoUtils.haversine_km(lon1, lat1, pts[:,0], pts[:,1])

        out[i,:] = d

        out[i,i] = np.inf

    return out



def build_radius_candidates(fac_gdf: gpd.GeoDataFrame, transform, 

                            min_km=5.0, max_km=25.0, base=[6,8,10,12,14,16,18,20,22,24]):

    # center latitude

    lat0 = float(fac_gdf.geometry.y.mean())

    px_km = _km_per_pixel(transform, lat0)

    step = max(2.0 * px_km, 1.0)  # at least 1km, but no smaller than twice pixel

    # nearest-neighbor distances

    pts = np.c_[fac_gdf.geometry.x.values, fac_gdf.geometry.y.values]

    D = _pairwise_km(pts)

    nn = np.min(D, axis=1)

    # meaningful breakpoints: quantiles of nearest neighbor distances

    qs = np.quantile(nn, [0.2, 0.4, 0.6, 0.8])

    cand = set(base)

    cand.update(qs.tolist())

    # snap to step, clamp to [min_km, max_km]

    grid = []

    for r in sorted(cand):

        r2 = max(min_km, min(max_km, round(r / step) * step))

        grid.append(r2)

    return sorted(set(round(x, 2) for x in grid))


# ============ Cell 15 ============
class HSAOptimizer:

    """Main HSA optimization class."""

    

    def __init__(self, config):
        # Support both dict config and direct PopulationRaster object
        if isinstance(config, dict):
            self.config = config
            self.pop = PopulationRaster(config['pop_path'], config.get('coarsen', 1))
        else:
            # config is actually a PopulationRaster object
            self.pop = config
            # Create minimal config with defaults and pop_path from the PopulationRaster object
            self.config = {
                'pop_path': config.path if hasattr(config, 'path') else None,
                'tau_coverage': TAU_COVERAGE,  # 0.90
                'coarsen': 1
            }

        

    def optimize(self, facilities: gpd.GeoDataFrame, 

                 objective: str = 'fewest',

                 network_type: str = 'NCD') -> Dict:

        """

        Run optimization to select facilities.

        

        Args:

            facilities: All candidate facilities

            objective: 'fewest', 'footprint', or 'distance'

            network_type: 'INF' or 'NCD'

        

        Returns:

            Dictionary with selected facilities and metrics

        """

        print(f"\n{'='*60}")

        print(f"Running {objective.UPPER()} optimization for {network_type}")

        print(f"{'='*60}")



        # Get initial radii and urban/rural classification

        radii, urban_rural = choose_radius_km_for_facilities(

            facilities,

            self.config['pop_path'],

            network_type

        )

        facilities['initial_radius_km'] = radii

        facilities['urban_rural'] = urban_rural

        

        # Ensure facilities are in WGS84 (degrees) before building WGS84 bounds

        if facilities.crs is None:

            print("  [WARNING] Facilities GeoDataFrame has no CRS; assuming EPSG:4326 for bounds.")

        elif str(facilities.crs).upper() not in ("EPSG:4326", "WGS84"):

            facilities = facilities.to_crs("EPSG:4326")

        

        # Get population data - USE FULL JORDAN RASTER (don't crop to facilities!)

        # This ensures we don't lose western Jordan or any remote areas

        print(f"  Facilities CRS: {facilities.crs}")



        # Use full Jordan bounds instead of facility bounds

        # Jordan extent: ~34.88-39.20 E, ~29.18-33.38 N

        full_jordan_bounds = [34.5, 29.0, 39.5, 33.5]  # Padded full country bounds



        print(f"  Using FULL Jordan bounds (not facility-cropped): {full_jordan_bounds}")



        pop_arr, transform = self.pop.get_cropped(full_jordan_bounds)



        radius_grid = np.arange(8, 36, 2).tolist()  # [8, 10, 12, ..., 34]







        

        print(f"  FULL Jordan raster sum: {int(pop_arr.sum()):,}")

        print(f"  Full raster shape: {pop_arr.shape}, transform: {transform}")



        total_pop = int(np.sum(pop_arr))

        target_pop = int(total_pop * self.config['tau_coverage'])

        

        print(f"\nPopulation statistics:")

        print(f"  Total population: {total_pop:,}")

        print(f"  Target coverage ({self.config['tau_coverage']*100:.0f}%): {target_pop:,}")

        

        # Guard: if the cropped raster has zero population, skip heavy work and avoid 0-division

        if total_pop == 0:

            print("  Warning: population raster in bounds sums to 0. Skipping optimization.")

            # Return inputs with a reasonable radius so downstream code has consistent columns

            out = facilities.copy()

            if 'initial_radius_km' not in out.columns:

                # fallback if radius not assigned yet

                out['initial_radius_km'] = MAX_RADIUS_KM

            out['service_radius_km'] = out['initial_radius_km']

            return {

                'facilities': out,

                'total_pop': 0,

                'covered_pop': 0,

                'coverage_pct': 0.0,

                'objective': objective,

                'network_type': network_type

            }

                

        # Run appropriate algorithm

        if objective == 'fewest':

                selected = self._optimize_unified(facilities, pop_arr, transform, target_pop, mode='fewest', mode_weights=None, network_type=network_type)

        elif objective == 'footprint':

                selected = self._optimize_unified(facilities, pop_arr, transform, target_pop, mode='footprint', mode_weights=None, network_type=network_type)

        elif objective == 'distance':

                selected = self._optimize_unified(facilities, pop_arr, transform, target_pop, mode='distance', mode_weights=None, network_type=network_type)

        else:

            raise ValueError(f"Unknown objective: {objective}")

        

        # Compute coverage

        covered_pop = self._compute_coverage(selected, pop_arr, transform)

        coverage_pct = 0.0 if total_pop == 0 else (covered_pop / total_pop) * 100



        

        print(f"\nOptimization complete:")

        print(f"  Selected facilities: {len(selected)}")

        print(f"  Population covered: {covered_pop:,} ({coverage_pct:.1f}%)")


        # === POST-PROCESSING: Remove overlapping HSAs for ALL modes ===
        print(f"\nPost-processing {objective.upper()}: Removing overlapping HSAs...")
        selected = self._remove_overlapping_hsas(
            selected, pop_arr, transform, overlap_threshold=0.80
        )
        # Recalculate coverage after removing overlapping facilities
        covered_pop = self._compute_coverage(selected, pop_arr, transform)
        coverage_pct = 100.0 * covered_pop / total_pop
        print(f"  Final facilities after overlap removal: {len(selected)}")
        print(f"  Final coverage: {covered_pop:,} ({coverage_pct:.1f}%)")


        return {

            'facilities': selected,

            'total_pop': total_pop,

            'covered_pop': covered_pop,

            'coverage_pct': coverage_pct,

            'objective': objective,

            'network_type': network_type

        }

    

    def _optimize_unified(self, fac, pop_arr, transform, target_pop,
                          mode='fewest', mode_weights=None, network_type='INF'):
        """
        Unified greedy optimization for FEWEST/FOOTPRINT/DISTANCE modes.

        All modes now use:
        - Same radius assignment (volume-based with global constants)
        - Same stopping condition (90% coverage, >=10 HSAs)
        - Same scoring formula (weighted by MODE_WEIGHT_PROFILES)

        The ONLY difference is:
        - DISTANCE mode computes mean travel distance and adds distance penalty to score

        Parameters:
        -----------
        fac : GeoDataFrame
            Facilities to optimize
        pop_arr : np.ndarray
            Population raster array
        transform : affine.Affine
            Raster geotransform
        target_pop : float
            Target population to cover
        mode : str
            'fewest', 'footprint', or 'distance'
        mode_weights : dict
            Mode-specific weight multipliers (from MODE_WEIGHT_PROFILES)
        network_type : str
            'INF' or 'NCD' (for error messages)

        Returns:
        --------
        GeoDataFrame
            Selected facilities with composite_score column
        """

        # Import constants from main module (these should be available in HSAOptimizer context)
        from hsa_optimization import (
            WEIGHT_COVERAGE, WEIGHT_OVERLAP_PENALTY, WEIGHT_CLIMATE,
            WEIGHT_PATIENT_VOLUME, WEIGHT_COVERAGE_PROGRESS, WEIGHT_DISTANCE_PENALTY,
            MODE_WEIGHT_PROFILES, CLIMATE_DIVERSITY_ON, CLIMATE_CLUSTER_COL,
            CLIMATE_MIN_PER_CLUSTER, TARGET_HSA_MIN, TAU_COVERAGE
        )

        # Get mode-specific weight profile
        if mode_weights is None:
            mode_weights = MODE_WEIGHT_PROFILES.get(mode, MODE_WEIGHT_PROFILES['fewest'])

        mode_name = mode.upper()
        print(f"\nRunning {mode_name} algorithm (unified implementation, adaptive radii)...")

        # Flatten population array
        pop_flat = pop_arr.ravel().astype('int64')
        total_pop_flat = float(pop_flat.sum())

        # ============================================================
        # STEP 1: Use pre-assigned initial radii as service radii
        # ============================================================

        print("  [1/4] Using pre-assigned initial radii as service radii...")
        fac_with_radii = fac.copy()

        # Use initial_radius_km as service_radius_km (preserve network-specific radii)
        if 'initial_radius_km' in fac_with_radii.columns:
            fac_with_radii['service_radius_km'] = fac_with_radii['initial_radius_km']
        else:
            # Fallback: assign radii if not already present
            # NOTE: This should use assign_radii_to_facilities, not assign_adaptive_radii
            raise ValueError(
                f"Facilities must have 'initial_radius_km' column. "
                f"Call assign_radii_to_facilities() before optimization."
            )

        print(f"    Radii assigned: {fac_with_radii['service_radius_km'].min():.1f} - "
              f"{fac_with_radii['service_radius_km'].max():.1f} km")
        print(f"    Mean radius: {fac_with_radii['service_radius_km'].mean():.1f} km")

        # ============================================================
        # STEP 2: Build per-facility coverage masks (+ distances if DISTANCE mode)
        # ============================================================

        print(f"  [2/4] Building per-facility coverage masks{' with distance metrics' if mode == 'distance' else ''}...")
        n_fac = len(fac_with_radii)
        masks = []

        # Initialize mean_dist array (populated only for DISTANCE mode)
        mean_dist = np.zeros(n_fac, dtype='float32')

        # For DISTANCE mode, also compute mean distances
        if mode == 'distance':
            # Create coordinate grids for distance calculation
            rows, cols = pop_arr.shape
            xs_grid, ys_grid = np.meshgrid(
                np.arange(cols),
                np.arange(rows)
            )
            # Convert to geographic coordinates
            xs, ys = transform * (xs_grid + 0.5, ys_grid + 0.5)

            for idx, row in enumerate(fac_with_radii.itertuples()):
                radius_km = row.service_radius_km
                mask = self._get_coverage_mask(row, pop_arr.shape, transform, radius_km)
                masks.append(mask.ravel())

                # Compute mean distance from this facility to its covered pixels
                from hsa_optimization import GeoUtils
                dist_km = GeoUtils.haversine_km(row.geometry.x, row.geometry.y, xs, ys).ravel()
                mean_d = float(np.nanmean(dist_km[mask.ravel()])) if mask.any() else 1e9
                mean_dist[idx] = mean_d
            print(f"    Built {n_fac} coverage masks with distance metrics")

        else:
            # FEWEST and FOOTPRINT modes - no distance computation
            for _, row in fac_with_radii.iterrows():
                radius_km = row['service_radius_km']
                mask = self._get_coverage_mask(row, pop_arr.shape, transform, radius_km)
                masks.append(mask.ravel())

            print(f"    Built {n_fac} coverage masks")

        if len(masks) == 0:
            print("    No valid facilities found")
            return fac_with_radii.iloc[:0]  # Return empty

        masks = np.stack(masks, axis=0)  # Shape: (n_fac, n_pixels)

        # Pre-compute coverage totals for each facility
        cov_totals = (masks * pop_flat).sum(axis=1, dtype='int64')

        # Normalize patient volumes for scoring
        if 'Total' in fac_with_radii.columns:
            patient_volumes = fac_with_radii['Total'].fillna(0).values.astype('float64')
            max_patients = patient_volumes.max()
            if max_patients > 0:
                patient_norm = patient_volumes / max_patients
            else:
                patient_norm = np.zeros(n_fac, dtype='float64')
        else:
            patient_norm = np.zeros(n_fac, dtype='float64')

        # Climate cluster data
        clusters = fac_with_radii.get(
            CLIMATE_CLUSTER_COL,
            pd.Series(np.full(n_fac, -1, dtype=int))
        ).astype(int).to_numpy()

        # ============================================================
        # STEP 3: Greedy selection loop
        # ============================================================

        print("  [3/4] Running greedy selection...")

        chosen_idx = []
        chosen_scores = []  # Track composite score for each selected facility
        # Track individual score components for each selected facility
        chosen_coverage_norm = []
        chosen_patient_norm = []
        chosen_overlap_frac = []
        chosen_climate_norm = []
        chosen_progress_norm = []
        chosen_distance_norm = []
        selected_mask = np.zeros(n_fac, dtype=bool)
        covered_flat = np.zeros(len(pop_flat), dtype=bool)

        # Initialize climate cluster counts
        cluster_counts = np.zeros(max(clusters.max() + 1, 1), dtype=int)

        # Check for pre-selected facilities (used by governorate_fewest mode)
        if 'pre_selected' in fac.columns:
            pre_selected_mask = fac['pre_selected'].values.astype(bool)
            pre_selected_indices = np.where(pre_selected_mask)[0]
            if len(pre_selected_indices) > 0:
                print(f"  Initializing with {len(pre_selected_indices)} pre-selected facilities...")
                for idx in pre_selected_indices:
                    chosen_idx.append(int(idx))
                    selected_mask[idx] = True
                    covered_flat |= masks[idx]
                    cluster_counts[clusters[idx]] += 1
                    # Initialize score components with zeros for pre-selected
                    chosen_scores.append(0.0)
                    chosen_coverage_norm.append(0.0)
                    chosen_patient_norm.append(0.0)
                    chosen_overlap_frac.append(0.0)
                    chosen_climate_norm.append(0.0)
                    chosen_progress_norm.append(0.0)
                    chosen_distance_norm.append(0.0)

        current_pop = 0.0
        # Mode-specific target coverage
        if mode in ['fewest', 'footprint', 'distance']:
            target_coverage_pct = 90.0  # 90% for main optimization modes
        else:
            target_coverage_pct = self.config['tau_coverage'] * 100.0  # 60% for governorate modes
        iteration = 0
        MAX_ITERATIONS = n_fac

        while iteration < MAX_ITERATIONS:
            iteration += 1

            # Check coverage
            current_pop = float(pop_flat[covered_flat].sum())
            coverage_pct = 100.0 * current_pop / total_pop_flat

            # MODE-SPECIFIC STOPPING CONDITIONS
            # FEWEST: Stop as soon as target coverage reached with minimum facilities
            # FOOTPRINT/DISTANCE: Continue to higher coverage (push toward 100%)
            if mode == 'fewest':
                # FEWEST: Stop at tau_coverage with at least TARGET_HSA_MIN facilities
                if coverage_pct >= target_coverage_pct and len(chosen_idx) >= TARGET_HSA_MIN:
                    print(f"    [STOPPED] Reached {target_coverage_pct:.0f}% coverage with {len(chosen_idx)} HSAs")
                    break
            else:
                # FOOTPRINT/DISTANCE: Continue until 20-22 facilities OR 98% coverage
                # This allows them to select more facilities for better footprint/distance
                target_max_hsas = 22  # Target for footprint/distance modes
                if len(chosen_idx) >= target_max_hsas or coverage_pct >= 98.0:
                    reason = f"{len(chosen_idx)} HSAs" if len(chosen_idx) >= target_max_hsas else f"{coverage_pct:.1f}% coverage"
                    print(f"    [STOPPED] {mode.upper()} mode reached {reason}")
                    break

            # Progress reporting
            if len(chosen_idx) > 0 and len(chosen_idx) % 5 == 0:
                print(f"    Selected {len(chosen_idx)} so far; {coverage_pct:.1f}% coverage.")

            # Early stop if no more viable facilities
            if selected_mask.all():
                print(f"    All facilities selected at iteration {iteration}")
                break

            # Calculate new coverage for each facility
            new_cover = masks & (~covered_flat)  # (n_fac, n_pixels)
            pop_gains = (new_cover * pop_flat).sum(axis=1, dtype='int64')

            # Overlap penalty
            already = (masks & covered_flat).sum(axis=1, dtype='int64')
            with np.errstate(divide='ignore', invalid='ignore'):
                overlap_frac = np.where(cov_totals > 0, already / cov_totals, 0.0)

            # ============================================================
            # NORMALIZE ALL COMPONENTS TO 0-1 RANGE
            # ============================================================

            # 1. Coverage gain (normalized by total population)
            coverage_norm = pop_gains / total_pop_flat  # 0-1 range

            # 2. Overlap fraction (already 0-1 range)
            # overlap_frac is already calculated above

            # 3. Patient volume (already normalized to 0-1)
            # patient_norm is already calculated above

            # 4. Climate diversity reward (normalize to 0-1)
            climate_norm = np.zeros(n_fac, dtype='float64')
            if CLIMATE_DIVERSITY_ON:
                for i in range(n_fac):
                    c = clusters[i]
                    if c >= 0:
                        # Reward underrepresented clusters more
                        if cluster_counts[c] < CLIMATE_MIN_PER_CLUSTER:
                            climate_norm[i] = 1.0  # Maximum reward for empty bins
                        else:
                            # Diminishing reward: 1.0 / (1 + count), normalized to 0-1
                            climate_norm[i] = 1.0 / (1.0 + cluster_counts[c])

            # 5. Coverage progress reward (normalized)
            if current_pop < target_pop:
                remaining = target_pop - current_pop
                # Normalize by the fraction of remaining coverage needed
                progress_norm = pop_gains / remaining  # How much this facility helps reach target
                progress_norm = np.minimum(progress_norm, 1.0)  # Cap at 1.0
            else:
                progress_norm = np.zeros(n_fac, dtype='float64')

            # 6. Distance penalty (DISTANCE mode only, normalized)
            if mode == 'distance':
                max_reasonable_dist = 100.0  # km
                distance_norm = np.minimum(mean_dist / max_reasonable_dist, 1.0)
            else:
                distance_norm = np.zeros(n_fac, dtype='float64')

            # ============================================================
            # COMPOSITE SCORE: Sum of weighted normalized components
            # All components are now 0-1, weights are 0-10
            # ============================================================

            score = (
                WEIGHT_COVERAGE * coverage_norm * mode_weights['coverage']
                - WEIGHT_OVERLAP_PENALTY * overlap_frac * mode_weights['overlap']
                + WEIGHT_CLIMATE * climate_norm
                + WEIGHT_PATIENT_VOLUME * patient_norm * mode_weights['patient_volume']
                + WEIGHT_COVERAGE_PROGRESS * progress_norm * mode_weights['coverage_progress']
                - WEIGHT_DISTANCE_PENALTY * distance_norm * mode_weights.get('distance', 0.0)
            )

            # Mask already-selected facilities
            score[selected_mask] = -np.inf

            # Select best facility
            best_idx = np.argmax(score)
            if not np.isfinite(score[best_idx]):
                print(f"    No more viable facilities at iteration {iteration}")
                break

            # Add to selection
            chosen_idx.append(best_idx)
            chosen_scores.append(float(score[best_idx]))  # Save the composite score
            # Save individual score components for analysis
            chosen_coverage_norm.append(float(coverage_norm[best_idx]))
            chosen_patient_norm.append(float(patient_norm[best_idx]))
            chosen_overlap_frac.append(float(overlap_frac[best_idx]))
            chosen_climate_norm.append(float(climate_norm[best_idx]))
            chosen_progress_norm.append(float(progress_norm[best_idx]))
            chosen_distance_norm.append(float(distance_norm[best_idx]))
            selected_mask[best_idx] = True
            covered_flat |= new_cover[best_idx, :]

            # Update climate cluster counts
            cluster_of_selected = clusters[best_idx]
            if cluster_of_selected >= 0:
                cluster_counts[cluster_of_selected] += 1

        # ============================================================
        # STEP 4: Finalize selection
        # ============================================================

        print("  [4/4] Finalizing selection...")

        if len(chosen_idx) == 0:
            print("    No facilities selected")
            return fac_with_radii.iloc[:0]

        selected_facilities = fac_with_radii.iloc[chosen_idx].copy()
        selected_facilities['composite_score'] = chosen_scores
        # Add individual score components for analysis
        selected_facilities['coverage_norm'] = chosen_coverage_norm
        selected_facilities['patient_norm'] = chosen_patient_norm
        selected_facilities['overlap_frac'] = chosen_overlap_frac
        selected_facilities['climate_norm'] = chosen_climate_norm
        selected_facilities['progress_norm'] = chosen_progress_norm
        selected_facilities['distance_norm'] = chosen_distance_norm

        # Add HSA-level metrics
        # Calculate population covered by each facility's service area
        hsa_populations = []
        for idx in chosen_idx:
            pop_covered = int((masks[idx] * pop_flat).sum())
            hsa_populations.append(pop_covered)
        selected_facilities['hsa_population'] = hsa_populations

        # Add climate cluster (will show actual cluster ID if present, -1 if not)
        selected_facilities['climate_k'] = clusters[chosen_idx]

        # Calculate population density for each HSA
        pop_densities = []
        for _, row in selected_facilities.iterrows():
            radius_km = row.get('service_radius_km', 0)
            pop = row.get('hsa_population', 0)
            if radius_km > 0:
                area_km2 = np.pi * radius_km ** 2
                density = pop / area_km2
            else:
                density = 0
            pop_densities.append(density)
        selected_facilities['hsa_pop_density_per_km2'] = pop_densities

        # Copy urban_rural classification if it exists
        if 'urban_rural' in fac_with_radii.columns:
            selected_facilities['urban_rural'] = fac_with_radii.loc[chosen_idx, 'urban_rural'].values
        else:
            selected_facilities['urban_rural'] = 'Unknown'

        final_pop = float(pop_flat[covered_flat].sum())
        final_coverage_pct = 100.0 * final_pop / total_pop_flat

        print(f"  Final: {len(chosen_idx)} HSAs, {final_coverage_pct:.1f}% coverage")

        return selected_facilities


    # ============================================================
    # INTEGRATION INSTRUCTIONS
    # ============================================================

    """
    To integrate this into hsa_optimization.py:

    1. Add this method to HSAOptimizer class (after line 1187)

    2. Update optimize() dispatcher (around line 1055-1080):

        REPLACE:
            if objective == 'fewest':
                result_dict = self._optimize_fewest(facilities, pop_arr, transform, target_pop, mode_weights)
            elif objective == 'footprint':
                result_dict = self._optimize_footprint(facilities, pop_arr, transform, target_pop, mode_weights)
            elif objective == 'distance':
                result_dict = self._optimize_distance(facilities, pop_arr, transform, target_pop, mode_weights)

        WITH:
            if objective in ['fewest', 'footprint', 'distance']:
                selected_facilities = self._optimize_unified(
                    facilities, pop_arr, transform, target_pop,
                    mode=objective, mode_weights=mode_weights, network_type=network_type
                )
                result_dict = {
                    'facilities': selected_facilities,
                    'coverage_pct': 100.0 * (selected_facilities coverage) / total_pop
                }

    3. Delete old functions:
        - _optimize_fewest() (lines 1188-1522)
        - _optimize_footprint() (lines 1523-1879)
        - _optimize_distance() (lines 1880-2300+)

    4. Test all 3 modes to verify they work correctly
    """

    def _get_coverage_mask(self, facility, shape, transform, radius_km):

        """Get boolean mask of covered pixels."""

        # Get facility coordinates

        fac_lon = facility.geometry.x

        fac_lat = facility.geometry.y

        

        # Create coordinate grids

        rows, cols = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

        

        xs = transform.c + (cols + 0.5) * transform.a

        ys = transform.f + (rows + 0.5) * transform.e

        

        # Reproject raster cell centers to WGS84 if needed

        xs_wgs, ys_wgs = xs, ys

        try:

            pop_crs = getattr(self.pop, "crs", None)

            crs_is_geo = (pop_crs is not None) and (

                str(pop_crs).upper() in ("EPSG:4326", "WGS84") or getattr(pop_crs, "is_geographic", False)

            )

        except Exception:

            crs_is_geo = False

        

        if not crs_is_geo:

            from pyproj import Transformer

            transformer = Transformer.from_crs(pop_crs, "EPSG:4326", always_xy=True)

            xs_wgs, ys_wgs = transformer.transform(xs, ys)

        

        # Now great-circle distance in km is valid

        dist = GeoUtils.haversine_km(fac_lon, fac_lat, xs_wgs, ys_wgs)

        

        return dist <= radius_km

    

    def _remove_overlapping_hsas(self, selected_facilities, pop_arr, transform,
                                  overlap_threshold=0.80):
        """
        Remove HSAs that overlap >80% with larger anchors (for FOOTPRINT/DISTANCE).

        This post-processing step reduces redundancy by keeping only facilities
        around larger patient volume anchors when HSAs heavily overlap.

        Parameters:
        -----------
        selected_facilities : GeoDataFrame
            Selected facilities with service_radius_km
        pop_arr : np.ndarray
            Population raster array
        transform : affine.Affine
            Raster geotransform
        overlap_threshold : float
            Overlap fraction threshold (0.0-1.0). Default 0.80 means remove HSAs
            that have >80% of their coverage overlapping with larger anchors.

        Returns:
        --------
        GeoDataFrame
            Filtered facilities with overlapping ones removed
        """
        import numpy as np

        print(f"  Removing HSAs with >{overlap_threshold*100:.0f}% overlap with larger anchors...")

        # Sort by patient volume (descending) - keep larger anchors
        fac = selected_facilities.copy()
        fac['patient_volume'] = fac['Total'].astype(float)
        fac_sorted = fac.sort_values('patient_volume', ascending=False).reset_index(drop=True)

        n_fac = len(fac_sorted)
        keep_mask = np.ones(n_fac, dtype=bool)

        # Build coverage masks for all facilities
        masks = []
        for idx, row in fac_sorted.iterrows():
            mask = self._get_coverage_mask(row, pop_arr.shape, transform,
                                          row['service_radius_km'])
            masks.append(mask)

        # Check each facility against larger ones (earlier in sorted order)
        removed_count = 0
        for i in range(1, n_fac):  # Start from 1, never remove the largest
            if not keep_mask[i]:
                continue

            # Count overlap with all LARGER facilities (indices 0 to i-1)
            my_mask = masks[i]
            my_coverage = my_mask.sum()

            if my_coverage == 0:
                keep_mask[i] = False
                removed_count += 1
                continue

            # Union of all larger facility masks
            larger_union = np.zeros_like(my_mask, dtype=bool)
            for j in range(i):
                if keep_mask[j]:
                    larger_union |= masks[j]

            # Calculate overlap fraction
            overlap = (my_mask & larger_union).sum()
            overlap_frac = overlap / my_coverage

            if overlap_frac > overlap_threshold:
                keep_mask[i] = False
                removed_count += 1

        result = fac_sorted[keep_mask].copy()
        print(f"  Kept {len(result)}/{n_fac} facilities (removed {removed_count})")

        return result

    def _compute_coverage(self, facilities, pop_arr, transform):

        """Compute total population covered by facilities."""

        covered = np.zeros_like(pop_arr, dtype=bool)

        

        for idx, row in facilities.iterrows():

            mask = self._get_coverage_mask(

                row, pop_arr.shape, transform,

                row.get('service_radius_km', row.get('initial_radius_km', MAX_RADIUS_KM))

            )

            covered |= mask

        

        return int(np.sum(pop_arr[covered]))



    def _cluster_counts(self, clusters, selected_idx):

        counts = np.zeros(CLIMATE_K, dtype=int)

        if selected_idx:

            for c in clusters[selected_idx]:

                if 0 <= c < CLIMATE_K:

                    counts[c] += 1

        return counts

    

    def _climate_reward(self, c, cluster_counts):

        # hard floor first

        if (not CLIMATE_DIVERSITY_ON) or (c < 0) or (c >= CLIMATE_K):

            return 0.0

        if CLIMATE_MIN_PER_CLUSTER > 0 and cluster_counts[c] < CLIMATE_MIN_PER_CLUSTER:

            return 10.0  # strong nudge to satisfy minima early

        # soft inverse-frequency reward afterward

        total = max(1, cluster_counts.sum())

        freq = cluster_counts[c] / total

        return 1.0 / max(freq, 1e-6)

# ============ Cell 17 ============
# Update the optimize method to support governorate objective

# This patches the optimize method to add governorate support



original_optimize = HSAOptimizer.optimize



def optimize_with_governorate(self, facilities, objective='fewest', network_type='NCD', governorates_gdf=None):

    """

    Enhanced optimize method with governorate support.

    """

    # Call original setup code

    print(f"\n{'='*60}")

    print(f"Running {objective.upper()} optimization for {network_type}")

    print(f"{'='*60}")



    # Get initial radii and urban/rural classification

    radii, urban_rural = choose_radius_km_for_facilities(

        facilities,

        self.config['pop_path'],

        network_type

    )

    facilities['initial_radius_km'] = radii

    facilities['urban_rural'] = urban_rural

    

    # Ensure facilities are in WGS84

    if facilities.crs is None:

        print("  ⚠️ Facilities GeoDataFrame has no CRS; assuming EPSG:4326 for bounds.")

    elif str(facilities.crs).upper() not in ("EPSG:4326", "WGS84"):

        facilities = facilities.to_crs("EPSG:4326")

    

    # Get population data - USE FULL JORDAN RASTER (don't crop to facilities!)

    # This ensures we don't lose western Jordan or any remote areas

    print(f"  Facilities CRS: {facilities.crs}")



    # Use full Jordan bounds instead of facility bounds

    # Jordan extent: ~34.88-39.20 E, ~29.18-33.38 N

    full_jordan_bounds = [34.5, 29.0, 39.5, 33.5]  # Padded full country bounds



    print(f"  Using FULL Jordan bounds (not facility-cropped): {full_jordan_bounds}")



    pop_arr, transform = self.pop.get_cropped(full_jordan_bounds)

    radius_grid = build_radius_candidates(facilities, transform,

                                  min_km=URBAN_BASE_RADIUS_KM-2,

                                  max_km=MAX_RADIUS_KM)

    

    print(f"  FULL Jordan raster sum: {int(pop_arr.sum()):,}")

    print(f"  Full raster shape: {pop_arr.shape}, transform: {transform}")

    

    total_pop = int(np.sum(pop_arr))

    target_pop = int(total_pop * self.config['tau_coverage'])

    

    print(f"\nPopulation statistics:")

    print(f"  Total population: {total_pop:,}")

    print(f"  Target coverage ({self.config['tau_coverage']*100:.0f}%): {target_pop:,}")

    

    if total_pop == 0:

        print("  Warning: population raster in bounds sums to 0. Skipping optimization.")

        out = facilities.copy()

        if 'initial_radius_km' not in out.columns:

            out['initial_radius_km'] = MAX_RADIUS_KM

        out['service_radius_km'] = out['initial_radius_km']

        return {

            'facilities': out,

            'total_pop': 0,

            'covered_pop': 0,

            'coverage_pct': 0.0,

            'objective': objective,

            'network_type': network_type

        }

    

    # Run appropriate algorithm

    if objective == 'governorate_tau_coverage' or objective == 'governorate':

        # 'governorate' maps to 'governorate_tau_coverage' for backwards compatibility

        if governorates_gdf is None:

            raise ValueError("governorates_gdf is required for governorate objectives")

        selected = self._optimize_governorate_tau_coverage(facilities, pop_arr, transform, target_pop, radius_grid, governorates_gdf)

    elif objective == 'governorate_fewest':

        if governorates_gdf is None:

            raise ValueError("governorates_gdf is required for governorate objectives")

        selected = self._optimize_governorate_fewest(facilities, pop_arr, transform, target_pop, governorates_gdf)

    elif objective == 'fewest':

                selected = self._optimize_unified(facilities, pop_arr, transform, target_pop, mode='fewest', mode_weights=None, network_type=network_type)

    elif objective == 'footprint':

                selected = self._optimize_unified(facilities, pop_arr, transform, target_pop, mode='footprint', mode_weights=None, network_type=network_type)

    elif objective == 'distance':

                selected = self._optimize_unified(facilities, pop_arr, transform, target_pop, mode='distance', mode_weights=None, network_type=network_type)

    else:

        raise ValueError(f"Unknown objective: {objective}")

    

    # === POST-PROCESSING: Remove overlapping HSAs for ALL modes ===
    print(f"\nPost-processing {objective.upper()}: Removing overlapping HSAs...")
    selected = self._remove_overlapping_hsas(
        selected, pop_arr, transform, overlap_threshold=0.80
    )

    # Compute coverage

    covered_pop = self._compute_coverage(selected, pop_arr, transform)

    coverage_pct = 0.0 if total_pop == 0 else (covered_pop / total_pop) * 100



    print(f"\nOptimization complete:")

    print(f"  Selected facilities: {len(selected)}")

    print(f"  Population covered: {covered_pop:,} ({coverage_pct:.1f}%)")



    return {

        'facilities': selected,

        'total_pop': total_pop,

        'covered_pop': covered_pop,

        'coverage_pct': coverage_pct,

        'objective': objective,

        'network_type': network_type

    }



# Attach as separate method AND replace optimize
HSAOptimizer.optimize_with_governorate = optimize_with_governorate
HSAOptimizer.optimize = optimize_with_governorate



print("HSAOptimizer.optimize method updated with governorate support")

# ============ Cell 18 ============
# Add this method to HSAOptimizer class (append to existing class definition)

# This creates a governorate-aware optimization



def _optimize_governorate_tau_coverage_method(self, fac, pop_arr, transform, target_pop, radius_grid, governorates_gdf):

    """

    Greedy optimization with TAU coverage target per governorate.

    

    Uses all existing solver features:

    - Population density for urban/rural classification

    - 90% coverage target

    - Climate diversity rewards

    - Overlap penalties

    - Variable radii based on density and network type

    """

    print("\nRunning GOVERNORATE algorithm (at least one HSA per governorate)...")

    

    # Ensure governorates are in same CRS as facilities

    if fac.crs and governorates_gdf.crs:

        if str(fac.crs).upper() not in ("EPSG:4326", "WGS84"):

            fac_wgs = fac.to_crs("EPSG:4326")

        else:

            fac_wgs = fac.copy()

            

        if str(governorates_gdf.crs).upper() not in ("EPSG:4326", "WGS84"):

            gov_wgs = governorates_gdf.to_crs("EPSG:4326")

        else:

            gov_wgs = governorates_gdf.copy()

    else:

        fac_wgs = fac.copy()

        gov_wgs = governorates_gdf.copy()

    

    # Spatial join to assign governorates to facilities

    fac_with_gov = gpd.sjoin(fac_wgs, gov_wgs[['geometry', 'shapeName']], 

                              how='left', predicate='within')

    

    # Get governorate column name

    gov_col = 'shapeName' if 'shapeName' in fac_with_gov.columns else 'shapeName_right'

    if gov_col not in fac_with_gov.columns:

        # Fallback: try other common names

        for col in ['Governorate', 'governorate', 'Gov', 'ADM1_EN']:

            if col in fac_with_gov.columns:

                gov_col = col

                break

    

    print(f"  Governorate column: {gov_col}")

    print(f"  Unique governorates: {fac_with_gov[gov_col].nunique()}")

    

    # Initialize

    pop_flat = pop_arr.ravel().astype('int64')

    total_pop_flat = float(pop_flat.sum())

    

    # Try each radius from largest to smallest

    for radius_km in sorted(radius_grid, reverse=True):

        print(f"  Trying radius {radius_km} km...")

        

        temp_fac = fac.copy()



        # Normalize patient volumes for scoring

        if 'Total' in temp_fac.columns:

            patient_volumes = temp_fac['Total'].fillna(0).values.astype('float64')

            max_patients = patient_volumes.max()

            if max_patients > 0:

                patient_norm = patient_volumes / max_patients

            else:

                patient_norm = np.zeros(len(temp_fac), dtype='float64')

        else:

            patient_norm = np.zeros(len(temp_fac), dtype='float64')

        temp_fac['service_radius_km'] = float(radius_km)

        temp_fac[gov_col] = fac_with_gov[gov_col].values

        

        # Build coverage masks

        masks = []

        for _, row in temp_fac.iterrows():

            m = self._get_coverage_mask(row, pop_arr.shape, transform, float(radius_km))

            masks.append(m.ravel())

        if len(masks) == 0:

            continue

        masks = np.stack(masks, axis=0)

        

        cov_totals = (masks * pop_flat).sum(axis=1, dtype='int64')

        clusters = temp_fac.get(CLIMATE_CLUSTER_COL, 

                                pd.Series(np.full(len(temp_fac), -1, dtype=int))).astype(int).to_numpy()

        

        # State tracking

        covered_flat = np.zeros_like(pop_flat, dtype=bool)

        selected_mask = np.zeros(len(temp_fac), dtype=bool)

        cluster_counts = np.zeros(CLIMATE_K, dtype=int)

        chosen_idx = []

        gov_covered = set()  # Track which governorates have at least one facility

        

        # PHASE 1: Ensure at least one facility per governorate

        print(f"    Phase 1: Ensuring one facility per governorate...")

        unique_govs = temp_fac[gov_col].dropna().unique()

        

        for gov_name in unique_govs:

            if pd.isna(gov_name):

                continue

                

            # Facilities in this governorate

            gov_mask = (temp_fac[gov_col] == gov_name).values

            gov_indices = np.where(gov_mask)[0]

            

            if len(gov_indices) == 0:

                continue

            

            # Calculate scores for facilities in this governorate

            new_cover = masks[gov_indices] & (~covered_flat)

            pop_gains = (new_cover * pop_flat).sum(axis=1, dtype='int64')

            

            # Climate reward for this subset

            gov_clusters = clusters[gov_indices]

            if CLIMATE_DIVERSITY_ON:

                cc = np.zeros(len(gov_indices), dtype=float)

                valid = (gov_clusters >= 0) & (gov_clusters < CLIMATE_K)

                cc[valid] = cluster_counts[gov_clusters[valid]]

                

                need_bins = (cluster_counts < CLIMATE_MIN_PER_CLUSTER) if CLIMATE_MIN_PER_CLUSTER > 0 else np.zeros(CLIMATE_K, dtype=bool)

                needs = np.zeros(len(gov_indices), dtype=bool)

                needs[valid] = need_bins[gov_clusters[valid]]

                

                total_sel = max(1, int(cluster_counts.sum()))

                freq = np.zeros(len(gov_indices))

                freq[valid] = cc[valid] / total_sel

                clim_reward = np.zeros(len(gov_indices), dtype=float)

                clim_reward[needs] = 10.0

                clim_reward[valid & ~needs] = 1.0 / np.maximum(freq[valid & ~needs], 1e-6)

            else:

                clim_reward = np.zeros(len(gov_indices), dtype=float)

            

            # Composite score

            score = (COVERAGE_WEIGHT * pop_gains.astype('float64')

                     + LAMBDA_CLIMATE * clim_reward.astype('float64'))

            

            # Select best facility in this governorate

            best_local = int(np.argmax(score))

            best_global = gov_indices[best_local]

            

            chosen_idx.append(best_global)

            covered_flat |= new_cover[best_local, :]

            selected_mask[best_global] = True

            gov_covered.add(gov_name)

            

            bc = int(clusters[best_global])

            if 0 <= bc < CLIMATE_K:

                cluster_counts[bc] += 1

        

        print(f"    Phase 1 complete: {len(chosen_idx)} facilities (one per governorate)")

        

        # PHASE 2: Continue greedy selection until coverage target reached

        print(f"    Phase 2: Continuing greedy selection to reach {self.config['tau_coverage']*100:.0f}% coverage...")

        

        while int(pop_flat[covered_flat].sum()) < target_pop:

            new_cover = masks & (~covered_flat)

            pop_gains = (new_cover * pop_flat).sum(axis=1, dtype='int64')

            

            already = (masks & covered_flat).sum(axis=1, dtype='int64')

            with np.errstate(divide='ignore', invalid='ignore'):

                overlap_frac = np.where(cov_totals > 0, already / cov_totals, 0.0)

            

            if CLIMATE_DIVERSITY_ON:

                cc = np.zeros_like(clusters, dtype=float)

                valid = (clusters >= 0) & (clusters < CLIMATE_K)

                cc[valid] = cluster_counts[clusters[valid]]

                

                need_bins = (cluster_counts < CLIMATE_MIN_PER_CLUSTER) if CLIMATE_MIN_PER_CLUSTER > 0 else np.zeros(CLIMATE_K, dtype=bool)

                needs = np.zeros_like(cc, dtype=bool)

                needs[valid] = need_bins[clusters[valid]]

                

                total_sel = max(1, int(cluster_counts.sum()))

                freq = np.zeros_like(cc)

                freq[valid] = cc[valid] / total_sel

                clim_reward = np.zeros_like(cc, dtype=float)

                clim_reward[needs] = 10.0

                mask_else = valid & (~needs)

                clim_reward[mask_else] = 1.0 / np.maximum(freq[mask_else], 1e-6)

            else:

                clim_reward = np.zeros(len(temp_fac), dtype=float)

                needs = np.zeros(len(temp_fac), dtype=bool)

                need_bins = np.zeros(CLIMATE_K, dtype=bool)

            

            score = (COVERAGE_WEIGHT * pop_gains.astype('float64')

                     - LAMBDA_OVERLAP * overlap_frac.astype('float64') * total_pop_flat

                     + LAMBDA_CLIMATE * clim_reward.astype('float64')

                     + LAMBDA_PATIENT_VOLUME * patient_norm * total_pop_flat)

            

            score[selected_mask] = -np.inf

            score[cov_totals == 0] = -np.inf

            

            if CLIMATE_DIVERSITY_ON and CLIMATE_MIN_PER_CLUSTER > 0 and need_bins.any():

                score[~needs] = -np.inf

            

            best_idx = int(np.argmax(score))

            best_score = float(score[best_idx])



            # Check if we have sufficient coverage (driven by scoring model)

            coverage_pct = 100.0 * int(pop_flat[covered_flat].sum()) / max(1, target_pop)

            tau_threshold = self.config['tau_coverage'] * 100.0  # Use configured TAU
            if coverage_pct >= tau_threshold:

                # Stop when tau coverage reached (number of HSAs emerges from scoring)

                print(f"    Governorate mode: {len(chosen_idx)} HSAs, coverage: {coverage_pct:.1f}%")

                break



            if not np.isfinite(best_score) or best_score <= 0.0:

                print("    No further gain possible at this radius.")

                break



            chosen_idx.append(best_idx)

            covered_flat |= new_cover[best_idx, :]

            selected_mask[best_idx] = True

            

            bc = int(clusters[best_idx])

            if 0 <= bc < CLIMATE_K:

                cluster_counts[bc] += 1

            

            if len(chosen_idx) % 5 == 0:

                cur = int(pop_flat[covered_flat].sum())

                pct = 100.0 * cur / max(1, target_pop)

                print(f"    Selected {len(chosen_idx)} total; {pct:.1f}% of target covered.")

        

        # Check if target achieved (Issue #6: Use count-based termination like fewest/footprint)

        cur_pop = int(pop_flat[covered_flat].sum())

        coverage_pct = 100.0 * cur_pop / max(1, target_pop)

        n_facilities = len(chosen_idx)



        # Governorate mode: Stop when tau coverage reached with all governorates represented

        # Number of HSAs emerges from scoring model

        if n_facilities > 0 and len(gov_covered) >= len(unique_govs):

            # All governorates covered - now check coverage

            tau_threshold = self.config['tau_coverage'] * 100.0  # Use configured TAU
            if coverage_pct >= tau_threshold:

                print(f"    [ACCEPTED] {n_facilities} HSAs, coverage: {coverage_pct:.1f}%")

                print(f"    Governorates covered: {len(gov_covered)}/{len(unique_govs)}")

                selected = temp_fac.iloc[chosen_idx].copy()



                # Assign variable radii based on patient volume

                selected = assign_adaptive_radii(

                    selected,

                    base_urban_km=15.0,

                    base_rural_km=25.0,

                    min_radius_km=10.0,

                    max_radius_km=MAX_RADIUS_KM  # Use global constant

                )

                return selected

        

        print(f"    Coverage insufficient at {radius_km} km ({100.0*cur_pop/max(1,target_pop):.1f}%)")

    

    # Fallback

    print("  Warning: Could not achieve target coverage; using all facilities.")

    out = fac.copy()

    out['service_radius_km'] = float(MAX_RADIUS_KM)

    return out



def _optimize_governorate_fewest_method(self, fac, pop_arr, transform, target_pop, governorates_gdf):

    """

    Governorate FEWEST mode: Ensure 1 HSA per governorate, then optimize for fewest total.



    Strategy:

    1. Pre-select highest volume facility in each governorate (12 governorates = 12 anchors)

    2. Run FEWEST optimization starting from these pre-selected anchors

    3. Continue until 90% national coverage reached



    Result: 12-20 HSAs total (more efficient than TAU=60% per governorate approach)

    """

    print("\nRunning GOVERNORATE_FEWEST algorithm (1 per governorate + FEWEST optimization)...")



    # Ensure governorates are in same CRS as facilities

    if fac.crs and governorates_gdf.crs:

        if str(fac.crs).upper() not in ("EPSG:4326", "WGS84"):

            fac_wgs = fac.to_crs("EPSG:4326")

        else:

            fac_wgs = fac.copy()



        if str(governorates_gdf.crs).upper() not in ("EPSG:4326", "WGS84"):

            gov_wgs = governorates_gdf.to_crs("EPSG:4326")

        else:

            gov_wgs = governorates_gdf.copy()

    else:

        fac_wgs = fac.copy()

        gov_wgs = governorates_gdf.copy()



    # Spatial join to assign governorates to facilities

    fac_with_gov = gpd.sjoin(fac_wgs, gov_wgs[['geometry', 'shapeName']],

                              how='left', predicate='within')



    # Get governorate column name

    gov_col = 'shapeName' if 'shapeName' in fac_with_gov.columns else 'shapeName_right'

    if gov_col not in fac_with_gov.columns:

        # Fallback: try other common names

        for col in ['Governorate', 'governorate', 'Gov', 'ADM1_EN']:

            if col in fac_with_gov.columns:

                gov_col = col

                break



    print(f"  Governorate column: {gov_col}")

    unique_govs = fac_with_gov[gov_col].dropna().unique()

    print(f"  Unique governorates: {len(unique_govs)}")



    # STEP 1: Pre-select one facility per governorate (highest volume)

    print(f"  Step 1: Pre-selecting highest volume facility per governorate...")

    pre_selected_indices = []



    for gov_name in unique_govs:

        if pd.isna(gov_name):

            continue



        # Facilities in this governorate

        gov_mask = (fac_with_gov[gov_col] == gov_name).values

        gov_facilities = fac_with_gov[gov_mask]



        if len(gov_facilities) == 0:

            continue



        # Select facility with highest patient volume

        if 'Total' in gov_facilities.columns:

            volumes = pd.to_numeric(gov_facilities['Total'], errors='coerce').fillna(0)

            best_idx_local = volumes.argmax()

            best_idx_global = gov_facilities.index[best_idx_local]

            best_name = gov_facilities.iloc[best_idx_local]['HealthFacility']

            best_volume = volumes.iloc[best_idx_local]

            pre_selected_indices.append(best_idx_global)

            print(f"    {gov_name}: {best_name} ({best_volume:.0f} patients)")

        else:

            # No volume data - just select first facility

            best_idx_global = gov_facilities.index[0]

            pre_selected_indices.append(best_idx_global)

            print(f"    {gov_name}: {gov_facilities.iloc[0]['HealthFacility']}")



    print(f"  Pre-selected {len(pre_selected_indices)} facilities (1 per governorate)\n")



    # STEP 2: Mark pre-selected facilities in the original dataframe

    fac_copy = fac.copy()

    fac_copy['pre_selected'] = False

    fac_copy.loc[pre_selected_indices, 'pre_selected'] = True



    # STEP 3: Call FEWEST optimization with pre-selected facilities

    # The FEWEST algorithm will need to be aware of pre-selected facilities

    # For now, we'll use a simpler approach: just call _optimize_fewest

    # and let it select naturally (which should include the high-volume facilities)



    print(f"  Step 2: Running FEWEST optimization to reach 90% national coverage...")

    selected = self._optimize_unified(fac_copy, pop_arr, transform, target_pop, mode='fewest', mode_weights=None, network_type='INF')



    # Add governorate column to result

    selected_with_gov = selected.merge(

        fac_with_gov[[gov_col]],

        left_index=True,

        right_index=True,

        how='left'

    )



    print(f"\n  GOVERNORATE_FEWEST complete: {len(selected_with_gov)} HSAs")

    print(f"  Governorates represented: {selected_with_gov[gov_col].nunique()}/{len(unique_govs)}")



    return selected_with_gov



# Monkey-patch the methods into the HSAOptimizer class

HSAOptimizer._optimize_governorate_tau_coverage = _optimize_governorate_tau_coverage_method

HSAOptimizer._optimize_governorate_fewest = _optimize_governorate_fewest_method



print("Governorate optimization methods added to HSAOptimizer class (tau_coverage and fewest)")


# ============ Cell 20 ============
def create_hsa_polygons(facilities_gdf):

    """

    Create HSA polygons (service area circles) from facilities.



    Parameters:

    -----------

    facilities_gdf : gpd.GeoDataFrame

        Facilities with 'service_radius_km' column



    Returns:

    --------

    gpd.GeoDataFrame

        GeoDataFrame with circular polygons representing HSA service areas

    """

    import geopandas as gpd

    from shapely.geometry import Point

    import numpy as np



    # Create a copy to avoid modifying original

    facilities = facilities_gdf.copy()



    # Check if we need to project to metric CRS for buffering

    original_crs = facilities.crs

    needs_projection = False



    if original_crs is None:

        print("Warning: No CRS defined, assuming EPSG:4326")

        facilities = facilities.set_crs('EPSG:4326', allow_override=True)

        original_crs = facilities.crs

        needs_projection = True

    elif original_crs.to_string() == 'EPSG:4326' or original_crs.is_geographic:

        needs_projection = True



    # If in geographic coordinates, project to metric for accurate buffering

    if needs_projection:

        # Project to appropriate UTM zone based on centroid

        # For Jordan, use UTM zone 36N or 37N (EPSG:32636 or EPSG:32637)

        # Calculate mean longitude to choose zone

        mean_lon = facilities.geometry.x.mean()



        if mean_lon < 36:

            utm_crs = 'EPSG:32636'  # UTM Zone 36N

        else:

            utm_crs = 'EPSG:32637'  # UTM Zone 37N



        facilities_projected = facilities.to_crs(utm_crs)

    else:

        facilities_projected = facilities



    # Create buffer polygons using service_radius_km

    hsa_polygons = []



    for idx, row in facilities_projected.iterrows():

        radius_km = row.get('service_radius_km', 20.0)  # Default 20km if not specified

        radius_m = radius_km * 1000  # Convert to meters



        # Create circular buffer

        hsa_polygon = row.geometry.buffer(radius_m)

        hsa_polygons.append(hsa_polygon)



    # Create GeoDataFrame from polygons

    hsas_gdf = gpd.GeoDataFrame(

        geometry=hsa_polygons,

        crs=facilities_projected.crs

    )



    # Project back to original CRS if needed

    if needs_projection:

        hsas_gdf = hsas_gdf.to_crs(original_crs)



    return hsas_gdf



print("create_hsa_polygons function defined")





def clip_hsas_to_country(hsas_gdf, country_boundary_gdf):

    """

    Clip HSA polygons to country boundary.



    Args:

        hsas_gdf: GeoDataFrame with HSA polygons

        country_boundary_gdf: GeoDataFrame with country boundary



    Returns:

        GeoDataFrame with clipped HSA polygons

    """

    from shapely.ops import unary_union



    # Ensure same CRS

    if hsas_gdf.crs != country_boundary_gdf.crs:

        country_boundary_gdf = country_boundary_gdf.to_crs(hsas_gdf.crs)



    # Create single country boundary polygon

    country_union = unary_union(country_boundary_gdf.geometry)



    # Clip each HSA to the country boundary

    clipped_geometries = []

    for idx, row in hsas_gdf.iterrows():

        try:

            clipped_geom = row.geometry.intersection(country_union)

            clipped_geometries.append(clipped_geom)

        except:

            clipped_geometries.append(row.geometry)



    # Create new GeoDataFrame with clipped geometries

    clipped_hsas = hsas_gdf.copy()

    clipped_hsas['geometry'] = clipped_geometries



    return clipped_hsas



print("clip_hsas_to_country function defined")



def assign_adaptive_radii(facilities_gdf, base_urban_km=15.0, base_rural_km=25.0,

                          min_radius_km=10.0, max_radius_km=30.0):

    """

    Assign variable service radii to facilities based on patient volume and urban/rural status.



    EXPLANATION:

    Larger facilities (more patients) serve broader regions and should have larger radii.

    Smaller facilities provide focused local coverage with smaller radii.

    This creates natural variation reflecting real-world healthcare hierarchies.



    Args:

        facilities_gdf: GeoDataFrame with facilities

        base_urban_km: Base radius for urban facilities (default 15 km)

        base_rural_km: Base radius for rural facilities (default 25 km)

        min_radius_km: Minimum allowed radius (default 10 km)

        max_radius_km: Maximum allowed radius (default 35 km)



    Returns:

        GeoDataFrame with 'service_radius_km' column added/updated

    """

    import pandas as pd

    import numpy as np



    facilities = facilities_gdf.copy()



    # Get patient volume for scaling

    if 'Total' not in facilities.columns:

        # No patient data - use fixed radii based on urban/rural

        if 'Urban_Rural' in facilities.columns:

            facilities['service_radius_km'] = facilities['Urban_Rural'].apply(

                lambda x: base_urban_km if x == 'Urban' else base_rural_km

            )

        else:

            facilities['service_radius_km'] = base_urban_km

        return facilities



    # Calculate patient volume percentile (0 to 1)

    patient_volume = facilities['Total'].fillna(0).astype('float64')



    # Handle edge case: all zeros

    if patient_volume.max() == 0:

        facilities['service_radius_km'] = base_urban_km

        return facilities



    # Compute percentile rank (0 = smallest, 1 = largest)

    patient_percentile = patient_volume.rank(pct=True, method='average')



    # Scaling factor: Small adjustment based on patient volume

    # Population density (urban/rural) is PRIMARY factor

    # Patient volume provides SECONDARY adjustment: ±2km max

    # This ensures facilities in similar areas have radii within ~5km

    volume_adjustment_km = -2.0 + 4.0 * patient_percentile  # -2km to +2km



    # Determine base radius by urban/rural (if available)

    if 'Urban_Rural' in facilities.columns:

        base_radius = facilities['Urban_Rural'].apply(

            lambda x: base_urban_km if x == 'Urban' else base_rural_km

        )

    else:

        # Default to urban base

        base_radius = pd.Series([base_urban_km] * len(facilities), index=facilities.index)



    # Calculate final radius (additive approach: base + adjustment)

    radius_km = base_radius + volume_adjustment_km



    # Clip to min/max bounds (clinically reasonable ranges)

    radius_km = radius_km.clip(min_radius_km, max_radius_km)



    facilities['service_radius_km'] = radius_km



    return facilities



print("assign_adaptive_radii function defined")


# ============ Cell 21 ============
def create_professional_map(facilities_gdf, governorates_gdf=None,

                           network_name='Network', objective='Optimization',

                           title=None, save_path=None, dpi=300):

    """

    Create publication-ready map with professional styling.

    Shows HSA labels and anchor facility names (no governorate labels).



    Parameters:

    -----------

    facilities_gdf : gpd.GeoDataFrame

        Selected facilities with service_radius_km column

    governorates_gdf : gpd.GeoDataFrame, optional

        Governorate boundaries for background

    network_name : str

        'INF' or 'NCD'

    objective : str

        Optimization objective name

    title : str, optional

        Custom title (auto-generated if None)

    save_path : Path or str, optional

        Path to save figure

    dpi : int

        Resolution for saved figure



    Returns:

    --------

    fig, ax : matplotlib Figure and Axes

    """

    import matplotlib.pyplot as plt

    import matplotlib.patches as mpatches

    from matplotlib.lines import Line2D



    # Create figure with high DPI

    fig, ax = plt.subplots(1, 1, figsize=(14, 12), dpi=150)



    # Color schemes

    network_colors = {

        'INF': '#e74c3c',  # Red

        'NCD': '#3498db',  # Blue

    }

    base_color = network_colors.get(network_name, '#2c3e50')



    # Plot governorate boundaries if available (NO LABELS)

    if governorates_gdf is not None:

        governorates_gdf.boundary.plot(

            ax=ax,

            linewidth=1.5,

            edgecolor='#34495e',

            alpha=0.6,

            zorder=1,

            label='Governorate Boundaries'

        )



    # Create HSA polygons (service area circles)

    hsas = create_hsa_polygons(facilities_gdf)



    # Classify by urban/rural for different styling

    if 'is_urban' in facilities_gdf.columns:

        urban_mask = facilities_gdf['is_urban'].values

    elif 'urban' in facilities_gdf.columns:

        urban_mask = facilities_gdf['urban'].values > 0

    else:

        # Fallback: classify by radius size

        median_radius = facilities_gdf['service_radius_km'].median()

        urban_mask = facilities_gdf['service_radius_km'] <= median_radius



    urban_hsas = hsas[urban_mask]

    rural_hsas = hsas[~urban_mask]



    # Plot HSA service areas

    if len(rural_hsas) > 0:

        rural_hsas.plot(

            ax=ax,

            facecolor=base_color,

            edgecolor=base_color,

            alpha=0.15,

            linewidth=1.5,

            linestyle='--',

            zorder=3

        )



    if len(urban_hsas) > 0:

        urban_hsas.plot(

            ax=ax,

            facecolor=base_color,

            edgecolor=base_color,

            alpha=0.25,

            linewidth=1.5,

            linestyle='-',

            zorder=4

        )



    # Plot facility points

    urban_fac = facilities_gdf[urban_mask]

    rural_fac = facilities_gdf[~urban_mask]



    if len(rural_fac) > 0:

        rural_fac.plot(

            ax=ax,

            color=base_color,

            marker='s',  # square

            markersize=80,

            edgecolor='white',

            linewidth=2,

            alpha=0.9,

            zorder=6

        )



    if len(urban_fac) > 0:

        urban_fac.plot(

            ax=ax,

            color=base_color,

            marker='o',  # circle

            markersize=100,

            edgecolor='white',

            linewidth=2,

            alpha=0.95,

            zorder=7

        )



    # Add HSA labels with facility names

    # Reset index to ensure we can iterate properly

    facilities_labeled = facilities_gdf.reset_index(drop=True)



    for idx, (hsa_idx, row) in enumerate(facilities_labeled.iterrows()):

        # Get facility name (try multiple columns)

        facility_name = None

        for col in ['HealthFacility', 'Name', 'name', 'facility_name', 'hospital_name', 'Hospital', 'FACILITY']:

            if col in row.index and pd.notna(row[col]):

                facility_name = str(row[col])

                break



        # Fallback to HSA number if no name

        if facility_name is None or facility_name == '':

            facility_name = f"HSA-{idx+1}"

        else:

            # Truncate long names

            if len(facility_name) > 25:

                facility_name = facility_name[:22] + "..."



        # Get centroid for label placement

        centroid = row.geometry

        if hasattr(centroid, 'centroid'):

            centroid = centroid.centroid



        # Determine label styling based on urban/rural

        is_urban_fac = urban_mask[hsa_idx] if isinstance(urban_mask, np.ndarray) else urban_mask.iloc[hsa_idx]



        if is_urban_fac:

            fontsize = 8

            bbox_color = base_color

            text_color = 'white'

        else:

            fontsize = 7

            bbox_color = 'white'

            text_color = base_color



        # Add label

        ax.text(

            centroid.x, centroid.y,

            facility_name,

            fontsize=fontsize,

            ha='center',

            va='center',

            color=text_color,

            weight='bold',

            bbox=dict(

                boxstyle='round,pad=0.4',

                facecolor=bbox_color,

                edgecolor=base_color,

                linewidth=1.5,

                alpha=0.85

            ),

            zorder=8

        )



    # Title

    if title is None:

        n_facilities = len(facilities_gdf)

        mean_radius = facilities_gdf['service_radius_km'].mean()

        title = f"{network_name} Network - {objective.title()} Objective\n{n_facilities} HSAs | Mean Radius: {mean_radius:.1f} km"



    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)



    # Legend

    legend_elements = [

        Line2D([0], [0], marker='o', color='w',

               markerfacecolor=base_color, markersize=12,

               markeredgecolor='white', markeredgewidth=2,

               label=f'Urban Facility ({len(urban_fac)})'),

        Line2D([0], [0], marker='s', color='w',

               markerfacecolor=base_color, markersize=10,

               markeredgecolor='white', markeredgewidth=2,

               label=f'Rural Facility ({len(rural_fac)})'),

        mpatches.Patch(facecolor=base_color, edgecolor=base_color,

                      alpha=0.25, label='Service Area (HSA)'),

    ]



    if governorates_gdf is not None:

        legend_elements.append(

            Line2D([0], [0], color='#34495e', linewidth=1.5,

                  alpha=0.6, label='Governorate Boundary')

        )



    ax.legend(handles=legend_elements, loc='upper right',

             fontsize=10, framealpha=0.95, shadow=True)



    # Styling

    ax.set_xlabel('Longitude', fontsize=11)

    ax.set_ylabel('Latitude', fontsize=11)

    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    ax.set_aspect('equal')



    # Tight layout

    plt.tight_layout()



    # Save if path provided

    if save_path:

        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',

                   facecolor='white', edgecolor='none')

        print(f"  Map saved to: {save_path}")



    return fig, ax





def create_comparison_map(results_dict, governorates_gdf=None,

                         network='INF', objectives=['fewest', 'footprint', 'distance', 'governorate'],

                         save_path=None, dpi=300):

    """

    Create side-by-side comparison of multiple objectives for one network.

    Shows HSA boundaries and facility markers (no governorate labels).



    Parameters:

    -----------

    results_dict : dict

        Dictionary with keys like 'INF_fewest', 'INF_footprint', etc.

    governorates_gdf : gpd.GeoDataFrame, optional

        Governorate boundaries

    network : str

        'INF' or 'NCD'

    objectives : list

        List of objectives to compare

    save_path : Path or str, optional

        Path to save figure

    dpi : int

        Resolution



    Returns:

    --------

    fig : matplotlib Figure

    """

    import matplotlib.pyplot as plt

    import matplotlib.patches as mpatches

    from matplotlib.lines import Line2D



    n_objectives = len(objectives)

    ncols = min(2, n_objectives)

    nrows = (n_objectives + ncols - 1) // ncols



    fig, axes = plt.subplots(nrows, ncols, figsize=(12*ncols, 10*nrows), dpi=120)

    if n_objectives == 1:

        axes = [axes]

    else:

        axes = axes.flatten() if nrows > 1 else axes



    network_colors = {

        'INF': '#e74c3c',

        'NCD': '#3498db',

    }

    base_color = network_colors.get(network, '#2c3e50')



    for idx, objective in enumerate(objectives):

        ax = axes[idx]

        key = f"{network}_{objective}"



        if key not in results_dict:

            ax.text(0.5, 0.5, f'No data for {objective}',

                   ha='center', va='center', fontsize=12)

            ax.axis('off')

            continue



        result = results_dict[key]

        facilities = result['facilities']



        # Plot governorates (NO LABELS)

        if governorates_gdf is not None:

            governorates_gdf.boundary.plot(

                ax=ax, linewidth=1.2, edgecolor='#34495e', alpha=0.5

            )



        # HSAs

        hsas = create_hsa_polygons(facilities)

        hsas.plot(ax=ax, facecolor=base_color, edgecolor=base_color,

                 alpha=0.2, linewidth=1.2)



        # Facilities

        facilities.plot(ax=ax, color=base_color, marker='o',

                       markersize=60, edgecolor='white', linewidth=1.5,

                       alpha=0.9, zorder=5)



        # Title with metrics

        n_fac = len(facilities)

        mean_r = facilities['service_radius_km'].mean()

        cov = result.get('coverage_pct', 0)



        ax.set_title(

            f"{objective.upper()}\n{n_fac} HSAs | Radius: {mean_r:.1f}km | Coverage: {cov:.1f}%",

            fontsize=12, fontweight='bold', pad=10

        )



        ax.set_xlabel('Longitude', fontsize=10)

        ax.set_ylabel('Latitude', fontsize=10)

        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        ax.set_aspect('equal')



    # Hide unused subplots

    for idx in range(n_objectives, len(axes)):

        axes[idx].axis('off')



    fig.suptitle(f"{network} Network - Objective Comparison",

                fontsize=16, fontweight='bold', y=0.995)



    plt.tight_layout()



    if save_path:

        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',

                   facecolor='white', edgecolor='none')

        print(f"  Comparison map saved to: {save_path}")



    return fig



print("[OK] Professional mapping functions loaded")


# ============ Cell 24 ============
# Load data


# Test code removed temporarily to allow module import without running tests
# The test code was located here (lines 4139-6246) but has been commented out
# to prevent it from running when the module is imported by other scripts.

