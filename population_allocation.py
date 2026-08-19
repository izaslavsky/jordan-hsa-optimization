"""
Patient Allocation Module - Gravity Model Implementation
=========================================================

Supports two allocation modes:
1. HARD ALLOCATION: Each pixel assigned to ONE facility (max attractiveness)
2. PROBABILISTIC ALLOCATION: Each pixel's population SPLIT across ALL reachable facilities
   based on gravity model probabilities

Usage (Hard Allocation - Original):
    allocator = PopulationAllocator(pop_raster_path, facilities_gdf, params)
    allocated = allocator.allocate_all_pixels()
    hsa_summary = allocator.aggregate_by_hsa(allocated, hsa_anchors)

Usage (Probabilistic Allocation - New):
    allocator = PopulationAllocator(pop_raster_path, facilities_gdf, params)
    allocated = allocator.allocate_all_pixels_probabilistic()
    hsa_summary = allocator.aggregate_facilities_to_hsas(allocated, hsa_anchors)

Author: HSA Research Team
Date: 2025-11-27 (Updated 2026-01)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from multiprocessing import cpu_count
import concurrent.futures
import math
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
from scipy.spatial import cKDTree

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class GeoUtils:
    """Geographic utility functions"""

    @staticmethod
    def haversine_km(lon1, lat1, lon2, lat2):
        """
        Calculate Haversine distance between two points in kilometers

        Args:
            lon1, lat1: Coordinates of point 1 (degrees)
            lon2, lat2: Coordinates of point 2 (degrees)

        Returns:
            Distance in kilometers
        """
        R = 6371.0  # Earth radius in km

        # Convert to radians
        lon1_rad, lat1_rad = np.radians(lon1), np.radians(lat1)
        lon2_rad, lat2_rad = np.radians(lon2), np.radians(lat2)

        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c


class PopulationAllocator:
    """
    Allocates patients to facilities using gravity model

    Gravity Model Formula:
        Attractiveness(facility) = Volume^α / Distance^β
        Probability(facility) = Attractiveness(facility) / Σ Attractiveness(all facilities)

    Hard Assignment:
        Each pixel assigned to facility with maximum attractiveness

    Performance Optimizations:
        - Spatial indexing (KD-Tree) for fast nearest facility lookup
        - Vectorized distance calculations
        - Pre-computed facility attractiveness constants
    """

    def __init__(self, pop_raster_path: str, facilities_gdf: gpd.GeoDataFrame,
                 params: Optional[Dict] = None):
        """
        Initialize allocator

        Args:
            pop_raster_path: Path to population raster (GeoTIFF)
            facilities_gdf: GeoDataFrame with facilities (must have 'Total' column for patient volume)
            params: Dict with keys:
                - alpha: Facility size weight (default 0.75)
                - beta: Distance decay (default 1.5)
                - max_distance_km: Maximum travel distance (default 100)
                - sample_rate: Sample every Nth pixel (default 5)
        """
        self.pop_raster_path = Path(pop_raster_path)
        self.facilities = facilities_gdf.copy()

        # Default parameters
        default_params = {
            'alpha': 0.75,
            'beta': 1.5,
            'max_distance_km': 100.0,
            'sample_rate': 5  # Process every 5th pixel
        }

        self.params = {**default_params, **(params or {})}

        # Validate facilities
        if 'Total' not in self.facilities.columns:
            raise ValueError("Facilities GeoDataFrame must have 'Total' column with patient volumes")

        # CRITICAL FIX: Ensure facilities are in EPSG:4326 (WGS84 lat/lon)
        if self.facilities.crs and self.facilities.crs.to_epsg() != 4326:
            print(f"  Reprojecting facilities from {self.facilities.crs} to EPSG:4326")
            self.facilities = self.facilities.to_crs(epsg=4326)

        # Extract facility info
        self.facilities['lon'] = self.facilities.geometry.x
        self.facilities['lat'] = self.facilities.geometry.y
        self.facilities['volume'] = pd.to_numeric(self.facilities['Total'], errors='coerce')

        # Remove facilities with zero/invalid volume
        valid_mask = self.facilities['volume'] > 0
        if not valid_mask.all():
            print(f"  Warning: Removing {(~valid_mask).sum()} facilities with zero/invalid volume")
            self.facilities = self.facilities[valid_mask].copy()

        # OPTIMIZATION: Build spatial index (KD-Tree) for fast nearest neighbor queries
        self._build_spatial_index()

        # OPTIMIZATION: Pre-compute volume^alpha for all facilities
        self.facilities['volume_alpha'] = self.facilities['volume'] ** self.params['alpha']

        # Pre-built name → DataFrame-index lookup for O(1) facility resolution
        self._fac_name_to_df_idx = dict(zip(
            self.facilities['HealthFacility'].values,
            self.facilities.index,
        ))

        print(f"PopulationAllocator initialized:")
        print(f"  Facilities: {len(self.facilities)}")
        print(f"  Parameters: alpha={self.params['alpha']}, beta={self.params['beta']}, "
              f"max_dist={self.params['max_distance_km']}km")
        print(f"  Spatial index: KD-Tree built for fast lookups")
        print(f"  Facility lon range: [{self.facilities['lon'].min():.2f}, {self.facilities['lon'].max():.2f}]")
        print(f"  Facility lat range: [{self.facilities['lat'].min():.2f}, {self.facilities['lat'].max():.2f}]")

    def _build_spatial_index(self):
        """Build KD-Tree spatial index for fast nearest facility queries"""
        # Convert to radians for haversine calculations
        self.facility_coords_rad = np.radians(
            self.facilities[['lon', 'lat']].values
        )

        # Build KD-Tree on lon/lat coordinates (for approximate nearest neighbors)
        # Note: This uses Euclidean distance as approximation, but we refine with haversine
        self.kdtree = cKDTree(self.facilities[['lon', 'lat']].values)

    def compute_attractiveness(self, pixel_lon: float, pixel_lat: float,
                              facility_row: pd.Series) -> float:
        """
        Compute attractiveness of a facility for a pixel

        Args:
            pixel_lon, pixel_lat: Pixel coordinates
            facility_row: Row from facilities DataFrame

        Returns:
            Attractiveness score (0 if beyond max distance)
        """
        # Calculate distance
        distance = GeoUtils.haversine_km(
            pixel_lon, pixel_lat,
            facility_row['lon'], facility_row['lat']
        )

        # Check max distance constraint
        if distance > self.params['max_distance_km']:
            return 0.0

        # Prevent division by zero (if pixel is exactly at facility)
        if distance < 0.01:  # Within 10m
            distance = 0.01

        # Gravity model: volume^α / distance^β
        volume = facility_row['volume']
        attractiveness = (volume ** self.params['alpha']) / (distance ** self.params['beta'])

        return attractiveness

    def allocate_pixel(self, pixel_lon: float, pixel_lat: float,
                      pixel_pop: float) -> Optional[Dict]:
        """
        Allocate one pixel to a facility (OPTIMIZED VERSION)

        Args:
            pixel_lon, pixel_lat: Pixel coordinates
            pixel_pop: Population in pixel

        Returns:
            Dict with allocation info, or None if no facilities in range
        """
        # OPTIMIZATION: Use KD-Tree to find candidate facilities within max distance
        # Query radius in degrees (approximate: 1 degree ≈ 111km at equator, but vary with latitude)
        # At Jordan's latitude (~31°N), 1 degree lon ≈ 95km, 1 degree lat ≈ 111km
        # Use conservative estimate to not miss facilities
        max_dist_degrees = self.params['max_distance_km'] / 90.0  # More conservative than 111

        # Find all facilities within approximate radius
        candidate_indices = self.kdtree.query_ball_point([pixel_lon, pixel_lat], max_dist_degrees)

        if not candidate_indices:
            return None  # No facilities within range

        # OPTIMIZATION: Vectorized distance calculation for all candidates
        candidate_rows = self.facilities.iloc[candidate_indices]

        # Calculate haversine distances (vectorized)
        distances = GeoUtils.haversine_km(
            pixel_lon, pixel_lat,
            candidate_rows['lon'].values,
            candidate_rows['lat'].values
        )

        # Filter by exact max distance
        valid_mask = distances <= self.params['max_distance_km']

        if not valid_mask.any():
            return None

        # Apply valid mask
        valid_distances = distances[valid_mask]
        valid_rows = candidate_rows[valid_mask]
        valid_indices = np.array(candidate_indices)[valid_mask]

        # Prevent division by zero
        valid_distances = np.maximum(valid_distances, 0.01)

        # OPTIMIZATION: Vectorized attractiveness calculation
        # attractiveness = volume^α / distance^β
        volume_alpha = valid_rows['volume_alpha'].values
        attractiveness = volume_alpha / (valid_distances ** self.params['beta'])

        # Hard assignment: choose facility with max attractiveness
        max_idx = np.argmax(attractiveness)
        assigned_idx = valid_indices[max_idx]
        total_attr = attractiveness.sum()

        return {
            'lon': pixel_lon,
            'lat': pixel_lat,
            'population': pixel_pop,
            'facility_idx': self.facilities.index[assigned_idx],
            'facility_id': valid_rows.iloc[max_idx]['HealthFacility'],
            'probability': attractiveness[max_idx] / total_attr,
            'num_candidates': len(valid_distances)
        }

    def allocate_pixel_probabilistic(self, pixel_lon: float, pixel_lat: float,
                                      pixel_pop: float) -> Optional[Dict]:
        """
        Allocate one pixel to ALL facilities within range using probabilistic allocation.

        Instead of assigning population to ONE facility (hard assignment), this method
        SPLITS the pixel's population across ALL reachable facilities based on their
        gravity model probabilities.

        Args:
            pixel_lon, pixel_lat: Pixel coordinates
            pixel_pop: Population in pixel

        Returns:
            Dict with keys:
                - 'lon', 'lat': pixel coordinates
                - 'total_population': total pixel population
                - 'allocations': List of {facility_id, allocated_pop, probability, distance_km}
            Returns None if no facilities in range.
        """
        # Use KD-Tree to find candidate facilities within max distance
        max_dist_degrees = self.params['max_distance_km'] / 90.0
        candidate_indices = self.kdtree.query_ball_point([pixel_lon, pixel_lat], max_dist_degrees)

        if not candidate_indices:
            return None

        # Get candidate rows
        candidate_rows = self.facilities.iloc[candidate_indices]

        # Calculate haversine distances (vectorized)
        distances = GeoUtils.haversine_km(
            pixel_lon, pixel_lat,
            candidate_rows['lon'].values,
            candidate_rows['lat'].values
        )

        # Filter by exact max distance
        valid_mask = distances <= self.params['max_distance_km']

        if not valid_mask.any():
            return None

        # Apply valid mask
        valid_distances = distances[valid_mask]
        valid_rows = candidate_rows[valid_mask]

        # Prevent division by zero
        valid_distances = np.maximum(valid_distances, 0.01)

        # Calculate attractiveness for each facility
        volume_alpha = valid_rows['volume_alpha'].values
        attractiveness = volume_alpha / (valid_distances ** self.params['beta'])

        # Calculate probabilities (sum to 1)
        total_attr = attractiveness.sum()
        probabilities = attractiveness / total_attr

        # Calculate allocated population for each facility
        allocated_pops = probabilities * pixel_pop

        # Build allocations list
        allocations = []
        for i in range(len(valid_rows)):
            allocations.append({
                'facility_id': valid_rows.iloc[i]['HealthFacility'],
                'allocated_pop': allocated_pops[i],
                'probability': probabilities[i],
                'distance_km': valid_distances[i]
            })

        return {
            'lon': pixel_lon,
            'lat': pixel_lat,
            'total_population': pixel_pop,
            'allocations': allocations,
            'num_candidates': len(allocations)
        }

    def allocate_all_pixels_probabilistic(self, progress_interval: int = 50000,
                                           return_pixel_allocations: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Allocate all populated pixels using PROBABILISTIC allocation.

        Unlike allocate_all_pixels() which assigns each pixel to ONE facility,
        this method SPLITS each pixel's population across ALL reachable facilities
        based on gravity model probabilities.

        Args:
            progress_interval: Print progress every N pixels
            return_pixel_allocations: If True, also return pixel-level allocations
                                      showing PRIMARY facility for each pixel

        Returns:
            Tuple of (facility_df, pixel_df):
                - facility_df: DataFrame with columns [facility_id, allocated_population]
                - pixel_df: DataFrame with columns [lon, lat, population, facility_id, probability, num_candidates]
                            Shows PRIMARY (max probability) facility for each pixel
        """
        print(f"\nAllocating population pixels to facilities (PROBABILISTIC)...")

        # Track allocations per facility
        facility_allocations = {}  # facility_id -> total allocated population
        pixel_allocations = []  # List for pixel-level data
        total_pop = 0
        allocated_pop = 0
        unallocated_pop = 0
        pixel_count = 0

        # Open population raster
        with rasterio.open(self.pop_raster_path) as src:
            pop_data = src.read(1)
            transform = src.transform

            height, width = pop_data.shape
            sample_rate = self.params['sample_rate']

            print(f"  Raster size: {height} x {width} = {height*width:,} pixels")
            if sample_rate > 1:
                print(f"  Sampling rate: 1/{sample_rate} (processing every {sample_rate}th pixel)")

            first_pixel_logged = False

            # Iterate through pixels
            for row in range(0, height, sample_rate):
                for col in range(0, width, sample_rate):
                    pop = pop_data[row, col]

                    if pop <= 0:
                        continue

                    # Convert pixel coordinates to geographic coordinates
                    lon, lat = transform * (col + 0.5, row + 0.5)

                    if not first_pixel_logged:
                        print(f"  First populated pixel: lon={lon:.6f}, lat={lat:.6f}, pop={pop:.1f}")
                        first_pixel_logged = True

                    total_pop += pop
                    pixel_count += 1

                    # Probabilistic allocation
                    allocation = self.allocate_pixel_probabilistic(lon, lat, pop)

                    if allocation:
                        allocated_pop += pop
                        # Add to facility totals
                        for alloc in allocation['allocations']:
                            fac_id = alloc['facility_id']
                            if fac_id not in facility_allocations:
                                facility_allocations[fac_id] = 0
                            facility_allocations[fac_id] += alloc['allocated_pop']

                        # Track pixel-level allocation (primary facility)
                        if return_pixel_allocations:
                            # Find primary (max probability) facility
                            primary = max(allocation['allocations'], key=lambda x: x['probability'])
                            # Get facility_idx for compatibility with old format
                            fac_match = self.facilities[self.facilities['HealthFacility'] == primary['facility_id']]
                            fac_idx = fac_match.index[0] if not fac_match.empty else -1
                            pixel_allocations.append({
                                'lon': lon,
                                'lat': lat,
                                'population': pop,
                                'facility_idx': fac_idx,
                                'facility_id': primary['facility_id'],
                                'probability': primary['probability'],
                                'num_candidates': allocation['num_candidates']
                            })
                    else:
                        unallocated_pop += pop

                    # Progress report
                    if pixel_count % progress_interval == 0:
                        pct_done = (row * width + col) / (height * width) * 100
                        pct_allocated = (allocated_pop / total_pop * 100) if total_pop > 0 else 0
                        print(f"    Progress: {pct_done:5.1f}% scanned | "
                              f"{pixel_count:,} pixels | "
                              f"{pct_allocated:.1f}% population allocated")

        # Create DataFrame from facility allocations
        facility_df = pd.DataFrame([
            {'facility_id': fac_id, 'allocated_population': alloc_pop}
            for fac_id, alloc_pop in facility_allocations.items()
        ])

        # Sort by allocated population
        if len(facility_df) > 0:
            facility_df = facility_df.sort_values('allocated_population', ascending=False).reset_index(drop=True)

        # Create pixel allocations DataFrame
        pixel_df = pd.DataFrame(pixel_allocations) if return_pixel_allocations else None

        # Summary
        print(f"\n  Probabilistic allocation complete:")
        print(f"    Total population: {total_pop:,.0f}")
        print(f"    Allocated population: {allocated_pop:,.0f} ({allocated_pop/total_pop*100:.1f}%)")
        print(f"    Unallocated population: {unallocated_pop:,.0f} ({unallocated_pop/total_pop*100:.1f}%)")
        print(f"    Pixels processed: {pixel_count:,}")
        print(f"    Facilities receiving allocation: {len(facility_df)}")
        if len(facility_df) > 0:
            print(f"    Sum of facility allocations: {facility_df['allocated_population'].sum():,.0f}")
        if pixel_df is not None:
            print(f"    Pixel allocations tracked: {len(pixel_df):,}")

        return facility_df, pixel_df

    def allocate_all_pixels(self, progress_interval: int = 50000) -> pd.DataFrame:
        """
        Allocate all populated pixels to facilities (OPTIMIZED VERSION)

        Args:
            progress_interval: Print progress every N pixels

        Returns:
            DataFrame with columns: lon, lat, population, facility_idx, facility_id, probability
        """
        print(f"\nAllocating population pixels to facilities...")

        allocations = []
        total_pop = 0
        allocated_pop = 0
        unallocated_pop = 0
        pixel_count = 0

        # Open population raster
        with rasterio.open(self.pop_raster_path) as src:
            pop_data = src.read(1)
            transform = src.transform

            # Get dimensions
            height, width = pop_data.shape
            sample_rate = self.params['sample_rate']

            print(f"  Raster size: {height} x {width} = {height*width:,} pixels")
            if sample_rate > 1:
                print(f"  Sampling rate: 1/{sample_rate} (processing every {sample_rate}th pixel)")

            # OPTIMIZATION: Pre-allocate lists for better performance
            total_pixels_estimate = height * width // (sample_rate ** 2) // 10  # Rough estimate
            allocations = []

            # Debug: Print first pixel coordinates
            first_pixel_logged = False

            # Iterate through pixels
            for row in range(0, height, sample_rate):
                for col in range(0, width, sample_rate):
                    pop = pop_data[row, col]

                    # Skip non-populated pixels
                    if pop <= 0:
                        continue

                    # Convert pixel coordinates to geographic coordinates
                    lon, lat = transform * (col + 0.5, row + 0.5)

                    # Debug: Log first populated pixel
                    if not first_pixel_logged:
                        print(f"  First populated pixel: lon={lon:.6f}, lat={lat:.6f}, pop={pop:.1f}")
                        first_pixel_logged = True

                    total_pop += pop
                    pixel_count += 1

                    # Allocate pixel
                    allocation = self.allocate_pixel(lon, lat, pop)

                    if allocation:
                        allocations.append(allocation)
                        allocated_pop += pop
                    else:
                        unallocated_pop += pop

                    # Progress report (less frequent for better performance)
                    if pixel_count % progress_interval == 0:
                        pct_done = (row * width + col) / (height * width) * 100
                        pct_allocated = (allocated_pop / total_pop * 100) if total_pop > 0 else 0
                        pixels_per_sec = pixel_count / ((row * width + col) / (height * width) * 100 + 0.001)
                        print(f"    Progress: {pct_done:5.1f}% scanned | "
                              f"{pixel_count:,} pixels | "
                              f"{pct_allocated:.1f}% population allocated | "
                              f"Speed: ~{pixels_per_sec:.0f} px/% scanned")

        # Create DataFrame
        df = pd.DataFrame(allocations)

        # Summary
        print(f"\n  Allocation complete:")
        print(f"    Total population: {total_pop:,.0f}")
        print(f"    Allocated population: {allocated_pop:,.0f} ({allocated_pop/total_pop*100:.1f}%)")
        print(f"    Unallocated population: {unallocated_pop:,.0f} ({unallocated_pop/total_pop*100:.1f}%)")
        print(f"    Pixels processed: {pixel_count:,}")
        print(f"    Pixels allocated: {len(df):,}")

        return df

    def _process_chunk(self, chunk: Tuple[int, int]) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Process a chunk of rows for parallel allocation.

        Args:
            chunk: (start_row, end_row)

        Returns:
            (allocations_df, stats_dict)
        """
        start_row, end_row = chunk
        allocations = []
        total_pop = 0
        allocated_pop = 0
        unallocated_pop = 0
        pixel_count = 0

        with rasterio.open(self.pop_raster_path) as src:
            width = src.width
            sample_rate = self.params['sample_rate']
            window = Window(0, start_row, width, end_row - start_row)
            pop_data = src.read(1, window=window)
            transform = src.transform

            for row in range(start_row, end_row, sample_rate):
                row_off = row - start_row
                if row_off >= pop_data.shape[0]:
                    break
                for col in range(0, width, sample_rate):
                    pop = pop_data[row_off, col]
                    if pop <= 0:
                        continue

                    lon, lat = transform * (col + 0.5, row + 0.5)

                    total_pop += pop
                    pixel_count += 1

                    allocation = self.allocate_pixel(lon, lat, pop)
                    if allocation:
                        allocations.append(allocation)
                        allocated_pop += pop
                    else:
                        unallocated_pop += pop

        df = pd.DataFrame(allocations)
        stats = {
            'total_pop': total_pop,
            'allocated_pop': allocated_pop,
            'unallocated_pop': unallocated_pop,
            'pixel_count': pixel_count,
            'allocated_pixels': len(df)
        }
        return df, stats

    def _process_chunk_probabilistic(self, chunk: Tuple[int, int]) -> Tuple[Dict[str, float], list, Dict[str, float]]:
        """
        Process a chunk of rows for probabilistic parallel allocation.

        Args:
            chunk: (start_row, end_row)

        Returns:
            (facility_allocations_dict, pixel_allocations_list, stats_dict)
        """
        start_row, end_row = chunk
        facility_allocations = {}  # facility_id -> total allocated population
        pixel_allocations = []  # List of pixel-level allocations
        total_pop = 0
        allocated_pop = 0
        unallocated_pop = 0
        pixel_count = 0

        with rasterio.open(self.pop_raster_path) as src:
            width = src.width
            sample_rate = self.params['sample_rate']
            window = Window(0, start_row, width, end_row - start_row)
            pop_data = src.read(1, window=window)
            transform = src.transform

            for row in range(start_row, end_row, sample_rate):
                row_off = row - start_row
                if row_off >= pop_data.shape[0]:
                    break
                for col in range(0, width, sample_rate):
                    pop = pop_data[row_off, col]
                    if pop <= 0:
                        continue

                    lon, lat = transform * (col + 0.5, row + 0.5)

                    total_pop += pop
                    pixel_count += 1

                    allocation = self.allocate_pixel_probabilistic(lon, lat, pop)
                    if allocation:
                        allocated_pop += pop
                        for alloc in allocation['allocations']:
                            fac_id = alloc['facility_id']
                            if fac_id not in facility_allocations:
                                facility_allocations[fac_id] = 0
                            facility_allocations[fac_id] += alloc['allocated_pop']

                        # Track pixel-level allocation (primary facility)
                        primary = max(allocation['allocations'], key=lambda x: x['probability'])
                        # O(1) facility index lookup
                        fac_idx = self._fac_name_to_df_idx.get(primary['facility_id'], -1)
                        pixel_allocations.append({
                            'lon': lon,
                            'lat': lat,
                            'population': pop,
                            'facility_idx': fac_idx,
                            'facility_id': primary['facility_id'],
                            'probability': primary['probability'],
                            'num_candidates': allocation['num_candidates']
                        })
                    else:
                        unallocated_pop += pop

        stats = {
            'total_pop': total_pop,
            'allocated_pop': allocated_pop,
            'unallocated_pop': unallocated_pop,
            'pixel_count': pixel_count,
            'num_facilities': len(facility_allocations)
        }
        return facility_allocations, pixel_allocations, stats

    def allocate_all_pixels_probabilistic_parallel(self, progress_interval: int = 50000,
                                                    num_workers: Optional[int] = None,
                                                    return_pixel_allocations: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Parallel probabilistic allocation using multiprocessing.

        Args:
            progress_interval: Unused in parallel mode (kept for API parity)
            num_workers: Number of worker processes (defaults to CPU count - 1)
            return_pixel_allocations: If True, also return pixel-level allocations

        Returns:
            Tuple of (facility_df, pixel_df):
                - facility_df: DataFrame with columns [facility_id, allocated_population]
                - pixel_df: DataFrame with columns [lon, lat, population, facility_id, probability, num_candidates]
        """
        if num_workers is None:
            num_workers = max(cpu_count() - 1, 1)

        print(f"\nAllocating population pixels to facilities (PROBABILISTIC PARALLEL)...")
        print(f"Using {num_workers} parallel workers")

        with rasterio.open(self.pop_raster_path) as src:
            height, width = src.shape
            sample_rate = self.params['sample_rate']

        print(f"  Raster size: {height} x {width} = {height*width:,} pixels")
        if sample_rate > 1:
            print(f"  Sampling rate: 1/{sample_rate} (processing every {sample_rate}th pixel)")

        chunk_size = math.ceil(height / num_workers)
        chunks = [
            (i * chunk_size, min((i + 1) * chunk_size, height))
            for i in range(num_workers)
            if i * chunk_size < height
        ]

        # Submit chunks as separate futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_chunk_probabilistic, chunk) for chunk in chunks]

            all_facility_allocations = {}
            all_pixel_allocations = []
            stats_list = []
            total_futures = len(futures)
            completed = 0

            for fut in concurrent.futures.as_completed(futures):
                try:
                    chunk_facility_allocs, chunk_pixel_allocs, stats = fut.result()
                except Exception as e:
                    raise RuntimeError(f"Error in worker during probabilistic allocation: {e}")

                # Merge chunk facility allocations into global
                for fac_id, alloc_pop in chunk_facility_allocs.items():
                    if fac_id not in all_facility_allocations:
                        all_facility_allocations[fac_id] = 0
                    all_facility_allocations[fac_id] += alloc_pop

                # Collect pixel allocations
                if return_pixel_allocations:
                    all_pixel_allocations.extend(chunk_pixel_allocs)

                stats_list.append(stats)

                completed += 1
                pct = (completed / total_futures) * 100
                print(f"    Parallel allocation progress: {completed}/{total_futures} chunks ({pct:.0f}%)")

        # Create facility DataFrame
        facility_df = pd.DataFrame([
            {'facility_id': fac_id, 'allocated_population': alloc_pop}
            for fac_id, alloc_pop in all_facility_allocations.items()
        ])

        if len(facility_df) > 0:
            facility_df = facility_df.sort_values('allocated_population', ascending=False).reset_index(drop=True)

        # Create pixel DataFrame
        pixel_df = pd.DataFrame(all_pixel_allocations) if return_pixel_allocations and all_pixel_allocations else None

        total_pop = sum(s['total_pop'] for s in stats_list)
        allocated_pop = sum(s['allocated_pop'] for s in stats_list)
        unallocated_pop = sum(s['unallocated_pop'] for s in stats_list)
        pixel_count = sum(s['pixel_count'] for s in stats_list)

        print(f"\n  Probabilistic allocation complete:")
        print(f"    Total population: {total_pop:,.0f}")
        if total_pop > 0:
            print(f"    Allocated population: {allocated_pop:,.0f} ({allocated_pop/total_pop*100:.1f}%)")
            print(f"    Unallocated population: {unallocated_pop:,.0f} ({unallocated_pop/total_pop*100:.1f}%)")
        print(f"    Pixels processed: {pixel_count:,}")
        print(f"    Facilities receiving allocation: {len(facility_df)}")
        if len(facility_df) > 0:
            print(f"    Sum of facility allocations: {facility_df['allocated_population'].sum():,.0f}")
        if pixel_df is not None:
            print(f"    Pixel allocations tracked: {len(pixel_df):,}")

        return facility_df, pixel_df

    def allocate_all_pixels_parallel(self, progress_interval: int = 50000,
                                     num_workers: Optional[int] = None) -> pd.DataFrame:
        """
        Parallel allocation using multiprocessing.

        Args:
            progress_interval: Unused in parallel mode (kept for API parity)
            num_workers: Number of worker processes (defaults to CPU count - 1)

        Returns:
            DataFrame with allocations
        """
        if num_workers is None:
            num_workers = max(cpu_count() - 1, 1)

        print(f"\nAllocating population pixels to facilities (parallel)...")
        print(f"Using {num_workers} parallel workers")

        with rasterio.open(self.pop_raster_path) as src:
            height, width = src.shape
            sample_rate = self.params['sample_rate']

        print(f"  Raster size: {height} x {width} = {height*width:,} pixels")
        if sample_rate > 1:
            print(f"  Sampling rate: 1/{sample_rate} (processing every {sample_rate}th pixel)")

        chunk_size = math.ceil(height / num_workers)
        chunks = [
            (i * chunk_size, min((i + 1) * chunk_size, height))
            for i in range(num_workers)
            if i * chunk_size < height
        ]

        # Submit chunks as separate futures so we can report progress as they complete
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_chunk, chunk) for chunk in chunks]

            results = []
            stats_list = []
            alloc_dfs = []
            total_futures = len(futures)
            completed = 0

            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    # If a worker fails, surface an informative error
                    raise RuntimeError(f"Error in worker during allocation: {e}")

                df_chunk, stats = res
                alloc_dfs.append(df_chunk)
                stats_list.append(stats)
                results.append(res)

                completed += 1
                pct = (completed / total_futures) * 100
                print(f"    Parallel allocation progress: {completed}/{total_futures} chunks ({pct:.0f}%)")
        df = pd.concat(alloc_dfs, ignore_index=True) if alloc_dfs else pd.DataFrame()

        total_pop = sum(s['total_pop'] for s in stats_list)
        allocated_pop = sum(s['allocated_pop'] for s in stats_list)
        unallocated_pop = sum(s['unallocated_pop'] for s in stats_list)
        pixel_count = sum(s['pixel_count'] for s in stats_list)

        print(f"\n  Allocation complete:")
        print(f"    Total population: {total_pop:,.0f}")
        if total_pop > 0:
            print(f"    Allocated population: {allocated_pop:,.0f} ({allocated_pop/total_pop*100:.1f}%)")
            print(f"    Unallocated population: {unallocated_pop:,.0f} ({unallocated_pop/total_pop*100:.1f}%)")
        print(f"    Pixels processed: {pixel_count:,}")
        print(f"    Pixels allocated: {len(df):,}")

        return df

    # ------------------------------------------------------------------
    # Vectorized helpers
    # ------------------------------------------------------------------

    def _haversine_batch(self, pixel_lons: np.ndarray, pixel_lats: np.ndarray) -> np.ndarray:
        """Return (N, F) km distance matrix between N pixels and F facilities."""
        R = 6371.0
        fac_lons = self.facilities['lon'].values   # (F,)
        fac_lats = self.facilities['lat'].values   # (F,)
        # Broadcasting: (N,1) op (1,F) → (N,F)
        dlat = np.radians(fac_lats[None, :] - pixel_lats[:, None])
        dlon = np.radians(fac_lons[None, :] - pixel_lons[:, None])
        lat1 = np.radians(pixel_lats[:, None])
        lat2 = np.radians(fac_lats[None, :])
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    def _compute_gravity_batch(self, dist_km: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gravity-model probabilities for an (N, F) distance matrix.

        Returns:
            probs   (N, F)  — row-normalised probabilities; rows of uncovered pixels are 0
            covered (N,)    — True where at least one facility is within max_distance_km
        """
        beta      = self.params['beta']
        max_dist  = self.params['max_distance_km']
        vol_alpha = self.facilities['volume_alpha'].values   # (F,)

        in_range  = dist_km <= max_dist                      # (N, F) bool
        safe_dist = np.maximum(dist_km, 0.01)
        attract   = vol_alpha[None, :] / (safe_dist ** beta) # (N, F)
        attract   = np.where(in_range, attract, 0.0)

        row_sum   = attract.sum(axis=1, keepdims=True)       # (N, 1)
        covered   = row_sum.ravel() > 0                      # (N,)
        probs     = np.where(
            covered[:, None],
            attract / np.maximum(row_sum, 1e-15),
            0.0,
        )
        return probs, covered

    def allocate_all_pixels_probabilistic_vectorized(
            self,
            batch_size: int = 50000,
            return_pixel_allocations: bool = True,
            progress_interval: int = 200000) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Vectorized probabilistic allocation using batched NumPy matrix operations.

        Identical results to allocate_all_pixels_probabilistic but ~50-200x faster:
        replaces the per-pixel Python loop with batched N×F distance and gravity
        computations, all in NumPy.

        Memory per batch: batch_size × n_facilities × 8 bytes (float64).
        At batch_size=50_000 and 188 facilities: ~75 MB — tune down if needed.

        Args:
            batch_size: Pixels per processing batch.
            return_pixel_allocations: If True, return pixel-level primary-facility table.
            progress_interval: Print progress every N pixels processed.

        Returns:
            (facility_df, pixel_df) — same schema as allocate_all_pixels_probabilistic.
        """
        print(f"\nAllocating population pixels to facilities (VECTORIZED)...")

        fac_ids    = self.facilities['HealthFacility'].values  # (F,)
        fac_df_idx = self.facilities.index.values              # (F,) DataFrame index labels
        n_fac      = len(fac_ids)

        # ── 1. Read raster, extract all populated sampled pixels at once ───
        with rasterio.open(self.pop_raster_path) as src:
            pop_data  = src.read(1)
            transform = src.transform
            height, width = pop_data.shape

        sample_rate = self.params['sample_rate']
        print(f"  Raster size: {height} x {width} = {height * width:,} pixels")
        if sample_rate > 1:
            print(f"  Sampling: every {sample_rate}th pixel "
                  f"({sample_rate ** 2}x reduction)")

        sampled             = pop_data[::sample_rate, ::sample_rate]  # (H_s, W_s)
        local_rows, local_cols = np.where(sampled > 0)
        orig_rows = local_rows * sample_rate
        orig_cols = local_cols * sample_rate
        pixel_pops = sampled[local_rows, local_cols].astype(np.float64)

        # Vectorised pixel-centre geographic coordinates via the raster affine transform
        ta, tb, tc = transform.a, transform.b, transform.c
        td, te, tf = transform.d, transform.e, transform.f
        pixel_lons = tc + ta * (orig_cols + 0.5) + tb * (orig_rows + 0.5)
        pixel_lats = tf + td * (orig_cols + 0.5) + te * (orig_rows + 0.5)

        n_pixels  = len(pixel_pops)
        total_pop = float(pixel_pops.sum())

        print(f"  Populated sampled pixels: {n_pixels:,}")
        print(f"  Total sampled population: {total_pop:,.0f}")

        if n_pixels == 0:
            print("  WARNING: No populated pixels found after sampling.")
            return pd.DataFrame(columns=['facility_id', 'allocated_population']), None

        n_batches = math.ceil(n_pixels / batch_size)
        print(f"  Batches: {n_batches} (batch_size={batch_size:,})")

        # ── 2. Process in batches, accumulating facility totals ────────────
        facility_totals = np.zeros(n_fac, dtype=np.float64)
        allocated_pop   = 0.0
        pixel_count     = 0

        # Per-batch column arrays for pixel-level output (avoids append per pixel)
        px_lons_list  = []
        px_lats_list  = []
        px_pops_list  = []
        px_fidx_list  = []
        px_fid_list   = []
        px_prob_list  = []
        px_ncand_list = []

        for batch_idx in range(n_batches):
            s = batch_idx * batch_size
            e = min(s + batch_size, n_pixels)

            b_lons = pixel_lons[s:e]
            b_lats = pixel_lats[s:e]
            b_pops = pixel_pops[s:e]

            dist_km        = self._haversine_batch(b_lons, b_lats)   # (N_b, F)
            probs, covered = self._compute_gravity_batch(dist_km)     # (N_b, F), (N_b,)

            # Accumulate: facility_totals[f] += Σ_i  pop_i * prob[i, f]
            facility_totals += (b_pops[:, None] * probs).sum(axis=0)
            allocated_pop   += float(b_pops[covered].sum())
            pixel_count     += len(b_pops)

            if return_pixel_allocations and covered.any():
                primary_pos  = np.argmax(probs, axis=1)               # (N_b,)
                cov          = covered
                px_lons_list.append(b_lons[cov])
                px_lats_list.append(b_lats[cov])
                px_pops_list.append(b_pops[cov])
                px_fidx_list.append(fac_df_idx[primary_pos[cov]])
                px_fid_list.append(fac_ids[primary_pos[cov]])
                px_prob_list.append(
                    probs[np.arange(len(b_pops)), primary_pos][cov]
                )
                px_ncand_list.append(
                    (dist_km[cov] <= self.params['max_distance_km']).sum(axis=1)
                )

            if pixel_count % progress_interval < batch_size or batch_idx == n_batches - 1:
                pct       = pixel_count / n_pixels * 100
                pct_alloc = allocated_pop / total_pop * 100 if total_pop > 0 else 0
                print(f"    Progress: {pct:5.1f}% | {pixel_count:,} pixels | "
                      f"{pct_alloc:.1f}% population allocated")

        # ── 3. Build output DataFrames ─────────────────────────────────────
        facility_df = pd.DataFrame({
            'facility_id':         fac_ids,
            'allocated_population': facility_totals,
        })
        facility_df = (
            facility_df[facility_df['allocated_population'] > 0]
            .sort_values('allocated_population', ascending=False)
            .reset_index(drop=True)
        )

        if return_pixel_allocations and px_lons_list:
            pixel_df = pd.DataFrame({
                'lon':            np.concatenate(px_lons_list),
                'lat':            np.concatenate(px_lats_list),
                'population':     np.concatenate(px_pops_list),
                'facility_idx':   np.concatenate(px_fidx_list),
                'facility_id':    np.concatenate(px_fid_list),
                'probability':    np.concatenate(px_prob_list),
                'num_candidates': np.concatenate(px_ncand_list),
            })
        else:
            pixel_df = None

        unallocated_pop = total_pop - allocated_pop
        print(f"\n  Vectorized allocation complete:")
        print(f"    Total population: {total_pop:,.0f}")
        print(f"    Allocated population: {allocated_pop:,.0f} "
              f"({allocated_pop / total_pop * 100:.1f}%)")
        print(f"    Unallocated population: {unallocated_pop:,.0f} "
              f"({unallocated_pop / total_pop * 100:.1f}%)")
        print(f"    Pixels processed: {pixel_count:,}")
        print(f"    Facilities receiving allocation: {len(facility_df)}")
        if len(facility_df) > 0:
            print(f"    Sum of facility allocations: "
                  f"{facility_df['allocated_population'].sum():,.0f}")
        if pixel_df is not None:
            print(f"    Pixel allocations tracked: {len(pixel_df):,}")

        return facility_df, pixel_df

    def aggregate_by_facility(self, allocations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate allocated population by facility

        Args:
            allocations_df: Output from allocate_all_pixels()

        Returns:
            DataFrame with facility-level summaries
        """
        # Group by facility
        grouped = allocations_df.groupby('facility_idx').agg({
            'population': 'sum',
            'probability': 'mean',
            'num_candidates': 'mean'
        }).reset_index()

        # Join with facility info
        fac_info = self.facilities[['HealthFacility', 'volume']].copy()
        result = grouped.merge(fac_info, left_on='facility_idx', right_index=True, how='left')

        # Rename columns
        result.columns = ['facility_idx', 'allocated_population', 'mean_probability',
                         'mean_candidates', 'facility_name', 'original_volume']

        # Calculate ratio
        result['allocation_ratio'] = result['allocated_population'] / result['original_volume']

        # Sort by allocated population
        result = result.sort_values('allocated_population', ascending=False).reset_index(drop=True)

        return result

    def aggregate_by_hsa(self, allocations_df: pd.DataFrame,
                        hsa_anchors: gpd.GeoDataFrame,
                        all_facilities: Optional[gpd.GeoDataFrame] = None,
                        network_type: Optional[str] = None,
                        optimization_mode: Optional[str] = None) -> pd.DataFrame:
        """
        Aggregate allocated population by HSA catchment areas (all facilities within HSA circles)

        This creates the MODELING DATASET - one row per HSA with allocated patients.
        For each HSA anchor, sums up ALL facilities within its service radius.
        This eliminates double-counting across HSAs while including all facilities.

        Args:
            allocations_df: Output from allocate_all_pixels()
            hsa_anchors: GeoDataFrame with HSA anchor facilities (with service_radius_km)
            all_facilities: GeoDataFrame with ALL facilities in network
            network_type: 'INF' or 'NCD'
            optimization_mode: e.g., 'fewest', 'footprint', etc.

        Returns:
            DataFrame ready for climate modeling with columns:
                anchor_id, anchor_name, network_type, optimization_mode,
                allocated_patients, num_facilities_in_hsa
        """
        # Handle empty allocations
        if len(allocations_df) == 0:
            raise ValueError("No pixels were allocated! Check that facilities and population raster "
                           "are in compatible coordinate systems and within max_distance.")

        if all_facilities is None:
            all_facilities = self.facilities
        if network_type is None:
            network_type = "unknown"
        if optimization_mode is None:
            optimization_mode = "unspecified"

        # Ensure both are in same CRS (WGS84)
        if hsa_anchors.crs and hsa_anchors.crs.to_epsg() != 4326:
            hsa_anchors = hsa_anchors.to_crs(epsg=4326)
        if all_facilities.crs and all_facilities.crs.to_epsg() != 4326:
            all_facilities = all_facilities.to_crs(epsg=4326)

        print(f"\nAggregating by HSA catchment areas...")
        anchors = hsa_anchors[['HealthFacility', 'service_radius_km', 'geometry']].copy()

        def _point_xy(geom):
            if geom is None:
                return None
            if geom.geom_type == "Point":
                return geom.x, geom.y
            center = geom.representative_point()
            return center.x, center.y

        # Precompute distances from each facility to each anchor
        facility_to_anchor = []
        for _, fac in all_facilities.iterrows():
            fac_name = fac['HealthFacility']
            fac_point = fac.geometry
            fac_xy = _point_xy(fac_point)
            if fac_xy is None:
                continue

            # Compute distances to all anchors
            def _dist(anchor_geom):
                anchor_xy = _point_xy(anchor_geom)
                if anchor_xy is None:
                    return np.nan
                return GeoUtils.haversine_km(
                    anchor_xy[0], anchor_xy[1],
                    fac_xy[0], fac_xy[1]
                )

            distances_km = anchors.geometry.apply(_dist)

            # Keep anchors where facility is inside service radius
            within_radius = distances_km <= anchors['service_radius_km'].values
            if not within_radius.any():
                continue

            eligible = anchors[within_radius].copy()
            eligible['distance_km'] = distances_km[within_radius].values
            closest_anchor = eligible.sort_values('distance_km', ascending=True).iloc[0]

            facility_to_anchor.append({
                'facility_id': fac_name,
                'anchor_name': closest_anchor['HealthFacility'],
                'distance_km': closest_anchor['distance_km']
            })

        facility_to_anchor_df = pd.DataFrame(facility_to_anchor)

        # Sum allocated population per facility, then map to anchors
        facility_allocations = allocations_df.groupby('facility_id', as_index=False)['population'].sum()
        facility_allocations = facility_allocations.merge(
            facility_to_anchor_df, on='facility_id', how='inner'
        )

        # Aggregate by anchor
        hsa_results = facility_allocations.groupby('anchor_name', as_index=False).agg(
            allocated_patients=('population', 'sum'),
            num_facilities_in_hsa=('facility_id', 'nunique')
        )

        # Add a few facility names for reference
        facility_lists = facility_allocations.groupby('anchor_name')['facility_id'].apply(
            lambda s: ', '.join(s.head(5))
        ).reset_index(name='facilities_in_hsa')
        hsa_results = hsa_results.merge(facility_lists, on='anchor_name', how='left')

        # Ensure anchors with zero assigned facilities still appear
        anchors_names = anchors[['HealthFacility']].rename(columns={'HealthFacility': 'anchor_name'})
        hsa_results = anchors_names.merge(
            hsa_results, on='anchor_name', how='left'
        )
        hsa_results['allocated_patients'] = hsa_results['allocated_patients'].fillna(0)
        hsa_results['num_facilities_in_hsa'] = hsa_results['num_facilities_in_hsa'].fillna(0).astype(int)
        hsa_results['facilities_in_hsa'] = hsa_results['facilities_in_hsa'].fillna('')

        for _, row in hsa_results.iterrows():
            print(f"  {row['anchor_name']}: {row['num_facilities_in_hsa']} facilities, "
                  f"{row['allocated_patients']:,.0f} people")

        hsa_summary = pd.DataFrame(hsa_results)

        # Add metadata
        hsa_summary['network_type'] = network_type
        hsa_summary['optimization_mode'] = optimization_mode
        hsa_summary['anchor_id'] = range(1, len(hsa_summary) + 1)

        # Reorder columns
        hsa_summary = hsa_summary[['anchor_id', 'anchor_name', 'network_type',
                                   'optimization_mode', 'allocated_patients',
                                   'num_facilities_in_hsa', 'facilities_in_hsa']]

        # Sort by allocated patients
        hsa_summary = hsa_summary.sort_values('allocated_patients', ascending=False).reset_index(drop=True)

        # Re-number anchor_id after sorting
        hsa_summary['anchor_id'] = range(1, len(hsa_summary) + 1)

        print(f"\nTotal HSA allocated population: {hsa_summary['allocated_patients'].sum():,.0f}")

        return hsa_summary

    def create_comparison_report(self, allocations_df: pd.DataFrame,
                                hsa_anchors: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Create comparison report: allocated vs. HSA circle population

        Args:
            allocations_df: Output from allocate_all_pixels()
            hsa_anchors: GeoDataFrame with HSA anchors (must have HSA population from circles)

        Returns:
            DataFrame comparing allocation methods
        """
        # Aggregate by facility
        facility_summary = self.aggregate_by_facility(allocations_df)

        # Get anchor IDs
        anchor_ids = set(hsa_anchors['HealthFacility'].values)

        # Filter to anchors only
        anchor_summary = facility_summary[facility_summary['facility_name'].isin(anchor_ids)].copy()

        # Try to join with HSA population if available
        if 'hsa_population' in hsa_anchors.columns:
            hsa_pop = hsa_anchors[['HealthFacility', 'hsa_population']].copy()
            hsa_pop.columns = ['facility_name', 'hsa_circle_population']
            anchor_summary = anchor_summary.merge(hsa_pop, on='facility_name', how='left')

        return anchor_summary

    def aggregate_facilities_to_hsas(self, facility_allocations: pd.DataFrame,
                                      hsa_anchors: gpd.GeoDataFrame,
                                      all_facilities: gpd.GeoDataFrame,
                                      network_type: str = "unknown",
                                      optimization_mode: str = "unspecified",
                                      max_assignment_distance_km: float = 100.0,
                                      fallback_radius_multiplier: float = 1.5,
                                      fallback_min_distance_km: float = 30.0,
                                      major_facility_pop_threshold: float = 25000.0,
                                      major_facility_volume_threshold: Optional[float] = None,
                                      major_facility_volume_quantile: float = 0.80,
                                      require_same_governorate_for_major: bool = True,
                                      alpha: Optional[float] = None,
                                      beta: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate facility-level populations to HSAs using three-case logic.

        This method handles the PROBABILISTIC allocation workflow where each pixel's
        population is distributed across multiple facilities. It then maps each
        facility's total allocated population to HSAs based on spatial containment.

        THREE CASES for facility-to-HSA assignment:
        1. Facility inside EXACTLY 1 HSA → 100% of population to that HSA
        2. Facility OUTSIDE ALL HSAs → Assign to an admissible nearby HSA anchor.
           Large/hospital facilities are excluded for anchor promotion when the
           only fallback is implausibly distant or cross-governorate.
        3. Facility inside 2+ OVERLAPPING HSAs → Allocate proportionally based on
           gravity scores to each containing HSA's anchor

        Args:
            facility_allocations: DataFrame with columns [facility_id, allocated_population]
                                  Output from allocate_all_pixels_probabilistic()
            hsa_anchors: GeoDataFrame with HSA anchor facilities (must have service_radius_km)
            all_facilities: GeoDataFrame with ALL facilities (must have HealthFacility, geometry)
            network_type: 'INF' or 'NCD'
            optimization_mode: e.g., 'fewest', 'footprint', etc.
            max_assignment_distance_km: Max distance for assigning facilities outside all HSAs
            fallback_radius_multiplier: Outside-HSA fallback may extend this many
                                      service radii from an anchor
            fallback_min_distance_km: Minimum absolute fallback allowance for
                                      small-radius anchors
            major_facility_pop_threshold: Allocated-population threshold above
                                          which Case 2 requires a plausible anchor
            major_facility_volume_threshold: Diagnosis/volume threshold for major
                                             facility classification. If None,
                                             computed from all facility volumes.
            major_facility_volume_quantile: Quantile used when deriving the major
                                            facility volume threshold.
            require_same_governorate_for_major: Major outside-HSA facilities will
                                                not be assigned cross-governorate
            alpha: Facility size weight for gravity model (default: use self.params['alpha'])
            beta: Distance decay for gravity model (default: use self.params['beta'])

        Returns:
            Tuple of (hsa_summary_df, facility_assignment_df):
                - hsa_summary: DataFrame with columns [anchor_id, anchor_name, network_type,
                              optimization_mode, allocated_patients, num_facilities_in_hsa,
                              facilities_in_hsa]
                - facility_assignment: DataFrame showing how each facility was assigned
                              (includes assignment_case, assigned_hsa(s), etc.)
        """
        print(f"\n{'='*80}")
        print("AGGREGATING FACILITIES TO HSAs (THREE-CASE LOGIC)")
        print(f"{'='*80}")

        # Use gravity parameters from allocator if not specified
        if alpha is None:
            alpha = self.params.get('alpha', 0.75)
        if beta is None:
            beta = self.params.get('beta', 1.5)

        print(f"  Gravity model parameters: alpha={alpha}, beta={beta}")
        print(
            "  Case 2 fallback guard: "
            f"max={max_assignment_distance_km:.1f}km, "
            f"radius_multiplier={fallback_radius_multiplier:.2f}, "
            f"min_allowance={fallback_min_distance_km:.1f}km, "
            f"major_pop_threshold={major_facility_pop_threshold:,.0f}"
        )

        # Ensure consistent CRS
        if hsa_anchors.crs and hsa_anchors.crs.to_epsg() != 4326:
            hsa_anchors = hsa_anchors.to_crs(epsg=4326)
        if all_facilities.crs and all_facilities.crs.to_epsg() != 4326:
            all_facilities = all_facilities.to_crs(epsg=4326)

        if major_facility_volume_threshold is None:
            volume_col = None
            for candidate_col in ('Total', 'total_diagnoses', 'volume', 'patient_volume'):
                if candidate_col in all_facilities.columns:
                    volume_col = candidate_col
                    break
            if volume_col is not None:
                positive_volume = pd.to_numeric(all_facilities[volume_col], errors='coerce')
                positive_volume = positive_volume[positive_volume > 0]
                if len(positive_volume) > 0:
                    major_facility_volume_threshold = float(
                        positive_volume.quantile(major_facility_volume_quantile)
                    )
                else:
                    major_facility_volume_threshold = float('inf')
            else:
                major_facility_volume_threshold = float('inf')
        print(f"  Major facility volume threshold: {major_facility_volume_threshold:,.0f}")

        # Helper function to get coordinates
        def _get_coords(geom):
            if geom is None or geom.is_empty:
                return None
            if geom.geom_type == "Point":
                return (geom.x, geom.y)
            return (geom.representative_point().x, geom.representative_point().y)

        def _get_row_coords(row):
            lon = None
            lat = None
            for lon_col in ('lon', 'Longitude', 'LONGITUDE'):
                if lon_col in row and pd.notna(row.get(lon_col)):
                    lon = float(row.get(lon_col))
                    break
            for lat_col in ('lat', 'Latitude', 'LATITUDE'):
                if lat_col in row and pd.notna(row.get(lat_col)):
                    lat = float(row.get(lat_col))
                    break
            if lon is not None and lat is not None:
                return (lon, lat)
            return _get_coords(row.geometry)

        # Build HSA anchor info (including volumes for gravity model)
        anchor_cols = ['HealthFacility', 'service_radius_km', 'geometry']
        for optional_col in ('governorate', 'healthfacilitytype', 'facility_type'):
            if optional_col in hsa_anchors.columns and optional_col not in anchor_cols:
                anchor_cols.append(optional_col)
        for optional_col in ('lon', 'lat', 'Longitude', 'Latitude'):
            if optional_col in hsa_anchors.columns and optional_col not in anchor_cols:
                anchor_cols.append(optional_col)
        anchors = hsa_anchors[anchor_cols].copy()
        anchors['anchor_coords'] = anchors.apply(_get_row_coords, axis=1)

        # Get anchor volumes from hsa_anchors or all_facilities
        if 'Total' in hsa_anchors.columns:
            anchors['volume'] = hsa_anchors['Total'].values
        elif 'patient_volume' in hsa_anchors.columns:
            anchors['volume'] = hsa_anchors['patient_volume'].values
        else:
            # Look up volumes from all_facilities
            anchor_volumes = []
            for anchor_name in anchors['HealthFacility']:
                match = all_facilities[all_facilities['HealthFacility'] == anchor_name]
                if not match.empty and 'Total' in match.columns:
                    anchor_volumes.append(match.iloc[0]['Total'])
                elif not match.empty and 'volume' in match.columns:
                    anchor_volumes.append(match.iloc[0]['volume'])
                else:
                    anchor_volumes.append(1000)  # Default volume
            anchors['volume'] = anchor_volumes

        # Build lists for efficient iteration
        anchor_coords_list = [c for c in anchors['anchor_coords'].values if c is not None]
        anchor_names_list = anchors['HealthFacility'].values.tolist()
        anchor_radii_list = anchors['service_radius_km'].values.tolist()
        anchor_volumes_list = anchors['volume'].values.tolist()
        anchor_governorates_list = (
            anchors['governorate'].fillna('').astype(str).str.strip().values.tolist()
            if 'governorate' in anchors.columns else [''] * len(anchors)
        )

        # Process each facility
        facility_assignments = []
        hsa_populations = {name: 0.0 for name in anchor_names_list}
        hsa_facility_lists = {name: [] for name in anchor_names_list}

        excluded_facilities = []
        case_counts = {'case_1': 0, 'case_2': 0, 'case_3': 0, 'excluded': 0}

        for _, fac_row in facility_allocations.iterrows():
            fac_id = fac_row['facility_id']
            fac_pop = fac_row['allocated_population']

            # Get facility coordinates
            fac_match = all_facilities[all_facilities['HealthFacility'] == fac_id]
            if fac_match.empty:
                print(f"  WARNING: Facility '{fac_id}' not found in all_facilities")
                continue

            fac_geom = fac_match.iloc[0].geometry
            fac_coords = _get_row_coords(fac_match.iloc[0])
            if fac_coords is None:
                continue
            fac_info = fac_match.iloc[0]
            fac_type = ''
            for type_col in ('healthfacilitytype', 'facility_type', 'FacilityType', 'type'):
                if type_col in fac_match.columns and pd.notna(fac_info.get(type_col)):
                    fac_type = str(fac_info.get(type_col)).strip()
                    break
            fac_governorate = ''
            if 'governorate' in fac_match.columns and pd.notna(fac_info.get('governorate')):
                fac_governorate = str(fac_info.get('governorate')).strip()
            fac_volume = np.nan
            for volume_col in ('Total', 'total_diagnoses', 'volume', 'patient_volume'):
                if volume_col in fac_match.columns and pd.notna(fac_info.get(volume_col)):
                    fac_volume = float(pd.to_numeric(fac_info.get(volume_col), errors='coerce'))
                    break
            is_hospital = 'hospital' in fac_type.lower()
            is_major_facility = (
                is_hospital
                or fac_pop >= major_facility_pop_threshold
                or (pd.notna(fac_volume) and fac_volume >= major_facility_volume_threshold)
            )

            # Calculate distance to all HSA anchors
            containing_hsas = []
            distances_to_anchors = []

            for i, (anchor_name, anchor_radius, anchor_volume, anchor_governorate) in enumerate(
                    zip(anchor_names_list, anchor_radii_list, anchor_volumes_list, anchor_governorates_list)):
                anchor_coords = anchor_coords_list[i]
                dist_km = GeoUtils.haversine_km(
                    fac_coords[0], fac_coords[1],
                    anchor_coords[0], anchor_coords[1]
                )
                distances_to_anchors.append((anchor_name, dist_km, anchor_radius, anchor_volume, anchor_governorate))

                # Check if facility is within this HSA's service radius
                if dist_km <= anchor_radius:
                    containing_hsas.append((anchor_name, dist_km, anchor_radius, anchor_volume))

            # Determine assignment case
            if len(containing_hsas) == 1:
                # CASE 1: Inside exactly ONE HSA
                case_counts['case_1'] += 1
                anchor_name = containing_hsas[0][0]
                hsa_populations[anchor_name] += fac_pop
                hsa_facility_lists[anchor_name].append(fac_id)

                facility_assignments.append({
                    'facility_id': fac_id,
                    'allocated_population': fac_pop,
                    'assignment_case': 'Case 1: Inside 1 HSA',
                    'primary_hsa': anchor_name,
                    'distance_to_primary_km': containing_hsas[0][1],
                    'num_containing_hsas': 1,
                    'all_containing_hsas': anchor_name,
                    'excluded': False
                })

            elif len(containing_hsas) == 0:
                # CASE 2: Outside ALL HSAs - find nearest admissible anchor.
                # Do not let major facilities be absorbed by distant small anchors.
                distances_to_anchors.sort(key=lambda x: x[1])
                nearest = distances_to_anchors[0]
                nearest_name, nearest_dist, nearest_radius, _, nearest_governorate = nearest

                admissible = []
                rejected = []
                for anchor_name, dist_km, anchor_radius, anchor_volume, anchor_governorate in distances_to_anchors:
                    anchor_limit = min(
                        max_assignment_distance_km,
                        max(float(anchor_radius) * fallback_radius_multiplier, fallback_min_distance_km)
                    )
                    same_governorate = (
                        bool(fac_governorate)
                        and bool(anchor_governorate)
                        and fac_governorate.lower() == anchor_governorate.lower()
                    )
                    if dist_km <= anchor_limit:
                        candidate = (anchor_name, dist_km, anchor_radius, anchor_volume, anchor_governorate, same_governorate, anchor_limit)
                        admissible.append(candidate)
                    else:
                        rejected.append((anchor_name, dist_km, anchor_limit))

                same_gov_admissible = [c for c in admissible if c[5]]
                if same_gov_admissible:
                    chosen = same_gov_admissible[0]
                    fallback_note = 'same-governorate fallback'
                elif (
                    admissible
                    and is_major_facility
                    and require_same_governorate_for_major
                    and fac_governorate
                    and admissible[0][1] <= fallback_min_distance_km
                ):
                    chosen = admissible[0]
                    fallback_note = 'nearby cross-governorate fallback'
                elif admissible and not (is_major_facility and require_same_governorate_for_major and fac_governorate):
                    chosen = admissible[0]
                    fallback_note = 'nearest admissible fallback'
                else:
                    chosen = None
                    if not admissible:
                        fallback_note = (
                            f"nearest anchor {nearest_name} is {nearest_dist:.1f}km away; "
                            f"allowed fallback is {max(float(nearest_radius) * fallback_radius_multiplier, fallback_min_distance_km):.1f}km"
                        )
                    else:
                        fallback_note = (
                            f"major facility requires same-governorate fallback; "
                            f"nearest admissible anchor is {admissible[0][0]}"
                        )

                if chosen is not None:
                    nearest_name, nearest_dist, _, _, _, _, _ = chosen
                    case_counts['case_2'] += 1
                    hsa_populations[nearest_name] += fac_pop
                    hsa_facility_lists[nearest_name].append(fac_id)

                    facility_assignments.append({
                        'facility_id': fac_id,
                        'allocated_population': fac_pop,
                        'assignment_case': 'Case 2: Outside all HSAs (nearest)',
                        'primary_hsa': nearest_name,
                        'distance_to_primary_km': nearest_dist,
                        'num_containing_hsas': 0,
                        'all_containing_hsas': '',
                        'excluded': False,
                        'is_major_facility': bool(is_major_facility),
                        'facility_type': fac_type,
                        'facility_governorate': fac_governorate,
                        'fallback_note': fallback_note
                    })
                else:
                    # Excluded - no plausible fallback; upstream HSA selection should
                    # consider promoting this facility to an anchor.
                    case_counts['excluded'] += 1
                    excluded_facilities.append({
                        'facility_id': fac_id,
                        'allocated_population': fac_pop,
                        'nearest_anchor': nearest_name,
                        'distance_km': nearest_dist,
                        'reason': fallback_note
                    })

                    facility_assignments.append({
                        'facility_id': fac_id,
                        'allocated_population': fac_pop,
                        'assignment_case': 'EXCLUDED: Requires anchor promotion',
                        'primary_hsa': None,
                        'distance_to_primary_km': nearest_dist,
                        'num_containing_hsas': 0,
                        'all_containing_hsas': '',
                        'excluded': True,
                        'is_major_facility': bool(is_major_facility),
                        'facility_type': fac_type,
                        'facility_governorate': fac_governorate,
                        'fallback_note': fallback_note
                    })

            else:
                # CASE 3: Inside 2+ OVERLAPPING HSAs - proportional allocation using gravity model
                case_counts['case_3'] += 1

                # Calculate gravity-based weights for each containing HSA
                # Using full gravity model: weight = volume^α / distance^β
                weights = []
                for anchor_name, dist_km, _, anchor_volume in containing_hsas:
                    # Avoid division by zero
                    dist_km = max(dist_km, 0.01)
                    # Gravity model: volume^α / distance^β
                    weight = (anchor_volume ** alpha) / (dist_km ** beta)
                    weights.append((anchor_name, weight, dist_km))

                total_weight = sum(w[1] for w in weights)

                # Allocate proportionally
                hsa_allocations = []
                for anchor_name, weight, dist_km in weights:
                    proportion = weight / total_weight
                    pop_share = fac_pop * proportion
                    hsa_populations[anchor_name] += pop_share
                    hsa_allocations.append(f"{anchor_name}:{proportion:.1%}")

                # For facility list, assign to closest containing HSA
                closest_containing = min(containing_hsas, key=lambda x: x[1])
                hsa_facility_lists[closest_containing[0]].append(fac_id)

                facility_assignments.append({
                    'facility_id': fac_id,
                    'allocated_population': fac_pop,
                    'assignment_case': 'Case 3: Inside 2+ HSAs (proportional)',
                    'primary_hsa': closest_containing[0],
                    'distance_to_primary_km': closest_containing[1],
                    'num_containing_hsas': len(containing_hsas),
                    'all_containing_hsas': '; '.join(hsa_allocations),
                    'excluded': False
                })

        # Print summary
        print(f"\nFacility Assignment Summary:")
        print(f"  Case 1 (Inside 1 HSA):           {case_counts['case_1']:4d} facilities")
        print(f"  Case 2 (Outside, assigned):      {case_counts['case_2']:4d} facilities")
        print(f"  Case 3 (Overlapping, proportional): {case_counts['case_3']:4d} facilities")
        print(f"  EXCLUDED (no admissible fallback): {case_counts['excluded']:4d} facilities")
        print(f"  Total:                           {sum(case_counts.values()):4d} facilities")

        if excluded_facilities:
            print(f"\nExcluded Facilities (no admissible fallback; anchor promotion needed):")
            excluded_pop_total = sum(f['allocated_population'] for f in excluded_facilities)
            print(f"  Count: {len(excluded_facilities)}")
            print(f"  Total excluded population: {excluded_pop_total:,.0f}")
            for ef in excluded_facilities[:5]:  # Show first 5
                print(f"    - {ef['facility_id']}: {ef['allocated_population']:,.0f} people "
                      f"(nearest: {ef['nearest_anchor']}, {ef['distance_km']:.1f}km)")
            if len(excluded_facilities) > 5:
                print(f"    ... and {len(excluded_facilities) - 5} more")

        # Build HSA summary DataFrame
        hsa_results = []
        for anchor_name in anchor_names_list:
            fac_list = hsa_facility_lists[anchor_name]
            hsa_results.append({
                'anchor_name': anchor_name,
                'allocated_patients': hsa_populations[anchor_name],
                'num_facilities_in_hsa': len(fac_list),
                'facilities_in_hsa': ', '.join(fac_list[:5]) + ('...' if len(fac_list) > 5 else '')
            })

        hsa_summary = pd.DataFrame(hsa_results)
        hsa_summary['network_type'] = network_type
        hsa_summary['optimization_mode'] = optimization_mode
        hsa_summary = hsa_summary.sort_values('allocated_patients', ascending=False).reset_index(drop=True)
        hsa_summary['anchor_id'] = range(1, len(hsa_summary) + 1)

        # Reorder columns
        hsa_summary = hsa_summary[['anchor_id', 'anchor_name', 'network_type',
                                   'optimization_mode', 'allocated_patients',
                                   'num_facilities_in_hsa', 'facilities_in_hsa']]

        facility_assignment_df = pd.DataFrame(facility_assignments)

        # Print HSA summary
        print(f"\nHSA Population Summary:")
        print(f"{'HSA Anchor':<45} {'Population':>15} {'Facilities':>12}")
        print("-" * 75)
        for _, row in hsa_summary.iterrows():
            print(f"  {row['anchor_name']:<43} {row['allocated_patients']:>15,.0f} {row['num_facilities_in_hsa']:>12d}")
        print("-" * 75)
        print(f"  {'TOTAL':<43} {hsa_summary['allocated_patients'].sum():>15,.0f} "
              f"{hsa_summary['num_facilities_in_hsa'].sum():>12d}")

        print(f"\n{'='*80}")

        return hsa_summary, facility_assignment_df


def allocate_population_for_hsa_mode(hsa_geojson_path: str,
                                   pop_raster_path: str,
                                   network_type: str,
                                   optimization_mode: str,
                                   output_dir: Path,
                                   params: Optional[Dict] = None,
                                   boundary_version: str = "v7") -> Dict[str, pd.DataFrame]:
    """
    Convenience function to allocate population for one HSA mode

    Args:
        hsa_geojson_path: Path to HSA GeoJSON file (e.g., INF_fewest_hsas_v2.geojson)
        pop_raster_path: Path to population raster
        network_type: 'INF' or 'NCD'
        optimization_mode: e.g., 'fewest', 'footprint', etc.
        output_dir: Directory for output files
        params: Optional allocation parameters

    Returns:
        Dict with keys: 'hsa_summary', 'facility_summary', 'allocations'
    """
    print("="*80)
    print(f"POPULATION ALLOCATION: {network_type} - {optimization_mode.upper()}")
    print("="*80)

    # Load HSA anchors
    hsa_anchors = gpd.read_file(hsa_geojson_path)
    print(f"\nLoaded {len(hsa_anchors)} HSA anchor facilities")

    # Load ALL facilities for the network (for allocation)
    # Assuming standard file naming
    all_fac_path = Path(hsa_geojson_path).parent.parent / 'data' / f'{network_type}_hospitals_projected_total_WITH_CLIMATE.gpkg'
    all_facilities = gpd.read_file(all_fac_path)
    print(f"Loaded {len(all_facilities)} total facilities for allocation")

    # Initialize allocator
    allocator = PopulationAllocator(pop_raster_path, all_facilities, params)

    # Allocate all pixels
    allocations = allocator.allocate_all_pixels()

    # Aggregate by HSA (MODELING DATASET)
    # Pass all_facilities so we can sum up facilities within each HSA circle
    hsa_summary = allocator.aggregate_by_hsa(allocations, hsa_anchors, all_facilities, network_type, optimization_mode)

    # Aggregate by facility (for validation)
    facility_summary = allocator.aggregate_by_facility(allocations)

    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. HSA modeling dataset (PRIMARY OUTPUT)
    hsa_output = output_dir / f'{network_type}_{optimization_mode}_allocated_patients.csv'
    hsa_summary.to_csv(hsa_output, index=False)
    print(f"\n  Saved HSA modeling dataset: {hsa_output.name}")

    # 2. Facility-level summary (for validation)
    fac_output = output_dir / f'{network_type}_{optimization_mode}_facility_allocations_{boundary_version}.csv'
    facility_summary.to_csv(fac_output, index=False)
    print(f"  Saved facility summary: {fac_output.name}")

    # 3. Detailed allocations (optional - can be large)
    detail_output = output_dir / f'{network_type}_{optimization_mode}_allocation_details_{boundary_version}.csv'
    allocations.to_csv(detail_output, index=False)
    print(f"  Saved allocation details: {detail_output.name}")

    print("\n" + "="*80)

    return {
        'hsa_summary': hsa_summary,
        'facility_summary': facility_summary,
        'allocations': allocations
    }


def allocate_population_probabilistic(hsa_geojson_path: str,
                                     all_facilities_path: str,
                                     pop_raster_path: str,
                                     network_type: str,
                                     optimization_mode: str,
                                     output_dir: Path,
                                     params: Optional[Dict] = None,
                                     use_vectorized: bool = True,
                                     use_parallel: bool = True,
                                     boundary_version: str = "v7") -> Dict[str, pd.DataFrame]:
    """
    Two-step probabilistic population allocation workflow.

    STEP 1: Allocate each population pixel to ALL 188 facilities using probabilistic
            gravity model (population split based on attractiveness probabilities)

    STEP 2: Aggregate facility populations to HSAs using three-case logic:
            - Case 1: Facility inside exactly 1 HSA → 100% to that HSA
            - Case 2: Facility outside all HSAs → assign to nearest anchor within 100km
            - Case 3: Facility inside 2+ HSAs → proportional allocation by gravity

    Args:
        hsa_geojson_path: Path to HSA GeoJSON file (e.g., INF_footprint_hsas_v2.geojson)
        all_facilities_path: Path to ALL facilities GeoPackage/CSV
        pop_raster_path: Path to population raster
        network_type: 'INF' or 'NCD'
        optimization_mode: e.g., 'fewest', 'footprint', etc.
        output_dir: Directory for output files
        params: Optional allocation parameters (alpha, beta, max_distance_km, sample_rate)
        use_vectorized: Use batched NumPy matrix ops (~50-200x faster, same results).
                        Falls back to use_parallel if False.
        use_parallel: Used only when use_vectorized=False.

    Returns:
        Dict with keys:
            - 'hsa_summary': HSA-level population totals
            - 'facility_allocations': Facility-level population totals from pixels
            - 'facility_assignments': How each facility was assigned to HSAs
    """
    print("="*80)
    print(f"PROBABILISTIC POPULATION ALLOCATION: {network_type} - {optimization_mode.upper()}")
    print("="*80)
    print("\nThis workflow uses TWO-STEP allocation:")
    print("  Step 1: Probabilistic pixel → ALL facilities (gravity model)")
    print("  Step 2: Facility → HSA aggregation (three-case logic)")

    # Load HSA anchors
    hsa_anchors = gpd.read_file(hsa_geojson_path)
    print(f"\nLoaded {len(hsa_anchors)} HSA anchor facilities")

    # Load ALL facilities
    all_facilities_path = Path(all_facilities_path)
    if all_facilities_path.suffix == '.gpkg':
        all_facilities = gpd.read_file(all_facilities_path)
    else:
        all_fac_df = pd.read_csv(all_facilities_path)
        all_facilities = gpd.GeoDataFrame(
            all_fac_df,
            geometry=gpd.points_from_xy(all_fac_df['lon'], all_fac_df['lat']),
            crs='EPSG:4326'
        )

    print(f"Loaded {len(all_facilities)} total facilities for allocation")

    # Initialize allocator with ALL facilities
    allocator = PopulationAllocator(pop_raster_path, all_facilities, params)

    # STEP 1: Probabilistic allocation to ALL facilities
    print("\n" + "-"*40)
    print("STEP 1: Probabilistic pixel allocation")
    print("-"*40)

    if use_vectorized:
        facility_allocations, pixel_allocations = allocator.allocate_all_pixels_probabilistic_vectorized()
    elif use_parallel:
        facility_allocations, pixel_allocations = allocator.allocate_all_pixels_probabilistic_parallel()
    else:
        facility_allocations, pixel_allocations = allocator.allocate_all_pixels_probabilistic()

    # STEP 2: Aggregate facilities to HSAs
    print("\n" + "-"*40)
    print("STEP 2: Facility → HSA aggregation")
    print("-"*40)

    hsa_summary, facility_assignments = allocator.aggregate_facilities_to_hsas(
        facility_allocations,
        hsa_anchors,
        all_facilities,
        network_type=network_type,
        optimization_mode=optimization_mode,
        max_assignment_distance_km=params.get('max_distance_km', 100.0) if params else 100.0,
        alpha=params.get('alpha', 0.75) if params else 0.75,
        beta=params.get('beta', 1.5) if params else 1.5
    )

    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. HSA modeling dataset (PRIMARY OUTPUT)
    hsa_output = output_dir / f'{network_type}_{optimization_mode}_hsa_populations_probabilistic_{boundary_version}.csv'
    hsa_summary.to_csv(hsa_output, index=False)
    print(f"\n  Saved HSA population summary: {hsa_output.name}")

    # 2. Facility-level allocations from pixels
    fac_alloc_output = output_dir / f'{network_type}_{optimization_mode}_facility_allocations_probabilistic_{boundary_version}.csv'
    facility_allocations.to_csv(fac_alloc_output, index=False)
    print(f"  Saved facility allocations: {fac_alloc_output.name}")

    # 3. Facility assignment details (shows three-case logic)
    fac_assign_output = output_dir / f'{network_type}_{optimization_mode}_facility_hsa_assignments_{boundary_version}.csv'
    facility_assignments.to_csv(fac_assign_output, index=False)
    print(f"  Saved facility-to-HSA assignments: {fac_assign_output.name}")

    # 4. Pixel-level allocations (for downstream compatibility)
    if pixel_allocations is not None:
        pixel_output = output_dir / f'pixel_allocations_{network_type}_{optimization_mode}_{boundary_version}.csv'
        pixel_allocations.to_csv(pixel_output, index=False)
        print(f"  Saved pixel allocations: {pixel_output.name}")

    print("\n" + "="*80)
    print("PROBABILISTIC ALLOCATION COMPLETE")
    print("="*80)

    return {
        'hsa_summary': hsa_summary,
        'facility_allocations': facility_allocations,
        'facility_assignments': facility_assignments,
        'pixel_allocations': pixel_allocations
    }
