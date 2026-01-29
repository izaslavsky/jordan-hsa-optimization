"""
Patient Allocation Module - Gravity Model Implementation
=========================================================

Eliminates double-counting by allocating each population pixel to exactly ONE facility
using a gravity model based on distance and facility size.

Usage:
    allocator = PatientAllocator(pop_raster_path, facilities_gdf, params)
    allocated = allocator.allocate_all_pixels()
    hsa_summary = allocator.aggregate_by_hsa(allocated, hsa_anchors)

Author: HSA Research Team
Date: 2025-11-27
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


class PatientAllocator:
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

        print(f"PatientAllocator initialized:")
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

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self._process_chunk, chunks))

        alloc_dfs = [res[0] for res in results]
        stats_list = [res[1] for res in results]
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


def allocate_patients_for_hsa_mode(hsa_geojson_path: str,
                                   pop_raster_path: str,
                                   network_type: str,
                                   optimization_mode: str,
                                   output_dir: Path,
                                   params: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to allocate patients for one HSA mode

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
    print(f"PATIENT ALLOCATION: {network_type} - {optimization_mode.upper()}")
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
    allocator = PatientAllocator(pop_raster_path, all_facilities, params)

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
    fac_output = output_dir / f'{network_type}_{optimization_mode}_facility_allocations.csv'
    facility_summary.to_csv(fac_output, index=False)
    print(f"  Saved facility summary: {fac_output.name}")

    # 3. Detailed allocations (optional - can be large)
    detail_output = output_dir / f'{network_type}_{optimization_mode}_allocation_details.csv'
    allocations.to_csv(detail_output, index=False)
    print(f"  Saved allocation details: {detail_output.name}")

    print("\n" + "="*80)

    return {
        'hsa_summary': hsa_summary,
        'facility_summary': facility_summary,
        'allocations': allocations
    }
