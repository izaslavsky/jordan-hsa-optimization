"""
Patient Allocation Module - Gravity Model Implementation
=========================================================

Eliminates double-counting by allocating each population pixel to exactly ONE facility
using a gravity model based on distance and facility size.

Usage:
    allocator = PatientAllocator(pop_raster_path, facilities_gdf, params)
    allocated = allocator.allocate_all_pixels()
    hsa_summary = allocator.aggregate_by_hsa(allocated, hsa_anchors)

Author: Claude Code
Date: 2025-11-27
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
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
                - sample_rate: Sample every Nth pixel (default 1 = all pixels)
        """
        self.pop_raster_path = Path(pop_raster_path)
        self.facilities = facilities_gdf.copy()

        # Default parameters
        default_params = {
            'alpha': 0.75,
            'beta': 1.5,
            'max_distance_km': 100.0,
            'sample_rate': 1  # Process every pixel
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
                        all_facilities: gpd.GeoDataFrame,
                        network_type: str,
                        optimization_mode: str) -> pd.DataFrame:
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

        # Ensure both are in same CRS (WGS84)
        if hsa_anchors.crs and hsa_anchors.crs.to_epsg() != 4326:
            hsa_anchors = hsa_anchors.to_crs(epsg=4326)
        if all_facilities.crs and all_facilities.crs.to_epsg() != 4326:
            all_facilities = all_facilities.to_crs(epsg=4326)

        # For each HSA anchor, find all facilities within its service radius
        hsa_results = []

        print(f"\nAggregating by HSA catchment areas...")
        for idx, hsa in hsa_anchors.iterrows():
            anchor_name = hsa['HealthFacility']
            anchor_point = hsa.geometry
            service_radius_km = hsa['service_radius_km']

            # Calculate distances from anchor to all facilities (haversine)
            facility_distances = all_facilities.geometry.apply(
                lambda fac_geom: GeoUtils.haversine_km(
                    anchor_point.x, anchor_point.y,
                    fac_geom.x, fac_geom.y
                )
            )

            # Find facilities within service radius
            facilities_in_hsa = all_facilities[facility_distances <= service_radius_km]['HealthFacility'].values

            # Sum allocated population for all facilities in this HSA
            hsa_allocation = allocations_df[allocations_df['facility_id'].isin(facilities_in_hsa)]
            total_allocated = hsa_allocation['population'].sum()

            hsa_results.append({
                'anchor_name': anchor_name,
                'allocated_patients': total_allocated,
                'num_facilities_in_hsa': len(facilities_in_hsa),
                'facilities_in_hsa': ', '.join(facilities_in_hsa[:5])  # First 5 for reference
            })

            print(f"  {anchor_name}: {len(facilities_in_hsa)} facilities, {total_allocated:,.0f} people")

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
