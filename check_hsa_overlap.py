#!/usr/bin/env python3
"""Executable invariant: no two HSA anchors in a delineation bundle may be
near-duplicates of one another.

The optimizer prunes overlapping HSAs while every anchor still sits where it was
scored, but anchor upgrade and major-facility promotion move and add anchors
afterwards, and an upgraded anchor keeps the radius fitted to the facility it
replaced. This check reads the written geojsons and fails if any ordered anchor
pair exceeds the overlap threshold in either direction.

Overlap is measured on circle area, which needs no population raster and is the
stricter of the two bases for desert-adjacent anchors. Pass --population to
measure on WorldPop cells instead, matching the threshold the pipeline prunes
with.

Usage:
    python check_hsa_overlap.py                      # every *_hsas_v*.geojson in out/
    python check_hsa_overlap.py --out-dir out_test --threshold 0.80
    python check_hsa_overlap.py --population         # population-weighted basis
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

METRIC_CRS = 'EPSG:32637'


def violations(gdf, threshold):
    name_col = 'HealthFacility' if 'HealthFacility' in gdf.columns else 'FacilityName'
    pts = gpd.GeoDataFrame(
        gdf, geometry=gpd.points_from_xy(gdf['lon'], gdf['lat']), crs='EPSG:4326'
    ).to_crs(METRIC_CRS)
    radii = pd.to_numeric(gdf['service_radius_km'], errors='coerce').fillna(0.0).values
    circles = [p.buffer(r * 1000.0) for p, r in zip(pts.geometry, radii)]

    rows = []
    for i, ci in enumerate(circles):
        if ci.area <= 0:
            continue
        for j, cj in enumerate(circles):
            if i == j:
                continue
            frac = ci.intersection(cj).area / ci.area
            if frac > threshold:
                rows.append((str(gdf[name_col].iloc[i]), str(gdf[name_col].iloc[j]),
                             frac, pts.geometry.iloc[i].distance(pts.geometry.iloc[j]) / 1000.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='out')
    ap.add_argument('--threshold', type=float, default=0.80)
    ap.add_argument('--population', action='store_true',
                    help='measure overlap on WorldPop cells instead of circle area')
    ap.add_argument('--version', default=None,
                    help='restrict to one boundary version (e.g. v7). Only v7 is '
                         'built now, so superseded v6/v8 bundles left in the '
                         'directory otherwise report failures that no longer '
                         'correspond to anything the pipeline produces.')
    args = ap.parse_args()

    pattern = f"*_hsas_{args.version}.geojson" if args.version else "*_hsas_v*.geojson"
    files = sorted(Path(args.out_dir).glob(pattern))
    # *_circles.geojson are pre-clipping intermediates, not delineations.
    files = [f for f in files if '_circles' not in f.name]
    if not files:
        print(f"No *_hsas_v*.geojson found in {args.out_dir}", file=sys.stderr)
        return 2

    optimizer = None
    if args.population:
        import hsa_optimization
        from hsa_optimization import HSAOptimizer
        optimizer = HSAOptimizer({'pop_path': 'data/jor_ppp_2020_UNadj.tif', 'coarsen': 4})

    failed = 0
    for path in files:
        gdf = gpd.read_file(path)
        if 'lat' not in gdf.columns or 'service_radius_km' not in gdf.columns:
            print(f"  [skip] {path.name}: no lat/service_radius_km columns")
            continue
        if optimizer is not None:
            import hsa_optimization
            report = hsa_optimization.anchor_overlap_report(
                gdf, overlap_threshold=args.threshold, optimizer=optimizer
            )
            bad = [(r.anchor_a, r.anchor_b, r.overlap_frac, r.distance_km)
                   for r in report.itertuples()]
        else:
            bad = violations(gdf, args.threshold)
        status = 'FAIL' if bad else 'ok'
        print(f"  [{status}] {path.name}: {len(gdf)} anchors, {len(bad)} violation(s)")
        for a, b, frac, dist in bad:
            print(f"        {a} is {frac*100:.1f}% inside {b} ({dist:.2f} km apart)")
        failed += len(bad)

    basis = 'populated cells' if args.population else 'circle area'
    print(f"\ncheck_hsa_overlap ({basis}, threshold {args.threshold:.0%}): "
          f"{'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
