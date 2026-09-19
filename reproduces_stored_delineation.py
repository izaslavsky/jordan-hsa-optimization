"""
Reproduction check for anything that re-runs the delineation.

Re-runs the baseline selection and asserts the resulting anchor set is identical
to the one already stored for that run. Anchor *count* is not enough: equal
counts with different membership is the failure this is meant to catch, and is
what a coverage figure of 90.925% against a stored 90.668% turned out to mean.

Run this before trusting any new analysis that re-optimizes. It found that
passing a copy of the facility frame to optimize() left initial_radius_km unset,
so the coverage repair sized candidates with the rural default and picked
AL-Nadeem (252,802) over Arish (231,037).

Usage:
    python reproduces_stored_delineation.py INF out_INF_footprint_v7
    python reproduces_stored_delineation.py NCD out_NCD_footprint_v7
"""
import os, sys, warnings
# Resolve the repo from this file's own location: an absolute path here works
# on exactly one machine and silently breaks the check everywhere else.
_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
warnings.filterwarnings('ignore')
import importlib.util, copy, geopandas as gpd
spec = importlib.util.spec_from_file_location('ws', 'weight_sensitivity_reoptimize.py')
ws = importlib.util.module_from_spec(spec); spec.loader.exec_module(ws)
import hsa_optimization as H
from hsa_optimization import HSAOptimizer

net, mode = sys.argv[1], 'footprint'
run = sys.argv[2]
fac = ws.load_facilities(__import__('pathlib').Path(run), net)
sel, stats = ws.run_one(H, HSAOptimizer, fac, net, mode,
                        copy.deepcopy(ws.PROFILES[mode]), 0.90,
                        'data/jor_ppp_2020_UNadj.tif')
got = sorted(str(x).strip() for x in sel['HealthFacility'])
stored = gpd.read_file(f'{run}/{net}_{mode}_hsas_v7.geojson')
want = sorted(str(x).strip() for x in stored['FacilityName'])
print(f'{net}: re-optimized {len(got)} anchors @ {stats["coverage_pct_final"]:.3f}%')
print(f'{net}: stored       {len(want)} anchors')
print(f'{net}: IDENTICAL    {got == want}')
if got != want:
    sys.exit(1)
if got != want:
    print('  only re-optimized:', sorted(set(got) - set(want)))
    print('  only stored      :', sorted(set(want) - set(got)))
