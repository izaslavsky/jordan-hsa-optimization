#!/usr/bin/env python3
"""S4.3 alternative-coverage delineations. Runs HSA_FINAL at TAU=0.95 and 0.99
for INF and NCD in isolated output dirs (never touching the canonical 90% v7
geojsons in out/), then records the v7 HSA count per mode.

Only TAU_COVERAGE, config['tau_coverage'], and NETWORK are changed; every other
delineation parameter matches the canonical run. Usage:
  python run_coverage_sensitivity.py --dry-run     # patch + verify config only
  python run_coverage_sensitivity.py --network INF --cov 0.95
  python run_coverage_sensitivity.py --all         # all 4 combos, sequential
"""
import argparse, json, os, re, shutil, sys, time
from pathlib import Path
import nbformat
try:
    from nbconvert.preprocessors import ExecutePreprocessor
except ModuleNotFoundError:            # nbconvert is optional, as in run_pipeline
    ExecutePreprocessor = None

BASE = Path(__file__).resolve().parent
MODES = ['fewest','footprint','distance','governorate_tau_coverage','governorate_fewest']
INPUTS = ['{net}_Facilities_Climate_Features_with_clusters.csv',
          '{net}_footprint_diagnosis_counts_pivot.csv',
          '{net}_diagnosis_counts_pivot.csv']  # last may not exist; copy if present

def stage(net, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    for tmpl in INPUTS:
        f = tmpl.format(net=net); s = BASE/'out'/f  # noqa: hardcode - reads the canonical delineation masters
        if s.exists(): shutil.copy2(s, outdir/f)

def patch_nb(net, cov):
    nb = nbformat.read(BASE/'HSA_FINAL.ipynb', as_version=4)
    cfg = nb.cells[1]  # config cell
    src = cfg.source
    src = re.sub(r'^NETWORK\s*=\s*.*$', f'NETWORK = "{net}"', src, flags=re.M)
    src = re.sub(r'^TAU_COVERAGE\s*=\s*[0-9.]+', f'TAU_COVERAGE = {cov}', src, flags=re.M)
    src = re.sub(r"('tau_coverage':\s*)[0-9.]+", rf"\g<1>{cov}", src)
    cfg.source = src
    # Keep only through the delineation cell (index 5, writes the geojsons);
    # the downstream mapping/metadata cells are unneeded and memory-heavy.
    nb.cells = nb.cells[:6]
    # Override ALGORITHM_VARIANTS to v7 only (we only read *_v7.geojson); this
    # cuts the delineation work and peak memory by ~3x vs running v6/v7/v8.
    ov = nbformat.v4.new_code_cell(
        'ALGORITHM_VARIANTS = [{"version":"v7",'
        '"label":"Anchor upgrade/demotion + major-orphan promotion",'
        '"UPGRADE_WEAK_SELECTED_ANCHORS":True,'
        '"PROMOTE_MAJOR_UNCOVERED_FACILITIES":True,'
        '"ENABLE_SATELLITE_BUBBLE_HSAS":False}]')
    if nb.get('nbformat_minor',5) < 5: ov.pop('id', None)
    nb.cells.insert(2, ov)  # right after the config cell
    return nb

def verify_patch(nb, net, cov):
    s = nb.cells[1].source
    assert f'NETWORK = "{net}"' in s, "NETWORK not patched"
    assert f'TAU_COVERAGE = {cov}' in s, "TAU_COVERAGE not patched"
    assert f"'tau_coverage': {cov}" in s, "config tau_coverage not patched"
    return True

def run_one(net, cov, execute=True):
    tag = f"{net}_cov{int(round(cov*100))}"
    outdir = BASE/f"out_{tag}"
    nb = patch_nb(net, cov); verify_patch(nb, net, cov)
    if not execute:
        print(f"[dry] {tag}: config patched OK (NETWORK={net}, TAU={cov})")
        return None
    stage(net, outdir)
    # Pin the kernel to the 3.11 env: the 'python3' kernelspec launches bare
    # 'python', which is base 3.13 unless the env bin leads PATH.
    env_bin = str(Path(sys.executable).parent)
    os.environ['PATH'] = env_bin + os.pathsep + os.environ.get('PATH','')
    os.environ['HSA_OUT_DIR']=str(outdir); os.environ['HSA_DATA_DIR']=str(BASE/'data')
    os.environ['PYTHONHASHSEED']='42'
    # Sentinel first cell: prove the kernel is the 3.11 env, not base 3.13.
    probe = nbformat.v4.new_code_cell("import sys; print('KERNEL_PY', sys.version.split()[0], sys.executable)")
    if nb.get('nbformat_minor',5) < 5: probe.pop('id', None)
    nb.cells.insert(0, probe)
    print(f"[run] {tag}: executing HSA_FINAL (isolated {outdir.name}) ...", flush=True)
    t0=time.time()
    exec_nb = BASE/f"HSA_FINAL_{tag}_executed.ipynb"
    if ExecutePreprocessor is not None:
        ep = ExecutePreprocessor(timeout=5400, kernel_name='python3', allow_errors=True)
        try:
            ep.preprocess(nb, {'metadata': {'path': str(BASE)}})
        finally:
            nbformat.write(nb, exec_nb)
            print(f"[nb-saved] {exec_nb.name}", flush=True)
    else:
        # Same fallback run_pipeline uses: `jupyter execute` ships with
        # jupyter_client/nbclient and needs no nbconvert.
        import subprocess
        nbformat.write(nb, exec_nb)
        subprocess.run(["jupyter", "execute", "--timeout=5400", "--inplace", str(exec_nb)],
                       cwd=str(BASE), check=False)
        print(f"[nb-saved] {exec_nb.name}", flush=True)
    import geopandas as gpd
    counts={}
    for m in MODES:
        g = outdir/f"{net}_{m}_hsas_v7.geojson"
        counts[m] = len(gpd.read_file(g)) if g.exists() else None
    print(f"[done] {tag} in {time.time()-t0:.0f}s  counts={counts}", flush=True)
    res_path = BASE/"coverage_sensitivity_results.json"
    allres = json.load(open(res_path)) if res_path.exists() else {}
    allres[tag]=counts; json.dump(allres, open(res_path,'w'), indent=2)
    return counts

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--network'); ap.add_argument('--cov', type=float)
    ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--all', action='store_true')
    a=ap.parse_args()
    if a.dry_run:
        for net in ['INF','NCD']:
            for cov in [0.95,0.99]: run_one(net,cov,execute=False)
    elif a.all:
        for net in ['INF','NCD']:
            for cov in [0.95,0.99]: run_one(net,cov,execute=True)
    else:
        run_one(a.network, a.cov, execute=True)
