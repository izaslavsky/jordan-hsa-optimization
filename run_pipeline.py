#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end pipeline runner.

Executes all local-computation notebooks in sequence for a single
network/mode/boundary/disease combination. GEE extraction notebooks require
human interaction (Earth Engine auth, Drive polling) and cannot be automated
here; the script checks for their outputs and skips steps whose prereqs are
missing.

Requirements:
    pip install nbformat nbconvert   # nbconvert optional; falls back to `jupyter execute`

---------------------------------------------------------------------
CORRECT EXECUTION ORDER (two-phase workflow)
---------------------------------------------------------------------

Phase 1 — run BEFORE HSA boundaries exist:
    GEE Step A  GEE_local_Climate_Features_by_Facilities.ipynb
                Extracts climate at each *facility* location.
                Needs only facility coordinates (no HSA boundaries).
                Prereq for Step 1 below.

    python run_pipeline.py ... --only-steps 1,2
                Step 1: delineates HSA boundaries (v6/v7/v8).
                Step 2: allocates population to HSAs.

Phase 2 — run AFTER HSA boundaries exist (output of Step 1):
    GEE Step B  GEE_local_HSA_Weekly_Climate_Lagged.ipynb
                Aggregates weekly climate per HSA polygon.
                Prereq for Step 3.
    GEE Step C  GEE_local_HSA_Daily_Climate.ipynb
                Aggregates daily climate per HSA polygon.
                Prereq for Step 5.

    python run_pipeline.py ... --only-steps 3,4,5
                Steps 3–5: build weekly and daily panels and run the
                weekly models.

    python run_dlnm_primary_sensitivity.py
                Runs the publication daily DLNM after Step 5 succeeds.

If you run the script without --only-steps, it will execute steps 1
and 2, then skip 3 and 5 (GEE outputs missing) and run step 4 with
whatever data is already present — which is usually not what you want.
---------------------------------------------------------------------

Usage:
    # Phase 1
    python run_pipeline.py \\
        --network SYNMODINF --hsa-mode footprint \\
        --boundary-version v7 --disease-focus diarrheal \\
        --study-start 2022-07-01 --study-end 2024-01-31 \\
        --week-start 2019-01-07 --week-end 2024-01-29 \\
        --ml-start-date 2022-06-27 --ml-end-date 2024-01-29 \\
        --only-steps 1,2

    # (run GEE Steps B and C here)

    # Phase 2
    python run_pipeline.py \\
        --network SYNMODINF --hsa-mode footprint \\
        --boundary-version v7 --disease-focus diarrheal \\
        --study-start 2022-07-01 --study-end 2024-01-31 \\
        --week-start 2019-01-07 --week-end 2024-01-29 \\
        --ml-start-date 2022-06-27 --ml-end-date 2024-01-29 \\
        --only-steps 3,4,5

    python run_dlnm_primary_sensitivity.py
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import nbformat
except ImportError:
    print("ERROR: nbformat is required.  pip install nbformat nbconvert", file=sys.stderr)
    sys.exit(1)

try:
    from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
    _USE_NBCONVERT_API = True
except ImportError:
    _USE_NBCONVERT_API = False

BASE_DIR = Path(__file__).resolve().parent

# Fixed seed passed to every notebook kernel via PYTHONHASHSEED (controls Python
# hash randomization, which is the source of non-determinism in the greedy optimizer)
# and via numpy/random seeds injected as a prepended cell.
RANDOM_SEED = 42

# ── Pre-flight import check ────────────────────────────────────────────────────
# All third-party packages required by the 6 pipeline notebooks and the local
# .py modules they import. Checked once at startup so a missing package fails
# immediately rather than after 20+ minutes of notebook execution.
_REQUIRED_PACKAGES = [
    "adjustText",
    "affine",
    "dlnm",
    "geopandas",
    "matplotlib",
    "numpy",
    "pandas",
    "rasterio",
    "scipy",
    "seaborn",
    "shapely",
    "statsmodels",
    "tqdm",
]

def _preflight_check() -> None:
    missing = []
    for pkg in _REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("ERROR: missing Python packages (install before running the pipeline):", file=sys.stderr)
        for pkg in missing:
            print(f"  pip install {pkg}", file=sys.stderr)
        sys.exit(1)

# ── Step registry ──────────────────────────────────────────────────────────────
#
# gee_prereq: key in _gee_outputs_exist() that must be True to run the step.
# params:     variable names that will be injected into this notebook's cells.
#
STEPS = [
    {
        "num":       1,
        "name":      "HSA delineation",
        "notebook":  "HSA_FINAL.ipynb",
        "desc":      "Delineates v6/v7/v8 boundaries; produces all three in one run",
        "gee_prereq": "climate_features",
        "params":    ["NETWORK", "HSA_MODE"],
    },
    {
        "num":       2,
        "name":      "Population allocation",
        "notebook":  "Population_Allocation_Probabilistic_v2.ipynb",
        "desc":      "Gravity-model population assignment for the chosen BOUNDARY_VERSION",
        "params":    ["NETWORK", "HSA_MODE", "BOUNDARY_VERSION"],
    },
    {
        "num":       3,
        "name":      "Weekly modeling dataset",
        "notebook":  "Generate_Modeling_Dataset.ipynb",
        "desc":      "Weekly climate-health panel (calls generate_weekly/prepare_ml scripts)",
        "gee_prereq": "weekly_climate",
        "params":    ["NETWORK", "HSA_MODE", "BOUNDARY_VERSION", "DISEASE_FOCUS",
                      "WEEK_START", "WEEK_END", "ML_START_DATE", "ML_END_DATE"],
    },
    {
        "num":       4,
        "name":      "Weekly climate-health models",
        "notebook":  "run_climate_health_modeling.ipynb",
        "desc":      "Comprehensive + parsimonious + ML models for the weekly panel",
        "params":    ["NETWORK", "HSA_MODE", "BOUNDARY_VERSION", "DISEASE_FOCUS"],
    },
    {
        "num":       5,
        "name":      "Daily modeling dataset",
        "notebook":  "Generate_Daily_Modeling_Dataset.ipynb",
        "desc":      "Daily climate-health panel with 14-day lag matrix",
        "gee_prereq": "daily_climate",
        "params":    ["NETWORK", "HSA_MODE", "BOUNDARY_VERSION", "STUDY_START", "STUDY_END"],
    },
]


# ── GEE prerequisite detection ────────────────────────────────────────────────

def _gee_outputs_exist(out_dir: Path, network: str, boundary_version: str) -> dict:
    """Return dict of {prereq_key: bool} indicating which GEE outputs are ready."""
    ver = boundary_version.upper()

    # Climate features CSV — produced by GEE_local_Climate_Features_by_Facilities
    cf_candidates = list(out_dir.glob(f"{network}_Facilities_Climate_Features*.csv"))
    if not cf_candidates:
        cf_candidates = list((BASE_DIR / "out").glob(f"{network}_Facilities_Climate_Features*.csv"))

    # Weekly climate dir — produced by GEE_local_HSA_Weekly_Climate_Lagged
    weekly_dir = out_dir / f"DRIVE_CLIMATE_BY_HSA_DOWNLOAD_{ver}" / "FINAL_HSA_CLIMATE"

    # Daily climate dir — produced by GEE_local_HSA_Daily_Climate
    daily_dir = out_dir / f"DRIVE_CLIMATE_BY_HSA_DOWNLOAD_DAILY_{ver}"

    return {
        "climate_features": bool(cf_candidates),
        "weekly_climate":   weekly_dir.exists() and bool(list(weekly_dir.glob("*.csv"))),
        "daily_climate":    daily_dir.exists() and bool(list(daily_dir.rglob("*.csv"))),
    }


# ── Parameter injection ───────────────────────────────────────────────────────

def _patch_notebook(nb, params: dict):
    """
    In every code cell, replace simple variable assignments for any key in params.
    Matches lines of the form:  VAR = anything   (start of line, MULTILINE).
    Replaces the entire RHS with a quoted string value.
    Does not touch variable *uses* (if/print/etc.) because those don't start
    with the variable name followed by whitespace and '='.
    """
    pattern_cache = {
        var: re.compile(rf'^({re.escape(var)}\s*=\s*).*$', re.MULTILINE)
        for var in params
    }
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        src = cell.source
        for var, val in params.items():
            src = pattern_cache[var].sub(
                lambda m, v=val: m.group(1) + f'"{v}"',
                src,
            )
        cell.source = src
    return nb


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _ticker(stop: threading.Event, nb_name: str, interval: int = 30) -> None:
    """Print a heartbeat line every `interval` seconds until stop is set."""
    t0 = time.time()
    while not stop.wait(interval):
        print(f"  ... still running {nb_name}  [{_fmt_elapsed(time.time() - t0)} elapsed]",
              flush=True)


def _seed_cell(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Return a seed cell whose format matches the target notebook.

    new_code_cell() adds an 'id' field (nbformat 4.5+). Notebooks saved in
    the older 4.4 format don't allow it, which causes a harmless but noisy
    validation warning from NbClientApp. Strip 'id' when the notebook
    doesn't use cell IDs.
    """
    cell = nbformat.v4.new_code_cell(
        f"import random, os\n"
        f"import numpy as np\n"
        f"random.seed({RANDOM_SEED})\n"
        f"np.random.seed({RANDOM_SEED})\n"
        f"# PYTHONHASHSEED={RANDOM_SEED} is set in the kernel environment by run_pipeline.py"
    )
    if nb.get("nbformat_minor", 5) < 5:
        cell.pop("id", None)
    return cell


def _run_notebook(nb_path: Path, params: dict, run_dir: Path, timeout: int) -> tuple[bool, float]:
    """
    Patch params into a copy of the notebook, execute it, and save the output.
    Returns (success, elapsed_seconds).
    """
    nb = nbformat.read(nb_path, as_version=4)
    nb = _patch_notebook(nb, params)
    # Prepend seed cell so numpy/random are fixed inside the kernel too.
    nb.cells.insert(0, _seed_cell(nb))
    out_nb = run_dir / f"{nb_path.stem}_executed.ipynb"

    # PYTHONHASHSEED must be set before the interpreter starts, so pass it
    # through the kernel's environment rather than setting it inside the notebook.
    kernel_env = {**os.environ, "PYTHONHASHSEED": str(RANDOM_SEED)}

    stop = threading.Event()
    ticker = threading.Thread(target=_ticker, args=(stop, nb_path.name), daemon=True)
    t0 = time.time()
    ticker.start()

    try:
        if _USE_NBCONVERT_API:
            ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
            # Temporarily propagate PYTHONHASHSEED so the kernel subprocess inherits it.
            old = os.environ.get("PYTHONHASHSEED")
            os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
            try:
                ep.preprocess(nb, {"metadata": {"path": str(BASE_DIR)}})
                success = True
            except CellExecutionError:
                success = False
            except Exception as exc:
                print(f"  Execution error: {exc}")
                success = False
            finally:
                if old is None:
                    os.environ.pop("PYTHONHASHSEED", None)
                else:
                    os.environ["PYTHONHASHSEED"] = old
            nbformat.write(nb, out_nb)
        else:
            # Fallback: jupyter execute (ships with jupyter_client/nbclient, no nbconvert needed).
            # Write tmp_in beside the project root so the kernel's cwd is BASE_DIR and
            # relative paths like DATA_DIR = Path("data") resolve correctly.
            tmp_in = BASE_DIR / f"_in_{nb_path.name}"
            nbformat.write(nb, tmp_in)
            result = subprocess.run(
                ["jupyter", "execute", f"--timeout={timeout}", "--inplace", str(tmp_in)],
                cwd=BASE_DIR,
                env=kernel_env,
            )
            if result.returncode == 0:
                tmp_in.rename(out_nb)
            else:
                tmp_in.unlink(missing_ok=True)
            success = result.returncode == 0
    finally:
        stop.set()
        ticker.join()

    return success, time.time() - t0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _preflight_check()

    parser = argparse.ArgumentParser(
        description="Run the HSA climate-health pipeline end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--network",          default="SYNMODINF",
                        help="Network prefix, e.g. SYNMODINF or SYNMODNCD "
                             "(default: SYNMODINF)")
    parser.add_argument("--hsa-mode",         default="footprint",
                        help="HSA mode, e.g. footprint  (default: footprint)")
    parser.add_argument("--boundary-version", default="v7",
                        help="Boundary bundle: v6, v7, or v8  (default: v7)")
    parser.add_argument("--disease-focus",    default=None,
                        help="Disease target: diarrheal or hypertension. "
                             "Derived from --network if not set.")
    parser.add_argument("--study-start",      default="2022-07-01",
                        help="Daily DLNM study window start (default: 2022-07-01). "
                             "First day after the June 2022 HMIS reporting gap.")
    parser.add_argument("--study-end",        default="2024-01-31",
                        help="Daily DLNM study window end   (default: 2024-01-31).")
    parser.add_argument("--week-start",       default="2019-01-07",
                        help="Weekly disease count aggregation start (default: 2019-01-07). "
                             "Full HMIS history; earlier than the ML window.")
    parser.add_argument("--week-end",         default="2024-01-29",
                        help="Weekly disease count aggregation end   (default: 2024-01-29).")
    parser.add_argument("--ml-start-date",    default="2022-06-27",
                        help="Weekly ML modeling window start (default: 2022-06-27). "
                             "Monday of the first week where GEE climate data is available. "
                             "Four days earlier than --study-start due to ISO week alignment.")
    parser.add_argument("--ml-end-date",      default="2024-01-29",
                        help="Weekly ML modeling window end   (default: 2024-01-29).")
    parser.add_argument("--out-dir",          default="out",
                        help="Pipeline output directory (default: out)")
    parser.add_argument("--skip-steps",       default="",
                        help="Comma-separated step numbers to skip, e.g. 1,2")
    parser.add_argument("--only-steps",       default="",
                        help="Comma-separated step numbers to run, e.g. 3,4")
    parser.add_argument("--timeout",          type=int, default=7200,
                        help="Per-notebook execution timeout in seconds (default: 7200)")
    args = parser.parse_args()

    NETWORK          = args.network
    HSA_MODE         = args.hsa_mode
    BOUNDARY_VERSION = args.boundary_version
    DISEASE_FOCUS    = (
        args.disease_focus
        or ("diarrheal" if NETWORK.upper().endswith("INF") else "hypertension")
    )
    OUT_DIR = Path(args.out_dir)
    if not OUT_DIR.is_absolute():
        OUT_DIR = BASE_DIR / OUT_DIR

    params = {
        "NETWORK":          NETWORK,
        "HSA_MODE":         HSA_MODE,
        "BOUNDARY_VERSION": BOUNDARY_VERSION,
        "DISEASE_FOCUS":    DISEASE_FOCUS,
        "STUDY_START":      args.study_start,
        "STUDY_END":        args.study_end,
        "WEEK_START":       args.week_start,
        "WEEK_END":         args.week_end,
        "ML_START_DATE":    args.ml_start_date,
        "ML_END_DATE":      args.ml_end_date,
    }

    skip = {int(x) for x in args.skip_steps.split(",") if x.strip().isdigit()}
    only = {int(x) for x in args.only_steps.split(",") if x.strip().isdigit()}

    run_dir = BASE_DIR / "_pipeline_runs" / f"{NETWORK}_{HSA_MODE}_{BOUNDARY_VERSION}"
    run_dir.mkdir(parents=True, exist_ok=True)

    gee = _gee_outputs_exist(OUT_DIR, NETWORK, BOUNDARY_VERSION)

    print("=" * 70)
    print("HSA PIPELINE RUNNER")
    print("=" * 70)
    print(f"  Network:          {NETWORK}")
    print(f"  HSA mode:         {HSA_MODE}")
    print(f"  Boundary version: {BOUNDARY_VERSION}")
    print(f"  Disease focus:    {DISEASE_FOCUS}")
    print(f"  Study window:     {args.study_start} to {args.study_end}  (daily)")
    print(f"  Weekly range:     {args.week_start} to {args.week_end}")
    print(f"  ML date range:    {args.ml_start_date} to {args.ml_end_date}")
    print(f"  Output dir:       {OUT_DIR}")
    print(f"  Run artifacts:    {run_dir}")
    print()
    print("  GEE prerequisites:")
    for key, ok in gee.items():
        status = "OK" if ok else "MISSING — run the corresponding GEE notebook first"
        print(f"    {'[OK]' if ok else '[--]'} {key}: {status}")
    print()

    failed   = []
    skipped  = []
    pipeline_t0 = time.time()

    for step in STEPS:
        num = step["num"]

        if only and num not in only:
            continue
        if num in skip:
            print(f"  [SKIP] Step {num}: {step['name']}")
            skipped.append(num)
            continue

        nb_path = BASE_DIR / step["notebook"]
        if not nb_path.exists():
            print(f"  [SKIP] Step {num}: {step['name']} — {step['notebook']} not found")
            skipped.append(num)
            continue

        prereq = step.get("gee_prereq")
        if prereq and not gee.get(prereq, False):
            print(f"\n  [SKIP] Step {num}: {step['name']}")
            print(f"         GEE output '{prereq}' not found.")
            print(f"         Run the GEE notebook for this step, then re-run the pipeline.")
            skipped.append(num)
            continue

        step_params = {k: params[k] for k in step.get("params", []) if k in params}

        start_ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'─' * 70}")
        print(f"  [RUN ] Step {num}/{len(STEPS)}: {step['name']}  ({start_ts})")
        print(f"         {step['desc']}")
        print(f"         Notebook: {step['notebook']}")
        print(f"         Params:   {step_params}")
        print(f"{'─' * 70}")

        ok, elapsed = _run_notebook(nb_path, step_params, run_dir, timeout=args.timeout)
        end_ts = datetime.now().strftime("%H:%M:%S")

        if ok:
            print(f"  [OK] Step {num} finished in {_fmt_elapsed(elapsed)}  ({end_ts})")
        else:
            out_nb = run_dir / f"{nb_path.stem}_executed.ipynb"
            print(f"  [FAIL] Step {num} failed after {_fmt_elapsed(elapsed)}  ({end_ts})")
            print(f"         Inspect the executed notebook for tracebacks: {out_nb}")
            failed.append(num)
            print()
            print("  Stopping pipeline on first failure.")
            break

    total_elapsed = time.time() - pipeline_t0

    print()
    print("=" * 70)
    if skipped:
        print(f"  Skipped:  steps {skipped}")
    if failed:
        print(f"  FAILED:   steps {failed}  (total time: {_fmt_elapsed(total_elapsed)})")
        print()
        sys.exit(1)
    else:
        completed = [s["num"] for s in STEPS
                     if (not only or s["num"] in only)
                     and s["num"] not in skip
                     and s["num"] not in skipped
                     and s["num"] not in failed]
        print(f"  Completed: steps {completed}  (total time: {_fmt_elapsed(total_elapsed)})")
        print()
        print(f"  Executed notebooks saved to: {run_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
