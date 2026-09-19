#!/usr/bin/env python3
"""
stage_isolated_run.py — copy the COMPLETE delineation/allocation dependency set
for one network+mode+version from a source output dir into an isolated output
dir, so an isolated pipeline run (e.g. a non-footprint mode whose per-HSA climate
must not collide with footprint's) has every prerequisite present.

This replaces ad-hoc hand-copying, which repeatedly missed files. The file set
is derived from what the weekly/daily pipeline actually reads (delineation is
all-modes and disease-agnostic, so these are safe to reuse across disease foci).

It does NOT copy the per-HSA climate download directory; that is mode-specific
and produced by the GEE notebook into the isolated dir separately.

Usage:
    python stage_isolated_run.py --network INF --hsa-mode fewest \
        --boundary-version v7 --from out --to out_INF_fewest_v7
    # add --check to report without copying
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def required_items(net: str, mode: str, ver: str) -> list[str]:
    """Filenames (relative to an out dir) the weekly/daily pipeline needs, plus
    the boundary-independent facility climate CSV."""
    return [
        f"{net}_{mode}_hsas_{ver}.geojson",                       # delineation
        f"pixel_allocations_{net}_{mode}_{ver}.csv",              # pixel allocation
        f"{net}_{mode}_facility_hsa_assignments_{ver}.csv",       # gravity assignment
        f"{net}_{mode}_allocation_details_{ver}.csv",             # allocation detail
        f"{net}_{mode}_facility_allocations_probabilistic_{ver}.csv",
        f"{net}_{mode}_hsa_populations_probabilistic_{ver}.csv",
        f"{net}_Facilities_Climate_Features_with_clusters.csv",   # facility climate (GEE step A)
        f"{net}_footprint_diagnosis_counts_pivot.csv",            # patient volumes per facility
        # The pivot is network-scoped and mode-agnostic (identical bytes across
        # run directories), but the gravity sensitivity needs it for facility
        # attractiveness and now fails rather than substituting random volumes.
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage a complete isolated pipeline run.")
    ap.add_argument("--network", required=True)
    ap.add_argument("--hsa-mode", required=True)
    ap.add_argument("--boundary-version", default="v7")
    ap.add_argument("--from", dest="src", default="out", help="Source output dir (default: out)")
    ap.add_argument("--to", dest="dst", required=True, help="Isolated output dir to stage into")
    ap.add_argument("--check", action="store_true", help="Report only; do not copy")
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    if not src.is_absolute():
        src = BASE_DIR / src
    if not dst.is_absolute():
        dst = BASE_DIR / dst
    dst.mkdir(parents=True, exist_ok=True)

    items = required_items(args.network, args.hsa_mode, args.boundary_version)
    missing_src, copied, already = [], [], []
    for name in items:
        s, d = src / name, dst / name
        if d.exists():
            already.append(name)
        elif s.exists():
            if not args.check:
                shutil.copy2(s, d)
            copied.append(name)
        else:
            missing_src.append(name)

    verb = "would copy" if args.check else "copied"
    for n in copied:      print(f"  [{verb}] {n}")
    for n in already:     print(f"  [present] {n}")
    for n in missing_src: print(f"  [MISSING in source] {n}")

    # Remind about the climate download dir (mode-specific; GEE-produced)
    clim = dst / f"DRIVE_CLIMATE_BY_HSA_DOWNLOAD_{args.boundary_version.upper()}" / "FINAL_HSA_CLIMATE"
    n_clim = len(list(clim.glob("*.csv"))) if clim.exists() else 0
    print(f"\n  Per-HSA weekly climate in isolated dir: {n_clim} file(s) at {clim}")
    if n_clim == 0:
        print("    Run the weekly GEE notebook (MODE-isolated Drive folder) to populate it.")

    if missing_src:
        print(f"\nstage_isolated_run: {len(missing_src)} required item(s) missing in "
              f"source '{src}'. Run delineation+allocation (--only-steps 1,2) there first.")
        return 1
    print(f"\nstage_isolated_run: complete dependency set present in {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
