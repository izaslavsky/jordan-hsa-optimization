"""
Package a minimal shareable results archive.

Included files:
  out/<network>_<mode>_map[_<version>].gpkg
  out/<network>_<mode>_facility_hsa_assignments[_<version>].csv
  out/modeling/<network>_<mode>_modeling_dataset[_<version>].csv
  out/modeling/<network>_<mode>_daily_modeling_dataset[_<version>].csv
  out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD[_<VERSION>]/FINAL_HSA_CLIMATE/*.csv

Usage:
  python package_results.py --network INF --mode footprint --version v7
  python package_results.py --mode footprint            # no version suffix
  python package_results.py                             # interactive prompts
"""

import argparse
import sys
import zipfile
from pathlib import Path

OUT_DIR = Path(__file__).parent / "out"


def prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else default


def build_paths(network: str, mode: str, version: str) -> tuple[list[Path], list[Path]]:
    """Return (found, missing) lists."""
    ver_lo = f"_{version.lower()}" if version else ""
    ver_hi = f"_{version.upper()}" if version else ""

    modeling_dir = OUT_DIR / "modeling"

    targets = [
        OUT_DIR / f"{network}_{mode}_map{ver_lo}.gpkg",
        OUT_DIR / f"{network}_{mode}_facility_hsa_assignments{ver_lo}.csv",
        modeling_dir / f"{network}_{mode}_modeling_dataset{ver_lo}.csv",
        modeling_dir / f"{network}_{mode}_daily_modeling_dataset{ver_lo}.csv",
    ]

    climate_dir = OUT_DIR / f"DRIVE_CLIMATE_BY_HSA_DOWNLOAD{ver_hi}" / "FINAL_HSA_CLIMATE"
    targets.extend(sorted(climate_dir.glob("*.csv")) if climate_dir.is_dir() else [climate_dir / "*.csv"])

    found, missing = [], []
    for p in targets:
        if "*" in p.name:
            missing.append(p)
        elif p.exists():
            found.append(p)
        else:
            missing.append(p)

    return found, missing


def make_archive(files: list[Path], output: Path) -> None:
    root = OUT_DIR.parent
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(root))
    print(f"\nWrote {output.name}  ({output.stat().st_size / 1_048_576:.1f} MB, {len(files)} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package HSA results for sharing")
    parser.add_argument("--network", help="Network identifier (e.g. INF, NCD, SYNMODINF)", default="")
    parser.add_argument("--mode",    help="Modeling mode (e.g. footprint, distance, fewest)")
    parser.add_argument("--version", help="Version ID without underscore (e.g. v7); omit for unversioned files", default="")
    parser.add_argument("--output",  help="Output zip path (default: hsa_results_<mode>[_<version>].zip)")
    args = parser.parse_args()

    network = args.network or prompt("Network (e.g. INF, NCD, SYNMODINF)", default="INF")
    mode    = args.mode    or prompt("Mode (footprint / distance / fewest)")
    version = args.version if args.version is not None else prompt("Version (e.g. v7, or leave blank for none)")

    if not mode:
        sys.exit("Mode is required.")

    found, missing = build_paths(network, mode, version)

    if missing:
        print("Not found (will be skipped):")
        for p in missing:
            print(f"  {p.relative_to(OUT_DIR.parent)}")

    if not found:
        sys.exit("No files found — check mode and version.")

    print(f"\nFound {len(found)} file(s):")
    for p in found:
        print(f"  {p.relative_to(OUT_DIR.parent)}")

    ver_tag = f"_{version.lower()}" if version else ""
    default_out = Path(__file__).parent / f"hsa_results_{mode}{ver_tag}.zip"
    output = Path(args.output) if args.output else default_out

    make_archive(found, output)


if __name__ == "__main__":
    main()
