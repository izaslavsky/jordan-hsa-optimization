"""
Package a minimal shareable results archive for one network + mode + disease.

Included files (all scoped to a single network so INF and NCD never mix):
  out/<network>_<mode>_map[_<version>].gpkg
  out/<network>_<mode>_facility_hsa_assignments[_<version>].csv
  out/modeling/<network>_<mode>_modeling_dataset[_<version>].csv
  out/modeling/<network>_<mode>_daily_modeling_dataset[_<version>].csv
  out/DRIVE_CLIMATE_BY_HSA_DOWNLOAD[_<VERSION>]/FINAL_HSA_CLIMATE/<network>_HSA_*.csv
  manifest.json  (network, mode, version, disease group, file counts)

The shared climate download folder holds both INF_HSA_* and NCD_HSA_* files, so
the climate glob is filtered by the network prefix. Without that filter the
archive leaked the other network's climate data and confused downstream parsers.

The archive name and manifest carry the disease group (resolved from
groups_of_diagnoses via disease_focus) so HEFTE can select and compare diseases.

Usage:
  python package_results.py --network INF --mode footprint --version v7 --disease-focus diarrheal
  python package_results.py --mode footprint            # no version suffix
  python package_results.py                             # interactive prompts
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

# Resolved from the environment rather than hardcoded to "out": each run now
# writes to its own directory, and packaging from "out" silently bundled a
# previous run's files while the fresh ones sat elsewhere.
OUT_DIR = Path(os.environ.get("HSA_OUT_DIR",
                              os.environ.get("PIPELINE_OUT_DIR",
                                             Path(__file__).parent / "out")))  # noqa: hardcode - env fallback
if not OUT_DIR.is_absolute():
    OUT_DIR = Path(__file__).parent / OUT_DIR


def prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else default


def resolve_disease(network: str, focus: str) -> tuple[str, str]:
    """Return (canonical_group_label, slug) for the disease focus, e.g.
    ('Diarrheal Diseases', 'diarrheal_diseases'). Falls back to the raw focus
    string if disease_focus cannot resolve it (e.g. groups table unavailable)."""
    if not focus:
        return "", ""
    try:
        from disease_focus import canonical_group, slug
        label = canonical_group(network, focus)
        return label, slug(label)
    except Exception as exc:  # pragma: no cover - resolver is best-effort here
        print(f"  (disease_focus could not resolve '{focus}': {exc}; using it verbatim)")
        fallback = focus.strip()
        slug_fb = "_".join(fallback.lower().split())
        return fallback, slug_fb


def build_paths(network: str, mode: str, version: str) -> tuple[list[Path], list[Path]]:
    """Return (found, missing) lists. Climate CSVs are scoped to <network>_HSA_*."""
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
    # Filter by network prefix: the folder is shared across INF and NCD.
    climate_glob = f"{network}_HSA_*.csv"
    if climate_dir.is_dir():
        climate_files = sorted(climate_dir.glob(climate_glob))
        targets.extend(climate_files if climate_files else [climate_dir / climate_glob])
    else:
        targets.append(climate_dir / climate_glob)

    found, missing = [], []
    for p in targets:
        if "*" in p.name:
            missing.append(p)
        elif p.exists():
            found.append(p)
        else:
            missing.append(p)

    return found, missing


def make_archive(files: list[Path], output: Path, manifest: dict) -> None:
    root = OUT_DIR.parent
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for f in files:
            zf.write(f, f.relative_to(root))
    print(f"\nWrote {output.name}  ({output.stat().st_size / 1_048_576:.1f} MB, {len(files)} files + manifest.json)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package HSA results for sharing")
    parser.add_argument("--network", help="Network identifier (e.g. INF, NCD, SYNMODINF)", default="")
    parser.add_argument("--mode",    help="Modeling mode (e.g. footprint, distance, fewest)")
    parser.add_argument("--version", help="Version ID without underscore (e.g. v7); omit for unversioned files", default="")
    parser.add_argument("--disease-focus", dest="disease_focus",
                        help="Disease group focus (e.g. diarrheal, hypertension); resolved to the canonical "
                             "groups_of_diagnoses label and embedded in the name and manifest", default="")
    parser.add_argument("--output",  help="Output zip path (default includes network, mode, disease, version)")
    args = parser.parse_args()

    network = args.network or prompt("Network (e.g. INF, NCD, SYNMODINF)", default="INF")
    mode    = args.mode    or prompt("Mode (footprint / distance / fewest)")
    version = args.version if args.version is not None else prompt("Version (e.g. v7, or leave blank for none)")
    focus   = args.disease_focus if args.disease_focus is not None else prompt("Disease focus (e.g. diarrheal)")

    if not mode:
        sys.exit("Mode is required.")

    disease_label, disease_slug = resolve_disease(network, focus)

    found, missing = build_paths(network, mode, version)

    if missing:
        print("Not found (will be skipped):")
        for p in missing:
            print(f"  {p.relative_to(OUT_DIR.parent)}")

    if not found:
        sys.exit("No files found — check network, mode, and version.")

    print(f"\nFound {len(found)} file(s) for {network} / {mode}"
          + (f" / {disease_label}" if disease_label else "") + ":")
    for p in found:
        print(f"  {p.relative_to(OUT_DIR.parent)}")

    ver_tag = f"_{version.lower()}" if version else ""
    disease_tag = f"_{disease_slug}" if disease_slug else ""
    default_out = Path(__file__).parent / f"hsa_results_{network}_{mode}{disease_tag}{ver_tag}.zip"
    output = Path(args.output) if args.output else default_out

    manifest = {
        "network": network,
        "hsa_mode": mode,
        "boundary_version": version,
        "disease_focus_input": focus,
        "disease_group": disease_label,   # canonical groups_of_diagnoses label
        "disease_slug": disease_slug,
        "n_files": len(found),
        "files": [str(p.relative_to(OUT_DIR.parent)) for p in found],
    }

    make_archive(found, output, manifest)


if __name__ == "__main__":
    main()
