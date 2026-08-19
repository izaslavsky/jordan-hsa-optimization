"""
Generate data/hsa_metadata.csv from facility coordinates and public JMP data.

Reads the facility coordinates CSV (any SYNMOD* or real network file) and joins
each facility to governorate-level JMP 2025 sanitation values stored in
data/jmp_2025_jordan_governorate.csv. Covers all facilities so the file stays
valid regardless of which facilities become HSA anchors in v6/v7/v8.

Sources:
  JMP values  — WHO/UNICEF JMP 2025, Jordan (washdata.org), 2022 indicator,
                 urban/rural weighted by 2015 Jordan DoS Census urbanization rate;
                 see data/jmp_2025_jordan_governorate.csv for per-governorate values
  Urban bonus — derived from Jordan DoS 2015 governorate urbanization tiers;
                 stored in jmp_2025_jordan_governorate.csv

Usage:
    python generate_hsa_metadata.py [--network INF] [--out data/hsa_metadata.csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

FAC_TYPE_SCORE = {
    "Primary Center":             1,
    "Comprehensive Center":       2,
    "Hospital":                   3,
    "Educational Hospital":       3,
    "Specialized Medical Center": 4,
}

NOTE_INFRA = (
    "JMP 2025 Jordan; 2022 urban/rural safely managed sanitation weighted by "
    "governorate urbanization rate (Jordan DoS Census 2015)"
)


def load_jmp_table(data_dir: Path) -> pd.DataFrame:
    jmp_path = data_dir / "jmp_2025_jordan_governorate.csv"
    if not jmp_path.exists():
        sys.exit(f"ERROR: JMP reference file not found: {jmp_path}")
    jmp = pd.read_csv(jmp_path)
    required = {"governorate", "jmp_san_pct_2022", "jmp_wat_pct_2022", "urban_bonus"}
    missing = required - set(jmp.columns)
    if missing:
        sys.exit(f"ERROR: jmp_2025_jordan_governorate.csv missing columns: {missing}")
    return jmp.set_index("governorate")


def build_metadata(coords_path: Path, jmp: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(coords_path, encoding="utf-8-sig")

    required = {"healthfacility", "healthfacilitytype", "governorate"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: coordinates file missing columns: {missing}")

    rows = []
    unknown_govs = set()
    unknown_types = set()

    for _, row in df.iterrows():
        gov = row["governorate"]
        fac_type = row["healthfacilitytype"]
        hsa_id = str(row["healthfacility"]).strip().replace(" ", "_")

        if gov in jmp.index:
            jmp_row = jmp.loc[gov]
            san = float(jmp_row["jmp_san_pct_2022"])
            wat = float(jmp_row["jmp_wat_pct_2022"])
            urban_bonus = float(jmp_row["urban_bonus"])
        else:
            unknown_govs.add(gov)
            san = wat = float("nan")
            urban_bonus = 0.0

        score = FAC_TYPE_SCORE.get(fac_type)
        if score is None:
            unknown_types.add(fac_type)
            score = 1

        infra_quality = round(san / 100, 5) if pd.notna(san) else float("nan")

        rows.append({
            "hsa_id":              hsa_id,
            "fac_type":            fac_type,
            "governorate":         gov,
            "fac_type_score":      score,
            "urban_bonus":         urban_bonus,
            "infra_quality":       infra_quality,
            "note_infra":          NOTE_INFRA,
            "jmp_san_pct":         san,
            "jmp_wat_pct":         wat,
            "infra_quality_label": f"{san:.1f}% safely managed sanitation"
                                   if pd.notna(san) else "unknown",
        })

    if unknown_govs:
        print(
            f"WARNING: no JMP data for governorate(s): {unknown_govs}. "
            "Add them to data/jmp_2025_jordan_governorate.csv.",
            file=sys.stderr,
        )
    if unknown_types:
        print(
            f"WARNING: unknown facility type(s): {unknown_types}. "
            "Defaulting fac_type_score=1.",
            file=sys.stderr,
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network", default="INF",
        help="Network prefix (INF or NCD). Locates {network}_facility_coordinates.csv "
             "or SYNMOD{network}_facility_coordinates.csv",
    )
    parser.add_argument(
        "--coords", default=None,
        help="Explicit path to facility coordinates CSV (overrides --network)",
    )
    parser.add_argument(
        "--out", default="data/hsa_metadata.csv",
        help="Output path (default: data/hsa_metadata.csv)",
    )
    args = parser.parse_args()

    data_dir = Path("data")
    jmp = load_jmp_table(data_dir)

    if args.coords:
        coords_path = Path(args.coords)
    else:
        real_path = data_dir / f"{args.network}_facility_coordinates.csv"
        syn_path  = data_dir / f"SYNMOD{args.network}_facility_coordinates.csv"
        coords_path = real_path if real_path.exists() else syn_path

    if not coords_path.exists():
        sys.exit(f"ERROR: coordinates file not found: {coords_path}")

    print(f"Reading:  {coords_path}")
    print(f"JMP ref:  {data_dir / 'jmp_2025_jordan_governorate.csv'}")
    meta = build_metadata(coords_path, jmp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_path, index=False)
    print(f"Wrote {len(meta)} rows to {out_path}")


if __name__ == "__main__":
    main()
