#!/usr/bin/env python3
"""
Generate daily diarrheal case counts per HSA.

Input:
  data/{NETWORK}_patient_visits.csv  (or SYNMOD{NETWORK}_patient_visits.csv)
  {OUT_DIR}/{NETWORK}_{HSA_MODE}_facility_hsa_assignments_{BOUNDARY_VERSION}.csv

Output:
  {OUT_DIR}/{NETWORK}_{HSA_MODE}_daily_diarrheal_{BOUNDARY_VERSION}.csv
  Columns: hsa_id, date, diarrheal_count (float, gravity-weighted), is_reporting_gap

Study period dates, NETWORK, and HSA_MODE must be passed by the calling notebook.
Reporting gap dates are read from data/reporting_gaps.csv (daily DLNM pipeline only).
"""

import re
import sys
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR        = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = Path(os.environ.get("HSA_OUT_DIR", os.environ.get("PIPELINE_OUT_DIR", "out")))

_DISEASE_CATEGORY = {
    "INF":    "Diarrheal Diseases",
    "SYNINF": "Diarrheal Diseases",
    "NCD":    "hypertension",
    "SYNNCD": "hypertension",
}


def parse_hsa_weights(s: str) -> dict:
    """Parse 'HSA1:92.2%; HSA2:7.8%' -> {'HSA1': 0.922, 'HSA2': 0.078}."""
    if not isinstance(s, str) or not s.strip():
        return {}
    out = {}
    for part in s.split(";"):
        part = part.strip()
        m = re.match(r"^(.+):([\d.]+)%", part)
        if m:
            out[m.group(1).strip()] = float(m.group(2)) / 100.0
    return out


def normalize_hsa_name(name: str) -> str:
    return name.strip().replace(" ", "_")


def load_reporting_gaps(gaps_path: Path) -> list:
    if not gaps_path.exists():
        print(f"  WARNING: gaps file not found: {gaps_path} -- no dates will be flagged")
        return []
    df = pd.read_csv(gaps_path)
    return df["date"].tolist()


def main():
    parser = argparse.ArgumentParser(description="Generate daily disease counts per HSA")
    parser.add_argument("--network",      required=True,
                        help="Network prefix (INF or NCD)")
    parser.add_argument("--hsa-mode",     required=True,
                        help="HSA mode (e.g. footprint)")
    parser.add_argument("--study-start",  required=True,
                        help="First date of study window (YYYY-MM-DD)")
    parser.add_argument("--study-end",    required=True,
                        help="Last date of study window (YYYY-MM-DD)")
    parser.add_argument("--out-dir",      default=str(DEFAULT_OUT_DIR),
                        help="Pipeline output directory")
    parser.add_argument("--boundary-version", default="v7",
                        help="HSA boundary bundle version (v6, v7, v8). Default: v7.")
    parser.add_argument("--patient-file", default=None,
                        help="Path to patient visits CSV. Defaults to "
                             "data/{NETWORK}_patient_visits.csv (real) or "
                             "data/SYNMOD{NETWORK}_patient_visits.csv (synthetic).")
    parser.add_argument("--gaps-file",    default=None,
                        help="CSV with reporting gap dates (column: date). "
                             "Default: data/reporting_gaps.csv.")
    args = parser.parse_args()

    NETWORK         = args.network
    HSA_MODE        = args.hsa_mode
    CODE_VALID_FROM = args.study_start
    STUDY_END       = args.study_end

    OUT_DIR = Path(args.out_dir)
    if not OUT_DIR.is_absolute():
        OUT_DIR = BASE_DIR / OUT_DIR

    ALLOC_FILE = OUT_DIR / f"{NETWORK}_{HSA_MODE}_facility_hsa_assignments_{args.boundary_version}.csv"
    OUT_FILE   = OUT_DIR / f"{NETWORK}_{HSA_MODE}_daily_diarrheal_{args.boundary_version}.csv"

    if args.patient_file:
        PATIENT_FILE = Path(args.patient_file)
        if not PATIENT_FILE.is_absolute():
            PATIENT_FILE = BASE_DIR / PATIENT_FILE
    else:
        _real = BASE_DIR / "data" / f"{NETWORK}_patient_visits.csv"
        _syn  = BASE_DIR / "data" / f"SYNMOD{NETWORK}_patient_visits.csv"
        PATIENT_FILE = _real if _real.exists() else _syn

    gaps_path = Path(args.gaps_file) if args.gaps_file else BASE_DIR / "data" / "reporting_gaps.csv"
    if not gaps_path.is_absolute():
        gaps_path = BASE_DIR / gaps_path
    REPORTING_GAPS = load_reporting_gaps(gaps_path)

    disease_cat = _DISEASE_CATEGORY.get(NETWORK.upper(), "Diarrheal Diseases")

    print("=" * 60)
    print(f"DAILY DISEASE COUNTS -- {NETWORK} {HSA_MODE}")
    print("=" * 60)
    print(f"  Study window:     {CODE_VALID_FROM} to {STUDY_END}")
    print(f"  Disease category: {disease_cat}")
    print(f"  Patient file:     {PATIENT_FILE}")

    # 1. Load allocations
    print("\n[1/4] Loading facility->HSA allocations...")
    alloc = pd.read_csv(ALLOC_FILE)
    alloc = alloc[~alloc["excluded"]].copy()

    facility_weights: dict[str, dict[str, float]] = {}
    for _, row in alloc.iterrows():
        fid = row["facility_id"]
        weights = parse_hsa_weights(str(row.get("all_containing_hsas", "")))
        if not weights:
            weights = {row["primary_hsa"]: 1.0}
        facility_weights[fid] = {normalize_hsa_name(h): w for h, w in weights.items()}

    print(f"  {len(facility_weights)} facilities mapped")

    # 2. Load patient visits
    print("\n[2/4] Loading and filtering patient visits...")
    pv = pd.read_csv(PATIENT_FILE, low_memory=False)
    pv["date"] = pd.to_datetime(pv["datetimediagnosisentered"], errors="coerce").dt.date
    pv = pv.dropna(subset=["date"])

    visits = pv[
        (pv["general_category"] == disease_cat)
        & (pv["date"] >= pd.to_datetime(CODE_VALID_FROM).date())
        & (pv["date"] <= pd.to_datetime(STUDY_END).date())
    ].copy()

    print(f"  {len(visits):,} visits in study window")

    # 3. Expand each visit to weighted HSA-day records
    print("\n[3/4] Applying gravity-model weights...")
    records = []
    unmatched_facilities = set()
    for _, row in visits.iterrows():
        fac = row["healthfacility"]
        weights = facility_weights.get(fac)
        if weights is None:
            unmatched_facilities.add(fac)
            continue
        for hsa, w in weights.items():
            records.append({"hsa_id": hsa, "date": str(row["date"]), "weight": w})

    if unmatched_facilities:
        print(f"  WARNING: {len(unmatched_facilities)} facilities not in allocation table")
        print(f"    Examples: {sorted(unmatched_facilities)[:5]}")

    weighted = pd.DataFrame(records)
    print(f"  {len(weighted):,} weighted visit-HSA records")

    # 4. Aggregate to HSA-day
    print("\n[4/4] Aggregating to HSA-day...")
    daily = (
        weighted.groupby(["hsa_id", "date"])["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "diarrheal_count"})
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    # Build complete grid: all HSAs x all study days -> fill 0 where no cases
    all_hsas  = sorted(daily["hsa_id"].unique())
    all_dates = pd.date_range(CODE_VALID_FROM, STUDY_END).date
    grid = pd.MultiIndex.from_product([all_hsas, all_dates], names=["hsa_id", "date"])
    daily = (
        daily.set_index(["hsa_id", "date"])
        .reindex(grid, fill_value=0)
        .reset_index()
    )

    # Flag reporting gaps as NaN (not true disease-free days)
    if REPORTING_GAPS:
        gap_set  = set(pd.to_datetime(REPORTING_GAPS).date)
        gap_mask = daily["date"].isin(gap_set)
        daily.loc[gap_mask, "diarrheal_count"] = np.nan
        daily["is_reporting_gap"] = gap_mask.astype(int)
        print(f"  Flagged {gap_mask.sum()} HSA-day rows as reporting gaps")
    else:
        daily["is_reporting_gap"] = 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_FILE, index=False)
    print(f"  Saved: {OUT_FILE}")
    print(f"  Shape: {daily.shape}")
    print(f"  HSAs: {daily['hsa_id'].nunique()}")
    print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"  Total weighted cases: {daily['diarrheal_count'].sum():.0f}")
    print(f"  Non-zero days: {(daily['diarrheal_count'] > 0).sum():,}")

    print("\n  Top HSAs by total cases:")
    top = daily.groupby("hsa_id")["diarrheal_count"].sum().nlargest(5)
    for hsa, n in top.items():
        print(f"    {hsa}: {n:.0f}")


if __name__ == "__main__":
    main()
