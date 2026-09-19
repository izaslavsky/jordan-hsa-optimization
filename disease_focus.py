#!/usr/bin/env python3
"""
disease_focus.py — single source of truth for disease-group and outcome naming.

No script or notebook should hardcode a disease-group label, the group-column
name, or the outcome-column name. Everything flows through here, reading the
canonical spelling from data/{NETWORK}_groups_of_diagnoses.csv (INF model:
column `General_Diagnosis`, one canonical spelling per group).

Typical use:
    from disease_focus import canonical_group, group_column, OUTCOME_COL
    label = canonical_group("NCD", "hypertension")   # -> "Hypertension"
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

# Candidate names for the group column, in preference order (INF model first).
GROUP_COLUMN_CANDIDATES = ("General_Diagnosis", "General_Category")


def slug(label: str) -> str:
    """Filesystem/column-safe token derived from a canonical group label.
    'Diarrheal Diseases' -> 'diarrheal_diseases'; 'Hypertension' -> 'hypertension'.
    Derived, never hardcoded, so outputs are marked with the resolved value."""
    import re
    s = str(label).strip().lower()
    s = re.sub(r"[^\w]+", "_", s)      # non-word runs -> underscore
    return re.sub(r"_+", "_", s).strip("_")


def weekly_outcome_col(label: str) -> str:
    """Weekly (attendance-adjusted) outcome column name for a focus label."""
    return f"{slug(label)}_count_adjusted"


def daily_outcome_col(label: str) -> str:
    """Daily outcome column name for a focus label."""
    return f"{slug(label)}_count"


def group_column(df: pd.DataFrame) -> str:
    """Return the group-label column present in df, agnostic to INF/NCD naming
    and to case. Raises if none of the known candidates is present."""
    lower = {c.lower(): c for c in df.columns}
    for cand in GROUP_COLUMN_CANDIDATES:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(
        f"No group column found in columns {list(df.columns)}; "
        f"expected one of {GROUP_COLUMN_CANDIDATES}"
    )


def load_groups(network: str, data_dir="data") -> pd.DataFrame:
    """Load the authoritative groups_of_diagnoses table for a network,
    falling back to the SYNMOD synthetic copy when the real file is absent."""
    base = Path(data_dir)
    real = base / f"{network}_groups_of_diagnoses.csv"
    if real.exists():
        return pd.read_csv(real)
    synth = base / f"SYNMOD{network}_groups_of_diagnoses.csv"
    if synth.exists():
        return pd.read_csv(synth)
    raise FileNotFoundError(
        f"No groups_of_diagnoses table for {network} in {base} "
        f"(tried {real.name} and {synth.name})"
    )


def canonical_group(network: str, focus: str, data_dir="data") -> str:
    """Resolve a disease-focus keyword to the canonical group label exactly as
    spelled in the authoritative groups table, matching case-insensitively.
    Exact (case-insensitive) match wins; otherwise a unique substring match is
    accepted (e.g. 'diarrheal' -> 'Diarrheal Diseases')."""
    g = load_groups(network, data_dir)
    col = group_column(g)
    labels = sorted(g[col].dropna().astype(str).unique())
    # Normalize both sides to the slug form so a canonical label, a keyword, or a
    # slug all resolve ('Diarrheal Diseases' == 'diarrheal' == 'diarrheal_diseases').
    fl = slug(focus)

    exact = [lab for lab in labels if slug(lab) == fl]
    if exact:
        return exact[0]

    partial = sorted({lab for lab in labels if fl in slug(lab)})
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ValueError(
            f"Disease focus '{focus}' is ambiguous for {network}; "
            f"matches {partial}. Pass the exact group label."
        )
    raise ValueError(
        f"Disease focus '{focus}' not found among {network} groups: {labels}"
    )


def canonicalize_series(series: pd.Series, network: str, data_dir="data") -> pd.Series:
    """Map a column of raw group labels (e.g. patient_visits.general_category,
    which may carry casing variants) onto the canonical spellings from the
    authoritative table. Unknown values are left unchanged."""
    g = load_groups(network, data_dir)
    col = group_column(g)
    canon = {lab.lower(): lab for lab in g[col].dropna().astype(str).unique()}
    return series.astype(str).map(lambda v: canon.get(v.strip().lower(), v))

