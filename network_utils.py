"""Helpers for normalizing network labels across real and synthetic datasets."""

from __future__ import annotations


def canonical_network_family(network: str | None) -> str:
    """Return the disease family represented by a network label."""
    normalized = (network or "").strip().upper()

    if normalized.endswith("INF") or normalized == "SYN":
        return "INF"
    if normalized.endswith("NCD"):
        return "NCD"

    raise ValueError(f"Unsupported network label: {network!r}")


def is_inf_network(network: str | None) -> bool:
    return canonical_network_family(network) == "INF"


def default_disease_focus(network: str | None) -> str:
    return "diarrheal" if is_inf_network(network) else "hypertension"


def secondary_label(network: str | None) -> str:
    return "infectious" if is_inf_network(network) else "ncd"


def default_target_col(network: str | None) -> str:
    return "diarrheal_count_adjusted" if is_inf_network(network) else "hypertension_count_adjusted"
