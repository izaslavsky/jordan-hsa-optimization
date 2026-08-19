#!/usr/bin/env python3
"""Run the prespecified daily DLNM primary and sensitivity analyses.

Primary analysis
    All 19 HSAs in the v7 daily modeling dataset.

Sensitivity analysis
    HSAs with a mean of at least one diarrheal visit per day.

Both analyses use the primary cohort's exposure spline knots, precipitation
reference (median non-zero exposure), precipitation contrast (90th percentile
of non-zero exposure), and representative sanitation levels. This makes the
sensitivity analysis directly comparable to the primary analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from dlnm.dlnm_crossbasis import build_crossbasis, cumulative_rr, ns_basis


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    REPO_ROOT
    / "out/modeling/SYNMODINF_footprint_daily_modeling_dataset_v7.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "out/modeling/daily_dlnm_primary_sensitivity_v7"
LAG_VALUES = np.arange(0, 15, dtype=float)
LAG_ALL_KNOTS = np.array([0.0, 3.0, 7.0, 14.0])
PRECIP_COLUMNS = ["P_precip"] + [f"P_precip_lag{k}" for k in range(1, 15)]
TEMP_COLUMNS = ["T_mean_C"] + [f"T_mean_C_lag{k}" for k in range(1, 15)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 19-HSA primary and >=1 case/day sensitivity DLNMs."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sensitivity-threshold",
        type=float,
        default=1.0,
        help="Minimum HSA mean daily diarrheal count (default: 1.0).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_quasi_poisson(x: np.ndarray, y: np.ndarray):
    model = sm.GLM(
        y,
        sm.add_constant(np.asarray(x, dtype=float), has_constant="add"),
        family=sm.families.Poisson(),
    )
    return model.fit(scale="X2")


def nested_f_test(restricted, full) -> tuple[float, float, int]:
    deviance_difference = restricted.deviance - full.deviance
    df_difference = int(round(restricted.df_resid - full.df_resid))
    if df_difference <= 0:
        return np.nan, np.nan, df_difference
    f_statistic = (deviance_difference / df_difference) / full.scale
    p_value = stats.f.sf(f_statistic, df_difference, full.df_resid)
    return float(f_statistic), float(p_value), df_difference


def base_design(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    hsa_dummies = pd.get_dummies(df["hsa_id"], drop_first=True, dtype=float)

    n_days = int(df["day_of_study"].max()) + 1
    n_interior_knots = max(5, n_days // 90)
    time_spline, time_knots = ns_basis(
        df["day_of_study"].to_numpy(),
        n_interior_knots=n_interior_knots,
    )
    dow_dummies = pd.get_dummies(
        df["day_of_week"], prefix="dow", drop_first=True, dtype=float
    )

    calendar_columns = ["is_ramadan", "is_eid_fitr", "is_eid_adha"]
    calendar = np.column_stack(
        [
            df[column].to_numpy(dtype=float)
            if column in df.columns
            else np.zeros(len(df), dtype=float)
            for column in calendar_columns
        ]
    )

    design = np.column_stack(
        [hsa_dummies.to_numpy(), time_spline, dow_dummies.to_numpy(), calendar]
    )
    metadata = {
        "n_hsa_fixed_effects": int(hsa_dummies.shape[1]),
        "n_time_spline_columns": int(time_spline.shape[1]),
        "time_interior_knots": int(n_interior_knots),
        "time_all_knots": [float(value) for value in time_knots],
        "n_dow_indicators": int(dow_dummies.shape[1]),
        "calendar_indicators": calendar_columns,
    }
    return design, metadata


def fit_exposure_models(
    df: pd.DataFrame,
    lag_columns: list[str],
    exposure_knots: np.ndarray,
) -> dict:
    outcome = df["diarrheal_count"].to_numpy(dtype=float)
    sanitation = df["infra_quality"].fillna(df["infra_quality"].mean()).to_numpy()
    sanitation_center = float(sanitation.mean())
    sanitation_centered = sanitation - sanitation_center

    base_x, base_metadata = base_design(df)
    exposure_history = df[lag_columns].to_numpy(dtype=float)
    crossbasis, crossbasis_names, crossbasis_metadata = build_crossbasis(
        exposure_history,
        exp_n_int=1,
        lag_n_int=2,
        exp_all_knots=exposure_knots,
        lag_all_knots=LAG_ALL_KNOTS,
        lag_values=LAG_VALUES,
    )
    interaction = crossbasis * sanitation_centered[:, None]

    base_result = fit_quasi_poisson(base_x, outcome)
    main_result = fit_quasi_poisson(np.column_stack([base_x, crossbasis]), outcome)
    interaction_result = fit_quasi_poisson(
        np.column_stack([base_x, crossbasis, interaction]), outcome
    )

    f_main, p_main, df_main = nested_f_test(base_result, main_result)
    f_interaction, p_interaction, df_interaction = nested_f_test(
        main_result, interaction_result
    )

    n_crossbasis = crossbasis.shape[1]
    first_crossbasis_parameter = base_x.shape[1] + 1
    main_slice = slice(
        first_crossbasis_parameter,
        first_crossbasis_parameter + n_crossbasis,
    )
    interaction_slice = slice(
        first_crossbasis_parameter + n_crossbasis,
        first_crossbasis_parameter + 2 * n_crossbasis,
    )
    coefficients = np.asarray(interaction_result.params)
    covariance = np.asarray(interaction_result.cov_params())

    return {
        "base_result": base_result,
        "main_result": main_result,
        "interaction_result": interaction_result,
        "F_main": f_main,
        "p_main": p_main,
        "df_main": df_main,
        "F_interaction": f_interaction,
        "p_interaction": p_interaction,
        "df_interaction": df_interaction,
        "dispersion": float(interaction_result.scale),
        "sanitation_center": sanitation_center,
        "crossbasis_metadata": crossbasis_metadata,
        "crossbasis_names": crossbasis_names,
        "beta_main": coefficients[main_slice],
        "beta_interaction": coefficients[interaction_slice],
        "variance_main": covariance[main_slice, main_slice],
        "variance_interaction": covariance[interaction_slice, interaction_slice],
        "variance_cross": covariance[main_slice, interaction_slice],
        "base_metadata": base_metadata,
    }


def cumulative_contrast(
    fitted: dict,
    sanitation_value: float,
    exposure_value: float,
    reference_value: float,
) -> tuple[float, float, float]:
    centered = sanitation_value - fitted["sanitation_center"]
    beta = fitted["beta_main"] + centered * fitted["beta_interaction"]
    covariance = (
        fitted["variance_main"]
        + centered**2 * fitted["variance_interaction"]
        + centered
        * (fitted["variance_cross"] + fitted["variance_cross"].T)
    )
    log_rr, standard_error = cumulative_rr(
        beta,
        covariance,
        fitted["crossbasis_metadata"],
        np.array([exposure_value]),
        reference_exp=reference_value,
    )
    rr = float(np.exp(log_rr[0]))
    lower = float(np.exp(log_rr[0] - 1.96 * standard_error[0]))
    upper = float(np.exp(log_rr[0] + 1.96 * standard_error[0]))
    return rr, lower, upper


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, parse_dates=["date"])
    df = df.sort_values(["hsa_id", "date"]).reset_index(drop=True)
    required = {
        "hsa_id",
        "date",
        "diarrheal_count",
        "infra_quality",
        "day_of_study",
        "day_of_week",
        *PRECIP_COLUMNS,
        *TEMP_COLUMNS,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    hsa_means = df.groupby("hsa_id")["diarrheal_count"].mean().sort_index()
    primary_hsas = hsa_means.index.tolist()
    sensitivity_hsas = hsa_means[
        hsa_means >= args.sensitivity_threshold
    ].index.tolist()
    if len(primary_hsas) != 19:
        raise ValueError(
            f"Expected 19 HSAs in the primary dataset; found {len(primary_hsas)}."
        )

    primary_precipitation = df[PRECIP_COLUMNS].to_numpy(dtype=float).ravel()
    primary_nonzero_precipitation = primary_precipitation[
        primary_precipitation > 0
    ]
    precip_reference = float(np.percentile(primary_nonzero_precipitation, 50))
    precip_contrast = float(np.percentile(primary_nonzero_precipitation, 90))
    precip_interior_knot = float(np.percentile(primary_nonzero_precipitation, 80))
    precip_knots = np.array(
        [
            float(primary_precipitation.min()),
            precip_interior_knot,
            float(primary_precipitation.max()),
        ]
    )

    primary_temperature = df[TEMP_COLUMNS].to_numpy(dtype=float).ravel()
    temperature_interior_knot = float(np.percentile(primary_temperature, 50))
    temperature_knots = np.array(
        [
            float(primary_temperature.min()),
            temperature_interior_knot,
            float(primary_temperature.max()),
        ]
    )

    hsa_sanitation = df.groupby("hsa_id")["infra_quality"].first().sort_values()
    representative_sanitation = {
        "lower-sanitation (five-HSA mean)": float(hsa_sanitation.iloc[:5].mean()),
        "higher-sanitation (five-HSA mean)": float(hsa_sanitation.iloc[-5:].mean()),
    }

    cohorts = {
        "primary_all_hsas": primary_hsas,
        "sensitivity_mean_daily_ge_1": sensitivity_hsas,
    }
    model_rows: list[dict] = []
    rr_rows: list[dict] = []
    temperature_rows: list[dict] = []
    cohort_rows: list[dict] = []
    cohort_metadata: dict[str, dict] = {}

    print(f"Input: {input_csv}")
    print(
        f"Primary exposure contrast: {precip_contrast:.4f} vs "
        f"{precip_reference:.4f} mm/day"
    )

    for cohort_name, hsa_ids in cohorts.items():
        cohort_df = df[df["hsa_id"].isin(hsa_ids)].copy().reset_index(drop=True)
        print(f"\n{cohort_name}: {len(hsa_ids)} HSAs, {len(cohort_df)} HSA-days")

        precipitation_fit = fit_exposure_models(
            cohort_df, PRECIP_COLUMNS, precip_knots
        )
        temperature_fit = fit_exposure_models(
            cohort_df, TEMP_COLUMNS, temperature_knots
        )

        model_rows.append(
            {
                "cohort": cohort_name,
                "n_hsas": len(hsa_ids),
                "n_hsa_days": len(cohort_df),
                "dispersion": precipitation_fit["dispersion"],
                "F_precipitation_main": precipitation_fit["F_main"],
                "df_precipitation_main": precipitation_fit["df_main"],
                "p_precipitation_main": precipitation_fit["p_main"],
                "F_precipitation_interaction": precipitation_fit["F_interaction"],
                "df_precipitation_interaction": precipitation_fit["df_interaction"],
                "p_precipitation_interaction": precipitation_fit["p_interaction"],
                "precipitation_reference_mm": precip_reference,
                "precipitation_contrast_mm": precip_contrast,
                "precipitation_interior_knot_mm": precip_interior_knot,
                "sanitation_center": precipitation_fit["sanitation_center"],
            }
        )
        temperature_rows.append(
            {
                "cohort": cohort_name,
                "n_hsas": len(hsa_ids),
                "F_temperature_main": temperature_fit["F_main"],
                "df_temperature_main": temperature_fit["df_main"],
                "p_temperature_main": temperature_fit["p_main"],
                "F_temperature_interaction": temperature_fit["F_interaction"],
                "df_temperature_interaction": temperature_fit["df_interaction"],
                "p_temperature_interaction": temperature_fit["p_interaction"],
                "temperature_interior_knot_C": temperature_interior_knot,
            }
        )

        for sanitation_label, sanitation_value in representative_sanitation.items():
            rr, lower, upper = cumulative_contrast(
                precipitation_fit,
                sanitation_value,
                precip_contrast,
                precip_reference,
            )
            rr_rows.append(
                {
                    "cohort": cohort_name,
                    "sanitation_level": sanitation_label,
                    "sanitation_value": sanitation_value,
                    "precipitation_reference_mm": precip_reference,
                    "precipitation_contrast_mm": precip_contrast,
                    "cumulative_RR": rr,
                    "CI95_lower": lower,
                    "CI95_upper": upper,
                }
            )

        cohort_metadata[cohort_name] = {
            "hsa_ids": [str(hsa_id) for hsa_id in hsa_ids],
            "n_hsas": len(hsa_ids),
            "n_hsa_days": len(cohort_df),
            "base_design": precipitation_fit["base_metadata"],
        }
        for hsa_id, mean_count in hsa_means.items():
            cohort_rows.append(
                {
                    "cohort": cohort_name,
                    "hsa_id": hsa_id,
                    "mean_daily_diarrheal_count": mean_count,
                    "included": hsa_id in hsa_ids,
                }
            )

        print(
            "  precipitation: "
            f"F_main={precipitation_fit['F_main']:.3f}, "
            f"p={precipitation_fit['p_main']:.6g}; "
            f"F_interaction={precipitation_fit['F_interaction']:.3f}, "
            f"p={precipitation_fit['p_interaction']:.6g}"
        )

    model_df = pd.DataFrame(model_rows)
    rr_df = pd.DataFrame(rr_rows)
    temperature_df = pd.DataFrame(temperature_rows)
    cohort_df = pd.DataFrame(cohort_rows)
    model_df.to_csv(output_dir / "dlnm_model_summary.csv", index=False)
    rr_df.to_csv(output_dir / "dlnm_rr_contrasts.csv", index=False)
    temperature_df.to_csv(output_dir / "dlnm_temperature_screen.csv", index=False)
    cohort_df.to_csv(output_dir / "dlnm_cohort_hsas.csv", index=False)

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "input_csv": str(input_csv),
        "input_sha256": sha256(input_csv),
        "model": {
            "family": "Poisson GLM with Pearson chi-square dispersion",
            "fixed_effects": "HSA",
            "seasonality": "natural spline of study day; one interior knot per approximately 90 days, minimum five",
            "calendar_adjustment": [
                "day of week",
                "Ramadan",
                "Eid al-Fitr",
                "Eid al-Adha",
            ],
            "lag_days": [0, 14],
            "lag_all_knots": LAG_ALL_KNOTS.tolist(),
            "precipitation_exposure_all_knots": precip_knots.tolist(),
            "temperature_exposure_all_knots": temperature_knots.tolist(),
            "interaction": "cross-basis multiplied by mean-centered infra_quality",
        },
        "contrast": {
            "precipitation_reference_mm": precip_reference,
            "precipitation_contrast_mm": precip_contrast,
            "definition": "primary-cohort median versus 90th percentile among non-zero daily precipitation values across lag columns",
            "representative_sanitation": representative_sanitation,
        },
        "sensitivity_threshold_mean_daily_cases": args.sensitivity_threshold,
        "cohorts": cohort_metadata,
    }
    (output_dir / "dlnm_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Daily DLNM primary and sensitivity analyses",
        "",
        f"Input SHA-256: `{metadata['input_sha256']}`",
        "",
        (
            f"The cumulative precipitation contrast is {precip_contrast:.2f} "
            f"versus {precip_reference:.2f} mm/day (90th percentile versus "
            "median of non-zero precipitation in the primary cohort)."
        ),
        "",
    ]
    for row in model_rows:
        lines.extend(
            [
                f"## {row['cohort']}",
                "",
                (
                    f"{row['n_hsas']} HSAs and {row['n_hsa_days']:,} HSA-days; "
                    f"dispersion = {row['dispersion']:.2f}; precipitation main "
                    f"effect F = {row['F_precipitation_main']:.2f}, "
                    f"p = {row['p_precipitation_main']:.4g}; sanitation interaction "
                    f"F = {row['F_precipitation_interaction']:.2f}, "
                    f"p = {row['p_precipitation_interaction']:.4g}."
                ),
                "",
            ]
        )
        for rr_row in [item for item in rr_rows if item["cohort"] == row["cohort"]]:
            lines.append(
                f"- {rr_row['sanitation_level']}: RR = {rr_row['cumulative_RR']:.2f} "
                f"(95% CI {rr_row['CI95_lower']:.2f}–{rr_row['CI95_upper']:.2f})"
            )
        lines.append("")
    (output_dir / "dlnm_primary_sensitivity_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"\nWrote reproducible outputs to {output_dir}")


if __name__ == "__main__":
    main()
