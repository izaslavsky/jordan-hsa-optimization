#!/usr/bin/env python3
"""
Step 3: DLNM cross-basis analysis — Python implementation of Gasparrini's approach.

Constructs a distributed lag non-linear model (DLNM) cross-basis by taking the
tensor product of:
  - a natural cubic spline basis over exposure values (precipitation or temperature)
  - a natural cubic spline basis over the lag dimension (weeks 0–3)

Fits quasi-Poisson GLMs with the cross-basis and tests whether infrastructure
quality (JMP sanitation %) modifies the climate-diarrhea relationship.

Two analysis passes:
  Pass A — pooled model (all HSAs together, HSA fixed effects):
    Tests overall climate-diarrhea cross-basis signal and its interaction
    with infra_quality. Direct Python analog of Gasparrini's pooled approach.

  Pass B — per-HSA models (13 HSAs with mean >= 2 cases/week):
    Extracts HSA-specific cumulative RR at the 75th pct precipitation.
    Meta-regression of these RRs on infra_quality via WLS.

Outputs in out/dlnm/crossbasis/:
  model_comparison.csv          — F-tests for cross-basis terms
  cumulative_rr_precip.png      — cumulative lag-response curve (pooled)
  exposure_response_surface.png — 2D heat map of RR over exposure × lag
  per_hsa_cumrr.csv             — per-HSA cumulative RRs + infra_quality
  meta_regression.csv           — WLS meta-regression of RR on infra_quality
  meta_regression.png           — scatter plot + regression line
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "out/dlnm/dlnm_dataset.csv"
DEFAULT_META  = BASE_DIR / "data" / "hsa_metadata.csv"
DEFAULT_OUTPUT = BASE_DIR / "out/dlnm/crossbasis"

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DLNM cross-basis analysis (Python)")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--meta-csv",  default=str(DEFAULT_META))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--min-mean-cases", type=float, default=2.0,
                        help="Min mean weekly cases for per-HSA models (default 2)")
    return parser.parse_args(argv)

# ---------------------------------------------------------------------------
# Natural cubic spline basis (truncated power series, natural constraints)
# ---------------------------------------------------------------------------
# This implementation gives explicit knot control and consistent prediction at
# new points — essential for DLNM where the same basis must be evaluated at
# the exposure grid used for the response curve.

def _ns_basis_from_knots(x, all_knots):
    """
    Natural cubic spline columns using the Reinsch truncated-power representation.
    all_knots includes two boundary knots (first and last) and n interior knots.
    Returns array of shape (len(x), n_interior_knots + 1).
    """
    x = np.asarray(x, dtype=float)
    interior = all_knots[1:-1]
    xK_1 = all_knots[-2]   # second-to-last knot
    xK   = all_knots[-1]   # last boundary knot

    def _h(t, xi):
        # Natural cubic spline basis column for interior knot xi
        pxi  = np.maximum(t - xi,   0) ** 3
        pK1  = np.maximum(t - xK_1, 0) ** 3
        pK   = np.maximum(t - xK,   0) ** 3
        scale = xK - xK_1
        d_xi  = (pxi  - pK ) / scale
        d_K1  = (pK1  - pK ) / scale
        return d_xi - d_K1

    cols = [x.copy()]  # linear term
    for xi in interior:
        cols.append(_h(x, xi))
    return np.column_stack(cols)


def ns_basis(x, n_interior_knots=1, knots=None, all_knots=None):
    """
    Natural cubic spline basis.

    Pass one of:
      - knots (interior knot positions) — boundary knots taken from data range
      - all_knots (boundary + interior) — used as-is for prediction at new points
      - n_interior_knots — knots placed at quantiles of x

    Returns (basis_array, all_knots) where all_knots can be reused for prediction.
    """
    x = np.asarray(x, dtype=float)
    if all_knots is not None:
        return _ns_basis_from_knots(x, all_knots), all_knots
    if knots is not None:
        interior = np.asarray(knots)
    else:
        pcts = np.linspace(0, 100, n_interior_knots + 2)[1:-1]
        interior = np.percentile(x, pcts)
    all_k = np.concatenate([[x.min()], interior, [x.max()]])
    return _ns_basis_from_knots(x, all_k), all_k


# ---------------------------------------------------------------------------
# Cross-basis constructor (exposure × lag tensor product)
# ---------------------------------------------------------------------------

def build_crossbasis(Q_matrix, exp_n_int=1, lag_n_int=1,
                     exp_all_knots=None, lag_all_knots=None,
                     lag_values=None):
    """
    Build a DLNM cross-basis from an exposure matrix Q.

    exp_n_int / lag_n_int : number of interior knots (placed at quantiles).
    exp_all_knots / lag_all_knots : pre-computed knot arrays (for per-HSA
        models that must share the same basis as the pooled model).
    lag_values : actual lag positions (e.g. [1,2,3,5,7,10,14] for daily lags).
        Defaults to np.arange(n_lags) if not supplied.

    Returns (cb, col_names, meta) where meta carries knots for prediction.
    """
    n_obs, n_lags = Q_matrix.shape
    lags = np.arange(n_lags, dtype=float) if lag_values is None else np.asarray(lag_values, dtype=float)

    # Fit exposure basis on all exposure values across all lags and obs
    all_exp = Q_matrix.flatten()
    _, exp_all_knots = ns_basis(all_exp, n_interior_knots=exp_n_int,
                                 all_knots=exp_all_knots)

    # Lag basis evaluated at each integer lag
    _, lag_all_knots = ns_basis(lags, n_interior_knots=lag_n_int,
                                 all_knots=lag_all_knots)
    B_lag, _ = ns_basis(lags, all_knots=lag_all_knots)

    n_exp_cols = len(exp_all_knots) - 1   # interior knots + 1 linear term
    n_lag_cols = B_lag.shape[1]

    # Cross-basis: sum over lags of kron(B_exp[l], B_lag[l])
    cb = np.zeros((n_obs, n_exp_cols * n_lag_cols))
    for l in range(n_lags):
        B_exp_l, _ = ns_basis(Q_matrix[:, l], all_knots=exp_all_knots)
        cb += np.einsum("ni,j->nij", B_exp_l, B_lag[l]).reshape(n_obs, -1)

    col_names = [f"cb_e{e}_l{l}"
                 for e in range(n_exp_cols) for l in range(n_lag_cols)]
    meta = {
        "n_exp_cols": n_exp_cols, "n_lag_cols": n_lag_cols,
        "lags": lags, "B_lag": B_lag,
        "exp_all_knots": exp_all_knots, "lag_all_knots": lag_all_knots,
    }
    return cb, col_names, meta


def predict_rr(coef_cb, vcov_cb, meta, exp_grid, lag_grid, reference_exp=0.0):
    """
    Log-RR surface over (exposure × lag) grid relative to reference_exp.
    Returns (logRR, seRR), each shape (len(exp_grid), len(lag_grid)).
    """
    exp_ak = meta["exp_all_knots"]
    lag_ak = meta["lag_all_knots"]

    B_lag_grid, _ = ns_basis(lag_grid.astype(float), all_knots=lag_ak)
    B_exp_ref,  _ = ns_basis(np.array([reference_exp]), all_knots=exp_ak)
    B_exp_ref = B_exp_ref[0]

    n_e, n_l = len(exp_grid), len(lag_grid)
    logRR = np.zeros((n_e, n_l))
    seRR  = np.zeros((n_e, n_l))

    for i, exp_val in enumerate(exp_grid):
        B_exp_val, _ = ns_basis(np.array([exp_val]), all_knots=exp_ak)
        B_exp_val = B_exp_val[0]
        for j in range(n_l):
            b_lag = B_lag_grid[j]
            contrast = np.kron(B_exp_val, b_lag) - np.kron(B_exp_ref, b_lag)
            logRR[i, j] = coef_cb @ contrast
            seRR[i, j]  = np.sqrt(np.maximum(contrast @ vcov_cb @ contrast, 0))

    return logRR, seRR


def cumulative_rr(coef_cb, vcov_cb, meta, exp_grid, reference_exp=0.0):
    """Cumulative (sum over lags) log-RR and SE for each exposure value."""
    exp_ak    = meta["exp_all_knots"]
    B_lag_sum = meta["B_lag"].sum(axis=0)

    B_exp_ref, _ = ns_basis(np.array([reference_exp]), all_knots=exp_ak)
    B_exp_ref = B_exp_ref[0]

    cum_logRR = np.zeros(len(exp_grid))
    cum_se    = np.zeros(len(exp_grid))
    for i, exp_val in enumerate(exp_grid):
        B_exp_val, _ = ns_basis(np.array([exp_val]), all_knots=exp_ak)
        B_exp_val = B_exp_val[0]
        contrast = np.kron(B_exp_val, B_lag_sum) - np.kron(B_exp_ref, B_lag_sum)
        cum_logRR[i] = coef_cb @ contrast
        cum_se[i]    = np.sqrt(np.maximum(contrast @ vcov_cb @ contrast, 0))
    return cum_logRR, cum_se


def main(argv=None):
    args = parse_args(argv)

    INPUT_CSV  = Path(args.input_csv)
    META_CSV   = Path(args.meta_csv)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DLNM CROSS-BASIS ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------------------
    print("\n[1/7] Loading data...")
    df   = pd.read_csv(INPUT_CSV)
    meta_df = pd.read_csv(META_CSV)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(["hsa_id", "week_start"]).reset_index(drop=True)
    print(f"  {len(df)} obs, {df['hsa_id'].nunique()} HSAs")

    # Filter for per-HSA analysis
    hsa_keep = meta_df[~meta_df["low_count_flag"]]["hsa_id"].tolist()
    df_full = df[df["hsa_id"].isin(hsa_keep)].copy()
    print(f"  {len(df_full)} obs after removing {df['hsa_id'].nunique() - len(hsa_keep)} low-count HSAs")

    # ---------------------------------------------------------------------------
    # 2. Build cross-basis for precipitation (pooled analysis)
    # ---------------------------------------------------------------------------
    print("\n[2/7] Building precipitation cross-basis (pooled)...")

    Q_precip_cols = ["Q_precip_w0", "Q_precip_w1", "Q_precip_w2", "Q_precip_w3"]
    Q_precip = df_full[Q_precip_cols].values  # (n_obs, 4 lags)

    # Interior knot at 80th pct of NON-ZERO precipitation.
    # Jordan precipitation is ~48% zero-weeks; placing the knot at the overall
    # median lands at 0 (boundary), degenerating the spline. The 80th pct of
    # non-zero values (≈2.37 mm/day) puts the knot where the distribution has
    # meaningful mass and the response is likely non-linear.
    nonzero_precip = Q_precip[Q_precip > 0].flatten()
    exp_int_knot = np.percentile(nonzero_precip, 80)
    exp_all_knots = np.array([0.0, exp_int_knot, Q_precip.max()])
    print(f"  Exposure interior knot: {exp_int_knot:.4f} mm/day "
          f"(80th pct of non-zero weeks)")

    CB_precip, cb_col_names, cb_meta_precip = build_crossbasis(
        Q_precip, exp_n_int=1, lag_n_int=1,
        exp_all_knots=exp_all_knots,
    )
    print(f"  Cross-basis shape: {CB_precip.shape}  ({len(cb_col_names)} columns)")
    print(f"  Exposure knots: {cb_meta_precip['exp_all_knots'].round(4)}")
    print(f"  Lag knots:      {cb_meta_precip['lag_all_knots'].round(4)}")

    # Also build temperature cross-basis (3 lags)
    Q_temp_cols = ["Q_temp_w0", "Q_temp_w1", "Q_temp_w2"]
    Q_temp = df_full[Q_temp_cols].values
    CB_temp, cb_temp_names, cb_meta_temp = build_crossbasis(
        Q_temp, exp_n_int=1, lag_n_int=1,
    )

    # ---------------------------------------------------------------------------
    # 3. Assemble pooled design matrices
    # ---------------------------------------------------------------------------
    print("\n[3/7] Fitting pooled quasi-Poisson models...")

    outcome = df_full["diarrheal_count_adjusted"].values.astype(float)
    ns_cols = [c for c in df_full.columns if c.startswith("ns_time_")]
    hsa_dummies = pd.get_dummies(df_full["hsa_id"], drop_first=True, dtype=float)

    def fit_qp(X, y, label):
        model = sm.GLM(y, sm.add_constant(X.astype(float), has_constant="add"),
                       family=sm.families.Poisson())
        res = model.fit(scale="X2")
        print(f"  {label}: deviance={res.deviance:.1f}  df={res.df_resid}  φ={res.scale:.2f}")
        return res

    def ftest(res_r, res_f, label):
        dD  = res_r.deviance - res_f.deviance
        ddf = res_r.df_resid  - res_f.df_resid
        F   = (dD / ddf) / res_f.scale
        p   = 1 - stats.f.cdf(F, ddf, res_f.df_resid)
        sig = "**" if p < 0.05 else "  "
        print(f"  {sig} {label:<45s}  F({ddf},{res_f.df_resid:.0f})={F:.3f}  p={p:.4f}")
        return {"comparison": label, "delta_dev": dD, "delta_df": ddf,
                "F": F, "p_value": p, "phi": res_f.scale}

    # Base components
    base_X = pd.concat([hsa_dummies, df_full[ns_cols]], axis=1).reset_index(drop=True)

    X_base    = base_X.values
    X_cb_p    = np.c_[base_X.values, CB_precip]
    X_cb_t    = np.c_[base_X.values, CB_temp]
    X_cb_both = np.c_[base_X.values, CB_precip, CB_temp]

    res_base   = fit_qp(pd.DataFrame(X_base),   outcome, "M_base  (HSA + time spline)")
    res_precip = fit_qp(pd.DataFrame(X_cb_p),   outcome, "M_precip (+ precip cross-basis)")
    res_temp   = fit_qp(pd.DataFrame(X_cb_t),   outcome, "M_temp   (+ temp cross-basis)")
    res_both   = fit_qp(pd.DataFrame(X_cb_both),outcome, "M_both   (+ both cross-bases)")

    print("\n  Model comparison (F-tests):")
    comp_rows = [
        ftest(res_base,   res_precip, "precip cross-basis vs base"),
        ftest(res_base,   res_temp,   "temp cross-basis vs base"),
        ftest(res_base,   res_both,   "both cross-bases vs base"),
        ftest(res_precip, res_both,   "both vs precip (temp add)"),
    ]
    pd.DataFrame(comp_rows).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # ---------------------------------------------------------------------------
    # 4. Extract cross-basis coefficients for precipitation (from M_precip)
    # ---------------------------------------------------------------------------
    print("\n[4/7] Extracting precipitation cross-basis response...")

    params   = res_precip.params
    vcov_all = np.asarray(res_precip.cov_params())

    # Identify cross-basis parameter positions
    n_base = X_base.shape[1] + 1  # +1 for const
    cb_idx = slice(n_base, n_base + CB_precip.shape[1])

    coef_cb = np.asarray(params)[cb_idx]
    vcov_cb = vcov_all[cb_idx, :][:, cb_idx]

    # Exposure grid: 0 to 95th pct of observed precipitation
    exp_grid = np.linspace(0, np.percentile(Q_precip, 95), 50)
    lag_grid = np.arange(4, dtype=float)  # lags 0,1,2,3

    reference_exp = 0.0  # RR = 1 at zero rainfall

    # Compute cumulative RR
    cum_logRR, cum_se = cumulative_rr(coef_cb, vcov_cb, cb_meta_precip,
                                       exp_grid, reference_exp)

    # Compute lag-specific RR surface
    logRR_surface, seRR_surface = predict_rr(coef_cb, vcov_cb, cb_meta_precip,
                                              exp_grid, lag_grid, reference_exp)

    # ---------------------------------------------------------------------------
    # 5. Interaction with infrastructure quality (pooled)
    # ---------------------------------------------------------------------------
    print("\n[5/7] Testing infrastructure quality × precipitation interaction...")

    # Centre infra_quality at its mean for interpretability
    infra = df_full["infra_quality"].values
    infra_c = infra - infra.mean()

    # Interaction cross-basis: CB_precip × infra_quality_centred
    CB_precip_x_infra = CB_precip * infra_c[:, None]

    X_interact = np.c_[base_X.values, CB_precip, CB_precip_x_infra]
    res_interact = fit_qp(pd.DataFrame(X_interact), outcome,
                          "M_interact (+cb×infra_quality)")

    ftest_interact = ftest(res_precip, res_interact,
                            "interaction (cb×infra_quality) vs precip")
    comp_rows.append(ftest_interact)
    pd.DataFrame(comp_rows).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Extract interaction cross-basis coefs
    n_main = X_base.shape[1] + 1 + CB_precip.shape[1]
    n_int  = CB_precip.shape[1]
    int_idx = slice(n_main, n_main + n_int)
    coef_int = np.asarray(res_interact.params)[int_idx]
    vcov_all_int = np.asarray(res_interact.cov_params())
    vcov_int = vcov_all_int[int_idx, :][:, int_idx]

    # Cumulative RR at mean infra_quality vs ±1 SD
    sd_infra = infra_c.std()
    cum_logRR_lo, cum_se_lo = cumulative_rr(coef_cb + coef_int * (-sd_infra),
                                             vcov_cb, cb_meta_precip,
                                             exp_grid, reference_exp)
    cum_logRR_hi, cum_se_hi = cumulative_rr(coef_cb + coef_int * (+sd_infra),
                                             vcov_cb, cb_meta_precip,
                                             exp_grid, reference_exp)

    # ---------------------------------------------------------------------------
    # 6. Per-HSA analysis — cumulative RR at 75th pct precipitation
    # ---------------------------------------------------------------------------
    print("\n[6/7] Per-HSA models — cumulative RR at 75th pct precipitation...")

    exp_ref75 = np.percentile(Q_precip, 75)
    print(f"  75th pct precipitation: {exp_ref75:.4f} mm/day")

    per_hsa_rows = []

    for hsa in hsa_keep:
        sub = df_full[df_full["hsa_id"] == hsa].copy()
        if len(sub) < 20:
            continue

        Q_h = sub[Q_precip_cols].values
        ns_h = sub[ns_cols].astype(float)
        y_h  = sub["diarrheal_count_adjusted"].values.astype(float)

        # Use same knots as pooled model for comparability
        CB_h, _, meta_h = build_crossbasis(
            Q_h, exp_n_int=1, lag_n_int=1,
            exp_all_knots=cb_meta_precip["exp_all_knots"],
            lag_all_knots=cb_meta_precip["lag_all_knots"],
        )

        X_h = np.c_[np.ones(len(sub)), ns_h.values, CB_h]

        try:
            model_h = sm.GLM(y_h, X_h, family=sm.families.Poisson())
            res_h   = model_h.fit(scale="X2")
        except Exception as e:
            print(f"  {hsa}: fit failed ({e})")
            continue

        n_base_h = 1 + ns_h.shape[1]
        cb_idx_h = slice(n_base_h, n_base_h + CB_h.shape[1])
        coef_h   = np.asarray(res_h.params)[cb_idx_h]
        vcov_h   = np.asarray(res_h.cov_params())[cb_idx_h, :][:, cb_idx_h]

        cum_lr, cum_s = cumulative_rr(coef_h, vcov_h, meta_h,
                                       np.array([exp_ref75]), reference_exp)
        hsa_infra = meta_df.loc[meta_df["hsa_id"] == hsa, "jmp_san_pct"].values[0]
        hsa_gov   = meta_df.loc[meta_df["hsa_id"] == hsa, "governorate"].values[0]

        per_hsa_rows.append({
            "hsa_id": hsa,
            "governorate": hsa_gov,
            "jmp_san_pct": hsa_infra,
            "n_obs": len(sub),
            "cum_logRR_75pct": cum_lr[0],
            "cum_se_75pct": cum_s[0],
            "cum_RR_75pct": np.exp(cum_lr[0]),
            "cum_RR_lo95": np.exp(cum_lr[0] - 1.96 * cum_s[0]),
            "cum_RR_hi95": np.exp(cum_lr[0] + 1.96 * cum_s[0]),
            "dispersion": res_h.scale,
        })
        print(f"  {hsa:<48s}  φ={res_h.scale:.2f}  "
              f"cumRR={np.exp(cum_lr[0]):.3f} [{np.exp(cum_lr[0]-1.96*cum_s[0]):.3f},"
              f"{np.exp(cum_lr[0]+1.96*cum_s[0]):.3f}]  san={hsa_infra:.1f}%")

    hsa_rr_df = pd.DataFrame(per_hsa_rows)
    hsa_rr_df.to_csv(OUTPUT_DIR / "per_hsa_cumrr.csv", index=False)

    # Meta-regression: weighted least squares of cumRR ~ infra_quality
    # Weight = 1/variance = 1/se²
    if len(hsa_rr_df) >= 5:
        weights = 1 / (hsa_rr_df["cum_se_75pct"] ** 2)
        X_meta = sm.add_constant(hsa_rr_df["jmp_san_pct"].values, has_constant='add')
        meta_model = sm.WLS(hsa_rr_df["cum_logRR_75pct"].values, X_meta,
                            weights=weights)
        meta_res = meta_model.fit()
        slope    = meta_res.params[1]
        slope_se = meta_res.bse[1]
        slope_p  = meta_res.pvalues[1]
        print(f"\n  Meta-regression (logRR ~ sanitation %): "
              f"β={slope:.4f} ± {slope_se:.4f}  p={slope_p:.4f}")

        meta_out = pd.DataFrame({
            "term": ["intercept", "jmp_san_pct"],
            "coef": meta_res.params,
            "se":   meta_res.bse,
            "p_value": meta_res.pvalues,
        })
        meta_out.to_csv(OUTPUT_DIR / "meta_regression.csv", index=False)
    else:
        print("  Too few HSAs for meta-regression.")
        slope, slope_se, slope_p = None, None, None

    # ---------------------------------------------------------------------------
    # 7. Attributable fractions
    # ---------------------------------------------------------------------------
    print("\n[7/9] Computing attributable fractions (precipitation above zero)...")

    # Counterfactual: all precipitation set to reference (0 mm/day)
    # Use M_interact for proper infra-quality-adjusted counterfactual
    fitted_obs = res_interact.fittedvalues

    # Counterfactual cross-basis at exposure = 0 for all lags and obs
    Q_counterfactual = np.zeros_like(Q_precip)
    CB_counter, _, _ = build_crossbasis(
        Q_counterfactual, exp_n_int=1, lag_n_int=1,
        exp_all_knots=cb_meta_precip["exp_all_knots"],
        lag_all_knots=cb_meta_precip["lag_all_knots"],
    )
    CB_counter_x_infra = CB_counter * infra_c[:, None]

    # Extract CB coefs from the interaction model (separate from coef_cb which is from res_precip)
    # params_int layout: [const, base_X..., CB_precip..., CB_precip_x_infra...]
    n_cb_start_int = base_X.shape[1] + 1   # +1 for constant prepended by fit_qp
    cb_idx_int = slice(n_cb_start_int, n_cb_start_int + CB_precip.shape[1])
    coef_cb_int = np.asarray(res_interact.params)[cb_idx_int]

    # Compute counterfactual via delta in linear predictor — avoids reconstructing X
    delta_lp = (
        (CB_counter @ coef_cb_int + CB_counter_x_infra @ coef_int) -
        (CB_precip  @ coef_cb_int + CB_precip_x_infra  @ coef_int)
    )
    mu_counter = np.exp(np.log(fitted_obs) + delta_lp)

    # Attributable fraction per observation: (fitted - counterfactual) / fitted
    af_obs = (fitted_obs - mu_counter) / fitted_obs

    # Aggregate by HSA
    df_full_copy = df_full.copy().reset_index(drop=True)
    df_full_copy["af"] = af_obs.values
    df_full_copy["fitted"] = fitted_obs.values
    df_full_copy["counterfactual"] = mu_counter

    af_by_hsa = df_full_copy.groupby("hsa_id").agg(
        total_observed=("diarrheal_count_adjusted", "sum"),
        total_fitted=("fitted", "sum"),
        total_counterfactual=("counterfactual", "sum"),
        jmp_san_pct=("jmp_san_pct", "first"),
    ).reset_index()
    af_by_hsa["AF_pct"] = (
        (af_by_hsa["total_fitted"] - af_by_hsa["total_counterfactual"])
        / af_by_hsa["total_fitted"] * 100
    )
    af_by_hsa = af_by_hsa.sort_values("jmp_san_pct")
    af_by_hsa.to_csv(OUTPUT_DIR / "attributable_fraction_by_hsa.csv", index=False)

    # Summary by infra tertile
    af_by_hsa["infra_tertile"] = pd.qcut(
        af_by_hsa["jmp_san_pct"], q=3, labels=["Low", "Medium", "High"]
    )
    af_tertile = af_by_hsa.groupby("infra_tertile", observed=True)["AF_pct"].mean()
    print(f"  Mean attributable fraction (precipitation) by infra quality tertile:")
    for tertile, af in af_tertile.items():
        print(f"    {tertile}: {af:.2f}%")

    # ---------------------------------------------------------------------------
    # 8. Plots
    # ---------------------------------------------------------------------------
    print("\n[8/9] Creating plots...")

    # --- Plot A: Stratified cumulative RR by infra quality tertile ---
    # Use interaction model: effective coef at infra_q = coef_cb + coef_int * (q_c)
    infra_tertile_vals = {
        "Low sanitation": np.percentile(infra_c, 17),
        "Medium sanitation": np.percentile(infra_c, 50),
        "High sanitation": np.percentile(infra_c, 83),
    }
    tertile_colors = {"Low sanitation": "#d62728", "Medium sanitation": "#ff7f0e",
                      "High sanitation": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    RR_cum = np.exp(cum_logRR)
    RR_lo  = np.exp(cum_logRR - 1.96 * cum_se)
    RR_hi  = np.exp(cum_logRR + 1.96 * cum_se)
    ax.plot(exp_grid, RR_cum, "k-", linewidth=2, alpha=0.4, label="Pooled average")
    ax.fill_between(exp_grid, RR_lo, RR_hi, alpha=0.08, color="black")

    for label, q_c_val in infra_tertile_vals.items():
        eff_coef = coef_cb + coef_int * q_c_val
        lr, se_ = cumulative_rr(eff_coef, vcov_cb, cb_meta_precip, exp_grid)
        san_pct = (q_c_val + infra.mean()) * 100
        ax.plot(exp_grid, np.exp(lr), linewidth=2, color=tertile_colors[label],
                label=f"{label} ({san_pct:.0f}%)")
        ax.fill_between(exp_grid, np.exp(lr - 1.96 * se_), np.exp(lr + 1.96 * se_),
                        alpha=0.12, color=tertile_colors[label])

    ax.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    ax.axvline(exp_ref75, color="gray", linewidth=0.7, linestyle="--",
               label=f"75th pct ({exp_ref75:.2f} mm/day)")
    ax.set_xlabel("Weekly mean precipitation (mm/day)")
    ax.set_ylabel("Cumulative RR (lags 0–3 weeks)")
    ax.set_title("Precipitation–Diarrhea: Cumulative Lag-Response\nby Sanitation Infrastructure Quality")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot B: RR surface (pooled) ---
    ax2 = axes[1]
    RR_surface = np.exp(logRR_surface)
    vmin = max(RR_surface.min(), 0.7)
    vmax = min(RR_surface.max(), 1.4)
    im = ax2.contourf(lag_grid, exp_grid, RR_surface, levels=20,
                       cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax2, label="RR")
    ax2.set_xlabel("Lag (weeks)")
    ax2.set_ylabel("Weekly mean precipitation (mm/day)")
    ax2.set_title("RR Surface: Precipitation × Lag (Pooled)")
    ax2.set_xticks(lag_grid)
    ax2.set_xticklabels(["Week 0", "Week 1", "Week 2", "Week 3"])
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cumulative_rr_precip.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: cumulative_rr_precip.png")

    # --- Plot C: Per-HSA meta-regression scatter ---
    if len(hsa_rr_df) >= 5:
        fig, ax = plt.subplots(figsize=(9, 6))
        # Correct error bars: asymmetric on RR scale (lower/upper based on log-scale SE)
        rr_vals   = hsa_rr_df["cum_RR_75pct"].values
        logRR_vals = hsa_rr_df["cum_logRR_75pct"].values
        se_vals   = hsa_rr_df["cum_se_75pct"].values
        err_lo    = rr_vals - np.exp(logRR_vals - 1.96 * se_vals)
        err_hi    = np.exp(logRR_vals + 1.96 * se_vals) - rr_vals

        sc = ax.scatter(hsa_rr_df["jmp_san_pct"], rr_vals,
                        s=hsa_rr_df["n_obs"] / 3,
                        c=hsa_rr_df["jmp_san_pct"],
                        cmap="RdYlGn", vmin=60, vmax=85,
                        alpha=0.85, zorder=3)
        ax.errorbar(hsa_rr_df["jmp_san_pct"], rr_vals,
                    yerr=[err_lo, err_hi],
                    fmt="none", color="gray", alpha=0.4, linewidth=0.8)

        x_line = np.linspace(hsa_rr_df["jmp_san_pct"].min() - 1,
                              hsa_rr_df["jmp_san_pct"].max() + 1, 100)
        y_line = np.exp(meta_res.params[0] + meta_res.params[1] * x_line)
        ax.plot(x_line, y_line, "k-", linewidth=2,
                label=f"WLS: β={slope:.4f}/ppt  p={slope_p:.3f}")

        for _, row in hsa_rr_df.iterrows():
            short = (row["hsa_id"].replace("_Hospital", "")
                     .replace("_Comprehensive_Center", " CC")
                     .replace("_Primary_Center", " PC").replace("_", " "))
            ax.annotate(short, (row["jmp_san_pct"], row["cum_RR_75pct"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=7)

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Safely managed sanitation, 2022 (%, JMP × census weights)")
        ax.set_ylabel(f"Cumulative RR at 75th pct precip ({exp_ref75:.2f} mm/day)")
        ax.set_title("Per-HSA Cumulative RR vs. Sanitation Quality\n"
                     "(point size ∝ n weeks; colour = sanitation %)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Sanitation %")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "meta_regression.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: meta_regression.png")

    # --- Plot D: Attributable fraction by HSA ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d62728" if row["jmp_san_pct"] < 67 else
              "#ff7f0e" if row["jmp_san_pct"] < 74 else "#2ca02c"
              for _, row in af_by_hsa.iterrows()]
    bars = ax.barh(af_by_hsa["hsa_id"].str.replace("_", " ").str.replace(" Hospital", ""),
                   af_by_hsa["AF_pct"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Attributable fraction: precipitation effect (%)")
    ax.set_title("% of Diarrheal Cases Attributable to Precipitation\n"
                 "(red=low sanitation, orange=medium, green=high)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "attributable_fraction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: attributable_fraction.png")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nModel comparison (all vs seasonal baseline):")
    for r in comp_rows:
        sig = "SIGNIFICANT" if r["p_value"] < 0.05 else "not significant"
        print(f"  {r['comparison']:<50s}  F={r['F']:.3f}  p={r['p_value']:.4f}  [{sig}]")

    print(f"\nPrecipitation cross-basis (cumulative RR at key percentiles):")
    for pct in [50, 75, 90, 95]:
        exp_val = np.percentile(Q_precip, pct)
        idx = np.argmin(np.abs(exp_grid - exp_val))
        rr = RR_cum[idx]
        lo = RR_lo[idx]
        hi = RR_hi[idx]
        print(f"  {pct}th pct ({exp_val:.3f} mm/day): RR={rr:.3f}  95%CI [{lo:.3f}, {hi:.3f}]")

    if slope is not None:
        print(f"\nMeta-regression (logRR ~ sanitation %):")
        print(f"  Slope: {slope:.5f} per pct point  (p={slope_p:.4f})")
        print(f"  Interpretation: {'better' if slope < 0 else 'worse'} sanitation associated "
              f"with {'lower' if slope < 0 else 'higher'} precipitation-diarrhea RR")

    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
