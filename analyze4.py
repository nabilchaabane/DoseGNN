#!/usr/bin/env python3
"""
Analyse dose prediction error and trust-score reliability from
dist_error_*.csv files.

Usage:
    python analyse_distance.py dist_error_*.csv
    python analyse_distance.py --source-organ liver dist_error_*.csv

Each row is one supervoxel under one source organ. Metrics are per
(supervoxel, source-organ) prediction task unless noted.

Framing:
    The MC inverse-variance weight w = 1/sigma^2 is the physical ground
    truth for measurement reliability. The trust head is evaluated as a
    learned surrogate for that signal, computable from features at
    inference time without the MC uncertainty map.

DVH trust metrics:
    For each endpoint X in {D95, D50, D2}, the code reports two trust
    summaries of the +/- trust_band volume window around the supervoxel
    that defines the endpoint:

      trust_at_X  : volume-weighted mean trust over the band
      gap_X       : pair_mean_trust - trust_at_X
"""

import sys
import glob
import pandas as pd
import numpy as np


# ------------------------------------------------------------------ constants
NEAR_FAR_CUT = 30.0
TRUST_FRACS  = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.00]
DIST_BINS    = [0, 5, 10, 15, 20, 30, 40, 50, 100, np.inf]
DIST_LABELS  = ["0-5", "5-10", "10-15", "15-20", "20-30",
                "30-40", "40-50", "50-100", "100+"]

CLINICAL_ORGANS = {
    "liver", "lungs", "kidneys", "pancreas", "spleen", "stomach",
    "esophagus", "heart", "brain", "thyroid", "prostate", "testes",
    "ovaries", "uterus", "thymus", "adrenals", "gall_bladder",
    "urinary_bladder", "rectosigmoid", "left_colon", "right_colon",
    "small_intestine", "salivary_glands", "breast_glandular",
    "spinal_cord",
}


# ------------------------------------------------------------------ loading
def load_files(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    print(files)
    if not files:
        raise SystemExit("No files matched.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {len(files)} file(s), {len(df):,} rows")
    return df


def prep(df):
    n_raw = len(df)
    df = df.dropna(subset=["distance_cm", "abs_error_decades"]).copy()
    df["censored"] = df["censored"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df["trust"] = (1.0 / (1.0 + np.exp(-df["trust_logit"]))
                   if "trust_logit" in df.columns else np.nan)
    df["log_w"] = np.log10(df["weight"].clip(lower=1e-12))
    df["dist_bin"] = pd.cut(df["distance_cm"], bins=DIST_BINS,
                            labels=DIST_LABELS, right=False)
    df["is_far"] = df["distance_cm"] >= NEAR_FAR_CUT
    return df, n_raw


def print_filter_summary(df, n_raw):
    n = len(df)
    n_cens = int(df["censored"].sum())
    print("\n--- Filter summary ---")
    print(f"  raw rows       {n_raw:>12,}")
    print(f"  after dropna   {n:>12,}")
    print(f"  censored       {n_cens:>12,}  ({100*n_cens/max(n,1):5.1f}%)")
    print(f"  non-censored   {n-n_cens:>12,}  ({100*(n-n_cens)/max(n,1):5.1f}%)")


# ------------------------------------------------------------------ helpers
def _stats(g):
    n = len(g)
    if n == 0:
        return pd.Series({"n": 0, "mae": np.nan, "mae_w": np.nan, "median": np.nan})
    e, w = g["abs_error_decades"], g["weight"]
    mae_w = float((w * e).sum() / w.sum()) if w.sum() > 0 else np.nan
    return pd.Series({"n": n, "mae": float(e.mean()), "mae_w": mae_w,
                      "median": float(e.median())})


def per_phantom_table(df):
    rows = []
    for ph, g in df.groupby("phantom", observed=True):
        row = {"phantom": ph}
        row.update({f"all_{k}":    v for k, v in _stats(g).items()})
        row.update({f"nocens_{k}": v for k, v in _stats(g[~g["censored"]]).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def near_far_table(df):
    rows = []
    for tag, g in [("near (<30cm)", df[~df["is_far"]]),
                   ("far  (>=30cm)", df[df["is_far"]])]:
        s = _stats(g[~g["censored"]])
        rows.append({"region": tag, **s.to_dict()})
    return pd.DataFrame(rows)


def per_phantom_correlation_table(dvh, endpoints=("D95", "D50", "D2"),
                                   min_pairs=5):
    """
    One row per phantom: Spearman rho between trust_at_X and X_err_dec,
    for each endpoint X. Phantoms with fewer than `min_pairs` rows for an
    endpoint get NaN for that column.
    """
    from scipy.stats import spearmanr
    rows = []
    for ph, g in dvh.groupby("phantom", observed=True):
        row = {"phantom": ph, "n_pairs": len(g)}
        for X in endpoints:
            sub = g.dropna(subset=[f"trust_at_{X}", f"{X}_err_dec"])
            if len(sub) >= min_pairs and sub[f"trust_at_{X}"].nunique() > 1:
                r, _ = spearmanr(sub[f"trust_at_{X}"], sub[f"{X}_err_dec"])
                row[f"rho_{X}"] = float(r) if np.isfinite(r) else np.nan
            else:
                row[f"rho_{X}"] = np.nan
        rows.append(row)
    tab = pd.DataFrame(rows).sort_values("rho_D95")
    summary = {"phantom": "MEDIAN"}
    for X in endpoints:
        summary[f"rho_{X}"] = tab[f"rho_{X}"].median()
    summary["n_pairs"] = tab["n_pairs"].sum()
    tab = pd.concat([tab, pd.DataFrame([summary])], ignore_index=True)
    return tab


def within_phantom_quartiles(dvh, col, min_per_group=8):
    """
    Assign quartile labels (0..3) to `col`, computed within each phantom.
    Phantoms with fewer than `min_per_group` rows get NaN.
    """
    def _assign(s):
        if len(s) < min_per_group or s.nunique() < 2:
            return pd.Series(np.nan, index=s.index)
        try:
            return pd.qcut(s.rank(method="first"), q=4, labels=False)
        except ValueError:
            return pd.Series(np.nan, index=s.index)
    q = dvh.groupby("phantom", observed=True)[col].transform(_assign)
    return q


def within_phantom_correlations(dvh, col, err_col, min_pairs=5):
    """Return list of per-phantom Spearman rho between `col` and `err_col`."""
    from scipy.stats import spearmanr
    rhos = []
    for ph, g in dvh.groupby("phantom", observed=True):
        g = g.dropna(subset=[col, err_col])
        if len(g) >= min_pairs and g[col].nunique() > 1:
            r, _ = spearmanr(g[col], g[err_col])
            if np.isfinite(r):
                rhos.append(r)
    return rhos


# ------------------------------------------------------------------ DVH
def dvh_metrics(df, min_n=20, dose_floor_log=-10.0, trust_band=0.05,
                min_n_band=3):
    """
    Volume-weighted DVH metrics per (phantom, source, target) pair.

    D95 = near-min (5th volume percentile)
    D50 = median   (50th volume percentile)
    D2  = near-max (98th volume percentile)

    For each endpoint we report two trust summaries of the +/- trust_band
    volume window around the endpoint-defining supervoxel:

        trust_at_X : volume-weighted mean trust over the band
        gap_X      : pair_mean_trust - trust_at_X

    A pair is skipped if its target organ is not clinically relevant, if
    it has fewer than `min_n` non-censored supervoxels, or if its maximum
    true log-dose is below `dose_floor_log`.
    """
    if "volume_cm3" not in df.columns:
        print("  [skip] no volume_cm3 column; rerun save_distance_error_csv")
        return pd.DataFrame()

    df = df.dropna(subset=["volume_cm3", "true_log_dose", "pred_log_dose"]).copy()
    df = df[df["volume_cm3"] > 0]
    df = df[df["target_organ"].isin(CLINICAL_ORGANS)]
    df = df[~df["censored"]]
    def wpercentile_with_trust(values, weights, trust, q, band, min_n_band):
        """
        Returns (percentile, band_mean_trust, trust_at_defining_sv).
        """
        order = np.argsort(values)
        v = np.asarray(values)[order]
        w = np.asarray(weights)[order]
        t = np.asarray(trust)[order]
        cw = np.cumsum(w)
        if cw[-1] <= 0:
            return np.nan, np.nan, np.nan
        total = cw[-1]
        target = q / 100.0 * total
        idx = min(int(np.searchsorted(cw, target, side="left")), len(v) - 1)
        pct = float(v[idx])
        trust_defining = float(t[idx])

        # band mean (unchanged, for comparison)
        eff_band = band
        n = len(v)
        if n > 0 and n * 2 * eff_band < min_n_band:
            eff_band = min(0.5, min_n_band / (2.0 * n))
        lo = target - eff_band * total
        hi = target + eff_band * total
        mask = (cw >= lo) & (cw <= hi)
        if mask.any():
            trust_band_mean = float((t[mask] * w[mask]).sum() / w[mask].sum())
        else:
            trust_band_mean = trust_defining
        return pct, trust_band_mean, trust_defining
    
    rows = []
    for (ph, src, tgt), g in df.groupby(
            ["phantom", "source_organ", "target_organ"], observed=True):
        if len(g) < min_n:
            continue
        if g["true_log_dose"].max() < dose_floor_log:
            continue

        vol   = g["volume_cm3"].values
        tlog  = g["true_log_dose"].values
        plog  = g["pred_log_dose"].values
        trust = g["trust"].values

        # pair-level trust statistics (volume-weighted)
        pair_vol = float(vol.sum())
        if pair_vol > 0:
            pair_mean_trust = float((trust * vol).sum() / pair_vol)
        else:
            pair_mean_trust = np.nan

        D95_t, tr_D95, tr_sv_D95 = wpercentile_with_trust(tlog, vol, trust, 5,  trust_band, min_n_band)
        D50_t, tr_D50, tr_sv_D50 = wpercentile_with_trust(tlog, vol, trust, 50, trust_band, min_n_band)
        D2_t,  tr_D2,  tr_sv_D2  = wpercentile_with_trust(tlog, vol, trust, 98, trust_band, min_n_band)
        D95_p, _, _ = wpercentile_with_trust(plog, vol, trust, 5,  trust_band, min_n_band)
        D50_p, _, _ = wpercentile_with_trust(plog, vol, trust, 50, trust_band, min_n_band)
        D2_p,  _, _ = wpercentile_with_trust(plog, vol, trust, 98, trust_band, min_n_band)

        rows.append({
            "phantom": ph,
            "source_organ": src,
            "target_organ": tgt,
            "n": len(g),
            "volume_cm3": pair_vol,
            "pair_mean_trust": pair_mean_trust,

            "D95_true_log": D95_t,
            "D95_pred_log": D95_p,
            "D50_true_log": D50_t,
            "D50_pred_log": D50_p,
            "D2_true_log":  D2_t,
            "D2_pred_log":  D2_p,
            "D95_err_dec":  abs(D95_p - D95_t),
            "D50_err_dec":  abs(D50_p - D50_t),
            "D2_err_dec":   abs(D2_p  - D2_t),

            "trust_at_D95":     tr_D95,
            "trust_at_D50":     tr_D50,
            "trust_at_D2":      tr_D2,
            "trust_defining_D95": tr_sv_D95,
            "trust_defining_D50": tr_sv_D50,
            "trust_defining_D2":  tr_sv_D2,

            "gap_D95": pair_mean_trust - tr_D95,
            "gap_D50": pair_mean_trust - tr_D50,
            "gap_D2":  pair_mean_trust - tr_D2,
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ trust
def trust_weight_agreement(df):
    """Correlation between trust and log10(weight), over ALL nodes."""
    from scipy.stats import spearmanr, pearsonr

    g = df.dropna(subset=["trust", "log_w"])
    r_p, _  = pearsonr(g["trust"],       g["log_w"])
    r_s, _  = spearmanr(g["trust"],      g["log_w"])
    r_pl, _ = pearsonr(g["trust_logit"], g["log_w"])
    return {
        "n": int(len(g)),
        "pearson_trust_vs_logw":  float(r_p),
        "spearman_trust_vs_logw": float(r_s),
        "pearson_logit_vs_logw":  float(r_pl),
    }


def trust_by_weight_decile(df):
    pieces = []
    for ph, g in df.groupby("phantom", observed=True):
        g = g.dropna(subset=["trust", "weight"]).copy()
        if len(g) == 0 or g["weight"].nunique() < 2:
            continue
        try:
            g["w_decile"] = pd.qcut(g["weight"].rank(method="first"),
                                    q=10, labels=False)
        except ValueError:
            continue
        pieces.append(g)
    if not pieces:
        return pd.DataFrame()
    allb = pd.concat(pieces, ignore_index=True)
    tab = (allb.groupby("w_decile", observed=True)
               .agg(n=("trust", "count"),
                    mean_weight=("weight", "mean"),
                    mean_log_w=("log_w", "mean"),
                    mean_trust=("trust", "mean"),
                    median_trust=("trust", "median"),
                    mean_err=("abs_error_decades", "mean"),
                    censored_frac=("censored", "mean"))
               .reset_index())
    return tab


def trust_weight_quartile_table(df):
    pieces = []
    for ph, g in df.groupby("phantom", observed=True):
        g = g.dropna(subset=["trust", "weight"]).copy()
        if len(g) == 0:
            continue
        g["w_q"] = pd.qcut(g["weight"].rank(method="first"),
                           q=4, labels=["W1", "W2", "W3", "W4"])
        g["t_q"] = pd.qcut(g["trust"].rank(method="first"),
                           q=4, labels=["T1", "T2", "T3", "T4"])
        pieces.append(g)
    if not pieces:
        return pd.DataFrame()
    allb = pd.concat(pieces, ignore_index=True)
    tab = (allb.groupby(["w_q", "t_q"], observed=True)
               .agg(n=("abs_error_decades", "count"),
                    frac=("abs_error_decades", lambda s: len(s) / len(allb)),
                    mae=("abs_error_decades", "mean"),
                    median=("abs_error_decades", "median"),
                    censored_frac=("censored", "mean"),
                    mean_trust=("trust", "mean"),
                    mean_log_w=("log_w", "mean"))
               .reset_index())
    return tab


def coverage_quantile(df):
    rows = []
    for ph, g in df.groupby("phantom", observed=True):
        g = g.dropna(subset=["trust"])
        if len(g) == 0:
            continue
        n = len(g)
        r_trust  = g["trust"].rank(method="first", ascending=False)
        r_weight = g["weight"].rank(method="first", ascending=False)
        for f in TRUST_FRACS:
            k = n if f >= 1.0 else int(round(f * n))
            kt = r_trust  <= k
            kw = r_weight <= k
            rows.append({
                "phantom": ph, "top_frac": f,
                "frac_kept_trust": float(kt.mean()),
                "mae_trust":  float(g.loc[kt, "abs_error_decades"].mean()),
                "frac_kept_weight": float(kw.mean()),
                "mae_weight": float(g.loc[kw, "abs_error_decades"].mean()),
            })
    cov = pd.DataFrame(rows)
    if cov.empty:
        return cov
    out = (cov.groupby("top_frac")
              .agg(frac_kept_trust=("frac_kept_trust", "mean"),
                   mae_trust=("mae_trust", "mean"),
                   frac_kept_weight=("frac_kept_weight", "mean"),
                   mae_weight=("mae_weight", "mean"),
                   n_phantoms=("phantom", "nunique"))
              .reset_index())
    return out


# ------------------------------------------------------------------ main
def main(patterns, source_organ=None):
    df_raw = load_files(patterns)

    if source_organ is not None:
        before = len(df_raw)
        df_raw = df_raw[df_raw["source_organ"] == source_organ].copy()
        print(f"\n--- Filtering to source_organ='{source_organ}' ---")
        print(f"  rows before: {before:>12,}")
        print(f"  rows after:  {len(df_raw):>12,}")
        if len(df_raw) == 0:
            raise SystemExit(f"No rows for source_organ='{source_organ}'.")

    df, n_raw = prep(df_raw)
    print_filter_summary(df, n_raw)

    # ---- 1. Model accuracy ----
    print("\n--- Per-phantom MAE ---")
    per_ph = per_phantom_table(df)
    cols = ["phantom", "all_n", "all_mae", "all_mae_w",
            "nocens_n", "nocens_mae", "nocens_mae_w"]
    print(per_ph[cols].to_string(index=False, float_format="%.4f"))
    print("\n--- Mean across phantoms ---")
    for tag, label in [("all", "all nodes"), ("nocens", "non-censored")]:
        print(f"  {label:24s}  "
              f"MAE {per_ph[f'{tag}_mae'].mean():.4f}  "
              f"MAE_w {per_ph[f'{tag}_mae_w'].mean():.4f}  "
              f"fold {10**per_ph[f'{tag}_mae'].mean():.3f}")
    per_ph.to_csv("mae_per_phantom.csv", index=False, float_format="%.6f")

    print("\n--- Near vs far (non-censored) ---")
    nf = near_far_table(df)
    print(nf.to_string(index=False, float_format="%.4f"))
    nf.to_csv("mae_near_far.csv", index=False, float_format="%.6f")

    # ---- 2. DVH ----
    print("\n--- DVH metrics (volume-weighted, clinically relevant organs) ---")
    dvh = dvh_metrics(df)
    if dvh.empty:
        print("  (no volume_cm3 column, or no organ pair with >= 20 supervoxels)")
    else:
        summary = (dvh.groupby(["source_organ", "target_organ"])
                      .agg(n_phantoms=("phantom", "nunique"),
                           n_tasks=("n", "sum"),
                           D95_err_dec_median=("D95_err_dec", "median"),
                           D50_err_dec_median=("D50_err_dec", "median"),
                           D2_err_dec_median=("D2_err_dec", "median"),
                           trust_at_D95_median=("trust_at_D95", "median"),
                           trust_at_D50_median=("trust_at_D50", "median"),
                           trust_at_D2_median=("trust_at_D2", "median"))
                      .reset_index()
                      .sort_values("D95_err_dec_median", ascending=False))

        summary_print = summary[summary["n_phantoms"] >= 3]
        print(f"\n  DVH summary (pairs present on >= 3 phantoms; "
              f"{len(summary) - len(summary_print)} pairs with fewer "
              f"phantoms written to CSV only):")
        print(summary_print.to_string(index=False, float_format="%.4f"))

        print("\n  overall medians across (source, target) pairs:")
        for c in ["D95_err_dec", "D50_err_dec", "D2_err_dec"]:
            med = dvh[c].median()
            print(f"    {c:14s}  {med:.4f} decades  (x{10**med:.3f})")

        # --- correlations ---
        print("\n  Spearman correlation with DVH error (within-phantom):")
        print("    trust_at_X = mean trust over band around endpoint")
        print("    gap_X      = pair_mean_trust - trust_at_X")
        
        for X in ["D95", "D50", "D2"]:
            print(f"    {X}:")
            for col in [f"trust_at_{X}", f"trust_defining_{X}"]:
                rhos = within_phantom_correlations(
                    dvh, col, f"{X}_err_dec", min_pairs=5)
                if rhos:
                    a = np.array(rhos)
                    print(f"      {col:20s}  median {np.median(a):+.3f}  "
                          f"mean {np.mean(a):+.3f}  "
                          f"[{a.min():+.3f},{a.max():+.3f}]  n={len(a)}")

        # --- per-phantom correlation table (D95 only) ---
        print("\n  Per-phantom Spearman rho vs D95 error:")
        ppc_rows = []
        for ph, g in dvh.groupby("phantom", observed=True):
            row = {"phantom": ph, "n_pairs": len(g)}
            for col in ["trust_at_D95", "gap_D95"]:
                rhos = within_phantom_correlations(
                    g.assign(**{"__dummy__": 0}), col, "D95_err_dec",
                    min_pairs=5)
                row[col] = rhos[0] if rhos else np.nan
            ppc_rows.append(row)
        ppc = pd.DataFrame(ppc_rows).sort_values("gap_D95")
        print(ppc.to_string(index=False, float_format="%.3f"))
        ppc.to_csv("trust_dvh_correlation_per_phantom.csv",
                   index=False, float_format="%.6f")

        # --- stratification by trust_at_D95 ---
        print("\n  DVH error stratified by trust_at_D95 (within-phantom quartiles):")
        dd = dvh.dropna(subset=["trust_at_D95"]).copy()
        dd["trust_q"] = within_phantom_quartiles(dd, "trust_at_D95")
        dd = dd.dropna(subset=["trust_q"])
        if not dd.empty:
            dd["trust_q"] = dd["trust_q"].astype(int).map(
                {0: "Q1 (low)", 1: "Q2", 2: "Q3", 3: "Q4 (high)"})
            strat = (dd.groupby("trust_q", observed=True)
                       .agg(n=("D95_err_dec", "count"),
                            n_phantoms=("phantom", "nunique"),
                            trust_mean=("trust_at_D95", "mean"),
                            D95_err_med=("D95_err_dec", "median"),
                            D50_err_med=("D50_err_dec", "median"),
                            D2_err_med=("D2_err_dec", "median"))
                       .reindex(["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
                       .reset_index())
            print(strat.to_string(index=False, float_format="%.4f"))

        dvh.to_csv("dvh_per_phantom.csv", index=False, float_format="%.6f")
        summary.to_csv("dvh_summary.csv", index=False, float_format="%.6f")
        print("  saved dvh_per_phantom.csv, dvh_summary.csv")

    # ---- 3. Trust vs weight ----
    print("\n--- Trust vs weight agreement (all nodes) ---")
    tw = trust_weight_agreement(df)
    for k, v in tw.items():
        print(f"  {k:24s}  {v:.4f}" if isinstance(v, float)
              else f"  {k:24s}  {v}")
    pd.DataFrame([tw]).to_csv("trust_weight_agreement.csv",
                              index=False, float_format="%.6f")

    print("\n--- Trust by weight decile ---")
    dec = trust_by_weight_decile(df)
    if dec.empty:
        print("  (could not bin)")
    else:
        print(dec.to_string(index=False, float_format="%.4f"))
        dec.to_csv("trust_by_weight_decile.csv",
                   index=False, float_format="%.6f")

    print("\n--- Trust x weight quartile table (4x4) ---")
    ct = trust_weight_quartile_table(df)
    print(ct.to_string(index=False, float_format="%.4f"))
    ct.to_csv("trust_weight_quartile.csv", index=False, float_format="%.6f")

    # ---- 4. Coverage-accuracy ----
    print("\n--- Coverage vs accuracy (top fraction by trust vs weight) ---")
    covq = coverage_quantile(df)
    if covq.empty:
        print("  (no trust column)")
    else:
        print(covq.to_string(index=False, float_format="%.4f"))
        covq.to_csv("trust_coverage_quantile.csv",
                    index=False, float_format="%.6f")


if __name__ == "__main__":
    args = sys.argv[1:]
    source_organ = None
    rest = []
    i = 0
    while i < len(args):
        if args[i] == "--source-organ":
            source_organ = args[i + 1]; i += 2; continue
        if args[i].startswith("--source-organ="):
            source_organ = args[i].split("=", 1)[1]; i += 1; continue
        rest.append(args[i]); i += 1
    patterns = [a for a in rest if not a.startswith("--")]
    if not patterns:
        print("Usage: python analyse_distance.py [--source-organ NAME] "
              "dist_error_*.csv")
        sys.exit(1)
    main(patterns, source_organ=source_organ)