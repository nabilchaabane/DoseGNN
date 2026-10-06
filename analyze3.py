#!/usr/bin/env python3
"""
Analyse distance vs error from dist_error_*.csv files.

Usage:
    python analyse_distance.py dist_error_*.csv
    python analyse_distance.py --source-organ liver dist_error_*.csv

Each row is one supervoxel under one source organ. Metrics are per
(supervoxel, source-organ) prediction task unless noted.
"""

import sys
import glob
import pandas as pd
import numpy as np

NEAR_FAR_CUT     = 30.0
TRUST_FRACS      = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00]
TRUST_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DIST_BINS   = [0, 5, 10, 15, 20, 30, 50, 100, np.inf]
DIST_LABELS = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-50", "50-100", "100+"]


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


def binned_table(df):
    return (df.groupby("dist_bin", observed=True)
              .agg(n=("abs_error_decades", "count"),
                   mean_err=("abs_error_decades", "mean"),
                   median_err=("abs_error_decades", "median"),
                   mean_dist=("distance_cm", "mean"),
                   mean_log_dose=("true_log_dose", "mean"),
                   max_log_dose=("true_log_dose", "max"),
                   max_pred_log_dose=("pred_log_dose", "max"),
                   mean_trust=("trust", "mean"))
              .reset_index())


def per_organ_pivots(df):
    sub = df[df["source_organ"].isin(df["source_organ"].value_counts().index)]
    pivot = (sub.groupby(["source_organ", "dist_bin"], observed=True)
                ["abs_error_decades"].mean().unstack("dist_bin"))
    counts = (sub.groupby(["source_organ", "dist_bin"], observed=True)
                 ["abs_error_decades"].count().unstack("dist_bin"))
    return pivot, counts


def per_organ_pivots_by_score(df, score_col):
    edges = [0, 0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 1.00]
    labels_ascending = ["bottom 10%", "70-90%", "50-70%", "30-50%",
                        "20-30%", "10-20%", "5-10%", "top 0-5%"]
    pieces = []
    for ph, g in df.groupby("phantom", observed=True):
        g = g.dropna(subset=[score_col]).copy()
        if len(g) == 0 or g[score_col].nunique() < 2:
            continue
        try:
            g["score_bin"] = pd.qcut(g[score_col], q=edges, labels=labels_ascending)
        except ValueError:
            # weight distribution too concentrated; fall back to rank-based binning
            g["score_bin"] = pd.qcut(
                g[score_col].rank(method="first"),
                q=edges, labels=labels_ascending,
            )
        pieces.append(g)
    if not pieces:
        return pd.DataFrame(), pd.DataFrame()
    allb = pd.concat(pieces, ignore_index=True)
    pivot = (allb.groupby(["source_organ", "score_bin"], observed=True)
                 ["abs_error_decades"].mean().unstack("score_bin"))
    counts = (allb.groupby(["source_organ", "score_bin"], observed=True)
                  ["abs_error_decades"].count().unstack("score_bin"))
    col_order = ["top 0-5%", "5-10%", "10-20%", "20-30%",
                 "30-50%", "50-70%", "70-90%", "bottom 10%"]
    return pivot.reindex(columns=col_order), counts.reindex(columns=col_order)


def coverage_quantile(df):
    rows = []
    for ph, g in df.groupby("phantom", observed=True):
        g = g.dropna(subset=["trust"])
        if len(g) == 0:
            continue
        for f in TRUST_FRACS:
            if f >= 1.0:
                kt = kd = kw = pd.Series(True, index=g.index)
            else:
                kt = g["trust"] >= g["trust"].quantile(1.0 - f)
                kd = g["distance_cm"] <= g["distance_cm"].quantile(f)
                kw = g["weight"] >= g["weight"].quantile(1.0 - f)
            rows.append({
                "phantom": ph, "top_frac": f,
                "frac_kept_trust": float(kt.mean()),
                "mae_trust":  float(g.loc[kt, "abs_error_decades"].mean()),
                "frac_kept_dist":  float(kd.mean()),
                "mae_dist":   float(g.loc[kd, "abs_error_decades"].mean()),
                "frac_kept_weight": float(kw.mean()),
                "mae_weight": float(g.loc[kw, "abs_error_decades"].mean()),
            })
    cov = pd.DataFrame(rows)
    if cov.empty:
        return cov
    out = (cov.groupby("top_frac")
              .agg(frac_kept_trust=("frac_kept_trust", "mean"),
                   mae_trust=("mae_trust", "mean"),
                   frac_kept_dist=("frac_kept_dist", "mean"),
                   mae_dist=("mae_dist", "mean"),
                   frac_kept_weight=("frac_kept_weight", "mean"),
                   mae_weight=("mae_weight", "mean"),
                   n_phantoms=("phantom", "nunique"))
              .reset_index())
    out["gain_trust_vs_dist"]   = out["mae_dist"]   - out["mae_trust"]
    out["gain_trust_vs_weight"] = out["mae_weight"] - out["mae_trust"]
    return out


def coverage_absolute(df):
    rows = []
    for ph, g in df.groupby("phantom", observed=True):
        g = g.dropna(subset=["trust"])
        if len(g) == 0:
            continue
        for p in TRUST_THRESHOLDS:
            keep = g["trust"] > p
            rows.append({
                "phantom": ph, "threshold": p,
                "frac_kept": float(keep.mean()),
                "mae_kept":  float(g.loc[keep, "abs_error_decades"].mean()) if keep.any() else np.nan,
                "mae_kept_median": float(g.loc[keep, "abs_error_decades"].median()) if keep.any() else np.nan,
            })
    cov = pd.DataFrame(rows)
    if cov.empty:
        return cov
    return (cov.groupby("threshold")
               .agg(frac_kept=("frac_kept", "mean"),
                    mae_kept=("mae_kept", "mean"),
                    mae_kept_median=("mae_kept_median", "mean"),
                    n_phantoms=("phantom", "nunique"))
               .reset_index())


def main(patterns, write_sample=False, source_organ=None):
    df_raw = load_files(patterns)

    if source_organ is not None:
        before = len(df_raw)
        df_raw = df_raw[df_raw["source_organ"] == source_organ].copy()
        print(f"\n--- Filtering to source_organ='{source_organ}' ---")
        print(f"  rows before: {before:>12,}")
        print(f"  rows after:  {len(df_raw):>12,}")
        if len(df_raw) == 0:
            avail = sorted(df_raw["source_organ"].unique()) if False else "see files"
            raise SystemExit(f"No rows for source_organ='{source_organ}'.")

    df, n_raw = prep(df_raw)
    print_filter_summary(df, n_raw)

    corr = df["distance_cm"].corr(df["abs_error_decades"])
    print(f"\nPearson correlation (distance vs abs error): {corr:.3f}")

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

    print("\n--- Error vs distance (binned) ---")
    binned = binned_table(df)
    print(binned.to_string(index=False, float_format="%.3f"))
    binned.to_csv("distance_vs_error_binned.csv", index=False, float_format="%.4f")

    print("\n--- Mean error by distance bin and source organ ---")
    pivot, counts = per_organ_pivots(df)
    print(pivot.round(3).to_string())
    print("\n--- Mean fold error by distance bin and source organ ---")
    print((10 ** pivot).round(3).to_string())
    pivot.to_csv("source_organ_vs_distance_mean_err.csv", float_format="%.4f")
    counts.to_csv("source_organ_vs_distance_counts.csv")

    print("\n--- Mean error by source organ and TRUST bin ---")
    tp, tc = per_organ_pivots_by_score(df, "trust")
    if not tp.empty:
        tp, tc = tp.reindex(pivot.index), tc.reindex(pivot.index)
        print(tp.round(3).to_string())
        tp.to_csv("source_organ_vs_trustbin_mean_err.csv", float_format="%.4f")
        tc.to_csv("source_organ_vs_trustbin_counts.csv")

    print("\n--- Mean error by source organ and WEIGHT bin ---")
    wp, wc = per_organ_pivots_by_score(df, "weight")
    if not wp.empty:
        wp, wc = wp.reindex(pivot.index), wc.reindex(pivot.index)
        print(wp.round(3).to_string())
        wp.to_csv("source_organ_vs_weightbin_mean_err.csv", float_format="%.4f")
        wc.to_csv("source_organ_vs_weightbin_counts.csv")

    print("\n--- Coverage vs accuracy (quantile) ---")
    covq = coverage_quantile(df)
    if covq.empty:
        print("  (no trust column)")
    else:
        print(covq.to_string(index=False, float_format="%.4f"))
        covq.to_csv("trust_coverage_quantile.csv", index=False, float_format="%.6f")

    print("\n--- Coverage vs accuracy (absolute thresholds) ---")
    cova = coverage_absolute(df)
    if cova.empty:
        print("  (no trust column)")
    else:
        print(cova.to_string(index=False, float_format="%.4f"))
        cova.to_csv("trust_coverage_absolute.csv", index=False, float_format="%.6f")

    if write_sample:
        sample = df.sample(n=min(50000, len(df)), random_state=42)
        sample[["distance_cm", "abs_error_decades", "source_organ",
                "target_organ", "trust"]].to_csv(
                    "distance_vs_error_sample.csv", index=False, float_format="%.4f")


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
    write_sample = "--write-sample" in rest
    patterns = [a for a in rest if not a.startswith("--")]
    if not patterns:
        print("Usage: python analyse_distance.py [--write-sample] "
              "[--source-organ NAME] dist_error_*.csv")
        sys.exit(1)
    main(patterns, write_sample=write_sample, source_organ=source_organ)