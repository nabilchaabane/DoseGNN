#!/usr/bin/env python3
"""
Analyse distance vs error from one or more dist_error_*.csv files.

Usage:
    python analyse_distance.py dist_error_10f125040_arms_highres.csv
    python analyse_distance.py dist_error_*.csv          # multiple files
    python analyse_distance.py --write-sample dist_error_*.csv

Assumes CSVs written by train_gnn.save_distance_error_csv contain:
    phantom, source_organ, target_organ, target_id, distance_cm,
    abs_error_decades, true_log_dose, pred_log_dose, trust_logit,
    weight, censored
"""

import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------------ constants
NEAR_FAR_CUT     = 30.0                       # cm
TRUST_FRACS      = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00]
TRUST_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DIST_BINS   = [0, 5, 10, 15, 20, 30, 50, 100, np.inf]
DIST_LABELS = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-50", "50-100", "100+"]


# ------------------------------------------------------------------ loading
def load_files(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    print(files)
    if not files:
        raise SystemExit("No files matched.")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(files)} file(s), {len(df):,} rows")
    return df


def prep(df):
    """Clean, tag, and derive all columns used downstream."""
    n_raw = len(df)
    df = df.dropna(subset=["distance_cm", "abs_error_decades"]).copy()

    # dtype-agnostic censored parse
    df["censored"] = df["censored"].astype(str).str.lower().isin(["true", "1", "1.0"])

    # trust probability from logit
    if "trust_logit" in df.columns:
        df["trust"] = 1.0 / (1.0 + np.exp(-df["trust_logit"]))
    else:
        df["trust"] = np.nan

    # distance bins and near/far flag
    df["dist_bin"] = pd.cut(df["distance_cm"], bins=DIST_BINS,
                            labels=DIST_LABELS, right=False)
    df["is_far"]   = df["distance_cm"] >= NEAR_FAR_CUT
    return df, n_raw


def print_filter_summary(df, n_raw):
    n = len(df)
    n_cens   = int(df["censored"].sum())
    n_nocens = int((~df["censored"]).sum())
    print("\n--- Filter summary ---")
    print(f"  raw rows                {n_raw:>12,}")
    print(f"  after dropna            {n:>12,}")
    print(f"  censored                {n_cens:>12,}  ({100*n_cens/max(n,1):5.1f}%)")
    print(f"  non-censored            {n_nocens:>12,}  ({100*n_nocens/max(n,1):5.1f}%)")


# ------------------------------------------------------------------ helpers
def _stats(g):
    """Unweighted and weighted MAE, median, and n for a subgroup."""
    n = len(g)
    if n == 0:
        return pd.Series({"n": 0, "mae": np.nan, "mae_w": np.nan, "median": np.nan})
    e = g["abs_error_decades"]
    w = g["weight"]
    mae   = float(e.mean())
    mae_w = float((w * e).sum() / w.sum()) if w.sum() > 0 else np.nan
    return pd.Series({"n": n, "mae": mae, "mae_w": mae_w,
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
    """Non-censored only, near vs far."""
    rows = []
    for tag, g in [("near (<30cm)",  df[~df["is_far"]]),
                   ("far  (>=30cm)", df[ df["is_far"]])]:
        s = _stats(g[~g["censored"]])
        rows.append({"region": tag, **s.to_dict()})
    return pd.DataFrame(rows)


def binned_table(df):
    return (df.groupby("dist_bin", observed=True)
              .agg(
                  n=("abs_error_decades", "count"),
                  mean_err=("abs_error_decades", "mean"),
                  median_err=("abs_error_decades", "median"),
                  mean_dist=("distance_cm", "mean"),
                  mean_log_dose=("true_log_dose", "mean"),
                  max_log_dose=("true_log_dose", "max"),
                  max_pred_log_dose=("pred_log_dose", "max"),
                  mean_trust=("trust", "mean"),
              )
              .reset_index())

def per_organ_pivots_trust_weight(df):
    """Same shape as per_organ_pivots, but for mean trust and mean weight."""
    sub = df[df["source_organ"].isin(df["source_organ"].value_counts().index)]
    trust_pivot = (sub.groupby(["source_organ", "dist_bin"], observed=True)
                      ["trust"].mean().unstack("dist_bin"))
    weight_pivot = (sub.groupby(["source_organ", "dist_bin"], observed=True)
                       ["weight"].mean().unstack("dist_bin"))
    return trust_pivot, weight_pivot


def per_organ_pivots(df):
    sub = df[df["source_organ"].isin(df["source_organ"].value_counts().index)]
    pivot = (sub.groupby(["source_organ", "dist_bin"], observed=True)
                ["abs_error_decades"].mean().unstack("dist_bin"))
    counts = (sub.groupby(["source_organ", "dist_bin"], observed=True)
                 ["abs_error_decades"].count().unstack("dist_bin"))
    return pivot, counts

def per_organ_pivots_by_score(df, score_col):
    """
    Bin nodes by a per-phantom quantile of `score_col` (trust or weight).
    Higher score = more trustworthy/well-measured.
    Columns are labeled top-first: "top 0-5%", ..., "bottom 10%".
    """
    # ascending quantile edges → first bin is the LOWEST score
    edges = [0, 0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 1.00]
    labels_ascending = [
        "bottom 10%",   # lowest 10% of scores
        "70-90%",
        "50-70%",
        "30-50%",
        "20-30%",
        "10-20%",
        "5-10%",
        "top 0-5%",     # highest 5% of scores
    ]

    pieces = []
    for ph, g in df.groupby("phantom", observed=True):
        g = g.dropna(subset=[score_col]).copy()
        if len(g) == 0:
            continue
        # qcut on the raw score, with ascending labels
        # this requires the score to be non-constant; if it is, drop the phantom
        if g[score_col].nunique() < 2:
            continue
        g["score_bin"] = pd.qcut(g[score_col], q=edges, labels=labels_ascending)
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
    pivot  = pivot.reindex(columns=col_order)
    counts = counts.reindex(columns=col_order)
    return pivot, counts

def draw_heatmap(pivot, counts, min_n=20,
                 out="source_organ_vs_distance_heatmap.png"):
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]
    counts = counts.loc[row_order]
    masked = pivot.where(counts >= min_n)

    fig, ax = plt.subplots(figsize=(11, max(6, 0.32 * len(pivot))))
    sns.heatmap(masked, ax=ax, cmap="viridis", annot=True, fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Mean |error| (decades)"},
                mask=masked.isna())
    ax.set_xlabel("Distance bin (cm)")
    ax.set_ylabel("Source organ")
    ax.set_title(f"Mean absolute error (decades): source organ vs distance\n"
                 f"(cells with n < {min_n} masked)")
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap saved to {out}")
    return pivot, counts


# ------------------------------------------------------------------ trust
def coverage_quantile(df):
    """Per phantom, rank nodes by trust, distance, and weight, keeping the
    top f fraction under each ranking. Averaged across phantoms."""
    rows = []
    for ph, g in df.groupby("phantom", observed=True):
        g = g.dropna(subset=["trust"])
        if len(g) == 0:
            continue
        for f in TRUST_FRACS:
            if f >= 1.0:
                keep_trust = pd.Series(True, index=g.index)
                keep_dist  = pd.Series(True, index=g.index)
                keep_w     = pd.Series(True, index=g.index)
            else:
                # trust: higher = keep
                cut_t = g["trust"].quantile(1.0 - f)
                keep_trust = g["trust"] >= cut_t
                # distance: smaller = keep
                cut_d = g["distance_cm"].quantile(f)
                keep_dist = g["distance_cm"] <= cut_d
                # weight: higher = keep
                cut_w = g["weight"].quantile(1.0 - f)
                keep_w = g["weight"] >= cut_w

            rows.append({
                "phantom": ph,
                "top_frac": f,
                "frac_kept_trust": float(keep_trust.mean()),
                "mae_trust":  float(g.loc[keep_trust, "abs_error_decades"].mean()),
                "frac_kept_dist":  float(keep_dist.mean()),
                "mae_dist":   float(g.loc[keep_dist,  "abs_error_decades"].mean()),
                "frac_kept_weight": float(keep_w.mean()),
                "mae_weight": float(g.loc[keep_w, "abs_error_decades"].mean()),
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
                "phantom": ph,
                "threshold": p,
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


# ------------------------------------------------------------------ main
def main(patterns, write_sample=False):
    df_raw = load_files(patterns)
    df, n_raw = prep(df_raw)
    print_filter_summary(df, n_raw)

    # 1. Overall correlation
    corr = df["distance_cm"].corr(df["abs_error_decades"])
    print(f"\nPearson correlation (distance vs abs error): {corr:.3f}")

    # 2. Per-phantom table (headline)
    print("\n--- Per-phantom MAE ---")
    per_ph = per_phantom_table(df)
    cols = ["phantom",
            "all_n",    "all_mae",    "all_mae_w",
            "nocens_n", "nocens_mae", "nocens_mae_w"]
    print(per_ph[cols].to_string(index=False, float_format="%.4f"))
    print("\n--- Mean across phantoms ---")
    for tag, label in [("all", "all nodes"),
                       ("nocens", "non-censored")]:
        print(f"  {label:24s}  "
              f"MAE {per_ph[f'{tag}_mae'].mean():.4f}  "
              f"MAE_w {per_ph[f'{tag}_mae_w'].mean():.4f}  "
              f"fold {10**per_ph[f'{tag}_mae'].mean():.3f}")
    per_ph.to_csv("mae_per_phantom.csv", index=False, float_format="%.6f")
    print("Saved mae_per_phantom.csv")

    # 3. Near vs far (non-censored)
    print("\n--- Near vs far (non-censored) ---")
    nf = near_far_table(df)
    print(nf.to_string(index=False, float_format="%.4f"))
    nf.to_csv("mae_near_far.csv", index=False, float_format="%.6f")
    print("Saved mae_near_far.csv")

    # 4. Binned
    print("\n--- Error vs distance (binned) ---")
    binned = binned_table(df)
    print(binned.to_string(index=False, float_format="%.3f"))
    binned.to_csv("distance_vs_error_binned.csv",
                  index=False, float_format="%.4f")
    print("Saved distance_vs_error_binned.csv")

    # 5. Per-organ pivot and heatmap
    print("\n--- Mean error by distance bin and source organ ---")
    pivot, counts = per_organ_pivots(df)
    print(pivot.round(3).to_string())
    print("\n--- Mean fold error by distance bin and source organ ---")
    print((10 ** pivot).round(3).to_string())
    pivot, counts = draw_heatmap(pivot, counts)
    pivot.to_csv("source_organ_vs_distance_mean_err.csv", float_format="%.4f")
    counts.to_csv("source_organ_vs_distance_counts.csv")
    print("Saved source_organ_vs_distance_mean_err.csv, "
          "source_organ_vs_distance_counts.csv")

    # 5c. Per-organ error, binned by trust quantile
    print("\n--- Mean error by source organ and TRUST bin ---")
    trust_err_pivot, trust_err_counts = per_organ_pivots_by_score(df, "trust")
    # reorder rows to match the error-by-distance pivot (worst at top)
    trust_err_pivot  = trust_err_pivot.reindex(pivot.index)
    trust_err_counts = trust_err_counts.reindex(pivot.index)
    print(trust_err_pivot.round(3).to_string())
    trust_err_pivot.to_csv("source_organ_vs_trustbin_mean_err.csv",
                           float_format="%.4f")
    trust_err_counts.to_csv("source_organ_vs_trustbin_counts.csv")

    print("\n--- Mean error by source organ and WEIGHT bin ---")
    weight_err_pivot, weight_err_counts = per_organ_pivots_by_score(df, "weight")
    weight_err_pivot  = weight_err_pivot.reindex(pivot.index)
    weight_err_counts = weight_err_counts.reindex(pivot.index)
    print(weight_err_pivot.round(3).to_string())
    weight_err_pivot.to_csv("source_organ_vs_weightbin_mean_err.csv",
                            float_format="%.4f")
    weight_err_counts.to_csv("source_organ_vs_weightbin_counts.csv")
    print("Saved per-organ error pivots by trust bin and weight bin.")


    print("Saved source_organ_vs_distance_mean_trust.csv, "
          "source_organ_vs_distance_mean_weight.csv")

    # 6. Coverage-accuracy: quantile (with distance baseline)
    print("\n--- Coverage vs accuracy (trust, quantile, with distance baseline) ---")
    covq = coverage_quantile(df)
    if covq.empty:
        print("  (no trust column in the CSVs)")
    else:
        print(covq.to_string(index=False, float_format="%.4f"))
        covq.to_csv("trust_coverage_quantile.csv",
                    index=False, float_format="%.6f")
        print("Saved trust_coverage_quantile.csv")

    # 7. Coverage-accuracy: absolute thresholds (kept for reference)
    print("\n--- Coverage vs accuracy (trust, absolute thresholds) ---")
    cova = coverage_absolute(df)
    if cova.empty:
        print("  (no trust column in the CSVs)")
    else:
        print(cova.to_string(index=False, float_format="%.4f"))
        cova.to_csv("trust_coverage_absolute.csv",
                    index=False, float_format="%.6f")
        print("Saved trust_coverage_absolute.csv")

    # 8. Optional plot sample
    if write_sample:
        sample = df.sample(n=min(50000, len(df)), random_state=42)
        sample[["distance_cm", "abs_error_decades", "source_organ",
                "target_organ", "trust"]].to_csv(
                    "distance_vs_error_sample.csv",
                    index=False, float_format="%.4f")
        print("Random sample saved to distance_vs_error_sample.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    write_sample = "--write-sample" in args
    patterns = [a for a in args if not a.startswith("--")]
    if not patterns:
        print("Usage: python analyse_distance.py [--write-sample] dist_error_*.csv")
        sys.exit(1)
    main(patterns, write_sample=write_sample)