#!/usr/bin/env python3
"""
Analyse distance vs error from one or more dist_error_*.csv files.

Usage:
    python analyse_distance.py dist_error_10f125040_arms_highres.csv
    python analyse_distance.py dist_error_*.csv          # multiple files
"""

import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def main(patterns):
    df = load_files(patterns)

    # Basic cleaning
    df = df.dropna(subset=["distance_cm", "abs_error_decades"])
    df["censored"] = df["censored"].astype(bool)
    # df = df[~df["censored"]]              # uncomment to drop censored nodes completely

    print(f"Using {len(df):,} nodes after filtering")

    # ------------------------------------------------------------------
    # 1. Overall correlation
    # ------------------------------------------------------------------
    corr = df["distance_cm"].corr(df["abs_error_decades"])
    print(f"\nPearson correlation (distance vs abs error): {corr:.3f}")

    # ------------------------------------------------------------------
    # 2. Binned analysis
    # ------------------------------------------------------------------
    bins = [0, 5, 10, 15, 20, 30, 50, 100, np.inf]
    labels = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-50", "50-100", "100+"]
    df["dist_bin"] = pd.cut(df["distance_cm"], bins=bins, labels=labels, right=False)

    binned = (df.groupby("dist_bin", observed=True)
                .agg(
                    n=("abs_error_decades", "count"),
                    mean_err=("abs_error_decades", "mean"),
                    median_err=("abs_error_decades", "median"),
                    mean_dist=("distance_cm", "mean"),
                    mean_log_dose=("true_log_dose", "mean"),
                    max_log_dose=("true_log_dose", "max"),
                    max_pred_log_dose=("pred_log_dose", "max"),
                )
                .reset_index())

    print("\n--- Error vs Distance (binned) ---")
    print(binned.to_string(index=False, float_format="%.3f"))

    # ------------------------------------------------------------------
    # 3. Same thing but split by source organ (optional, can be long)
    # ------------------------------------------------------------------
    print("\n--- Mean error by distance bin and source organ ---")
    src_list = df["source_organ"].value_counts().index.tolist()
    sub = df[df["source_organ"].isin(src_list)]

    pivot = (sub.groupby(["source_organ", "dist_bin"], observed=True)["abs_error_decades"]
               .mean()
               .unstack("dist_bin"))
    print(pivot.round(3).to_string())

    ## Convert from decades to fold
    pivot_folds = 10 ** pivot
    print("\n--- Mean fold error by distance bin and source organ ---")
    print(pivot_folds.round(3).to_string())

    # ------------------------------------------------------------------
    # 3b. Heatmap: source organ vs distance bin (mean abs error)
    # ------------------------------------------------------------------
    # Count matrix (for annotations / masking sparse cells)
    count_pivot = (sub.groupby(["source_organ", "dist_bin"], observed=True)["abs_error_decades"]
                     .count()
                     .unstack("dist_bin"))

    # Reorder rows by overall mean error (worst at top) for readability
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]
    count_pivot = count_pivot.loc[row_order]

    # Optional: mask cells with too few samples so they don't dominate the colour scale
    min_n = 20
    masked = pivot.where(count_pivot >= min_n)

    fig, ax = plt.subplots(figsize=(11, max(6, 0.32 * len(pivot))))
    sns.heatmap(
        masked,
        ax=ax,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean |error| (decades)"},
        mask=masked.isna(),
    )
    ax.set_xlabel("Distance bin (cm)")
    ax.set_ylabel("Source organ")
    ax.set_title(f"Mean absolute error (decades): source organ vs distance\n"
                 f"(cells with n < {min_n} masked)")
    plt.tight_layout()

    heatmap_out = "source_organ_vs_distance_heatmap.png"
    fig.savefig(heatmap_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap saved to {heatmap_out}")

    # Also save the underlying pivot tables as CSV for further inspection
    pivot.to_csv("source_organ_vs_distance_mean_err.csv", float_format="%.4f")
    count_pivot.to_csv("source_organ_vs_distance_counts.csv")
    print("Pivot tables saved to source_organ_vs_distance_mean_err.csv "
          "and source_organ_vs_distance_counts.csv")

    # ------------------------------------------------------------------
    # 3c. Overall MAE per phantom
    # ------------------------------------------------------------------
    print("\n--- Overall MAE per phantom ---")

    AIR_LIKE = {"air", "lungs"}
    df["is_air"] = df["target_organ"].isin(AIR_LIKE)

    def _mae(g):
        n = len(g)
        if n == 0:
            return pd.Series({"n": 0, "mae": np.nan, "mae_w": np.nan,
                              "median": np.nan})
        e = g["abs_error_decades"]
        w = g["weight"]
        mae   = e.mean()
        mae_w = float((w * e).sum() / w.sum()) if w.sum() > 0 else np.nan
        return pd.Series({"n": n, "mae": mae, "mae_w": mae_w,
                          "median": e.median()})

    rows = []
    for ph, g in df.groupby("phantom", observed=True):
        row = {"phantom": ph}
        r = _mae(g)
        row.update({f"all_{k}": v for k, v in r.items()})
        r = _mae(g[~g["censored"]])
        row.update({f"nocens_{k}": v for k, v in r.items()})
        r = _mae(g[~g["censored"] & ~g["is_air"]])
        row.update({f"tissue_{k}": v for k, v in r.items()})
        rows.append(row)

    per_phantom = pd.DataFrame(rows)
    cols = ["phantom",
            "all_n",    "all_mae",    "all_mae_w",
            "nocens_n", "nocens_mae", "nocens_mae_w",
            "tissue_n", "tissue_mae", "tissue_mae_w"]
    print(per_phantom[cols].to_string(index=False, float_format="%.4f"))

    print("\n--- Mean across phantoms ---")
    for tag, label in [("all", "all nodes"),
                       ("nocens", "non-censored"),
                       ("tissue", "non-censored, non-air")]:
        print(f"  {label:24s}  "
              f"MAE {per_phantom[f'{tag}_mae'].mean():.4f}  "
              f"MAE_w {per_phantom[f'{tag}_mae_w'].mean():.4f}  "
              f"fold {10**per_phantom[f'{tag}_mae'].mean():.3f}")

    per_phantom.to_csv("mae_per_phantom.csv",
                       index=False, float_format="%.6f")
    print("\nPer-phantom MAE saved to mae_per_phantom.csv")

    # ------------------------------------------------------------------
    # 4. Save binned results
    # ------------------------------------------------------------------
    out = "distance_vs_error_binned.csv"
    binned.to_csv(out, index=False, float_format="%.4f")
    print(f"\nBinned summary saved to {out}")

    # Optional: save a smaller sample for plotting
    sample = df.sample(n=min(50000, len(df)), random_state=42)
    sample[["distance_cm", "abs_error_decades", "source_organ", "target_organ"]].to_csv(
        "distance_vs_error_sample.csv", index=False, float_format="%.4f")
    print("Random sample saved to distance_vs_error_sample.csv (for plotting)")

    # ------------------------------------------------------------------
    # 6. Coverage-accuracy curve from the trust score
    # ------------------------------------------------------------------
    if "trust_logit" in df.columns:
        df["trust"] = 1.0 / (1.0 + np.exp(-df["trust_logit"]))

        # Per-phantom first, then average, so phantoms are equal-weight
        thresholds = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                            0.6, 0.7, 0.8, 0.9, 0.95])
        rows = []
        for ph, g in df.groupby("phantom", observed=True):
            for p in thresholds:
                keep = g["trust"] > p
                if keep.any():
                    rows.append({
                        "phantom": ph,
                        "threshold": p,
                        "frac_kept": keep.mean(),
                        "n_kept": int(keep.sum()),
                        "mae_kept": g.loc[keep, "abs_error_decades"].mean(),
                        "mae_kept_median": g.loc[keep, "abs_error_decades"].median(),
                    })
        cov = pd.DataFrame(rows)

        # average across phantoms (equal weight per phantom)
        cov_mean = (cov.groupby("threshold")
                    .agg(frac_kept=("frac_kept", "mean"),
                            mae_kept=("mae_kept", "mean"),
                            mae_kept_median=("mae_kept_median", "mean"),
                            n_phantoms=("phantom", "nunique"))
                    .reset_index())
        print("\n--- Coverage vs accuracy (trust filter) ---")
        print(cov_mean.to_string(index=False, float_format="%.4f"))
        cov.to_csv("trust_coverage_accuracy.csv", index=False, float_format="%.6f")
    print("Saved trust_coverage_accuracy.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyse_distance.py dist_error_*.csv")
        sys.exit(1)
    main(sys.argv[1:])