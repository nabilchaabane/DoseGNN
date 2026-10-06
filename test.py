import glob
import pandas as pd
import numpy as np

from train_gnn_two_heads import DoseGNN

# --- load all CSVs the same way analyse_distance.py does ---
# files = sorted(glob.glob("matrix/dist_error_*.csv"))
# print("files:", files)
# dfs = [pd.read_csv(f) for f in files]
# df = pd.concat(dfs, ignore_index=True)
# print(f"Loaded {len(files)} file(s), {len(df):,} rows")

# # --- normalize censored to a real bool, per file before concat would be safer,
# #     but doing it after concat is fine as long as the dtype is uniform ---
# print("censored dtype:", df["censored"].dtype)
# df["censored"] = df["censored"].astype(str).str.lower().isin(["true", "1", "1.0"])
# print("censored counts:")
# print(df["censored"].value_counts(dropna=False))

# # --- invert log10 ---
# df["D_true"] = 10 ** df["true_log_dose"]
# df["D_pred"] = 10 ** df["pred_log_dose"]

# # --- distance bins, same as analyse_distance.py ---
# bins   = [0, 5, 10, 15, 20, 30, 50, 100, np.inf]
# labels = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-50", "50-100", "100+"]
# df["dist_bin"] = pd.cut(df["distance_cm"], bins=bins, labels=labels, right=False)

# # --- 1. per-bin x censored-flag: the key table ---
# tab = (df.groupby(["dist_bin", "censored"], observed=True)
#          .agg(n=("abs_error_decades", "count"),
#               mean_err=("abs_error_decades", "mean"),
#               median_err=("abs_error_decades", "median"),
#               mean_true_log=("true_log_dose", "mean"),
#               mean_dist=("distance_cm", "mean"))
#          .reset_index())
# print("\n--- per distance bin x censored ---")
# print(tab.to_string(index=False, float_format="%.3f"))

# # --- 2. per-bin: all nodes vs non-censored only ---
# all_nodes = (df.groupby("dist_bin", observed=True)
#                .agg(n_all=("abs_error_decades", "count"),
#                     mean_err_all=("abs_error_decades", "mean"),
#                     median_err_all=("abs_error_decades", "median"))
#                .reset_index())
# noncens = (df[~df["censored"]].groupby("dist_bin", observed=True)
#              .agg(n_noncens=("abs_error_decades", "count"),
#                   mean_err_noncens=("abs_error_decades", "mean"),
#                   median_err_noncens=("abs_error_decades", "median"))
#              .reset_index())

# merged = all_nodes.merge(noncens, on="dist_bin", how="left")
# merged["delta_mean"]   = merged["mean_err_noncens"]   - merged["mean_err_all"]
# merged["delta_median"] = merged["median_err_noncens"] - merged["median_err_all"]
# print("\n--- per bin: all vs non-censored ---")
# print(merged.to_string(index=False, float_format="%.3f"))

# # --- 3. overall numbers ---
# print("\n--- overall (all phantoms pooled) ---")
# print(f"all nodes       n={len(df):>7,}  mean={df['abs_error_decades'].mean():.3f}  "
#       f"median={df['abs_error_decades'].median():.3f}")
# sub = df[~df["censored"]]
# print(f"non-censored    n={len(sub):>7,}  mean={sub['abs_error_decades'].mean():.3f}  "
#       f"median={sub['abs_error_decades'].median():.3f}")

# # --- 4. per phantom: does the pattern hold everywhere? ---
# print("\n--- per phantom: all vs non-censored ---")
# per_ph = (df.groupby("phantom")
#             .apply(lambda d: pd.Series({
#                 "n_all":       len(d),
#                 "n_censored":  int(d["censored"].sum()),
#                 "mean_all":    d["abs_error_decades"].mean(),
#                 "mean_noncens": d.loc[~d["censored"], "abs_error_decades"].mean(),
#             }), include_groups=False)
#             .reset_index())
# per_ph["delta_mean"] = per_ph["mean_noncens"] - per_ph["mean_all"]
# print(per_ph.to_string(index=False, float_format="%.3f"))

# near_cens = df[(df["censored"]) & (df["distance_cm"] < 30)]
# print(f"n = {len(near_cens):,}")
# print(near_cens["target_organ"].value_counts().head(20))
# print()
# print(near_cens[["distance_cm","target_organ","source_organ",
#                  "abs_error_decades","pred_log_dose"]]
#       .sort_values("abs_error_decades", ascending=False)
#       .head(30).to_string())


# df["censored"] = df["censored"].astype(str).str.lower().isin(["true","1","1.0"])

# sub = df[~df["censored"]]

# # bucket by weight (proxy for inverse variance)
# sub = sub.assign(w_bin=pd.qcut(sub["weight"], q=10, duplicates="drop"))
# tab = sub.groupby("w_bin", observed=True).agg(
#     n=("abs_error_decades", "count"),
#     mean_w=("weight", "mean"),
#     mean_err=("abs_error_decades", "mean"),
#     median_err=("abs_error_decades", "median"),
#     mean_true_log=("true_log_dose", "mean"),
#     mean_dist=("distance_cm", "mean"),
# ).reset_index()
# print(tab.to_string(index=False, float_format="%.4f"))

# # the two numbers
# unweighted = sub["abs_error_decades"].mean()
# weighted = (sub["weight"] * sub["abs_error_decades"]).sum() / sub["weight"].sum()
# print(f"\nunweighted MAE: {unweighted:.4f}")
# print(f"weighted   MAE: {weighted:.4f}")
# print(f"ratio:          {unweighted/weighted:.2f}x")




#!/usr/bin/env python3
"""
Plot the distribution of trust probabilities from dist_error_*.csv files.

Usage:
    python plot_trust_dist.py matrix1/*.csv
"""

import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise SystemExit("No files matched.")
    print(f"Loading {len(files)} file(s)...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"  {len(df):,} rows")
    return df


def prep(df):
    df = df.dropna(subset=["trust_logit"]).copy()
    df["censored"] = df["censored"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df["trust"] = 1.0 / (1.0 + np.exp(-df["trust_logit"]))
    # label the head was trained against
    df["label"] = (~df["censored"]) & (df["abs_error_decades"] < 0.2)
    return df


def main():
    df = prep(load(sorted(glob.glob("matrix1/dist_error_*.csv"))))

    print(f"\nn = {len(df):,}")
    print(f"base rate (label == 1): {df['label'].mean():.4f}")
    print(f"trust_logit percentiles:")
    print(df["trust_logit"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).to_string())
    print(f"\ntrust probability percentiles:")
    print(df["trust"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).to_string())
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"  frac(trust > {t:.1f}) = {(df['trust'] > t).mean():.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- 1. histogram of trust probabilities, all nodes ---
    ax = axes[0, 0]
    ax.hist(df["trust"], bins=100, range=(0, 1),
            color="steelblue", edgecolor="none")
    ax.axvline(0.5, color="red", ls="--", lw=1, label="p = 0.5")
    ax.set_xlabel("trust probability")
    ax.set_ylabel("count")
    ax.set_title(f"Distribution of trust probability (all nodes, n={len(df):,})")
    ax.legend()

    # --- 2. histogram split by label (positive vs negative) ---
    ax = axes[0, 1]
    pos = df.loc[df["label"],  "trust"]
    neg = df.loc[~df["label"], "trust"]
    bins = np.linspace(0, 1, 101)
    ax.hist(neg, bins=bins, alpha=0.6, color="firebrick",
            label=f"label=0 (n={len(neg):,})", density=True)
    ax.hist(pos, bins=bins, alpha=0.6, color="seagreen",
            label=f"label=1 (n={len(pos):,})", density=True)
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.set_xlabel("trust probability")
    ax.set_ylabel("density")
    ax.set_title("Trust probability by true label")
    ax.legend()

    # --- 3. histogram of trust probability, binned by distance ---
    ax = axes[1, 0]
    dist_bins   = [0, 10, 20, 30, 50, 100, np.inf]
    dist_labels = ["0-10", "10-20", "20-30", "30-50", "50-100", "100+"]
    df["dbin"] = pd.cut(df["distance_cm"], bins=dist_bins,
                        labels=dist_labels, right=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(dist_labels)))
    for c, lab in zip(colors, dist_labels):
        sub = df.loc[df["dbin"] == lab, "trust"]
        if len(sub) > 0:
            ax.hist(sub, bins=bins, histtype="step", lw=1.8,
                    color=c, label=f"{lab} cm (n={len(sub):,})", density=True)
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.set_xlabel("trust probability")
    ax.set_ylabel("density")
    ax.set_title("Trust probability by distance bin")
    ax.legend(fontsize=8)

    # --- 4. trust vs true dose scatter ---
    ax = axes[1, 1]
    sample = df.sample(n=min(30000, len(df)), random_state=0)
    ax.scatter(sample["true_log_dose"], sample["trust"],
               s=1, alpha=0.15, color="steelblue", rasterized=True)
    ax.axhline(0.5, color="red", ls="--", lw=1)
    ax.set_xlabel("true log10(dose)")
    ax.set_ylabel("trust probability")
    ax.set_title("Trust vs true dose (30k sample)")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out = "trust_probability_distribution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # --- additional: fraction above threshold, by distance bin ---
    print("\n--- fraction with trust > 0.5, by distance bin ---")
    tab = (df.groupby("dbin", observed=True)
             .agg(n=("trust", "count"),
                  mean_trust=("trust", "mean"),
                  frac_gt_0p3=("trust", lambda s: (s > 0.3).mean()),
                  frac_gt_0p5=("trust", lambda s: (s > 0.5).mean()),
                  frac_gt_0p7=("trust", lambda s: (s > 0.7).mean()),
                  base_rate=("label", "mean"))
             .reset_index())
    print(tab.to_string(index=False, float_format="%.4f"))
    tab.to_csv("trust_by_distance.csv", index=False, float_format="%.6f")
    print("Saved trust_by_distance.csv")


if __name__ == "__main__":
    main()