#!/usr/bin/env python3
"""
Correlation between trust probability and MC weight.

Usage:
    python plot_trust_vs_weight.py matrix1/dist_error_*.csv
"""

import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


def load(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise SystemExit("No files matched.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {len(files)} file(s), {len(df):,} rows")
    return df


def prep(df):
    df = df.dropna(subset=["trust_logit", "weight"]).copy()
    df["censored"] = df["censored"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df["trust"] = 1.0 / (1.0 + np.exp(-df["trust_logit"]))
    df["log_w"] = np.log10(df["weight"].clip(lower=1e-12))
    df["label"] = (~df["censored"]) & (df["abs_error_decades"] < 0.2)
    return df


def main(patterns):
    df = prep(load(patterns))
    print(f"n = {len(df):,}")

    # ------------------------------------------------------------------
    # Global correlation
    # ------------------------------------------------------------------
    r_p,   p_p   = pearsonr(df["trust"],   df["log_w"])
    r_s,   p_s   = spearmanr(df["trust"],  df["log_w"])
    r_p_l, _     = pearsonr(df["trust_logit"], df["log_w"])
    print("\n--- Global correlation trust vs log10(weight) ---")
    print(f"  Pearson  (trust vs log_w):    r = {r_p:.4f}   p = {p_p:.2e}")
    print(f"  Spearman (trust vs log_w):    r = {r_s:.4f}   p = {p_s:.2e}")
    print(f"  Pearson  (logit vs log_w):    r = {r_p_l:.4f}")

    # same but excluding censored (the population where w is dominated by the floor)
    nz = df[~df["censored"]]
    r_s_nz, _ = spearmanr(nz["trust"], nz["log_w"])
    print(f"  Spearman (non-censored only): r = {r_s_nz:.4f}   n = {len(nz):,}")

    # ------------------------------------------------------------------
    # Binned by weight decile: mean trust, mean error
    # ------------------------------------------------------------------
    df = df.assign(w_bin=pd.qcut(df["log_w"], q=10, duplicates="drop"))
    tab = (df.groupby("w_bin", observed=True)
             .agg(n=("trust", "count"),
                  mean_w=("weight", "mean"),
                  mean_log_w=("log_w", "mean"),
                  mean_trust=("trust", "mean"),
                  median_trust=("trust", "median"),
                  mean_err=("abs_error_decades", "mean"),
                  mean_dist=("distance_cm", "mean"),
                  base_rate=("label", "mean"))
             .reset_index())
    print("\n--- By weight decile ---")
    print(tab.to_string(index=False, float_format="%.4f"))
    tab.to_csv("trust_vs_weight_deciles.csv", index=False, float_format="%.6f")

    # ------------------------------------------------------------------
    # Incremental value: trust beyond weight
    # Rank by trust vs rank by weight, coverage-accuracy curve
    # ------------------------------------------------------------------
    print("\n--- Coverage vs accuracy: trust vs weight ---")
    fracs = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00]
    rows = []
    for ph, g in df.groupby("phantom", observed=True):
        for f in fracs:
            if f >= 1.0:
                keep_t = pd.Series(True, index=g.index)
                keep_w = pd.Series(True, index=g.index)
            else:
                cut_t = g["trust"].quantile(1.0 - f)
                keep_t = g["trust"] >= cut_t
                cut_w = g["weight"].quantile(1.0 - f)
                keep_w = g["weight"] >= cut_w
            rows.append({
                "phantom": ph, "top_frac": f,
                "mae_trust":  float(g.loc[keep_t, "abs_error_decades"].mean()),
                "mae_weight": float(g.loc[keep_w, "abs_error_decades"].mean()),
            })
    cov = pd.DataFrame(rows)
    cov_mean = (cov.groupby("top_frac")
                   .agg(mae_trust=("mae_trust", "mean"),
                        mae_weight=("mae_weight", "mean"),
                        n_phantoms=("phantom", "nunique"))
                   .reset_index())
    cov_mean["gap"] = cov_mean["mae_weight"] - cov_mean["mae_trust"]
    print(cov_mean.to_string(index=False, float_format="%.4f"))
    cov_mean.to_csv("trust_vs_weight_coverage.csv",
                    index=False, float_format="%.6f")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (1) hexbin trust vs log weight (density)
    ax = axes[0, 0]
    hb = ax.hexbin(df["log_w"], df["trust"], gridsize=60, bins="log",
                   cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="log10(count)")
    ax.set_xlabel("log10(weight)")
    ax.set_ylabel("trust probability")
    ax.set_title(f"Trust vs weight (all nodes, n={len(df):,})")

    # (2) same, split by label
    ax = axes[0, 1]
    pos = df[df["label"]]
    neg = df[~df["label"]]
    ax.scatter(neg["log_w"], neg["trust"], s=1, alpha=0.05,
               color="firebrick", rasterized=True, label="label=0")
    ax.scatter(pos["log_w"], pos["trust"], s=1, alpha=0.05,
               color="seagreen", rasterized=True, label="label=1")
    ax.set_xlabel("log10(weight)")
    ax.set_ylabel("trust probability")
    ax.set_title("Trust vs weight, split by label")
    ax.legend(markerscale=8)

    # (3) coverage-accuracy: trust vs weight
    ax = axes[1, 0]
    ax.plot(cov_mean["top_frac"], cov_mean["mae_trust"],
            "o-", label="rank by trust")
    ax.plot(cov_mean["top_frac"], cov_mean["mae_weight"],
            "s-", label="rank by weight")
    ax.set_xlabel("top fraction kept")
    ax.set_ylabel("MAE among kept")
    ax.set_title("Coverage-accuracy: trust vs weight")
    ax.legend()
    ax.grid(alpha=0.3)

    # (4) mean trust per weight decile
    ax = axes[1, 1]
    x = np.arange(len(tab))
    ax.bar(x, tab["mean_trust"], color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in tab["mean_log_w"]],
                       rotation=45, ha="right")
    ax.set_xlabel("weight decile (mean log10 w)")
    ax.set_ylabel("mean trust probability")
    ax.set_title("Mean trust per weight decile")

    plt.tight_layout()
    fig.savefig("trust_vs_weight.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved trust_vs_weight.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_trust_vs_weight.py dist_error_*.csv")
        sys.exit(1)
    main(sys.argv[1:])