#!/usr/bin/env python3
"""
dvh_vs_distance.py -- DVH error vs distance, with optional zero-threshold.

Usage:
    python dvh_vs_distance.py --clinical-only --zero-below -10 dist_error_*.csv
    python dvh_vs_distance.py --clinical-only --max-dist 15 --zero-below -10 dist_error_*.csv

Reads dist_error_*.csv. Required columns:
    phantom, source_organ, target_organ,
    true_log_dose, pred_log_dose, volume_cm3, distance_cm
Optional: censored

--zero-below T
    If both true_log_dose and pred_log_dose at a point are < T,
    treat the error and bias at that point as 0 (not |pred-true|).
    Threshold is in log10 space. Recommended: -10 (i.e. 1e-10 of the
    reference dose -- effectively numerical zero). Use -3 for 0.1% of
    prescription if you want a stricter physical cutoff.
    Default: no zeroing.
"""
import argparse
import glob

import numpy as np
import pandas as pd


DEFAULT_QUANTILES = (5, 50, 95)
CLINICAL_ORGANS = {
    "liver", "lungs", "kidneys", "pancreas", "spleen", "stomach",
    "esophagus", "heart", "brain", "thyroid", "prostate", "testes",
    "ovaries", "uterus", "thymus", "adrenals", "gall_bladder",
    "urinary_bladder", "rectosigmoid", "left_colon", "right_colon",
    "small_intestine", "salivary_glands", "breast_glandular",
    "spinal_cord",
}
DIST_BINS   = [0, 5, 10, 15, 20, 30, 40, 50, 75, 100, np.inf]
DIST_LABELS = ["0-5", "5-10", "10-15", "15-20", "20-30",
               "30-40", "40-50", "50-75", "75-100", "100+"]
MIN_N = 20


# ----------------------------------------------------------------- helpers
def wquantile(v, w, q):
    order = np.argsort(v, kind="stable")
    v_s, w_s = v[order], w[order]
    cw = np.cumsum(w_s)
    total = cw[-1]
    if total <= 0:
        return np.nan, -1
    target = q / 100.0 * total
    pos = int(np.searchsorted(cw, target, side="left"))
    pos = min(pos, len(v_s) - 1)
    if pos == 0:
        val = float(v_s[0])
    else:
        c_lo, c_hi = cw[pos - 1], cw[pos]
        if c_hi <= c_lo:
            val = float(v_s[pos])
        else:
            f = (target - c_lo) / (c_hi - c_lo)
            val = float(v_s[pos - 1] + f * (v_s[pos] - v_s[pos - 1]))
    return val, int(order[pos])


def _err(pred, true, zero_below):
    """Error and bias, zeroed when both values are below threshold."""
    if zero_below is not None and true < zero_below and pred < zero_below:
        return 0.0, 0.0, True
    return abs(pred - true), pred - true, False


# ----------------------------------------------------------------- per-task table
def per_task_table(df, quantiles, cens_mode, min_n, zero_below):
    d = df if cens_mode == "all" else df[~df["censored"]]
    rows = []
    for (ph, src, tgt), g in d.groupby(
            ["phantom", "source_organ", "target_organ"], observed=True):
        if len(g) < min_n:
            continue
        v_true = g["true_log_dose"].to_numpy(dtype=float)
        v_pred = g["pred_log_dose"].to_numpy(dtype=float)
        w      = g["volume_cm3"].to_numpy(dtype=float)
        dist   = g["distance_cm"].to_numpy(dtype=float)
        if not np.isfinite(dist).all():
            continue

        mean_dist = float(dist.mean())
        for q in quantiles:
            Dt, i_def = wquantile(v_true, w, q)
            Dp, _     = wquantile(v_pred, w, q)
            if i_def < 0:
                continue

            # err on the quantile value: zero if both Dq values are below
            ec, bc, zc = _err(Dp, Dt, zero_below)

            # err at the defining supervoxel: zero if both pieces are below
            p_def = float(v_pred[i_def])
            t_def = float(v_true[i_def])
            et, bt, zt = _err(p_def, t_def, zero_below)

            rows.append({
                "phantom":      ph,
                "source_organ": src,
                "target_organ": tgt,
                "quantile":     q,
                "cens_mode":    cens_mode,
                "mean_dist":    mean_dist,
                "dist_at_def":  float(dist[i_def]),
                "err_classical": ec,
                "err_true_loc":  et,
                "bias_classical": bc,
                "bias_true_loc":  bt,
                "zeroed_classical": zc,
                "zeroed_true_loc":  zt,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- reporting
def print_endpoint_distance(tab, quantiles):
    print("\n--- Where do DVH endpoints sit? (median distance in cm) ---")
    hdr = f"  {'q':>4}  {'mean_dist':>10}  {'dist_at_def':>12}  {'n':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for q in quantiles:
        s = tab[tab["quantile"] == q]
        if s.empty:
            continue
        print(f"  {q:>4}  {s['mean_dist'].median():>10.2f}  "
              f"{s['dist_at_def'].median():>12.2f}  {len(s):>6}")


def print_error_vs_distance_corr(tab, quantiles):
    from scipy.stats import spearmanr
    print("\n--- Spearman rho between distance and error (per quantile) ---")
    print(f"  {'q':>4}  {'rho_class':>10}  {'rho_true':>10}  {'n':>6}")
    for q in quantiles:
        s = tab[tab["quantile"] == q].dropna(
            subset=["dist_at_def", "err_classical", "err_true_loc"])
        if len(s) < 20:
            continue
        r_c, _ = spearmanr(s["dist_at_def"], s["err_classical"])
        r_t, _ = spearmanr(s["dist_at_def"], s["err_true_loc"])
        print(f"  {q:>4}  {r_c:>+10.3f}  {r_t:>+10.3f}  {len(s):>6}")


def print_dvh_error_by_distance(tab, quantiles, zero_below):
    print(f"\n--- DVH error by distance bin (bin on 'dist_at_def') ---")
    if zero_below is not None:
        print(f"    (zeroing applied: both values < {zero_below})")
    for q in quantiles:
        s = tab[tab["quantile"] == q].copy()
        if s.empty:
            continue
        s["dist_bin"] = pd.cut(s["dist_at_def"], bins=DIST_BINS,
                               labels=DIST_LABELS, right=False)
        g = (s.groupby("dist_bin", observed=True)
              .agg(n=("err_classical", "count"),
                   zeroed=("zeroed_classical", "mean"),
                   err_class=("err_classical", "median"),
                   err_true=("err_true_loc", "median"),
                   bias=("bias_classical", "median"))
              .reset_index())
        print(f"\n  q={q}:")
        print(g.to_string(index=False, float_format="%.3f"))


def print_pointwise_by_distance(df, cens_mode="uncensored", zero_below=None):
    d = df if cens_mode == "all" else df[~df["censored"]]
    d = d.dropna(subset=["true_log_dose", "pred_log_dose", "distance_cm"])
    if d.empty:
        return
    d = d.copy()

    if zero_below is not None:
        z = (d["true_log_dose"] < zero_below) & (d["pred_log_dose"] < zero_below)
        d["err"]  = np.where(z, 0.0, (d["pred_log_dose"] - d["true_log_dose"]).abs())
        d["bias"] = np.where(z, 0.0,  d["pred_log_dose"] - d["true_log_dose"])
    else:
        d["err"]  = (d["pred_log_dose"] - d["true_log_dose"]).abs()
        d["bias"] =  d["pred_log_dose"] - d["true_log_dose"]

    d["dist_bin"] = pd.cut(d["distance_cm"], bins=DIST_BINS,
                           labels=DIST_LABELS, right=False)
    g = (d.groupby("dist_bin", observed=True)
          .agg(n=("err", "count"),
               err_median=("err", "median"),
               err_mean=("err", "mean"),
               bias=("bias", "median"))
          .reset_index())
    print(f"\n--- Pointwise error by distance ({cens_mode}) ---")
    if zero_below is not None:
        print(f"    (zeroing applied: both values < {zero_below})")
    print(g.to_string(index=False, float_format="%.3f"))


def print_zeroed_summary(tab):
    if tab.empty or "zeroed_classical" not in tab.columns:
        return
    print("\n--- Zeroed-out tasks (both values below threshold) ---")
    for q in sorted(tab["quantile"].unique()):
        s = tab[tab["quantile"] == q]
        n_class = int(s["zeroed_classical"].sum())
        n_true  = int(s["zeroed_true_loc"].sum())
        n_total = len(s)
        print(f"  q={q:>4}: classical {n_class:>5}/{n_total:<6} "
              f"({100*n_class/max(n_total,1):5.1f}%)   "
              f"true_loc  {n_true:>5}/{n_total:<6} "
              f"({100*n_true/max(n_total,1):5.1f}%)")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--source-organ", default=None)
    ap.add_argument("--quantiles", default=None,
                    help="comma-separated, e.g. 5,50,98")
    ap.add_argument("--clinical-only", action="store_true")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--zero-below", type=float, default=None,
                    help="zero error when BOTH true and pred < this "
                         "(log10 space). Recommended: -10")
    ap.add_argument("--max-dist", type=float, default=None,
                    help="drop supervoxels beyond this distance (cm)")
    a = ap.parse_args()

    min_n = a.min_n
    zero_below = a.zero_below
    quantiles = (tuple(int(x) for x in a.quantiles.split(","))
                 if a.quantiles else DEFAULT_QUANTILES)

    files = []
    for p in a.patterns:
        files.extend(glob.glob(p))
    if not files:
        raise SystemExit("No files matched.")
    print(f"Loading {len(files)} file(s)...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"  {len(df):,} rows raw")

    need = {"phantom", "source_organ", "target_organ",
            "true_log_dose", "pred_log_dose", "volume_cm3", "distance_cm"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["true_log_dose", "pred_log_dose",
                           "volume_cm3", "distance_cm"]).copy()
    df = df[df["volume_cm3"] > 0]
    df["censored"] = (
        df["censored"].astype(str).str.lower().isin(["true", "1", "1.0"])
        if "censored" in df.columns else False)

    if a.source_organ is not None:
        before = len(df)
        df = df[df["source_organ"] == a.source_organ]
        print(f"  filtered source_organ='{a.source_organ}': "
              f"{before:,} -> {len(df):,}")
    if a.clinical_only:
        before = len(df)
        df = df[df["target_organ"].isin(CLINICAL_ORGANS)]
        print(f"  restricted to clinical targets: "
              f"{before:,} -> {len(df):,}")
    if a.max_dist is not None:
        before = len(df)
        df = df[df["distance_cm"] <= a.max_dist]
        print(f"  max distance {a.max_dist} cm: "
              f"{before:,} -> {len(df):,}")

    n = len(df)
    n_cens = int(df["censored"].sum())
    print("\n--- Filter summary ---")
    print(f"  rows        {n:>10,}")
    print(f"  censored    {n_cens:>10,}  ({100 * n_cens / max(n, 1):5.1f}%)")
    print(f"  uncensored  {n - n_cens:>10,}  "
          f"({100 * (n - n_cens) / max(n, 1):5.1f}%)")
    if zero_below is not None:
        n_zero = int(((df["true_log_dose"] < zero_below) &
                      (df["pred_log_dose"] < zero_below)).sum())
        print(f"  zero-below threshold: {zero_below}  "
              f"({n_zero:,} rows = {100*n_zero/max(n,1):.1f}% "
              f"will be zeroed when both sides are below)")

    print("\nComputing per-task DVH metrics...")
    tab_unc = per_task_table(df, quantiles, "uncensored", min_n, zero_below)
    if tab_unc.empty:
        raise SystemExit(f"No tasks with >= {min_n} supervoxels.")

    print("\n" + "=" * 78)
    print("DVH vs DISTANCE (uncensored)")
    print("=" * 78)

    print_endpoint_distance(tab_unc, quantiles)
    print_error_vs_distance_corr(tab_unc, quantiles)
    print_zeroed_summary(tab_unc)
    print_dvh_error_by_distance(tab_unc, quantiles, zero_below)
    print_pointwise_by_distance(df, "uncensored", zero_below)

    tab_unc.to_csv("dvh_vs_distance.csv",
                   index=False, float_format="%.6f")
    print("\nSaved dvh_vs_distance.csv")


if __name__ == "__main__":
    main()