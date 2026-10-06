#!/usr/bin/env python3
"""
Analyse dose prediction error and trust-score reliability from
dist_error_*.csv files.

Usage:
    python analyse_distance.py dist_error_*.csv
    python analyse_distance.py --source-organ liver dist_error_*.csv

Framing:
    The MC inverse-variance weight w = 1/sigma^2 is the physical ground
    truth for measurement reliability. It is not available at inference
    time (the MC uncertainty map is a simulation artefact). The trust
    head is evaluated as a learned surrogate for that signal that IS
    computable at inference from input features alone.

Sections:
    1. Filter summary
    2. Pointwise MAE (overall, by region, near/far)
    3. DVH metrics (region-filtered to clinically relevant pairs)
    4. DVH error vs trust score (within-phantom Spearman)
    5. DVH weight analysis: is the endpoint-defining supervoxel reliable?
    6. Trust vs MC inverse-variance weight agreement (surrogate check)
    7. Coverage vs accuracy (trust vs weight as ranking signals)

Region note:
    Organs that belong to multiple regions (esophagus, thyroid, lungs,
    heart, small_intestine) are assigned to the first region they appear
    in ORGAN_REGIONS. This keeps each pair in exactly one region for
    grouping. It is a reporting convenience, not an anatomical claim.

Excluded structures:
    spinal_cord, testes, ovaries were removed from CLINICAL_ORGANS and
    from the region definitions. At the supervoxel resolution used here
    (~10 mm organ tissue), these structures contain too few supervoxels
    per phantom for stable D95/D50/D2 endpoints. This is a resolution
    limit, not a clinical judgement. Full DVH results including these
    structures are a supplementary limitation.
"""

import sys
import glob
import pandas as pd
import numpy as np


# ------------------------------------------------------------------ constants
NEAR_FAR_CUT = 30.0
TRUST_FRACS  = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.00]

CLINICAL_ORGANS = {
    "liver", "lungs", "kidneys", "pancreas", "spleen", "stomach",
    "esophagus", "heart", "brain", "thyroid", "prostate",
    "uterus", "thymus", "adrenals", "gall_bladder",
    "urinary_bladder", "rectosigmoid", "left_colon", "right_colon",
    "small_intestine", "salivary_glands", "breast_glandular",
}

ORGAN_REGIONS = {
    "head_and_neck": {"brain", "thyroid", "salivary_glands", "esophagus"},
    "thorax":        {"lungs", "heart", "esophagus",
                      "breast_glandular", "thymus", "thyroid"},
    "abdomen":       {"liver", "kidneys", "pancreas", "spleen",
                      "stomach", "gall_bladder", "adrenals",
                      "small_intestine", "lungs", "heart"},
    "pelvis":        {"prostate", "uterus",
                      "urinary_bladder", "rectosigmoid",
                      "left_colon", "right_colon", "small_intestine"},
}

REGION_ORDER = ["head_and_neck", "thorax", "abdomen", "pelvis"]


def _build_primary_region_map():
    m = {}
    for region in REGION_ORDER:
        for org in ORGAN_REGIONS[region]:
            m.setdefault(org, region)
    return m


ORGAN_PRIMARY_REGION = _build_primary_region_map()


def is_clinically_relevant_pair(src, tgt):
    if src not in ORGAN_PRIMARY_REGION or tgt not in ORGAN_PRIMARY_REGION:
        return False
    if src == tgt:
        return True
    return any((src in organs) and (tgt in organs)
               for organs in ORGAN_REGIONS.values())


def pair_region(src, tgt):
    return ORGAN_PRIMARY_REGION.get(src, "unknown")


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
    df = df[df["source_organ"].isin(CLINICAL_ORGANS)]
    if "target_organ" in df.columns:
        df = df[df["target_organ"].isin(CLINICAL_ORGANS)]

    df["censored"] = (df["censored"].astype(str).str.lower()
                      .isin(["true", "1", "1.0"]))
    df["trust"] = (1.0 / (1.0 + np.exp(-df["trust_logit"]))
                   if "trust_logit" in df.columns else np.nan)
    df["log_w"] = np.log10(df["weight"].clip(lower=1e-12))
    df["is_far"] = df["distance_cm"] >= NEAR_FAR_CUT
    df["source_region"] = df["source_organ"].map(ORGAN_PRIMARY_REGION)
    df["target_region"] = df["target_organ"].map(ORGAN_PRIMARY_REGION)
    return df, n_raw


def print_filter_summary(df, n_raw):
    n = len(df)
    n_cens = int(df["censored"].sum())
    print("\n--- Filter summary ---")
    print(f"  raw rows       {n_raw:>12,}")
    print(f"  after dropna   {n:>12,}")
    print(f"  censored       {n_cens:>12,}  ({100*n_cens/max(n,1):5.1f}%)")
    print(f"  non-censored   {n-n_cens:>12,}  ({100*(n-n_cens)/max(n,1):5.1f}%)")


# ------------------------------------------------------------------ stats
def _stats(g):
    n = len(g)
    if n == 0:
        return pd.Series({"n": 0, "mean": np.nan, "std": np.nan,
                          "median": np.nan, "p90": np.nan})
    e = g["abs_error_decades"]
    return pd.Series({
        "n": n,
        "mean": float(e.mean()),
        "std": float(e.std(ddof=1)) if n > 1 else 0.0,
        "median": float(e.median()),
        "p90": float(e.quantile(0.90)),
    })


def print_pointwise_summary(df):
    d = df[~df["censored"]]
    print("\n--- Pointwise MAE (non-censored) ---")

    print(f"\n  Overall and near/far:")
    print(f"  {'region':16s}  {'n':>10s}  "
          f"{'mean':>8s}  {'std':>8s}  {'median':>8s}  {'p90':>8s}")
    for tag, sub in [("overall", d),
                     ("near (<30cm)", d[~d["is_far"]]),
                     ("far  (>=30cm)", d[d["is_far"]])]:
        s = _stats(sub)
        print(f"  {tag:16s}  {int(s['n']):>10,}  "
              f"{s['mean']:>8.4f}  {s['std']:>8.4f}  "
              f"{s['median']:>8.4f}  {s['p90']:>8.4f}")

    print("\n  Per-phantom mean, pooled as mean ± std across phantoms:")
    per_ph = []
    for ph, g in d.groupby("phantom", observed=True):
        e = g["abs_error_decades"]
        if len(e) > 0:
            per_ph.append(float(e.mean()))
    a = np.array(per_ph)
    print(f"    mean of phantom means : {a.mean():.4f}")
    print(f"    std across phantoms   : {a.std(ddof=1):.4f}")
    print(f"    range                 : [{a.min():.4f}, {a.max():.4f}]")

    print(f"\n  By SOURCE region (non-censored, all distances):")
    print(f"  {'region':16s}  {'n':>10s}  "
          f"{'mean':>8s}  {'std':>8s}  {'median':>8s}  {'p90':>8s}")
    for region in REGION_ORDER:
        s = _stats(d[d["source_region"] == region])
        if s["n"] == 0:
            continue
        print(f"  {region:16s}  {int(s['n']):>10,}  "
              f"{s['mean']:>8.4f}  {s['std']:>8.4f}  "
              f"{s['median']:>8.4f}  {s['p90']:>8.4f}")

    print(f"\n  By TARGET region (non-censored, all distances):")
    print(f"  {'region':16s}  {'n':>10s}  "
          f"{'mean':>8s}  {'std':>8s}  {'median':>8s}  {'p90':>8s}")
    for region in REGION_ORDER:
        s = _stats(d[d["target_region"] == region])
        if s["n"] == 0:
            continue
        print(f"  {region:16s}  {int(s['n']):>10,}  "
              f"{s['mean']:>8.4f}  {s['std']:>8.4f}  "
              f"{s['median']:>8.4f}  {s['p90']:>8.4f}")


# ------------------------------------------------------------------ DVH
def dvh_metrics(df, min_n=20, dose_floor_log=-10.0, trust_band=0.05,
                min_n_band=3):
    """
    Volume-weighted DVH metrics per (phantom, source, target) pair.

    Records, at each endpoint-defining supervoxel:
      - trust score
      - MC inverse-variance weight w
      - percentile rank of that weight within the pair
    and the volume-weighted mean w over the +/- trust_band window.
    """
    if "volume_cm3" not in df.columns:
        print("  [skip] no volume_cm3 column; rerun save_distance_error_csv")
        return pd.DataFrame()

    df = df.dropna(subset=["volume_cm3", "true_log_dose",
                           "pred_log_dose"]).copy()
    df = df[df["volume_cm3"] > 0]
    df = df[~df["censored"]]

    def wpercentile_with_trust(values, volumes, trust, mc_w,
                               q, band, min_n_band):
        """
        Returns (pct, trust_band_mean, trust_defining,
                 w_defining, w_band_mean).
        """
        order = np.argsort(values)
        v  = np.asarray(values)[order]
        vw = np.asarray(volumes)[order]
        t  = np.asarray(trust)[order]
        wo = np.asarray(mc_w)[order]
        cw = np.cumsum(vw)
        if cw[-1] <= 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        total = cw[-1]
        target = q / 100.0 * total
        idx = min(int(np.searchsorted(cw, target, side="left")), len(v) - 1)
        pct = float(v[idx])
        trust_defining = float(t[idx])
        w_defining = float(wo[idx])

        eff_band = band
        n = len(v)
        if n > 0 and n * 2 * eff_band < min_n_band:
            eff_band = min(0.5, min_n_band / (2.0 * n))
        lo = target - eff_band * total
        hi = target + eff_band * total
        mask = (cw >= lo) & (cw <= hi)
        if mask.any():
            trust_band_mean = float((t[mask] * vw[mask]).sum() / vw[mask].sum())
            w_band_mean = float((wo[mask] * vw[mask]).sum() / vw[mask].sum())
        else:
            trust_band_mean = trust_defining
            w_band_mean = w_defining
        return pct, trust_band_mean, trust_defining, w_defining, w_band_mean

    rows = []
    for (ph, src, tgt), g in df.groupby(
            ["phantom", "source_organ", "target_organ"], observed=True):

        if not is_clinically_relevant_pair(src, tgt):
            continue
        if len(g) < min_n:
            continue
        if g["true_log_dose"].max() < dose_floor_log:
            continue

        vol   = g["volume_cm3"].values
        tlog  = g["true_log_dose"].values
        plog  = g["pred_log_dose"].values
        trust = g["trust"].values
        w_mc  = g["weight"].values

        pair_vol = float(vol.sum())
        pair_mean_trust = (float((trust * vol).sum() / pair_vol)
                           if pair_vol > 0 else np.nan)
        pair_median_w = float(np.median(w_mc))
        pair_mean_w   = float(np.mean(w_mc))

        def pct_rank_in_pair(x, arr):
            return float((arr <= x).mean()) * 100.0

        D95_t, tr_D95, _, w_D95, wb_D95 = wpercentile_with_trust(
            tlog, vol, trust, w_mc, 5,  trust_band, min_n_band)
        D50_t, tr_D50, _, w_D50, wb_D50 = wpercentile_with_trust(
            tlog, vol, trust, w_mc, 50, trust_band, min_n_band)
        D2_t,  tr_D2,  _, w_D2,  wb_D2  = wpercentile_with_trust(
            tlog, vol, trust, w_mc, 98, trust_band, min_n_band)
        D95_p, _, _, _, _ = wpercentile_with_trust(
            plog, vol, trust, w_mc, 5,  trust_band, min_n_band)
        D50_p, _, _, _, _ = wpercentile_with_trust(
            plog, vol, trust, w_mc, 50, trust_band, min_n_band)
        D2_p,  _, _, _, _ = wpercentile_with_trust(
            plog, vol, trust, w_mc, 98, trust_band, min_n_band)

        rows.append({
            "phantom": ph,
            "source_organ": src,
            "target_organ": tgt,
            "region": pair_region(src, tgt),
            "n": len(g),
            "volume_cm3": pair_vol,
            "pair_mean_trust": pair_mean_trust,
            "pair_median_w": pair_median_w,
            "pair_mean_w": pair_mean_w,

            "D95_err_dec":  abs(D95_p - D95_t),
            "D50_err_dec":  abs(D50_p - D50_t),
            "D2_err_dec":   abs(D2_p  - D2_t),

            "trust_at_D95": tr_D95,
            "trust_at_D50": tr_D50,
            "trust_at_D2":  tr_D2,

            "w_at_D95": w_D95,
            "w_at_D50": w_D50,
            "w_at_D2":  w_D2,
            "w_band_D95": wb_D95,
            "w_band_D50": wb_D50,
            "w_band_D2":  wb_D2,
            "w_pctile_D95": pct_rank_in_pair(w_D95, w_mc),
            "w_pctile_D50": pct_rank_in_pair(w_D50, w_mc),
            "w_pctile_D2":  pct_rank_in_pair(w_D2,  w_mc),
        })

    return pd.DataFrame(rows)


def print_dvh_by_region(dvh):
    """Region-level DVH summary — the primary table for the paper."""
    print("\n  DVH error per REGION (clinically relevant pairs):")
    print(f"  {'region':16s}  {'n_pairs':>8s}  {'n_ph':>5s}  "
          f"{'D95 (mean ± std)':>20s}  "
          f"{'D50 (mean ± std)':>20s}  "
          f"{'D2 (mean ± std)':>20s}")
    for region in REGION_ORDER:
        sub = dvh[dvh["region"] == region]
        if sub.empty:
            continue
        def _ms(col):
            e = sub[col]
            return f"{e.mean():.3f} ± {e.std(ddof=1) if len(e) > 1 else 0.0:.3f}"
        print(f"  {region:16s}  {len(sub):>8d}  "
              f"{sub['phantom'].nunique():>5d}  "
              f"{_ms('D95_err_dec'):>20s}  "
              f"{_ms('D50_err_dec'):>20s}  "
              f"{_ms('D2_err_dec'):>20s}")

    print("\n  Per-region, per-phantom mean pooled across phantoms:")
    print(f"  {'region':16s}  {'endpoint':>8s}  "
          f"{'mean':>8s}  {'std':>8s}  {'n_ph':>5s}")
    for region in REGION_ORDER:
        sub = dvh[dvh["region"] == region]
        if sub.empty:
            continue
        for name, col in [("D95", "D95_err_dec"),
                          ("D50", "D50_err_dec"),
                          ("D2",  "D2_err_dec")]:
            per_ph = (sub.groupby("phantom", observed=True)[col]
                         .mean().dropna().values)
            if len(per_ph) == 0:
                continue
            print(f"  {region:16s}  {name:>8s}  "
                  f"{per_ph.mean():>8.4f}  "
                  f"{per_ph.std(ddof=1) if len(per_ph) > 1 else 0.0:>8.4f}  "
                  f"{len(per_ph):>5d}")

    print("\n  Overall across all regions:")
    print(f"  {'endpoint':>8s}  {'n_pairs':>8s}  "
          f"{'mean':>8s}  {'std':>8s}  {'median':>8s}  {'p90':>8s}")
    for name, col in [("D95", "D95_err_dec"),
                      ("D50", "D50_err_dec"),
                      ("D2",  "D2_err_dec")]:
        e = dvh[col]
        print(f"  {name:>8s}  {len(e):>8d}  "
              f"{e.mean():>8.4f}  {e.std(ddof=1):>8.4f}  "
              f"{e.median():>8.4f}  {e.quantile(0.90):>8.4f}")


def print_dvh_by_pair(dvh, min_phantoms=3):
    """Per (src, tgt) table — detail."""
    g = dvh.groupby(["source_organ", "target_organ", "region"],
                    observed=True)
    rows = []
    for (src, tgt, reg), sub in g:
        n_ph = sub["phantom"].nunique()
        if n_ph < min_phantoms:
            continue
        rows.append({
            "region": reg,
            "source_organ": src,
            "target_organ": tgt,
            "n_phantoms": n_ph,
            "n_tasks": int(sub["n"].sum()),
            "D95_mean": sub["D95_err_dec"].mean(),
            "D95_std":  sub["D95_err_dec"].std(ddof=1) if len(sub) > 1 else 0.0,
            "D50_mean": sub["D50_err_dec"].mean(),
            "D50_std":  sub["D50_err_dec"].std(ddof=1) if len(sub) > 1 else 0.0,
            "D2_mean":  sub["D2_err_dec"].mean(),
            "D2_std":   sub["D2_err_dec"].std(ddof=1) if len(sub) > 1 else 0.0,
        })
    tab = pd.DataFrame(rows)
    if tab.empty:
        return
    tab = tab.sort_values(["region", "D95_mean"], ascending=[True, False])

    def _fmt(m, s):
        return f"{m:.3f} ± {s:.3f}"
    tab["D95"] = [_fmt(m, s) for m, s in zip(tab["D95_mean"], tab["D95_std"])]
    tab["D50"] = [_fmt(m, s) for m, s in zip(tab["D50_mean"], tab["D50_std"])]
    tab["D2"]  = [_fmt(m, s) for m, s in zip(tab["D2_mean"],  tab["D2_std"])]

    print(f"\n  Per (source, target) — pairs on >= {min_phantoms} phantoms, "
          f"sorted by region then D95:")
    print(tab[["region", "source_organ", "target_organ", "n_phantoms",
               "n_tasks", "D95", "D50", "D2"]].to_string(index=False))


def dvh_trust_correlation(dvh):
    from scipy.stats import spearmanr
    print("\n--- DVH error vs trust score (within-phantom Spearman) ---")
    print("  Negative rho = higher trust associates with lower DVH error.")

    print(f"\n  Per-phantom rho, pooled across phantoms:")
    print(f"  {'endpoint':>8s}  {'mean':>8s}  {'std':>8s}  "
          f"{'min':>8s}  {'max':>8s}  {'n_ph':>5s}")
    for X in ["D95", "D50", "D2"]:
        rhos = []
        for ph, g in dvh.groupby("phantom", observed=True):
            sub = g.dropna(subset=[f"trust_at_{X}", f"{X}_err_dec"])
            if len(sub) >= 5 and sub[f"trust_at_{X}"].nunique() > 1:
                r, _ = spearmanr(sub[f"trust_at_{X}"], sub[f"{X}_err_dec"])
                if np.isfinite(r):
                    rhos.append(r)
        if not rhos:
            print(f"  {X:>8s}  (no phantom with enough pairs)")
            continue
        a = np.array(rhos)
        print(f"  {X:>8s}  {a.mean():>+8.3f}  "
              f"{a.std(ddof=1) if len(a) > 1 else 0.0:>8.3f}  "
              f"{a.min():>+8.3f}  {a.max():>+8.3f}  {len(a):>5d}")

    print(f"\n  Within-region rho (phantom-centred):")
    print(f"  {'region':16s}  {'endpoint':>8s}  {'rho':>8s}  {'n':>7s}")
    for region in REGION_ORDER:
        sub = dvh[dvh["region"] == region]
        if len(sub) < 20:
            continue
        for X in ["D95", "D50", "D2"]:
            s = sub.dropna(subset=[f"trust_at_{X}", f"{X}_err_dec"]).copy()
            if len(s) < 20:
                continue
            s["t_c"] = (s[f"trust_at_{X}"]
                        - s.groupby("phantom")[f"trust_at_{X}"].transform("mean"))
            s["e_c"] = (s[f"{X}_err_dec"]
                        - s.groupby("phantom")[f"{X}_err_dec"].transform("mean"))
            if s["t_c"].nunique() > 1 and s["e_c"].nunique() > 1:
                r, _ = spearmanr(s["t_c"], s["e_c"])
                print(f"  {region:16s}  {X:>8s}  {r:>+8.3f}  {len(s):>7d}")

    print("\n  Pooled across all pairs (phantom-centred):")
    print(f"  {'endpoint':>8s}  {'rho':>8s}  {'n':>7s}")
    for X in ["D95", "D50", "D2"]:
        g = dvh.dropna(subset=[f"trust_at_{X}", f"{X}_err_dec"]).copy()
        if len(g) < 20:
            continue
        g["t_c"] = (g[f"trust_at_{X}"]
                    - g.groupby("phantom")[f"trust_at_{X}"].transform("mean"))
        g["e_c"] = (g[f"{X}_err_dec"]
                    - g.groupby("phantom")[f"{X}_err_dec"].transform("mean"))
        if g["t_c"].nunique() > 1 and g["e_c"].nunique() > 1:
            r, _ = spearmanr(g["t_c"], g["e_c"])
            print(f"  {X:>8s}  {r:>+8.3f}  {len(g):>7d}")


# ------------------------------------------------------------------ DVH weight
def print_endpoint_weight_summary(dvh):
    """How does the defining supervoxel's weight compare to the pair?"""
    print("\n--- Endpoint supervoxels: are they low-weight? ---")
    print("  MC inverse-variance weight at the supervoxel that defines each")
    print("  DVH endpoint, compared to the pair's median weight.")
    print()
    print(f"  {'endpoint':>8s}  {'w_def/pair_med':>16s}  "
          f"{'%below_med':>11s}  {'%bottom_quart':>14s}  {'n':>6s}")
    for X in ["D95", "D50", "D2"]:
        sub = dvh.dropna(subset=[f"w_at_{X}", "pair_median_w"])
        if sub.empty:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (sub[f"w_at_{X}"] / sub["pair_median_w"]).replace(
                [np.inf, -np.inf], np.nan).dropna()
        ratio_med = float(ratio.median()) if len(ratio) else np.nan
        below_med = float((sub[f"w_at_{X}"] < sub["pair_median_w"]).mean()) * 100
        bot_q = float((sub[f"w_pctile_{X}"] < 25).mean()) * 100
        print(f"  {X:>8s}  {ratio_med:>16.3f}  "
              f"{below_med:>10.1f}%  {bot_q:>13.1f}%  {len(sub):>6d}")

    print("\n  Interpretation:")
    print("    w_def/pair_med < 1     : endpoint sits on a below-typical sv.")
    print("    %below_med > 50        : endpoint more often than not on a")
    print("                             below-median sv.")
    print("    %bottom_quart > 25     : endpoint over-represented in the")
    print("                             bottom quartile of pair weights.")

    print("\n  Ratio w_def / pair_median, by region:")
    print(f"  {'region':16s}  {'D95':>8s}  {'D50':>8s}  {'D2':>8s}  {'n':>6s}")
    for region in REGION_ORDER:
        sub = dvh[dvh["region"] == region]
        if sub.empty:
            continue
        def _med_ratio(X):
            s = sub.dropna(subset=[f"w_at_{X}", "pair_median_w"])
            if s.empty:
                return np.nan
            r = (s[f"w_at_{X}"] / s["pair_median_w"]).replace(
                [np.inf, -np.inf], np.nan).dropna()
            return float(r.median()) if len(r) else np.nan
        print(f"  {region:16s}  {_med_ratio('D95'):>8.3f}  "
              f"{_med_ratio('D50'):>8.3f}  {_med_ratio('D2'):>8.3f}  "
              f"{len(sub):>6d}")


def print_dvh_stratified_by_endpoint_weight(dvh):
    """DVH error stratified by within-phantom quartile of defining-sv weight."""
    print("\n--- DVH error stratified by defining-SV weight "
          "(within-phantom quartiles) ---")
    for X in ["D95", "D50", "D2"]:
        w_col = f"w_at_{X}"
        err_col = f"{X}_err_dec"
        sub = dvh.dropna(subset=[w_col, err_col]).copy()
        if len(sub) < 50:
            continue
        def _q(s):
            if len(s) < 8 or s.nunique() < 2:
                return pd.Series(np.nan, index=s.index)
            try:
                return pd.qcut(s.rank(method="first"), q=4, labels=False)
            except ValueError:
                return pd.Series(np.nan, index=s.index)
        sub["w_q"] = sub.groupby("phantom", observed=True)[w_col].transform(_q)
        sub = sub.dropna(subset=["w_q"])
        if sub.empty:
            continue
        sub["w_q"] = sub["w_q"].astype(int).map(
            {0: "Q1 (low w)", 1: "Q2", 2: "Q3", 3: "Q4 (high w)"})
        g = (sub.groupby("w_q", observed=True)
                .agg(n=(err_col, "count"),
                     n_ph=("phantom", "nunique"),
                     w_med=(w_col, "median"),
                     err_med=(err_col, "median"),
                     err_mean=(err_col, "mean"),
                     err_std=(err_col, "std"))
                .reindex(["Q1 (low w)", "Q2", "Q3", "Q4 (high w)"])
                .reset_index())
        print(f"\n  {X}:")
        print(g.to_string(index=False, float_format="%.4f"))


def dvh_weight_correlation(dvh):
    """Does the defining-sv weight predict DVH error?"""
    from scipy.stats import spearmanr
    print("\n--- Correlation: defining-SV weight vs DVH error ---")
    print("  Negative rho means lower weight at the defining supervoxel")
    print("  is associated with larger DVH error.")
    print(f"\n  {'endpoint':>8s}  {'per-ph mean':>12s}  {'std':>8s}  "
          f"{'min':>8s}  {'max':>8s}  {'pooled':>8s}  {'n_ph':>5s}")
    for X in ["D95", "D50", "D2"]:
        w_col = f"w_at_{X}"
        e_col = f"{X}_err_dec"
        rhos = []
        for ph, g in dvh.groupby("phantom", observed=True):
            s = g.dropna(subset=[w_col, e_col])
            if len(s) >= 5 and s[w_col].nunique() > 1:
                r, _ = spearmanr(s[w_col], s[e_col])
                if np.isfinite(r):
                    rhos.append(r)
        s = dvh.dropna(subset=[w_col, e_col]).copy()
        pooled = np.nan
        if len(s) >= 20:
            s["w_c"] = s[w_col] - s.groupby("phantom")[w_col].transform("mean")
            s["e_c"] = s[e_col] - s.groupby("phantom")[e_col].transform("mean")
            if s["w_c"].nunique() > 1 and s["e_c"].nunique() > 1:
                r, _ = spearmanr(s["w_c"], s["e_c"])
                pooled = r
        if not rhos:
            continue
        a = np.array(rhos)
        print(f"  {X:>8s}  {a.mean():>+12.3f}  "
              f"{a.std(ddof=1) if len(a) > 1 else 0.0:>8.3f}  "
              f"{a.min():>+8.3f}  {a.max():>+8.3f}  "
              f"{pooled:>+8.3f}  {len(a):>5d}")


# ------------------------------------------------------------------ trust
def trust_weight_agreement(df):
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


def trust_weight_agreement_per_phantom(df):
    from scipy.stats import spearmanr, pearsonr
    print("\n  Per-phantom trust vs log10(weight):")
    print(f"  {'stat':>10s}  {'mean':>8s}  {'std':>8s}  "
          f"{'min':>8s}  {'max':>8s}  {'n_ph':>5s}")
    pear, spear = [], []
    for ph, g in df.groupby("phantom", observed=True):
        g = g.dropna(subset=["trust", "log_w"])
        if len(g) < 50 or g["trust"].nunique() < 2 or g["log_w"].nunique() < 2:
            continue
        rp, _ = pearsonr(g["trust"], g["log_w"])
        rs, _ = spearmanr(g["trust"], g["log_w"])
        if np.isfinite(rp): pear.append(rp)
        if np.isfinite(rs): spear.append(rs)
    for name, vals in [("pearson", np.array(pear)),
                       ("spearman", np.array(spear))]:
        if len(vals) == 0:
            continue
        print(f"  {name:>10s}  {vals.mean():>8.4f}  "
              f"{vals.std(ddof=1) if len(vals) > 1 else 0.0:>8.4f}  "
              f"{vals.min():>8.4f}  {vals.max():>8.4f}  {len(vals):>5d}")

    print("\n  Per-source-region trust vs log10(weight):")
    print(f"  {'region':16s}  {'n':>10s}  {'pearson':>8s}  {'spearman':>8s}")
    for region in REGION_ORDER:
        sub = df[df["source_region"] == region].dropna(
            subset=["trust", "log_w"])
        if len(sub) < 50 or sub["trust"].nunique() < 2:
            continue
        rp, _ = pearsonr(sub["trust"], sub["log_w"])
        rs, _ = spearmanr(sub["trust"], sub["log_w"])
        print(f"  {region:16s}  {len(sub):>10,}  "
              f"{rp:>8.4f}  {rs:>8.4f}")


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
                "mae_trust":  float(g.loc[kt, "abs_error_decades"].mean()),
                "mae_weight": float(g.loc[kw, "abs_error_decades"].mean()),
            })
    cov = pd.DataFrame(rows)
    if cov.empty:
        return cov
    out = (cov.groupby("top_frac")
              .agg(mae_trust=("mae_trust", "mean"),
                   mae_trust_std=("mae_trust", "std"),
                   mae_weight=("mae_weight", "mean"),
                   mae_weight_std=("mae_weight", "std"),
                   n_phantoms=("phantom", "nunique"))
              .reset_index())
    return out


def coverage_by_region(df):
    """Coverage-vs-accuracy per source region."""
    print("\n  Coverage vs accuracy, per source region:")
    print(f"  {'region':16s}  {'top_frac':>8s}  "
          f"{'mae_trust':>10s}  {'mae_weight':>10s}")
    for region in REGION_ORDER:
        sub = df[df["source_region"] == region]
        if len(sub) < 500:
            continue
        rows = []
        for ph, g in sub.groupby("phantom", observed=True):
            g = g.dropna(subset=["trust"])
            if len(g) < 20:
                continue
            n = len(g)
            rt = g["trust"].rank(method="first", ascending=False)
            rw = g["weight"].rank(method="first", ascending=False)
            for f in [0.10, 0.50]:
                k = int(round(f * n))
                rows.append({
                    "top_frac": f,
                    "mae_trust":  float(g.loc[rt <= k,
                                              "abs_error_decades"].mean()),
                    "mae_weight": float(g.loc[rw <= k,
                                              "abs_error_decades"].mean()),
                })
        if not rows:
            continue
        r = pd.DataFrame(rows).groupby("top_frac").mean()
        for f in [0.10, 0.50]:
            if f in r.index:
                print(f"  {region:16s}  {f:>8.2f}  "
                      f"{r.loc[f, 'mae_trust']:>10.4f}  "
                      f"{r.loc[f, 'mae_weight']:>10.4f}")


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
    print_pointwise_summary(df)

    # ---- DVH ----
    print("\n--- DVH metrics (volume-weighted, clinically relevant pairs) ---")
    dvh = dvh_metrics(df)
    if dvh.empty:
        print("  (no volume_cm3 column, or no pair with >= 20 supervoxels)")
    else:
        print_dvh_by_region(dvh)
        print_dvh_by_pair(dvh)
        dvh_trust_correlation(dvh)

        # ---- NEW: DVH weight analysis ----
        print_endpoint_weight_summary(dvh)
        print_dvh_stratified_by_endpoint_weight(dvh)
        dvh_weight_correlation(dvh)

        dvh.to_csv("dvh_per_phantom.csv", index=False, float_format="%.6f")
        print("\n  saved dvh_per_phantom.csv")

    # ---- Trust vs weight ----
    print("\n--- Trust as a surrogate for MC inverse-variance weight ---")
    print("  w = 1/sigma^2 is not available at inference time. The trust head")
    print("  is trained to approximate it from input features alone.")
    tw = trust_weight_agreement(df)
    for k, v in tw.items():
        print(f"  {k:24s}  {v:.4f}" if isinstance(v, float)
              else f"  {k:24s}  {v}")
    pd.DataFrame([tw]).to_csv("trust_weight_agreement.csv",
                              index=False, float_format="%.6f")

    trust_weight_agreement_per_phantom(df)

    # ---- Coverage-accuracy ----
    print("\n--- Coverage vs accuracy (both ranking signals) ---")
    print("  MAE of the top-f fraction of supervoxels, ranked by trust")
    print("  and by the physical weight. Both are reported; neither is")
    print("  claimed superior.")
    covq = coverage_quantile(df)
    if covq.empty:
        print("  (no trust column)")
    else:
        print(covq.to_string(index=False, float_format="%.4f"))
        covq.to_csv("trust_coverage_quantile.csv",
                    index=False, float_format="%.6f")

    coverage_by_region(df)


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