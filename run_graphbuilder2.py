"""
run_graphbuilder2.py -- driver for GraphBuilder2: turn each dose map into a
supervoxel graph (.npz) ready for the GNN.

Usage:
    # test ONE map first (recommended before the full batch):
    python run_graphbuilder2.py --results results --work work \
        --primaries 1e+07 --window-mm 10 --out graphs2 --limit 1

    # then the full set:
    python run_graphbuilder2.py --results results --work work \
        --primaries 1e+07 --window-mm 10 --out graphs2
"""

import argparse
import glob
import os
import numpy as np
import itk

from GraphBuilder2 import GraphBuilder2
from source_organs import SOURCE_ORGANS


def parse_name(base, primaries):
    tail = f"_n{primaries}_dose.mhd"
    if not base.endswith(tail):
        return None
    core = base[: -len(tail)]
    for org in sorted(SOURCE_ORGANS, key=len, reverse=True):
        if core.endswith("_" + org):
            return core[: -(len(org) + 1)], org
    return None


def read_densities(db_path):
    dens = {}
    if not os.path.exists(db_path):
        return dens
    for line in open(db_path):
        line = line.strip()
        if line.startswith("mat") and "d=" in line:
            name = line.split(":")[0]
            try:
                lab = int(name[3:6])
                rho = float(line.split("d=")[1].split()[0])
                dens[lab] = rho
            except (ValueError, IndexError):
                continue
    return dens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--work", default="work")
    ap.add_argument("--primaries", default="1e+07")
    ap.add_argument("--window-mm", type=float, default=10.0,
                    help="supervoxel size (mm) for organ tissue")
    ap.add_argument("--bg-window-mm", type=float, default=None,
                    help="supervoxel size (mm) for background tissue "
                         "(fat/muscle/skin/bone); default 2.5x --window-mm")
    ap.add_argument("--min-volume", type=float, default=0.3)
    ap.add_argument("--keep-zero-dose", action="store_true",
                    help="keep supervoxels with no dose, at --dose-floor (Option B)")
    ap.add_argument("--dose-floor", type=float, default=1e-12,
                    help="Gy/decay assigned to zero-dose nodes: the MC detection "
                         "limit. Derive it from your data, don't guess.")
    ap.add_argument("--floor-weight-frac", type=float, default=0.1,
                    help="loss weight of floored nodes, as a fraction of the "
                         "median real-node weight")
    ap.add_argument("--out", default="graphs2")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(
        a.results, "*", f"*_n{a.primaries}_dose.mhd")))
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} dose maps | window {a.window_mm} mm")

    dens_cache = {}

    for i, dose_path in enumerate(files, 1):
        base = os.path.basename(dose_path)
        parsed = parse_name(base, a.primaries)
        if parsed is None:
            print(f"  [skip] cannot parse {base}")
            continue
        stem, organ = parsed
        out_path = os.path.join(a.out, f"{stem}__{organ}.npz")
        if os.path.exists(out_path):
            continue

        phantom_path = os.path.join(a.work, stem, f"{stem}.mhd")
        unc_path = dose_path.replace("_dose.mhd", "_dose_uncertainty.mhd")
        if not (os.path.exists(phantom_path) and os.path.exists(unc_path)):
            print(f"  [skip] missing inputs for {stem} {organ}")
            continue

        print(f"[{i}/{len(files)}] START {stem}  {organ} ...", flush=True)

        # ITK gives arrays as (z,y,x); GraphBuilder2 (like the ORIGINAL
        # GraphBuilder) expects (X,Y,Z), matching DataLoader's transpose.
        ph = itk.array_from_image(itk.imread(phantom_path)).transpose(2, 1, 0)
        d = itk.array_from_image(itk.imread(dose_path)).transpose(2, 1, 0)
        u = itk.array_from_image(itk.imread(unc_path)).transpose(2, 1, 0)
        img = itk.imread(phantom_path)
        spacing = [float(s) for s in itk.spacing(img)]   # already (x,y,z)

        if stem not in dens_cache:
            dens_cache[stem] = read_densities(
                os.path.join(a.work, stem, f"{stem}_materials.db"))
        densities = dens_cache[stem]

        gb = GraphBuilder2(window_mm=a.window_mm, bg_window_mm=a.bg_window_mm,
                           ignore_tag=0, min_volume_cm3=a.min_volume,
                           keep_zero_dose=a.keep_zero_dose,
                           dose_floor=a.dose_floor,
                           floor_weight_frac=a.floor_weight_frac)
        gb.build_supervoxels(ph, spacing)
        feats = gb.build_supervoxel_features(
            ph, densities, spacing, source_labels=SOURCE_ORGANS[organ])
        targeted = gb.build_target(feats, d, u, spacing)
        edge_index = gb.build_supervoxel_edges()

        if not targeted:
            print(f"  [skip] empty graph {stem} {organ}")
            continue

        # ---- assemble arrays, same schema as build_graphs.py -----------------
        ids = sorted(targeted.keys())
        id_map = {sv: k for k, sv in enumerate(ids)}
        organ_id = np.array([targeted[s]["organ"] for s in ids], dtype=np.int32)
        vol = np.array([targeted[s]["volume_cm3"] for s in ids], dtype=np.float32)
        mass = np.array([targeted[s]["mass_g"] for s in ids], dtype=np.float32)
        dens = np.array([targeted[s]["density"] for s in ids], dtype=np.float32)
        pos = np.array([[targeted[s]["cx"], targeted[s]["cy"], targeted[s]["cz"]]
                        for s in ids], dtype=np.float32)
        is_src = np.array([targeted[s]["is_source"] for s in ids],
                          dtype=np.float32)
        dose = np.array([targeted[s]["dose"] for s in ids], dtype=np.float64)
        censored = np.array([targeted[s].get("censored", False) for s in ids],
                            dtype=bool)
        sigma = np.array([targeted[s]["sigma"] for s in ids], dtype=np.float64)

        if is_src.any():
            src_c = pos[is_src > 0].mean(axis=0)
        else:
            src_c = pos.mean(axis=0)
        dist = np.linalg.norm(pos - src_c, axis=1).astype(np.float32)

        # ---- SOURCE-ORGAN GLOBAL FEATURES ----------------------------------
        # Self-dose is set mainly by the source organ's MASS/VOLUME: the same
        # energy per decay spread over a bigger organ gives proportionally less
        # dose (verified empirically -- S x V was constant across phantoms).
        # Each node currently knows only its OWN mass, not the total mass of the
        # emitting organ, so the model has to infer the 1/V law indirectly. We
        # broadcast the source organ's totals to every node, and also give the
        # ratio of node volume to source volume.
        if is_src.any():
            src_mass_tot = float(mass[is_src > 0].sum())
            src_vol_tot = float(vol[is_src > 0].sum())
        else:                       # should not happen, but stay finite
            src_mass_tot = float(mass.sum())
            src_vol_tot = float(vol.sum())
        src_mass_col = np.full(len(ids), src_mass_tot, dtype=np.float32)
        src_vol_col = np.full(len(ids), src_vol_tot, dtype=np.float32)
        # log of the inverse-volume law, the quantity self-dose is proportional to
        inv_vol_col = np.full(len(ids),
                              np.log10(1.0 / max(src_vol_tot, 1e-6)),
                              dtype=np.float32)

        feats_x = np.column_stack([
            vol, mass, dens, pos, dist, is_src,
            src_mass_col, src_vol_col, inv_vol_col,
        ]).astype(np.float32)

        floor = 1e-30
        y = np.log10(np.maximum(dose, floor)).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_sig = sigma / (np.maximum(dose, floor) * np.log(10.0))
        log_sig = np.where(np.isfinite(log_sig) & (log_sig > 0), log_sig, np.inf)
        w = (1.0 / log_sig ** 2).astype(np.float32)
        w = np.where(np.isfinite(w), w, 0.0)
        w[dose <= 0] = 0.0
        # Floored (censored) nodes: give them a small but nonzero weight so the
        # model learns "negligible dose here" without letting upper limits
        # dominate the loss against real measurements.
        if censored.any():
            real = w[(~censored) & (w > 0)]
            base = float(np.median(real)) if real.size else 1.0
            w[censored] = np.float32(base * a.floor_weight_frac)

        # remap edge indices from raw sv_id -> compact 0..N-1
        if edge_index.shape[1] > 0:
            valid = np.isin(edge_index[0], ids) & np.isin(edge_index[1], ids)
            e0 = np.array([id_map[v] for v in edge_index[0][valid]])
            e1 = np.array([id_map[v] for v in edge_index[1][valid]])
            ei = np.vstack([e0, e1]).astype(np.int64)
        else:
            ei = np.zeros((2, 0), dtype=np.int64)

        np.savez_compressed(
            out_path,
            x=feats_x, y=y, w=w, edge_index=ei,
            organ_id=organ_id, dose=dose, sigma=sigma, censored=censored,
            spacing=np.array(spacing), source_organ=organ, phantom=stem,
        )
        done_now = len(glob.glob(os.path.join(a.out, "*.npz")))
        print(f"[SAVED {done_now}] {stem}  {organ}: {len(ids):,} nodes, "
             f"{ei.shape[1]:,} edges -> {out_path}", flush=True)

    print(f"\ngraphs written to {a.out}/")


if __name__ == "__main__":
    main()
