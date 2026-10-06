"""
predict_and_compare.py -- train on all-but-one phantom, predict the held-out one,
and compare the predicted dose field against the Monte Carlo ground truth.
================================================================================
Produces:
  1. A scatter of predicted vs true log-dose per supervoxel (the honest view of
     where the model succeeds and fails).
  2. Error-vs-distance and error-vs-dose plots: is the error structured, or just
     noise in the far field?
  3. Per-organ comparison table (mean dose, true vs predicted).
  4. Optionally a rendered dose-map slice, by painting supervoxel predictions
     back onto the phantom voxel grid.

Usage:
    python predict_and_compare.py --graphs graphs2 \
        --hold-out 30f155090_arms_highres --organ liver --epochs 100

    # also paint the prediction back onto a phantom slice:
    python predict_and_compare.py --graphs graphs2 \
        --hold-out 30f155090_arms_highres --organ liver --epochs 100 \
        --work work --paint --slice-label 25
"""

import argparse
import glob
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader

# reuse the model / data code from the trainer
from train_gnn import (load_graphs, standardize, normalize_weights,
                       DoseGNN, weighted_loss,
                       build_organ_vocab, apply_organ_vocab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="graphs2")
    ap.add_argument("--hold-out", required=True,
                    help="phantom stem to hold out and predict")
    ap.add_argument("--organ", default=None,
                    help="restrict the comparison to one source organ")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--organ-dim", type=int, default=16,
                    help="organ-identity embedding size; 0 disables it")
    ap.add_argument("--out-prefix", default="compare")
    ap.add_argument("--exclude-young", action="store_true",
                    help="exclude newborns (00), 1-year-olds (01) and 5-year-olds (05); "
                         "keep only ages >= 10")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    graphs = load_graphs(a.graphs, exclude_young=a.exclude_young)
    train = [g for g in graphs if g.phantom != a.hold_out]
    test = [g for g in graphs if g.phantom == a.hold_out]
    if not test:
        raise SystemExit(f"no graphs for phantom '{a.hold_out}'. "
                         f"Available: {sorted(set(g.phantom for g in graphs))[:5]} ...")
    if a.organ:
        test = [g for g in test if g.source_organ == a.organ]
        if not test:
            raise SystemExit(f"no graph for organ '{a.organ}' in {a.hold_out}")
    print(f"train {len(train)} graphs | test {len(test)} graphs "
          f"({a.hold_out}{'/' + a.organ if a.organ else ''})")

    standardize(train, train + test)
    normalize_weights(train + test)
    organ_vocab = build_organ_vocab(graphs)
    apply_organ_vocab(train + test, organ_vocab)

    model = DoseGNN(train[0].x.shape[1], a.hidden, a.layers,
                    n_organs=(len(organ_vocab) + 1) if a.organ_dim else 0,
                    organ_dim=a.organ_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    tl = DataLoader(train, batch_size=a.batch_size, shuffle=True)

    print("training ...")
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for batch in tl:
            batch = batch.to(device)
            opt.zero_grad()
            loss = weighted_loss(model(batch), batch.y, batch.w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
        sched.step()
        if ep % 20 == 0 or ep == a.epochs:
            print(f"  ep {ep:3d} | train {tot/len(tl):.4f}")
            ## Save the result to a csv file with all arguments included in the file name
        out_csv = f"{a.out_prefix}_{a.hold_out}" + (f"_{a.organ}" if a.organ else "") + f"_bs{a.batch_size}" + f"_odim{a.organ_dim}" + f"_ep{ep}.csv"
        with open(out_csv, "a"):
            if ep == 1:
                # Write header
                print("epoch,hold_out,organ,batch_size,organ_dim,train_loss", file=open(out_csv, "a"))
            print(f"{ep},{a.hold_out},{a.organ},{a.batch_size},{a.organ_dim},{tot/len(tl):.4f}", file=open(out_csv, "a"))

            

    # ---------------- predict -------------------------------------------------
    model.eval()
    rows = []
    for g in test:
        gd = g.to(device)
        with torch.no_grad():
            pred = model(gd).cpu().numpy()
        rows.append({
            "organ": g.source_organ,
            "y_true": g.y.cpu().numpy(),
            "y_pred": pred,
            "w": g.w.cpu().numpy(),
            "censored": (g.censored.cpu().numpy()
                         if hasattr(g, "censored") and g.censored is not None
                         else np.zeros(len(g.y), dtype=bool)),
            "organ_id": (g.organ_id.cpu().numpy()
                         if hasattr(g, "organ_id") and g.organ_id is not None
                         else None),
            "x": g.x.cpu().numpy(),
        })

    y_true = np.concatenate([r["y_true"] for r in rows])
    y_pred = np.concatenate([r["y_pred"] for r in rows])
    w = np.concatenate([r["w"] for r in rows])
    cen = np.concatenate([r["censored"] for r in rows])
    # distance feature is column 6 (vol,mass,dens,x,y,z,dist,is_src) -- note the
    # features were standardized, so this is a z-score, still monotonic in dist
    dist = np.concatenate([r["x"][:, 6] for r in rows])

    err = y_pred - y_true
    real = ~cen                     # measured nodes only
    mae = np.abs(err[real]).mean() if real.any() else float("nan")
    w_real = np.where(real, w, 0.0)
    mae_w = (w_real * np.abs(err)).sum() / max(w_real.sum(), 1e-12)
    print("\n" + "=" * 58)
    print(f"HELD-OUT {a.hold_out}" + (f" / {a.organ}" if a.organ else ""))
    print(f"  nodes            : {len(y_true):,} "
          f"({int(real.sum()):,} measured / {int(cen.sum()):,} censored)")
    print(f"  MAE (unweighted) : {mae:.3f} decades  (x{10**mae:.2f} fold)")
    print(f"  MAE (weighted)   : {mae_w:.3f} decades  (x{10**mae_w:.2f} fold)")
    b = err[real].mean() if real.any() else 0.0
    print(f"  bias (mean err)  : {b:+.3f} decades "
          f"({'over' if b>0 else 'under'}-predicting)")
    if cen.any():
        print(f"  [censored nodes excluded from the metrics above; "
              f"their MAE is {np.abs(err[cen]).mean():.3f} dec]")
    # linear-space percentage errors
    d_t, d_p = 10.0 ** y_true[real], 10.0 ** y_pred[real]
    mape = np.abs(d_p - d_t) / np.maximum(d_t, 1e-30)
    smape = 2 * np.abs(d_p - d_t) / np.maximum(np.abs(d_p) + np.abs(d_t), 1e-30)
    print(f"  median MAPE      : {np.median(mape)*100:.1f}%")
    print(f"  median SMAPE     : {np.median(smape)*100:.1f}%")
    print("=" * 58)

    # ---------------- figures -------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

    # (1) predicted vs true
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    sc = ax[0].scatter(y_true, y_pred, c=np.log10(np.maximum(w, 1e-12)),
                       s=4, alpha=0.35, cmap="viridis")
    ax[0].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="perfect")
    ax[0].plot([lo, hi], [lo + 0.3, hi + 0.3], "r:", lw=1, label="x2")
    ax[0].plot([lo, hi], [lo - 0.3, hi - 0.3], "r:", lw=1)
    ax[0].set_xlabel("TRUE log10 dose (Monte Carlo)")
    ax[0].set_ylabel("PREDICTED log10 dose (GNN)")
    ax[0].set_title(f"{a.hold_out}\nMAE(w) {mae_w:.3f} dec = x{10**mae_w:.2f}")
    ax[0].legend(fontsize=8)
    plt.colorbar(sc, ax=ax[0], label="log10 weight", fraction=0.046)

    # (2) error vs true dose -- where does it fail?
    ax[1].scatter(y_true, err, s=4, alpha=0.3)
    ax[1].axhline(0, color="r", ls="--", lw=1.5)
    ax[1].axhline(0.3, color="r", ls=":", lw=1)
    ax[1].axhline(-0.3, color="r", ls=":", lw=1)
    ax[1].set_xlabel("TRUE log10 dose")
    ax[1].set_ylabel("error (pred - true), decades")
    ax[1].set_title("Error vs dose level")

    # (3) error distribution
    ax[2].hist(err, bins=80, color="steelblue", alpha=0.8)
    ax[2].axvline(0, color="r", ls="--")
    ax[2].set_xlabel("error (decades)")
    ax[2].set_ylabel("supervoxels")
    ax[2].set_title(f"Error distribution\nmedian {np.median(err):+.3f}, "
                    f"IQR {np.percentile(err,75)-np.percentile(err,25):.3f}")

    plt.tight_layout()
    out = f"{a.out_prefix}_{a.hold_out}" + (f"_{a.organ}" if a.organ else "") + ".png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")

    # ---------------- per-organ table ----------------------------------------
    if rows and rows[0]["organ_id"] is not None:
        print("\nPER-TARGET-ORGAN (mean log10 dose, this held-out phantom)")
        print(f"  {'label':>6s} {'n':>6s} {'true':>9s} {'pred':>9s} {'diff':>8s}")
        oid = np.concatenate([np.asarray(r["organ_id"]) for r in rows])
        for lab in np.unique(oid)[:25]:
            m = (oid == lab) & real
            if m.sum() < 5:
                continue
            print(f"  {int(lab):6d} {int(m.sum()):6d} "
                  f"{y_true[m].mean():9.3f} {y_pred[m].mean():9.3f} "
                  f"{(y_pred[m]-y_true[m]).mean():+8.3f}")


if __name__ == "__main__":
    main()
