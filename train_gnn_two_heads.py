"""
train_gnn.py -- first training run on the supervoxel dose graphs.
=================================================================
Loads the .npz graphs produced by run_graphbuilder2.py, trains a GNN to predict
log10(dose per decay) per supervoxel, and evaluates with LEAVE-ONE-PHANTOM-OUT
so the reported error is on anatomy the model has never seen -- which is the
actual claim of the study.

KEY CHOICES (all follow from the data characteristics measured earlier)
----------------------------------------------------------------------
* TARGET is log10(dose). Dose spans ~8 orders of magnitude; a linear target
  makes MSE hostage to the source organ alone.
* LOSS is INVERSE-VARIANCE WEIGHTED using the per-node w from the Monte Carlo
  uncertainty. The far field has a long noisy tail (median ~32% per supervoxel,
  but ~2-3% where dose actually is); without weighting the model chases noise.
* FEATURES ARE STANDARDIZED using statistics from the TRAINING phantoms only
  (volume cm3, mass g, density, x/y/z cm, distance-to-source cm, is_source).
  Raw scales differ by orders of magnitude.
* Metrics are reported in LOG space (MAE in decades) and as a fold-error in
  linear space, which is what a dosimetrist actually cares about.

Usage:
    python train_gnn.py --graphs graphs2 --epochs 60
    python train_gnn.py --graphs graphs2 --epochs 60 --hidden 128 --lr 1e-3
"""

import argparse
import glob
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv

from datetime import datetime

from source_organs import SOURCE_ORGANS

import csv

from sklearn.metrics import roc_auc_score

# Build reverse mapping: label -> name
# (If a label belongs to multiple groups we just keep the first one we see)
LABEL_TO_NAME = {}
for name, labels in SOURCE_ORGANS.items():
    for lab in labels:
        if lab not in LABEL_TO_NAME:
            LABEL_TO_NAME[lab] = name

# Optional: add a few common non-source labels if you want nicer names
EXTRA_NAMES = {
    1: "adipose",
    43: "skin",
    57: "air",
    # add more if you want
}
LABEL_TO_NAME.update(EXTRA_NAMES)

def organ_name(label):
    return LABEL_TO_NAME.get(int(label), f"id_{label}")

# ------------------------------------------------------------------ data
def load_graphs(folder, exclude_young=False):
    """Read every .npz into a PyG Data object, tagged with its phantom."""
    graphs = []
    for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        ## Check filename
        filename = os.path.basename(f)
        if exclude_young and (filename.startswith("00") or filename.startswith("01") or filename.startswith("05")):
            continue
        z = np.load(f, allow_pickle=True)
        x = torch.tensor(z["x"], dtype=torch.float)
        y = torch.tensor(z["y"], dtype=torch.float)
        w = torch.tensor(z["w"], dtype=torch.float)
        ei = torch.tensor(z["edge_index"], dtype=torch.long)
        d = Data(x=x, y=y, edge_index=ei)
        d.w = w
        d.phantom = str(z["phantom"])
        d.source_organ = str(z["source_organ"])
        # carried for per-organ analysis in predict_and_compare.py
        d.organ_id = torch.tensor(z["organ_id"], dtype=torch.long)
        d.raw_vol = x[:, 0].clone()
        d.raw_pos = x[:, 3:6].clone()
        if "censored" in z:
            d.censored = torch.tensor(z["censored"], dtype=torch.bool)
        # Keep the raw distance (column 6) before any standardisation
        d.raw_dist = x[:, 6].clone()
        graphs.append(d)

    return graphs

@torch.no_grad()
def save_distance_error_csv(model, loader, device, out_path, organ_name_fn):
    """Save per-supervoxel predictions + trust for later analysis."""
    model.eval()
    rows = []

    for batch in loader:
        batch = batch.to(device)
        dose, trust_logit = model(batch)
        pred = dose.cpu()
        trust = trust_logit.cpu()
        true = batch.y.cpu()
        w = batch.w.cpu()
        dist = batch.raw_dist.cpu()
        vol = batch.raw_vol.cpu()
        pos = batch.raw_pos.cpu()
        organ_ids = batch.organ_id.cpu()
        cens = (batch.censored.cpu() if hasattr(batch, "censored")
                else torch.zeros_like(true, dtype=torch.bool))
        source = batch.source_organ[0]
        phantom = batch.phantom[0]

        for i in range(len(true)):
            rows.append({
                "phantom": phantom,
                "source_organ": source,
                "target_organ": organ_name_fn(organ_ids[i].item()),
                "target_id": int(organ_ids[i]),
                "distance_cm": float(dist[i]),
                "abs_error_decades": float((pred[i] - true[i]).abs()),
                "true_log_dose": float(true[i]),
                "pred_log_dose": float(pred[i]),
                "trust_logit": float(trust[i]),
                "weight": float(w[i]),
                "censored": bool(cens[i]),
                "volume_cm3": float(vol[i]),
                "pos_x": float(pos[i, 0]),
                "pos_y": float(pos[i, 1]),
                "pos_z": float(pos[i, 2]),
            })

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"      Distance-error CSV saved: {out_path}  ({len(rows)} rows)")

# ----------------------------------------------------------------- model
class DoseGNN(torch.nn.Module):
    """GraphConv (not GCNConv): GCN's symmetric normalization washes out the
    magnitude information that matters here, and GraphConv keeps a separate
    self-transform, which is important because a node's own organ/geometry is
    highly predictive of its dose."""

    def __init__(self, in_dim, hidden=128, layers=4, dropout=0.1,
                 n_organs=0, organ_dim=16):
        super().__init__()
        # ORGAN IDENTITY EMBEDDING. Density alone cannot separate liver (1.05),
        # spleen (1.04), pancreas (1.05) and kidney (1.05), yet their elemental
        # composition and role in the attenuation chain differ. A learned
        # embedding lets the model discover which organs behave alike for dose.
        self.use_organ = n_organs > 0
        if self.use_organ:
            self.organ_emb = torch.nn.Embedding(n_organs, organ_dim)
            in_dim = in_dim + organ_dim

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.convs.append(GraphConv(d, hidden))
            self.bns.append(BatchNorm1d(hidden))
            d = hidden

        # shared trunk head
        self.trunk = torch.nn.Sequential(
            Linear(hidden, hidden), torch.nn.ReLU()
        )
        # two output heads on top of the shared trunk
        self.head_dose  = Linear(hidden, 1)
        self.head_trust = Linear(hidden, 1)

        self.dropout = dropout

    def forward(self, data):
        x, ei = data.x, data.edge_index
        if self.use_organ and hasattr(data, "organ_idx"):
            x = torch.cat([x, self.organ_emb(data.organ_idx)], dim=1)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.trunk(x)
        dose  = self.head_dose(h).view(-1)
        trust = self.head_trust(h).view(-1)
        return dose, trust


def weighted_loss(pred, target, w):
    """Inverse-variance weighted Huber. Huber (not MSE) so the noisy far-field
    tail can't dominate through a few large residuals."""
    r = pred - target
    huber = torch.where(r.abs() < 1.0, 0.5 * r ** 2, r.abs() - 0.5)
    return (w * huber).sum() / w.sum().clamp(min=1e-8)

def trust_loss(trust_logit, dose_pred, target, censored, tau=0.2):
    """Binary cross-entropy for the trust head.

    Label:
        0 if the node is censored (MC could not resolve it)
        1 if non-censored AND |dose_pred - target| < tau
        0 if non-censored AND |dose_pred - target| >= tau
    The label is computed under no_grad so the trust loss does not pull the
    dose head to make its own label easier.
    """
    with torch.no_grad():
        within_tol = (dose_pred - target).abs() < tau
        label = within_tol & (~censored)
    return F.binary_cross_entropy_with_logits(trust_logit, label.float())

@torch.no_grad()
def evaluate_trust(model, loader, device, tau=0.2,
                   thresholds=(0.5, 0.7, 0.8, 0.9)):
    model.eval()
    all_logits, all_err, all_cens = [], [], []
    for batch in loader:
        batch = batch.to(device)
        dose, trust_logit = model(batch)
        err = (dose - batch.y).abs()
        cens = getattr(batch, "censored", None)
        if cens is None:
            cens = torch.zeros_like(err, dtype=torch.bool)
        all_logits.append(trust_logit.cpu())
        all_err.append(err.cpu())
        all_cens.append(cens.cpu())

    logits = torch.cat(all_logits)
    errs   = torch.cat(all_err)
    cens   = torch.cat(all_cens)
    probs  = torch.sigmoid(logits)

    # --- primary AUC (all nodes, Option B label) ---
    labels_all = ((errs < tau) & (~cens)).float()
    if labels_all.min() == labels_all.max():
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels_all.numpy(), probs.numpy()))

    # --- secondary AUC (non-censored only) ---
    keep = ~cens
    lab_nc = (errs[keep] < tau).float()
    if keep.any() and lab_nc.min() != lab_nc.max():
        auc_nocens = float(roc_auc_score(lab_nc.numpy(), probs[keep].numpy()))
    else:
        auc_nocens = float("nan")

    frac, mae_t = {}, {}
    for p in thresholds:
        keep_p = probs > p
        frac[p]  = float(keep_p.float().mean())
        mae_t[p] = float(errs[keep_p].mean()) if keep_p.any() else float("nan")

    return {"auc": auc, "auc_nocens": auc_nocens, "n": int(len(errs)),
            "frac_trusted": frac, "mae_trusted": mae_t}

# ------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(model, loader, device):
    """Weighted MAE EXCLUDES censored (floored) nodes. Those are upper limits,
    not measurements: which specific far-field supervoxel received literally
    zero scoring events is a coin flip, so scoring the model against them
    measures Monte Carlo luck rather than model quality. They stay in the graph
    (message passing still sees them) and remain predictable at inference."""
    model.eval()
    tot_w = tot_wae = 0.0
    all_err = []
    for batch in loader:
        batch = batch.to(device)
        pred, _ = model(batch)
        err = (pred - batch.y).abs()
        w = batch.w.clone()
        if hasattr(batch, "censored") and batch.censored is not None:
            w = torch.where(batch.censored, torch.zeros_like(w), w)
        tot_wae += float((w * err).sum())
        tot_w += float(w.sum())
        # unweighted, restricted to nodes with real signal (w>0)
        m = w > 0
        if m.any():
            all_err.append(err[m].cpu())
    mae_w = tot_wae / max(tot_w, 1e-8)            # decades, weighted
    e = torch.cat(all_err) if all_err else torch.tensor([0.0])
    return mae_w, float(e.mean()), float(e.median())

@torch.no_grad()
def evaluate_per_organ(model, loader, device):
    """Return dict: source_organ -> weighted MAE (decades)."""
    model.eval()
    stats = {}
    for batch in loader:
        batch = batch.to(device)
        pred, _ = model(batch)
        err = (pred - batch.y).abs()
        w = batch.w.clone()
        if hasattr(batch, "censored") and batch.censored is not None:
            w = torch.where(batch.censored, torch.zeros_like(w), w)
        organ = batch.source_organ[0]
        if organ not in stats:
            stats[organ] = [0.0, 0.0]
        stats[organ][0] += float((w * err).sum())
        stats[organ][1] += float(w.sum())
    return {org: swerr / max(sw, 1e-8)
            for org, (swerr, sw) in stats.items()}


@torch.no_grad()
def evaluate_source_target_matrix(model, loader, device):
    """
    Returns a dict of dicts:
        matrix[source_organ][target_organ_id] = weighted MAE (decades)
    """
    model.eval()
    # (source, target_id) -> [sum_w_err, sum_w]
    stats = {}

    for batch in loader:
        batch = batch.to(device)
        pred, _ = model(batch)
        err = (pred - batch.y).abs()
        w = batch.w.clone()
        if hasattr(batch, "censored") and batch.censored is not None:
            w = torch.where(batch.censored, torch.zeros_like(w), w)

        source = batch.source_organ[0]
        target_ids = batch.organ_id.cpu().numpy()

        for t_id, e, wt in zip(target_ids, err.cpu().numpy(), w.cpu().numpy()):
            if wt <= 0:
                continue
            key = (source, int(t_id))
            if key not in stats:
                stats[key] = [0.0, 0.0]
            stats[key][0] += e * wt
            stats[key][1] += wt

    # Convert to nested dict
    matrix = {}
    for (src, tgt), (swerr, sw) in stats.items():
        if src not in matrix:
            matrix[src] = {}
        matrix[src][tgt] = swerr / max(sw, 1e-8)

    return matrix

def build_organ_vocab(graphs):
    labels = sorted({int(v) for g in graphs for v in g.organ_id.tolist()})
    return {lab: i + 1 for i, lab in enumerate(labels)}


def apply_organ_vocab(graphs, vocab):
    for g in graphs:
        idx = [vocab.get(int(v), 0) for v in g.organ_id.tolist()]
        g.organ_idx = torch.tensor(idx, dtype=torch.long)


def standardize(train_graphs, all_graphs):
    X = torch.cat([g.x for g in train_graphs], dim=0)
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)
    std[std < 1e-8] = 1.0
    for g in all_graphs:
        g.x = (g.x - mean) / std
    return mean, std


def normalize_weights(graphs):
    for g in graphs:
        w = g.w.clone()
        w[~torch.isfinite(w)] = 0.0
        m = w[w > 0].mean() if (w > 0).any() else torch.tensor(1.0)
        g.w = w / m


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="graphs2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--organ-dim", type=int, default=16)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--exclude-young", action="store_true")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--lambda-trust", type=float, default=0.5)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    graphs = load_graphs(f"/content/drive/MyDrive/{a.graphs}",
                         exclude_young=a.exclude_young)
    print(f"{len(graphs)} graphs loaded")
    organ_vocab = build_organ_vocab(graphs)
    print(f"{len(organ_vocab)} distinct organ labels (+1 unknown)")
    phantoms = sorted(set(g.phantom for g in graphs), reverse=True)
    print(f"{len(phantoms)} phantoms: {phantoms}")

    folds = phantoms if a.folds == 0 else phantoms[: a.folds]
    results = {}          # local MAE
    trust_results = {}

    out_dir = (f"/content/drive/MyDrive/res2/hidden{a.hidden}_layers{a.layers}"
               f"_lr{a.lr}_bs{a.batch_size}_odim{a.organ_dim}"
               f"{'_age10plus' if a.exclude_young else ''}")

    ## Check if directory exists, if not, create it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created directory: {out_dir}")

    for fi, held in enumerate(folds):
        # ---- THREE-WAY SPLIT -------------------------------------------------
        # test  = the held-out phantom (never touched during training/selection)
        # val   = ONE other phantom, used ONLY to pick the stopping epoch
        # train = the rest
        # Selecting the checkpoint on `val` rather than on `test` keeps the
        # reported number honest: the model never sees test performance while
        # deciding when to stop.
        others = [p for p in phantoms if p != held]
        val_phantom = others[fi % len(others)]

        test = [g for g in graphs if g.phantom == held]
        val = [g for g in graphs if g.phantom == val_phantom]
        train = [g for g in graphs if g.phantom not in (held, val_phantom)]
        if not test or not val:
            continue
        print(f"\n=== HOLD OUT {held}  (train {len(train)} / "
              f"val {len(val)} [{val_phantom}] / test {len(test)}) ===")

        standardize(train, train + val + test)
        normalize_weights(train + val + test)
        apply_organ_vocab(train + val + test, organ_vocab)

        model = DoseGNN(train[0].x.shape[1], a.hidden, a.layers,
                        n_organs=(len(organ_vocab) + 1) if a.organ_dim else 0,
                        organ_dim=a.organ_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=(a.amp and device == "cuda"))

        tl = DataLoader(train, batch_size=a.batch_size, shuffle=True)
        vl_val = DataLoader(val, batch_size=1)
        vl_test = DataLoader(test, batch_size=1)

        best_val = float("inf")
        best_state = None
        best_ep = 0
        patience_left = a.patience

        # One CSV file per held-out phantom
        out_csv = (f"{out_dir}/held_{held}"
                   f"_val{val_phantom}.csv")

        # Write header
        with open(out_csv, "w") as f:
            f.write("epoch,train_loss,train_mae,val_mae,test_mae,best_val,patience_left\n")

        for ep in range(1, a.epochs + 1):
            model.train()
            tot = 0.0
            for batch in tl:
                batch = batch.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    with torch.cuda.amp.autocast():
                        dose, trust_logit = model(batch)
                        loss = weighted_loss(dose, batch.y, batch.w)
                        if a.lambda_trust > 0:
                            cens = getattr(batch, "censored", None)
                            if cens is None:
                                cens = torch.zeros_like(dose, dtype=torch.bool)
                            loss = loss + a.lambda_trust * trust_loss(
                                trust_logit, dose, batch.y, cens, tau=a.tau)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt); scaler.update()
                else:
                    dose, trust_logit = model(batch)
                    loss = weighted_loss(dose, batch.y, batch.w)
                    if a.lambda_trust > 0:
                        cens = getattr(batch, "censored", None)
                        if cens is None:
                            cens = torch.zeros_like(dose, dtype=torch.bool)
                        loss = loss + a.lambda_trust * trust_loss(
                            trust_logit, dose, batch.y, cens, tau=a.tau)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                tot += float(loss.detach())
            sched.step()

            train_loss = tot / len(tl)

            # Evaluate only every eval_every epochs (or last epoch)
            if (ep % a.eval_every == 0) or (ep == a.epochs):
                train_mae, _, _ = evaluate(model, tl, device)
                vmw, _, _ = evaluate(model, vl_val, device)
                tmw, _, _ = evaluate(model, vl_test, device)

                per_organ = evaluate_per_organ(model, vl_test, device)

                trust_stats = None
                if a.lambda_trust > 0:
                    trust_stats = evaluate_trust(model, vl_test, device, tau=a.tau)

                if vmw < best_val - 1e-4:
                    best_val = vmw
                    best_ep = ep
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    patience_left = a.patience
                else:
                    patience_left -= 1

                # ---------- clean per-epoch print ----------
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
                    f"ep {ep:3d} | loss {train_loss:.4f} | "
                    f"train {train_mae:.3f} | val {vmw:.3f} | test {tmw:.3f} "
                    f"(best {best_val:.3f} @ep{best_ep})")

                # Only worst & best source organs
                if per_organ:
                    sorted_orgs = sorted(per_organ.items(), key=lambda x: x[1])
                    best_org, best_mae = sorted_orgs[0]
                    worst_org, worst_mae = sorted_orgs[-1]
                    print(f"    Source organs:  BEST  {best_org:20s} {best_mae:.3f}   "
                        f"WORST {worst_org:20s} {worst_mae:.3f}")

                # Keep AUC (important)
                if trust_stats is not None:
                    print(f"    trust: AUC {trust_stats['auc']:.3f}  "
                        f"AUC_nc {trust_stats['auc_nocens']:.3f}  "
                        f"frac@0.9 {trust_stats['frac_trusted'].get(0.9, float('nan')):.3f}")
                # CSV
                with open(out_csv, "a") as f:
                    f.write(f"{ep},{train_loss:.6f},{train_mae:.6f},"
                            f"{vmw:.6f},{tmw:.6f},{best_val:.6f},{patience_left}\n")
            else:
                with open(out_csv, "a") as f:
                    f.write(f"{ep},{train_loss:.6f},,,,{best_val:.6f},{patience_left}\n")

            if patience_left <= 0:
                print(f"  early stop at ep {ep}")
                break

        # ---------- final evaluation for this held-out phantom ----------
        if best_state is not None:
            model.load_state_dict(best_state)
        mw, mu, md = evaluate(model, vl_test, device)
        results[held] = mw

        print(f"  --> {held}: MAE {mw:.3f} dec")

        if a.lambda_trust > 0:
            trust_stats = evaluate_trust(model, vl_test, device, tau=a.tau)
            trust_results[held] = trust_stats

        # still save the detailed CSVs
        dist_csv = f"{out_dir}/dist_error_{held}.csv"
        save_distance_error_csv(model, vl_test, device, dist_csv, organ_name)

        matrix = evaluate_source_target_matrix(model, vl_test, device)
        matrix_csv = f"{out_dir}/matrix_{held}.csv"
        with open(matrix_csv, "w") as f:
            f.write("source_organ,target_organ,weighted_mae\n")
            for src in sorted(matrix.keys()):
                for tgt, mae in sorted(matrix[src].items()):
                    f.write(f"{src},{organ_name(tgt)},{mae:.6f}\n")

    # ==================================================================
    # FINAL SUMMARY – only best and worst phantoms
    # ==================================================================
    print("\n" + "=" * 72)
    print("LEAVE-ONE-PHANTOM-OUT SUMMARY  (best & worst only)")
    print("=" * 72)

    if results:
        # ----- Local weighted MAE -----
        best_ph = min(results, key=results.get)
        worst_ph = max(results, key=results.get)
        print(f"\nLocal weighted MAE:")
        print(f"  BEST  {best_ph:32s}  {results[best_ph]:.3f} dec")
        print(f"  WORST {worst_ph:32s}  {results[worst_ph]:.3f} dec")
        print(f"  MEAN  {'':32s}  {np.mean(list(results.values())):.3f} dec")

        # ----- Trust AUC (kept – important) -----
        if trust_results:
            aucs = {ph: trust_results[ph]["auc"] for ph in trust_results
                    if not np.isnan(trust_results[ph]["auc"])}
            if aucs:
                best_auc_ph = max(aucs, key=aucs.get)   # higher AUC is better
                worst_auc_ph = min(aucs, key=aucs.get)
                print(f"\nTrust AUC:")
                print(f"  BEST  {best_auc_ph:32s}  {aucs[best_auc_ph]:.3f}")
                print(f"  WORST {worst_auc_ph:32s}  {aucs[worst_auc_ph]:.3f}")
                print(f"  MEAN  {'':32s}  {np.mean(list(aucs.values())):.3f}")

    print("=" * 72)


if __name__ == "__main__":
    main()
