"""
train_gnn.py -- supervoxel dose graphs with per-node and per-organ heads.
=======================================================================
Loads the .npz graphs produced by run_graphbuilder2.py, trains a GNN with
leave-one-phantom-out evaluation. Three heads share the trunk:

  * DOSE HEAD  -- per-node log10(dose).
  * TRUST HEAD -- per-node reliability score, trained against
                  |err| < tau, as before.
  * DVH HEAD   -- per-(phantom, source, target) endpoint reliability.
                  Takes pooled per-organ features and outputs three
                  probabilities, one per endpoint, for whether the
                  predicted D95 / D50 / D2 is within tolerance of the
                  truth.

The per-node trust head is unchanged from the original architecture. The
DVH head is a second-stage readout on pooled per-organ features.

Usage:
    python train_gnn.py --graphs graphs2 --epochs 60
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

LABEL_TO_NAME = {}
for name, labels in SOURCE_ORGANS.items():
    for lab in labels:
        if lab not in LABEL_TO_NAME:
            LABEL_TO_NAME[lab] = name

EXTRA_NAMES = {1: "adipose", 43: "skin", 57: "air"}
LABEL_TO_NAME.update(EXTRA_NAMES)


def organ_name(label):
    return LABEL_TO_NAME.get(int(label), f"id_{label}")


# ------------------------------------------------------------------ data
def load_graphs(folder, exclude_young=False):
    graphs = []
    for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        filename = os.path.basename(f)
        if exclude_young and (filename.startswith("00")
                              or filename.startswith("01")
                              or filename.startswith("05")):
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
        d.organ_id = torch.tensor(z["organ_id"], dtype=torch.long)
        d.raw_vol = x[:, 0].clone()
        d.raw_pos = x[:, 3:6].clone()
        if "censored" in z:
            d.censored = torch.tensor(z["censored"], dtype=torch.bool)
        d.raw_dist = x[:, 6].clone()
        graphs.append(d)
    return graphs


# ------------------------------------------------------------------ pooling
def pool_group_stats(values, volume, key):
    """
    Pool per-node `values` by integer `key` (usually batch_idx * K + organ_id).

    Returns:
        stats: (n_groups, 9) tensor of float32:
            [mean, min, max, std, D95, D50, D2, log(sum vol), log(n)]
        unique_keys: (n_groups,) tensor of the group keys.
    """
    unique_keys, inverse = torch.unique(key, return_inverse=True)
    n_groups = int(unique_keys.numel())
    dev = values.device

    out = torch.zeros(n_groups, 9, device=dev, dtype=torch.float32)
    vals_f = values.float()
    vol_f = volume.float()

    for gi in range(n_groups):
        m = inverse == gi
        d = vals_f[m]
        v = vol_f[m]
        n = int(d.numel())
        if n == 0:
            continue

        out[gi, 0] = d.mean()
        out[gi, 1] = d.min()
        out[gi, 2] = d.max()
        out[gi, 3] = d.std() if n > 1 else 0.0
        out[gi, 7] = torch.log(v.sum().clamp(min=1e-8))
        out[gi, 8] = float(np.log(n))

        if n >= 2:
            order = torch.argsort(d)
            d_sorted = d[order]
            v_sorted = v[order]
            cw = torch.cumsum(v_sorted, dim=0)
            total = cw[-1]
            if total > 0:
                for j, q in enumerate([5.0, 50.0, 98.0]):
                    target = (q / 100.0) * total
                    pos = int(torch.searchsorted(cw, target).clamp(max=n - 1).item())
                    out[gi, 4 + j] = d_sorted[pos]
            else:
                out[gi, 4] = out[gi, 5] = out[gi, 6] = out[gi, 0]
        else:
            out[gi, 4] = out[gi, 5] = out[gi, 6] = out[gi, 0]

    return out, unique_keys


# ------------------------------------------------------------------ model
class DVHHead(torch.nn.Module):
    """Small MLP that takes pooled per-organ features and outputs one
    logit per endpoint: P(|D_X_pred - D_X_true| < tau)."""

    def __init__(self, in_dim=12, hidden=32, n_endpoints=3):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            Linear(in_dim, hidden), torch.nn.ReLU(),
            Linear(hidden, hidden), torch.nn.ReLU(),
            Linear(hidden, n_endpoints),
        )

    def forward(self, x):
        return self.mlp(x)


class DoseGNN(torch.nn.Module):
    """GraphConv trunk with per-node dose + trust heads and a per-organ
    DVH head."""

    def __init__(self, in_dim, hidden=128, layers=4, dropout=0.1,
                 n_organs=0, organ_dim=16, dvh_hidden=32):
        super().__init__()
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

        self.trunk = torch.nn.Sequential(
            Linear(hidden, hidden), torch.nn.ReLU()
        )
        self.head_dose  = Linear(hidden, 1)
        self.head_trust = Linear(hidden, 1)

        # 9 dose stats + 3 trust stats = 12 features per organ group
        self.dvh_head = DVHHead(in_dim=12, hidden=dvh_hidden, n_endpoints=3)

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
        dose = self.head_dose(h).view(-1)
        trust = self.head_trust(h).view(-1)
        return dose, trust


# ----------------------------------------------------------------- losses
def weighted_loss(pred, target, w):
    r = pred - target
    huber = torch.where(r.abs() < 1.0, 0.5 * r ** 2, r.abs() - 0.5)
    return (w * huber).sum() / w.sum().clamp(min=1e-8)


def trust_loss(trust_logit, dose_pred, target, censored, tau=0.2):
    """Per-node BCE, unchanged from the original architecture."""
    with torch.no_grad():
        within_tol = (dose_pred - target).abs() < tau
        label = within_tol & (~censored)
    return F.binary_cross_entropy_with_logits(trust_logit, label.float())


def make_organ_key(batch_idx, organ_id):
    """Combine (batch index, organ id) into one int64 key."""
    return batch_idx.to(torch.long) * 100000 + organ_id.to(torch.long)


def dvh_features_and_labels(dose, trust, dose_true, volume, organ_id, batch_idx,
                            tau=0.1):
    """
    Pool dose, trust and true dose per (batch, organ) group.
    Returns:
        feats:  (n_groups, 12) input features for the DVH head
        labels: (n_groups, 3) binary targets
        valid:  (n_groups,) mask over groups with enough nodes and
                meaningful dose
    """
    key = make_organ_key(batch_idx, organ_id)
    dose_stats, _  = pool_group_stats(dose,       volume, key)
    trust_stats, _ = pool_group_stats(trust,      volume, key)
    true_stats, _  = pool_group_stats(dose_true,  volume, key)

    feats = torch.cat([
        dose_stats,                  # 9 columns
        trust_stats[:, [0, 1, 3]],   # mean, min, std
    ], dim=1)

    d_pred = dose_stats[:, 4:7]      # D95, D50, D2 (predicted)
    d_true = true_stats[:, 4:7]
    labels = ((d_pred - d_true).abs() < tau).float()

    n_nodes  = torch.exp(dose_stats[:, 8])
    max_true = true_stats[:, 2]
    valid = (n_nodes >= 20) & (max_true >= -10.0)

    return feats, labels, valid


def dvh_loss(dvh_head, dose, trust, dose_true, volume, organ_id, batch_idx,
             tau=0.1):
    """DVH-head BCE. Detaches dose/trust so the head trains on frozen
    features; the dose head is not pulled by this loss."""
    with torch.no_grad():
        feats, labels, valid = dvh_features_and_labels(
            dose.detach(), trust.detach(), dose_true,
            volume, organ_id, batch_idx, tau=tau)
    if not valid.any():
        return torch.zeros((), device=dose.device, requires_grad=True)
    logits = dvh_head(feats[valid])
    return F.binary_cross_entropy_with_logits(logits, labels[valid])


# ----------------------------------------------------------------- eval
@torch.no_grad()
def save_distance_error_csv(model, loader, device, out_path, organ_name_fn):
    model.eval()
    rows = []
    for batch in loader:
        batch = batch.to(device)
        dose, trust_logit = model(batch)
        pred = dose.cpu()
        trust = trust_logit.cpu()
        true = batch.y.cpu()
        w    = batch.w.cpu()
        dist = batch.raw_dist.cpu()
        vol  = batch.raw_vol.cpu()
        pos  = batch.raw_pos.cpu()
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
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"      Distance-error CSV saved: {out_path}  ({len(rows)} rows)")


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

    labels_all = ((errs < tau) & (~cens)).float()
    if labels_all.min() == labels_all.max():
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels_all.numpy(), probs.numpy()))

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
        mae_t[p] = (float(errs[keep_p].mean()) if keep_p.any()
                    else float("nan"))
    return {"auc": auc, "auc_nocens": auc_nocens, "n": int(len(errs)),
            "frac_trusted": frac, "mae_trusted": mae_t}


@torch.no_grad()
def evaluate_dvh(model, loader, device, tau=0.1):
    """
    Compute AUC of the DVH head's probability against the per-endpoint
    label, pooled across all test groups.
    Returns a dict: {"D95": auc, "D50": auc, "D2": auc, "n": n_groups}
    """
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        dose, trust = model(batch)
        feats, labels, valid = dvh_features_and_labels(
            dose.detach(), trust.detach(), batch.y,
            batch.raw_vol, batch.organ_id, batch.batch, tau=tau)
        if not valid.any():
            continue
        logits = model.dvh_head(feats[valid]).cpu()
        all_logits.append(logits)
        all_labels.append(labels[valid].cpu())

    if not all_logits:
        return {"D95": float("nan"), "D50": float("nan"), "D2": float("nan"),
                "n": 0}

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    out = {"n": int(len(labels))}
    names = ["D95", "D50", "D2"]
    for j, name in enumerate(names):
        y = labels[:, j]
        if y.min() == y.max():
            out[name] = float("nan")
        else:
            out[name] = float(roc_auc_score(y, probs[:, j]))
    return out


@torch.no_grad()
def evaluate(model, loader, device):
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
        m = w > 0
        if m.any():
            all_err.append(err[m].cpu())
    mae_w = tot_wae / max(tot_w, 1e-8)
    e = torch.cat(all_err) if all_err else torch.tensor([0.0])
    return mae_w, float(e.mean()), float(e.median())


@torch.no_grad()
def evaluate_per_organ(model, loader, device):
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
    model.eval()
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
        for t_id, e, wt in zip(target_ids, err.cpu().numpy(),
                               w.cpu().numpy()):
            if wt <= 0:
                continue
            key = (source, int(t_id))
            if key not in stats:
                stats[key] = [0.0, 0.0]
            stats[key][0] += e * wt
            stats[key][1] += wt
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
    ap.add_argument("--tau", type=float, default=0.2,
                    help="per-node trust tolerance")
    ap.add_argument("--dvh-tau", type=float, default=0.1,
                    help="per-endpoint DVH tolerance; label is 1 if "
                         "|D_X_pred - D_X_true| < dvh_tau")
    ap.add_argument("--lambda-trust", type=float, default=0.5)
    ap.add_argument("--lambda-dvh", type=float, default=0.5,
                    help="weight of the DVH head loss")
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
    results = {}
    trust_results = {}
    dvh_results = {}

    out_dir = (f"/content/drive/MyDrive/res3/hidden{a.hidden}_layers{a.layers}"
               f"_lr{a.lr}_bs{a.batch_size}_odim{a.organ_dim}"
               f"{'_age10plus' if a.exclude_young else ''}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created directory: {out_dir}")

    for fi, held in enumerate(folds):
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
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                           T_max=a.epochs)
        scaler = torch.cuda.amp.GradScaler(
            enabled=(a.amp and device == "cuda"))

        tl = DataLoader(train, batch_size=a.batch_size, shuffle=True)
        vl_val = DataLoader(val, batch_size=1)
        vl_test = DataLoader(test, batch_size=1)

        best_val = float("inf")
        best_state = None
        best_ep = 0
        patience_left = a.patience

        out_csv = f"{out_dir}/held_{held}_val{val_phantom}.csv"
        with open(out_csv, "w") as f:
            f.write("epoch,train_loss,train_mae,val_mae,test_mae,"
                    "best_val,patience_left\n")

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
                                cens = torch.zeros_like(dose,
                                                        dtype=torch.bool)
                            loss = loss + a.lambda_trust * trust_loss(
                                trust_logit, dose, batch.y, cens,
                                tau=a.tau)
                    scaler.scale(loss).backward()

                    if a.lambda_dvh > 0:
                        # DVH head trains on detached pooled features
                        dvh_l = dvh_loss(model.dvh_head,
                                         dose.detach(), trust_logit.detach(),
                                         batch.y, batch.raw_vol,
                                         batch.organ_id, batch.batch,
                                         tau=a.dvh_tau)
                        scaler.scale(a.lambda_dvh * dvh_l).backward()

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
                    if a.lambda_dvh > 0:
                        dvh_l = dvh_loss(model.dvh_head,
                                         dose.detach(), trust_logit.detach(),
                                         batch.y, batch.raw_vol,
                                         batch.organ_id, batch.batch,
                                         tau=a.dvh_tau)
                        (a.lambda_dvh * dvh_l).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                tot += float(loss.detach())
            sched.step()

            train_loss = tot / len(tl)

            if (ep % a.eval_every == 0) or (ep == a.epochs):
                train_mae, _, _ = evaluate(model, tl, device)
                vmw, _, _ = evaluate(model, vl_val, device)
                tmw, _, _ = evaluate(model, vl_test, device)
                per_organ = evaluate_per_organ(model, vl_test, device)

                trust_stats = None
                if a.lambda_trust > 0:
                    trust_stats = evaluate_trust(model, vl_test, device,
                                                 tau=a.tau)
                dvh_stats = None
                if a.lambda_dvh > 0:
                    dvh_stats = evaluate_dvh(model, vl_test, device,
                                             tau=a.dvh_tau)

                if vmw < best_val - 1e-4:
                    best_val = vmw
                    best_ep = ep
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    patience_left = a.patience
                else:
                    patience_left -= 1

                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
                      f"ep {ep:3d} | train_loss {train_loss:.4f} | "
                      f"train_mae {train_mae:.3f} | "
                      f"val {vmw:.3f} | test {tmw:.3f} dec "
                      f"(best val {best_val:.3f} @ep{best_ep})")

                print("    Test per-source-organ:")
                for org, mae in sorted(per_organ.items(),
                                       key=lambda x: -x[1]):
                    print(f"      {org:20s}  {mae:.3f} dec (x{10**mae:.2f})")

                if trust_stats is not None:
                    print(f"    trust: AUC {trust_stats['auc']:.3f}  "
                          f"AUC_nc {trust_stats['auc_nocens']:.3f}")
                if dvh_stats is not None:
                    print(f"    DVH head: AUC D95 {dvh_stats['D95']:.3f}  "
                          f"D50 {dvh_stats['D50']:.3f}  "
                          f"D2 {dvh_stats['D2']:.3f}  "
                          f"(n_groups {dvh_stats['n']})")

                with open(out_csv, "a") as f:
                    f.write(f"{ep},{train_loss:.6f},{train_mae:.6f},"
                            f"{vmw:.6f},{tmw:.6f},"
                            f"{best_val:.6f},{patience_left}\n")

                organ_csv = out_csv.replace(".csv", "_per_organ.csv")
                write_header = not os.path.exists(organ_csv)
                with open(organ_csv, "a") as f:
                    if write_header:
                        f.write("epoch,source_organ,weighted_mae\n")
                    for org, mae in per_organ.items():
                        f.write(f"{ep},{org},{mae:.6f}\n")
            else:
                with open(out_csv, "a") as f:
                    f.write(f"{ep},{train_loss:.6f},,,"
                            f"{best_val:.6f},{patience_left}\n")

            if patience_left <= 0:
                print(f"  early stop at ep {ep} "
                      f"(no val improvement for {a.patience} epochs)")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        mw, mu, md = evaluate(model, vl_test, device)
        results[held] = (mw, mu, md)

        if a.lambda_trust > 0:
            trust_stats = evaluate_trust(model, vl_test, device, tau=a.tau)
            trust_results[held] = trust_stats
            print(f"  --> trust: AUC {trust_stats['auc']:.3f}  "
                  f"AUC_nc {trust_stats['auc_nocens']:.3f}")
        if a.lambda_dvh > 0:
            dvh_stats = evaluate_dvh(model, vl_test, device, tau=a.dvh_tau)
            dvh_results[held] = dvh_stats
            print(f"  --> DVH head: AUC D95 {dvh_stats['D95']:.3f}  "
                  f"D50 {dvh_stats['D50']:.3f}  "
                  f"D2 {dvh_stats['D2']:.3f}  "
                  f"(n_groups {dvh_stats['n']})")

        dist_csv = f"{out_dir}/dist_error_{held}.csv"
        save_distance_error_csv(model, vl_test, device, dist_csv, organ_name)

        matrix = evaluate_source_target_matrix(model, vl_test, device)
        matrix_csv = f"{out_dir}/matrix_{held}.csv"
        print(f"  --> {held}: weighted MAE {mw:.3f} decades "
              f"(x{10**mw:.2f} fold), median {md:.3f}  [ep {best_ep}]")
        with open(matrix_csv, "w") as f:
            f.write("source_organ,target_organ,weighted_mae\n")
            for src in sorted(matrix.keys()):
                for tgt, mae in sorted(matrix[src].items()):
                    f.write(f"{src},{organ_name(tgt)},{mae:.6f}\n")

        if trust_results:
            trust_csv = f"{out_dir}/trust_per_phantom.csv"
            with open(trust_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["phantom", "auc", "auc_nocens", "n"])
                for ph, s in sorted(trust_results.items()):
                    writer.writerow([ph, f"{s['auc']:.6f}",
                                     f"{s['auc_nocens']:.6f}", s["n"]])
            print(f"\nPer-phantom trust summary written to {trust_csv}")

        if dvh_results:
            dvh_csv = f"{out_dir}/dvh_head_per_phantom.csv"
            with open(dvh_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["phantom", "auc_D95", "auc_D50", "auc_D2",
                                 "n_groups"])
                for ph, s in sorted(dvh_results.items()):
                    writer.writerow([
                        ph,
                        f"{s['D95']:.6f}",
                        f"{s['D50']:.6f}",
                        f"{s['D2']:.6f}",
                        s["n"],
                    ])
            print(f"Per-phantom DVH head summary written to {dvh_csv}")

    print("\n" + "=" * 62)
    print("LEAVE-ONE-PHANTOM-OUT SUMMARY")
    for k, (mw, mu, md) in results.items():
        print(f"  {k:32s} MAE(w) {mw:.3f} dec  = x{10**mw:.2f} fold error")
    if results:
        avg = np.mean([v[0] for v in results.values()])
        print(f"\n  MEAN weighted MAE: {avg:.3f} decades "
              f"(x{10**avg:.2f} fold)")
    if dvh_results:
        print("\n  DVH head mean AUC across phantoms:")
        for endpoint in ["D95", "D50", "D2"]:
            vals = [s[endpoint] for s in dvh_results.values()
                    if np.isfinite(s[endpoint])]
            if vals:
                print(f"    {endpoint}: {np.mean(vals):.3f}  "
                      f"(min {min(vals):.3f}, max {max(vals):.3f}, "
                      f"n={len(vals)})")
    print("=" * 62)


if __name__ == "__main__":
    main()