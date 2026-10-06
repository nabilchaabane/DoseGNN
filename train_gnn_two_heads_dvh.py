"""
train_gnn.py -- supervoxel dose graphs with a DVH-aware trust head
               + explicit global DVH quantile loss + clean DVH reporting.
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
import csv
from sklearn.metrics import roc_auc_score

from source_organs import SOURCE_ORGANS


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


# ------------------------------------------------------------------ DVH features
def compute_dvh_features(pred_dose, volume, organ_id, batch_idx,
                         quantiles=(5, 50, 98)):
    N = pred_dose.shape[0]
    nq = len(quantiles)
    dev = pred_dose.device
    out = torch.zeros(N, 2 + 2 * nq, device=dev, dtype=torch.float32)

    key = batch_idx * 100000 + organ_id
    for k in torch.unique(key):
        m = key == k
        idx_m = torch.where(m)[0]
        d = pred_dose[idx_m].float()
        w = volume[idx_m].float()
        n = d.numel()
        if n < 2:
            continue
        order = torch.argsort(d)
        d_sorted = d[order]
        w_sorted = w[order]
        cw = torch.cumsum(w_sorted, dim=0)
        total = cw[-1]
        if total <= 0:
            continue

        rank = torch.empty(n, device=dev, dtype=torch.float32)
        rank[order] = torch.arange(n, device=dev, dtype=torch.float32) / (n - 1)
        vol_frac = torch.empty(n, device=dev, dtype=torch.float32)
        vol_frac[order] = cw / total

        out[idx_m, 0] = rank
        out[idx_m, 1] = vol_frac

        for j, q in enumerate(quantiles):
            target = (q / 100.0) * total
            pos = int(torch.searchsorted(cw, target).clamp(max=n - 1).item())
            qval = d_sorted[pos]
            out[idx_m, 2 + j] = d - qval
            out[idx_m, 2 + nq + j] = qval
    return out.to(pred_dose.dtype)


# ------------------------------------------------------------------ DVH quantile loss (global soft constraint)
def dvh_quantile_loss(pred, target, volume, organ_id, batch_idx,
                      quantiles=(5, 50, 98), reduction="mean"):
    device = pred.device
    key = batch_idx * 100_000 + organ_id
    losses = []

    for k in torch.unique(key):
        m = key == k
        d_pred = pred[m]
        d_true = target[m]
        w = volume[m].float()
        n = d_pred.numel()
        if n < 2 or w.sum() <= 0:
            continue

        order_p = torch.argsort(d_pred)
        cw_p = torch.cumsum(w[order_p], dim=0)
        total = cw_p[-1]

        order_t = torch.argsort(d_true)
        cw_t = torch.cumsum(w[order_t], dim=0)

        for q in quantiles:
            target_vol = (q / 100.0) * total
            pos_p = torch.searchsorted(cw_p, target_vol).clamp(max=n - 1)
            q_pred = d_pred[order_p][pos_p]
            pos_t = torch.searchsorted(cw_t, target_vol).clamp(max=n - 1)
            q_true = d_true[order_t][pos_t]
            losses.append((q_pred - q_true).abs())

    if not losses:
        return torch.tensor(0.0, device=device)
    losses = torch.stack(losses)
    return losses.mean() if reduction == "mean" else losses.sum()


# ----------------------------------------------------------------- model
class DoseGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden=128, layers=4, dropout=0.1,
                 n_organs=0, organ_dim=16, dvh_feat_dim=8):
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
        self.head_trust = Linear(hidden + dvh_feat_dim, 1)

        self.dropout = dropout
        self.dvh_feat_dim = dvh_feat_dim

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

        raw_vol  = getattr(data, "raw_vol",  None)
        organ_id = getattr(data, "organ_id", None)
        batch_idx = getattr(data, "batch", None)
        if raw_vol is None or organ_id is None or batch_idx is None:
            dvh_feats = torch.zeros(h.shape[0], self.dvh_feat_dim,
                                    device=h.device, dtype=h.dtype)
        else:
            dvh_feats = compute_dvh_features(
                dose.detach(), raw_vol, organ_id, batch_idx)

        h_trust = torch.cat([h, dvh_feats], dim=1)
        trust = self.head_trust(h_trust).view(-1)
        vol_frac = dvh_feats[:, 1]
        return dose, trust, vol_frac


# ----------------------------------------------------------------- losses
def weighted_loss(pred, target, w):
    r = pred - target
    huber = torch.where(r.abs() < 1.0, 0.5 * r ** 2, r.abs() - 0.5)
    return (w * huber).sum() / w.sum().clamp(min=1e-8)


def dvh_proximity_weight(vol_frac, quantiles=(0.05, 0.50, 0.98), sigma=0.02):
    acc = torch.zeros_like(vol_frac)
    for q in quantiles:
        acc = acc + torch.exp(-((vol_frac - q) ** 2) / (2 * sigma ** 2))
    return acc.clamp(max=1.0)


def trust_loss(trust_logit, dose_pred, target, censored, vol_frac,
               tau=0.2, lambda_dvh=5.0, sigma=0.02):
    with torch.no_grad():
        within_tol = (dose_pred - target).abs() < tau
        label = within_tol & (~censored)
    prox = dvh_proximity_weight(vol_frac, sigma=sigma)
    node_w = 1.0 + lambda_dvh * prox
    bce = F.binary_cross_entropy_with_logits(
        trust_logit, label.float(), reduction="none")
    return (node_w * bce).sum() / node_w.sum().clamp(min=1e-8)


# ----------------------------------------------------------------- eval
@torch.no_grad()
def evaluate_dvh_detailed(model, loader, device, quantiles=(5, 50, 98)):
    """
    Returns a dict with mean absolute error for each quantile
    and the overall mean.
    """
    model.eval()
    errs_per_q = {q: [] for q in quantiles}

    for batch in loader:
        batch = batch.to(device)
        pred, _, _ = model(batch)

        key = batch.batch * 100_000 + batch.organ_id
        for k in torch.unique(key):
            m = key == k
            d_pred = pred[m]
            d_true = batch.y[m]
            w = batch.raw_vol[m].float()
            n = d_pred.numel()
            if n < 2 or w.sum() <= 0:
                continue

            order_p = torch.argsort(d_pred)
            cw_p = torch.cumsum(w[order_p], dim=0)
            total = cw_p[-1]

            order_t = torch.argsort(d_true)
            cw_t = torch.cumsum(w[order_t], dim=0)

            for q in quantiles:
                target_vol = (q / 100.0) * total
                pos_p = torch.searchsorted(cw_p, target_vol).clamp(max=n-1)
                q_pred = d_pred[order_p][pos_p]
                pos_t = torch.searchsorted(cw_t, target_vol).clamp(max=n-1)
                q_true = d_true[order_t][pos_t]
                errs_per_q[q].append(float((q_pred - q_true).abs()))

    result = {}
    all_errs = []
    for q in quantiles:
        if errs_per_q[q]:
            mean_q = float(np.mean(errs_per_q[q]))
            result[f"D{q}"] = mean_q
            all_errs.extend(errs_per_q[q])
        else:
            result[f"D{q}"] = float("nan")

    result["mean"] = float(np.mean(all_errs)) if all_errs else float("nan")
    return result


@torch.no_grad()
def save_distance_error_csv(model, loader, device, out_path, organ_name_fn):
    model.eval()
    rows = []
    for batch in loader:
        batch = batch.to(device)
        dose, trust_logit, _ = model(batch)
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
        dose, trust_logit, _ = model(batch)
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
def evaluate(model, loader, device):
    model.eval()
    tot_w = tot_wae = 0.0
    all_err = []
    for batch in loader:
        batch = batch.to(device)
        pred, _, _ = model(batch)
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
        pred, _, _ = model(batch)
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
        pred, _, _ = model(batch)
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
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--lambda-trust", type=float, default=0.5)
    ap.add_argument("--lambda-dvh", type=float, default=5.0)
    ap.add_argument("--lambda-dvh-q", type=float, default=2.0,
                    help="weight of the global DVH quantile residual loss")
    ap.add_argument("--dvh-sigma", type=float, default=0.02)
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
    dvh_results = {}      # detailed DVH errors
    trust_results = {}

    out_dir = (f"/content/drive/MyDrive/res2/hidden{a.hidden}_layers{a.layers}"
               f"_lr{a.lr}_bs{a.batch_size}_odim{a.organ_dim}"
               f"_ldvhq{a.lambda_dvh_q}"
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

        out_csv = f"{out_dir}/held_{held}_val{val_phantom}.csv"
        with open(out_csv, "w") as f:
            f.write("epoch,train_loss,train_mae,val_mae,test_mae,"
                    "best_val,patience_left,D5,D50,D98,DVH_mean\n")

        for ep in range(1, a.epochs + 1):
            model.train()
            tot = 0.0
            for batch in tl:
                batch = batch.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                if scaler.is_enabled():
                    with torch.cuda.amp.autocast():
                        dose, trust_logit, vol_frac = model(batch)
                        loss = weighted_loss(dose, batch.y, batch.w)
                        if a.lambda_dvh_q > 0:
                            loss = loss + a.lambda_dvh_q * dvh_quantile_loss(
                                dose, batch.y, batch.raw_vol,
                                batch.organ_id, batch.batch)
                        if a.lambda_trust > 0:
                            cens = getattr(batch, "censored", None)
                            if cens is None:
                                cens = torch.zeros_like(dose, dtype=torch.bool)
                            loss = loss + a.lambda_trust * trust_loss(
                                trust_logit, dose, batch.y, cens, vol_frac,
                                tau=a.tau, lambda_dvh=a.lambda_dvh,
                                sigma=a.dvh_sigma)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    dose, trust_logit, vol_frac = model(batch)
                    loss = weighted_loss(dose, batch.y, batch.w)
                    if a.lambda_dvh_q > 0:
                        loss = loss + a.lambda_dvh_q * dvh_quantile_loss(
                            dose, batch.y, batch.raw_vol,
                            batch.organ_id, batch.batch)
                    if a.lambda_trust > 0:
                        cens = getattr(batch, "censored", None)
                        if cens is None:
                            cens = torch.zeros_like(dose, dtype=torch.bool)
                        loss = loss + a.lambda_trust * trust_loss(
                            trust_logit, dose, batch.y, cens, vol_frac,
                            tau=a.tau, lambda_dvh=a.lambda_dvh,
                            sigma=a.dvh_sigma)
                    loss.backward()
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

                # Detailed DVH errors on the current test set
                dvh = evaluate_dvh_detailed(model, vl_test, device)

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

                # DVH errors on the current test set
                print(f"    DVH  D5={dvh['D5']:.3f}  D50={dvh['D50']:.3f}  "
                    f"D98={dvh['D98']:.3f}  mean={dvh['mean']:.3f} dec")

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
                            f"{vmw:.6f},{tmw:.6f},{best_val:.6f},{patience_left},"
                            f"{dvh['D5']:.6f},{dvh['D50']:.6f},{dvh['D98']:.6f},"
                            f"{dvh['mean']:.6f}\n")
            else:
                with open(out_csv, "a") as f:
                    f.write(f"{ep},{train_loss:.6f},,,,{best_val:.6f},{patience_left},,,,\n")

            if patience_left <= 0:
                print(f"  early stop at ep {ep}")
                break

        # ---------- final evaluation for this held-out phantom ----------
        if best_state is not None:
            model.load_state_dict(best_state)

        mw, mu, md = evaluate(model, vl_test, device)
        results[held] = mw

        dvh = evaluate_dvh_detailed(model, vl_test, device)
        dvh_results[held] = dvh

        print(f"  --> {held}: MAE {mw:.3f} dec | "
              f"DVH  D5={dvh['D5']:.3f}  D50={dvh['D50']:.3f}  "
              f"D98={dvh['D98']:.3f}  mean={dvh['mean']:.3f}")

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

        # ----- DVH mean quantile error -----
        dvh_means = {ph: dvh_results[ph]["mean"] for ph in dvh_results}
        best_dvh = min(dvh_means, key=dvh_means.get)
        worst_dvh = max(dvh_means, key=dvh_means.get)
        print(f"\nDVH mean quantile error:")
        print(f"  BEST  {best_dvh:32s}  {dvh_means[best_dvh]:.3f} dec "
            f"(D5={dvh_results[best_dvh]['D5']:.3f}  "
            f"D50={dvh_results[best_dvh]['D50']:.3f}  "
            f"D98={dvh_results[best_dvh]['D98']:.3f})")
        print(f"  WORST {worst_dvh:32s}  {dvh_means[worst_dvh]:.3f} dec "
            f"(D5={dvh_results[worst_dvh]['D5']:.3f}  "
            f"D50={dvh_results[worst_dvh]['D50']:.3f}  "
            f"D98={dvh_results[worst_dvh]['D98']:.3f})")
        print(f"  MEAN  {'':32s}  {np.mean(list(dvh_means.values())):.3f} dec")

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