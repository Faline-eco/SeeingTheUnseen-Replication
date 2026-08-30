"""Train an anchor-free detection head on DINOv3 patch embeddings.

Answers whether a detector reading *embeddings* of a rendering does as well as
one reading its pixels. Consumes the `.npy` grids from `encode_embeddings.py`
and reuses the **exact label files** of the matching YOLO dataset, so the split,
the frame set and the ground truth are identical to the pixel-space runs.
mAP is computed with ultralytics' own `ap_per_class`, so the numbers drop
straight into the comparison table.

Resolution caveat
-----------------
DINOv3 is patch-16: a 1024x1024 render becomes a 64x64 grid, and an animal here
measures ~1.2 x 1.1 cells. A head predicting directly on that grid would be
reporting patch stride, not embedding quality, so a small decoder upsamples
64 -> 256 (effective stride 4) before the heads — comparable to the finest level
a YOLO detector uses. The features are still stride-16 in origin; the decoder
only lets the head localise within a cell.

    py scripts\\train_embedding_detector.py --dataset alfs_thermal
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------------- data ----

class EmbeddingDet(Dataset):
    """(embedding grid, boxes) pairs; boxes are YOLO cx cy w h, normalised."""

    def __init__(self, emb_dir: Path, label_dir: Path, out_hw: int,
                 stems: set[str] | None = None, use_dims: int = 0):
        self.emb_dir = emb_dir
        self.out_hw = out_hw
        # PCA components are nested: the first K columns of a D-dim encoding are
        # exactly the K-dim encoding of the same basis. Truncating here gives a
        # lower-dimensionality arm for free, so resolution and dimensionality can
        # be varied one at a time instead of together.
        self.use_dims = use_dims
        self.items: list[tuple[Path, Path]] = []
        for lp in sorted(label_dir.glob("*.txt")):
            if stems is not None and lp.stem not in stems:
                continue
            ep = emb_dir / f"{lp.stem}.npy"
            if ep.exists():
                self.items.append((ep, lp))
        if not self.items:
            raise SystemExit(f"no (embedding, label) pairs for {emb_dir} / {label_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ep, lp = self.items[i]
        emb = np.load(ep).astype(np.float32)              # (H, W, D)
        if self.use_dims:
            emb = emb[..., :self.use_dims]
        x = torch.from_numpy(emb).permute(2, 0, 1)        # (D, H, W)
        boxes = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 5:
                boxes.append([float(v) for v in p[1:]])
        return x, torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), i


def collate(batch):
    xs, bs, idx = zip(*batch)
    return torch.stack(xs), list(bs), list(idx)


# ---------------------------------------------------------------- model ----

def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.SiLU(inplace=True),
    )


class EmbeddingDetector(nn.Module):
    """Decoder + CenterNet-style heads over a patch-embedding grid.

    Heads follow CenterNet: a Gaussian centre heatmap, a sub-cell offset, and
    box size. That suits targets roughly one feature cell across far better than
    an anchor-based head, which would need anchors smaller than the stride.
    """

    def __init__(self, in_dim: int, width: int = 128, up: int = 4):
        super().__init__()
        self.stem = _block(in_dim, width)
        ups = []
        c = width
        for _ in range(int(math.log2(up))):
            ups += [nn.ConvTranspose2d(c, c // 2, 2, stride=2),
                    nn.BatchNorm2d(c // 2), nn.SiLU(inplace=True)]
            c //= 2
        self.up = nn.Sequential(*ups)
        self.refine = _block(c, c)
        self.hm = nn.Conv2d(c, 1, 1)
        self.wh = nn.Conv2d(c, 2, 1)
        self.off = nn.Conv2d(c, 2, 1)
        # Heatmap prior: most cells are background (CenterNet convention).
        nn.init.constant_(self.hm.bias, -4.6)

    def forward(self, x):
        f = self.refine(self.up(self.stem(x)))
        return self.hm(f), self.wh(f).sigmoid(), self.off(f).sigmoid()


# --------------------------------------------------------------- targets ---

def build_targets(boxes_list, hw: int, device):
    """CenterNet targets: Gaussian heatmap + wh/offset at object centres."""
    b = len(boxes_list)
    hm = torch.zeros(b, 1, hw, hw, device=device)
    wh = torch.zeros(b, 2, hw, hw, device=device)
    off = torch.zeros(b, 2, hw, hw, device=device)
    mask = torch.zeros(b, 1, hw, hw, device=device)
    for i, boxes in enumerate(boxes_list):
        for cx, cy, w, h in boxes.tolist():
            fx, fy = cx * hw, cy * hw
            ix, iy = int(min(fx, hw - 1)), int(min(fy, hw - 1))
            # Gaussian radius from box size, floor 1 cell (targets are tiny).
            r = max(1, int(0.3 * max(w, h) * hw))
            y0, y1 = max(0, iy - r), min(hw, iy + r + 1)
            x0, x1 = max(0, ix - r), min(hw, ix + r + 1)
            if y1 > y0 and x1 > x0:
                yy = torch.arange(y0, y1, device=device).view(-1, 1)
                xx = torch.arange(x0, x1, device=device).view(1, -1)
                g = torch.exp(-((yy - iy) ** 2 + (xx - ix) ** 2) / (2 * (r / 3 + 1e-6) ** 2))
                hm[i, 0, y0:y1, x0:x1] = torch.maximum(hm[i, 0, y0:y1, x0:x1], g)
            hm[i, 0, iy, ix] = 1.0
            wh[i, :, iy, ix] = torch.tensor([w, h], device=device)
            off[i, :, iy, ix] = torch.tensor([fx - ix, fy - iy], device=device)
            mask[i, 0, iy, ix] = 1.0
    return hm, wh, off, mask


def focal_loss(pred_logits, gt):
    """CenterNet penalty-reduced focal loss."""
    p = pred_logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = -((1 - p) ** 2) * torch.log(p) * pos
    neg_loss = -((1 - gt) ** 4) * (p ** 2) * torch.log(1 - p) * neg
    n = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n


# ---------------------------------------------------------------- decode ---

@torch.no_grad()
def decode(hm_logits, wh, off, topk: int = 100, thr: float = 0.01):
    """Peaks -> xyxy boxes in normalised coordinates, with scores."""
    b, _, h, w = hm_logits.shape
    hm = hm_logits.sigmoid()
    keep = (F.max_pool2d(hm, 3, stride=1, padding=1) == hm).float()
    hm = hm * keep
    out = []
    for i in range(b):
        flat = hm[i, 0].reshape(-1)
        k = min(topk, flat.numel())
        scores, idx = flat.topk(k)
        sel = scores > thr
        scores, idx = scores[sel], idx[sel]
        ys, xs = (idx // w).float(), (idx % w).float()
        ox, oy = off[i, 0].reshape(-1)[idx], off[i, 1].reshape(-1)[idx]
        bw, bh = wh[i, 0].reshape(-1)[idx], wh[i, 1].reshape(-1)[idx]
        cx, cy = (xs + ox) / w, (ys + oy) / h
        out.append(torch.stack([cx - bw / 2, cy - bh / 2,
                                cx + bw / 2, cy + bh / 2, scores], dim=1))
    return out


@torch.no_grad()
def evaluate(model, loader, device, gt_labels: dict | None = None):
    """mAP via ultralytics' ap_per_class, so numbers match the YOLO runs."""
    from ultralytics.utils.metrics import ap_per_class, box_iou

    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    model.eval()
    for x, boxes_list, idxs in loader:
        x = x.to(device, non_blocking=True)
        preds = decode(*model(x))
        for p, boxes, di in zip(preds, boxes_list, idxs):
            if gt_labels is not None:
                boxes = gt_labels[loader.dataset.items[di][1].stem]
            gt = boxes.to(device)
            if gt.numel():
                g = torch.stack([gt[:, 0] - gt[:, 2] / 2, gt[:, 1] - gt[:, 3] / 2,
                                 gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3] / 2], 1)
            else:
                g = torch.zeros(0, 4, device=device)
            stats["target_cls"].append(torch.zeros(len(g)))
            if not len(p):
                continue
            stats["conf"].append(p[:, 4].cpu())
            stats["pred_cls"].append(torch.zeros(len(p)))
            correct = torch.zeros(len(p), len(iouv), dtype=torch.bool)
            if len(g):
                iou = box_iou(g, p[:, :4])
                for k, t in enumerate(iouv):
                    y, xi = torch.where(iou >= t)
                    if len(y):
                        m = torch.cat([torch.stack([y, xi], 1),
                                       iou[y, xi][:, None]], 1).cpu().numpy()
                        m = m[m[:, 2].argsort()[::-1]]
                        m = m[np.unique(m[:, 1], return_index=True)[1]]
                        m = m[np.unique(m[:, 0], return_index=True)[1]]
                        correct[m[:, 1].astype(int), k] = True
            stats["tp"].append(correct)
    if not stats["conf"]:
        return {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0}
    tp = torch.cat(stats["tp"]).numpy()
    conf = torch.cat(stats["conf"]).numpy()
    pcls = torch.cat(stats["pred_cls"]).numpy()
    tcls = torch.cat(stats["target_cls"]).numpy()
    res = ap_per_class(tp, conf, pcls, tcls, plot=False, names={0: "animal"})
    # Contract: (tp, fp, p, r, f1, ap, unique_classes, p_curve, ...). Unpack by
    # name -- indexing positionally reads unique_classes as `ap` and fails only
    # later, inside the metric arithmetic.
    _tp, _fp, p, r, _f1, ap = res[0], res[1], res[2], res[3], res[4], res[5]
    ap = np.atleast_2d(ap)                      # (n_classes, n_iou_thresholds)
    return {"mAP50": float(ap[:, 0].mean()), "mAP50-95": float(ap.mean()),
            "precision": float(np.mean(p)), "recall": float(np.mean(r))}


# ------------------------------------------------------------------ main ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="alfs_thermal",
                    help="Name shared by the embedding dir and the YOLO dataset.")
    ap.add_argument("--labels-dataset", default=None,
                    help="Take labels from this YOLO dataset instead of "
                         "--dataset. Needed for the resolution arm: its "
                         "embeddings live under the 1024px name, but every arm "
                         "must be scored against one common ground truth "
                         "(the 1024px and 2048px label sets differ by ~1% of "
                         "box size from integer-pixel rounding).")
    ap.add_argument("--embeddings", type=Path, default=Path("D:/embeddings"))
    ap.add_argument("--yolo-datasets", type=Path, default=Path("D:/yolo_datasets"))
    ap.add_argument("--eval-labels", default=None,
                    help="YOLO dataset whose val labels to score against "
                         "(e.g. alfs_thermal_centralgt for a like-for-like "
                         "comparison with the ortho/raw detectors).")
    ap.add_argument("--out", type=Path, default=Path("D:/embedding_runs"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--up", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--stems-file", type=Path, default=None,
                    help="Restrict to these '<flight>_<frame>' stems. Used to "
                         "hold the frame set identical across a resolution "
                         "comparison, so only the patch stride differs.")
    ap.add_argument("--tag", default=None,
                    help="Run name suffix, e.g. '1x' / '2x'.")
    ap.add_argument("--use-dims", type=int, default=0,
                    help="Use only the first K PCA components of the cached "
                         "embeddings (0 = all). PCA components are nested, so "
                         "this yields the K-dim arm without re-encoding — the "
                         "way to vary dimensionality and render resolution "
                         "independently rather than confounding them.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_dir = args.embeddings / args.dataset
    lab_ds = args.labels_dataset or args.dataset
    lab = args.yolo_datasets / lab_ds / "labels"

    stems = None
    if args.stems_file:
        stems = {l.strip() for l in args.stems_file.read_text().splitlines() if l.strip()}
    tr = EmbeddingDet(emb_dir, lab / "train", 0, stems, args.use_dims)
    va = EmbeddingDet(emb_dir, lab / "val", 0, stems, args.use_dims)
    stored_dim = np.load(tr.items[0][0]).shape[-1]
    if args.use_dims > stored_dim:
        raise SystemExit(f"--use-dims {args.use_dims} exceeds the {stored_dim} "
                         f"dims stored in {emb_dir}")
    in_dim = args.use_dims or stored_dim
    grid = np.load(tr.items[0][0]).shape[0]
    out_hw = grid * args.up
    trunc = f" (truncated from {stored_dim})" if args.use_dims else ""
    print(f"{args.dataset}: {len(tr)} train / {len(va)} val | "
          f"grid {grid}x{grid}x{in_dim}{trunc} -> head {out_hw}x{out_hw}", flush=True)

    # Optional like-for-like GT (e.g. central-frame boxes) for the val pass.
    gt_labels = None
    if args.eval_labels:
        gt_labels = {}
        for lp in (args.yolo_datasets / args.eval_labels / "labels" / "val").glob("*.txt"):
            rows = [[float(v) for v in l.split()[1:5]]
                    for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
            gt_labels[lp.stem] = torch.tensor(rows, dtype=torch.float32).reshape(-1, 4)
        print(f"  scoring against {args.eval_labels} labels "
              f"({sum(len(v) for v in gt_labels.values())} boxes)", flush=True)

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate,
                       num_workers=args.workers, pin_memory=True)

    model = EmbeddingDetector(in_dim, args.width, args.up).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device)
    print(f"  head: {n_par/1e6:.2f} M params", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    run = args.out / (f"{args.dataset}_{args.tag}" if args.tag else args.dataset)
    run.mkdir(parents=True, exist_ok=True)
    best, best_ep, hist = -1.0, -1, []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, boxes_list, _ in dl_tr:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device):
                hm, wh, off = model(x)
                thm, twh, toff, m = build_targets(boxes_list, hm.shape[-1], device)
                l_hm = focal_loss(hm.float(), thm)
                n = m.sum().clamp(min=1)
                l_wh = (F.l1_loss(wh.float() * m, twh * m, reduction="sum") / n)
                l_off = (F.l1_loss(off.float() * m, toff * m, reduction="sum") / n)
                loss = l_hm + 5.0 * l_wh + l_off
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
        sched.step()

        met = evaluate(model, dl_va, device, gt_labels)
        fit = 0.1 * met["mAP50"] + 0.9 * met["mAP50-95"]
        hist.append({"epoch": ep, "loss": tot / max(len(dl_tr), 1), **met})
        print(f"  ep {ep:3d}  loss {tot/max(len(dl_tr),1):7.4f}  "
              f"mAP50 {met['mAP50']:.4f}  mAP50-95 {met['mAP50-95']:.4f}  "
              f"P {met['precision']:.4f}  R {met['recall']:.4f}", flush=True)
        if fit > best:
            best, best_ep = fit, ep
            torch.save({"model": model.state_dict(), "in_dim": in_dim,
                        "width": args.width, "up": args.up, "metrics": met},
                       run / "best.pt")
        (run / "results.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
        if ep - best_ep >= args.patience:
            print(f"  early stop (best epoch {best_ep})", flush=True)
            break

    best_met = next(h for h in hist if h["epoch"] == best_ep)
    summary = {"name": f"emb_{args.dataset}" + (f"_{args.tag}" if args.tag else ""), "mAP50": round(best_met["mAP50"], 4),
               "mAP50-95": round(best_met["mAP50-95"], 4),
               "precision": round(best_met["precision"], 4),
               "recall": round(best_met["recall"], 4),
               "best_epoch": best_ep, "epochs_run": len(hist),
               "minutes": round((time.time() - t0) / 60, 1),
               "params_M": round(n_par / 1e6, 2),
               "eval_labels": args.eval_labels or lab_ds,
               # Recorded so a run's numbers can never be compared across arms
               # without it being visible which ground truth produced them.
               "labels_dataset": lab_ds,
               "use_dims": args.use_dims or stored_dim,
               "stored_dims": stored_dim}
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBEST epoch {best_ep}: mAP50 {best_met['mAP50']:.4f} "
          f"mAP50-95 {best_met['mAP50-95']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
