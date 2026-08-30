"""Individual example figures: what each cell detects on held-out frames.

Legend, consistent across every figure this writes:

    solid outline    animal visible in the central view
    dashed outline   animal hidden in the central view (the study population)
    green            found by this representation at IoU 0.5
    red              missed by this representation

Only ground truth is drawn; pass --show-fp to add detections that match no
annotation. Every box is a coloured stroke over a wider black one, which
is what keeps the colours legible at print size and stops them being read as
part of the false-colour embedding panels underneath.

One clean PNG per representation per frame, with no titles or annotations burnt
in, so each can be dropped into a paper with its caption supplied there. A
companion markdown describes them and carries the per-cell counts.

Two things are deliberate and easy to get wrong:

  * The **detector always runs on the 128x128 cell** it was trained on, but the
    two sensor rows are *displayed* on the 2048px render. Boxes are in
    normalised coordinates, so they map to either. Showing a 128x128 cell where a
    2048px render exists throws away all the detail a reader needs to judge the
    imagery.
  * Ground truth follows `occlusion_metrics.py`: a merged box counts as occluded
    when it matches no central-frame box at IoU 0.5, and the reviewer's hidden
    flags are indexed over that occluded subset. The merged annotation also
    carries the same animal twice in places, so near-duplicate boxes are
    collapsed -- otherwise one animal gets two outlines, sometimes in two
    different colours.

Frames must belong to the chosen fold's VALIDATION split: a frame drawn with a
head that trained on it is not evidence of anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.patches import Rectangle   # noqa: E402
import matplotlib.patheffects as pe        # noqa: E402
from PIL import Image                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_embedding_detector import EmbeddingDetector, decode   # noqa: E402

# display name -> (cell the head consumes, render tree to draw on or None)
CELLS = [
    ("raw / single",    "realortho_single", "geo-referenced2_2k"),
    ("raw / multi",     "realalfs_multi",   "alfs_2k"),
    ("DINOv3 / single", "embed_single",     None),
    ("DINOv3 / multi",  "embed_multi",      None),
    ("V-JEPA / single", "vjepa_single",     None),
    ("V-JEPA / multi",  "vjepa_multi",      None),
]
SLUG = {"realortho_single": "raw-single", "realalfs_multi": "raw-multi",
        "embed_single": "dinov3-single", "embed_multi": "dinov3-multi",
        "vjepa_single": "vjepa-single", "vjepa_multi": "vjepa-multi"}


def highpass(a: np.ndarray, k: int = 17) -> np.ndarray:
    """Subtract a blurred copy of each channel.

    The leading principal components of a feature grid are dominated by absolute
    position rather than content, and the multi-view cells are worse than the
    single-view ones: averaging 31 views cancels content variation while the
    positional component, common to every view, survives intact. Removing the
    low-frequency part before projecting to colour lets the remaining structure
    show. This is a DISPLAY transform only -- the detector reads the untouched
    grid, so the boxes drawn on the figure still come from the real input.

    Two passes of a separable box blur, which is close enough to Gaussian here
    and needs no scipy.
    """
    out = a.astype(np.float32)
    pad = k // 2
    for _ in range(2):
        for _axis in range(2):
            p = np.pad(out, ((pad, pad), (0, 0), (0, 0)), mode="edge")
            c = np.concatenate([np.zeros((1,) + p.shape[1:], np.float32),
                                np.cumsum(p, 0)], 0)
            out = ((c[k:] - c[:-k]) / k).transpose(1, 0, 2)
    return a - out


def to_rgb(a: np.ndarray) -> np.ndarray:
    """(H,W,C) -> displayable RGB. 3 channels pass through; wider uses 3 PCs."""
    a = np.nan_to_num(a.astype(np.float32))
    if a.shape[-1] == 3:
        v = a
    else:
        flat = a.reshape(-1, a.shape[-1])
        flat = flat - flat.mean(0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(flat, full_matrices=False)
            v = (flat @ vt[:3].T).reshape(*a.shape[:2], 3)
        except np.linalg.LinAlgError:
            v = a[..., :3]
    lo = np.percentile(v, 1, axis=(0, 1), keepdims=True)
    hi = np.percentile(v, 99, axis=(0, 1), keepdims=True)
    return np.clip((v - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def read_yolo(p: Path) -> np.ndarray:
    if not p.exists():
        return np.zeros((0, 4), np.float32)
    out = []
    for ln in p.read_text().split("\n"):
        f = ln.split()
        if len(f) >= 5:
            cx, cy, w, h = (float(v) for v in f[1:5])
            out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.asarray(out, np.float32) if out else np.zeros((0, 4), np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    i = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (i / np.maximum(aa[:, None] + ab[None, :] - i, 1e-9)).astype(np.float32)


def ground_truth(stem, lab, cen, hidden, iou=0.5, dedup=0.45):
    gt = read_yolo(lab / f"{stem}.txt")
    if not len(gt):
        return gt, np.zeros(0, bool)
    cgt = read_yolo(cen / f"{stem}.txt")
    occ = (iou_matrix(gt, cgt).max(1) < iou if len(cgt) else np.ones(len(gt), bool))
    hid = np.zeros(len(gt), bool)
    flags = hidden.get(stem)
    if flags is not None and len(flags) == int(occ.sum()):
        for j, f in zip(np.flatnonzero(occ), flags):
            hid[j] = bool(f)
    order = np.argsort(-((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])))
    m = iou_matrix(gt, gt)
    keep, taken = [], np.zeros(len(gt), bool)
    for j in order:
        if taken[j]:
            continue
        grp = np.flatnonzero((m[j] >= dedup) & ~taken)
        taken[grp] = True
        keep.append((j, bool(hid[grp].any())))
    idx = [k for k, _ in keep]
    return gt[idx], np.asarray([h for _, h in keep], bool)


def load_head(runs: Path, mod: str, cell: str, fold: str, device):
    best, best_map = None, -1.0
    for d in sorted(runs.glob(f"cell_{mod}_{cell}_{cell}_f{fold}_s*")):
        s, c = d / "summary.json", d / "best.pt"
        if not (s.is_file() and c.is_file()):
            continue
        m = json.loads(s.read_text()).get("mAP50-95", -1)
        if m > best_map:
            best_map, best = m, c
    if best is None:
        return None
    ck = torch.load(best, map_location=device)
    model = EmbeddingDetector(ck["in_dim"], ck["width"], ck["up"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


@torch.no_grad()
def run_head(model, cell_arr, conf, device):
    x = torch.from_numpy(cell_arr.astype(np.float32)).permute(2, 0, 1)[None].to(device)
    hm, wh, off = model(x)
    d = decode(hm, wh, off)[0].cpu().numpy()
    return d[d[:, 4] >= conf] if len(d) else np.zeros((0, 5), np.float32)


def crop_window(gt, margin):
    x0 = max(0.0, gt[:, 0].min() - margin); x1 = min(1.0, gt[:, 2].max() + margin)
    y0 = max(0.0, gt[:, 1].min() - margin); y1 = min(1.0, gt[:, 3].max() + margin)
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (float(np.clip(cx - side / 2, 0, 1 - side)),
            float(np.clip(cy - side / 2, 0, 1 - side)), side)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", default="thermal", choices=("thermal", "rgb"))
    ap.add_argument("--flight", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--stems", required=True)
    ap.add_argument("--root", type=Path,
                    default=Path("/scratch/bambi/datasets/alfs_embed"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--show-fp", action="store_true",
                    help="Also draw detections matching no annotation, in "
                         "dotted yellow. Off by default: they crowd the panels "
                         "and the count is in the companion JSON anyway.")
    ap.add_argument("--lw", type=float, default=4.0,
                    help="White stroke width of the ground-truth boxes, in "
                         "points. A wider black stroke is drawn underneath; see "
                         "--halo.")
    ap.add_argument("--halo", type=float, default=2.0,
                    help="How much wider the black backing stroke is, in points, "
                         "so half of it shows on each side. 3pt rather than 2pt "
                         "because a green box on a green false-colour region "
                         "needs a real dark ring to separate it.")
    ap.add_argument("--viz", default="hp", choices=("hp", "pca"),
                    help="Display transform for the embedding cells. hp "
                         "suppresses the positional ramp first (default); "
                         "pca projects the raw grid, which on multi-view "
                         "cells shows little but the ramp.")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.06)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mod, fold = args.modality, args.fold
    lab = args.root / "yolo_datasets" / f"alfs_2k_{mod}_f{fold}" / "labels" / "val"
    cen = args.root / "yolo_datasets" / f"alfs_2k_{mod}_f{fold}_centralgt" / "labels" / "val"
    hidden = json.loads((args.root / "zenodo_labels" /
                         f"{mod}_hidden_mask_all.json").read_text())
    runs = args.root / "embedding_runs_multiseed"

    heads = {}
    for _, cell, _ in CELLS:
        heads[cell] = load_head(runs, mod, cell, fold, device)
        if heads[cell] is None:
            print(f"  WARNING no fold-{fold} head for {cell}")

    args.out.mkdir(parents=True, exist_ok=True)
    notes = []
    for stem in [s.strip() for s in args.stems.split(",") if s.strip()]:
        gt, hid = ground_truth(stem, lab, cen, hidden, args.iou)
        if not len(gt):
            print(f"  {stem}: no ground truth, skipped")
            continue
        x0n, y0n, side = crop_window(gt, args.margin)
        frame = stem.split("_", 1)[1]
        rec = {"stem": stem, "n_gt": int(len(gt)), "n_hidden": int(hid.sum()),
               "cells": {}}

        for name, cell, tree in CELLS:
            cf = args.root / "embeddings" / f"cell_{mod}_{cell}" / f"{stem}.npy"
            if not cf.exists() or heads[cell] is None:
                continue
            cell_arr = np.load(cf).astype(np.float32)
            det = run_head(heads[cell], cell_arr, args.conf, device)
            m = iou_matrix(gt, det[:, :4]) if len(det) else np.zeros((len(gt), 0))
            hit = m.max(1) >= args.iou if m.size else np.zeros(len(gt), bool)
            # Any detection that lands on an annotated animal is a hit for drawing
            # purposes, even if a better one claimed that box -- the metric is
            # one-to-one, but painting a second correct box red would mislead.
            used = set(np.flatnonzero(m.max(0) >= args.iou).tolist()) if m.size else set()

            if tree is not None:
                src = args.root / "matched_dataset_new" / tree / args.flight / mod / f"{frame}.png"
                disp = (np.asarray(Image.open(src).convert("RGB"), np.float32) / 255.0
                        if src.exists() else to_rgb(cell_arr))
                interp = "lanczos"
            else:
                disp = to_rgb(highpass(cell_arr) if args.viz == "hp" else cell_arr)
                interp = "nearest"
            S = disp.shape[0]
            a, b = int(x0n * S), int(y0n * S)
            w = max(1, int(side * S))
            sub = disp[b:b + w, a:a + w]

            fig, ax = plt.subplots(figsize=(5.0, 5.0))
            ax.imshow(np.clip(sub, 0, 1), interpolation=interp)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            W = sub.shape[1]
            # Ground truth, coloured by whether this representation found it.
            # Each box is a coloured stroke over a wider black one: the halo is
            # what makes the colours legible at print size, and in particular
            # keeps them from being read as part of the false-colour embedding
            # panels underneath.
            for j, bx in enumerate(gt):
                px = [(bx[0]-x0n)/side*W, (bx[1]-y0n)/side*W,
                      (bx[2]-x0n)/side*W, (bx[3]-y0n)/side*W]
                r = Rectangle((px[0], px[1]), px[2]-px[0], px[3]-px[1],
                              fill=False, lw=args.lw,
                              edgecolor="#2ecc40" if hit[j] else "#ff4136",
                              # Fine dashes: matplotlib scales the pattern by the
                              # line width, and these boxes are only ~30 px a
                              # side, so a coarser one puts a single dash on each
                              # edge and reads as corner brackets.
                              linestyle=(0, (1.1, 0.75)) if hid[j] else "-")
                # withStroke re-renders the same path -- dashes included -- so
                # the halo aligns instead of drifting out of phase.
                r.set_path_effects([pe.withStroke(linewidth=args.lw + args.halo,
                                                  foreground="black")])
                ax.add_patch(r)
            if args.show_fp:
                for i, d in enumerate(det):
                    if i in used:
                        continue
                    px = [(d[0]-x0n)/side*W, (d[1]-y0n)/side*W,
                          (d[2]-x0n)/side*W, (d[3]-y0n)/side*W]
                    if px[2] < 0 or px[3] < 0 or px[0] > W or px[1] > W:
                        continue
                    r = Rectangle((px[0], px[1]), px[2]-px[0], px[3]-px[1],
                                  fill=False, lw=args.lw * 0.75,
                                  edgecolor="#ffdc00", linestyle=(0, (0.9, 0.7)))
                    r.set_path_effects([pe.withStroke(linewidth=args.lw * 0.75 + args.halo,
                                                      foreground="black")])
                    ax.add_patch(r)
            dst = args.out / f"{mod}_{stem}_{SLUG[cell]}.png"
            fig.savefig(dst, dpi=150, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            rec["cells"][cell] = {"tp": int(hit.sum()),
                                  "hidden_tp": int((hit & hid).sum()),
                                  "fp": int(len(det) - len(used)),
                                  "file": dst.name}
            print(f"wrote {dst}")
        notes.append(rec)

    (args.out / f"notes_{mod}.json").write_text(json.dumps(notes, indent=2),
                                                encoding="utf-8")
    print(f"wrote {args.out / f'notes_{mod}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
