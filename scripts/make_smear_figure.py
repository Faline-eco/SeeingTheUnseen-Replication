"""Crops showing what synthetic-aperture integration does to a MOVING animal.

The aperture focuses on the DEM plane, so static ground detail sharpens while
anything that moved during the ~9 m sweep integrates into a streak. The
annotation records the same thing independently: a track's merged box is the
hull of its per-frame boxes, so a moving animal's merged box is far larger than
its central-frame box. Both are drawn, so the smear can be read off the image and
checked against the geometry.

    solid   merged box  (hull over every contributing frame)
    dotted  central box (the animal in the central frame alone)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

TREES = {"alfs": "alfs_2k", "ortho": "geo-referenced2_2k"}


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


def draw(img, x0n, y0n, side, gt, cgt, dst, lw=1.8):
    S = img.shape[0]
    a, b = int(x0n * S), int(y0n * S)
    w = max(1, int(side * S))
    sub = img[b:b + w, a:a + w]
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(np.clip(sub, 0, 1), interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    W = sub.shape[1]
    for boxes, col, ls in ((cgt, "#39cccc", (0, (2, 2))), (gt, "#ffdc00", "-")):
        for bx in boxes:
            px = [(bx[0]-x0n)/side*W, (bx[1]-y0n)/side*W,
                  (bx[2]-x0n)/side*W, (bx[3]-y0n)/side*W]
            if px[2] < 0 or px[3] < 0 or px[0] > W or px[1] > W:
                continue
            ax.add_patch(Rectangle((px[0], px[1]), px[2]-px[0], px[3]-px[1],
                                   fill=False, lw=lw, edgecolor=col, linestyle=ls))
    fig.savefig(dst, dpi=170, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flight", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--modality", default="thermal")
    ap.add_argument("--root", type=Path,
                    default=Path("/scratch/bambi/datasets/alfs_embed"))
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    mod, FL, FR = args.modality, args.flight, args.frame
    stem = f"{FL}_{FR}"
    lab = args.root/"yolo_datasets"/f"alfs_2k_{mod}_f{args.fold}"/"labels"/"val"
    cen = args.root/"yolo_datasets"/f"alfs_2k_{mod}_f{args.fold}_centralgt"/"labels"/"val"
    gt, cgt = read_yolo(lab/f"{stem}.txt"), read_yolo(cen/f"{stem}.txt")
    if not len(gt):
        print(f"no ground truth for {stem} in fold {args.fold}")
        return 1

    print(f"{stem}: {len(gt)} merged boxes, {len(cgt)} central boxes")
    ratios = []
    for i, b in enumerate(gt):
        aw, ah = b[2]-b[0], b[3]-b[1]
        j = int(np.argmin(np.hypot((cgt[:, 0]+cgt[:, 2])/2 - (b[0]+b[2])/2,
                                   (cgt[:, 1]+cgt[:, 3])/2 - (b[1]+b[3])/2))) if len(cgt) else -1
        if j >= 0:
            cw, ch = cgt[j][2]-cgt[j][0], cgt[j][3]-cgt[j][1]
            r = (aw*ah)/(cw*ch)
            ratios.append((r, i))
            print(f"  animal {i}: merged {aw*2048:.0f}x{ah*2048:.0f} px, "
                  f"central {cw*2048:.0f}x{ch*2048:.0f} px, area x{r:.2f}")
    args.out.mkdir(parents=True, exist_ok=True)

    mg = args.margin
    x0 = max(0.0, gt[:, 0].min()-mg); x1 = min(1.0, gt[:, 2].max()+mg)
    y0 = max(0.0, gt[:, 1].min()-mg); y1 = min(1.0, gt[:, 3].max()+mg)
    side = max(x1-x0, y1-y0)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    win = (float(np.clip(cx-side/2, 0, 1-side)), float(np.clip(cy-side/2, 0, 1-side)), side)

    zoom = None
    if ratios:
        _, i = max(ratios)
        b = gt[i]
        zs = max(b[2]-b[0], b[3]-b[1]) * 3.0
        zx = float(np.clip((b[0]+b[2])/2 - zs/2, 0, 1-zs))
        zy = float(np.clip((b[1]+b[3])/2 - zs/2, 0, 1-zs))
        zoom = (zx, zy, zs)
        print(f"  zoom on animal {i}")

    for key, tree in TREES.items():
        src = args.root/"matched_dataset_new"/tree/FL/mod/f"{FR}.png"
        if not src.exists():
            print(f"  missing {src}")
            continue
        img = np.asarray(Image.open(src).convert("RGB"), np.float32)/255.0
        draw(img, *win, gt, cgt, args.out/f"{mod}_{stem}_{key}.png")
        if zoom:
            draw(img, *zoom, gt, cgt, args.out/f"{mod}_{stem}_{key}_zoom.png", lw=2.4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
