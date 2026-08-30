"""Build an OWL-format ground-truth CSV from our YOLO label files.

OWL is a point detector: its CSV is one row per animal with columns
`images,x,y,labels` in PIXEL coordinates of the full image. Our labels are
normalised YOLO boxes, so each box contributes its centre.

Images with no animals still belong in the evaluation set -- they are where
false positives come from -- but the OWL CSVDataset keys on annotation rows, so
empty images are emitted with a sentinel row the way tools/infer.py does
(x=0, y=0, labels=1 is what infer.py uses for unannotated inference input).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="Take only the first N images (smoke tests).")
    ap.add_argument("--no-empty", action="store_true",
                    help="Omit images with no animals entirely. Required for "
                         "TRAINING: the sentinel row below is a point at (0,0), "
                         "which as a training target is a phantom animal in the "
                         "corner of every empty frame. Inference wants the "
                         "sentinels (every image must be scored); training must "
                         "not have them.")
    ap.add_argument("--with-animals-first", action="store_true",
                    help="Order images with animals first, so a small --limit "
                         "still contains positives.")
    args = ap.parse_args()

    exts = (".jpg", ".jpeg", ".png", ".tif")
    imgs = sorted(p for p in args.images.iterdir() if p.suffix.lower() in exts)

    def n_boxes(p: Path) -> int:
        lp = args.labels / f"{p.stem}.txt"
        if not lp.exists():
            return 0
        return sum(1 for ln in lp.read_text().split("\n") if ln.strip())

    if args.with_animals_first:
        imgs.sort(key=lambda p: (-n_boxes(p), p.name))
    if args.limit:
        imgs = imgs[: args.limit]

    rows, n_pts, n_empty = [], 0, 0
    for p in imgs:
        with Image.open(p) as im:
            W, H = im.size
        lp = args.labels / f"{p.stem}.txt"
        boxes = []
        if lp.exists():
            for ln in lp.read_text().split("\n"):
                f = ln.split()
                if len(f) >= 5:
                    boxes.append(tuple(float(v) for v in f[1:5]))
        if not boxes:
            n_empty += 1
            if args.no_empty:
                continue
            rows.append((p.name, 0, 0, 1))       # sentinel, as tools/infer.py does
            continue
        for cx, cy, _w, _h in boxes:
            rows.append((p.name, round(cx * W), round(cy * H), 1))
            n_pts += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["images", "x", "y", "labels"])
        w.writerows(rows)

    print(f"[gt] {len(imgs)} images -> {args.out}")
    print(f"     {n_pts} animal points, {n_empty} empty images"
          f"{' (omitted)' if args.no_empty else ' (sentinel rows)'}")
    if imgs:
        with Image.open(imgs[0]) as im:
            print(f"     image size: {im.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
