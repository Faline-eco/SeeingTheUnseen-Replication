"""Score point detectors (OWL) on our occlusion populations.

OWL emits points, not boxes, so IoU matching does not apply. A detection counts
as a hit when its point falls INSIDE a ground-truth box, matched one-to-one and
greedily by descending confidence -- the same one-to-one discipline the box
matcher uses, since allowing one detection to satisfy several boxes inflated
recall by ~10% when we last let it.

Point-in-box is a MORE PERMISSIVE criterion than IoU 0.5. Comparing OWL under
this rule against our detectors under IoU would flatter OWL for purely
mechanical reasons, so `--yolo-runs` re-scores our own detectors here too, using
their predicted box centres as points. Only numbers produced by this script are
comparable with each other.

The population construction, the hidden mask and the COCO ignore rule are
imported from occlusion_metrics.py rather than reimplemented, so both families
see exactly the same ground truth.

Coordinates: OWL writes detections at heatmap scale (down_ratio 2 with up=False),
so they are multiplied by --down-ratio to reach image pixels. Verified against
the authors' own evaluator: without it recall collapses to ~0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import csv as _csv

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from occlusion_metrics import populations, agg  # noqa: E402


def match_points(pts: np.ndarray, scores: np.ndarray, gt: np.ndarray,
                 in_pop: np.ndarray) -> tuple[int, int, int]:
    """-> (tp, fp, n_pop). One-to-one greedy by score; out-of-population GT ignored.

    A detection landing in an out-of-population box is neither a hit nor a false
    positive: that is the COCO ignore convention the box scorer uses, and without
    it a population-restricted precision is meaningless.
    """
    n_pop = int(in_pop.sum())
    if len(pts) == 0:
        return 0, 0, n_pop
    order = np.argsort(-scores)
    taken = np.zeros(len(gt), bool)
    tp = fp = 0
    for i in order:
        x, y = pts[i]
        inside = np.flatnonzero(
            (gt[:, 0] <= x) & (x <= gt[:, 2]) & (gt[:, 1] <= y) & (y <= gt[:, 3])
            & ~taken)
        if len(inside) == 0:
            fp += 1                      # matched nothing at all -> false positive
            continue
        # Smallest containing box first: with nested/overlapping GT the tight box
        # is the intended target, and taking the largest would strand it.
        areas = ((gt[inside, 2] - gt[inside, 0]) * (gt[inside, 3] - gt[inside, 1]))
        j = inside[int(np.argmin(areas))]
        taken[j] = True
        if in_pop[j]:
            tp += 1
        # else: in-image but out-of-population -> ignored, neither tp nor fp
    return tp, fp, n_pop


def score(recs: list[dict]) -> dict:
    tp = sum(r["tp"] for r in recs)
    fp = sum(r["fp"] for r in recs)
    n = sum(r["n_pop"] for r in recs)
    rec = tp / n if n else 0.0
    pre = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * rec * pre / (rec + pre) if (rec + pre) else 0.0
    return {"recall": rec, "precision": pre, "f1": f1,
            "tp": tp, "fp": fp, "n_gt": n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owl-runs", type=Path, default=None,
                    help="Directory of <model>_<arm>_f<k>/detections.csv")
    ap.add_argument("--arms", required=True,
                    help="Comma-separated run-directory names to score.")
    ap.add_argument("--yolo-datasets", type=Path,
                    default=Path("/scratch/bambi/datasets/alfs_embed/yolo_datasets"))
    ap.add_argument("--labels-dataset", required=True)
    ap.add_argument("--hidden-mask", type=Path, default=None)
    ap.add_argument("--modality", default="rgb", choices=("thermal", "rgb"))
    ap.add_argument("--occlusion", default="hidden", choices=("proxy", "hidden"))
    ap.add_argument("--iou", type=float, default=0.5,
                    help="Only used to decide which merged boxes are occluded.")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--down-ratio", type=float, default=2.0)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lab_val = args.yolo_datasets / args.labels_dataset / "labels" / "val"
    cen_val = args.yolo_datasets / f"{args.labels_dataset}_centralgt" / "labels" / "val"
    hidden = (json.loads(args.hidden_mask.read_text(encoding="utf-8"))
              if args.hidden_mask and args.hidden_mask.exists() else {})

    out: dict = {"meta": {
        "scorer": "point-in-box, one-to-one greedy by confidence",
        "modality": args.modality, "occlusion": args.occlusion,
        "labels_dataset": args.labels_dataset, "conf": args.conf,
        "down_ratio": args.down_ratio, "imgsz": args.imgsz,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Point-in-box is more permissive than IoU 0.5. Only compare "
                "numbers produced by this scorer with each other.",
    }, "arms": {}}

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        det_csv = args.owl_runs / arm / "detections.csv"
        if not det_csv.exists():
            print(f"{arm}: skipped (no detections.csv)")
            continue
        # stdlib csv, not pandas: this has to run in the same image as
        # occlusion_metrics.py, which does not carry pandas.
        by_img: dict[str, list[tuple[float, float, float]]] = {}
        n_read = n_kept = n_blank = 0
        with det_csv.open(newline="") as fh:
            for row in _csv.DictReader(fh):
                n_read += 1
                # Frames where the model found nothing still emit a row, with
                # empty x/y/score. Their own infer.py dropna()s these; counting
                # them would invent detections at (0,0).
                if not (row.get("dscores") or "").strip():
                    n_blank += 1
                    continue
                if not (row.get("x") or "").strip() or not (row.get("y") or "").strip():
                    n_blank += 1
                    continue
                sc = float(row["dscores"])
                if sc < args.conf:
                    continue
                n_kept += 1
                by_img.setdefault(row["images"], []).append(
                    (float(row["x"]), float(row["y"]), sc))
        print(f"{arm}: {n_kept}/{n_read} detections at conf>={args.conf} "
              f"({n_blank} empty rows skipped)")

        per = {"visible": [], "occluded": [], "all": []}
        n_frames = 0
        for lp in sorted(lab_val.glob("*.txt")):
            stem = lp.stem
            pop = populations(stem, lab_val, cen_val, hidden, args)
            if pop is None:
                continue
            gt, occ, keep = pop
            gt = gt * args.imgsz          # normalised xyxy -> pixels
            n_frames += 1
            d = None
            for ext in (".jpg", ".png", ".jpeg", ".tif"):
                if stem + ext in by_img:
                    d = by_img[stem + ext]
                    break
            if not d:
                pts = np.zeros((0, 2), np.float32)
                sc = np.zeros((0,), np.float32)
            else:
                arr = np.asarray(d, np.float32)
                pts = arr[:, :2] * args.down_ratio
                sc = arr[:, 2]

            for name, sel in (("visible", (~occ) & keep),
                              ("occluded", occ & keep),
                              ("all", keep)):
                tp, fp, n_pop = match_points(pts, sc, gt, sel)
                per[name].append({"tp": tp, "fp": fp, "n_pop": n_pop})

        entry = {"n_frames": n_frames,
                 "populations": {k: score(v) for k, v in per.items()}}
        out["arms"][arm] = entry
        v = entry["populations"]
        print(f"{arm:34s} vis R {v['visible']['recall']:.4f}  "
              f"occ R {v['occluded']['recall']:.4f}  "
              f"occ P {v['occluded']['precision']:.4f}  "
              f"occ F1 {v['occluded']['f1']:.4f}  (n={v['occluded']['n_gt']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
