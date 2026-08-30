"""Full metric set per arm per occlusion population, written to JSON.

`occlusion_recall.py` prints recall and nothing else, so every headline number in
this study has been hand-transcribed from a terminal into markdown. This records
the whole thing instead: TP/FP/FN, precision, recall, F1, AP50 and AP50-95, per
seed and aggregated, for each population.

Scoring a *subset* of the ground truth needs one non-obvious rule. If the
population is "hidden animals", a detection that correctly finds a *visible*
animal is neither a true positive (it is not in the population) nor a false
positive (it is not an error). Counting it as FP would make precision a function
of how many out-of-population animals happen to share the frame, which is not a
property of the method. Such detections are therefore **ignored**, following the
COCO crowd/ignore convention. Detections matching nothing remain false positives.

Recall is unaffected by this choice; precision and AP are not, so the aggregate
row and the per-population rows are not comparable to each other by construction.

    python occlusion_metrics.py --occlusion hidden --modality thermal \
        --labels-dataset alfs_2k_thermal_f1 --hidden-mask <...> \
        --arms <run>:<dataset> --out results_f1.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from occlusion_recall import iou_matrix, read_boxes, xywh_to_xyxy   # noqa: E402
from train_embedding_detector import (                              # noqa: E402
    EmbeddingDet, EmbeddingDetector, collate, decode,
)

IOUV = np.linspace(0.5, 0.95, 10)


def match(dets: np.ndarray, gt: np.ndarray, thr: float) -> np.ndarray:
    """Greedy highest-confidence-first assignment -> index of matched GT, or -1.

    `dets` must already be sorted by descending confidence.
    """
    out = np.full(len(dets), -1, int)
    if not len(dets) or not len(gt):
        return out
    iou = iou_matrix(dets, gt)
    taken = np.zeros(len(gt), bool)
    for i in range(len(dets)):
        j = int(np.argmax(np.where(taken, -1.0, iou[i])))
        if iou[i, j] >= thr and not taken[j]:
            out[i] = j
            taken[j] = True
    return out


def ap_from(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """All-point-interpolated AP; 0 if the population is empty."""
    if n_gt == 0 or not len(tp):
        return 0.0
    o = np.argsort(-conf)
    tp = tp[o]
    ctp = np.cumsum(tp)
    rec = ctp / n_gt
    pre = ctp / np.arange(1, len(tp) + 1)
    mrec = np.concatenate(([0.0], rec, [rec[-1]]))
    mpre = np.concatenate(([1.0], pre, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def score(records: list[dict], conf_thr: float) -> dict:
    """records: per-frame {dets(N,5 sorted), in_pop(M,), out_pop(M,), gt(M,4)}."""
    n_gt = sum(int(r["in_pop"].sum()) for r in records)
    res = {"n_gt": n_gt}

    # --- counting metrics at the fixed operating point ------------------
    tp = fp = 0
    for r in records:
        d = r["dets"]
        d = d[d[:, 4] >= conf_thr]
        m = match(d[:, :4], r["gt"], 0.5)
        for k in range(len(d)):
            j = m[k]
            if j < 0:
                fp += 1                      # matched nothing: a real error
            elif r["in_pop"][j]:
                tp += 1
            # else: matched an out-of-population animal -> ignored
    fn = n_gt - tp
    res.update(tp=tp, fp=fp, fn=fn,
               recall=tp / n_gt if n_gt else 0.0,
               precision=tp / (tp + fp) if (tp + fp) else 0.0)
    res["f1"] = (2 * res["precision"] * res["recall"] /
                 (res["precision"] + res["recall"])
                 if (res["precision"] + res["recall"]) else 0.0)

    # --- AP over the full confidence range ------------------------------
    aps = []
    for thr in IOUV:
        flags, confs = [], []
        for r in records:
            d = r["dets"]
            m = match(d[:, :4], r["gt"], thr)
            for k in range(len(d)):
                j = m[k]
                if j >= 0 and not r["in_pop"][j]:
                    continue                 # ignored
                flags.append(1.0 if j >= 0 else 0.0)
                confs.append(d[k, 4])
        aps.append(ap_from(np.asarray(flags), np.asarray(confs), n_gt))
    res["ap50"] = aps[0]
    res["ap75"] = aps[5]
    res["ap50_95"] = float(np.mean(aps))
    return res


def agg(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": len(a), "values": [float(v) for v in a]}



def populations(stem, lab_val, cen_val, hidden, args):
    """-> (gt_xyxy, occluded, keep) or None when the frame has no merged GT."""
    merged = read_boxes(lab_val / f"{stem}.txt")
    if not len(merged):
        return None
    m_xy = xywh_to_xyxy(merged)
    c_xy = xywh_to_xyxy(read_boxes(cen_val / f"{stem}.txt"))
    occ = (iou_matrix(m_xy, c_xy).max(1) < args.iou
           if len(c_xy) else np.ones(len(m_xy), bool))
    keep = np.ones(len(m_xy), bool)
    if args.occlusion == "hidden" and occ.any():
        fl = hidden.get(stem)
        if fl is None or len(fl) != int(occ.sum()):
            keep[occ] = False
        else:
            for j, f in zip(np.flatnonzero(occ), fl):
                if not f:
                    keep[j] = False
    return m_xy, occ, keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path,
                    default=Path("/scratch/bambi/datasets/alfs_embed/embedding_runs_multiseed"))
    ap.add_argument("--embeddings", type=Path,
                    default=Path("/scratch/bambi/datasets/alfs_embed/embeddings"))
    ap.add_argument("--yolo-datasets", type=Path,
                    default=Path("/scratch/bambi/datasets/alfs_embed/yolo_datasets"))
    ap.add_argument("--labels-dataset", default="alfs_2k_thermal")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--occlusion", default="hidden", choices=("proxy", "hidden"))
    ap.add_argument("--hidden-mask", type=Path, default=None)
    ap.add_argument("--modality", default="thermal", choices=("thermal", "rgb"))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--yolo-runs", type=Path, default=None,
                    help="Score YOLO checkpoints instead of embedding heads. "
                         "Arms become '<run-name>:<image-dataset>'. Exists so "
                         "YOLO goes through the SAME matcher and ignore rule as "
                         "the grid cells -- the older yolo_occlusion_recall.py "
                         "matched each GT to any detection, which inflates "
                         "recall by ~10% and is not comparable with these "
                         "numbers.")
    ap.add_argument("--yolo-images", type=Path,
                    default=Path("/scratch/bambi/datasets/alfs_embed/yolo_images"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_val = args.yolo_datasets / args.labels_dataset / "labels" / "val"
    cen_val = args.yolo_datasets / f"{args.labels_dataset}_centralgt" / "labels" / "val"
    hidden = (json.loads(args.hidden_mask.read_text(encoding="utf-8"))
              if args.hidden_mask and args.hidden_mask.exists() else {})

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip() or None
    except Exception:
        commit = None

    out: dict = {"meta": {
        "modality": args.modality, "occlusion": args.occlusion,
        "labels_dataset": args.labels_dataset, "iou": args.iou, "conf": args.conf,
        "hidden_mask": str(args.hidden_mask) if args.hidden_mask else None,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": commit,
        "note": "Out-of-population detections are ignored, not counted as FP; "
                "population rows are therefore not comparable to 'all'.",
    }, "arms": {}}

    if args.yolo_runs:
        from ultralytics import YOLO
        for spec in [a.strip() for a in args.arms.split(",") if a.strip()]:
            arm, _, ds = spec.partition(":")
            runs = sorted(d for d in args.yolo_runs.glob(f"{arm}_s*")
                          if (d / "weights" / "best.pt").exists())
            img_dir = args.yolo_images / ds / "images" / "val"
            if not runs or not img_dir.is_dir():
                print(f"{arm}: skipped (runs={len(runs)}, imgs={img_dir.is_dir()})")
                continue
            per_seed = {"visible": [], "occluded": [], "all": []}
            seeds = []
            stems = sorted(p.stem for p in lab_val.glob("*.txt"))
            for run in runs:
                seeds.append(run.name.rsplit("_s", 1)[-1])
                model = YOLO(str(run / "weights" / "best.pt"))
                recs = []
                for i in range(0, len(stems), 16):
                    batch = stems[i:i + 16]
                    res = model.predict([str(img_dir / f"{x}.jpg") for x in batch],
                                        imgsz=args.imgsz, conf=0.001,
                                        verbose=False, device=0)
                    for stem, r in zip(batch, res):
                        pop = populations(stem, lab_val, cen_val, hidden, args)
                        if pop is None:
                            continue
                        m_xy, occ, keep = pop
                        if len(r.boxes):
                            d = np.concatenate(
                                [r.boxes.xyxyn.cpu().numpy(),
                                 r.boxes.conf.cpu().numpy()[:, None]], 1)
                            d = d[np.argsort(-d[:, 4])]
                        else:
                            d = np.zeros((0, 5), np.float32)
                        recs.append({"dets": d, "gt": m_xy, "occ": occ, "keep": keep})
                for pop_name, sel in (("visible", lambda r: (~r["occ"]) & r["keep"]),
                                      ("occluded", lambda r: r["occ"] & r["keep"]),
                                      ("all", lambda r: r["keep"])):
                    rr = [{"dets": r["dets"], "gt": r["gt"], "in_pop": sel(r)} for r in recs]
                    per_seed[pop_name].append(score(rr, args.conf))
            entry = {"dataset": ds, "seeds": seeds, "n_runs": len(runs),
                     "populations": {}}
            for pop_name, rows in per_seed.items():
                entry["populations"][pop_name] = {
                    "n_gt": rows[0]["n_gt"] if rows else 0,
                    "per_seed": rows,
                    **{k: agg([r[k] for r in rows]) for k in
                       ("recall", "precision", "f1", "ap50", "ap75", "ap50_95")},
                }
            out["arms"][arm] = entry
            v = entry["populations"]
            print(f"{arm:28s} vis R {v['visible']['recall']['mean']:.4f}  "
                  f"occ R {v['occluded']['recall']['mean']:.4f}  "
                  f"occ P {v['occluded']['precision']['mean']:.4f}  "
                  f"occ AP50 {v['occluded']['ap50']['mean']:.4f}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
        return 0

    for spec in [a.strip() for a in args.arms.split(",") if a.strip()]:
        arm, _, ds = spec.partition(":")
        runs = sorted(p for p in args.runs.glob(f"{arm}_s*") if (p / "best.pt").exists())
        emb = args.embeddings / ds
        if not runs or not emb.is_dir():
            print(f"{arm}: skipped (runs={len(runs)}, emb={emb.is_dir()})")
            continue

        per_seed: dict[str, list[dict]] = {"visible": [], "occluded": [], "all": []}
        seeds = []
        for run in runs:
            ck = torch.load(run / "best.pt", map_location=device)
            summ = json.loads((run / "summary.json").read_text())
            seeds.append(run.name.rsplit("_s", 1)[-1])
            va = EmbeddingDet(emb, lab_val, 0, None, summ.get("use_dims", 0))
            model = EmbeddingDetector(ck["in_dim"], ck["width"], ck["up"]).to(device)
            model.load_state_dict(ck["model"]); model.eval()
            dl = torch.utils.data.DataLoader(va, batch_size=8, shuffle=False,
                                             num_workers=2, collate_fn=collate)
            recs: list[dict] = []
            with torch.no_grad():
                for x, _bl, idxs in dl:
                    hm, wh, off = model(x.to(device))
                    dets = decode(hm, wh, off)
                    for bi, gi in enumerate(idxs):
                        stem = va.items[int(gi)][1].stem
                        merged = read_boxes(lab_val / f"{stem}.txt")
                        if not len(merged):
                            continue
                        m_xy = xywh_to_xyxy(merged)
                        c_xy = xywh_to_xyxy(read_boxes(cen_val / f"{stem}.txt"))
                        occ = (iou_matrix(m_xy, c_xy).max(1) < args.iou
                               if len(c_xy) else np.ones(len(m_xy), bool))
                        keep = np.ones(len(m_xy), bool)
                        if args.occlusion == "hidden":
                            fl = hidden.get(stem)
                            if occ.any():
                                if fl is None or len(fl) != int(occ.sum()):
                                    keep[occ] = False
                                else:
                                    for j, f in zip(np.flatnonzero(occ), fl):
                                        if not f:
                                            keep[j] = False
                        d = dets[bi].cpu().numpy() if len(dets[bi]) else np.zeros((0, 5), np.float32)
                        d = d[np.argsort(-d[:, 4])] if len(d) else d
                        recs.append({"dets": d, "gt": m_xy,
                                     "occ": occ, "keep": keep})
            for pop, sel in (("visible", lambda r: (~r["occ"]) & r["keep"]),
                             ("occluded", lambda r: r["occ"] & r["keep"]),
                             ("all", lambda r: r["keep"])):
                rr = [{"dets": r["dets"], "gt": r["gt"], "in_pop": sel(r)} for r in recs]
                per_seed[pop].append(score(rr, args.conf))
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        entry = {"dataset": ds, "seeds": seeds, "n_runs": len(runs), "populations": {}}
        for pop, rows in per_seed.items():
            entry["populations"][pop] = {
                "n_gt": rows[0]["n_gt"] if rows else 0,
                "per_seed": rows,
                **{k: agg([r[k] for r in rows]) for k in
                   ("recall", "precision", "f1", "ap50", "ap75", "ap50_95")},
            }
        out["arms"][arm] = entry
        v = entry["populations"]
        print(f"{arm:52s} vis R {v['visible']['recall']['mean']:.4f}  "
              f"occ R {v['occluded']['recall']['mean']:.4f}  "
              f"occ P {v['occluded']['precision']['mean']:.4f}  "
              f"occ AP50 {v['occluded']['ap50']['mean']:.4f}  "
              f"occ mAP {v['occluded']['ap50_95']['mean']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
