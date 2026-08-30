"""Does the embedded light field recover animals hidden in the central frame?

This is the direct test of the occlusion hypothesis, which the headline recall
number does not isolate. The ALFS and embedded-LF arms both beat the single-view
baseline on recall, but "more animals found" is not the same claim as "animals
that the central view could not see".

The split comes free with the data. Two ground-truth sets exist per frame:

  * central GT  (``<frame>_central.txt``) — boxes for animals visible in the
    central frame itself;
  * merged GT   (``<frame>.txt``)          — one box per track over the whole
    aperture, so it also covers animals only some other view could see.

A merged box that matches no central box is therefore an animal the central
view did not show. Recall on that subset is the quantity of interest; recall on
the matched subset is the control. If the light field is doing what it is meant
to, the gap between arms should be much larger on the occluded subset than on
the visible one.

    python occlusion_recall.py --runs <dir> --arms alfs_2k_thermal_2k_64,embfield_thermal_embLF_64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_embedding_detector import (                     # noqa: E402
    EmbeddingDet, EmbeddingDetector, collate, decode,
)


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Na, Nb) IoU between two sets of xyxy boxes."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ar_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(ar_a[:, None] + ar_b[None, :] - inter, 1e-9)).astype(np.float32)


def read_boxes(p: Path) -> np.ndarray:
    if not p.exists():
        return np.zeros((0, 4), np.float32)
    rows = [[float(v) for v in ln.split()[1:5]]
            for ln in p.read_text(encoding="utf-8").splitlines()
            if len(ln.split()) == 5]
    return np.asarray(rows, np.float32).reshape(-1, 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path,
                    default=Path("/scratch/bambi/datasets/alfs_embed/embedding_runs_multiseed"))
    ap.add_argument("--embeddings", type=Path,
                    default=Path("/scratch/bambi/datasets/alfs_embed/embeddings"))
    ap.add_argument("--yolo-datasets", type=Path,
                    default=Path("/scratch/bambi/datasets/alfs_embed/yolo_datasets"))
    ap.add_argument("--labels-dataset", default="alfs_2k_thermal",
                    help="Merged-GT dataset; its _centralgt sibling supplies "
                         "the central boxes.")
    ap.add_argument("--arms", required=True,
                    help="Comma-separated '<run-prefix>:<embedding-dataset>' "
                         "pairs; seeds are pooled. Given explicitly because the "
                         "run directory name concatenates dataset and tag with "
                         "the same separator, so it cannot be split reliably.")
    ap.add_argument("--occlusion", default="proxy",
                    choices=("proxy", "reviewed", "hidden"),
                    help="'proxy' calls a box occluded when it is in the merged "
                         "aperture GT but has no central-frame counterpart, i.e. "
                         "the animal is *invisible* centrally. 'reviewed' uses "
                         "the human occlusion ranges from the Zenodo release, "
                         "where the animal IS annotated centrally but flagged "
                         "partially hidden. These are different populations, "
                         "not two estimates of one — report both.")
    ap.add_argument("--hidden-mask", type=Path, default=None,
                    help="For --occlusion hidden: JSON from "
                         "decompose_invisible.py marking which "
                         "invisible boxes were actually imaged.")
    ap.add_argument("--box-occlusion", type=Path,
                    default=Path("D:/zenodo_labels"),
                    help="Directory holding <modality>_val_box_occlusion.json.")
    ap.add_argument("--modality", default="thermal", choices=("thermal", "rgb"))
    ap.add_argument("--by-frame", action="store_true",
                    help="Report recall over ALL boxes of a frame, split by "
                         "whether the frame contains a hidden animal. The "
                         "per-box columns ask 'can this method find a hidden "
                         "animal'; this asks 'is a scene with occluders harder "
                         "overall', which is a different question and the one a "
                         "detector actually faces at inference.")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.3,
                    help="Detection confidence floor. Recall is reported at a "
                         "fixed threshold so the arms are compared at the same "
                         "operating point, not at each one's own best.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_val = args.yolo_datasets / args.labels_dataset / "labels" / "val"
    cen_val = args.yolo_datasets / f"{args.labels_dataset}_centralgt" / "labels" / "val"

    # In reviewed mode the boxes scored are the *central* ones (that is what the
    # reviewer annotated), each carrying a state in frame order.
    box_states: dict[str, list[str]] = {}
    hidden: dict[str, list[int]] = {}
    if args.occlusion == 'hidden':
        if not args.hidden_mask or not args.hidden_mask.exists():
            raise SystemExit('--occlusion hidden needs --hidden-mask')
        hidden = json.loads(args.hidden_mask.read_text(encoding='utf-8'))
    if args.occlusion == "reviewed":
        f = args.box_occlusion / f"{args.modality}_val_box_occlusion.json"
        if not f.exists():
            raise SystemExit(f"missing {f}; run build_track_sidecar.py first")
        box_states = json.loads(f.read_text(encoding="utf-8"))
        lab_val = cen_val
        n_occ = sum(s.count("occluded") for s in box_states.values())
        n_cle = sum(s.count("clear") for s in box_states.values())
        n_unl = sum(s.count("unlabelled") for s in box_states.values())
        print(f"reviewed occlusion: {n_cle} clear, {n_occ} occluded, "
              f"{n_unl} unlabelled (excluded)")

    print(f"merged GT : {lab_val}")
    print(f"central GT: {cen_val}")
    print(f"IoU {args.iou}, confidence >= {args.conf}\n")
    if args.by_frame:
        print(f"{'arm':38s} {'seeds':>5} {'R clean-fr':>12} {'R occl-fr':>12} {'n_box':>7}")
    else:
        print(f"{'arm':38s} {'seeds':>5} {'R visible':>12} {'R occluded':>12} {'n_occ':>7}")
    print("-" * 80)

    for spec in [a.strip() for a in args.arms.split(",") if a.strip()]:
        arm, _, ds = spec.partition(":")
        if not ds:
            print(f"{spec:38s}  expected '<run-prefix>:<embedding-dataset>'")
            continue
        runs = sorted(p for p in args.runs.glob(f"{arm}_s*") if (p / "best.pt").exists())
        if not runs:
            print(f"{arm:38s}  no runs found under {args.runs}")
            continue
        emb_dir = args.embeddings / ds
        if not emb_dir.is_dir():
            print(f"{arm:38s}  embeddings not found: {emb_dir}")
            continue
        rec_vis, rec_occ, n_occ_tot = [], [], 0
        for run in runs:
            ck = torch.load(run / "best.pt", map_location=device)
            summ = json.loads((run / "summary.json").read_text())
            # summary records the *effective* width, so passing it is correct
            # whether or not the run truncated.
            va = EmbeddingDet(emb_dir, lab_val, 0, None, summ.get("use_dims", 0))
            model = EmbeddingDetector(ck["in_dim"], ck["width"], ck["up"]).to(device)
            model.load_state_dict(ck["model"])
            model.eval()

            dl = torch.utils.data.DataLoader(va, batch_size=8, shuffle=False,
                                             num_workers=2, collate_fn=collate)
            hit_v = tot_v = hit_o = tot_o = 0
            with torch.no_grad():
                for x, boxes_list, idxs in dl:
                    hm, wh, off = model(x.to(device))
                    dets = decode(hm, wh, off)
                    for bi, gi in enumerate(idxs):
                        stem = va.items[int(gi)][1].stem
                        merged = read_boxes(lab_val / f"{stem}.txt")
                        if not len(merged):
                            continue
                        m_xy = xywh_to_xyxy(merged)
                        keep = np.ones(len(m_xy), bool)
                        if args.occlusion == "reviewed":
                            st = box_states.get(stem, [])
                            if len(st) != len(m_xy):
                                continue          # alignment lost; skip frame
                            arr = np.array(st)
                            occluded = arr == "occluded"
                            # 'unlabelled' is neither group: counting it as
                            # clear would dilute the control with unknowns.
                            keep = arr != "unlabelled"
                        else:
                            central = read_boxes(cen_val / f"{stem}.txt")
                            c_xy = xywh_to_xyxy(central)
                            # An animal the central view did not show: a merged
                            # box with no central counterpart.
                            occluded = (iou_matrix(m_xy, c_xy).max(1) < args.iou
                                        if len(c_xy) else np.ones(len(m_xy), bool))
                            if args.occlusion == "hidden":
                                # Keep only invisible boxes the central view
                                # actually imaged; the rest are wider-footprint
                                # cases and say nothing about occlusion.
                                #
                                # A frame with no invisible boxes has no mask
                                # entry, and must still contribute its VISIBLE
                                # boxes — skipping it would restrict the control
                                # group to frames that happen to contain a hidden
                                # animal, i.e. the cluttered ones.
                                flags = hidden.get(stem)
                                keep = np.ones(len(m_xy), bool)
                                if not occluded.any():
                                    pass                     # visible-only frame
                                elif flags is None or len(flags) != int(occluded.sum()):
                                    keep[occluded] = False   # unclassifiable: drop
                                else:
                                    idx = np.flatnonzero(occluded)
                                    for j, f in zip(idx, flags):
                                        if not f:
                                            keep[j] = False
                        d = dets[bi]
                        d = d[d[:, 4] >= args.conf] if len(d) else d
                        d_xy = d[:, :4].cpu().numpy() if len(d) else np.zeros((0, 4), np.float32)
                        matched = (iou_matrix(m_xy, d_xy).max(1) >= args.iou
                                   if len(d_xy) else np.zeros(len(m_xy), bool))
                        if args.by_frame:
                            # Whole frame lands in one group; every kept box in
                            # it counts, hidden or not.
                            if (occluded & keep).any():
                                occ_k, vis_k = keep, np.zeros(len(m_xy), bool)
                            else:
                                occ_k, vis_k = np.zeros(len(m_xy), bool), keep
                        else:
                            occ_k = occluded & keep
                            vis_k = (~occluded) & keep
                        hit_o += int((matched & occ_k).sum())
                        tot_o += int(occ_k.sum())
                        hit_v += int((matched & vis_k).sum())
                        tot_v += int(vis_k.sum())
            rec_vis.append(hit_v / max(tot_v, 1))
            rec_occ.append(hit_o / max(tot_o, 1))
            n_occ_tot = tot_o
        if rec_vis:
            def f(v):
                return (f"{np.mean(v):.4f}+/-{np.std(v, ddof=1):.4f}"
                        if len(v) > 1 else f"{v[0]:.4f}")
            print(f"{arm:38s} {len(rec_vis):5d} {f(rec_vis):>12} "
                  f"{f(rec_occ):>12} {n_occ_tot:7d}")

    print("\nRecall on 'occluded' = animals present in the aperture-merged GT but\n"
          "not visible in the central frame. A light field that works should\n"
          "widen the gap between arms there far more than on visible animals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
