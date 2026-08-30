"""How many 'hidden' boxes are actually annotated in the central frame?

The hidden population is built geometrically: a merged box is 'invisible
centrally' when it matches no central box at IoU 0.5, and 'hidden' when the
central view nonetheless covered that ground. A merged box inflated by motion
smear can fail that IoU test while its animal IS annotated centrally -- flight
192 frame 640 is such a case. This bounds how often that happens by splitting
the hidden boxes on their best IoU against the central annotation.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np

R = Path("/scratch/bambi/datasets/alfs_embed")
ALFS = R/"matched_dataset_new"/"alfs_2k"


def read_yolo(p):
    if not p.exists():
        return np.zeros((0, 4), np.float32)
    rows = [[float(v) for v in ln.split()[1:5]] for ln in p.read_text().splitlines()
            if len(ln.split()) == 5]
    return np.asarray(rows, np.float32).reshape(-1, 4)


def xyxy(b):
    if not len(b):
        return b.reshape(0, 4)
    return np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2], 1)


def iou_mat(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:,None,0], b[None,:,0]); y1 = np.maximum(a[:,None,1], b[None,:,1])
    x2 = np.minimum(a[:,None,2], b[None,:,2]); y2 = np.minimum(a[:,None,3], b[None,:,3])
    i = np.clip(x2-x1, 0, None)*np.clip(y2-y1, 0, None)
    aa = (a[:,2]-a[:,0])*(a[:,3]-a[:,1]); ab = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return i/np.maximum(aa[:,None]+ab[None,:]-i, 1e-9)


for mod in ("thermal", "rgb"):
    mask = json.loads((R/"zenodo_labels"/f"{mod}_hidden_mask_all.json").read_text())
    c = Counter(); areas = []
    for stem, flags in mask.items():
        if not any(flags):
            continue
        flight, fr = stem.rsplit("_", 1)
        d = ALFS/flight/mod
        m = xyxy(read_yolo(d/f"{int(fr):06d}.txt"))
        cg = xyxy(read_yolo(d/f"{int(fr):06d}_central.txt"))
        if not len(m):
            continue
        best = iou_mat(m, cg).max(1) if len(cg) else np.zeros(len(m))
        invis = best < 0.5
        if int(invis.sum()) != len(flags):
            c["flag_length_mismatch"] += 1
            continue
        for b_iou, f, box in zip(best[invis], flags, m[invis]):
            if not f:
                continue
            c["hidden_total"] += 1
            if b_iou == 0:
                c["no_central_overlap"] += 1
            elif b_iou < 0.3:
                c["overlap_lt_0.3"] += 1
            else:
                c["overlap_0.3_to_0.5"] += 1
                areas.append(float(b_iou))
    t = c["hidden_total"]
    print(f"\n{mod}: {t} hidden boxes over {len(mask)} frames")
    if t:
        for k in ("no_central_overlap", "overlap_lt_0.3", "overlap_0.3_to_0.5"):
            print(f"   {k:<22} {c[k]:>6}  ({100*c[k]/t:5.2f} %)")
    if c["flag_length_mismatch"]:
        print(f"   (skipped {c['flag_length_mismatch']} frames: flag length mismatch)")
    if areas:
        a = np.array(areas)
        print(f"   of the 0.3-0.5 group: median IoU {np.median(a):.3f}, "
              f"{int((a >= 0.45).sum())} above 0.45")
