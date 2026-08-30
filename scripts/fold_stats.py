"""Per-flight statistics, so validation folds can be balanced rather than arbitrary.

A fold with few hidden animals gives a recall estimate dominated by sampling
noise, which would be indistinguishable from the scene variance we are trying to
measure. Balancing on hidden-box count is what makes the between-fold spread
interpretable.
"""
import json
import numpy as np
from pathlib import Path

R = Path("/scratch/bambi/datasets/alfs_embed")
ALFS = R / "matched_dataset_new/alfs_2k"

def rd(p):
    if not p.exists():
        return np.zeros((0, 4), np.float32)
    r = [[float(v) for v in l.split()[1:5]] for l in p.read_text().splitlines()
         if len(l.split()) == 5]
    return np.asarray(r, np.float32).reshape(-1, 4)

def xy(b):
    o = np.empty_like(b)
    o[:, 0] = b[:, 0]-b[:, 2]/2; o[:, 1] = b[:, 1]-b[:, 3]/2
    o[:, 2] = b[:, 0]+b[:, 2]/2; o[:, 3] = b[:, 1]+b[:, 3]/2
    return o

def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    i = np.clip(x2-x1, 0, None)*np.clip(y2-y1, 0, None)
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return (i/np.maximum(aa[:, None]+ab[None, :]-i, 1e-9)).astype(np.float32)

out = {}
for mod in ("thermal", "rgb"):
    pool = R / f"yolo_datasets/alfs_2k_{mod}/labels"
    stems = [p.stem for s in ("train", "val") for p in (pool/s).glob("*.txt")]
    per = {}
    for stem in stems:
        fl, fr = stem.rsplit("_", 1)
        d = ALFS/fl/mod
        m = rd(d/f"{fr}.txt"); c = rd(d/f"{fr}_central.txt")
        if not len(m):
            continue
        inv = iou(xy(m), xy(c)).max(1) < 0.5 if len(c) else np.ones(len(m), bool)
        cov = R/f"field_{mod}_rgb_single"/fl/mod/f"{fr}_cov.npy"
        hid = 0
        if inv.any() and cov.exists():
            seen = np.load(cov) > 0
            n = seen.shape[0]
            for b in xy(m)[inv]:
                x0 = max(int(b[0]*n)-1, 0); x1 = min(int(np.ceil(b[2]*n))+1, n)
                y0 = max(int(b[1]*n)-1, 0); y1 = min(int(np.ceil(b[3]*n))+1, n)
                patch = seen[y0:max(y1, y0+1), x0:max(x1, x0+1)]
                if patch.size and patch.mean() >= 0.5:
                    hid += 1
        e = per.setdefault(fl, {"frames": 0, "boxes": 0, "invisible": 0, "hidden": 0})
        e["frames"] += 1; e["boxes"] += len(m)
        e["invisible"] += int(inv.sum()); e["hidden"] += hid
    out[mod] = per
    tot = {k: sum(v[k] for v in per.values()) for k in ("frames", "boxes", "invisible", "hidden")}
    print(f"{mod}: {len(per)} flights, {tot}")

Path("/scratch/bambi/datasets/alfs_embed/zenodo_labels/flight_stats.json").write_text(json.dumps(out))
print("wrote flight_stats.json")
