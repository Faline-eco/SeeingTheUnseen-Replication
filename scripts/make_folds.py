"""Build balanced scene-level CV folds and their label datasets.

The published split (flights 10/211/212/213) draws 212 hidden thermal boxes, of
which flight 10 contributes 149. Its occlusion estimate therefore rests on
effectively one scene, and its seed-to-seed spread cannot express that. These
folds partition every flight present in both modalities, balanced greedily on
hidden-box count so no fold's estimate is dominated by sampling noise.

Label files are symlinked, not copied: the same ~10k text files are
re-partitioned five times and copies would be 400k files for no benefit.
"""
import json
import os
from pathlib import Path

R = Path("/scratch/bambi/datasets/alfs_embed")
K = 5
stats = json.loads((R/"zenodo_labels/flight_stats.json").read_text())
t, r = stats["thermal"], stats["rgb"]
common = sorted(set(t) & set(r), key=lambda f: -t[f]["hidden"])

# Greedy longest-processing-time: assign the largest remaining flight to the
# fold with the fewest hidden boxes so far. Balances a heavily skewed
# distribution far better than a random or round-robin split.
folds = [[] for _ in range(K)]
load = [0]*K
for f in common:
    i = min(range(K), key=lambda j: load[j])
    folds[i].append(f)
    load[i] += t[f]["hidden"]

print(f"{K} folds over {len(common)} flights")
for i, fl in enumerate(folds):
    th = sum(t[f]["hidden"] for f in fl); rh = sum(r[f]["hidden"] for f in fl)
    fr = sum(t[f]["frames"] for f in fl)
    print(f"  fold {i}: {len(fl):2d} flights, {fr:5d} frames, "
          f"thermal hidden {th:5d}, rgb hidden {rh:5d}")

(R/"zenodo_labels/cv_folds.json").write_text(json.dumps(folds))

# ---- build per-fold label datasets -------------------------------------
for mod in ("thermal", "rgb"):
    for suffix in ("", "_centralgt"):
        src = R/f"yolo_datasets/alfs_2k_{mod}{suffix}/labels"
        pool = {}
        for split in ("train", "val"):
            for p in (src/split).glob("*.txt"):
                pool[p.stem] = p
        for k in range(K):
            val_fl = set(folds[k])
            dst = R/f"yolo_datasets/alfs_2k_{mod}_f{k}{suffix}/labels"
            for split in ("train", "val"):
                (dst/split).mkdir(parents=True, exist_ok=True)
            n = {"train": 0, "val": 0}
            for stem, p in pool.items():
                fl = stem.rsplit("_", 1)[0]
                split = "val" if fl in val_fl else "train"
                link = dst/split/f"{stem}.txt"
                if not link.exists():
                    os.symlink(p, link)
                n[split] += 1
            if suffix == "":
                print(f"  {mod} f{k}: train {n['train']}, val {n['val']}")
print("done")
