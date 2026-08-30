"""Per-fold YOLO image datasets, so the capacity controls sit on the same
scene-level CV protocol as the embedding grid.

The staged datasets use the published split (flights 10/211/212/213), which is
70 % one flight and inflates effect sizes -- the grid's 5.53x became 2.49x under
CV. Leaving YOLO on that split would make the overview mix protocols.

Images and labels are symlinked, not copied: the same ~39k jpgs are
re-partitioned five times, and copies would be ~30 GB for nothing.

Flights absent from cv_folds.json (present in one modality only) are never a
validation fold, so they always go to train -- the same rule make_folds.py used
for the label datasets.
"""
import json
import os
from pathlib import Path

R = Path("/scratch/bambi/datasets/alfs_embed")
IMG = R / "yolo_images"
ARMS = ["ortho_thermal", "alfs_thermal", "ortho_rgb", "alfs_rgb"]
folds = json.loads((R / "zenodo_labels/cv_folds.json").read_text())

for arm in ARMS:
    src = IMG / arm
    pool = {}
    for split in ("train", "val"):
        for p in (src / "images" / split).glob("*.jpg"):
            lab = src / "labels" / split / f"{p.stem}.txt"
            pool[p.stem] = (p, lab if lab.exists() else None)
    for k, val_flights in enumerate(folds):
        vf = set(val_flights)
        dst = IMG / f"{arm}_f{k}"
        for sub in ("images", "labels"):
            for split in ("train", "val"):
                (dst / sub / split).mkdir(parents=True, exist_ok=True)
        n = {"train": 0, "val": 0}
        for stem, (img, lab) in pool.items():
            split = "val" if stem.rsplit("_", 1)[0] in vf else "train"
            li = dst / "images" / split / img.name
            if not li.exists():
                os.symlink(img, li)
            if lab is not None:
                ll = dst / "labels" / split / lab.name
                if not ll.exists():
                    os.symlink(lab, ll)
            n[split] += 1
        (dst / "data.yaml").write_text(
            f"path: {dst}\ntrain: images/train\nval: images/val\n"
            f"nc: 1\nnames:\n  0: animal\n", encoding="utf-8")
        print(f"  {arm}_f{k}: train {n['train']}, val {n['val']}")
print("done")
