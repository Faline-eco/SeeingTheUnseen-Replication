"""Build a matched-ground-truth val set for the ALFS vs ortho comparison.

The ALFS datasets are labelled with *merged* boxes — one per track, hulled over
the whole aperture — because that is what the integral actually shows: a moving
animal is smeared along its trajectory, and animals visible only in neighbouring
frames still appear. That is the right training target, but it is the wrong
thing to score against when comparing with an orthographic detector, whose
ground truth only ever contains the central frame's boxes. Scoring two models
against two different label sets makes the mAP delta meaningless.

This creates ``alfs_<modality>_centralgt``: the **same ALFS val images** paired
with the **central-frame** labels, i.e. the identical ground truth the ortho
detector is scored on. Images are hard-linked, so the duplicate costs no space.

Verified upstream: the ALFS ``_central.txt`` labels are bit-identical to the
ortho ``.txt`` labels (same MOT boxes, same virtual camera), so the ortho
dataset's val labels can be reused directly.

    py scripts\\make_central_gt_val.py --datasets D:\\yolo_datasets
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)          # NTFS hard link: no extra bytes on disk
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, required=True)
    ap.add_argument("--modalities", default="rgb,thermal")
    args = ap.parse_args()

    for modality in (m.strip() for m in args.modalities.split(",")):
        alfs = args.datasets / f"alfs_{modality}"
        ortho = args.datasets / f"ortho_{modality}"
        out = args.datasets / f"alfs_{modality}_centralgt"
        if not (alfs / "images" / "val").is_dir() or not (ortho / "labels" / "val").is_dir():
            print(f"[skip] {modality}: need both {alfs.name} and {ortho.name}")
            continue

        (out / "images" / "val").mkdir(parents=True, exist_ok=True)
        (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

        n_img = n_lbl = missing = 0
        for img in sorted((alfs / "images" / "val").glob("*.jpg")):
            label = ortho / "labels" / "val" / f"{img.stem}.txt"
            if not label.exists():
                missing += 1
                continue
            link_or_copy(img, out / "images" / "val" / img.name)
            link_or_copy(label, out / "labels" / "val" / label.name)
            n_img += 1
            n_lbl += sum(1 for ln in label.read_text(encoding="utf-8").splitlines()
                         if ln.strip())

        # train/ points at val/ only so `yolo val` resolves; nothing trains here.
        (out / "data.yaml").write_text(
            f"# ALFS val images with central-frame ground truth, for a\n"
            f"# like-for-like comparison against the orthographic detector.\n"
            f"path: {out.as_posix()}\n"
            f"train: images/val\n"
            f"val: images/val\n"
            f"nc: 1\nnames:\n  0: animal\n", encoding="utf-8")
        print(f"{out.name}: {n_img} images, {n_lbl} boxes"
              + (f", {missing} without a central label" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
