"""Assemble the six YOLO datasets for the raw / ortho / ALFS comparison.

Builds ``{raw,ortho,alfs} x {rgb,thermal}`` over an **identical set of central
frames**, so any difference in detector performance is attributable to the
imaging modality and not to which frames each model happened to see.

Frame selection
---------------
A (flight, frame) pair is included only when all three sources can supply it:

* raw   — ``flights_root/<id>/frames/<mod>/NNNNNN.jpg`` plus MOT labels,
* ortho — ``output_root/<id>/<mod>/NNNNNN.png`` plus its projected ``.txt``,
* alfs  — ``alfs_output_root/<id>/<mod>/NNNNNN.png`` plus its ``.txt``.

Frames whose projected label file exists but is empty are dropped from *all*
sources: a box that missed the DEM in one modality would otherwise appear as a
false negative there and a true positive elsewhere.

Split
-----
Two modes, both at flight level so no frame leaks between splits:

* ``--split-mode official`` (default) follows the BAMBI dataset's own
  ``flight_metadata/splits.json``. This is what makes a comparison against
  externally released detectors meaningful: models trained on the official
  train split have never seen the official val/test flights. Note that the
  per-flight ``*_metadata.json`` in this dataset copy claims ``"split": "train"``
  for *every* flight, which contradicts ``splits.json`` — the official file wins.
* ``--split-mode seeded`` falls back to a seeded 80/20 assignment, used when no
  official split applies. Flights are assigned largest-first to whichever split
  is furthest below its target so the frame ratio stays close.

Images are written as 3-channel JPEG (RGBA composited over black), which is what
the detector consumes anyway and keeps the six datasets to ~15 GB in total.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from georef.config import load_config                      # noqa: E402
from georef.discovery import list_available_flight_ids, resolve_flight  # noqa: E402
from georef.labels import load_mot                         # noqa: E402
from georef.video_frames import SAMPLING_STEP              # noqa: E402

SOURCES = ("raw", "ortho", "alfs")
MODALITIES = ("rgb", "thermal")
CLASS_NAMES = {0: "animal"}
JPEG_QUALITY = 95


@dataclass
class FrameRef:
    flight_id: str
    frame_idx: int
    # source -> (image path, yolo label rows as "cls cx cy w h")
    per_source: dict[str, tuple[Path, list[str]]] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return f"{self.flight_id}_{self.frame_idx:06d}"


def _clip_row(row: str) -> str | None:
    """Clip a YOLO row to the unit square, or drop it if nothing remains.

    Projected boxes — especially merged ALFS hulls spanning a whole aperture —
    can extend past the render edge. YOLO refuses such labels at load time
    ("non-normalized or out of bounds coordinates") and silently drops them,
    which turns a real animal into background. Clipping keeps the visible part
    of the box instead.
    """
    parts = row.split()
    if len(parts) != 5:
        return None
    cls = parts[0]
    cx, cy, w, h = (float(v) for v in parts[1:])
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    if x2 - x1 <= 1e-9 or y2 - y1 <= 1e-9:
        return None
    return f"{cls} {(x1 + x2) / 2} {(y1 + y2) / 2} {x2 - x1} {y2 - y1}"


def _read_yolo(path: Path) -> list[str] | None:
    """Rows of a projected YOLO label file, clipped; None when absent."""
    if not path.exists():
        return None
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        clipped = _clip_row(ln.strip())
        if clipped is not None:
            rows.append(clipped)
    return rows


def _raw_labels(labels: list, width: int, height: int) -> list[str]:
    """MOT boxes in original frame pixels -> normalised YOLO rows."""
    rows = []
    for lb in labels:
        xs, ys = lb.corners[0::2], lb.corners[1::2]
        # MOT boxes can also run past the frame edge; clip for the same reason.
        x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
        y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(f"{lb.class_id} {(x1 + x2) / 2 / width} {(y1 + y2) / 2 / height} "
                    f"{(x2 - x1) / width} {(y2 - y1) / height}")
    return rows


def collect_frames(cfg, flight_ids: list[str], modality: str,
                   label_file: str) -> list[FrameRef]:
    """Central frames available, with labels, in all three sources."""
    out: list[FrameRef] = []
    for fid in flight_ids:
        assets = resolve_flight(cfg, fid)
        ma = assets.modalities.get(modality)
        if ma is None or ma.mot is None:
            continue
        mot = load_mot(ma.mot)
        raw_dir = ma.frames_dir
        ortho_dir = cfg.paths.output_root / fid / modality
        alfs_dir = cfg.paths.alfs_output_root / fid / modality

        # Resolution of the raw frames, needed to normalise the MOT boxes.
        probe = next((raw_dir / f"{i:06d}.jpg" for i in sorted(mot)
                      if (raw_dir / f"{i:06d}.jpg").exists()), None)
        if probe is None:
            continue
        img = cv2.imread(str(probe))
        if img is None:
            continue
        raw_h, raw_w = img.shape[:2]

        for idx in sorted(mot):
            if idx % SAMPLING_STEP:
                continue
            raw_img = raw_dir / f"{idx:06d}.jpg"
            ortho_img = ortho_dir / f"{idx:06d}.png"
            alfs_img = alfs_dir / f"{idx:06d}.png"
            if not (raw_img.exists() and ortho_img.exists() and alfs_img.exists()):
                continue
            ortho_rows = _read_yolo(ortho_dir / f"{idx:06d}.txt")
            alfs_rows = _read_yolo(alfs_dir / f"{idx:06d}{label_file}")
            raw_rows = _raw_labels(mot[idx], raw_w, raw_h)
            # Require a non-empty label set everywhere; an empty one means the
            # box failed to project and the frame is not comparable.
            if not raw_rows or not ortho_rows or not alfs_rows:
                continue
            ref = FrameRef(flight_id=fid, frame_idx=idx)
            ref.per_source["raw"] = (raw_img, raw_rows)
            ref.per_source["ortho"] = (ortho_img, ortho_rows)
            ref.per_source["alfs"] = (alfs_img, alfs_rows)
            out.append(ref)
    return out


def official_split(counts: dict[str, int], splits_json: Path,
                   val_splits: tuple[str, ...]) -> dict[str, str]:
    """Map flight ids onto the dataset's own split file.

    `val_splits` names which official splits become our evaluation set (e.g.
    ``("val", "test")``); everything in official ``train`` becomes ours.
    Flights in neither are dropped, so a flight can never be trained on by us
    and evaluated on by an externally released model, or vice versa.
    """
    official = json.loads(splits_json.read_text(encoding="utf-8"))
    by_flight: dict[str, str] = {}
    for split_name, ids in official.items():
        for fid in ids:
            by_flight[str(fid)] = split_name

    out: dict[str, str] = {}
    unknown = []
    for fid in counts:
        official_name = by_flight.get(fid)
        if official_name in val_splits:
            out[fid] = "val"
        elif official_name == "train":
            out[fid] = "train"
        else:
            unknown.append(fid)
    if unknown:
        print(f"  dropping {len(unknown)} flight(s) absent from {splits_json.name}: "
              f"{sorted(unknown, key=int)}")
    return out


def split_flights(counts: dict[str, int], val_fraction: float,
                  seed: int) -> dict[str, str]:
    """Flight-level split targeting `val_fraction` of *frames*, not of flights.

    Flights vary from a handful of labelled frames to several hundred, so a
    plain random split of flight ids gives a wildly variable val size. Assigning
    the largest flights first to whichever split is furthest below its target
    keeps the frame ratio close while still splitting on flight boundaries.
    """
    rng = random.Random(seed)
    flights = sorted(counts, key=lambda f: (-counts[f], f))
    # Shuffle within equal-size groups so the split is not a deterministic
    # function of flight id alone.
    rng.shuffle(flights)
    flights.sort(key=lambda f: -counts[f])

    total = sum(counts.values())
    target_val = total * val_fraction
    assigned = {"train": 0, "val": 0}
    out: dict[str, str] = {}
    for fid in flights:
        val_deficit = target_val - assigned["val"]
        train_deficit = (total - target_val) - assigned["train"]
        split = "val" if val_deficit / max(target_val, 1) > \
            train_deficit / max(total - target_val, 1) else "train"
        out[fid] = split
        assigned[split] += counts[fid]
    return out


def _write_image(src: Path, dst: Path) -> bool:
    """Copy one frame as 3-channel JPEG, compositing any alpha over black."""
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
    return bool(cv2.imwrite(str(dst), img,
                            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]))


def build_dataset(root: Path, source: str, modality: str, frames: list[FrameRef],
                  split_of: dict[str, str], workers: int,
                  labels_only: bool = False) -> dict:
    ds = root / f"{source}_{modality}"
    for split in ("train", "val"):
        if not labels_only:
            (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / split).mkdir(parents=True, exist_ok=True)

    jobs, counts = [], {"train": 0, "val": 0}
    boxes = {"train": 0, "val": 0}
    for ref in frames:
        split = split_of[ref.flight_id]
        img_src, rows = ref.per_source[source]
        jobs.append((ref.stem, img_src, ds / "images" / split / f"{ref.stem}.jpg"))
        (ds / "labels" / split / f"{ref.stem}.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8")
        counts[split] += 1
        boxes[split] += len(rows)

    failed: set[str] = set()
    # The embedding detectors read embeddings, never the rendered pixels, so
    # transcoding 10k 2048px PNGs to JPEG for them is pure cost. Labels still go
    # through the identical split/clip path, which is what keeps a label set
    # built here comparable with one built for another resolution.
    if not labels_only:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for (stem, _, _), ok in zip(jobs, pool.map(lambda a: _write_image(*a[1:]), jobs)):
                if not ok:
                    failed.add(stem)

    if labels_only:
        # No data.yaml: it would point YOLO at an images/ tree that does not
        # exist here, failing at load time for a non-obvious reason.
        (ds / "LABELS_ONLY").write_text(
            f"# {source} / {modality} — labels only, no images.\n"
            "Built for the embedding detectors, which read cached DINOv3 grids\n"
            "instead of pixels. Not usable as an ultralytics dataset.\n",
            encoding="utf-8")
    else:
        (ds / "data.yaml").write_text(
            f"# {source} / {modality} — generated by build_yolo_datasets.py\n"
            f"path: {ds.as_posix()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(CLASS_NAMES)}\n"
            f"names:\n" + "".join(f"  {k}: {v}\n" for k, v in CLASS_NAMES.items()),
            encoding="utf-8")

    return {"dataset": ds.name, "train_images": counts["train"],
            "val_images": counts["val"], "train_boxes": boxes["train"],
            "val_boxes": boxes["val"], "failed_images": len(failed)}, failed


def drop_stems(root: Path, source: str, modality: str, stems: set[str],
               split_of: dict[str, str], frames: list[FrameRef]) -> int:
    """Remove a set of frames from one built dataset.

    Used to reconcile across sources: if an image failed to convert in *any*
    source, the frame is dropped from *all* of them, so the six models keep
    seeing an identical frame set.
    """
    ds = root / f"{source}_{modality}"
    split_by_stem = {ref.stem: split_of[ref.flight_id] for ref in frames}
    removed = 0
    for stem in stems:
        split = split_by_stem.get(stem)
        if split is None:
            continue
        for path in (ds / "images" / split / f"{stem}.jpg",
                     ds / "labels" / split / f"{stem}.txt"):
            if path.exists():
                path.unlink()
                removed += 1
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config.yaml")
    ap.add_argument("--out", type=Path, required=True,
                    help="Root directory to build the six datasets under.")
    ap.add_argument("--flight-ids", default="all")
    ap.add_argument("--modalities", default="rgb,thermal")
    ap.add_argument("--sources", default="raw,ortho,alfs")
    ap.add_argument("--split-mode", default="official", choices=("official", "seeded"))
    ap.add_argument("--splits-json", type=Path, default=None,
                    help="BAMBI flight_metadata/splits.json (required for "
                         "--split-mode official).")
    ap.add_argument("--val-splits", default="val,test",
                    help="Official split names to use as our evaluation set.")
    ap.add_argument("--val-fraction", type=float, default=0.2,
                    help="Only used with --split-mode seeded.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--alfs-labels", default="merged", choices=("merged", "central"),
                    help="Which ALFS label file to train on. merged = one box per "
                         "track over the aperture (NNNNNN.txt); central = the "
                         "central frame's boxes only (NNNNNN_central.txt).")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--labels-only", action="store_true",
                    help="Write labels but not images. For the embedding "
                         "detectors, which read cached DINOv3 grids rather than "
                         "pixels; avoids transcoding thousands of large PNGs.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.mode = "alfs"          # so resolve_flight fills in the ALFS paths
    flight_ids = (list_available_flight_ids(cfg) if args.flight_ids == "all"
                  else [f.strip() for f in args.flight_ids.split(",") if f.strip()])
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    label_file = ".txt" if args.alfs_labels == "merged" else "_central.txt"

    if args.split_mode == "official" and args.splits_json is None:
        raise SystemExit("--split-mode official requires --splits-json")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"split_mode": args.split_mode, "seed": args.seed,
               "val_fraction": args.val_fraction, "val_splits": args.val_splits,
               "alfs_labels": args.alfs_labels, "datasets": [], "splits": {}}

    for modality in modalities:
        print(f"\n=== {modality}: collecting frames common to {sources} ===")
        frames = collect_frames(cfg, flight_ids, modality, label_file)
        if not frames:
            print(f"  no comparable frames found for {modality}, skipping")
            continue
        per_flight = defaultdict(int)
        for ref in frames:
            per_flight[ref.flight_id] += 1
        if args.split_mode == "official":
            split_of = official_split(dict(per_flight), args.splits_json,
                                      tuple(v.strip() for v in args.val_splits.split(",")))
            # Flights outside the official split are dropped entirely.
            frames = [r for r in frames if r.flight_id in split_of]
        else:
            split_of = split_flights(dict(per_flight), args.val_fraction, args.seed)
        if not frames:
            print("  no frames left after splitting, skipping")
            continue
        n_val = sum(1 for r in frames if split_of[r.flight_id] == "val")
        print(f"  {len(frames)} frames over "
              f"{len({r.flight_id for r in frames})} flights; "
              f"val = {n_val} frames ({n_val / len(frames):.1%}) from "
              f"{len({r.flight_id for r in frames if split_of[r.flight_id] == 'val'})} flights")
        summary["splits"][modality] = split_of

        infos, all_failed = [], set()
        for source in sources:
            info, failed = build_dataset(args.out, source, modality, frames,
                                         split_of, args.workers, args.labels_only)
            infos.append(info)
            all_failed |= failed
            print(f"  {info['dataset']:16s} train {info['train_images']:6d} imgs / "
                  f"{info['train_boxes']:6d} boxes | val {info['val_images']:5d} / "
                  f"{info['val_boxes']:5d}" +
                  (f"  ({info['failed_images']} FAILED)" if info["failed_images"] else ""))

        if all_failed:
            # Keep the frame set identical across sources.
            print(f"  reconciling: dropping {len(all_failed)} frame(s) that failed "
                  f"in at least one source from all {len(sources)} datasets")
            for source, info in zip(sources, infos):
                drop_stems(args.out, source, modality, all_failed, split_of, frames)
                info["dropped_for_parity"] = len(all_failed)
        summary["datasets"].extend(infos)

    (args.out / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out / 'build_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
