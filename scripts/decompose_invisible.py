"""Why is a merged-GT animal missing from the central frame — occluded, or simply not in view?

The "invisible centrally" subset drives the multi-view claim, but it has always
been a geometric proxy: a merged-aperture box with no central-frame counterpart.
That lumps together two very different cases.

  * The central view covered that patch of ground and no animal was annotated
    there -> the animal was hidden (canopy, shadow, another body). This is the
    case the light-field argument is about.
  * The central view never covered that ground at all -> the aperture simply
    sees a wider footprint. Recovering those animals is real but trivial: more
    ground, more animals, no occlusion reasoning involved.

The single-view coverage map (cov > 0) says exactly which ground the central
view saw, so the two can be separated. A third signal cross-checks it: the
Zenodo track annotations give each track a trim range, so an animal outside its
own lifetime at that frame was not in the scene to begin with.

    python decompose_invisible.py --modality thermal
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_yolo(p: Path) -> np.ndarray:
    if not p.exists():
        return np.zeros((0, 4), np.float32)
    rows = [[float(v) for v in ln.split()[1:5]]
            for ln in p.read_text(encoding="utf-8").splitlines()
            if len(ln.split()) == 5]
    return np.asarray(rows, np.float32).reshape(-1, 4)


def to_xyxy(b):
    o = np.empty_like(b)
    o[:, 0] = b[:, 0] - b[:, 2] / 2
    o[:, 1] = b[:, 1] - b[:, 3] / 2
    o[:, 2] = b[:, 0] + b[:, 2] / 2
    o[:, 3] = b[:, 1] + b[:, 3] / 2
    return o


def iou_mat(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    i = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (i / np.maximum(aa[:, None] + ab[None, :] - i, 1e-9)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", default="thermal", choices=("thermal", "rgb"))
    ap.add_argument("--fields", type=Path,
                    default=Path("/scratch/bambi/datasets/alfs_embed"))
    ap.add_argument("--alfs-root", type=Path,
                    default=Path("/scratch/bambi/datasets/alfs_embed/matched_dataset_new/alfs_2k"))
    ap.add_argument("--yolo", type=Path,
                    default=Path("/scratch/bambi/datasets/alfs_embed/yolo_datasets"))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None,
                    help="Write {stem: [bool,...]} marking which "
                         "merged-GT boxes are genuinely hidden (the "
                         "central view covered that ground), so recall "
                         "can be restricted to them.")
    ap.add_argument("--labels-dataset", default=None,
                    help="Central-GT dataset supplying the frame list. "
                         "Defaults to alfs_2k_<mod>_centralgt.")
    ap.add_argument("--splits", default="val",
                    help="Comma-separated splits to scan. Use 'train,val' to "
                         "build one global mask: whether a box is hidden is a "
                         "property of the frame, not of which CV fold it lands "
                         "in, so one mask serves every fold.")
    ap.add_argument("--margin", type=int, default=1,
                    help="Cells of slack around the box when asking whether the "
                         "central view covered it; 0 would call a box straddling "
                         "the coverage edge 'not in view' on a single pixel.")
    args = ap.parse_args()
    mod = args.modality

    ds = args.labels_dataset or f"alfs_2k_{mod}_centralgt"
    stems = sorted({p.stem
                    for sp in args.splits.split(",")
                    for p in (args.yolo / ds / "labels" / sp.strip()).glob("*.txt")})
    c = Counter()
    per_frame_cov = []
    hidden_mask: dict[str, list[int]] = {}

    for stem in stems:
        flight, fr = stem.rsplit("_", 1)
        frame = int(fr)
        d = args.alfs_root / flight / mod
        merged = read_yolo(d / f"{frame:06d}.txt")
        central = read_yolo(d / f"{frame:06d}_central.txt")
        if not len(merged):
            continue
        m_xy, c_xy = to_xyxy(merged), to_xyxy(central)
        invis = (iou_mat(m_xy, c_xy).max(1) < args.iou if len(c_xy)
                 else np.ones(len(m_xy), bool))
        if not invis.any():
            continue

        covp = (args.fields / f"field_{mod}_rgb_single" / flight / mod /
                f"{frame:06d}_cov.npy")
        if not covp.exists():
            c["no_coverage_map"] += int(invis.sum())
            continue
        cov = np.load(covp)                      # single view: 1 where seen
        n = cov.shape[0]
        seen = cov > 0
        per_frame_cov.append(float(seen.mean()))

        flags = []
        for b in m_xy[invis]:
            x0 = max(int(b[0] * n) - args.margin, 0)
            x1 = min(int(np.ceil(b[2] * n)) + args.margin, n)
            y0 = max(int(b[1] * n) - args.margin, 0)
            y1 = min(int(np.ceil(b[3] * n)) + args.margin, n)
            patch = seen[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
            if patch.size == 0:
                c["outside_grid"] += 1; flags.append(0)
            elif patch.mean() >= 0.5:
                c["covered_but_unannotated"] += 1; flags.append(1)   # hidden
            elif patch.any():
                c["partly_covered"] += 1; flags.append(0)
            else:
                c["never_in_view"] += 1; flags.append(0)   # wider footprint

        hidden_mask[stem] = flags

    if args.out:
        args.out.write_text(json.dumps(hidden_mask), encoding='utf-8')
        print(f'wrote {args.out}')

    tot = sum(c[k] for k in ("covered_but_unannotated", "partly_covered",
                             "never_in_view", "outside_grid", "no_coverage_map"))
    print(f"modality: {mod}")
    print(f"  'invisible centrally' boxes: {tot}")
    print(f"  single-view coverage of the ortho canvas: "
          f"{100 * np.mean(per_frame_cov):.1f}% (mean over frames)")
    print()
    for k, label in (("covered_but_unannotated", "central view SAW that ground -> hidden"),
                     ("partly_covered", "box straddles the coverage edge"),
                     ("never_in_view", "central view never covered it -> wider footprint"),
                     ("outside_grid", "outside the grid entirely"),
                     ("no_coverage_map", "no coverage map available")):
        if c[k]:
            print(f"  {c[k]:5d}  ({100*c[k]/max(tot,1):5.1f} %)  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
