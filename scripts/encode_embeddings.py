"""Encode rendered frames into DINOv3 patch embeddings for embedding-space detection.

Runs DINOv3 ViT-H+/16 over already-rendered images (ALFS, ortho or raw) and
stores a PCA-reduced patch-embedding grid per frame, so a detection head can be
trained on embeddings instead of pixels.

Why PCA is not optional
-----------------------
DINOv3 returns ``(H/16, W/16, 1280)`` float32 — 20 MB for a 1024x1024 render,
i.e. **204 GB** for the thermal set alone. An unreduced cache of exactly this
kind reached 112 GB earlier in this project and filled the system disk. PCA to
64 dims in float16 is 0.5 MB/frame (~5 GB) and still supports both a D-dim
detection head and a 3-channel PCA-RGB variant (the first three components).

The PCA basis is fitted once on a subsample of *training-split* frames only and
saved alongside, so val/test frames are transformed by a basis that never saw
them.

    py scripts\\encode_embeddings.py --source alfs --modality thermal
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# alfspy lives in different places on the workstation and the DGX.
for _alfspy in (r"D:\alfs_py_embeddings\src",
                "/scratch/bambi/alfs_embed/code/alfspy_src"):
    if Path(_alfspy).is_dir():
        sys.path.insert(0, _alfspy)
        break

from georef.config import load_config                       # noqa: E402
from georef.labels import load_mot                          # noqa: E402
from georef.video_frames import SAMPLING_STEP               # noqa: E402

log = logging.getLogger("encode")

SOURCE_DIRS = {
    "alfs": ("alfs", ".png"),
    # The 2048px re-render, kept beside the 1024px set rather than replacing it
    # so both resolutions stay available for the comparison.
    "alfs_2k": ("alfs_2k", ".png"),
    "ortho": ("geo-referenced2", ".png"),
    "neural": ("neural_alfs", ".png"),
}


def project_out(x: np.ndarray, basis: np.ndarray, rank: int) -> np.ndarray:
    """INSID3 debiasing: F~ = F (I - B B^T), with B the top-`rank` columns.

    Kept byte-identical in behaviour to the V-JEPA project's `common.project_out`
    so both pipelines are debiased the same way; a difference here would show up
    as a representation effect in the comparison.
    """
    if rank <= 0:
        return x.astype(np.float32, copy=True)
    b = basis[:, :rank].astype(np.float32)
    shape = x.shape
    flat = x.reshape(-1, shape[-1]).astype(np.float32)
    flat = flat - (flat @ b) @ b.T
    return flat.reshape(shape)


def _load_image(path: Path) -> np.ndarray | None:
    """Read a render as 3-channel BGR, compositing alpha over black."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        return (img[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
    return img


def _scale(img: np.ndarray, scale: int) -> np.ndarray:
    """Upscale before DINOv3 so the patch grid — and thus effective stride — is finer."""
    if scale <= 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def collect_source_frames(cfg, modality: str,
                          splits: dict) -> list[tuple[str, int, Path]]:
    """(flight, frame, path) for every *source* frame an aperture actually reads.

    The embedded light field integrates DINOv3 features of the individual
    captures, not of a render, so this enumerates the aperture members rather
    than the central frames — including the neighbour-cache frames, which is
    where most of them live.

    Deliberately restricted to frames some labelled central's aperture touches.
    Encoding every cached frame would be several times the work for embeddings
    the integrator would never read.
    """
    from georef.discovery import resolve_flight
    from georef.video_frames import (aperture_indices, central_frame_indices,
                                     resolve_frame_path)
    out: list[tuple[str, int, Path]] = []
    for fid in sorted(splits, key=int):
        assets = resolve_flight(cfg, fid)
        if not assets.ready:
            continue
        ma = assets.modalities.get(modality)
        if ma is None or ma.mot is None or assets.poses is None:
            continue
        try:
            with open(assets.poses, "r", encoding="utf-8") as f:
                n_poses = len(json.load(f)["images"])
        except Exception as e:
            log.warning("flight %s: unreadable poses (%s)", fid, type(e).__name__)
            continue
        wanted: set[int] = set()
        for c in central_frame_indices(ma, n_poses):
            wanted.update(aperture_indices(c, cfg.alfs, n_poses))
        for idx in sorted(wanted):
            p = resolve_frame_path(ma, idx)
            if p is not None:
                out.append((fid, idx, p))
    return out


def collect_frames(cfg, root: Path, source: str, modality: str,
                   splits: dict) -> list[tuple[str, int, Path]]:
    """(flight, frame, image path) for every labelled central frame we can encode."""
    if source == "srcframes":
        return collect_source_frames(cfg, modality, splits)
    sub, ext = SOURCE_DIRS[source]
    out: list[tuple[str, int, Path]] = []
    for fid in sorted(splits, key=int):
        mot = cfg.paths.mot_root / modality / f"{fid}_accepted_{modality}_mot.txt"
        if not mot.exists():
            continue
        for idx in sorted(load_mot(mot)):
            if idx % SAMPLING_STEP:
                continue
            p = root / sub / fid / modality / f"{idx:06d}{ext}"
            if p.exists():
                out.append((fid, idx, p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config.yaml")
    ap.add_argument("--source", default="alfs",
                    choices=sorted(SOURCE_DIRS) + ["srcframes"],
                    help="A key of SOURCE_DIRS encodes rendered frames. "
                         "'srcframes' instead encodes the individual captures "
                         "an aperture reads, which is what the embedded light "
                         "field integrates.")
    ap.add_argument("--modality", default="thermal", choices=("thermal", "rgb"))
    ap.add_argument("--splits-json", type=Path,
                    default=Path(__file__).resolve().parents[1] / "bambi_splits_reference.json")
    ap.add_argument("--val-splits", default="test")
    ap.add_argument("--out", type=Path, default=Path("D:/embeddings"))
    ap.add_argument("--pca-dim", type=int, default=128,
                    help="PCA target dimensionality. On aggregate mAP, width "
                         "looks irrelevant: 64/128/256 gave 0.3530/0.3483/"
                         "0.3545 mAP50-95 over 3 seeds, inside one run's stdev. "
                         "But that average hides the cases the light field "
                         "exists for. Split by whether an animal is visible in "
                         "the central frame, the embedded light field recalls "
                         "0.4888 of the occluded ones at 128 dims versus 0.3408 "
                         "at 64 (3.3x stdev) — while the ALFS renders show no "
                         "such gap. Occluded animals are only ~23%% of val "
                         "boxes, so the effect is invisible in the mean. "
                         "Default 128; drop to 64 only where storage forces it, "
                         "and prefer encoding wide and truncating with "
                         "--use-dims, which keeps the question answerable. "
                         "(Separately: PCA reconstruction error is NOT a usable "
                         "proxy here — animal patches reconstruct 43-81%% worse "
                         "than background at every width, yet that penalty does "
                         "not track detection accuracy.)")
    ap.add_argument("--pca-fit-frames", type=int, default=400,
                    help="Training frames sampled to fit the PCA basis.")
    ap.add_argument("--model-dir", default=r"D:\DINOv3")
    ap.add_argument("--input-scale", type=int, default=1,
                    help="Upscale each render by this factor before DINOv3. "
                         "Patch-16 means the grid scales with it (1024->64x64, "
                         "2048->128x128), halving the effective stride. Costs "
                         "~8.8x time at scale 2, not 4x: ViT attention is "
                         "quadratic in token count.")
    ap.add_argument("--stems-file", type=Path, default=None,
                    help="Encode only these '<flight>_<frame>' stems, one per "
                         "line. Used to hold the frame set fixed across a "
                         "resolution comparison.")
    ap.add_argument("--debias-basis", type=Path, default=None,
                    help="positional_basis.npz from fit_positional_basis_dino.py. "
                         "Applies INSID3 debiasing F~ = F (I - B B^T) to the raw "
                         "patch tokens BEFORE PCA. Off by default; the published "
                         "embeddings were encoded without it.")
    ap.add_argument("--debias-rank", type=int, default=0,
                    help="Number of leading positional directions to project "
                         "out. 0 is a no-op even when a basis is given, so the "
                         "rank sweep needs no separate flag.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fit-only", action="store_true",
                    help="Fit the PCA basis, write pca.pkl and exit. Run this "
                         "once before launching sharded workers.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Split the frame list across N processes (one per GPU). "
                         "Every worker must share one PCA basis, so --fit-only "
                         "has to have produced pca.pkl first; workers refuse to "
                         "fit their own.")
    args = ap.parse_args()
    if not 0 <= args.shard < args.num_shards:
        ap.error(f"--shard must be in [0, {args.num_shards})")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config(args.config)
    root = cfg.paths.output_root.parent          # .../matched_dataset_new
    official = json.loads(args.splits_json.read_text(encoding="utf-8"))
    val_names = {v.strip() for v in args.val_splits.split(",")}
    split_of = {}
    for name, ids in official.items():
        for fid in ids:
            split_of[str(fid)] = "val" if name in val_names else "train"

    frames = collect_frames(cfg, root, args.source, args.modality, split_of)
    if args.stems_file:
        wanted = {s.strip() for s in args.stems_file.read_text().splitlines() if s.strip()}
        frames = [f for f in frames if f"{f[0]}_{f[1]:06d}" in wanted]
        log.info("restricted to %d frames from %s", len(frames), args.stems_file.name)
    if not frames:
        log.error("no frames found for %s/%s", args.source, args.modality)
        return 1
    train_frames = [f for f in frames if split_of.get(f[0]) == "train"]
    log.info("%d frames to encode (%d train, %d val)", len(frames),
             len(train_frames), len(frames) - len(train_frames))

    # Debiasing changes the feature distribution, so a PCA basis fitted on
    # undebiased tokens does not describe debiased ones. Reusing an existing
    # pca.pkl would silently transform by the wrong basis and the probe would
    # measure that mistake instead of the effect. Refuse rather than warn.
    debias_basis = None
    if args.debias_basis is not None:
        if args.debias_rank <= 0:
            log.warning("--debias-basis given with --debias-rank %d: this is a "
                        "no-op, encoding WITHOUT debiasing", args.debias_rank)
        else:
            z = np.load(args.debias_basis)
            debias_basis = z["basis"]
            evr = z["explained_variance_ratio"]
            log.info("INSID3 debiasing: rank %d of %d, removing %.2f%% of the "
                     "noise-feature variance", args.debias_rank,
                     debias_basis.shape[0], 100 * evr[:args.debias_rank].sum())

    # A debiased run ALWAYS writes to its own directory. This is not a
    # convenience: the undebiased embeddings and their pca.pkl are the baseline
    # the probe is measured against, and the whole comparison is lost if they are
    # refit in place. Deriving the name here makes that structurally impossible
    # rather than dependent on the caller remembering to pass a different --out
    # (and on nobody reaching for --overwrite to get past a complaint).
    suffix = f"_debias{args.debias_rank}" if debias_basis is not None else ""
    out_dir = args.out / f"{args.source}_{args.modality}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if suffix:
        log.info("debiased run -> %s (baseline %s_%s left untouched)",
                 out_dir.name, args.source, args.modality)
    pca_path = out_dir / "pca.pkl"
    have_basis = pca_path.exists() and not args.overwrite

    # Checked before the 4 GB model load, so a mis-ordered launch fails in
    # seconds instead of after every worker has pulled DINOv3 into VRAM.
    if not have_basis and args.num_shards > 1:
        # Four workers starting at once would each fit their own basis on their
        # own subsample and race to write pca.pkl. The surviving file would then
        # describe only one shard, and the other three shards' embeddings would
        # sit in a different linear space - invisible downstream, fatal to it.
        log.error("no PCA basis at %s and --num-shards=%d: run once with "
                  "--fit-only first so every worker shares one basis",
                  pca_path, args.num_shards)
        return 1

    # alfspy.embedding.__init__ imports the ModernGL render path, which a
    # headless container has no use for and cannot create a context in. The
    # extractor itself only needs cv2/numpy/torch/transformers, so ship it
    # standalone there and fall back to it.
    try:
        from alfspy.embedding import DinoV3Extractor
    except Exception as e:                       # pragma: no cover - container path
        log.info("alfspy.embedding unavailable (%s); using standalone extractor",
                 type(e).__name__)
        from dino_extractor import DinoV3Extractor
    log.info("loading DINOv3 from %s", args.model_dir)
    extractor = DinoV3Extractor(args.model_dir)

    # --- PCA basis, fitted on training frames only -------------------------
    if have_basis:
        pca = pickle.loads(pca_path.read_bytes())
        log.info("reusing PCA basis %s -> %d", pca.n_features_in_, pca.n_components_)
        if pca.n_components_ != args.pca_dim:
            # Silently encoding at the cached basis's width would produce a set
            # that looks like the requested one but isn't.
            log.error("pca.pkl is %d-dim but --pca-dim is %d; pass --overwrite "
                      "to refit, or point --out elsewhere",
                      pca.n_components_, args.pca_dim)
            return 1
    else:
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(0)
        pick = rng.choice(len(train_frames),
                          size=min(args.pca_fit_frames, len(train_frames)),
                          replace=False)
        log.info("fitting PCA on %d training frames", len(pick))
        chunks = []
        for n, i in enumerate(pick, 1):
            img = _load_image(train_frames[i][2])
            if img is None:
                continue
            img = _scale(img, args.input_scale)
            emb = extractor.extract(img).reshape(-1, 1280)
            if debias_basis is not None:
                emb = project_out(emb, debias_basis, args.debias_rank)
            # Subsample patches per frame: 400 frames x 4096 patches would be
            # 1.6M x 1280 and needlessly large to fit.
            sel = rng.choice(emb.shape[0], size=min(256, emb.shape[0]), replace=False)
            chunks.append(emb[sel])
            if n % 100 == 0:
                log.info("  %d/%d", n, len(pick))
        flat = np.concatenate(chunks, axis=0)
        pca = PCA(n_components=args.pca_dim, random_state=0).fit(flat)
        pca_path.write_bytes(pickle.dumps(pca))
        log.info("PCA 1280 -> %d, explained variance %.1f%%",
                 args.pca_dim, 100 * pca.explained_variance_ratio_.sum())

    if args.fit_only:
        log.info("--fit-only: basis written to %s, exiting", pca_path)
        return 0

    # --- encode ------------------------------------------------------------
    # Strided rather than blocked, so every shard sees a mix of flights and the
    # per-shard rate estimates stay comparable instead of one worker drawing
    # only the large flights.
    if args.num_shards > 1:
        frames = frames[args.shard::args.num_shards]
        log.info("shard %d/%d: %d frames", args.shard, args.num_shards, len(frames))
    t0 = time.time()
    written = skipped = failed = 0
    for n, (fid, idx, path) in enumerate(frames, 1):
        dst = out_dir / f"{fid}_{idx:06d}.npy"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        img = _load_image(path)
        if img is None:
            failed += 1
            continue
        img = _scale(img, args.input_scale)
        emb = extractor.extract(img)
        h, w, d = emb.shape
        flat = emb.reshape(-1, d)
        if debias_basis is not None:
            flat = project_out(flat, debias_basis, args.debias_rank)
        red = pca.transform(flat).reshape(h, w, -1).astype(np.float16)
        np.save(dst, red)
        written += 1
        if n % 250 == 0:
            rate = n / max(time.time() - t0, 1e-9)
            log.info("  %d/%d (%.1f frames/s, %.0f min left)", n, len(frames), rate,
                     (len(frames) - n) / max(rate, 1e-9) / 60)

    meta = {"source": args.source, "modality": args.modality,
            "pca_dim": args.pca_dim, "input_scale": args.input_scale,
            "debias_basis": str(args.debias_basis) if debias_basis is not None else None,
            "debias_rank": args.debias_rank if debias_basis is not None else 0,
            "frames": len(frames),
            "written": written, "skipped": skipped, "failed": failed,
            "grid": list(red.shape[:2]) if written else None,
            "split_of": {f[0]: split_of.get(f[0]) for f in frames}}
    # Per-shard filename: a shared meta.json would be overwritten by whichever
    # worker finished last, hiding the other three shards' failure counts.
    name = "meta.json" if args.num_shards == 1 else f"meta.shard{args.shard}.json"
    (out_dir / name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("done: %d written, %d skipped, %d failed in %.1f min",
             written, skipped, failed, (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
