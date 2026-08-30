"""Encode V-JEPA 2.1 features into a grid cell, matching the DINOv3 cells exactly.

Design C of `docker/PLAN_VJEPA_ABLATION.md`: V-JEPA enters the 2x2 as a
*representation*, not as a separate pipeline. Its tokens are stitched to the same
128x128 grid, PCA-reduced to the same 128 dims, and handed to the same CenterNet
head as every other cell. DINOv3 vs V-JEPA is then a pure representation swap.

Consequences of that choice, all deliberate:

  * the backbone is **frozen** and used once per frame — no LoRA, and none of
    `train_lora.py`'s `--limit-clips` cap, which exists only because adapting the
    encoder forces re-encoding every step;
  * the grid is **128x128**, i.e. 2048 px source at 16 px/cell, matching the
    DINOv3 cells rather than V-JEPA's native 64x64 at 1024 px. Resolution is not
    a free variable here: embedded ALFS fell 0.2437 -> 0.1918 going 2048 -> 1024,
    so equalising downward would trade a known effect for a saving.

Two cells:

  single  the central view only, repeated to fill the clip. No parallax, so the
          temporal axis carries nothing and this isolates the representation.
  multi   the real +-45 / stride-3 aperture, ortho-projected. The clip axis
          carries parallax, and V-JEPA integrates it itself.

    python encode_vjepa_cells.py --kind multi --stack-root <...>_stack \\
        --modality thermal --out <embeddings>/cell_thermal_vjepa_multi --fit-only
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("vjepa_cells")

# ImageNet statistics, which is what the vendored repo normalises with. Verified
# against the repo rather than assumed -- a wrong constant here would degrade
# every feature silently and look like "V-JEPA is worse".
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def tile_offsets(extent: int, tile: int) -> list[int]:
    """Left edges covering `extent` with `tile`-wide windows, stride = tile.

    The final window is pulled back to `extent - tile` rather than padded, so
    every tile is real image. That makes the last one overlap its neighbour,
    which `stitch` resolves by averaging.
    """
    if tile >= extent:
        return [0]
    offs = list(range(0, extent - tile + 1, tile))
    if offs[-1] != extent - tile:
        offs.append(extent - tile)
    return offs


def to_clip(views: np.ndarray, y: int, x: int, tile: int) -> torch.Tensor:
    """(S,H,W,C) uint8/float -> (1,T,C,tile,tile) normalised float32 crop."""
    crop = views[:, y:y + tile, x:x + tile, :].astype(np.float32) / 255.0
    crop = (crop - _MEAN) / _STD
    t = torch.from_numpy(crop).permute(0, 3, 1, 2).contiguous()   # (S,C,h,w)
    return t.unsqueeze(0)                                          # (1,S,C,h,w)


def stitch(tiles: dict[tuple[int, int], np.ndarray], grid: int,
           cells_per_tile: int, cell_px: int) -> np.ndarray:
    """Place per-tile (c,c,D) maps into a (grid,grid,D) canvas, averaging overlap."""
    d = next(iter(tiles.values())).shape[-1]
    acc = np.zeros((grid, grid, d), np.float32)
    cnt = np.zeros((grid, grid, 1), np.float32)
    for (y, x), m in tiles.items():
        cy, cx = y // cell_px, x // cell_px
        acc[cy:cy + cells_per_tile, cx:cx + cells_per_tile] += m
        cnt[cy:cy + cells_per_tile, cx:cx + cells_per_tile] += 1.0
    return acc / np.maximum(cnt, 1.0)


def encode_frame(encoder, encode_fn, views: np.ndarray, src: int, tile: int,
                 patch: int, tubelet: int, device: str, dtype,
                 ac_dtype=None) -> np.ndarray:
    """One frame's (S,H,W,C) view stack -> (grid, grid, D) pooled token map."""
    cells_per_tile = tile // patch
    cell_px = tile // cells_per_tile
    grid = src // cell_px
    out: dict[tuple[int, int], np.ndarray] = {}
    for y in tile_offsets(src, tile):
        for x in tile_offsets(src, tile):
            clip = to_clip(views, y, x, tile).to(device=device, dtype=dtype)
            with torch.inference_mode():
                if ac_dtype is None:
                    tok = encode_fn(encoder, clip)             # (1, T'*h*w, D)
                else:
                    with torch.autocast(device_type="cuda", dtype=ac_dtype):
                        tok = encode_fn(encoder, clip)
            n, d = tok.shape[1], tok.shape[2]
            hw = cells_per_tile * cells_per_tile
            if n % hw:
                raise RuntimeError(f"token count {n} not divisible by {hw}; "
                                   "tiling and model geometry disagree")
            # t-major: (T', h, w, D). Mean over T' -- for `single` every temporal
            # slice sees the same image, so this is a no-op there by construction.
            tok = tok.reshape(n // hw, cells_per_tile, cells_per_tile, d)
            out[(y, x)] = tok.float().mean(0).cpu().numpy()
    return stitch(out, grid, cells_per_tile, cell_px)


def load_views(path: Path, kind: str, clip_len: int) -> np.ndarray | None:
    """-> (T,H,W,3) uint8. `single` replicates the central view across the clip."""
    arr = np.load(path)
    if arr.ndim != 4:
        return None
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)          # grayscale on disk, 3ch for the model
    if kind == "single":
        centre = arr[len(arr) // 2]
        arr = np.repeat(centre[None], clip_len, axis=0)
    else:
        if len(arr) < clip_len:
            reps = -(-clip_len // len(arr))
            arr = np.tile(arr, (reps, 1, 1, 1))
        # tubelet 2 needs an even T; the aperture is 31, so one view is dropped
        # rather than duplicating a view and weighting it twice.
        arr = arr[:clip_len]
    return arr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack-root", type=Path, required=True,
                    help="Batch directory of per-view stacks from "
                         "render_embedding_field.py --emit stack.")
    ap.add_argument("--flight", default=None,
                    help="Encode only this flight's stacks. Without it every "
                         "worker processes every staged flight.")
    ap.add_argument("--kind", default="multi", choices=("single", "multi"))
    ap.add_argument("--modality", default="thermal", choices=("thermal", "rgb"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", default="2.1-vit-b-384")
    ap.add_argument("--vjepa-src", type=Path, default=Path("/scratch/bambi/vjepa2/src"))
    ap.add_argument("--src-px", type=int, default=2048)
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--clip-len", type=int, default=30,
                    help="Must be even (tubelet 2). The aperture is 31 views, so "
                         "the default drops one.")
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--pca-fit-frames", type=int, default=400)
    ap.add_argument("--train-stems", type=Path, default=None,
                    help="Stems eligible for the PCA fit, one per line. The basis "
                         "must never see val frames.")
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--autocast", default="fp16", choices=("fp16", "bf16", "off"),
                    help="Weights are always loaded fp32. Casting them directly "
                         "fails inside the vendored attention (query/key stay "
                         "fp32 while value casts); autocast casts per operation "
                         "and is consistent by construction. fp16 measured at "
                         "5.0x fp32 throughput with cosine 0.999991 against the "
                         "fp32 reference -- both faster and more faithful than "
                         "bf16 (0.999735).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debias-basis", type=Path, default=None,
                    help="INSID3 positional basis from fit_positional_basis_vjepa.py. "
                         "Given, the leading --debias-rank directions are projected "
                         "out of the tokens BEFORE the PCA, and --out gains a "
                         "_debias<rank> suffix so the baseline cells cannot be "
                         "overwritten. Must have been fitted for this --variant, "
                         "--tile and --clip-len.")
    ap.add_argument("--debias-rank", type=int, default=32,
                    help="Directions removed. 32 of 768 carried ~99% of the "
                         "noise response for DINOv3.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.clip_len % 2:
        ap.error("--clip-len must be even (tubelet size 2)")
    sys.path.insert(0, str(args.vjepa_src))
    from vjepa21 import encode as vj_encode, load_encoder   # noqa: E402

    # fp32 weights, always. See --autocast.
    dtype = torch.float32
    ac_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                "off": None}[args.autocast]

    # ---- INSID3 positional projection -------------------------------------
    # Applied to the stitched (grid, grid, D) map rather than to raw tokens.
    # The projection is linear and the map is a mean over the temporal axis and
    # over overlapping tiles, so projecting the mean equals the mean of the
    # projections -- identical result, one matmul per frame instead of per tile.
    #
    # It must happen BEFORE the PCA: the PCA basis is fitted in the debiased
    # space, which is why a debiased run cannot reuse the baseline pca.pkl.
    proj = None
    if args.debias_basis:
        z = np.load(args.debias_basis)
        b = z["basis"][:, :args.debias_rank]                    # (D, r)
        proj = (np.eye(b.shape[0], dtype=np.float32)
                - b @ b.T).astype(np.float32)                   # (D, D)
        # Derived, not taken from --out, so a debiased run structurally cannot
        # land in the baseline cell directory -- the same guarantee
        # encode_embeddings.py gives.
        suffix = f"_debias{args.debias_rank}"
        if not args.out.name.endswith(suffix):
            args.out = args.out.with_name(args.out.name + suffix)
        log.info("INSID3: removing %d of %d directions -> %s",
                 args.debias_rank, b.shape[0], args.out)

    args.out.mkdir(parents=True, exist_ok=True)

    # Refuse to write into a cell directory built for a different modality,
    # kind or backbone. The pipeline derives these paths from --modality so it
    # cannot collide, but a hand-typed --out can, and the failure would be
    # silent: mixed cells look perfectly well-formed.
    meta_path = args.out / "meta.json"
    if meta_path.exists():
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, now in (("modality", args.modality), ("kind", args.kind),
                         ("variant", args.variant),
                         ("debias_rank",
                          args.debias_rank if args.debias_basis else None)):
            if prev.get(key) not in (None, now):
                log.error("%s holds %s=%r but this run is %r. Refusing to mix.",
                          args.out, key, prev.get(key), now)
                return 1

    pca_path = args.out / "pca.pkl"

    # Restricted to one flight when asked. Globbing the whole staging root made
    # every GPU worker process every staged flight, so with 8 staged the three
    # workers did ~8x duplicated scanning and raced on the same outputs -- the
    # run was heading for ~40 h instead of ~9.
    pat = (f"{args.flight}/{args.modality}/*.npy" if args.flight
           else f"*/{args.modality}/*.npy")
    stacks = sorted(args.stack_root.glob(pat))
    stacks = [p for p in stacks if not p.name.endswith("_cov.npy")]
    if not stacks:
        log.error("no stacks under %s", args.stack_root)
        return 1
    log.info("%d frames in this batch", len(stacks))

    log.info("loading %s", args.variant)
    # torch.hub._parse_repo_info queries the GitHub API for the default branch
    # on every call, cached repo or not, so each encode makes a network request
    # that occasionally gets dropped (RemoteDisconnected). The weights
    # themselves are already cached under TORCH_HOME. Retry rather than lose the
    # flight: a dropped connection here is not a real failure.
    encoder = info = None
    for attempt in range(1, 6):
        try:
            encoder, info = load_encoder(args.variant, device=args.device, dtype=dtype)
            break
        except Exception as e:                       # noqa: BLE001 - any transport error
            if attempt == 5:
                raise
            wait = 5 * attempt
            log.warning("load_encoder attempt %d/5 failed (%s); retrying in %ds",
                        attempt, type(e).__name__, wait)
            time.sleep(wait)
    log.info("embed_dim %d, img_size %d, patch %d, tubelet %d",
             info["embed_dim"], info["img_size"], info["patch_size"],
             info["tubelet_size"])
    # A basis fitted for another variant has the wrong width. Caught here rather
    # than at the first matmul, where the traceback would not say why.
    if proj is not None and proj.shape[0] != info["embed_dim"]:
        log.error("--debias-basis is %d-dimensional but %s emits %d. The basis "
                  "must be fitted for this variant.",
                  proj.shape[0], args.variant, info["embed_dim"])
        return 1
    if args.tile != info["img_size"]:
        log.warning("--tile %d != model img_size %d; cells per tile will differ "
                    "from 24 and the grid will not be 128x128",
                    args.tile, info["img_size"])

    def stem_of(p: Path) -> str:
        return f"{p.parent.parent.name}_{p.stem}"

    # ---- PCA basis, fitted on training frames only ------------------------
    if pca_path.exists() and not args.overwrite:
        pca = pickle.loads(pca_path.read_bytes())
        log.info("reusing PCA basis %s -> %d", pca.n_features_in_, pca.n_components_)
    elif not args.fit_only:
        # Concurrent workers would each fit a basis on their own flight and race
        # to write it. The survivor would describe one flight, and every other
        # flight's cells would sit in a different linear space -- invisible
        # downstream and fatal to it. Refuse, exactly as encode_embeddings.py
        # does for its sharded workers.
        log.error("no PCA basis at %s. Run once with --fit-only on a batch "
                  "spanning several TRAINING flights before starting parallel "
                  "encoding; workers must not fit their own.", pca_path)
        return 1
    else:
        from sklearn.decomposition import PCA
        allow = None
        if args.train_stems and args.train_stems.exists():
            allow = set(args.train_stems.read_text(encoding="utf-8").split())
        pool = [p for p in stacks if allow is None or stem_of(p) in allow]
        if not pool:
            log.error("no training frames in this batch for the PCA fit; pass "
                      "--train-stems covering it, or fit on a batch that has some")
            return 1
        rng = np.random.default_rng(0)
        pick = rng.choice(len(pool), size=min(args.pca_fit_frames, len(pool)),
                          replace=False)
        log.info("fitting PCA on %d training frames", len(pick))
        chunks = []
        for n, i in enumerate(pick, 1):
            views = load_views(pool[i], args.kind, args.clip_len)
            if views is None:
                continue
            g = encode_frame(encoder, vj_encode, views, args.src_px, args.tile,
                             info["patch_size"], info["tubelet_size"],
                             args.device, dtype, ac_dtype)
            if proj is not None:
                g = g @ proj
            flat = g.reshape(-1, g.shape[-1])
            sel = rng.choice(len(flat), size=min(256, len(flat)), replace=False)
            chunks.append(flat[sel])
            if n % 25 == 0:
                log.info("  %d/%d", n, len(pick))
        flat = np.concatenate(chunks, 0)
        pca = PCA(n_components=args.pca_dim, random_state=0).fit(flat)
        pca_path.write_bytes(pickle.dumps(pca))
        log.info("PCA %d -> %d, explained variance %.1f%%", flat.shape[1],
                 args.pca_dim, 100 * pca.explained_variance_ratio_.sum())
    if args.fit_only:
        return 0

    # ---- encode ----------------------------------------------------------
    t0, written, skipped = time.time(), 0, 0
    for n, p in enumerate(stacks, 1):
        dst = args.out / f"{stem_of(p)}.npy"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        views = load_views(p, args.kind, args.clip_len)
        if views is None:
            continue
        g = encode_frame(encoder, vj_encode, views, args.src_px, args.tile,
                         info["patch_size"], info["tubelet_size"],
                         args.device, dtype, ac_dtype)
        if proj is not None:
            g = g @ proj
        h, w, d = g.shape
        red = pca.transform(g.reshape(-1, d)).reshape(h, w, -1).astype(np.float16)
        np.save(dst, red)
        written += 1
        if n % 25 == 0:
            rate = n / max(time.time() - t0, 1e-9)
            log.info("  %d/%d (%.2f frames/s)", n, len(stacks), rate)

    (args.out / "meta.json").write_text(json.dumps({
        "kind": args.kind, "modality": args.modality, "variant": args.variant,
        "autocast": args.autocast,
        "src_px": args.src_px, "tile": args.tile, "clip_len": args.clip_len,
        # cell_px == patch_size whenever tile divides by patch, so the grid is
        # simply src/patch. Spelled out rather than nested, because the earlier
        # form was correct but unreadable.
        "pca_dim": args.pca_dim,
        "debias_basis": str(args.debias_basis) if args.debias_basis else None,
        "debias_rank": args.debias_rank if args.debias_basis else None,
        "grid": [args.src_px // info["patch_size"]] * 2,
        "embed_dim": info["embed_dim"], "written": written, "skipped": skipped,
    }, indent=2), encoding="utf-8")
    log.info("[done] wrote %d, skipped %d in %.1f min",
             written, skipped, (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
