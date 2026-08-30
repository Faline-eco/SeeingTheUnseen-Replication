"""Embedded light field: integrate DINOv3 features across an aperture, in PyTorch.

The geometric ALFS integral averages *pixels* from many views; this averages
their *DINOv3 features* instead — "average-then-encode" becomes
"encode-then-average". Same aperture, same virtual camera, same DEM.

Why not alfspy.embedding.render
-------------------------------
That path integrates feature channels through the ModernGL renderer, four
channels per pass — 320 passes for 1280 dims — and OpenGL in a container has
produced renderings with artifacts on this project's DGX. This implementation
uses no GL at all: embree ray-casts the target pixels onto the DEM, the world
points are projected into each source view with the *same* clip matrices the
renderer uses, and features are gathered with `grid_sample`. It therefore runs
anywhere, including inside the training container.

Correctness is checkable rather than assumed: run with `--channels rgb` and the
output should reproduce the geometric ALFS render of the same frame, because the
integral is then literally the same operation on the same data (see
`--selftest`).

    py scripts\\render_embedding_field.py --flight-ids 15 --modality thermal
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# alfspy lives in different places on the workstation and the DGX.
for _alfspy in (r"D:\alfs_py_embeddings\src",
                "/scratch/bambi/alfs_embed/code/alfspy_src"):
    if Path(_alfspy).is_dir():
        sys.path.insert(0, _alfspy)
        break

from georef.config import load_config                       # noqa: E402
from georef.corrections import CorrectionProvider           # noqa: E402
from georef.discovery import resolve_flight                 # noqa: E402
from georef.video_frames import (                           # noqa: E402
    aperture_indices, central_frame_indices, resolve_frame_path,
)

log = logging.getLogger("embfield")


def _load_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        return (img[:, :, :3].astype(np.float32) * a).astype(np.uint8)
    return img


def stack_views(sampler, world_xyz, tgt_center, hit_mask, out_hw: int,
                n_channels: int, use_feat: bool, mask_sampler=None):
    """Per-source projected maps, (S, H, W, C), WITHOUT averaging.

    This is `integrate` stopped one step early. Arm B of the V-JEPA ablation
    needs the aperture views registered to the focus plane but *not* integrated,
    so the model performs the integration itself; averaging here is precisely
    what must not happen.

    Invalid samples are zeroed rather than left as garbage, matching the
    renderer's convention (a shot contributes nothing where it does not cover,
    or where the modality mask blanks it). `reduce_stack` below turns the output
    back into `integrate`'s result, and `--selftest-stack` asserts it does to
    floating-point precision.
    """
    sv = sampler.sample(world_xyz, tgt_center)
    vals = sv.feat if use_feat else sv.rgb            # (S, P, C)
    valid_b = sv.valid
    if mask_sampler is not None:
        mv = mask_sampler.sample(world_xyz, tgt_center)
        valid_b = valid_b & (mv.feat[..., 0] > 0.5)
    valid = valid_b.to(vals.dtype)                    # (S, P)
    vals = vals * valid.unsqueeze(-1)

    n_src = vals.shape[0]
    idx = torch.from_numpy(np.flatnonzero(hit_mask)).to(vals.device)
    out = torch.zeros(n_src, out_hw * out_hw, n_channels,
                      device=vals.device, dtype=vals.dtype)
    out[:, idx] = vals
    cov = torch.zeros(n_src, out_hw * out_hw, device=vals.device, dtype=vals.dtype)
    cov[:, idx] = valid
    return (out.reshape(n_src, out_hw, out_hw, n_channels),
            cov.reshape(n_src, out_hw, out_hw))


def reduce_stack(stack, cov, alpha_threshold: float):
    """Collapse a per-view stack back to the integrated field.

    Exists so the un-averaged path can be checked against the averaged one
    rather than trusted: any divergence means the stack is not the same data the
    validated integral was built from.
    """
    count = cov.sum(dim=0)                                     # (H, W)
    summed = stack.sum(dim=0)                                  # (H, W, C)
    mean = summed / count.clamp(min=1).unsqueeze(-1)
    mean[count < alpha_threshold] = 0.0
    return mean, count


def integrate(sampler, world_xyz, tgt_center, hit_mask, out_hw: int,
              n_channels: int, alpha_threshold: float, use_feat: bool,
              mask_sampler=None):
    """Mean of the per-source samples at each world point, as an (H, W, C) map.

    This is the geometric ALFS integral written out explicitly: gather each
    source's value at the point, average over the sources that actually cover
    it, and blank pixels covered by fewer than `alpha_threshold` sources — the
    same coverage rule `Renderer.render_integral` applies.
    """
    sv = sampler.sample(world_xyz, tgt_center)
    vals = sv.feat if use_feat else sv.rgb            # (S, P, C)
    valid_b = sv.valid
    if mask_sampler is not None:
        # The renderer multiplies every shot by the modality mask, and because
        # that multiplies alpha too, masked pixels contribute *nothing* to the
        # integral. Frames are letterboxed, so without this the black bars are
        # averaged in as if they were data - which is exactly what made the
        # RGB self-test correlate only ~0.66 with the reference render.
        mv = mask_sampler.sample(world_xyz, tgt_center)
        valid_b = valid_b & (mv.feat[..., 0] > 0.5)
    valid = valid_b.to(vals.dtype)                    # (S, P)
    count = valid.sum(dim=0)                          # (P,)
    summed = (vals * valid.unsqueeze(-1)).sum(dim=0)  # (P, C)
    mean = summed / count.clamp(min=1).unsqueeze(-1)
    mean[count < alpha_threshold] = 0.0

    out = torch.zeros(out_hw * out_hw, n_channels, device=vals.device, dtype=vals.dtype)
    idx = torch.from_numpy(np.flatnonzero(hit_mask)).to(vals.device)
    out[idx] = mean
    cov = torch.zeros(out_hw * out_hw, device=vals.device, dtype=vals.dtype)
    cov[idx] = count
    return out.reshape(out_hw, out_hw, n_channels), cov.reshape(out_hw, out_hw)


def main() -> int:
    from alfspy.neural.sampler import MultiViewSampler
    from alfspy.neural.geometry import stack_clip_matrices
    from alfspy.core.rendering import CtxShot, Resolution
    from alfspy.core.util.pyrrs import quaternion_from_eulers
    from alfspy.core.util.geo import get_aabb
    from alfspy.render.data import BaseSettings, CameraPositioningMode
    from alfspy.render.render import process_render_data, read_gltf
    from pyrr import Vector3
    import trimesh
    from georef.alfs import build_virtual_camera, ApertureShot
    from alfspy.core.convert.convert import pixel_to_world_coord

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config.yaml")
    ap.add_argument("--flight-ids", default="15")
    ap.add_argument("--modality", default="thermal", choices=("thermal", "rgb"))
    ap.add_argument("--channels", default="feat", choices=("feat", "rgb"),
                    help="'feat' integrates DINOv3 embeddings; 'rgb' integrates "
                         "raw colour and should reproduce the geometric ALFS "
                         "render (used by --selftest).")
    ap.add_argument("--embeddings", type=Path, default=Path("D:/embeddings_src"),
                    help="Per-source-frame DINOv3 .npy grids (see "
                         "encode_embeddings.py), named <flight>_<frame>.npy.")
    ap.add_argument("--out", type=Path, default=Path("D:/embedding_field"))
    ap.add_argument("--out-hw", type=int, default=128,
                    help="Output grid. 128 matches a 2048px render encoded at "
                         "patch-16, so the two embedding variants are directly "
                         "comparable.")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--alpha-threshold", type=float, default=2.0)
    ap.add_argument("--aperture", default="full", choices=("full", "single"),
                    help="'single' collapses the aperture to the central frame "
                         "alone. Combined with --channels this spans the whole "
                         "2x2 (single/multi view x sensor/embedding) through "
                         "one code path, so cells differ only in the variable "
                         "under study rather than in renderer, head or "
                         "coordinate frame.")
    ap.add_argument("--emit", default="mean", choices=("mean", "stack"),
                    help="'mean' writes the integrated field (the existing "
                         "behaviour). 'stack' writes the per-view registered "
                         "maps, (S, H, W, C), for V-JEPA arm B — registered but "
                         "not integrated, so the model integrates.")
    ap.add_argument("--selftest-stack", action="store_true",
                    help="Assert that reducing the per-view stack reproduces "
                         "the integrated field to floating-point precision "
                         "(~1e-7, not bit-identical: pre-zeroing the samples "
                         "associates the sum differently than masking during "
                         "it), then exit. Run before generating, not after.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute frames that already have output. Off by "
                         "default so an interrupted run resumes.")
    ap.add_argument("--selftest", action="store_true",
                    help="Integrate RGB and report agreement with the existing "
                         "geometric ALFS render instead of writing embeddings.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config(args.config)
    cfg.mode = "alfs"
    cfg.validate()
    if args.aperture == "single":
        # Collapse the aperture to the central frame. Everything else — virtual
        # camera, DEM, target grid, coverage rule — is untouched, so the
        # single-view cell differs from the multi-view one in view count alone.
        cfg.alfs.neighbors_before = 0
        cfg.alfs.neighbors_after = 0
        # With one view every covered pixel has coverage exactly 1, so the
        # renderer's "at least 2 views" rule would blank the entire output.
        args.alpha_threshold = min(args.alpha_threshold, 1.0)
        log.info("single-view mode: aperture=1, alpha_threshold=%.1f",
                 args.alpha_threshold)
    device = torch.device(args.device)
    use_feat = args.channels == "feat" and not args.selftest
    flights = [f.strip() for f in args.flight_ids.split(",") if f.strip()]

    for fid in flights:
        assets = resolve_flight(cfg, fid)
        if not assets.ready:
            log.warning("flight %s not ready: %s", fid, assets.missing_essential)
            continue
        ma = assets.modalities.get(args.modality)
        if ma is None or ma.mot is None:
            continue

        with open(assets.poses, "r", encoding="utf-8") as f:
            poses = json.load(f)
        n_poses = len(poses["images"])
        corr = CorrectionProvider(assets.correction)
        centrals = central_frame_indices(ma, n_poses)
        if args.max_frames:
            centrals = centrals[:args.max_frames]
        if not centrals:
            continue

        mesh_data, texture_data = read_gltf(str(assets.dem_glb))
        # Must be built before process_render_data, which mutates the arrays.
        tri = trimesh.Trimesh(vertices=np.asarray(mesh_data.vertices),
                              faces=np.asarray(mesh_data.indices))
        log.info("flight %s: %d centrals, DEM %d faces, embree=%s",
                 fid, len(centrals), len(tri.faces),
                 type(tri.ray).__module__.endswith("pyembree"))

        # GL-free stand-in for AlfsRenderer: we need only the camera geometry,
        # never a rendered pixel, so we take the two things make_camera reads
        # (the processed mesh's AABB and the base settings) and skip the
        # ModernGL context entirely. build_virtual_camera is bit-identical to
        # AlfsRenderer.virtual_camera (max|diff| 0.0 on view/proj/position).
        proc_mesh, _ = process_render_data(mesh_data, texture_data)
        mesh_aabb = get_aabb(proc_mesh.vertices)
        base_settings = BaseSettings(
            count=1,
            initial_skip=0,
            add_background=False,
            camera_position_mode=CameraPositioningMode.FirstShot,
            fovy=cfg.render.fovy_fallback,
            aspect_ratio=cfg.render.aspect_ratio,
            orthogonal=True,
            ortho_size=(cfg.render.ortho_width, cfg.render.ortho_height),
            correction=None,
            resolution=Resolution(cfg.render.render_width, cfg.render.render_height),
        )
        # A stack run ALWAYS writes under its own root. The per-view files carry
        # exactly the same names as the integrated fields ("<frame>.npy"), so
        # sharing a root would overwrite the existing field_* trees with data of
        # a different shape and rank -- silently, since nothing downstream
        # checks. Deriving the suffix here makes that impossible rather than
        # relying on the caller passing a different --out.
        root = args.out if args.emit == "mean" else Path(f"{args.out}_stack")
        out_dir = root / fid / args.modality
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        done = 0
        skipped = 0

        for c in centrals:
            # Resume support: a long integration run should not have to redo
            # completed frames after an interruption or a code change that
            # leaves the output identical. Both files must be present — a field
            # without its coverage map is a half-written frame.
            if not args.selftest and not args.overwrite:
                fp, cp = out_dir / f"{c:06d}.npy", out_dir / f"{c:06d}_cov.npy"
                if fp.exists() and cp.exists():
                    skipped += 1
                    continue
            aperture = []
            for idx in aperture_indices(c, cfg.alfs, n_poses):
                img = resolve_frame_path(ma, idx)
                if img is None:
                    continue
                aperture.append(ApertureShot(
                    frame_idx=idx, image=img, meta=poses["images"][idx],
                    correction=corr.for_frame(idx, central_frame_idx=c)))
            if not any(a.frame_idx == c for a in aperture):
                continue

            central = next(a for a in aperture if a.frame_idx == c)
            camera, _ = build_virtual_camera(
                None, mesh_aabb, base_settings, central.image, central.meta,
                central.correction.transform, cfg.render.fovy_fallback)

            # Source shots: geometry only; textures are features or pixels.
            shots, imgs, feats = [], [], []
            for a in aperture:
                rot = [v % 360.0 for v in a.meta["rotation"]]
                q = quaternion_from_eulers([np.deg2rad(v) for v in rot], "zyx")
                fov = a.meta.get("fovy", cfg.render.fovy_fallback)
                if isinstance(fov, (list, tuple)):
                    fov = fov[0] if fov and fov[0] is not None else cfg.render.fovy_fallback
                # ctx=None: CtxShot is lazy, so the texture is never bound
                # and no GL call is made. We use only its view/projection.
                shots.append(CtxShot(None, str(a.image), Vector3(a.meta["location"]),
                                     q, float(fov), 1, a.correction.transform, True))
                # In feature mode the RGB slot is never read: integrate() takes
                # sv.feat, and the sampler normalises coordinates so the two
                # streams need not share a resolution. Decoding 31 full JPEGs
                # per central frame only to grid_sample and discard them was the
                # dominant cost (3.17 s/frame, GPU at 0-2%), so feed a 1x1
                # placeholder instead and keep the real load for --channels rgb.
                if use_feat:
                    imgs.append(torch.zeros(3, 1, 1))
                else:
                    bgr = _load_bgr(a.image)
                    imgs.append(torch.from_numpy(
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1) / 255.0)
                if use_feat:
                    fp = args.embeddings / f"{fid}_{a.frame_idx:06d}.npy"
                    if not fp.exists():
                        feats = None
                        break
                    feats.append(torch.from_numpy(
                        np.load(fp).astype(np.float32)).permute(2, 0, 1))
            if use_feat and not feats:
                log.warning("flight %s frame %d: missing source embeddings", fid, c)
                continue

            images = torch.stack(imgs, 0).to(device)
            fstack = torch.stack(feats, 0).to(device) if use_feat else None
            clip = torch.from_numpy(stack_clip_matrices(shots)).to(device)
            centers = torch.from_numpy(np.stack(
                [np.asarray(s.camera.transform.position, np.float32) for s in shots])).to(device)
            sampler = MultiViewSampler(images, fstack, clip, centers)

            # Same modality mask the renderer applies to every shot, served
            # through a second sampler as a single channel.
            mask_sampler = None
            if cfg.alfs.use_mask and ma.mask is not None:
                mk = cv2.imread(str(ma.mask), cv2.IMREAD_UNCHANGED)
                if mk is not None:
                    if mk.ndim == 3:
                        mk = mk[:, :, 0]
                    mk = (mk.astype(np.float32) / 255.0)
                    mkt = torch.from_numpy(mk)[None, None].repeat(
                        len(shots), 1, 1, 1).to(device)
                    mask_sampler = MultiViewSampler(images, mkt, clip, centers)

            # Target grid -> DEM. The orthographic branch of
            # pixel_to_world_coord is used here and was broken upstream
            # (crash + wrong scale + wrong handedness); this depends on the
            # patched version in patches/alfs_py_embeddings.patch.
            hw = args.out_hw
            ys, xs = np.meshgrid(np.arange(hw), np.arange(hw), indexing="ij")
            xs = (xs.reshape(-1) + 0.5).astype(np.float32)
            ys = (ys.reshape(-1) + 0.5).astype(np.float32)
            res = pixel_to_world_coord(xs, ys, hw, hw, tri, camera, include_misses=True)
            hit = np.array([r is not None for r in res])
            if not hit.any():
                continue
            world = torch.from_numpy(np.stack(
                [np.asarray(r, np.float32) for r in res if r is not None])).to(device)
            tgt_c = torch.from_numpy(
                np.asarray(camera.transform.position, np.float32)).to(device)

            n_ch = fstack.shape[1] if use_feat else 3

            if args.emit == "stack" or args.selftest_stack:
                views, vcov = stack_views(sampler, world, tgt_c, hit, hw, n_ch,
                                          use_feat, mask_sampler)

            if args.selftest_stack:
                # The un-averaged path must be the same data the validated
                # integral is built from, or everything downstream of it is
                # measuring a different quantity.
                field, cov = integrate(sampler, world, tgt_c, hit, hw, n_ch,
                                       args.alpha_threshold, use_feat,
                                       mask_sampler)
                red, rcov = reduce_stack(views, vcov, args.alpha_threshold)
                dm = (red - field).abs().max().item()
                dc = (rcov - cov).abs().max().item()
                log.info("frame %d: stack->integral max|d| field %.3e cov %.3e "
                         "| views %d | %s", c, dm, dc, views.shape[0],
                         "OK" if dm < 1e-5 and dc < 1e-5 else "MISMATCH")
                done += 1
                for sh in shots:
                    sh.release()
                continue

            if args.emit == "stack":
                # uint8: these are re-read as images by the V-JEPA encoder, and
                # float16 would triple the batch footprint for no benefit --
                # the source frames are 8-bit to begin with.
                arr = (views.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                if arr.shape[-1] == 3:
                    # Thermal is grey, but the sources are JPEG: chroma
                    # subsampling leaves the channels unequal by a few levels, so
                    # an exact test never fires and the file is 3x larger for
                    # nothing. Collapse when the spread is compression-sized
                    # (<= 8/255 on a subsample); genuine RGB is far above that.
                    sub = arr[::8, ::64, ::64, :].astype(np.int16)
                    # Per-pixel channel spread at the 99th percentile, not the
                    # max. Measured on real thermal: max 11, p99 3, mean 1.1 --
                    # grey content plus JPEG chroma-subsampling noise, with a
                    # thin tail. Keying on the max lets a handful of pixels force
                    # 3x the storage; keying on p99 collapses thermal and still
                    # leaves genuine RGB (p99 ~250) far outside.
                    # (Taking max(per-pixel max) - min(per-pixel min) instead
                    # would measure image dynamic range and never fire at all.)
                    spread = (int(np.percentile(sub.max(-1) - sub.min(-1), 99))
                              if sub.size else 255)
                    if spread <= 8:
                        arr = arr.mean(-1, keepdims=True).round().astype(np.uint8)
                np.save(out_dir / f"{c:06d}.npy", arr)
                np.save(out_dir / f"{c:06d}_cov.npy",
                        vcov.sum(0).cpu().numpy().astype(np.uint8))
                done += 1
                for sh in shots:
                    sh.release()
                continue

            field, cov = integrate(sampler, world, tgt_c, hit, hw, n_ch,
                                   args.alpha_threshold, use_feat,
                                   mask_sampler)

            if args.selftest:
                ref_p = cfg.paths.alfs_output_root / fid / args.modality / f"{c:06d}.png"
                ref = _load_bgr(ref_p)
                if ref is None:
                    log.warning("no reference render at %s", ref_p)
                    continue
                ref = cv2.cvtColor(cv2.resize(ref, (hw, hw), interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                ours = field.cpu().numpy()
                m = (cov.cpu().numpy() >= args.alpha_threshold) & (ref.sum(-1) > 0)
                if m.sum() < 100:
                    continue
                a_, b_ = ours[m], ref[m]
                # Scale/offset differ: render_integral auto-contrasts, we do not.
                aa = (a_ - a_.mean()) / (a_.std() + 1e-8)
                bb = (b_ - b_.mean()) / (b_.std() + 1e-8)
                log.info("frame %d: covered px %d | corr %.4f | MAE(norm) %.4f",
                         c, int(m.sum()), float((aa * bb).mean()),
                         float(np.abs(aa - bb).mean()))
            else:
                np.save(out_dir / f"{c:06d}.npy",
                        field.cpu().numpy().astype(np.float16))
                np.save(out_dir / f"{c:06d}_cov.npy",
                        cov.cpu().numpy().astype(np.uint8))
            done += 1
            for s in shots:
                s.release()
        log.info("flight %s: %d frames in %.1f min (%.2f s/frame), %d skipped",
                 fid, done, (time.time() - t0) / 60,
                 (time.time() - t0) / max(done, 1), skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
