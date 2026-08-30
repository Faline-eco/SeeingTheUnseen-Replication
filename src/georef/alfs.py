"""Airborne light field sampling (ALFS) rendering.

Where the orthographic path projects a *single* frame onto the DEM, ALFS
integrates a whole synthetic aperture: the central frame plus its neighbours,
all projected onto the same DEM through the *same* virtual orthographic camera
and averaged. Ground features (which the DEM focuses correctly) add up sharply
while occluders above the focus plane -- foliage -- are spread out and wash
away, so animals hidden under canopy in any single view can become visible.

The virtual camera is built exactly as in ``projection.FlightRenderer``, from
the central frame alone, so an ALFS render and the orthographic render of the
same central frame are pixel-aligned and their labels live in the same space.

Mirrors ``alfspy.orthografic_projection`` (the ``project_orthogonal=False``
branch) but loads the DEM / GL context once per flight, keeps the neighbour
shots streaming, and fixes the below-threshold pixel handling (see
``_IntegralRenderer``).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, cast

import cv2
import moderngl as mgl
import numpy as np
from numpy.typing import NDArray
from pyrr import Quaternion

from alfspy.core.convert.convert import pixel_to_world_coord
from alfspy.core.rendering import CtxShot, Renderer, TextureData
from alfspy.core.rendering.data import Resolution
from alfspy.core.util.defs import TRANSPARENT
from alfspy.orthografic_projection import (
    get_axis_aligned_bounding_box, get_camera_for_frame, to_yolo_format,
)
from alfspy.render.render import make_camera

from .config import AlfsSettings, RenderSettings
from .corrections import CorrectionProvider, FrameCorrection
from .labels import Label
from .projection import FlightRenderer, make_shot, project_to_image

log = logging.getLogger("georef")


@dataclass
class ApertureShot:
    """One contributing view of a synthetic aperture."""
    frame_idx: int
    image: Path
    meta: dict                  # pose entry for this frame
    correction: FrameCorrection


def build_virtual_camera(ctx, mesh_aabb, base_settings, image, meta: dict,
                         correction, fovy_fallback: float):
    """The ALFS orthographic camera for one central frame, without rendering.

    Same construction as :meth:`AlfsRenderer.virtual_camera`, factored out so a
    consumer that only needs the camera — the embedding-field integrator, which
    is deliberately GL-free — can build it without a ModernGL context or a DEM
    texture upload. Pass ``ctx=None``: the shot is lazy, so no GL call is made
    unless its texture is bound, which never happens here.

    ``image`` is passed explicitly rather than read out of ``meta``, mirroring
    ``virtual_camera``'s ``(image, meta)`` split — the pose dicts do not carry
    their own path.
    """
    settings = base_settings
    settings.correction = correction
    shot = make_shot(ctx, str(image), meta, correction, fovy_fallback)
    camera = make_camera(
        mesh_aabb, [shot], settings,
        rotation=Quaternion.from_matrix(
            shot.get_view().inverse @ shot.get_correction().inverse
        ),
    )
    return camera, shot


def _shot_path(shot: CtxShot) -> str:
    return str(getattr(shot, "_img_file", "<in-memory>"))


def _evict_bad_cache_file(shot: CtxShot, cache_dir: Path | None) -> None:
    """Delete an undecodable image, but only from the neighbour cache.

    Source frames under ``flights_root`` are never touched — those are input
    data, not something this pipeline may regenerate.
    """
    path = getattr(shot, "_img_file", None)
    if not path or cache_dir is None:
        return
    path = Path(path)
    try:
        if not path.is_relative_to(cache_dir):
            return
        path.unlink()
        log.warning("evicted undecodable neighbour cache file %s", path)
    except OSError:
        pass


class _PrefetchLoader:
    """Bounded-lookahead JPEG decoder backed by one long-lived thread pool.

    alfspy's ``AsyncShotLoader`` spins up a fresh ``ThreadPoolExecutor`` per call
    and never shuts it down, which leaks a dozen threads per render — untenable
    across tens of thousands of integrals. This keeps a single pool for the
    whole flight and decodes at most `lookahead` shots ahead of the GPU.
    """

    def __init__(self, workers: int = 8, lookahead: int = 12):
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._lookahead = lookahead
        # Set per modality; undecodable files under it are evicted so the next
        # extraction pass regenerates them.
        self.cache_dir: Path | None = None

    def __call__(self, shots: Sequence[CtxShot]) -> Iterator[CtxShot]:
        n = len(shots)
        futures = {}

        def submit(i: int) -> None:
            if i < n:
                futures[i] = self._pool.submit(shots[i].load_tex_input)

        for i in range(min(self._lookahead, n)):
            submit(i)
        try:
            for i in range(n):
                try:
                    futures.pop(i).result()
                except Exception as e:
                    # One unreadable frame must not cost the whole integral —
                    # drop that view and carry on with the rest of the aperture.
                    # The file is almost always a truncated write into the
                    # neighbour cache, so evict it and let the next extraction
                    # pass regenerate it.
                    log.warning("aperture shot %s failed to load (%s); "
                                "rendering without it", _shot_path(shots[i]), e)
                    _evict_bad_cache_file(shots[i], self.cache_dir)
                    submit(i + self._lookahead)
                    continue
                submit(i + self._lookahead)   # keep the pool ahead of the GPU
                yield shots[i]
        finally:
            for future in futures.values():
                future.cancel()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class _IntegralRenderer(Renderer):
    """``Renderer`` with a corrected integral read-back.

    alfspy's :meth:`Renderer.render_integral` calls ``np.divide(..., where=mask)``
    without an ``out=`` array, so every pixel below ``alpha_threshold`` keeps
    whatever was in the freshly allocated (uninitialised) buffer. For a 70 m
    orthographic window only the middle of the frame is covered by enough shots,
    so that is a large fraction of the image. Here those pixels are written as
    transparent black instead, which also gives the dataset builder a clean
    signal for "no light field data".
    """

    def _use_mask(self, mask: Optional[TextureData]) -> None:
        """Upload the mask only when it actually changes.

        The base implementation releases and re-uploads the texture on every
        call; the mask is fixed per modality, so for a 1024x1024 f4 RGBA mask
        that would be a wasted 16 MB transfer per integral.
        """
        if mask is not None and mask is getattr(self, "_cached_mask", None):
            self._shot_prog[self._PAR_MASK_FLAG].value = self._VAL_TRUE
            return
        super()._use_mask(mask)
        self._cached_mask = mask

    def render_integral_clean(self, shots: Iterable[CtxShot],
                              mask: Optional[TextureData] = None,
                              alpha_threshold: float = 2.0,
                              auto_contrast: bool = True,
                              release_shots: bool = True) -> tuple[NDArray, float, int]:
        """Additively blend all shots and normalise by the per-pixel shot count.

        Returns ``(rgba_uint8, coverage, shots_used)`` where ``coverage`` is the
        fraction of pixels that met ``alpha_threshold`` and ``shots_used`` is
        how many views actually made it into the integral (the loader drops any
        it cannot decode).
        """
        self._use_mask(mask)

        self._ctx.enable(cast(int, mgl.BLEND))
        self._ctx.disable(cast(int, mgl.DEPTH_TEST))
        self._ctx.blend_func = mgl.ADDITIVE_BLENDING
        self._fbo.clear(color=TRANSPARENT)
        used = 0
        try:
            for shot in shots:
                self._project_shot(shot)
                used += 1
                if release_shots:
                    shot.release()

            raw = self._fbo.read(components=4, dtype="f4", clamp=False)
        finally:
            self._ctx.disable(cast(int, mgl.BLEND))
            self._ctx.enable(cast(int, mgl.DEPTH_TEST))

        integral = np.frombuffer(raw, dtype=np.single).reshape((*self._fbo.size[1::-1], 4))
        # The alpha channel counts how many shots covered each pixel, because
        # every shot texture contributes alpha 1.0 where it has data.
        alpha = integral[:, :, -1][:, :, np.newaxis]
        valid = alpha >= alpha_threshold

        out = np.zeros_like(integral)
        np.divide(integral, alpha, out=out, where=valid)

        coverage = float(valid.mean())
        if auto_contrast and valid.any():
            rgb_valid = np.broadcast_to(valid, out.shape).copy()
            rgb_valid[:, :, -1] = False          # stretch colour, leave alpha
            lo = float(np.min(out[rgb_valid]))
            hi = float(np.max(out[rgb_valid]))
            if hi > lo:
                out[rgb_valid] = (out[rgb_valid] - lo) / (hi - lo)
        out[:, :, -1] = valid[:, :, 0]           # alpha := coverage flag

        result = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)[::-1, ...]
        return result, coverage, used


class AlfsRenderer(FlightRenderer):
    """Per-flight ALFS renderer: one DEM + GL context, many integrals."""

    def __init__(self, dem_glb: str | Path, render: RenderSettings,
                 alfs: AlfsSettings, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        super().__init__(dem_glb, render, origin=origin)
        self.alfs = alfs
        self._mask: TextureData | None = None
        self._load = _PrefetchLoader()

    def close(self) -> None:
        self._load.close()
        super().close()

    def set_modality(self, mask_path: str | Path | None,
                     neighbors_dir: Path | None) -> bool:
        """Point the renderer at one modality's mask and neighbour cache."""
        self._load.cache_dir = neighbors_dir
        return self.set_mask(mask_path)

    def set_mask(self, mask_path: str | Path | None) -> bool:
        """Load the modality mask applied to every shot texture.

        The thermal and RGB frames are letterboxed inside a 1024x1024 canvas;
        without the mask those black bars are integrated as if they were image
        data. Returns whether a mask is now active.
        """
        self._mask = None
        if not self.alfs.use_mask or mask_path is None:
            return False
        img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            log.warning("Could not read mask %s; integrating without a mask", mask_path)
            return False
        # Same conversion the shot textures get, so mask and shot share a
        # colour layout; TextureData normalises 0..255 to 0..1 on upload.
        self._mask = TextureData(CtxShot._cvt_img(img))
        return True

    def _ensure_renderer(self, camera) -> _IntegralRenderer:
        if self._renderer is None:
            self._renderer = _IntegralRenderer(
                self.resolution, self.ctx, camera, self.mesh_data, self.texture_data
            )
        else:
            self._renderer.camera = camera
            self._renderer.apply_matrices()
        return cast(_IntegralRenderer, self._renderer)

    def virtual_camera(self, central: ApertureShot):
        """The orthographic camera one integral is rendered through.

        Exposed separately so other renderers — notably the neural blend — can
        render the *same* view, keeping their output pixel-aligned with both the
        geometric ALFS integral and the orthographic render of the same frame.
        Returns ``(camera, central_shot)``; the shot is reused by the caller
        rather than rebuilt, since constructing it touches the GL context.
        """
        settings = self._base_settings
        settings.correction = central.correction.transform
        central_shot = make_shot(self.ctx, str(central.image), central.meta,
                                 central.correction.transform, self.render.fovy_fallback)
        # Identical construction to the orthographic path (FirstShot positioning
        # over the single central shot) => identical virtual camera.
        camera = make_camera(
            self.mesh_aabb, [central_shot], settings,
            rotation=Quaternion.from_matrix(
                central_shot.get_view().inverse @ central_shot.get_correction().inverse
            ),
        )
        return camera, central_shot

    def render_aperture(self, aperture: Sequence[ApertureShot], central_idx: int,
                        save_path: str | Path) -> tuple[object, float, int]:
        """Render one integral to `save_path`.

        `aperture` must contain the central frame; the virtual camera is built
        from it alone so the result aligns with the orthographic render of the
        same frame. Returns ``(camera, coverage, shots_used)``.
        """
        central = next(a for a in aperture if a.frame_idx == central_idx)
        camera, central_shot = self.virtual_camera(central)
        renderer = self._ensure_renderer(camera)

        shots = [
            central_shot if a.frame_idx == central_idx
            else make_shot(self.ctx, str(a.image), a.meta, a.correction.transform,
                           self.render.fovy_fallback)
            for a in aperture
        ]
        try:
            image, coverage, used = renderer.render_integral_clean(
                self._load(shots),
                mask=self._mask,
                alpha_threshold=self.alfs.alpha_threshold,
                auto_contrast=self.alfs.auto_contrast,
                release_shots=True,
            )
        finally:
            for shot in shots:
                shot.release()

        cv2.imwrite(str(save_path), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
        return camera, coverage, used

    def project_aperture_labels(
        self, poses: dict, central_idx: int, aperture: Sequence[ApertureShot],
        labels_by_frame: dict[int, list[Label]], input_resolution: Resolution,
        corr: CorrectionProvider, single_shot_camera,
    ) -> tuple[list[tuple[int, str]], list[tuple[int, str]],
               list[tuple[int, int, list[float]]]]:
        """Project labels into the integral's image space.

        Returns ``(merged, central, utm)``:

        * ``merged`` — one YOLO box per *track*, the axis-aligned hull of that
          track's boxes over every contributing frame. A moving animal is
          smeared across the integral along its trajectory, so this is the box
          that actually covers it (matches alfspy's ``merge_labels_in_alfs``).
        * ``central`` — only the central frame's boxes, i.e. what the
          orthographic render of the same frame would carry.
        * ``utm`` — the central frame's boxes as global UTM corners, same format
          and meaning as the orthographic pipeline's ``_utm.txt``.
        """
        merged_by_track: dict[int, tuple[int, list[float]]] = {}
        central_out: list[tuple[int, str]] = []
        utm_out: list[tuple[int, int, list[float]]] = []

        for shot in aperture:
            frame_labels = labels_by_frame.get(shot.frame_idx, [])
            if not frame_labels:
                continue
            is_central = shot.frame_idx == central_idx
            # Each contributing frame is projected through its own physical
            # camera, but always into the central frame's virtual camera.
            frame_corr = corr.for_frame(shot.frame_idx, central_frame_idx=central_idx)
            camera = get_camera_for_frame(
                poses, shot.frame_idx, frame_corr.rotation_eulers, frame_corr.translation,
            )
            for label in frame_labels:
                xs = [int(float(label.corners[i])) for i in range(0, len(label.corners), 2)]
                ys = [int(float(label.corners[i])) for i in range(1, len(label.corners), 2)]
                world = pixel_to_world_coord(
                    xs, ys, input_resolution.width, input_resolution.height,
                    self.tri_mesh, camera, include_misses=True,
                )

                if is_central and len(world) == 4 and all(w is not None for w in world):
                    flat: list[float] = []
                    for w in world:
                        gx, gy, gz = np.asarray(w, dtype=float) + self.origin
                        flat.extend([float(gx), float(gy), float(gz)])
                    utm_out.append((label.class_id, label.track_id, flat))

                hits = [w for w in world if w is not None]
                if not hits:
                    continue
                px, py = project_to_image(
                    hits, self.resolution.width, self.resolution.height, single_shot_camera,
                )
                box = [float(np.min(px)), float(np.min(py)),
                       float(np.max(px)), float(np.max(py))]

                prev = merged_by_track.get(label.track_id)
                if prev is None:
                    merged_by_track[label.track_id] = (label.class_id, box)
                else:
                    _, acc = prev
                    acc[0] = min(acc[0], box[0])
                    acc[1] = min(acc[1], box[1])
                    acc[2] = max(acc[2], box[2])
                    acc[3] = max(acc[3], box[3])

                if is_central:
                    yolo = _to_yolo(box, self.resolution)
                    if yolo is not None:
                        central_out.append((label.class_id, yolo))

        merged_out: list[tuple[int, str]] = []
        for class_id, box in merged_by_track.values():
            yolo = _to_yolo(box, self.resolution)
            if yolo is not None:
                merged_out.append((class_id, yolo))
        return merged_out, central_out, utm_out


def _to_yolo(box: list[float], resolution: Resolution) -> str | None:
    """`[x_min, y_min, x_max, y_max]` -> normalised YOLO `cx cy w h`.

    The box is clipped to the image before conversion. Merged ALFS boxes are
    hulls over a whole aperture, so a track that drifts past the edge of the
    render yields corners outside it; ``to_yolo_format`` only drops boxes that
    are *entirely* outside and would otherwise emit coordinates > 1, which
    YOLO rejects at load time as "non-normalized or out of bounds" — silently
    dropping the label and turning a real animal into background.
    """
    x1 = max(0.0, min(box[0], float(resolution.width)))
    y1 = max(0.0, min(box[1], float(resolution.height)))
    x2 = max(0.0, min(box[2], float(resolution.width)))
    y2 = max(0.0, min(box[3], float(resolution.height)))
    if x2 <= x1 or y2 <= y1:
        return None
    aabb = get_axis_aligned_bounding_box([np.array([
        [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]],
    ])])
    return to_yolo_format(aabb, resolution.width, resolution.height)
