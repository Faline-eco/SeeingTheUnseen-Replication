"""Core orthographic projection: render a frame onto its DEM and project labels.

Wraps the alfspy rendering primitives the way the reference notebook (cell 27,
single-orthographic case) does, but loads the DEM / GL context once per flight
and renders many frames against it.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from pyrr import Quaternion, Vector3
from trimesh import Trimesh

from alfspy.core.convert.convert import pixel_to_world_coord, world_to_pixel_coord
from alfspy.core.rendering import (
    CtxShot, Renderer, Resolution, RenderResultMode, TextureData,
)
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.pyrrs import quaternion_from_eulers
from alfspy.render.data import BaseSettings, CameraPositioningMode
from alfspy.render.render import (
    make_camera, make_mgl_context, make_shot_loader,
    process_render_data, read_gltf, release_all,
)
from alfspy.orthografic_projection import (
    get_camera_for_frame, get_axis_aligned_bounding_box, to_yolo_format,
)

from .config import RenderSettings
from .corrections import FrameCorrection
from .labels import Label


def make_shot(ctx, image_path: str, meta: dict, correction, fovy_fallback: float) -> CtxShot:
    """Build a lazy CtxShot from a projection-ready pose entry.

    Replicates bambi.util.projection_util.create_shot so we don't pull in the
    heavy bambi_detection/torch dependency.
    """
    position = Vector3(meta["location"])
    rotation = [v % 360.0 for v in meta["rotation"]]
    if len(rotation) != 3:
        raise ValueError(f"Expected 3 rotation eulers, got {rotation}")
    quat = quaternion_from_eulers([np.deg2rad(v) for v in rotation], "zyx")

    fov = meta.get("fovy", fovy_fallback)
    if isinstance(fov, (list, tuple)):
        fov = fov[0] if fov and fov[0] is not None else fovy_fallback
    return CtxShot(ctx, image_path, position, quat, float(fov), 1, correction, lazy=True)


def project_to_image(points, width: int, height: int, camera) -> tuple:
    """World coordinates -> top-left-origin pixel coordinates.

    Same maths as ``alfspy.core.convert.world_to_pixel_coord`` but with the
    perspective divide done per point. The upstream helper divides an ``(N, 4)``
    array by an ``(N,)`` one, which only broadcasts when ``N == 4`` — so a label
    whose box has just three of its four corners landing on the DEM raises
    ``operands could not be broadcast together``. That is exactly the partial-hit
    case near the edge of the render, and losing those labels (or the whole
    frame) is not acceptable.

    For the orthographic render camera ``w`` is identically 1, so results match
    the upstream helper wherever the upstream helper works at all.
    """
    pts = np.asarray(points, dtype=float).reshape((-1, 3))
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    # np.asarray: pyrr returns a Matrix44 (not an ndarray) whenever a product
    # happens to come out 4x4, and Matrix44 has no elementwise division.
    view = np.asarray(camera.get_view(), dtype=float)
    proj = np.asarray(camera.get_proj(), dtype=float)
    ndc = np.asarray((homo @ view) @ proj, dtype=float)
    ndc = ndc / ndc[:, 3:4]
    xs = (ndc[:, 0] + 1.0) * width / 2.0
    ys = height - (ndc[:, 1] + 1.0) / 2.0 * height
    # Match the upstream rounding so labels stay comparable with earlier runs.
    return np.floor(xs + 0.5).astype(int), np.floor(ys + 0.5).astype(int)


class FlightRenderer:
    """Holds the per-flight DEM mesh + GL context; renders frames and labels."""

    def __init__(self, dem_glb: str | Path, render: RenderSettings,
                 origin: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.render = render
        self.resolution = Resolution(render.render_width, render.render_height)
        # UTM (easting, northing, altitude) of the DEM mesh's local (0,0,0).
        # Projected label corners are in local mesh metres; adding this yields
        # global coordinates. Defaults to zero => local coords (no DEM origin).
        self.origin = np.asarray(origin, dtype=float)

        mesh_data, texture_data = read_gltf(str(dem_glb))
        if mesh_data is None:
            raise ValueError(f"Could not extract mesh from DEM {dem_glb}")
        # Build the ray-cast mesh BEFORE process_render_data mutates the arrays.
        self.tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)

        self.mesh_data, self.texture_data = process_render_data(mesh_data, texture_data)
        self.mesh_aabb = get_aabb(self.mesh_data.vertices)
        self.ctx = make_mgl_context()
        # Created lazily on the first frame and then reused across the whole
        # flight so the (large) DEM texture is uploaded to the GPU only once.
        self._renderer: Renderer | None = None

        self._base_settings = BaseSettings(
            count=1,
            initial_skip=0,
            add_background=False,
            camera_position_mode=CameraPositioningMode.FirstShot,
            fovy=render.fovy_fallback,
            aspect_ratio=render.aspect_ratio,
            orthogonal=True,
            ortho_size=(render.ortho_width, render.ortho_height),
            correction=None,
            resolution=self.resolution,
        )

    def render_frame(self, image_path: str, meta: dict, correction: FrameCorrection,
                     save_path: str | Path):
        """Render one orthographic projection to `save_path`.

        Returns the single-shot Camera used (needed for label projection).
        """
        settings = self._base_settings
        settings.correction = correction.transform

        shot = make_shot(self.ctx, image_path, meta, correction.transform,
                         self.render.fovy_fallback)
        single_shot_camera = make_camera(
            self.mesh_aabb, [shot], settings,
            rotation=Quaternion.from_matrix(
                shot.get_view().inverse @ shot.get_correction().inverse
            ),
        )
        if self._renderer is None:
            self._renderer = Renderer(self.resolution, self.ctx, single_shot_camera,
                                      self.mesh_data, self.texture_data)
        else:
            # Reuse the renderer (and its uploaded DEM texture); just retarget
            # the orthographic camera for this frame.
            self._renderer.camera = single_shot_camera
            self._renderer.apply_matrices()

        shot_loader = make_shot_loader([shot])
        self._renderer.project_shots(
            shot_loader,
            RenderResultMode.ShotOnly,
            mask=None,
            integral=False,
            save=True,
            release_shots=True,
            save_name_iter=iter([str(save_path)]),
        )
        return single_shot_camera

    def project_labels(self, poses: dict, frame_idx: int, labels: list[Label],
                       input_resolution: Resolution, correction: FrameCorrection,
                       single_shot_camera) -> tuple[list[tuple[int, str]],
                                                    list[tuple[int, int, list[float]]]]:
        """Project label corners onto the DEM in both local and UTM space.

        Returns ``(yolo_lines, utm_lines)`` where

        * ``yolo_lines`` is ``[(class_id, "cx cy w h"), ...]`` in the rendered
          orthographic image space (same as before).
        * ``utm_lines`` is ``[(class_id, track_id,
          [x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4]), ...]`` in global DEM world
          space: local mesh metres plus ``self.origin`` (UTM easting, northing,
          altitude). A label only contributes a UTM entry when all four input
          corners ray-cast onto the DEM, so corners stay aligned with the
          original TL/TR/BR/BL order.

        Mirrors ``alfspy.orthografic_projection.project_label`` but splits the
        ``pixel_to_world_coord`` / ``world_to_pixel_coord`` steps so we can
        capture the intermediate world-space hits.
        """
        camera = get_camera_for_frame(
            poses, frame_idx, correction.rotation_eulers, correction.translation,
        )
        yolo_out: list[tuple[int, str]] = []
        utm_out: list[tuple[int, int, list[float]]] = []
        for label in labels:
            xs = [int(float(label.corners[i])) for i in range(0, len(label.corners), 2)]
            ys = [int(float(label.corners[i])) for i in range(1, len(label.corners), 2)]

            world = pixel_to_world_coord(
                xs, ys, input_resolution.width, input_resolution.height,
                self.tri_mesh, camera, include_misses=True,
            )

            if len(world) == 4 and all(w is not None for w in world):
                flat: list[float] = []
                for w in world:
                    gx, gy, gz = (np.asarray(w, dtype=float) + self.origin)
                    flat.extend([float(gx), float(gy), float(gz)])
                utm_out.append((label.class_id, label.track_id, flat))

            hits = [w for w in world if w is not None]
            if not hits:
                continue
            np_poses = project_to_image(
                hits, self.resolution.width, self.resolution.height, single_shot_camera,
            )
            poly = [np.array(np_poses).T.reshape((-1, 1, 2))]
            aabb = get_axis_aligned_bounding_box(poly)
            yolo = to_yolo_format(aabb, self.resolution.width, self.resolution.height)
            if yolo is not None:
                yolo_out.append((label.class_id, yolo))
        return yolo_out, utm_out

    def close(self):
        release_all(self._renderer)
        release_all(self.ctx)


def read_dem_origin(dem_json: str | Path | None) -> tuple[float, float, float]:
    """UTM (easting, northing, altitude) of the DEM mesh's local origin.

    Read from the ``<id>_dem.json`` written alongside the DEM (.glb) at download
    time. Returns ``(0, 0, 0)`` when the file is missing or has no ``origin``,
    in which case projected coordinates stay in local mesh space.
    """
    if dem_json is None:
        return (0.0, 0.0, 0.0)
    try:
        with open(dem_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError:
        return (0.0, 0.0, 0.0)
    origin = data.get("origin")
    if not origin or len(origin) < 3:
        return (0.0, 0.0, 0.0)
    return (float(origin[0]), float(origin[1]), float(origin[2]))


def read_input_resolution(mask_path: str | Path | None,
                          fallback_image: str | Path | None) -> Resolution:
    """Resolution of the original frame space (label coords). Prefer the mask,
    fall back to reading a frame image."""
    for candidate in (mask_path, fallback_image):
        if candidate is None:
            continue
        img = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
        if img is not None:
            return Resolution(img.shape[1], img.shape[0])
    raise ValueError(f"Could not determine input resolution from {mask_path} / {fallback_image}")
