"""Flight correction loading, including per-frame-range fine corrections.

A correction JSON looks like:

    {
      "rotation":    {"x":0, "y":0, "z":-0.2057},
      "translation": {"x":0, "y":0, "z":-1.5},
      "fine_corrections": [
        {"start_frame": 7133, "end_frame": 11836,
         "rotation": {...}, "translation": {...}}
      ]
    }

`fine_corrections` is optional (the correction_data2 variant omits it).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyrr import Quaternion, Vector3

from alfspy.core.geo.transform import Transform


def _vec(d: dict, default=(0.0, 0.0, 0.0)) -> Vector3:
    return Vector3([d.get("x", default[0]), d.get("y", default[1]),
                    d.get("z", default[2])], dtype="f4")


@dataclass
class FrameCorrection:
    """Everything the renderer + label projection need for a single frame."""
    transform: Transform        # translation + rotation quaternion
    rotation_eulers: Vector3    # euler form (radians? -> see note) for get_camera_for_frame
    translation: Vector3


class CorrectionProvider:
    """Resolves the correction applicable to a given frame index."""

    def __init__(self, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._default = self._build(data)
        self._ranges: list[tuple[int, int, FrameCorrection]] = []
        for fc in data.get("fine_corrections", []) or []:
            self._ranges.append(
                (int(fc["start_frame"]), int(fc["end_frame"]), self._build(fc))
            )

    @staticmethod
    def _build(d: dict) -> FrameCorrection:
        translation = _vec(d.get("translation", {}))
        eulers = _vec(d.get("rotation", {}))
        # Note: matches notebook _load_correction -> Quaternion.from_eulers(eulers)
        # and get_camera_for_frame which consumes the raw euler vector.
        transform = Transform(translation, Quaternion.from_eulers(eulers))
        return FrameCorrection(transform=transform, rotation_eulers=eulers,
                               translation=translation)

    def for_frame(self, frame_idx: int,
                  central_frame_idx: int | None = None) -> FrameCorrection:
        """Correction for `frame_idx`.

        `central_frame_idx` is used by ALFS: a neighbouring frame that falls
        outside every fine-correction range inherits the correction of the
        integral's central frame rather than dropping to the flight default, so
        all shots of one aperture stay in the same corrected frame of reference.
        Mirrors ``alfspy.orthografic_projection.get_frame_correction``.
        """
        central: FrameCorrection | None = None
        for start, end, corr in self._ranges:
            if start <= frame_idx <= end:
                return corr
            if central_frame_idx is not None and start <= central_frame_idx <= end:
                central = corr
        return central if central is not None else self._default
