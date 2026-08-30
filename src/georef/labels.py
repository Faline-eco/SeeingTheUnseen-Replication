"""MOT label loading for the manually-filtered ("accepted") tracks.

Per-flight accepted MOT format (10 comma-separated columns):

    frame, track_id, bb_left, bb_top, bb_width, bb_height, conf(=1), -1, -1, -1

There is no species/class column, so projected YOLO labels are written with a
single configurable class id (default 0 = "animal"); the original track_id is
preserved on the in-memory label for traceability / future class mapping.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Single class for all accepted animal tracks (accepted MOT carries no species).
DEFAULT_CLASS_ID = 0


@dataclass
class Label:
    frame: int
    track_id: int
    class_id: int
    # Four corners TL,TR,BR,BL as a flat list [x1,y1,x2,y1,x2,y2,x1,y2] in
    # the ORIGINAL frame pixel space (matches alfspy.project_label input).
    corners: list[float]


def load_mot(path: str | Path, class_id: int = DEFAULT_CLASS_ID) -> dict[int, list[Label]]:
    """Parse an accepted MOT file into {frame_idx: [Label, ...]}."""
    by_frame: dict[int, list[Label]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no}: expected >=6 columns, got {len(parts)}")
            frame = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = (float(v) for v in parts[2:6])
            x2, y2 = x + w, y + h
            corners = [x, y, x2, y, x2, y2, x, y2]
            by_frame[frame].append(
                Label(frame=frame, track_id=track_id, class_id=class_id, corners=corners)
            )
    return dict(by_frame)
