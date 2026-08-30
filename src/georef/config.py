"""Typed configuration loaded from config.yaml (+ optional config.local.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_MODALITIES = ("rgb", "thermal")
VALID_CORRECTION_SOURCES = ("flights", "correction_data")
VALID_MODES = ("ortho", "alfs")
VALID_LABEL_MODES = ("merged", "central", "both")


@dataclass
class Paths:
    flights_root: Path
    correction_data: Path
    mot_root: Path
    output_root: Path
    # ALFS-only. `videos_root` holds <id>_matched_processed.mp4 (left half
    # thermal, right half rgb); neighbour frames extracted from it are cached
    # under `neighbor_frames_root`/<id>/<modality>/NNNNNN.jpg.
    videos_root: Path | None = None
    neighbor_frames_root: Path | None = None
    alfs_output_root: Path | None = None


@dataclass
class RenderSettings:
    ortho_width: int = 70
    ortho_height: int = 70
    render_width: int = 2048
    render_height: int = 2048
    fovy_fallback: float = 50.0
    aspect_ratio: float = 1.0


@dataclass
class AlfsSettings:
    """Airborne light field sampling (synthetic aperture) parameters.

    The aperture is defined in *video* frame indices around each central frame:
    ``neighbors_before``/``neighbors_after`` give its extent and ``stride`` the
    spacing between contributing shots. At ~30 fps and ~3 m/s ground speed one
    video frame is ~0.10 m, so the default (+-45 frames, stride 3) is 31 shots
    spanning ~9 m at ~0.3 m spacing.
    """
    neighbors_before: int = 45
    neighbors_after: int = 45
    stride: int = 3
    # Minimum number of overlapping shots a pixel needs to survive integration.
    alpha_threshold: float = 2.0
    auto_contrast: bool = True
    use_mask: bool = True          # apply the modality mask to each shot texture
    label_mode: str = "both"       # merged | central | both

    def neighbor_offsets(self) -> list[int]:
        """Frame-index offsets of the contributing shots, central (0) included."""
        before = [-o for o in range(self.stride, self.neighbors_before + 1, self.stride)]
        after = [o for o in range(self.stride, self.neighbors_after + 1, self.stride)]
        return sorted(before) + [0] + after

    @property
    def shot_count(self) -> int:
        return len(self.neighbor_offsets())


@dataclass
class Config:
    paths: Paths
    render: RenderSettings = field(default_factory=RenderSettings)
    alfs: AlfsSettings = field(default_factory=AlfsSettings)
    correction_source: str = "flights"
    mode: str = "ortho"          # ortho | alfs
    modalities: list[str] = field(default_factory=lambda: list(VALID_MODALITIES))
    overwrite: bool = False
    allow_dem_download: bool = False
    labeled_only: bool = False   # process only frames that have MOT labels
    # ALFS: decode any missing neighbour frames from the flight video before
    # rendering. Turn off when the neighbour cache is known to be complete.
    extract_neighbors: bool = True

    @property
    def active_output_root(self) -> Path:
        """Where this run writes. ALFS keeps its own root so the two modalities
        of renderings never overwrite each other."""
        if self.mode == "alfs" and self.paths.alfs_output_root is not None:
            return self.paths.alfs_output_root
        return self.paths.output_root

    def validate(self) -> None:
        if self.correction_source not in VALID_CORRECTION_SOURCES:
            raise ValueError(
                f"correction_source must be one of {VALID_CORRECTION_SOURCES}, "
                f"got {self.correction_source!r}"
            )
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        bad = [m for m in self.modalities if m not in VALID_MODALITIES]
        if bad:
            raise ValueError(f"Unknown modalities {bad}; valid: {VALID_MODALITIES}")
        for name in ("flights_root", "correction_data", "mot_root"):
            p = getattr(self.paths, name)
            if not p.exists():
                raise FileNotFoundError(f"Configured path {name}={p} does not exist")

        if self.mode != "alfs":
            return
        if self.alfs.label_mode not in VALID_LABEL_MODES:
            raise ValueError(f"alfs.label_mode must be one of {VALID_LABEL_MODES}, "
                             f"got {self.alfs.label_mode!r}")
        if self.alfs.stride < 1:
            raise ValueError(f"alfs.stride must be >= 1, got {self.alfs.stride}")
        if min(self.alfs.neighbors_before, self.alfs.neighbors_after) < 0:
            raise ValueError("alfs.neighbors_before/after must be >= 0")
        for name in ("neighbor_frames_root", "alfs_output_root"):
            if getattr(self.paths, name) is None:
                raise ValueError(f"mode=alfs requires paths.{name} in the config")
        # videos_root is read only to *decode* neighbour frames into the cache.
        # With extract_neighbors off the cache is already populated and the
        # videos need not be present at all — requiring them would block every
        # ALFS job on a machine that holds the frames but not the source videos.
        if self.extract_neighbors:
            if self.paths.videos_root is None:
                raise ValueError("extract_neighbors=true requires paths.videos_root")
            if not self.paths.videos_root.exists():
                raise FileNotFoundError(
                    f"Configured path videos_root={self.paths.videos_root} does not exist")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(config_path: str | Path) -> Config:
    """Load config.yaml; merge an adjacent config.local.yaml on top if present."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    local = config_path.with_name("config.local.yaml")
    if local.exists():
        with open(local, "r", encoding="utf-8") as f:
            data = _deep_merge(data, yaml.safe_load(f) or {})

    p = data["paths"]

    def _opt(key: str) -> Path | None:
        return Path(p[key]) if p.get(key) else None

    paths = Paths(
        flights_root=Path(p["flights_root"]),
        correction_data=Path(p["correction_data"]),
        mot_root=Path(p["mot_root"]),
        output_root=Path(p["output_root"]),
        videos_root=_opt("videos_root"),
        neighbor_frames_root=_opt("neighbor_frames_root"),
        alfs_output_root=_opt("alfs_output_root"),
    )
    render = RenderSettings(**(data.get("render") or {}))
    alfs = AlfsSettings(**(data.get("alfs") or {}))
    cfg = Config(
        paths=paths,
        render=render,
        alfs=alfs,
        correction_source=data.get("correction_source", "flights"),
        mode=data.get("mode", "ortho"),
        modalities=list(data.get("modalities", list(VALID_MODALITIES))),
        overwrite=bool(data.get("overwrite", False)),
        allow_dem_download=bool(data.get("allow_dem_download", False)),
        labeled_only=bool(data.get("labeled_only", False)),
        extract_neighbors=bool(data.get("extract_neighbors", True)),
    )
    return cfg
