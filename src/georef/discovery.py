"""Resolve and validate the per-flight input assets needed for projection."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config

# Mask filename suffix per modality (in correction_data / flight folder).
_MASK_SUFFIX = {"rgb": "mask_w", "thermal": "mask_t"}


@dataclass
class ModalityAssets:
    modality: str
    frames_dir: Path           # .../frames/<modality>
    mask: Path | None          # <id>_mask_w/t.png  (None if missing)
    mot: Path | None           # MOT/<modality>/<id>_accepted_<modality>_mot.txt
    frame_files: list[Path] = field(default_factory=list)
    # ALFS only: cache of neighbour frames decoded from the flight video.
    neighbors_dir: Path | None = None

    @property
    def has_frames(self) -> bool:
        return bool(self.frame_files)


@dataclass
class FlightAssets:
    flight_id: str
    flight_dir: Path
    poses: Path                # correction_data/<id>_matched_poses.json
    dem_glb: Path
    dem_json: Path
    correction: Path           # chosen per config.correction_source
    modalities: dict[str, ModalityAssets]
    output_dir: Path
    video: Path | None = None  # ALFS only: <videos_root>/<id>_matched_processed.mp4

    # Problems found during discovery (empty => ready to process).
    missing_essential: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.missing_essential


def list_available_flight_ids(cfg: Config) -> list[str]:
    """Numeric subfolders of flights_root that actually contain a frames/ dir."""
    ids: list[str] = []
    for child in cfg.paths.flights_root.iterdir():
        if child.is_dir() and child.name.isdigit() and (child / "frames").is_dir():
            ids.append(child.name)
    return sorted(ids, key=int)


def _frame_files(frames_dir: Path) -> list[Path]:
    if not frames_dir.is_dir():
        return []
    files = [p for p in frames_dir.iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.stem.isdigit()]
    return sorted(files, key=lambda p: int(p.stem))


def resolve_flight(cfg: Config, flight_id: str) -> FlightAssets:
    """Resolve every asset path for one flight and record what is missing."""
    cd = cfg.paths.correction_data
    flight_dir = cfg.paths.flights_root / flight_id

    poses = cd / f"{flight_id}_matched_poses.json"
    dem_glb = cd / f"{flight_id}_dem.glb"
    dem_json = cd / f"{flight_id}_dem.json"

    if cfg.correction_source == "flights":
        correction = flight_dir / f"{flight_id}_correction.json"
    else:
        correction = cd / f"{flight_id}_correction.json"

    modalities: dict[str, ModalityAssets] = {}
    for modality in cfg.modalities:
        frames_dir = flight_dir / "frames" / modality
        mask_name = f"{flight_id}_{_MASK_SUFFIX[modality]}.png"
        mask = cd / mask_name
        if not mask.exists():
            alt = flight_dir / mask_name        # fall back to flight folder
            mask = alt if alt.exists() else None
        mot = cfg.paths.mot_root / modality / f"{flight_id}_accepted_{modality}_mot.txt"
        nb_root = cfg.paths.neighbor_frames_root
        modalities[modality] = ModalityAssets(
            modality=modality,
            frames_dir=frames_dir,
            mask=mask,
            mot=mot if mot.exists() else None,
            frame_files=_frame_files(frames_dir),
            neighbors_dir=(nb_root / flight_id / modality) if nb_root else None,
        )

    video = None
    if cfg.paths.videos_root is not None:
        video = cfg.paths.videos_root / f"{flight_id}_matched_processed.mp4"

    assets = FlightAssets(
        flight_id=flight_id,
        flight_dir=flight_dir,
        poses=poses,
        dem_glb=dem_glb,
        dem_json=dem_json,
        correction=correction,
        modalities=modalities,
        output_dir=cfg.active_output_root / flight_id,
        video=video,
    )

    # --- essentials -------------------------------------------------------
    if not poses.exists():
        assets.missing_essential.append(f"poses: {poses}")
    if not dem_glb.exists():
        # Only essential when download is disabled.
        if cfg.allow_dem_download:
            assets.warnings.append(f"DEM missing, will attempt download: {dem_glb}")
        else:
            assets.missing_essential.append(f"dem: {dem_glb}")
    if not correction.exists():
        assets.missing_essential.append(f"correction: {correction}")
    if not dem_json.exists():
        assets.warnings.append(
            f"DEM origin missing ({dem_json}); UTM labels will be in local "
            f"mesh coordinates")
    if not any(m.has_frames for m in modalities.values()):
        assets.missing_essential.append(
            f"frames: none found under {flight_dir / 'frames'} for {cfg.modalities}"
        )

    # --- ALFS essentials --------------------------------------------------
    if cfg.mode == "alfs":
        # The video is only ever read to *decode* neighbour frames into the
        # cache. When the cache is already populated (extract_neighbors off)
        # the flight is perfectly renderable without it, so requiring it here
        # would reject every flight on a machine that holds the frames but not
        # the source videos.
        if cfg.extract_neighbors and (video is None or not video.exists()):
            assets.missing_essential.append(f"video: {video}")
        # ALFS renders only labelled central frames, so a flight with no MOT
        # file for any requested modality has nothing to contribute.
        if not any(ma.mot for ma in modalities.values()):
            assets.missing_essential.append(
                f"MOT labels: none found under {cfg.paths.mot_root} for {cfg.modalities}"
            )

    # --- warnings ---------------------------------------------------------
    for modality, ma in modalities.items():
        if not ma.has_frames:
            assets.warnings.append(f"[{modality}] no frames in {ma.frames_dir}")
        if ma.mask is None:
            assets.warnings.append(f"[{modality}] no mask found")
        if ma.mot is None:
            assets.warnings.append(f"[{modality}] no MOT labels (images only)")

    return assets
