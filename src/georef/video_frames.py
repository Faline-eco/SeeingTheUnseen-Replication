"""Neighbour-frame extraction from the processed flight videos.

The sampled dataset under ``flights_root`` only keeps every 10th video frame
(~3 fps), which is far too sparse to form a synthetic aperture. ALFS therefore
needs additional frames pulled straight out of

    <videos_root>/<id>_matched_processed.mp4

Those videos are 2048x1024 side-by-side: **left half = thermal, right half =
rgb**, each 1024x1024, and the video frame index is exactly the index into the
pose ``images`` array. Extracted frames are cached as

    <neighbor_frames_root>/<id>/<modality>/NNNNNN.jpg

Frames that already exist in the sampled ``frames/<modality>`` folder are not
re-extracted; :func:`resolve_frame_path` looks in both places.

Decoding is a single sequential pass using ``grab()`` for frames we do not want
and ``retrieve()`` only for those we do (~900 fps grab-only on the reference
machine, i.e. ~15 s for a 12k-frame video). Seeking is deliberately avoided so
frame indices cannot drift away from pose indices.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import AlfsSettings, Config
from .discovery import FlightAssets, ModalityAssets
from .labels import load_mot

log = logging.getLogger("georef")

# Column split of the side-by-side video, as fractions of the full width.
_MODALITY_HALF = {"thermal": 0, "rgb": 1}

# The sampled dataset keeps every Nth video frame; central frames must sit on
# that grid so ALFS and the orthographic renders share identical centrals.
SAMPLING_STEP = 10

JPEG_QUALITY = 95

# Threads encoding and writing extracted frames. The output share is
# latency-bound per file, so concurrency matters far more than CPU here.
WRITE_WORKERS = 16
WRITE_QUEUE_LIMIT = 256


@dataclass
class ExtractionResult:
    flight_id: str
    status: str = "ok"            # ok | skipped | failed
    reason: str = ""
    centrals: dict[str, int] = None       # modality -> number of central frames
    needed: dict[str, int] = None         # modality -> neighbour frames required
    written: dict[str, int] = None        # modality -> newly decoded frames
    existing: dict[str, int] = None       # modality -> already on disk
    seconds: float = 0.0
    # Video/pose index alignment probes (see _probe_indices).
    probes: int = 0
    probe_max_mad: float = 0.0
    problems: list[str] = None

    def __post_init__(self):
        for name in ("centrals", "needed", "written", "existing"):
            if getattr(self, name) is None:
                setattr(self, name, {})
        if self.problems is None:
            self.problems = []


def central_frame_indices(ma: ModalityAssets, n_poses: int) -> list[int]:
    """Labelled frames on the sampling grid — the ALFS central frames.

    These are exactly the frames the orthographic pipeline renders and labels
    for this modality, so both modalities of rendering cover the same set.
    """
    if ma.mot is None:
        return []
    labelled = load_mot(ma.mot)
    return sorted(
        idx for idx in labelled
        if idx % SAMPLING_STEP == 0 and 0 <= idx < n_poses
    )


def aperture_indices(central: int, alfs: AlfsSettings, n_poses: int) -> list[int]:
    """Frame indices contributing to one integral, clipped to available poses."""
    return [central + off for off in alfs.neighbor_offsets()
            if 0 <= central + off < n_poses]


def needed_neighbor_indices(centrals: list[int], alfs: AlfsSettings,
                            n_poses: int) -> set[int]:
    """Union of every aperture's frame indices over all central frames."""
    needed: set[int] = set()
    for c in centrals:
        needed.update(aperture_indices(c, alfs, n_poses))
    return needed


def resolve_frame_path(ma: ModalityAssets, idx: int) -> Path | None:
    """Image for one aperture frame: sampled frames first, neighbour cache next.

    Returns ``None`` when neither exists, so the caller can drop that shot from
    the aperture instead of failing the whole integral.
    """
    sampled = ma.frames_dir / f"{idx:06d}.jpg"
    if sampled.exists():
        return sampled
    if ma.neighbors_dir is not None:
        cached = ma.neighbors_dir / f"{idx:06d}.jpg"
        if cached.exists():
            return cached
    return None


def missing_indices(ma: ModalityAssets, needed: set[int]) -> set[int]:
    """Needed indices that are neither sampled nor already cached."""
    return {idx for idx in needed if resolve_frame_path(ma, idx) is None}


def extract_flight_neighbors(cfg: Config, assets: FlightAssets,
                             n_poses: int, overwrite: bool = False) -> ExtractionResult:
    """Decode every neighbour frame this flight's ALFS renders will need.

    Resumable: only indices missing from both the sampled frames folder and the
    neighbour cache are decoded, so re-running after a partial pass is cheap.
    """
    import time
    t0 = time.time()
    res = ExtractionResult(flight_id=assets.flight_id)

    if assets.video is None or not assets.video.exists():
        res.status = "failed"
        res.reason = f"video missing: {assets.video}"
        return res

    to_write: dict[int, list[tuple[str, Path]]] = {}
    for modality, ma in assets.modalities.items():
        if ma.neighbors_dir is None:
            continue
        centrals = central_frame_indices(ma, n_poses)
        needed = needed_neighbor_indices(centrals, cfg.alfs, n_poses)
        missing = needed if overwrite else missing_indices(ma, needed)
        res.centrals[modality] = len(centrals)
        res.needed[modality] = len(needed)
        res.existing[modality] = len(needed) - len(missing)
        res.written[modality] = 0
        if missing:
            ma.neighbors_dir.mkdir(parents=True, exist_ok=True)
        for idx in missing:
            to_write.setdefault(idx, []).append(
                (modality, ma.neighbors_dir / f"{idx:06d}.jpg")
            )

    if not to_write:
        res.seconds = round(time.time() - t0, 1)
        log.info("Flight %s: neighbour cache already complete (%s needed)",
                 assets.flight_id, res.needed)
        return res

    # Frames we decode purely to confirm the video is still in step with the
    # pose/label indices; they are compared in-pass and then discarded.
    probes = _probe_indices(assets, set(to_write))
    last_needed = max(max(to_write), max(probes) if probes else 0)
    cap = cv2.VideoCapture(str(assets.video))
    if not cap.isOpened():
        res.status = "failed"
        res.reason = f"could not open video {assets.video}"
        return res

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    def _encode(modality: str, out_path: Path, tile) -> tuple[str, bool]:
        return modality, bool(cv2.imwrite(str(out_path), tile, encode_params))

    # Decoding is fast (~900 fps grab-only) but each JPEG write is a separate
    # round trip to the network share, so serial writing caps out around 15
    # frames/s. Encoding and writing off-thread keeps the decoder saturated;
    # both cv2.imwrite and the socket wait release the GIL.
    pending: list = []
    try:
        with ThreadPoolExecutor(max_workers=WRITE_WORKERS) as pool:
            idx = 0
            truncated = False
            while idx <= last_needed:
                if idx not in to_write and idx not in probes:
                    if not cap.grab():
                        truncated = True
                        break
                    idx += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    truncated = True
                    break
                half = frame.shape[1] // 2
                for modality, out_path in to_write.get(idx, ()):
                    col = _MODALITY_HALF[modality]
                    # Copy: `frame` is reused by the next cap.read().
                    tile = frame[:, col * half:(col + 1) * half].copy()
                    pending.append(pool.submit(_encode, modality, out_path, tile))
                if idx in probes:
                    _check_probe(assets, idx, frame, res)
                # Bound the queue so a slow share cannot balloon memory.
                if len(pending) >= WRITE_QUEUE_LIMIT:
                    for future in pending[:WRITE_QUEUE_LIMIT // 2]:
                        modality, written = future.result()
                        res.written[modality] += written
                    del pending[:WRITE_QUEUE_LIMIT // 2]
                idx += 1

            for future in pending:
                modality, written = future.result()
                res.written[modality] += written
            pending.clear()

        if truncated:
            res.status = "failed"
            res.reason = f"video ended at frame {idx}, needed up to {last_needed}"
    finally:
        for future in pending:
            future.cancel()
        cap.release()

    if res.problems:
        res.status = "failed"
        res.reason = res.problems[0]

    res.seconds = round(time.time() - t0, 1)
    log.info("Flight %s: extracted %s neighbour frames (%s already present) in %ss "
             "[%d alignment probes, max MAD %.2f]",
             assets.flight_id, res.written, res.existing, res.seconds,
             res.probes, res.probe_max_mad)
    return res


# A decoded frame and the stored sampled frame differ only by JPEG round-trip
# (~0.3 mean absolute difference measured); anything above this is a real
# mismatch, not compression noise.
PROBE_MAD_LIMIT = 5.0


def _probe_indices(assets: FlightAssets, needed: set[int], count: int = 4) -> set[int]:
    """Sampling-grid indices to re-decode as an index-alignment check.

    Frames on the sampling grid already exist in ``frames/<modality>``, so they
    are never written to the neighbour cache and can serve as ground truth: if
    the video frame at index *i* does not match the stored frame *i*, then video
    indices have drifted out of step with pose and MOT indices and every
    integral built from them would be silently misregistered.

    Probes are spread across the range actually being decoded, so drift that
    only develops later in a video is still caught.
    """
    if not needed:
        return set()
    lo, hi = min(needed), max(needed)
    grid = [i for i in range(lo - lo % SAMPLING_STEP, hi + 1, SAMPLING_STEP)
            if any((ma.frames_dir / f"{i:06d}.jpg").exists()
                   for ma in assets.modalities.values())]
    if not grid:
        return set()
    step = max(1, len(grid) // count)
    return set(grid[::step][:count])


def _check_probe(assets: FlightAssets, idx: int, frame, res: ExtractionResult) -> None:
    """Compare one decoded video frame against the stored sampled frames."""
    import numpy as np

    half = frame.shape[1] // 2
    for modality, ma in assets.modalities.items():
        stored_path = ma.frames_dir / f"{idx:06d}.jpg"
        if not stored_path.exists():
            continue
        stored = cv2.imread(str(stored_path))
        col = _MODALITY_HALF[modality]
        decoded = frame[:, col * half:(col + 1) * half]
        if stored is None or stored.shape != decoded.shape:
            res.problems.append(f"[{modality}] frame {idx}: unreadable or shape mismatch")
            continue
        mad = float(np.abs(decoded.astype(np.int16) - stored.astype(np.int16)).mean())
        res.probes += 1
        res.probe_max_mad = max(res.probe_max_mad, round(mad, 3))
        if mad > PROBE_MAD_LIMIT:
            res.problems.append(
                f"[{modality}] frame {idx}: decoded video frame differs from the "
                f"sampled frame (MAD {mad:.2f}) — video/pose index drift?"
            )
