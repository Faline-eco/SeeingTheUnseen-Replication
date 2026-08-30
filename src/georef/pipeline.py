"""Flight-by-flight orchestration: render every sampled frame + project labels."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .alfs import ApertureShot, AlfsRenderer
from .config import Config
from .corrections import CorrectionProvider
from .discovery import FlightAssets, ModalityAssets, resolve_flight
from .labels import load_mot
from .projection import FlightRenderer, read_dem_origin, read_input_resolution
from .video_frames import (
    ExtractionResult, aperture_indices, central_frame_indices,
    extract_flight_neighbors, resolve_frame_path,
)

log = logging.getLogger("georef")


@dataclass
class ModalityResult:
    modality: str
    frames_total: int = 0
    rendered: int = 0
    skipped_existing: int = 0
    skipped_no_label: int = 0
    failed: int = 0
    label_files: int = 0
    labels_written: int = 0
    utm_label_files: int = 0
    utm_labels_written: int = 0
    # ALFS only
    skipped_no_frames: int = 0        # central frame image missing
    shots_used_mean: float = 0.0      # mean aperture size actually rendered
    coverage_mean: float = 0.0        # mean fraction of pixels above alpha_threshold
    merged_labels_written: int = 0


@dataclass
class FlightResult:
    flight_id: str
    status: str = "ok"          # ok | skipped | failed
    reason: str = ""
    seconds: float = 0.0
    dem_origin: tuple[float, float, float] | None = None  # UTM origin of the DEM
    modalities: dict[str, ModalityResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    extraction: ExtractionResult | None = None   # ALFS neighbour extraction


def _process_modality(renderer: FlightRenderer, cfg: Config, poses: dict,
                      corr: CorrectionProvider, ma: ModalityAssets,
                      out_dir: Path) -> ModalityResult:
    res = ModalityResult(modality=ma.modality, frames_total=len(ma.frame_files))
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_by_frame = load_mot(ma.mot) if ma.mot else {}
    input_resolution = read_input_resolution(
        ma.mask, ma.frame_files[0] if ma.frame_files else None
    )
    n_images = len(poses["images"])

    for frame_path in ma.frame_files:
        idx = int(frame_path.stem)
        img_out = out_dir / f"{frame_path.stem}.png"
        txt_out = out_dir / f"{frame_path.stem}.txt"
        utm_out = out_dir / f"{frame_path.stem}_utm.txt"
        frame_labels = labels_by_frame.get(idx, [])

        if cfg.labeled_only and not frame_labels:
            res.skipped_no_label += 1
            continue

        if not cfg.overwrite and img_out.exists() and (
            not frame_labels or (txt_out.exists() and utm_out.exists())
        ):
            res.skipped_existing += 1
            continue

        if idx >= n_images:
            res.failed += 1
            log.warning("[%s/%s] frame %d has no pose entry (poses=%d), skipping",
                        ma.modality, frame_path.name, idx, n_images)
            continue

        meta = poses["images"][idx]
        frame_corr = corr.for_frame(idx)
        try:
            cam = renderer.render_frame(str(frame_path), meta, frame_corr, img_out)
            res.rendered += 1
            if frame_labels:
                yolo, utm = renderer.project_labels(
                    poses, idx, frame_labels, input_resolution, frame_corr, cam,
                )
                with open(txt_out, "w", encoding="utf-8") as f:
                    for class_id, line in yolo:
                        f.write(f"{class_id} {line}\n")
                res.label_files += 1
                res.labels_written += len(yolo)
                # class track x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 (global UTM
                # easting/northing/altitude, mm precision).
                with open(utm_out, "w", encoding="utf-8") as f:
                    for class_id, track_id, coords in utm:
                        coord_str = " ".join(f"{c:.3f}" for c in coords)
                        f.write(f"{class_id} {track_id} {coord_str}\n")
                res.utm_label_files += 1
                res.utm_labels_written += len(utm)
        except Exception:
            res.failed += 1
            log.exception("[%s/%s] render/projection failed", ma.modality, frame_path.name)

    return res


def _write_yolo(path: Path, rows: list[tuple[int, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for class_id, line in rows:
            f.write(f"{class_id} {line}\n")


def _process_modality_alfs(renderer: AlfsRenderer, cfg: Config, poses: dict,
                           corr: CorrectionProvider, ma: ModalityAssets,
                           out_dir: Path) -> ModalityResult:
    """Render one integral per labelled central frame of this modality."""
    n_images = len(poses["images"])
    centrals = central_frame_indices(ma, n_images)
    res = ModalityResult(modality=ma.modality, frames_total=len(centrals))
    if not centrals:
        return res
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_by_frame = load_mot(ma.mot) if ma.mot else {}
    input_resolution = read_input_resolution(
        ma.mask, ma.frame_files[0] if ma.frame_files else None
    )
    renderer.set_modality(ma.mask, ma.neighbors_dir)

    want_merged = cfg.alfs.label_mode in ("merged", "both")
    want_central = cfg.alfs.label_mode in ("central", "both")
    shots_used: list[int] = []
    coverages: list[float] = []

    for idx in centrals:
        img_out = out_dir / f"{idx:06d}.png"
        # With label_mode=both the merged boxes are the primary label file and
        # the central-frame-only boxes sit alongside for ablation.
        merged_out = out_dir / f"{idx:06d}.txt"
        central_out = out_dir / (f"{idx:06d}_central.txt" if want_merged
                                 else f"{idx:06d}.txt")
        utm_out = out_dir / f"{idx:06d}_utm.txt"

        expected = [img_out, utm_out]
        if want_merged:
            expected.append(merged_out)
        if want_central:
            expected.append(central_out)
        if not cfg.overwrite and all(p.exists() for p in expected):
            res.skipped_existing += 1
            continue

        aperture: list[ApertureShot] = []
        for frame_idx in aperture_indices(idx, cfg.alfs, n_images):
            image = resolve_frame_path(ma, frame_idx)
            if image is None:
                continue
            aperture.append(ApertureShot(
                frame_idx=frame_idx,
                image=image,
                meta=poses["images"][frame_idx],
                correction=corr.for_frame(frame_idx, central_frame_idx=idx),
            ))
        if not any(a.frame_idx == idx for a in aperture):
            # No central frame image => no aperture to anchor the camera on.
            res.skipped_no_frames += 1
            log.warning("[%s] frame %d: central frame image missing, skipping",
                        ma.modality, idx)
            continue

        try:
            camera, coverage, n_shots = renderer.render_aperture(aperture, idx, img_out)
            res.rendered += 1
            shots_used.append(n_shots)
            coverages.append(coverage)

            merged, central, utm = renderer.project_aperture_labels(
                poses, idx, aperture, labels_by_frame, input_resolution, corr, camera,
            )
            if want_merged:
                _write_yolo(merged_out, merged)
                res.merged_labels_written += len(merged)
            if want_central:
                _write_yolo(central_out, central)
            res.label_files += 1
            res.labels_written += len(merged if want_merged else central)

            with open(utm_out, "w", encoding="utf-8") as f:
                for class_id, track_id, coords in utm:
                    coord_str = " ".join(f"{c:.3f}" for c in coords)
                    f.write(f"{class_id} {track_id} {coord_str}\n")
            res.utm_label_files += 1
            res.utm_labels_written += len(utm)
        except Exception:
            res.failed += 1
            log.exception("[%s] frame %d: ALFS render/projection failed",
                          ma.modality, idx)

    if shots_used:
        res.shots_used_mean = round(sum(shots_used) / len(shots_used), 2)
        res.coverage_mean = round(sum(coverages) / len(coverages), 4)
    return res


def process_flight(cfg: Config, flight_id: str) -> FlightResult:
    """Render all frames + labels for one flight. Never raises; reports status."""
    t0 = time.time()
    result = FlightResult(flight_id=flight_id)
    assets = resolve_flight(cfg, flight_id)
    result.warnings = list(assets.warnings)

    if not assets.ready:
        result.status = "skipped"
        result.reason = "; ".join(assets.missing_essential)
        log.warning("Flight %s SKIPPED: %s", flight_id, result.reason)
        return result

    renderer = None
    try:
        corr = CorrectionProvider(assets.correction)
        with open(assets.poses, "r", encoding="utf-8") as f:
            poses = json.load(f)

        origin = read_dem_origin(assets.dem_json)
        result.dem_origin = origin
        if origin == (0.0, 0.0, 0.0):
            log.warning("Flight %s: no DEM origin (%s); UTM labels will be in "
                        "local mesh coordinates", flight_id, assets.dem_json)
        if cfg.mode == "alfs":
            if cfg.extract_neighbors:
                extraction = extract_flight_neighbors(
                    cfg, assets, len(poses["images"]), overwrite=False
                )
                result.extraction = extraction
                if extraction.status != "ok":
                    result.status = "skipped"
                    result.reason = f"neighbour extraction: {extraction.reason}"
                    log.warning("Flight %s SKIPPED: %s", flight_id, result.reason)
                    return result
            renderer = AlfsRenderer(assets.dem_glb, cfg.render, cfg.alfs, origin=origin)
            for modality, ma in assets.modalities.items():
                if ma.mot is None:
                    log.info("Flight %s [%s]: no MOT labels, nothing to render",
                             flight_id, modality)
                    continue
                result.modalities[modality] = _process_modality_alfs(
                    renderer, cfg, poses, corr, ma, assets.output_dir / modality
                )
                r = result.modalities[modality]
                log.info("Flight %s [%s]: %d centrals -> %d rendered, %d skipped, "
                         "%d failed (mean %.1f shots, %.0f%% coverage)",
                         flight_id, modality, r.frames_total, r.rendered,
                         r.skipped_existing, r.failed, r.shots_used_mean,
                         r.coverage_mean * 100)
        else:
            renderer = FlightRenderer(assets.dem_glb, cfg.render, origin=origin)
            for modality, ma in assets.modalities.items():
                if not ma.has_frames:
                    continue
                log.info("Flight %s [%s]: %d frames%s", flight_id, modality,
                         len(ma.frame_files), "" if ma.mot else " (no labels)")
                result.modalities[modality] = _process_modality(
                    renderer, cfg, poses, corr, ma, assets.output_dir / modality
                )
    except Exception as e:
        result.status = "failed"
        result.reason = repr(e)
        log.exception("Flight %s FAILED", flight_id)
    finally:
        if renderer is not None:
            renderer.close()

    result.seconds = round(time.time() - t0, 1)
    if result.status == "ok":
        _write_manifest(cfg, assets, result)
    return result


def _write_manifest(cfg: Config, assets: FlightAssets, result: FlightResult) -> None:
    manifest = {
        "flight_id": assets.flight_id,
        "mode": cfg.mode,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": {
            "poses": str(assets.poses),
            "dem_glb": str(assets.dem_glb),
            "dem_json": str(assets.dem_json),
            "dem_origin": list(result.dem_origin) if result.dem_origin else None,
            "correction": str(assets.correction),
            "correction_source": cfg.correction_source,
            "modalities": {
                m: {"frames_dir": str(ma.frames_dir),
                    "mask": str(ma.mask) if ma.mask else None,
                    "mot": str(ma.mot) if ma.mot else None}
                for m, ma in assets.modalities.items()
            },
        },
        "render": vars(cfg.render),
        "results": {
            m: vars(r) for m, r in result.modalities.items()
        },
        "warnings": result.warnings,
        "seconds": result.seconds,
    }
    if cfg.mode == "alfs":
        manifest["alfs"] = dict(vars(cfg.alfs),
                                neighbor_offsets=cfg.alfs.neighbor_offsets())
        manifest["inputs"]["video"] = str(assets.video)
        manifest["inputs"]["neighbor_frames"] = {
            m: str(ma.neighbors_dir) for m, ma in assets.modalities.items()
        }
        if result.extraction is not None:
            manifest["neighbor_extraction"] = vars(result.extraction)
    # Manifests are per-mode: ortho and ALFS write to different output roots.
    assets.output_dir.mkdir(parents=True, exist_ok=True)
    with open(assets.output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def extract_neighbors(cfg: Config, flight_ids: list[str]) -> bool:
    """Decode neighbour frames for every flight without rendering anything.

    Useful to front-load the (I/O bound) video decoding before the GPU run, or
    to top up the cache after widening the aperture. Returns True if all
    flights extracted cleanly.
    """
    ok = True
    totals = {"written": 0, "needed": 0}
    for i, fid in enumerate(flight_ids, 1):
        assets = resolve_flight(cfg, fid)
        if not assets.ready:
            log.warning("Flight %s (%d/%d) SKIPPED: %s", fid, i, len(flight_ids),
                        "; ".join(assets.missing_essential))
            continue
        with open(assets.poses, "r", encoding="utf-8") as f:
            n_images = len(json.load(f)["images"])
        log.info("=== Flight %s (%d/%d) neighbour extraction ===", fid, i, len(flight_ids))
        res = extract_flight_neighbors(cfg, assets, n_images, overwrite=cfg.overwrite)
        totals["written"] += sum(res.written.values())
        totals["needed"] += sum(res.needed.values())
        if res.status != "ok":
            ok = False
            log.error("Flight %s extraction FAILED: %s", fid, res.reason)
            for problem in res.problems[1:]:
                log.error("Flight %s: %s", fid, problem)
    log.info("-------- EXTRACTION SUMMARY --------")
    log.info("%d frames decoded, %d aperture frames required in total",
             totals["written"], totals["needed"])
    return ok


def run(cfg: Config, flight_ids: list[str]) -> list[FlightResult]:
    """Process a list of flights sequentially, returning per-flight results."""
    results: list[FlightResult] = []
    for i, fid in enumerate(flight_ids, 1):
        log.info("=== Flight %s (%d/%d) ===", fid, i, len(flight_ids))
        results.append(process_flight(cfg, fid))
        print(f"Saved to {cfg.active_output_root}/{fid}")
    _log_summary(results)
    return results


def _log_summary(results: list[FlightResult]) -> None:
    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]
    log.info("-------- SUMMARY --------")
    log.info("flights: %d ok, %d skipped, %d failed", len(ok), len(skipped), len(failed))
    for r in ok:
        rendered = sum(m.rendered for m in r.modalities.values())
        labels = sum(m.label_files for m in r.modalities.values())
        log.info("  ok      %-5s %4d imgs, %3d label files, %ss",
                 r.flight_id, rendered, labels, r.seconds)
    for r in skipped:
        log.info("  skipped %-5s %s", r.flight_id, r.reason)
    for r in failed:
        log.info("  FAILED  %-5s %s", r.flight_id, r.reason)
