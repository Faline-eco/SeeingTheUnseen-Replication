"""Command-line entry point for the geo-referencing pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    VALID_LABEL_MODES, VALID_MODALITIES, VALID_MODES, load_config,
)
from .discovery import list_available_flight_ids, resolve_flight

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


def _parse_flight_ids(value: str, cfg) -> list[str]:
    if value.strip().lower() == "all":
        return list_available_flight_ids(cfg)
    ids = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in ids if not v.isdigit()]
    if bad:
        raise SystemExit(f"Invalid flight ids (must be integers): {bad}")
    return ids


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="georef",
        description="Flight-by-flight orthographic geo-referencing of drone "
                    "frames and MOT labels.",
    )
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                   help=f"Path to config.yaml (default: {_DEFAULT_CONFIG})")
    p.add_argument("--flight-ids", required=True,
                   help="Comma-separated flight ids, or 'all'.")
    p.add_argument("--mode", choices=VALID_MODES, default=None,
                   help="ortho = one orthographic projection per frame; "
                        "alfs = airborne light field sampling, integrating a "
                        "synthetic aperture of neighbouring frames around each "
                        "labelled central frame. Default: from config.")
    p.add_argument("--modalities", default=None,
                   help=f"Comma-separated subset of {VALID_MODALITIES}. "
                        "Default: from config.")
    alfs = p.add_argument_group("alfs", "Only used with --mode alfs")
    alfs.add_argument("--aperture", default=None, metavar="BEFORE,AFTER,STRIDE",
                      help="Synthetic aperture in video frames, e.g. 45,45,3 "
                           "(the default: 31 shots over ~9 m).")
    alfs.add_argument("--alpha-threshold", type=float, default=None,
                      help="Minimum overlapping shots for a pixel to survive.")
    alfs.add_argument("--label-mode", choices=VALID_LABEL_MODES, default=None,
                      help="merged = one box per track over the whole aperture; "
                           "central = central frame only; both = write both.")
    alfs.add_argument("--no-extract-neighbors", dest="extract_neighbors",
                      action="store_false", default=None,
                      help="Assume the neighbour frame cache is complete and "
                           "skip video decoding.")
    alfs.add_argument("--extract-only", action="store_true",
                      help="Decode neighbour frames and exit without rendering.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-render frames whose outputs already exist.")
    p.add_argument("--allow-dem-download", action="store_true",
                   help="Attempt to download missing DEMs (slow).")
    frames = p.add_mutually_exclusive_group()
    frames.add_argument("--labeled-only", dest="labeled_only", action="store_true",
                        default=None,
                        help="Process only frames that have MOT labels.")
    frames.add_argument("--all-frames", dest="labeled_only", action="store_false",
                        help="Process every frame, labeled or not (default).")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve and report assets per flight; render nothing.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p


def _dry_run(cfg, flight_ids: list[str]) -> int:
    import json

    from .video_frames import (
        central_frame_indices, needed_neighbor_indices, missing_indices,
    )

    is_alfs = cfg.mode == "alfs"
    header = (f"Dry run [{cfg.mode}]: {len(flight_ids)} flight(s); "
              f"modalities={cfg.modalities}; correction_source={cfg.correction_source}")
    if is_alfs:
        header += (f"\naperture: -{cfg.alfs.neighbors_before}..+{cfg.alfs.neighbors_after} "
                   f"stride {cfg.alfs.stride} => {cfg.alfs.shot_count} shots/render; "
                   f"labels={cfg.alfs.label_mode}")
    else:
        header += f"; frames={'labeled-only' if cfg.labeled_only else 'all'}"
    print(header + "\n")

    n_ready = 0
    totals = {"centrals": 0, "needed": 0, "missing": 0}
    for fid in flight_ids:
        a = resolve_flight(cfg, fid)
        status = "READY" if a.ready else "SKIP "
        n_ready += a.ready
        print(f"[{status}] flight {fid}")
        n_images = 0
        if is_alfs and a.poses.exists():
            with open(a.poses, "r", encoding="utf-8") as f:
                n_images = len(json.load(f)["images"])
        for m, ma in a.modalities.items():
            extras = []
            if ma.mask is None:
                extras.append("no-mask")
            extras.append("labels" if ma.mot else "no-labels")
            if is_alfs:
                centrals = central_frame_indices(ma, n_images) if a.ready else []
                needed = needed_neighbor_indices(centrals, cfg.alfs, n_images)
                missing = missing_indices(ma, needed) if needed else set()
                totals["centrals"] += len(centrals)
                totals["needed"] += len(needed)
                totals["missing"] += len(missing)
                print(f"         {m:7s}: {len(centrals):5d} centrals, "
                      f"{len(needed):6d} aperture frames "
                      f"({len(missing):6d} to decode)  ({', '.join(extras)})")
            else:
                print(f"         {m:7s}: {len(ma.frame_files):4d} frames  "
                      f"({', '.join(extras)})")
        for miss in a.missing_essential:
            print(f"         MISSING: {miss}")

    print(f"\n{n_ready}/{len(flight_ids)} flights ready to process.")
    if is_alfs:
        print(f"ALFS totals: {totals['centrals']} integrals to render, "
              f"{totals['needed']} aperture frames referenced, "
              f"{totals['missing']} still to decode from video.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg = load_config(args.config)
    if args.mode:
        cfg.mode = args.mode
    if args.modalities:
        cfg.modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    if args.overwrite:
        cfg.overwrite = True
    if args.allow_dem_download:
        cfg.allow_dem_download = True
    if args.labeled_only is not None:   # CLI overrides config when given
        cfg.labeled_only = args.labeled_only
    if args.aperture:
        try:
            before, after, stride = (int(v) for v in args.aperture.split(","))
        except ValueError:
            raise SystemExit(f"--aperture expects BEFORE,AFTER,STRIDE, got {args.aperture!r}")
        cfg.alfs.neighbors_before = before
        cfg.alfs.neighbors_after = after
        cfg.alfs.stride = stride
    if args.alpha_threshold is not None:
        cfg.alfs.alpha_threshold = args.alpha_threshold
    if args.label_mode:
        cfg.alfs.label_mode = args.label_mode
    if args.extract_neighbors is not None:
        cfg.extract_neighbors = args.extract_neighbors
    cfg.validate()

    flight_ids = _parse_flight_ids(args.flight_ids, cfg)
    if not flight_ids:
        raise SystemExit("No flights to process.")

    if args.dry_run:
        return _dry_run(cfg, flight_ids)

    if args.extract_only:
        if cfg.mode != "alfs":
            raise SystemExit("--extract-only requires --mode alfs")
        from .pipeline import extract_neighbors
        return 0 if extract_neighbors(cfg, flight_ids) else 1

    # Import here so --dry-run / --help don't require a GL context.
    from .pipeline import run
    results = run(cfg, flight_ids)
    failed = sum(1 for r in results if r.status == "failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
