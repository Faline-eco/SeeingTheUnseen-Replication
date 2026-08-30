"""Train the ALFS detectors and score them against the released ortho baselines.

Trains one YOLO26x per modality on the ALFS renderings, then evaluates the
publicly released orthographic detectors
(https://huggingface.co/cpraschl/bambi-orthographic-detectors) on the **same
val frames**, so both sides of the comparison are measured over an identical
frame set with identical thresholds.

    py scripts\\train_yolo.py --datasets D:\\yolo_datasets --out D:\\yolo_runs \\
        --sources alfs --ortho-weights D:\\bambi_models

Targets are small — a 1.5 m animal inside a 70 m orthographic window at 1024 px
is roughly 22 px across — so training runs at the native 1024 px rather than the
usual 640, and the default batch is sized for a 16 GB card.

Results land in `<out>/comparison.json` and are printed as a table of
mAP50 / mAP50-95 / precision / recall per model.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SOURCES = ("raw", "ortho", "alfs")
MODALITIES = ("rgb", "thermal")
# Released orthographic baselines, keyed by modality. These are already trained,
# so they are only ever validated — never fine-tuned here.
ORTHO_BASELINE = {"rgb": "detector_1k_rgb.pt", "thermal": "detector_1k_thermal.pt"}


def _metrics(box, name: str, minutes: float, weights: str, extra: dict) -> dict:
    return {
        "name": name,
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "minutes": round(minutes, 1),
        "weights": weights,
        **extra,
    }


def eval_baseline(data_yaml: Path, weights: Path, name: str, args) -> dict:
    """Validate an already-trained checkpoint on our val split.

    Same imgsz / conf / IoU as the trained models, so the numbers in the table
    are directly comparable rather than merely quoted from a model card.
    """
    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO(str(weights))
    # workers=0: spawning a fresh dataloader pool in a process that has already
    # run one deadlocks ultralytics on Windows (the val step hangs indefinitely
    # at 10 W with the process alive). Val sets here are a few thousand images,
    # so single-process loading costs a couple of minutes and cannot wedge.
    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device,
                        batch=args.batch, project=str(args.out), name=f"{name}_val",
                        exist_ok=True, plots=True, workers=0)
    return _metrics(metrics.box, name, (time.time() - t0) / 60, str(weights),
                    {"trained_here": False, "classes": list(model.names.values())})


def train_one(data_yaml: Path, name: str, args) -> dict:
    from ultralytics import YOLO

    model = YOLO(args.model)
    t0 = time.time()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.out),
        name=name,
        seed=args.seed,
        patience=args.patience,
        # Seeded but not bit-deterministic: deterministic=True also disables
        # cuDNN autotuning, and the run is long enough that the trade is not
        # worth it. Both models use the same seed and schedule.
        deterministic=False,
        exist_ok=True,
        val=True,
        plots=True,
    )
    # Re-validate the best checkpoint so every number in the table comes from
    # the same code path regardless of where early stopping landed.
    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device,
                        batch=args.batch, project=str(args.out),
                        name=f"{name}_val", exist_ok=True, workers=0)
    return _metrics(metrics.box, name, (time.time() - t0) / 60,
                    str(args.out / name / "weights" / "best.pt"),
                    {"trained_here": True, "epochs": args.epochs,
                     "arch": args.model})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", type=Path, required=True,
                    help="Root written by build_yolo_datasets.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="yolo26x.pt",
                    help="Matches the released orthographic baselines (YOLO26x).")
    ap.add_argument("--ortho-weights", type=Path, default=None,
                    help="Directory holding the released detector_1k_*.pt. When "
                         "given, each is validated on the matching ortho dataset "
                         "and added to the comparison table.")
    ap.add_argument("--sources", default="alfs")
    ap.add_argument("--modalities", default=",".join(MODALITIES))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1024)
    # Batch 4, not 8. YOLO26x at 1024 with batch 8 fills 15.9 of the 4080's
    # 16.4 GiB; at 97% occupancy the CUDA allocator thrashes and throughput
    # collapses to 0.068 it/s at 73 W (vs 8.0 it/s at batch 4) -- a 118x
    # difference that looks like a busy GPU, because utilisation still reads
    # 100%. Raise this only alongside a VRAM check.
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--retrain", action="store_true",
                    help="Retrain even when a best.pt already exists (default is "
                         "to evaluate the saved checkpoint).")
    ap.add_argument("--only", default=None,
                    help="Internal: run exactly one job in this process. The "
                         "top-level invocation re-execs itself with this per "
                         "job, because training two models back to back in one "
                         "process deadlocks ultralytics' Windows dataloader "
                         "workers (observed: a 6.5 h silent hang after the "
                         "second dataset's label cache was built).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "comparison.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {r["name"] for r in results}

    combos = [(s, m)
              for m in (v.strip() for v in args.modalities.split(","))
              for s in (v.strip() for v in args.sources.split(","))]

    for source, modality in combos:
        name = f"{source}_{modality}"
        data_yaml = args.datasets / name / "data.yaml"
        if not data_yaml.exists():
            print(f"[skip] {name}: no dataset at {data_yaml}")
            continue
        if name in done:
            print(f"[skip] {name}: already in {results_path.name}")
            continue

        if args.only is None:
            # Re-exec one job per process (see --only). Each child writes its own
            # results into comparison.json before exiting.
            print(f"\n{'=' * 70}\n=== launching {name} in a fresh process\n{'=' * 70}",
                  flush=True)
            rc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--datasets", str(args.datasets), "--out", str(args.out),
                 "--model", str(args.model), "--sources", source,
                 "--modalities", modality, "--epochs", str(args.epochs),
                 "--imgsz", str(args.imgsz), "--batch", str(args.batch),
                 "--device", args.device, "--workers", str(args.workers),
                 "--patience", str(args.patience), "--seed", str(args.seed),
                 "--only", name] + (["--retrain"] if args.retrain else []),
            ).returncode
            if rc != 0:
                print(f"[warn] {name} exited with code {rc}")
            results = json.loads(results_path.read_text()) if results_path.exists() else []
            done = {r["name"] for r in results}
            continue

        if args.only != name:
            continue

        existing = args.out / name / "weights" / "best.pt"
        if existing.exists() and not args.retrain:
            # Training finished but a later step (e.g. the val pass) died, so the
            # run never reached comparison.json. Score the saved checkpoint
            # instead of spending hours retraining it.
            print(f"\n{'=' * 70}\n=== {name}: reusing existing best.pt, evaluating only"
                  f"\n{'=' * 70}")
            results.append(eval_baseline(data_yaml, existing, name, args))
        else:
            print(f"\n{'=' * 70}\n=== training {name}\n{'=' * 70}")
            results.append(train_one(data_yaml, name, args))
        # Persist after each run so a crash never loses completed training.
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        # ALFS models train on merged (per-track, whole-aperture) labels because
        # that is what the integral shows. Score them a second time against the
        # central-frame ground truth as well — that is the label set the ortho
        # detector is measured on, so only this number is comparable to it.
        central = args.datasets / f"{name}_centralgt" / "data.yaml"
        if central.exists():
            alt = f"{name}_vs_centralgt"
            if alt not in done:
                print(f"\n--- re-scoring {name} against central-frame GT")
                weights = args.out / name / "weights" / "best.pt"
                results.append(eval_baseline(central, weights, alt, args))
                results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.ortho_weights:
        for modality in (v.strip() for v in args.modalities.split(",")):
            name = f"ortho_{modality}_released"
            weights = args.ortho_weights / ORTHO_BASELINE[modality]
            data_yaml = args.datasets / f"ortho_{modality}" / "data.yaml"
            if name in done:
                print(f"[skip] {name}: already in {results_path.name}")
                continue
            if not weights.exists() or not data_yaml.exists():
                print(f"[skip] {name}: missing {weights if not weights.exists() else data_yaml}")
                continue
            print(f"\n{'=' * 70}\n=== evaluating released baseline {name}\n{'=' * 70}")
            results.append(eval_baseline(data_yaml, weights, name, args))
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'model':26s} {'mAP50':>8s} {'mAP50-95':>9s} {'P':>8s} {'R':>8s} {'min':>7s}")
    print("-" * 72)
    for r in sorted(results, key=lambda r: r["name"]):
        print(f"{r['name']:26s} {r['mAP50']:8.4f} {r['mAP50-95']:9.4f} "
              f"{r['precision']:8.4f} {r['recall']:8.4f} {r['minutes']:7.1f}")
    print(f"\nWrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
