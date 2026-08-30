"""Phase-batched driver for the V-JEPA cells. Replaces vjepa_pipeline.py.

The threaded producer/consumer version overlapped projection and encoding, but
produced five distinct defects in a row -- wrong output root, a semaphore that
capped projection concurrency, an unbounded CPU thread pool, an encoder that
processed every staged flight instead of its own, and finally a deadlock whose
disk bound did not bound disk (a worker writes its ~49 GB stack *before* it
blocks, so the real footprint was max_ready + proj_workers flights, not
max_ready). Each fix exposed the next.

This trades the overlap for something whose invariants can be checked by reading
it:

  * two sequential phases per batch, no queue, no semaphore, no daemon threads;
  * disk footprint is exactly `--batch` flights, because nothing is projected
    while encoding runs;
  * a failure in one flight cannot stall the others -- there is nothing to
    stall on;
  * resume is per flight, so an interrupted run costs at most one batch.

The lost overlap is cheap here: projection is CPU-bound and encoding GPU-bound,
but the measured phase times are close enough that serialising them costs far
less than another day of debugging concurrency.

    python vjepa_batch.py --modality rgb --batch 6 --proj-workers 12 --gpus 0,1,2
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/scratch/bambi")
DS = ROOT / "datasets/alfs_embed"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def docker_cmd(image: str, gpu: str | None, threads: int | None,
               *args: str) -> list[str]:
    c = ["docker", "run", "--rm", "--ipc=host"]
    if gpu is not None:
        c += ["--gpus", f'"device={gpu}"']
    if threads:
        c += ["-e", f"OMP_NUM_THREADS={threads}", "-e", f"MKL_NUM_THREADS={threads}"]
    c += ["-v", f"{ROOT}:{ROOT}", "-w", str(ROOT / "alfs_embed/code"),
          "-e", f"PYTHONPATH=/opt/bambi:{ROOT}/alfs_embed/code:"
                f"{ROOT}/alfs_embed/code/src:{ROOT}/alfs_embed/code/alfspy_src",
          "-e", f"TORCH_HOME={ROOT}/vjepa2/torch_hub",
          image, *args]
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", default="rgb", choices=("thermal", "rgb"))
    ap.add_argument("--kinds", default="single,multi")
    ap.add_argument("--variant", default="2.1-vit-b-384")
    ap.add_argument("--batch", type=int, default=6,
                    help="Flights per batch. This IS the disk bound: nothing is "
                         "projected while encoding runs.")
    ap.add_argument("--proj-workers", type=int, default=12)
    ap.add_argument("--proj-threads", type=int, default=16)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--src-px", type=int, default=2048)
    ap.add_argument("--proj-image", default="bambi-embed:1.5")
    ap.add_argument("--enc-image", default="bambi-vjepa:1.0")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debias-basis", type=Path, default=None,
                    help="INSID3 basis for fit_positional_basis_vjepa.py. Sends "
                         "the cells to _debias<rank> directories.")
    ap.add_argument("--debias-rank", type=int, default=32)
    args = ap.parse_args()

    mod, kinds = args.modality, [k for k in args.kinds.split(",") if k]
    gpus = [g for g in args.gpus.split(",") if g]
    stack_root = DS / f"vjepa_stack_{mod}_stack"
    # The suffix is applied HERE, not left to encode_vjepa_cells.py, because
    # `done()` below counts existing cells to decide what to skip. Pointed at the
    # baseline directories -- which are already complete -- it would report every
    # flight finished and the debiased run would encode nothing.
    suf = f"_debias{args.debias_rank}" if args.debias_basis else ""
    out_dirs = {k: DS / "embeddings" / f"cell_{mod}_vjepa_{k}{suf}" for k in kinds}

    def n_expected(fid: str) -> int:
        d = DS / "matched_dataset_new/alfs_2k" / fid / mod
        return len([p for p in d.glob("*.txt")
                    if not p.stem.endswith("_central")]) if d.is_dir() else 0

    def done(fid: str) -> bool:
        want = n_expected(fid)
        return want > 0 and all(
            len(list(out_dirs[k].glob(f"{fid}_*.npy"))) >= want for k in kinds)

    flights = sorted(p.name for p in (DS / "matched_dataset_new/alfs_2k").iterdir()
                     if p.is_dir())
    todo = [f for f in flights if n_expected(f) > 0 and not done(f)]
    print(f"[plan] {len(todo)} flights, batches of {args.batch}, "
          f"{args.proj_workers} projection workers, GPUs {','.join(gpus)}",
          flush=True)
    if args.dry_run:
        return 0

    t0 = time.time()
    ok = failed = 0
    for bi in range(0, len(todo), args.batch):
        batch = todo[bi:bi + args.batch]
        el = (time.time() - t0) / 60
        print(f"[{el:7.1f}m] batch {bi//args.batch + 1}: projecting {batch}", flush=True)

        # ---- phase 1: project the batch -------------------------------
        with ThreadPoolExecutor(max_workers=args.proj_workers) as ex:
            res = list(ex.map(lambda f: (f, run(docker_cmd(
                args.proj_image, None, args.proj_threads,
                "python", "scripts/render_embedding_field.py",
                "--config", "config_dgx.yaml", "--flight-ids", f,
                "--modality", mod, "--channels", "rgb", "--aperture", "full",
                "--out-hw", str(args.src_px), "--emit", "stack",
                "--device", "cpu", "--out", str(DS / f"vjepa_stack_{mod}")))), batch))
        projected = [f for f, r in res if r.returncode == 0]
        for f, r in res:
            if r.returncode != 0:
                print(f"           PROJECT FAILED {f}: "
                      f"{r.stderr.strip().splitlines()[-1:]}", flush=True)
                failed += 1

        # ---- phase 2: encode the batch, one flight per GPU at a time ---
        el = (time.time() - t0) / 60
        print(f"[{el:7.1f}m]   encoding {len(projected)} flights", flush=True)
        jobs = [(f, k) for f in projected for k in kinds]

        def encode(job, _gpus=gpus):
            f, k = job
            gpu = _gpus[hash(f + k) % len(_gpus)]
            extra = (["--debias-basis", str(args.debias_basis),
                      "--debias-rank", str(args.debias_rank)]
                     if args.debias_basis else [])
            return job, run(docker_cmd(
                args.enc_image, gpu, None,
                "python", "scripts/encode_vjepa_cells.py",
                "--stack-root", str(stack_root), "--flight", f, "--kind", k,
                "--modality", mod, "--variant", args.variant,
                "--src-px", str(args.src_px),
                "--train-stems", str(DS / f"zenodo_labels/{mod}_train_stems.txt"),
                *extra, "--out", str(out_dirs[k])))

        with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
            for (f, k), r in ex.map(encode, jobs):
                if r.returncode != 0:
                    print(f"           ENCODE FAILED {f}/{k}: "
                          f"{r.stderr.strip().splitlines()[-1:]}", flush=True)
                    failed += 1

        # ---- phase 3: free the disk before the next batch --------------
        for f in projected:
            shutil.rmtree(stack_root / f, ignore_errors=True)
        ok += len(projected)
        el = (time.time() - t0) / 60
        print(f"[{el:7.1f}m]   batch done ({ok}/{len(todo)} flights, "
              f"{failed} failures)", flush=True)

    print(f"[done] {ok} flights, {failed} failures in {(time.time()-t0)/60:.1f} min",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
