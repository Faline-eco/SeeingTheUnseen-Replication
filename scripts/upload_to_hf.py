"""Publish trained checkpoints to the Hugging Face model repo.

Reads the token from the ``HF_TOKEN`` environment variable -- never pass it on
the command line and never commit it, since anything in ``argv`` shows up in
shell history and in ``ps``.

Files go up in batches rather than one commit each: 180 separate commits would
bury the repo history, and a single commit holding 6.6 GB of LFS pointers is
slow to retry when one upload fails. Batches of a dozen are a reasonable middle,
and a failed batch can be re-run because uploading an identical file is a no-op.

    HF_TOKEN=hf_... python upload_to_hf.py \
        --runs /scratch/bambi/datasets/alfs_embed/yolo_runs_cv \
        --prefix yolo --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import CommitOperationAdd

# per run directory: local relative path -> name in the repo
WANTED = {
    "weights/best.pt": "best.pt",     # ultralytics nests the weights
    "results.csv": "results.csv",     # per-epoch metrics
    "args.yaml": "args.yaml",         # the exact training configuration
}


def collect(runs: Path, prefix: str) -> list[tuple[Path, str]]:
    out = []
    for run in sorted(p for p in runs.iterdir() if p.is_dir()):
        for rel, name in WANTED.items():
            src = run / rel
            if src.is_file():
                out.append((src, f"{prefix}/{run.name}/{name}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True,
                    help="Directory of run subdirectories to publish.")
    ap.add_argument("--repo", default="cpraschl/SeeingTheUnseen")
    ap.add_argument("--prefix", default="yolo",
                    help="Path inside the repo to place <run>/<file> under.")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--message", default="Add YOLO26x cross-validation checkpoints")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("HF_TOKEN is not set", file=sys.stderr)
        return 2

    files = collect(args.runs, args.prefix)
    if not files:
        print(f"nothing to upload under {args.runs}", file=sys.stderr)
        return 1
    total = sum(s.stat().st_size for s, _ in files)
    print(f"{len(files)} files, {total / 1073741824:.2f} GB "
          f"from {len(set(d.split('/')[1] for _, d in files))} runs")
    if args.dry_run:
        for s, d in files[:6]:
            print(f"  {s}  ->  {d}")
        print(f"  ... and {len(files) - 6} more")
        return 0

    api = HfApi(token=token)
    batches = [files[i:i + args.batch] for i in range(0, len(files), args.batch)]
    for n, batch in enumerate(batches, 1):
        ops = [CommitOperationAdd(path_in_repo=d, path_or_fileobj=str(s))
               for s, d in batch]
        api.create_commit(repo_id=args.repo, repo_type="model", operations=ops,
                          commit_message=f"{args.message} ({n}/{len(batches)})")
        done = sum(s.stat().st_size for s, _ in files[:n * args.batch])
        print(f"  batch {n}/{len(batches)}: {done / 1073741824:.2f} GB uploaded",
              flush=True)
    print(f"done: {len(files)} files to {args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
