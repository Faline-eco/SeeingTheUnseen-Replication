"""Assert that warping sources through a stored grid reproduces the integrated field.

`view_aggregator.py` trains on stacks it warps at load time from the geometry
`render_embedding_field.py --emit grid` wrote. If that warp with uniform
weights does not equal the stored `embfield` the published `embed_multi` cell
was trained on, the aggregator is being compared against a different input and
nothing it shows can be attributed to the aggregation. So check, per frame:
max |online mean - stored field| and coverage agreement.

    python scripts/check_viewgrid.py --grids <emb>/viewgrid_thermal \
        --field <emb>/cell_thermal_embed_multi --n 20
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from view_aggregator import ViewStackDet, ViewAggregatorDetector, collate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=Path, required=True)
    ap.add_argument("--field", type=Path, required=True,
                    help="flat <stem>.npy store of the integrated field")
    ap.add_argument("--labels", type=Path, required=True,
                    help="a labels dir whose stems select the frames")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = ViewStackDet(args.grids, args.labels)
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViewAggregatorDetector(128, agg="mean").to(dev).eval()
    worst = 0.0
    for i in idx[:args.n]:
        x, boxes, _ = ds[i]
        stem = ds.items[i][1].stem
        ref_p = args.field / f"{stem}.npy"
        if not ref_p.exists():
            print(f"{stem}: no stored field, skipped")
            continue
        batch, _, _ = collate([(x, boxes, i)])
        with torch.no_grad():
            out, _ = model.attention(batch.to(dev))
        ours = out[0].permute(1, 2, 0).cpu().numpy()                 # (H, W, C)
        ref = np.load(ref_p).astype(np.float32)
        d = np.abs(ours - ref)
        scale = np.abs(ref).max() + 1e-6
        cov_p = ref_p.with_name(ref_p.stem + "_cov.npy")
        cov_note = ""
        if cov_p.exists():
            cnt = batch.valid[0].sum(0).reshape(ref.shape[0], ref.shape[1]).numpy()
            cov_note = f" | cov max|d| {np.abs(cnt - np.load(cov_p)).max():.0f}"
        worst = max(worst, float(d.max() / scale))
        print(f"{stem}: views {x[0].shape[0]:2d} | max|d| {d.max():.4f} "
              f"(rel {d.max()/scale:.2e}) | mean|d| {d.mean():.2e} | "
              f"|ref|max {scale:.2f}{cov_note}")
    print(f"\nworst relative max deviation: {worst:.2e}  "
          f"({'OK' if worst < 1e-2 else 'MISMATCH'}; fp16 grid coordinates "
          f"bound this at ~1e-3 of the feature range)")
    return 0 if worst < 1e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
