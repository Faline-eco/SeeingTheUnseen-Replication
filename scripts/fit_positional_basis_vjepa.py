"""Estimate V-JEPA 2.1's positional-bias subspace from content-free clips.

The INSID3 counterpart to `fit_positional_basis_dino.py`, for the video encoder
actually used by `encode_vjepa_cells.py`. Same recipe: push Gaussian *noise*
through the encoder, and take the leading eigenvectors of the token Gram matrix
as the positional subspace B, removed later by

    F~ = F (I_D - B B^T)

Why not the basis already sitting at /scratch/bambi/vjepa2/positional_basis.npz:
that one was fitted for `facebook/vjepa2-vitl-fpc64-256` -- ViT-L, 1024 dims,
clip length 64, 256x256 input. Our cells use 2.1-vit-b-384: 768 dims, clip
length 30, 384x384 tiles. The basis is specific to both the width and the token
grid, so reusing it would project features onto directions estimated for a
different model and a different geometry.

**The geometry must match the encoding run.** `--tile` and `--clip-len` take the
values `encode_vjepa_cells.py` will actually feed the encoder, for the same
reason the DINOv3 fitter takes a `--size`.

    python fit_positional_basis_vjepa.py \\
        --out /scratch/bambi/datasets/alfs_embed/zenodo_labels/vjepa_positional_basis.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Same ImageNet constants encode_vjepa_cells.py normalises with. Noise must go
# through the identical preprocessing, or the subspace is estimated for inputs
# the encoder never sees.
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def make_noise_clip(rng, t: int, hw: int, kind: str, std: float) -> torch.Tensor:
    """-> (1,T,C,hw,hw) normalised float32, matching `to_clip`'s output."""
    if kind == "uniform":
        arr = rng.integers(0, 256, size=(t, hw, hw, 3), dtype=np.uint8)
    else:
        arr = np.clip(rng.normal(127.0, std, size=(t, hw, hw, 3)),
                      0, 255).astype(np.uint8)
    crop = arr.astype(np.float32) / 255.0
    crop = (crop - _MEAN) / _STD
    return torch.from_numpy(crop).permute(0, 3, 1, 2).contiguous().unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", default="2.1-vit-b-384")
    ap.add_argument("--vjepa-src", type=Path, default=Path("/scratch/bambi/vjepa2/src"))
    ap.add_argument("--tile", type=int, default=384,
                    help="Tile edge fed to the encoder; must match --tile there.")
    ap.add_argument("--clip-len", type=int, default=30,
                    help="Must match --clip-len there (tubelet 2 needs it even).")
    ap.add_argument("--num-clips", type=int, default=32,
                    help="32 clips at 384px/30 frames is ~276k tokens, far more "
                         "than the 768 dimensions being estimated.")
    ap.add_argument("--noise", default="gaussian", choices=["gaussian", "uniform"])
    ap.add_argument("--noise-std", type=float, default=60.0,
                    help="Gaussian std in 0-255 pixel units; matches the DINOv3 fit.")
    ap.add_argument("--autocast", default="fp16", choices=("fp16", "bf16", "off"),
                    help="fp32 weights with autocast, as in the encoding run: "
                         "casting V-JEPA 2.1's weights directly to fp16 fails.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.clip_len % 2:
        ap.error("--clip-len must be even (tubelet size 2)")

    sys.path.insert(0, str(args.vjepa_src))
    from vjepa21 import encode as vj_encode, load_encoder   # noqa: E402

    ac_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                "off": None}[args.autocast]
    rng = np.random.default_rng(args.seed)

    print(f"[load] {args.variant}")
    encoder, info = load_encoder(args.variant, device=args.device,
                                 dtype=torch.float32)
    dim = info["embed_dim"]
    per_clip = (args.clip_len // info["tubelet_size"]) * (args.tile // info["patch_size"]) ** 2
    print(f"[plan] {args.num_clips} noise clips, {args.clip_len}x{args.tile}x{args.tile} "
          f"-> {per_clip} tokens each, dim {dim}")

    # Accumulate the Gram matrix rather than the tokens: the right singular
    # vectors of M are the eigenvectors of M^T M, and D is only 768 while the
    # token matrix would be ~276k x 768.
    gram = np.zeros((dim, dim), np.float64)
    n_tokens = 0
    t0 = time.time()

    for i in range(args.num_clips):
        clip = make_noise_clip(rng, args.clip_len, args.tile,
                               args.noise, args.noise_std).to(args.device)
        with torch.inference_mode():
            if ac_dtype is None:
                tok = vj_encode(encoder, clip)
            else:
                with torch.autocast(device_type="cuda", dtype=ac_dtype):
                    tok = vj_encode(encoder, clip)
        m = tok.float().reshape(-1, dim).cpu().numpy().astype(np.float64)
        gram += m.T @ m
        n_tokens += len(m)
        if (i + 1) % 8 == 0 or i + 1 == args.num_clips:
            print(f"  encoded {i + 1}/{args.num_clips} clips ({n_tokens} tokens)")

    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0, None), vecs[:, order]
    evr = vals / (vals.sum() + 1e-12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             basis=vecs.astype(np.float32),
             singular_values=np.sqrt(vals).astype(np.float32),
             explained_variance_ratio=evr.astype(np.float32))
    args.out.with_suffix(".json").write_text(json.dumps({
        "variant": args.variant, "tile": args.tile, "clip_len": args.clip_len,
        "num_clips": args.num_clips, "num_tokens": int(n_tokens),
        "noise": args.noise,
        "noise_std": args.noise_std if args.noise == "gaussian" else None,
        "hidden_size": dim, "patch_size": info["patch_size"],
        "tubelet_size": info["tubelet_size"], "autocast": args.autocast,
        "runtime_seconds": round(time.time() - t0, 2),
    }, indent=2), encoding="utf-8")

    print(f"\n[done] {time.time() - t0:.1f}s over {n_tokens} noise tokens")
    print("[spectrum] cumulative variance of the noise-feature subspace")
    for r in (1, 8, 32, 64, 128, 256, 512, dim):
        if r <= dim:
            print(f"  rank {r:4d}: {evr[:r].sum() * 100:6.2f}%")
    print("\nIf this is not concentrated in a few dozen directions, the noise "
          "response\nis not low-rank and projecting it out would remove semantic "
          "capacity too.")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
