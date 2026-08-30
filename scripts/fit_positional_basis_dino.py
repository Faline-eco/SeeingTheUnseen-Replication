"""Estimate DINOv3's positional-bias subspace from content-free inputs.

Follows the INSID3 recipe (arXiv 2603.28480), which is stated for DINOv3
features: a low-rank subspace encodes absolute patch position rather than
semantics, and is removed by projecting onto the orthogonal complement,

    F~ = F (I_D - B B^T)

B is estimated by pushing Gaussian *noise* images through the encoder. Noise
carries no semantic content, so whatever structure survives in the token
features is positional (plus the encoder's mean response), and the top right
singular vectors of the noise feature matrix span it.

This mirrors the V-JEPA project's `fit_positional_basis.py` deliberately, so the
two pipelines are debiased the same way and the representation axis is not
confounded by preprocessing. Differences are only those forced by DINOv3 being
an image model rather than a video one: noise *images* instead of clips, and no
tubelet dimension.

The full sorted basis and its singular values are stored, so the rank can be
chosen afterwards without re-encoding.

**The image size must match the encoding run.** The basis is position-specific:
one fitted at 1024x1024 does not describe the token grid produced at 2048x2048.
`--size` therefore takes the *post-`--input-scale`* edge length that
`encode_embeddings.py` will actually feed the extractor.

    python fit_positional_basis_dino.py --model-dir /scratch/bambi/DINOv3 \
        --size 2048 --out /scratch/bambi/datasets/alfs_embed/dino_positional_basis.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker"))


def make_noise_image(rng, h: int, w: int, kind: str, std: float) -> np.ndarray:
    """-> uint8 BGR (H, W, 3), matching what _load_image would hand the extractor."""
    if kind == "uniform":
        return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return np.clip(rng.normal(127.0, std, size=(h, w, 3)), 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="/scratch/bambi/datasets/alfs_embed/DINOv3",
                    help="Local HuggingFace DINOv3 directory (as used by encode_embeddings).")
    ap.add_argument("--out", type=Path, required=True, help="Output .npz holding the basis.")
    ap.add_argument("--size", type=int, default=2048,
                    help="Edge length fed to the extractor, i.e. the render size "
                         "times --input-scale. Must match the encoding run: the "
                         "positional subspace depends on the token grid.")
    ap.add_argument("--num-images", type=int, default=64,
                    help="Independent noise images. 64 at 2048px is ~1.05M tokens, "
                         "far more than the 1280 dimensions being estimated.")
    ap.add_argument("--noise", default="gaussian", choices=["gaussian", "uniform"])
    ap.add_argument("--noise-std", type=float, default=60.0,
                    help="Gaussian std in 0-255 pixel units; matches the V-JEPA fit.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from dino_extractor import DinoV3Extractor

    rng = np.random.default_rng(args.seed)
    print(f"[load] {args.model_dir}")
    extractor = DinoV3Extractor(args.model_dir)
    dim = extractor.embed_dim
    print(f"[plan] {args.num_images} noise images at {args.size}x{args.size} "
          f"-> {(args.size // extractor.patch_size) ** 2} tokens each, dim {dim}")

    # Accumulate the Gram matrix rather than every token: the right singular
    # vectors of M are the eigenvectors of M^T M, and D is only 1280, while the
    # token matrix would be ~1M x 1280.
    gram = np.zeros((dim, dim), np.float64)
    n_tokens = 0
    t0 = time.time()

    for i in range(args.num_images):
        img = make_noise_image(rng, args.size, args.size, args.noise, args.noise_std)
        emb = extractor.extract(img)                       # (h, w, D) float32
        m = emb.reshape(-1, dim).astype(np.float64)
        gram += m.T @ m
        n_tokens += len(m)
        if (i + 1) % 8 == 0 or i + 1 == args.num_images:
            print(f"  encoded {i + 1}/{args.num_images} images ({n_tokens} tokens)")

    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0, None), vecs[:, order]
    evr = vals / (vals.sum() + 1e-12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             basis=vecs.astype(np.float32),            # (D, D), col k = k-th positional direction
             singular_values=np.sqrt(vals).astype(np.float32),
             explained_variance_ratio=evr.astype(np.float32))
    args.out.with_suffix(".json").write_text(json.dumps({
        "model_dir": args.model_dir, "size": args.size,
        "num_images": args.num_images, "num_tokens": int(n_tokens),
        "noise": args.noise,
        "noise_std": args.noise_std if args.noise == "gaussian" else None,
        "hidden_size": dim, "patch_size": extractor.patch_size,
        "runtime_seconds": round(time.time() - t0, 2),
    }, indent=2), encoding="utf-8")

    print(f"\n[done] {time.time() - t0:.1f}s over {n_tokens} noise tokens")
    print("[spectrum] cumulative variance of the noise-feature subspace")
    for r in (1, 8, 32, 64, 128, 256, 512, dim):
        if r <= dim:
            print(f"  rank {r:4d}: {evr[:r].sum() * 100:6.2f}%")
    print("\nA subspace that is genuinely low-rank should reach most of its "
          "variance\nwithin a few dozen directions. If it does not, the noise "
          "response is not\nconcentrated and projecting it out will remove "
          "semantic capacity too --\nwhich is the case where this whole idea "
          "does not apply.")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
