"""Sensor cells built from the REAL renders rather than my integrator.

The grid's raw/multi cell is my PyTorch integrator's RGB output at 128x128. It
agrees with the production renderer at r=0.967, but the renderer also applies
auto-contrast (alfs.auto_contrast: true) and runs at 2048px, and either could
matter to a 0.28M head. This builds the alternative cell directly from the
rendered PNGs so the two can be trained head-to-head.

Downsampled with INTER_AREA to the same 128x128x3 the other cells use, so the
comparison isolates renderer-vs-reimplementation rather than resolution.
"""
import numpy as np, cv2, glob, os, sys
mod = sys.argv[1] if len(sys.argv) > 1 else "thermal"
# which render tree: the ALFS integral (multi view) or the single-frame
# orthographic projection (single view). Both come from the same renderer at
# 2048px, so multi-vs-single is measured without a renderer confound.
kind = sys.argv[2] if len(sys.argv) > 2 else "alfs"
REND = ("/scratch/bambi/datasets/alfs_embed/matched_dataset_new/alfs_2k" if kind == "alfs"
        else "/scratch/bambi/datasets/alfs_embed/matched_dataset_new/geo-referenced2_2k")
OUT = (f"/scratch/bambi/datasets/alfs_embed/embeddings/cell_{mod}_realalfs_multi" if kind == "alfs"
       else f"/scratch/bambi/datasets/alfs_embed/embeddings/cell_{mod}_realortho_single")
os.makedirs(OUT, exist_ok=True)
n = 0
for p in sorted(glob.glob(f"{REND}/*/{mod}/*.png")):
    fl = p.split("/")[-3]
    fr = os.path.basename(p)[:-4]
    dst = f"{OUT}/{fl}_{fr}.npy"
    if os.path.exists(dst):
        n += 1; continue
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None:
        continue
    if im.ndim == 3 and im.shape[2] == 4:
        a = im[:, :, 3:4].astype(np.float32) / 255.0
        im = im[:, :, :3].astype(np.float32) * a
    elif im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR).astype(np.float32)
    else:
        im = im.astype(np.float32)
    small = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA) / 255.0
    np.save(dst, small.astype(np.float16))
    n += 1
    if n % 2000 == 0:
        print(f"  {n}", flush=True)
print(f"[done] {n} -> {OUT}")
