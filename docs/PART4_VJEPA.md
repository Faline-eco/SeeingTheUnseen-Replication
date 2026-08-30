# Part 4 — V-JEPA 2 on the same occlusion axis

> **Superseded 2026-08-12.** The V-JEPA figures in this document (visible
> 0.9263 / hidden 0.2184, LoRA, image space, published single split) mixed head
> size, backbone adaptation, evaluation space and split all at once. V-JEPA is
> now a cell in the grid under design C -- frozen ViT-B, same 128x128 grid, same
> PCA width, same CenterNet head, same 5 folds -- giving thermal hidden recall
> **0.3752 +/- 0.1510** (multi) and **0.2436 +/- 0.0935** (single) against
> DINOv3's 0.3192 and 0.2125. See `paper/methodology.tex` Sec. "Foundation-model
> comparison". Do not quote the numbers below as current.


Measured 2026-08-07. V-JEPA 2 ViT-L, val flights 10 / 211 / 212 / 213, IoU 0.5,
conf >= 0.3 — the same operating point as Parts 1–3.

V-JEPA predicts in raw camera image space, not the DEM-plane ortho canvas, so
**only recall transfers**. mAP and precision do not: the two spaces have
different fields of view and different pixel scales. Every number here is a
recall.

Its occluded figure had also been quoted against a different population from the
rest of the grid — all centrally-invisible boxes, without the hidden /
wider-footprint split. This puts it on Part 1's population A.

## Alignment problem, and why it did not need a re-projection

`project_occluded_gt.py` drops boxes that fail to project (no DEM hit, behind
the camera, outside the frame) without recording which merged-GT index they came
from. The hidden mask is indexed by that list, so in any frame where a box was
dropped the two cannot be aligned — and a silent misalignment would assign hidden
flags to the wrong boxes.

Measured extent: **4 frames per modality**, 4 boxes of 243 (thermal) and 4 of 241
(rgb). Those frames are excluded and the exclusion is reported by the script.
`project_occluded_gt.py` now also writes
`<mod>_occluded_imagespace_idx.json`, so a future re-projection will not need it.

---

## Results

| model | modality | R visible | R occluded (all) | R hidden | retained |
|---|---|---|---|---|---|
| V-JEPA LoRA | thermal | **0.9263** | 0.1915 | 0.2184 | 24 % |
| V-JEPA frozen | thermal | 0.9159 | 0.1872 | 0.2087 | 23 % |
| V-JEPA LoRA | rgb | 0.8797 | 0.2189 | 0.2394 | 27 % |

n = 963 visible / 206 hidden (thermal), 964 / 213 (rgb).

Restricting to genuinely hidden animals *raises* the occluded number slightly
(0.1915 → 0.2184), because wider-footprint boxes — animals the central view never
covered — are the hardest cases and are now correctly excluded from a question
they never belonged in.

**Frozen vs LoRA is a null on occlusion.** 0.2087 vs 0.2184 on hidden animals,
0.9159 vs 0.9263 on visible ones. The 2.36 M LoRA adapters buy about one point
everywhere and nothing specific to occlusion. Whatever lets V-JEPA find a hidden
animal is in the pretrained features, not in the adaptation.

---

## Where V-JEPA sits in the grid (thermal, hidden animals, n≈206–212)

| method | view | space | R visible | R hidden | retained |
|---|---|---|---|---|---|
| YOLO26x ortho | single | pixels | 0.7774 | 0.0631 | 8 % |
| small head, realortho | single | pixels | 0.4958 | 0.0739 | 15 % |
| small head, realalfs | multi | pixels | 0.5955 | 0.1179 | 20 % |
| YOLO26x alfs | multi | pixels | 0.7464 | 0.1845 | 25 % |
| small head, embed | single | DINOv3 | 0.7761 | 0.1887 | 24 % |
| **V-JEPA 2 (LoRA)** | temporal clip | V-JEPA | **0.9263** | 0.2184 | 24 % |
| **small head, embed** | **multi** | **DINOv3** | 0.8407 | **0.4088** | **49 %** |

This is the cleanest statement of the whole comparison. V-JEPA is **the best
method in the study on animals that are visible** — 0.9263, ahead of every ortho
cell — and it is beaten nearly 2:1 on hidden animals by a 0.28 M head reading
DINOv3 features off a geometrically aligned light field.

Note where V-JEPA lands: at 24 % retention it is level with `embed/single`
(24 %) and `YOLO alfs` (25 %), and far below `embed/multi` (49 %). A 300 M-param
video model with temporal context reaches exactly the same occlusion robustness
as a small head on single-view features. Scale and temporal context do not
substitute for aperture.

One qualification on "single view": V-JEPA consumes a temporal clip, so it does
see multiple frames. What it does not do is align them to a focus plane. The
comparison therefore separates *temporal* multi-frame evidence from *geometric*
multi-view integration, and finds the geometry is what matters for occlusion —
consistent with Part 3's finding that partial occlusion (where geometry has
nothing to recover) is where multi-view stops helping.

---

## Caveats

* Image space, single checkpoint per variant, and a population that overlaps but
  is not identical to the ortho cells' (206 vs 212 hidden boxes thermal).
* The visible column is against MOT central GT, which is V-JEPA's own training
  ground truth definition; the ortho cells' visible column is against merged-GT
  boxes that matched a central box. These are close but not the same set, so the
  visible column is the weakest cross-method comparison in the table.
* No rgb frozen control was run.
