# Method compendium — what each arm computes, and where it belongs in the 2×2

Companion to `RESULTS.md`. Every method in the study, what it actually does, why
it sits in the quadrant it sits in, which checkpoint produced its numbers, and
the full visible/occluded results on a single consistent population.

---

## 1. What the two axes mean

The grid is a 2×2 of **mechanisms**, not of implementations. Getting the axes
precise matters, because two of the methods are easy to place wrongly.

### Axis X — single vs multi view

The question is **"how many geometrically registered viewpoints of the same
ground patch does the method integrate?"** — not "how many frames does it see".

*Multi view* here means the 31-shot synthetic aperture: ±45 video frames at
stride 3, ≈9 m of baseline at ≈0.3 m spacing, all projected onto the same DEM
focus plane. Content at the focus plane reinforces across shots; content off it
(canopy, mostly) smears. That is the mechanism the occlusion claim rests on.

A method that consumes many frames **without registering them to a focus plane**
is not exercising this mechanism. It has temporal context, which is a different
resource: it can track an animal that moves relative to its occluder, but it
cannot synthesise a view through the occluder.

### Axis Y — sensor space vs embedding space

The question is **"in what representation is the detector's input expressed?"**

*Sensor space* is intensity: a rendered image, downsampled to 128×128×3.
*Embedding space* is DINOv3 ViT-H+/16 patch tokens, PCA-reduced, on a
128×128×D grid.

This axis has an internal ordering that the 2×2 alone hides. When both axes are
active there are two ways to combine them, and they are **not** the same method:

| | what is integrated | when |
|---|---|---|
| **average-then-encode** ("embedded ALFS") | pixels | before the encoder |
| **encode-then-average** ("embedded light field") | feature vectors | after the encoder |

Both are legitimately multi-view *and* embedding-space, so both occupy the same
quadrant — but they differ by 50 % relative on hidden animals (§4). The order of
operations is a first-class design decision, not an implementation detail.

### Capacity is a third, orthogonal dimension

Detector capacity spans 0.21 M → ~60 M → ~300 M parameters across these methods
and is **not** one of the axes. It is deliberately controlled instead: the same
0.21/0.36 M head is used for all four grid cells, and YOLO26x / V-JEPA are added
as capacity controls on top. Without this, "multi-view helps" and "bigger model
helps" would be confounded.

---

## 2. Single view / sensor space

### 2.1 Orthorectified central frame + 0.21 M head — `realortho_single`

The reference cell: one camera frame resampled onto the 70×70 m DEM-plane ortho
window at 2048², then INTER_AREA-downsampled to 128×128×3. Renderer is alfspy in
`ortho` mode, so it shares every geometric convention with the ALFS cell — the
only difference is that the aperture contains one shot instead of 31.

This is the origin of both axes. Everything else is measured as a departure
from it.

*Checkpoints:* `cell_{thermal,rgb}_realortho_single_realortho_single_s{6 seeds}`

### 2.2 Orthorectified central frame + YOLO26x — `ortho_thermal` / `ortho_rgb`

Same geometry at 1024², a ~60 M-parameter detector at imgsz 1024, 100 epochs.
Its role is to answer "is the grid's conclusion an artifact of a small head?".
It is the strongest single-view sensor-space detector in the study and the
weakest non-control method on hidden animals — see §5.

*Checkpoints:* `models/yolo/{ortho_thermal,ortho_rgb}/best.pt`

### 2.3 PyTorch integrator, single view — `sensor_single` *(control, not a result)*

A GL-free reimplementation of the projection (embree ray-cast + `grid_sample`).
Retained because the study's early conclusions were built on it before the real
renderer was wired in, and it turned out to understate thermal by 63 % on
aggregate and to **invert** the multi-view finding on hidden animals. It is
published here as a cautionary control, not as a competing method.

*Checkpoints:* `cell_{thermal,rgb}_sensor_single_sensor_single_s*`

---

## 3. Multi view / sensor space

### 3.1 ALFS integral render + 0.21 M head — `realalfs_multi`

alfspy `render_integral` over the 31-shot aperture, focused on the DEM, 2048²,
downsampled identically to 2.1. Pixel intensities are averaged across shots
after projection, so an animal at the focus plane accumulates while the canopy
above it — which projects to a different place in each shot — washes out.

Paired with 2.1 this is the **clean single-vs-multi contrast in pixel space**:
same renderer, same window, same downsample, same head, same seeds.

*Checkpoints:* `cell_{thermal,rgb}_realalfs_multi_realalfs_multi_s*`

### 3.2 ALFS integral render + YOLO26x — `alfs_thermal` / `alfs_rgb`

Capacity control on the multi-view sensor cell. Also the arm that exposed the
ground-truth trap: scored against *central* GT it looks 39 % worse than the
single-view detector, because central GT counts every correctly recovered hidden
animal as a false positive. Against the neutral merged GT it is 2.00× (thermal) /
2.04× (rgb) better. See `RESULTS.md` §5.4.

*Checkpoints:* `models/yolo/{alfs_thermal,alfs_rgb}/best.pt`

### 3.3 PyTorch integrator, multi view — `sensor_multi` *(control)*

As 2.3. Its multi/single ratio on hidden animals is 1.18× and not separable,
against the real renderer's 1.60× — the single clearest demonstration that the
reimplementation was not an adequate stand-in.

---

## 4. Multi view / embedding space

Four distinct methods share this quadrant. They differ in *where* integration
happens and at what width, and those differences are large.

### 4.1 Embedded ALFS — `alfs_2k_*` — **average-then-encode**

DINOv3 is applied **to the ALFS render**. The 31 views have already been
collapsed into a single image by the renderer; the encoder never sees them
separately. Multi-view and embedding are both present, but serially.

Available at 64 / 128 / 256 PCA dims (nested components, so a run truncates via
`--use-dims` without re-encoding) and at 1024 px as a resolution control.

The mechanism has a specific weakness: whatever the pixel average destroys is
gone before the representation is computed. A fawn glimpsed through a gap in one
shot of thirty contributes ~3 % of that pixel's final intensity and is
indistinguishable from noise.

*Checkpoints:* `alfs_2k_{thermal,rgb}_2k_{64,128,256}_s*`,
`alfs_{thermal,rgb}_1024_64_s*`

### 4.2 Embedded light field — `embfield_*` and `cell_*_embed_multi` — **encode-then-average**

Each of the 31 views is encoded by DINOv3 **independently**, and the per-cell
feature vectors are then averaged. Integration happens in feature space.

The same fawn now contributes a full-strength feature vector to one of thirty
summands. Averaging a discriminative vector with thirty background vectors
attenuates it but does not destroy its direction — which is why this arm
outperforms 4.1 by ~50 % relative on hidden animals at matched width, while
being much closer on visible ones.

Two independent runs exist and agree within noise: the standalone 3-seed sweep
(`embfield_*_embLF_128`, thermal 0.4182) and the grid-controlled 6-seed cell
(`cell_*_embed_multi`, thermal 0.4088). The agreement is evidence that the grid's
preprocessing did not alter the method.

*Checkpoints:* `embfield_{thermal,rgb}_embLF_{64,128}_s*`,
`cell_{thermal,rgb}_embed_multi_embed_multi_s*`

### 4.3b V-JEPA 2.1 as a grid cell (design C, 2026-08-12) — supersedes 4.3

Frozen V-JEPA 2.1 ViT-B, tokens stitched to the same 128x128 grid, PCA to 128,
the same CenterNet head, the same 5 folds. A pure representation swap against
DINOv3, with nothing else varying.

| cell | thermal visible | thermal hidden |
|---|---|---|
| DINOv3 / single | 0.6309 +/- 0.0540 | 0.2125 +/- 0.0924 |
| DINOv3 / multi | 0.7107 +/- 0.0687 | 0.3192 +/- 0.1436 |
| V-JEPA / single | 0.6917 +/- 0.0575 | 0.2436 +/- 0.0935 |
| **V-JEPA / multi** | **0.7503 +/- 0.0652** | **0.3752 +/- 0.1510** |

V-JEPA leads by 1.15-1.18x on hidden animals (5/5 folds in the multi cell) at a
tenth of DINOv3's parameters. **The multi-view gain is representation-independent
-- 1.50x on DINOv3, 1.54x on V-JEPA** -- which is what promotes the aperture
effect from a property of one encoder to a property of the geometry.

The numbers in 4.3 below are from the LoRA/image-space pipeline and are
superseded.

### 4.3 V-JEPA 2 (LoRA and frozen) — temporal, **not geometric**

A ViT-L video encoder consuming a clip of **raw camera frames**, with 144 LoRA
adapters (2.36 M) plus a 3.81 M conv head; the frozen variant trains the head
only (3.81 M) and isolates adaptation from pretrained features.

**Placement.** It is unambiguously embedding-space. On the view axis it is a
genuine edge case, and the honest placement is *multi-frame but not multi-view*:
it integrates many frames, but never registers them to a focus plane, so it does
not exercise the synthetic-aperture mechanism. It is listed in this quadrant with
that qualifier because its temporal context is real and does contribute — but its
result (§5) is the study's evidence that temporal context is **not** a substitute
for geometric alignment.

It also predicts in raw image space rather than the ortho canvas, so **only
recall is comparable**; mAP and precision are not.

*Checkpoints:* `models/vjepa/{matched_thermal,matched_rgb,frozen_ref}.pt`

---

## 5. Single view / embedding space

### 5.1 DINOv3 of the central ortho view — `embed_single`

DINOv3 ViT-H+/16 patch tokens of the *single* orthorectified frame, PCA→128,
on the same 128×128 grid as every other cell. Holds the view count at one and
moves only the representation, so paired with 2.1 it is the **clean
sensor-vs-embedding contrast**, and paired with 4.2 the clean single-vs-multi
contrast *within* embedding space.

*Checkpoints:* `cell_{thermal,rgb}_embed_single_embed_single_s*`

---

## 6. Off-axis: raw camera frames + YOLO26x

`raw_thermal` / `raw_rgb` are trained on unrectified camera frames. They do not
project onto the ortho canvas, so the merged/central GT distinction — which is
defined on that canvas — has no counterpart for them, and they cannot be placed
in the grid or scored on the occlusion axis at all. Their aggregate mAP is
recorded in `RESULTS.md` §5.4 for provenance only.

*Checkpoints:* `models/yolo/{raw_thermal,raw_rgb}/best.pt`

---

## 7. Results — all methods, both populations

IoU 0.5, confidence ≥ 0.3. Hidden population: 212 boxes (thermal), 222 (rgb);
visible ≈900 / ≈890. Ranked by thermal.

### 7.1 Visible boxes

| # | method | quadrant | seeds | thermal | rgb |
|---|---|---|---|---|---|
| 1 | V-JEPA 2 + LoRA | temporal/embed | 1 | **0.9263** | 0.8797 |
| 2 | V-JEPA 2 frozen | temporal/embed | 1 | 0.9159 | — |
| 3 | Embedded light field 128d | multi/embed | 3 | 0.8429 ± 0.0244 | 0.8131 ± 0.0452 |
| 4 | Embedded LF 128d *(grid cell)* | multi/embed | 6 | 0.8407 ± 0.0209 | 0.8181 ± 0.0363 |
| 5 | Ortho render + YOLO26x | single/sensor | 1 | 0.7774 | 0.6927 |
| 6 | DINOv3 central view 128d | single/embed | 6 | 0.7761 ± 0.0230 | 0.7655 ± 0.0278 |
| 7 | Embedded light field 64d | multi/embed | 3 | 0.7753 ± 0.0347 | 0.7774 ± 0.0230 |
| 8 | ALFS render + YOLO26x | multi/sensor | 1 | 0.7464 | 0.6972 |
| 9 | Embedded ALFS 256d | multi/embed | 3 | 0.7417 ± 0.0157 | — |
| 10 | Embedded ALFS 64d | multi/embed | 3 | 0.7203 ± 0.0273 | 0.6555 ± 0.0748 |
| 11 | Embedded ALFS 128d | multi/embed | 3 | 0.7121 ± 0.0486 | — |
| 12 | Embedded ALFS @1024 px 64d | multi/embed | 3 | 0.6770 ± 0.0652 | 0.5043 ± 0.0285 |
| 13 | ALFS render + 0.21 M head | multi/sensor | 6 | 0.5955 ± 0.0686 | 0.4679 ± 0.0413 |
| 14 | Ortho render + 0.21 M head | single/sensor | 6 | 0.4958 ± 0.0464 | 0.4827 ± 0.0302 |
| 15 | PyTorch integrator multi *(control)* | multi/sensor | 6 | 0.3540 ± 0.0564 | 0.4695 ± 0.0393 |
| 16 | PyTorch integrator single *(control)* | single/sensor | 6 | 0.2341 ± 0.0326 | 0.4184 ± 0.0404 |

### 7.2 Occluded (fully hidden) boxes

| # | method | quadrant | seeds | thermal | rgb | Δ rank |
|---|---|---|---|---|---|---|
| 1 | Embedded light field 128d | multi/embed | 3 | **0.4182 ± 0.0571** | **0.4324 ± 0.1252** | +2 |
| 2 | Embedded LF 128d *(grid cell)* | multi/embed | 6 | 0.4088 ± 0.0537 | 0.4219 ± 0.0836 | +2 |
| 3 | Embedded ALFS 256d | multi/embed | 3 | 0.3522 ± 0.0354 | — | +6 |
| 4 | Embedded ALFS 128d | multi/embed | 3 | 0.2783 ± 0.0340 | — | +7 |
| 5 | Embedded light field 64d | multi/embed | 3 | 0.2673 ± 0.0260 | 0.3198 ± 0.0508 | +2 |
| 6 | Embedded ALFS 64d | multi/embed | 3 | 0.2437 ± 0.0268 | 0.3258 ± 0.0806 | +4 |
| 7 | V-JEPA 2 + LoRA | temporal/embed | 1 | 0.2184 | 0.2394 | **−6** |
| 8 | V-JEPA 2 frozen | temporal/embed | 1 | 0.2087 | — | **−6** |
| 9 | Embedded ALFS @1024 px 64d | multi/embed | 3 | 0.1918 ± 0.0526 | 0.1832 ± 0.0069 | +3 |
| 10 | DINOv3 central view 128d | single/embed | 6 | 0.1887 ± 0.0158 | 0.2590 ± 0.0358 | −4 |
| 11 | ALFS render + YOLO26x | multi/sensor | 1 | 0.1845 | 0.3832 | −3 |
| 12 | ALFS render + 0.21 M head | multi/sensor | 6 | 0.1179 ± 0.0248 | 0.1119 ± 0.0234 | +1 |
| 13 | Ortho render + 0.21 M head | single/sensor | 6 | 0.0739 ± 0.0279 | 0.1306 ± 0.0199 | +1 |
| 14 | Ortho render + YOLO26x | single/sensor | 1 | 0.0631 | 0.1729 | **−9** |
| 15 | PyTorch integrator multi *(control)* | multi/sensor | 6 | 0.0464 ± 0.0205 | 0.1029 ± 0.0355 | 0 |
| 16 | PyTorch integrator single *(control)* | single/sensor | 6 | 0.0393 ± 0.0183 | 0.1051 ± 0.0231 | 0 |

---

## 8. Reading the two tables together

**They are close to different experiments.** Visible recall spans 4.0× across
methods (0.23–0.93); hidden recall spans 10.6× (0.039–0.418). Nothing that ranks
methods on overall accuracy would recover the occlusion ordering.

**The disagreement is concentrated in one place: capacity.** The two
highest-capacity models — YOLO26x and V-JEPA 2 — are ranks 5 and 1 on visible
boxes and 14 and 7 on hidden ones. Both are excellent at extracting information
from pixels that imaged an animal, and neither can conjure information from
pixels that did not. This is the study's central asymmetry: **aperture changes
what was imaged; capacity only changes what is extracted from it.**

**Integration order is worth as much as a doubling of width.** Embedded LF at
128d (0.4182) beats embedded ALFS at 256d (0.3522) on hidden animals while using
half the dimensions, and beats embedded ALFS at matched 128d by 50 % relative.
Meanwhile their visible recalls sit within ~0.13 of each other. Whatever the
pixel average destroys, no amount of downstream representation recovers.

**Effects invisible on visible boxes.** Embedded ALFS 128d vs 256d is a null on
visible recall (0.7121 vs 0.7417, within noise) and a 27 % relative gap on hidden
recall. Embedded LF 64d→128d is +9 % visible and +56 % hidden. Any design
decision validated on aggregate accuracy is being validated on the wrong metric.

**Both tables agree on the controls.** The PyTorch integrator cells are last in
both, and the encode-then-average light field is top-4 in both, so this is not
pure re-ordering — the occlusion axis adds information rather than replacing it.

## 9. Caveats attached to these tables

* V-JEPA's visible column is scored against MOT central GT, its own ground-truth
  definition; every other row uses merged-GT boxes that matched a central box.
  Close but not identical sets, so its rank-1 visible position is the softest
  number in either table. Its hidden population is 206 boxes, not 212.
* Seed counts differ: `cell_*` 6, `alfs_2k_*`/`embfield_*` 3, YOLO and V-JEPA a
  single checkpoint. Ranks 3–6 and 7–11 should be read as bands, not an order.
* rgb has no 128d/256d embedded-ALFS runs — blank, not zero.
* The rgb multi/sensor row is a known anomaly: the 0.21 M head shows no
  multi-view gain (0.86×, a null) while YOLO shows 2.22× on identical data. In
  rgb the pixel-space light-field signal needs capacity and resolution that a
  0.21 M head at 128² does not have. See `PART3_RGB.md`.
* All recall at a fixed conf 0.3; precision is not in these tables.
