# BAMBI — multi-view and embedding-space animal detection

**Backup dated 2026-08-05.** Everything below was produced in this project and
is reproducible from the archived code, weights and metrics. Numbers are quoted
from `results_table.json`, which is generated directly from the run directories
(`collect_results.py`) rather than transcribed by hand.

---

## 1. What this compares, and what it does not

The goal is a comparison along two axes:

|                 | sensor space (pixels)      | embedding space (DINOv3)     |
|-----------------|----------------------------|------------------------------|
| **single view** | raw / orthographic frame   | encoded single frame         |
| **multi view**  | ALFS render (pixel mean)   | embedded light field, V-JEPA 2 |

**As of this backup the 2x2 is not yet complete, and the cells that do exist are
not mutually comparable.** They were produced by three different detectors:

| detector | params | input | used for |
|---|---|---|---|
| YOLO26x (ultralytics 8.4.87) | ~60 M | 2048 px images | pixel cells |
| CenterNet-lite (this repo) | 0.28–0.36 M | 128x128xD grids | embedding cells |
| V-JEPA 2 + LoRA + box head | ~300 M backbone | 1024 px clips | temporal cell |

A cross-cell difference therefore confounds *representation* with *head
capacity, input resolution and coordinate frame*. Statements like "pixels beat
embeddings" cannot be supported by the current table. Section 8 sets out the
design that would fix this.

What **is** established, because it was measured within one detector and one
evaluation protocol, is in section 5.

---

## 2. Dataset

Source: `matched_dataset_new` (BAMBI), 77 flights with thermal labels and 74
with rgb, paired thermal/rgb video with per-frame poses, a per-flight DEM and
per-flight fine corrections.

* **Frames.** Labelled "central" frames are multiples of 10. Each central frame
  anchors an aperture of **±45 frames at stride 3** (31 views including the
  centre). Neighbour frames were decoded from `<id>_matched_processed.mp4`
  (left half thermal, right half rgb) into a cache.
* **Split.** The reference split used throughout: flights **10, 211, 212, 213,
  214** held out for test, everything else train. *214 is absent from the
  dataset*, so the effective test set is four flights. This split was supplied
  by the project owner as the one the released detectors used; it has **not**
  been independently verified against those detectors' training manifests.
* **Ground truth.** Two label sets per frame, both produced by the same
  projection and clipping code:
  * `<frame>.txt` — **merged**: one box per track over the whole aperture, so it
    includes animals only some non-central view could see;
  * `<frame>_central.txt` — **central**: only animals visible in the central
    frame itself.
  The difference between them is what section 6 uses to define occlusion.

| modality | labelled centrals | train | val | merged boxes (val) | central boxes (val) |
|---|---|---|---|---|---|
| thermal | 10,459 | 9,772 | 687 | 1,170 | 968 |
| rgb | 9,045 | 8,352 | 693 | 1,163 | 969 |

### 2.1 Renders

| product | resolution | count | failures |
|---|---|---|---|
| ALFS thermal | 2048 px | 11,132 | 0 |
| ALFS rgb | 2048 px | 9,689 | 0 |
| ALFS thermal | 1024 px | 11,132 | 0 (earlier run) |
| ALFS rgb | 1024 px | 9,689 | 0 (earlier run) |

Frame-set parity between 1024 px and 2048 px was verified exact (0 missing,
0 extra) for both modalities, so resolution arms compare identical frames.

**469 of the 11,132 thermal renders are entirely blank** (alpha 0.000
everywhere) because those camera positions fall outside DEM coverage. The
dataset builder already excludes them, so no training or evaluation used them —
but they exist on disk and will be silently skipped by anything that re-derives
the frame list. Flights 156 (279), 17 (166), 159 (24).

### 2.2 Embeddings

DINOv3 ViT-H+/16, PCA-reduced, fp16. The PCA basis is fitted on **training
frames only** and stored beside the cache.

| cache | frames | grid | dims | variance kept | size |
|---|---|---|---|---|---|
| `alfs_thermal` (1024 px renders) | 11,132 | 64x64 | 64 | 91.4 % | 5.5 GB |
| `alfs_2k_thermal` (2048 px renders) | 11,132 | 128x128 | 256 | 97.4 % | 88 GB |
| `alfs_rgb` (1024 px) | 9,689 | 64x64 | 64 | 91.8 % | 4.8 GB |
| `alfs_2k_rgb` (2048 px) | 9,689 | 128x128 | 64 | 91.9 % | 19 GB |
| `srcframes_thermal` (source frames) | 138,220 | 64x64 | 128 | 92.6 % | 135 GB |
| `srcframes_rgb` (source frames) | 119,863 | 64x64 | 128 | 92.9 % | 118 GB |
| `embfield_thermal` (integrated) | 10,663 | 128x128 | 128 | — | 47 GB |
| `embfield_rgb` (integrated) | 9,221 | 128x128 | 128 | — | 37 GB |

PCA components are **nested**, so a K-dim arm is the first K columns of a wider
cache (`--use-dims`). Note that two *separately fitted* bases are not nested:
sklearn selects a randomised SVD at this scale, and the first 64 components of a
128-dim fit differ from a 64-dim fit by ~1.5e-3 relative. All dimensionality
comparisons here truncate a single basis rather than compare separate fits.

---

## 3. Methods

> **`METHODS.md` is the full compendium**: what each of the 16 arms computes,
> why it sits in the quadrant it sits in (including the two genuine edge cases —
> V-JEPA's temporal-but-unregistered frames, and raw-image-space YOLO which
> cannot be placed at all), which checkpoint produced each number, and the
> complete visible/occluded tables for every method on one consistent
> population. The sections below cover only the core arms.

### 3.1 ALFS render (multi view, sensor space)

alfspy's synthetic-aperture integral: every aperture view is projected onto the
DEM through the central frame's orthographic virtual camera and averaged, with
pixels covered by fewer than `alpha_threshold=2` views left transparent.

### 3.2 Embedded light field (multi view, embedding space)

`render_embedding_field.py`. The same integral, but over **DINOv3 features
instead of pixels** — "encode-then-average" rather than "average-then-encode".
Deliberately **GL-free**: embree ray-casts the target grid onto the DEM, world
points are projected into each source view with the renderer's own clip
matrices, and features are gathered with `grid_sample`. It therefore runs inside
a container, where ModernGL has previously produced renders with artifacts.

Two correctness checks, both against the known-good GL path:

* **camera**: `build_virtual_camera` is bit-identical to `AlfsRenderer.virtual_camera` — max|diff| **0.000e+00** on view, projection and position;
* **integral**: with `--channels rgb` it reproduces the geometric ALFS render at **r = 0.9996, MAE 0.012** (normalised).

The modality mask matters: the renderer multiplies every shot by it, including
alpha, so letterbox bars contribute nothing. Omitting it drops the self-test
correlation to ~0.66.

### 3.3 Detection head

CenterNet-style, anchor-free: Gaussian heatmap + width/height + offset, with
penalty-reduced focal loss. A small decoder upsamples the patch grid 4x before
the heads (128x128 -> 512x512), so the reported numbers are not dominated by
patch stride. 0.28 M parameters at 64 input dims, 0.36 M at 256.

Metrics come from ultralytics' `ap_per_class`, the same implementation the YOLO
baselines use. Fitness for early stopping is 0.1·mAP50 + 0.9·mAP50-95.

---

## 4. Reproduction

```bash
# 1. render (workstation; needs a real GL driver)
py -m georef.cli --config config_2k.yaml --flight-ids all \
    --modalities thermal --mode alfs --no-extract-neighbors

# 2. labels (same split/clip path for every arm)
py scripts/build_yolo_datasets.py --config config_2k.yaml \
    --out D:/yolo_datasets_2k --sources alfs --modalities thermal \
    --split-mode official --splits-json bambi_splits_reference.json \
    --val-splits test --labels-only

# 3. stage + encode (DGX)
docker/stage_to_dgx.sh thermal
SOURCE=alfs_2k docker/encode_dgx.sh thermal 4 256      # renders
SOURCE=srcframes docker/encode_dgx.sh thermal 4 128    # source frames

# 4. embedded light field
docker/integrate_dgx.sh thermal 4

# 5. train + evaluate
MOD=thermal docker/multiseed_dgx.sh
python scripts/occlusion_recall.py --labels-dataset alfs_2k_thermal \
    --arms alfs_2k_thermal_2k_64:alfs_2k_thermal,embfield_thermal_embLF_128:embfield_thermal
```

Container: `bambi-embed:1.5`, base `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`.
Dependencies are pinned to the workstation environment (`.venv-neural`) so
embeddings computed on either machine are comparable — notably numpy 2.4.4,
transformers 5.14.1, trimesh 4.6.6 with embreex, opencv-python-headless
5.0.0.93. `dgx_run.sh` refuses to start on a GPU another user occupies.

---

## 5. Results

All embedding-detector numbers are **3 seeds (1337, 7, 42)**, mean ± sample
stdev, evaluated against the **central** GT so they are like-for-like with the
pixel baselines. A difference smaller than ~2x the larger stdev is not
separable at n=3.

### 5.1 Render resolution (within the embedding detector)

| modality | 1024 px | 2048 px | change |
|---|---|---|---|
| thermal | 0.2425 ± 0.0238 | **0.3530 ± 0.0047** | **+46 %**, 4.6σ |
| rgb | 0.1386 ± 0.0124 | **0.2278 ± 0.0226** | **+64 %**, 3.9σ |

*(mAP50-95, 64 dims both sides, identical frames and GT.)*

**Established.** Replicated on two modalities. The gain is concentrated at
strict IoU, which is the signature of better localisation — consistent with the
mechanism: at 2048 px a patch covers 0.55 m of ground instead of 1.09 m, so an
animal spans ~2.4 cells instead of ~1.2.

### 5.2 PCA width

| dims | thermal mAP50-95 |
|---|---|
| 64 | 0.3530 ± 0.0047 |
| 128 | 0.3483 ± 0.0104 |
| 256 | 0.3545 ± 0.0154 |

**No effect on aggregate mAP** — the spread is smaller than one run's stdev,
and it is non-monotonic, which is what noise looks like. But see 6.2: width
*does* matter on the occluded subset, which the mean hides.

### 5.3 Embedded light field vs ALFS render

| modality | arm | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| thermal | ALFS 2k, 64d | 0.6845 ± 0.0055 | 0.3530 ± 0.0047 | 0.7751 ± 0.0107 | 0.6343 ± 0.0092 |
| thermal | **embLF, 128d** | **0.7367 ± 0.0229** | 0.3590 ± 0.0127 | 0.7387 ± 0.0606 | **0.7585 ± 0.0249** |
| thermal | embLF, 64d | 0.7325 ± 0.0118 | 0.3624 ± 0.0308 | 0.7744 ± 0.0249 | 0.7047 ± 0.0214 |
| rgb | ALFS 2k, 64d | 0.5742 ± 0.0294 | 0.2278 ± 0.0226 | — | — |
| rgb | **embLF, 128d** | **0.7116 ± 0.0089** | **0.2816 ± 0.0047** | — | — |
| rgb | embLF, 64d | 0.6719 ± 0.0471 | 0.2574 ± 0.0334 | — | — |

**Established.** Averaging features beats averaging pixels:

* thermal: **+7.0 recall points at identical precision** (0.7744 vs 0.7751 P;
  0.7047 vs 0.6343 R), i.e. mAP50 +7 % with mAP50-95 unchanged — more animals
  found, localised no better;
* rgb: **+24 % on both mAP50 and mAP50-95** — here localisation improves too.

### 5.4 Pixel detectors (YOLO26x, single seed, reference split)

Not comparable with 5.1–5.3 (different head — see section 1). Recorded for
provenance.

| arm | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| raw_thermal | 0.8823 | 0.7247 | 0.8903 | 0.8052 |
| raw_rgb | 0.8180 | 0.5282 | 0.8275 | 0.8018 |
| ortho_thermal | 0.8743 | 0.6609 | 0.8882 | 0.7789 |
| ortho_rgb | 0.7937 | 0.5149 | 0.8484 | 0.7389 |
| alfs_thermal | 0.7639 | 0.5230 | 0.8846 | 0.6504 |
| alfs_rgb | 0.6466 | 0.4219 | 0.8414 | 0.6096 |
| alfs_thermal vs central GT | 0.7812 | 0.4018 | 0.8757 | 0.6777 |
| alfs_rgb vs central GT | 0.5881 | 0.2289 | 0.7451 | 0.6037 |
| **released** ortho_thermal | 0.9938 | 0.8429 | 0.9778 | 0.9824 |
| **released** ortho_rgb | 0.9332 | 0.6119 | 0.8933 | 0.8901 |

> **Read this table by row, never across rows.** Each `ortho_*` row is scored
> against its own labels (= central GT) and each `alfs_*` row against its own
> (= merged GT), so `ortho_thermal 0.6609` and `alfs_thermal 0.5230` are not
> comparable, and the "vs central GT" rows are worse: central GT is the
> *single-view* truth, so it scores every correctly recovered hidden animal as a
> false positive — a penalty proportional to how well the method does the thing
> being measured. Scored against the same merged GT (Part 2):
>
> | | ortho | alfs | ratio |
> |---|---|---|---|
> | thermal mAP50-95 | 0.2654 | 0.5308 | **2.00x** |
> | rgb mAP50-95 | 0.2064 | 0.4201 | **2.04x** |
>
> The light field does not degrade YOLO; it doubles it. (My AP harness gives
> ortho/central 0.6661 against the 0.6609 recorded here — a ~0.005 offset from
> NMS/matching details, which is the gate that makes the rest of the table
> trustworthy.)

The released detectors' numbers are **not trustworthy as a baseline**: if they
were trained on all flights except the five held out, then these four test
flights were in their training set for every flight the split does not exclude —
and their near-perfect scores are consistent with that. Treat as an upper bound
under probable contamination, not as a comparison point.

### 5.5 V-JEPA 2 (external, `vjepa2_backup_2026-08-02`)

Same dataset, same held-out flights, same aperture definition, same
`ap_per_class` implementation.

| metric | thermal | rgb |
|---|---|---|
| AP@0.25 | 0.8925 | 0.8573 |
| AP@0.50 | 0.8423 | 0.7696 |
| mAP50-95 | 0.4001 | 0.3042 |

**Two reasons these cannot be placed in the 2x2 as they stand:**

1. **Different detector and evaluation set** — LoRA-adapted V-JEPA 2 with a box
   head, scored over 2,580 frames / 3,256 GT boxes in *image* coordinates,
   against this project's 687 frames / 968 boxes in the *orthographic* frame.
2. **Its ground truth structurally excludes occlusion.** `prepare_matched.py`
   copies the raw MOT file as GT, and MOT annotates only what is visible in each
   frame. An animal hidden in the central frame has no box, so a model that
   recovered it from neighbouring frames would be **penalised as a false
   positive**. V-JEPA 2 therefore cannot demonstrate an occlusion benefit under
   its current protocol, whatever its true capability.

---

## 6. Occlusion analysis

`occlusion_recall.py`. A merged-GT box that matches no central-GT box (IoU 0.5)
is an animal **the central view did not show**; recall on that subset is the
quantity of interest and recall on the matched subset is the control. Fixed
operating point: IoU 0.5, confidence ≥ 0.3, so arms are compared at the same
threshold rather than each at its own best.

### 6.1 thermal (268 occluded boxes)

| arm | R visible | R occluded |
|---|---|---|
| ALFS 2k, 64d | 0.7203 ± 0.0273 | 0.3060 ± 0.0261 |
| ALFS 2k, 128d | 0.7121 ± 0.0486 | 0.3221 ± 0.0334 |
| embLF, 64d | 0.7753 ± 0.0347 | 0.3408 ± 0.0141 |
| **embLF, 128d** | **0.8429 ± 0.0244** | **0.4888 ± 0.0454** |

embLF(128d) over ALFS(128d): **+18 % relative on visible, +52 % on occluded** —
the advantage is ~2.9x larger where the aperture is supposed to help.

### 6.2 rgb (266 occluded boxes)

| arm | R visible | R occluded |
|---|---|---|
| ALFS 2k, 64d | 0.6555 ± 0.0748 | 0.3784 ± 0.0735 |
| embLF, 64d | 0.7774 ± 0.0230 | 0.3885 ± 0.0544 |
| **embLF, 128d** | **0.8131 ± 0.0452** | 0.4787 ± 0.1039 |

embLF(128d) over ALFS(64d): **+24 % relative on visible, +27 % on occluded** —
*no* concentration, and the occluded difference is **not separable** (±0.1039).

~~**The occlusion-specific claim holds on thermal and does not replicate on
rgb.**~~ **Superseded — it does replicate.** Restricted to genuinely hidden
animals and with the real renderer in the sensor row, rgb embed/multi reaches
0.4219 ± 0.0836 at 52 % retention (thermal: 0.4088, 49 %). See
`PART3_RGB.md`.

### 6.3 PCA width, revisited

On the occluded subset thermal embLF gets **0.4888 (128d) vs 0.3408 (64d)**,
3.3σ — while the ALFS arms show no such gap (0.3221 vs 0.3060, n.s.). Occluded
animals are ~23 % of val boxes, so this is invisible in the aggregate mAP of
5.2. **A representation choice that looks free on the mean is not free on the
cases the method exists for.**

### 6.3b All methods re-scored on the hidden population

6.1-6.2 above are on the **proxy** population (268/266 boxes). Re-run on the
hidden population so every method in the study is finally like-for-like
(thermal 212 / rgb 222):

| method | order | dims | seeds | thermal vis -> hidden | rgb vis -> hidden |
|---|---|---|---|---|---|
| Embedded ALFS | avg->enc | 64 | 3 | 0.7203 -> 0.2437 | 0.6555 -> 0.3258 |
| Embedded ALFS | avg->enc | 128 | 3 | 0.7121 -> 0.2783 | - |
| Embedded ALFS | avg->enc | 256 | 3 | 0.7417 -> 0.3522 | - |
| Embedded ALFS @1024px | avg->enc | 64 | 3 | 0.6770 -> 0.1918 | 0.5043 -> 0.1832 |
| Embedded light field | enc->avg | 64 | 3 | 0.7753 -> 0.2673 | 0.7774 -> 0.3198 |
| Embedded light field | enc->avg | 128 | 3 | 0.8429 -> **0.4182** | 0.8131 -> **0.4324** |
| Embedded LF (grid cell) | enc->avg | 128 | 6 | 0.8407 -> 0.4088 | 0.8181 -> 0.4219 |

**Encode-then-average beats average-then-encode.** At matched 128d on thermal,
0.4182 vs 0.2783 — a 50 % relative gap on hidden animals, against a much smaller
gap on visible ones (0.8429 vs 0.7121). Integrating *pixels* destroys evidence
that survives integration in *feature* space: a fawn glimpsed through a gap in
one view averages away against canopy in the render, while its feature vector
still carries signal into the mean. Embedded ALFS needs 256 dims (0.3522) to
approach what the light field reaches at 128.

The embedded light field appears twice — standalone 3-seed (0.4182) and
grid-controlled 6-seed (0.4088) — agreeing within noise, i.e. the grid's
preprocessing did not change the method.

Resolution matters here too: embedded ALFS falls 0.2437 -> 0.1918 (thermal) and
0.3258 -> 0.1832 (rgb) going from 2048 to 1024 px.

### 6.4 Superseded: "occluded" was three populations, not one

Everything in 6.1–6.3 uses the **proxy** definition — a merged-GT box with no
central counterpart. That conflates three cases, and separating them changed the
conclusions. Detail in `PART1_VISIBLE_VS_OCCLUDED.md`; summary:

* **hidden** — invisible centrally *and* the central view imaged that ground.
  79.1 % of thermal, 83.5 % of rgb. This is the occlusion population.
  **Those two figures are from the early small sample** (rgb: 222 of 266 boxes,
  `PART3_YOLO_RGB.md`). Over all flights the shares are **73.4 %** thermal
  (10,046 of 13,688 centrally invisible boxes) and **77.8 %** rgb (8,771 of
  11,275), recomputed from `zenodo_labels/<mod>_hidden_mask_all.json`. The
  all-flights figures are the ones the paper quotes.
* **wider footprint** — the central view never covered it. Recovering these is
  real but says nothing about occlusion. Excluded.
* **reviewed** — annotated centrally but flagged partially hidden by a human.
  A different question entirely, reported separately.

A bug in the hidden branch restricted the *visible* control to frames containing
a hidden animal. Fixed; it moved raw/single visible recall 0.2758 → 0.4958. The
biased figure was itself informative and is now reported as the frame-level
population.

### 6.5 The grid on genuinely hidden animals (thermal, 212 boxes, 6 seeds)

| | single view | multi view | multi/single |
|---|---|---|---|
| raw (pixels, real renderer) | 0.0739 ± 0.0279 | 0.1179 ± 0.0248 | 1.60x |
| embeddings (DINOv3) | 0.1887 ± 0.0158 | **0.4088 ± 0.0537** | 2.17x |
| embed/raw | 2.55x | 3.47x | |

Retention — hidden recall as a fraction of the same cell's visible recall — is
the specificity test: 15 % / 20 % / 24 % for the first three cells, **49 %** for
embed/multi. The advantage is occlusion-specific, not a general accuracy offset.

### 6.6 Partial occlusion is a different phenomenon

Reviewed (human-flagged) partial occlusion, ratio of occluded to clear recall:

| method | thermal | rgb |
|---|---|---|
| raw / single | 1.21x | 0.70x |
| raw / multi | 1.02x | 0.69x |
| embed / single | 0.96x | — |
| embed / multi | 0.98x | — |
| YOLO single / multi | 1.03x / 1.12x | 0.70x / 0.69x |

Partial occlusion is **free in thermal and costs ~30 % in rgb for pixel-space
methods**, and multi-view does not help it in either. Adding the embedding cells
narrows this further — they pay nothing in either modality (rgb 1.03–1.07x):

| | thermal | rgb |
|---|---|---|
| pixel cells | free (0.98–1.21x) | costly (0.69–0.91x) |
| embedding cells | free (0.96–0.98x) | free (1.03–1.07x) |

So the asymmetry lives in the **representation**, not the sensor: raw rgb pixels
of a half-covered animal are ambiguous where DINOv3 features stay
discriminative.

The claim to make is therefore "compensates **total** occlusion", not
"handles occlusion".

### 6.7 Clutter is a separate cost that embeddings absorb

Recall on animals that are plainly visible, in frames that contain an occluder,
against the same cell's overall visible recall (thermal):

| cell | all frames | occluder frames | cost |
|---|---|---|---|
| raw / single | 0.4958 | 0.2758 | −44 % |
| raw / multi | 0.5955 | 0.3931 | −34 % |
| embed / single | 0.7761 | 0.7271 | −6 % |
| embed / multi | 0.8407 | 0.8215 | −2 % |

Part of what reads as "sees through canopy" is really "is robust to canopy".

### 6.8 Capacity (YOLO26x) — `PART2_YOLO_THERMAL.md`, `PART3_YOLO_RGB.md`

The worry that the finding was an artifact of a 0.28 M head is refuted: the
multi-view gain on hidden animals is **larger** with YOLO (2.92x thermal,
2.22x rgb) than with the small head (1.60x).

| | small head | YOLO26x |
|---|---|---|
| visible recall, single view | 0.4958 | 0.7774 |
| hidden recall, single view | 0.0739 | 0.0631 |

**Capacity buys performance on what is visible and nothing on what is not.**

### 6.9 V-JEPA 2 — `PART4_VJEPA.md`

Image space, so recall only. Thermal: visible **0.9263** (the best visible-animal
result in the study), hidden 0.2184 — 24 % retention, level with single-view
DINOv3 and far below embed/multi's 49 %. Frozen vs LoRA is a null on occlusion
(0.2087 vs 0.2184). A 300 M video model with temporal context does not reach
what geometric aperture integration gives a 0.28 M head.

---

### 6.10 INSID3 positional debiasing: small, asymmetric, not adopted

Dense transformer tokens carry a component encoding absolute patch position.
INSID3 removes it by estimating that subspace from content-free input and
projecting onto its orthogonal complement (`F~ = F(I - BB^T)`, rank 32).

Two premise measurements looked encouraging. The subspace is sharply low-rank --
1 of 1280 directions holds 44.9 % of the DINOv3 noise response, and for V-JEPA a
single direction holds 89.5 % of 768. And real tokens inside animal boxes put
*less* energy in it than background does (0.2673 vs 0.4810 at rank 32, ratio
0.56), so the positional component is disproportionately background.

Applied end to end across **four encoder x modality pairs** (120 runs, 5 folds x
3 seeds each, baseline and debiased scored in the same pass on the same folds):

| pair | single cell | multi cell |
|---|---|---|
| DINOv3 / thermal | recall up, precision down | 8/8 positive; precision p<0.02 |
| DINOv3 / rgb | 7/8 positive | 8/8 positive; **AP50:95 +16.8 %, p<0.05** |
| V-JEPA / thermal | **recall -10.0 %, p<0.02** (0/5); precision +3.0 %, p<0.01 | 6/8 positive, none significant |
| V-JEPA / rgb | **8/8 negative**; AP50 -12.0 %, p<0.1 | 6/8 positive, none significant |

**The asymmetry across all 64 comparisons: multi-view cells 28/32 positive,
single-view cells 11/32.** Every reliable positive is in a multi cell; every
reliable negative is in a single cell.

Plausible mechanism: the multi-view cell averages features from 31 views and a
world point lands at a different patch position in each, so the positional
component does not cancel under integration. Removing it beforehand should help
where features from different grid positions get mixed -- which is exactly the
multi cell. The single cell has no such mixing.

**Not adopted.** Effects are 1-3 %; only 2 of 64 comparisons reach p<0.05 and
none survives Bonferroni. All reported numbers use undebiased features. Full
record: `DEBIAS_STUDY.md`. Metrics: `metrics/debias32_*.json`,
`metrics/vjdebias32_*.json`.

**Third proxy that misled.** A low-rank subspace that sits mostly on background
does not imply that deleting it helps a detector: the head is trained on whatever
representation it is given, and a linear component this predictable is one it can
already discount.

---

### 6.11 Integration order at parity: it does not matter much

The draft describes two approaches — "create the ALFS render, then embed it" and
"embed the frames, then build the embedded light field". Until now only the
second existed as a cross-validated cell; the first sat at 64x64 / 64 dims on the
superseded single split. Both now run at full parity: 128x128 grid, 128 PCA dims,
same head, same 5 folds, same 3 seeds, same frames.

Two facts made the baseline arm free rather than a rebuild. PCA components are
nested, so the first 128 channels of the existing 256-dim `alfs_2k_thermal` store
ARE the 128-dim PCA (`--use-dims` truncates on read; the scorer reads `use_dims`
back out of each run's summary.json so evaluation truncates identically). And
both stores cover the same 52,295 labelled stems -- the 469 extra ALFS frames
carry no labels, so the two orders see identical data.

| arm | vis recall | hidden recall | hidden prec | hidden AP50 |
|---|---|---|---|---|
| encode-then-average (`embed_multi`) | **0.7107 ± 0.0687** | **0.3192 ± 0.1436** | 0.2059 | **0.1235** |
| average-then-encode (`alfsembed_multi`) | 0.6705 ± 0.0602 | 0.2815 ± 0.1091 | 0.2087 | 0.1158 |
| average-then-encode, debiased | 0.6810 ± 0.0653 | 0.2886 ± 0.1144 | 0.2062 | 0.1191 |

**Not separable at n = 5** -- nothing reaches p < 0.10. Direction favours
encode-then-average on recall (visible ratio 0.943, t = -1.93; hidden ratio
0.882, t = -1.42); precision and AP50:95 marginally favour average-then-encode at
ratios 1.01-1.02.

This matters for the paper's framing: the two approaches are **near-equivalent
detectors**, so the multi-view benefit reported throughout comes from aperture
integration itself, not from where in the pipeline the integration happens.

Debiasing this arm behaves like every other multi-view cell -- hidden recall
x1.025, AP50 x1.028, AP50:95 x1.043, each winning 4/5 folds, none significant.

**Fold 0 misled again.** It showed average-then-encode winning every metric; over
five folds the recall direction reverses. Second time in this study a fold-0 read
would have produced the wrong headline. Metrics:
`metrics/alfsembed_thermal_f{0..4}.json`. Open: rgb, and the single-view
analogue (`project -> encode`).

---

### 6.12 Aperture in pixel space, cross-validated (rgb)

The pixel-space row of the 2x2 on the five-fold protocol: the same YOLO26x
trained per arm, single ortho frame vs the 31-view ALFS integral. 1024 px,
batch 4, 100 epochs (patience 10), 5 folds x 3 seeds = 30 runs.

| arm | vis recall | hidden recall | hidden prec | hidden AP50 |
|---|---|---|---|---|
| `ortho_rgb` (single) | 0.5233 ± 0.0612 | 0.1273 ± 0.0620 | 0.1986 | 0.0529 |
| `alfs_rgb` (multi) | 0.5208 ± 0.0911 | **0.2159 ± 0.1015** | **0.3229** | **0.1434** |

| metric | ratio | t(4) | p | folds won |
|---|---|---|---|---|
| visible recall | 0.995 | -0.10 | ns | 2/5 |
| **hidden recall** | **1.696** | 3.20 | **< .05** | **5/5** |
| **hidden precision** | **1.626** | 4.56 | **< .02** | **5/5** |
| hidden AP50 | 2.710 | 5.70 | < .01 | 5/5 |
| hidden AP50:95 | 4.714 | 5.30 | < .01 | 5/5 |

**Visible recall is unchanged while hidden recall rises 1.70x, winning every
fold.** That is the cleanest available shape for the central claim: the aperture
buys occluded animals specifically, not a general improvement -- and with an
off-the-shelf detector and no foundation model, so it is a property of the
imagery rather than of the representation.

**Discount the AP columns.** These are scored against merged ground truth, the
hull of a track's boxes over every contributing frame, and that annotation is not
neutral between the arms: an ALFS integral smears a moving animal along its
trajectory and matches the hull well, while a single ortho frame shows one
instant and overlaps it poorly. The visible AP50:95 gain of 1.715x at t=12.71 is
almost certainly this effect, since visible recall did not move at all. Recall
and precision on hidden animals are the defensible figures -- presence is far
less sensitive to box geometry. Metrics: `metrics/yolocv_rgb_f{0..4}.json`.

**Thermal replicates it, more strongly.**

| arm | vis recall | hidden recall | hidden prec | hidden AP50 |
|---|---|---|---|---|
| `ortho_thermal` | 0.7178 ± 0.0644 | 0.1611 ± 0.0547 | 0.1947 | 0.0625 |
| `alfs_thermal` | 0.7156 ± 0.0580 | **0.3293 ± 0.1361** | **0.3135** | **0.2065** |

| metric | ratio | t(4) | p | folds won |
|---|---|---|---|---|
| visible recall | 0.997 | -0.12 | ns | 2/5 |
| **hidden recall** | **2.044** | 4.39 | **< .02** | **5/5** |
| **hidden precision** | **1.610** | 6.32 | **< .01** | **5/5** |
| hidden AP50 | 3.306 | 4.10 | < .02 | 5/5 |
| hidden AP50:95 | 5.687 | 3.99 | < .02 | 5/5 |

### Both modalities, side by side

| modality | visible recall | hidden recall | hidden precision |
|---|---|---|---|
| rgb | x0.995 (ns) | **x1.696** (p<.05, 5/5) | x1.626 (p<.02, 5/5) |
| thermal | x0.997 (ns) | **x2.044** (p<.02, 5/5) | x1.610 (p<.01, 5/5) |

Two sensors, independently, the same shape: **no change on visible animals,
1.7-2.0x on occluded ones, every fold in both.** This is the study's central
claim on the protocol that survives scrutiny, obtained with an off-the-shelf
detector and no foundation model -- so it is a property of the imagery rather
than of a representation we chose.

Metrics: `metrics/yolocv_{rgb,thermal}_f{0..4}.json`. The full 60-run sweep ran
30 rgb on A100s and 30 thermal on dgx1 V100s, split by arm so no paired
comparison straddles hardware (V100 measured 1.32x slower per epoch, same
ultralytics and torch build shipped to both machines).

---

### 6.13 An external detector disagrees -- and that is the point

Microsoft's OWL (Overhead Wildlife Locator, arXiv 2606.13911) is an independent,
published, point-based aerial wildlife detector. Three released variants were run
zero-shot over both rgb arms on all five folds. Full record: `OWL_BASELINE.md`.

| model / arm | vis recall | hidden recall | hidden prec | hidden F1 |
|---|---|---|---|---|
| OWL-C / ortho | 0.2452 | 0.1481 | 0.2542 | 0.1814 |
| OWL-C / alfs | 0.1840 | 0.0887 | 0.1836 | 0.1154 |
| OWL-T / ortho | 0.2131 | 0.1401 | 0.2589 | 0.1745 |
| OWL-T / alfs | 0.1536 | 0.0800 | 0.1626 | 0.1044 |
| **OWL-D / ortho** | **0.5128** | **0.3265** | **0.4246** | **0.3631** |
| OWL-D / alfs | 0.4788 | 0.2643 | 0.3317 | 0.2906 |

**The aperture hurts every zero-shot model** -- OWL-D hidden recall x0.809
(p<.05), precision x0.781 (p<.01), F1 x0.801 (p<.02), losing 0/5 folds. OWL-C and
OWL-T drop harder still (x0.60, x0.57).

That is the exact opposite of 6.12, and the two together isolate the cause with
an internal control: **the same ALFS images that a zero-shot OWL does worse on
give a trained YOLO a 70% gain.** The imagery is not degraded; the integral image
is out of distribution for a detector trained on conventional aerial photographs.

**Synthetic-aperture integration is not free preprocessing that can be bolted
onto an off-the-shelf detector -- the benefit requires training on aperture
imagery.** This says *when* the method pays off, and explains why a practitioner
running an existing detector over ALFS renders would wrongly conclude it fails.

Secondary: OWL-D is far the strongest zero-shot (0.3265 vs 0.148/0.140), and its
frozen backbone is DINOv3 ViT-H+/16 -- the same one our embedding cells use, so
an independent group corroborates that choice.

Scored point-in-box (a detection hits when its point lies inside a GT box),
one-to-one greedy by confidence, populations and ignore rule imported from
`occlusion_metrics.py`. **Point-in-box is more permissive than IoU 0.5**, so these
numbers are comparable with each other but not with the IoU tables above; the
companion pass re-scoring our detectors under the same rule is outstanding.

### 6.14 Fine-tuning OWL-D: penalty removed, benefit absent

OWL-D was fine-tuned from the released checkpoint on each fold, both arms,
backbone frozen as published, then evaluated identically to zero-shot.

| detector | what adapts | hidden recall alfs/ortho | p | folds won |
|---|---|---|---|---|
| OWL-D zero-shot | nothing | **x0.809** hurts | < .05 | 0/5 |
| OWL-D fine-tuned | decoder only, backbone frozen | **x0.985** neutral | ns | 2/5 |
| YOLO26x | end to end | **x1.696** helps | < .05 | 5/5 |

The zero-shot deficit disappears, so that part **was** unfamiliarity. But the
aperture gain never arrives: x0.985 is exactly neutral, and precision on hidden
animals is significantly worse on ALFS (x0.744, p<0.01, 0/5).

**The explanation we proposed did not hold.** The natural reading was that only
the DPT decoder adapted (the DINOv3 backbone stayed frozen, as published) while
YOLO26x adapts end to end -- so the aperture benefit might need the
*representation* to adapt. An unfrozen-backbone probe on two folds says
otherwise: the ratio went **down**, from 1.281 frozen to 1.041 unfrozen on those
folds. Unfreezing improved the model overall but improved the ortho arm more.

That probe is not clean -- batch dropped 16->4 and epochs 10->3 to fit 840 M
trainable parameters in 41 GB, so "unfrozen" is confounded with "less training",
and folds 0 and 3 happen to be the two most favourable to ALFS (frozen ratio
1.281 there against 0.985 over all five). We chose not to run the frozen-at-batch-4
control. So: **no evidence that unfreezing recovers the gain, and the trend runs
against it**; the frozen backbone is not demonstrably the cause.

Still untested: point/density supervision vs box regression, patch training vs
full frames, and the DPT decoder itself.

Incidentally this fills the `project -> encode` single-view gap noted in
`DEBIAS_STUDY.md` 9.1: frozen DINOv3 plus a trained head, on ortho vs ALFS pixel
images, is equivalent for hidden recall.

Full record: `OWL_BASELINE.md`. Metrics: `metrics/owlft_rgb_f{0..4}.json`,
`metrics/owlunfroz_rgb_f{0,3}.json`.

---


---

## 7. Defects found and fixed

Recorded because several were silent, and because they bound how much to trust
earlier numbers.

| defect | consequence | status |
|---|---|---|
| alfspy `render_integral` used `np.divide(where=)` without `out=` | sub-threshold pixels were uninitialised memory | fixed (`render_integral_clean`) |
| `world_to_pixel_coord` only broadcasts for exactly 4 points | label projection wrong for N≠4 | replaced by `project_to_image`, bit-identical where upstream works |
| orthographic `pixel_to_world_coord` crashed, wrong scale, wrong handedness | integrator geometry | patched; verified 70.0000 m span, 0.0000 px round-trip |
| out-of-bounds YOLO labels silently dropped at load | lost boxes | clipped at write time |
| `ap_per_class` unpacked positionally — index 6 is `unique_classes`, not `ap` | wrong mAP | named unpacking + known-answer test |
| two render pipelines ran concurrently on the same output | 1 of 72 PNGs truncated | detected, all output discarded and re-rendered |
| DGX frame tree missing 587 thermal / 408 rgb aperture frames | embedded LF would silently cover fewer frames than ALFS | staged; coverage now 11,132/11,132 and 9,689/9,689 |
| video required as an ALFS essential even with `extract_neighbors=false` | every flight rejected on a machine without source videos | required only when decoding |
| PCA basis refitted per shard under multi-GPU | shards would land in different linear spaces | guard: workers refuse to fit; `--fit-only` first |
| `--pca-dim` ignored when a cached basis existed | "PCA-128" run would silently reuse a 64-dim basis | dimension mismatch is now an error |
| ultralytics dataloader deadlock on Windows | 6.5 h silent hang | `workers=0` on val, per-model subprocess isolation |

**Three proxies that misled, all now recorded in the code:**

1. **PCA reconstruction error.** Animal patches reconstruct 43–81 % worse than
   background at *every* width, and the gap widens with more dims. This is real
   and reproducible (`pca_cost_probe.json`) — and it does **not** predict
   detection accuracy. It was the basis for encoding at 256 dims; the AP data
   showed 64 was enough.
2. **Aggregate mAP.** It showed dimensionality as a null result, which was the
   basis for encoding rgb source frames at 64 dims. The occluded subset showed
   otherwise (6.3), and that encode was redone at 128.
3. **Subspace energy.** The positional subspace is low-rank and sits mostly on
   background (ratio 0.56), which predicted that removing it would help. It did
   not (6.10). A linear component that predictable is one the head can already
   discount, so deleting it in advance saves an adjustment it was making anyway.
   Measuring a property of the representation does not establish that acting on
   it changes the detector.

---

## 8. What would make the 2x2 valid

Every cell through **one integrator and one head**, varying only the two axes:

| | `--channels rgb` (3ch) | `--channels feat` (128ch) |
|---|---|---|
| `--aperture single` | single view + sensor | single view + embedding |
| `--aperture full` | multi view + sensor | multi view + embedding |

All four then share the orthographic camera, the 128x128 grid, the labels, the
head and the seeds, so a difference is attributable to the axis under study. The
pixel path is already validated (r = 0.9996 vs the real renderer) and
`--aperture` is implemented. Estimated cost ~6–8 h on 4 A100s.

For V-JEPA 2 to join as a fifth cell it needs **merged-aperture GT in image
coordinates** — projecting each track's box from the views that do see it into
the central frame — otherwise the occluded case remains undefined for it.

---

## 9. Caveats

* **n = 3 seeds.** Differences under ~0.03 mAP50-95 are not separable. The
  occluded subsets (268 / 266 boxes) are thinner still.
* **"Occluded" is a proxy.** It means "in the merged GT but not the central GT",
  which conflates true occlusion with animals outside the central frame's
  footprint. The two have not been separated.
* **Early stopping is aggressive.** Arms peak between epochs 4 and 17 and then
  degrade; these are early-stopped snapshots of a small head.
* **The split is assumed, not verified** against the released detectors.
* **Flight 214** is absent, so the test set is four flights, not five.
* **Pixel baselines are single-seed**, embedding arms are 3-seed.
* The 1024 px arms used a separately fitted PCA basis (~1.5e-3 from a
  truncation), negligible against the resolution effect but not zero.

---

## 10. Archive contents

```
results_2026-08-05.tar.gz     34 trained heads (best.pt) + summaries + PCA bases
logs_2026-08-05.tar.gz        all run logs, docker/, code/scripts, code/src
results_table.json            every arm, aggregated, generated from run dirs
RESULTS.md                    this document
```

Not archived (regenerable, ~430 GB): embedding caches, integrated fields,
renders. Their provenance is section 2.
