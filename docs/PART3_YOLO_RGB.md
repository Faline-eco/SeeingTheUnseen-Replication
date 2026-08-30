# Part 3a — YOLO on rgb

Measured 2026-08-07. Same protocol as Part 2: YOLO26x, 100 epochs, imgsz 1024,
one checkpoint per arm, IoU 0.5, conf 0.3. `ortho_rgb` = raw/single,
`alfs_rgb` = raw/multi.

This is the **YOLO half** of Part 3. The embedding half still needs a real 2048
ortho render for rgb; the rgb sensor row currently uses the PyTorch integrator.

The rgb hidden mask did not exist and was generated first
(`decompose_invisible.py --modality rgb`): of 266 invisible boxes, **222 (83.5 %)
are genuinely hidden**, 21 straddle the coverage edge, 23 are wider-footprint.
Mean single-view coverage of the ortho canvas is 29.3 %.

---

## Recall on hidden animals (n = 214)

| arm | R visible | R hidden | retained |
|---|---|---|---|
| raw / single (ortho) | 0.6927 | 0.1729 | 25 % |
| raw / multi (alfs) | 0.6972 | 0.3832 | 55 % |

Multi-view gain: **2.22×** on hidden animals, with visible recall identical
(0.6927 vs 0.6972, a 0.6 % difference). The same signature as thermal — the
light field contributes on hidden animals and nowhere else.

## mAP, both arms against both ground truths

| model | GT | mAP50 | mAP50-95 |
|---|---|---|---|
| ortho (single) | central | 0.7926 | **0.5127** |
| ortho (single) | merged | 0.6153 | 0.2064 |
| alfs (multi) | central | 0.5762 | **0.2193** |
| alfs (multi) | merged | 0.6446 | 0.4201 |

The same trap as thermal, and the same size. Comparing the bolded cells (each
model against its own view's GT) says the light field halves performance;
comparing against the neutral merged GT says it doubles it:
**alfs 0.4201 vs ortho 0.2064 = 2.04×**, against thermal's 2.00×. That the ratio
lands on 2× in both modalities independently is the strongest indication that it
is a property of the geometry rather than of either training run.

## Precision and localization

| arm | dets | TP | precision | R@0.50 | R@0.70 | R@0.80 | R@0.90 | mean IoU of found GT |
|---|---|---|---|---|---|---|---|---|
| raw / single | 759 | 648 | **0.8538** | 0.5675 | 0.2614 | 0.0989 | 0.0155 | 0.6977 |
| raw / multi | 871 | 724 | 0.8312 | 0.6346 | 0.6010 | 0.4454 | 0.1092 | **0.8300** |

Precision is marginally *worse* for the light field here (0.8312 vs 0.8538),
unlike thermal where it was better — so the precision advantage seen in thermal
does not generalise and should not be claimed. Localization does: mean IoU 0.8300
vs 0.6977, and recall at IoU 0.80 is 4.5× better. Both modalities agree that the
integral render produces tighter boxes, which is why scoring at mAP50-95 rather
than mAP50 flatters it once the GT is neutral.

---

## The one real divergence: partial occlusion is not free in rgb

Reviewed (Zenodo human) partial-occlusion flags, n = 278 occluded / 278:

| arm | R clear | R occluded | ratio |
|---|---|---|---|
| **rgb** raw / single | 0.8192 | 0.5719 | **0.70×** |
| **rgb** raw / multi | 0.7427 | 0.5144 | **0.69×** |
| thermal raw / single | 0.8080 | 0.8311 | 1.03× |
| thermal raw / multi | 0.7039 | 0.7900 | 1.12× |

In thermal, partial occlusion costs nothing — Part 1 found the same with the
small head, across all four cells. In rgb it costs **30 %**, consistently for
both arms.

The plausible reading is that the two modalities encode an animal differently. A
thermal detection is a temperature anomaly: clip half of it behind a branch and
it is still an unambiguous hot blob against a cold background. An rgb detection
depends on texture, outline and colour, and partial cover destroys exactly those.
This is a hypothesis consistent with both tables, not something these experiments
tested directly.

What the table does establish is that **multi-view does not help partial
occlusion in either modality** (0.69× vs 0.70× in rgb; both near 1.0 in thermal).
The light field compensates total occlusion only. Where the animal is visible but
degraded, thirty extra views of the same degradation add nothing.

## Frame level (n = 396)

| arm | R clean frames | R occluded frames |
|---|---|---|
| raw / single | 0.6928 | 0.4116 |
| raw / multi | 0.6971 | 0.5278 |

---

## Summary across both modalities (YOLO, raw row)

| | thermal | rgb |
|---|---|---|
| hidden-animal gain, multi/single | 2.92× | 2.22× |
| mAP50-95 gain vs merged GT | 2.00× | 2.04× |
| visible-recall change | −4 % | +0.6 % |
| partial occlusion, cost vs clear | none | −30 % |
| localization, mean IoU single → multi | 0.74 → 0.85 | 0.70 → 0.83 |

## Still open in Part 3

* The embedding row for rgb. Needs the 2048 ortho render (~2 h) to build
  `realortho_single`, then the cells and their seeds.
* One checkpoint per arm, so no variance estimate on any rgb number here.
