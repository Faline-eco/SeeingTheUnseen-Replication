# YOLO26x cross-validation — every scored metric

All four arms of the image-space capacity control: `ortho` (single view) against
`alfs` (31-view integral), on both modalities, over the scene-level 5-fold split
with three seeds per fold. `±` is the between-fold standard deviation, which is
the error term the study uses; each fold value is itself a mean over its three
seeds. Ratios are ALFS ÷ ortho, paired by fold, with a paired t-test over the
five folds. Ratios at or beyond 1.5× (or 0.67×) are bold.

Scoring is `scripts/occlusion_metrics.py` at IoU 0.5 and confidence 0.3, against
the **merged** labels (`alfs_2k_<mod>_f<k>`) for both arms. Read §Caveats before
quoting anything below AP50.

Regenerate with `python scripts/yolo_md.py`; source data is
`metrics/yolocv_{thermal,rgb}_f{0..4}.json`.

## THERMAL


### Visible population

| metric | ortho (single) | ALFS (multi) | ratio | t(4) | p | folds won |
|---|---|---|---|---|---|---|
| recall | 0.7178 ± 0.0644 | 0.7156 ± 0.0580 | 0.997 | -0.12 | ns | 2/5 |
| precision | 0.7207 ± 0.0468 | 0.7088 ± 0.0540 | 0.983 | -0.57 | ns | 2/5 |
| F1 | 0.7169 ± 0.0402 | 0.7103 ± 0.0514 | 0.991 | -0.75 | ns | 3/5 |
| AP50 | 0.7211 ± 0.0507 | 0.7472 ± 0.0722 | 1.036 | 1.31 | ns | 4/5 |
| AP75 | 0.2321 ± 0.0557 | 0.6222 ± 0.0686 | **2.680** | 8.80 | **<.01** | 5/5 |
| AP50-95 | 0.3174 ± 0.0305 | 0.5226 ± 0.0545 | **1.646** | 8.36 | **<.01** | 5/5 |
| TP / FN | 52144 / 21023 | 52463 / 20704 | | | | |

24,389 ground-truth boxes summed over five folds (each counted once per fold, three seeds each).

### Hidden population

| metric | ortho (single) | ALFS (multi) | ratio | t(4) | p | folds won |
|---|---|---|---|---|---|---|
| recall | 0.1611 ± 0.0547 | 0.3293 ± 0.1361 | **2.044** | 4.39 | **<.02** | 5/5 |
| precision | 0.1947 ± 0.0318 | 0.3135 ± 0.0659 | **1.610** | 6.32 | **<.01** | 5/5 |
| F1 | 0.1734 ± 0.0430 | 0.3096 ± 0.0981 | **1.786** | 5.25 | **<.01** | 5/5 |
| AP50 | 0.0625 ± 0.0236 | 0.2065 ± 0.0990 | **3.306** | 4.10 | **<.02** | 5/5 |
| AP75 | 0.0070 ± 0.0035 | 0.0946 ± 0.0494 | **13.600** | 4.05 | **<.02** | 5/5 |
| AP50-95 | 0.0181 ± 0.0058 | 0.1027 ± 0.0521 | **5.687** | 3.99 | **<.02** | 5/5 |
| TP / FN | 4855 / 25283 | 9924 / 20214 | | | | |

10,046 ground-truth boxes summed over five folds (each counted once per fold, three seeds each).

### All population

| metric | ortho (single) | ALFS (multi) | ratio | t(4) | p | folds won |
|---|---|---|---|---|---|---|
| recall | 0.5440 ± 0.0721 | 0.5911 ± 0.0998 | 1.087 | 2.52 | <.1 | 4/5 |
| precision | 0.7383 ± 0.0427 | 0.7423 ± 0.0489 | 1.005 | 0.20 | ns | 3/5 |
| F1 | 0.6241 ± 0.0563 | 0.6531 ± 0.0715 | 1.046 | 2.67 | <.1 | 4/5 |
| AP50 | 0.5746 ± 0.0684 | 0.6489 ± 0.1001 | 1.129 | 2.87 | **<.05** | 4/5 |
| AP75 | 0.1818 ± 0.0549 | 0.5162 ± 0.0826 | **2.840** | 8.12 | **<.01** | 5/5 |
| AP50-95 | 0.2501 ± 0.0387 | 0.4405 ± 0.0693 | **1.761** | 7.38 | **<.01** | 5/5 |
| TP / FN | 56999 / 46306 | 62387 / 40918 | | | | |

34,435 ground-truth boxes summed over five folds (each counted once per fold, three seeds each).

## RGB


### Visible population

| metric | ortho (single) | ALFS (multi) | ratio | t(4) | p | folds won |
|---|---|---|---|---|---|---|
| recall | 0.5233 ± 0.0612 | 0.5208 ± 0.0911 | 0.995 | -0.10 | ns | 2/5 |
| precision | 0.7298 ± 0.0460 | 0.7415 ± 0.0595 | 1.016 | 0.49 | ns | 3/5 |
| F1 | 0.6067 ± 0.0442 | 0.6043 ± 0.0538 | 0.996 | -0.24 | ns | 2/5 |
| AP50 | 0.5679 ± 0.0517 | 0.5811 ± 0.0746 | 1.023 | 1.17 | ns | 4/5 |
| AP75 | 0.0814 ± 0.0617 | 0.3608 ± 0.0191 | **4.435** | 12.01 | **<.01** | 5/5 |
| AP50-95 | 0.1991 ± 0.0477 | 0.3416 ± 0.0302 | **1.715** | 12.71 | **<.01** | 5/5 |
| TP / FN | 34809 / 30435 | 34879 / 30365 | | | | |

21,748 ground-truth boxes summed over five folds (each counted once per fold, three seeds each).

### Hidden population

| metric | ortho (single) | ALFS (multi) | ratio | t(4) | p | folds won |
|---|---|---|---|---|---|---|
| recall | 0.1273 ± 0.0620 | 0.2159 ± 0.1015 | **1.696** | 3.20 | **<.05** | 5/5 |
| precision | 0.1986 ± 0.0315 | 0.3229 ± 0.0741 | **1.626** | 4.56 | **<.02** | 5/5 |
| F1 | 0.1472 ± 0.0561 | 0.2372 ± 0.0747 | **1.612** | 6.15 | **<.01** | 5/5 |
| AP50 | 0.0529 ± 0.0274 | 0.1434 ± 0.0624 | **2.710** | 5.70 | **<.01** | 5/5 |
| AP75 | 0.0029 ± 0.0027 | 0.0343 ± 0.0170 | **11.891** | 4.51 | **<.02** | 5/5 |
| AP50-95 | 0.0128 ± 0.0071 | 0.0604 ± 0.0268 | **4.714** | 5.30 | **<.01** | 5/5 |
| TP / FN | 3263 / 23050 | 5625 / 20688 | | | | |

8,771 ground-truth boxes summed over five folds (each counted once per fold, three seeds each).

### All population

| metric | ortho (single) | ALFS (multi) | ratio | t(4) | p | folds won |
|---|---|---|---|---|---|---|
| recall | 0.4029 ± 0.0905 | 0.4254 ± 0.1131 | 1.056 | 0.91 | ns | 3/5 |
| precision | 0.7467 ± 0.0424 | 0.7680 ± 0.0544 | 1.029 | 0.99 | ns | 3/5 |
| F1 | 0.5167 ± 0.0786 | 0.5357 ± 0.0857 | 1.037 | 1.40 | ns | 4/5 |
| AP50 | 0.4566 ± 0.0837 | 0.4914 ± 0.1029 | 1.076 | 2.88 | **<.05** | 4/5 |
| AP75 | 0.0676 ± 0.0544 | 0.2936 ± 0.0498 | **4.341** | 11.29 | **<.01** | 5/5 |
| AP50-95 | 0.1602 ± 0.0501 | 0.2830 ± 0.0525 | **1.766** | 12.16 | **<.01** | 5/5 |
| TP / FN | 38072 / 53485 | 40504 / 51053 | | | | |

30,519 ground-truth boxes summed over five folds (each counted once per fold, three seeds each).

## Caveats

**Recall and precision are the robust numbers. AP75 and AP50-95 are not.**

Both arms are scored against the same merged ground truth, and a merged box is
the axis-aligned hull of a track across the whole aperture. The ALFS render
shows the animal smeared along exactly that hull, so an ALFS detection covers it
naturally. The ortho render shows the animal at one instant — a much smaller
box — so even a perfectly placed ortho detection overlaps the hull poorly. Any
metric that demands high IoU therefore rewards ALFS for a property of the
annotation rather than of the detection.

That is what produces the extreme strict-IoU ratios, and they should not be read
as detection improvements:

| | AP75 ratio | on which population |
|---|---|---|
| thermal | **13.60×** | hidden |
| thermal | **2.68×** | visible |
| RGB | **11.89×** | hidden |
| RGB | **4.44×** | visible |

The visible row is the tell. Visible-animal *recall* is unchanged (0.997× thermal,
0.995× RGB) — the aperture does not find more animals that were already
visible — yet visible AP75 rises 2.68×. Nothing about occlusion can explain
that; it is the label geometry. `PART2_YOLO_THERMAL.md` measures the same effect
directly by scoring each arm against both annotations.

**FP counts are not population-specific.** The `fp` field is identical across
the visible, hidden and all rows because it is a frame-level count, so only TP
and FN are broken out above. Per-population *precision* is computed correctly,
using the COCO crowd convention: detections matching an out-of-population animal
are ignored rather than charged as false positives.

**Resolution.** These arms read a 1024x1024 JPEG export, whereas the embedding
arms encode the 2048x2048 PNG renders. Both YOLO arms use the same export, so
the ratios here are internally controlled, but the absolute values are not
directly comparable with the embedding tables.

## What is safe to quote

Hidden-animal recall, **2.044×** thermal and **1.696×** RGB, five of five folds
in both modalities, with hidden precision rising alongside it (1.610× and
1.626×) so the extra detections are not bought with false positives. Visible
recall is unchanged in both modalities, which is what makes the effect specific
to occlusion.
