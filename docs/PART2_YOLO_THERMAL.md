# Part 2 — does the occlusion result survive a high-capacity detector? (thermal)

Measured 2026-08-07. YOLO26x, 100 epochs, imgsz 1024, single checkpoint per arm
(no seed variance — treat each number as one draw). Same GT, same IoU 0.5, same
confidence floor 0.3 as Part 1.

Two runs cover the top row of the grid exactly: `ortho_thermal` is raw/single
(orthorectified central frame), `alfs_thermal` is raw/multi (integral render over
the aperture). Both predict in the DEM-plane ortho canvas, so Part 1's
populations transfer unchanged. `raw_thermal` is excluded: it predicts in camera
image space, which is the V-JEPA situation and belongs with Part 4.

---

## The premise was wrong

Part 2 existed to test a worry: multi-view improved the 0.28 M head but appeared
to *hurt* YOLO (mAP 0.6609 -> 0.4018), so the occlusion result might have been a
statement about a weak detector. It isn't. The apparent harm was a scoring
artifact, and the correction runs the other way.

**mAP for both arms against both ground truths** (my AP harness; it reproduces
the published ortho/central figure, 0.6661 vs 0.6609, which is the gate for
trusting the rest of the table):

| model | GT | mAP50 | mAP50-95 |
|---|---|---|---|
| ortho (single) | central | 0.8781 | **0.6661** |
| ortho (single) | merged | 0.6229 | 0.2654 |
| alfs (multi) | central | 0.7764 | **0.3796** |
| alfs (multi) | merged | 0.7671 | 0.5308 |

Each model scores best against the ground truth built from its own view geometry.
The two bolded cells are the numbers that have been quoted against each other —
and they come from *different rows of this table*. That comparison is not
like-for-like, and neither is the "fix" recorded earlier in `PLAN.md`, which
proposed scoring both against central GT.

Central GT is not neutral. It is the **single-view** ground truth: it omits every
animal the central frame could not see. Scoring a multi-view detector against it
converts each correct recovery of a hidden animal into a false positive. That
penalty is proportional to how well the method does the thing being studied.

Against the neutral ground truth — merged GT, the full aperture truth, which both
arms are scored against identically — the ordering reverses:

**alfs 0.5308 vs ortho 0.2654 mAP50-95: exactly 2×.**

---

## Recall on hidden animals

n = 206 (the 1024 tree yields 267 invisible boxes to the 2048 tree's 268, and 6
fall out of the mask, so YOLO's population is 3 % smaller than Part 1's 212).

| arm | R visible | R hidden | retained |
|---|---|---|---|
| raw / single (ortho) | 0.7774 | 0.0631 | 8 % |
| raw / multi (alfs) | 0.7464 | 0.1845 | 25 % |

Multi-view gain on hidden animals: **2.92×**, against the small head's 1.60×.
Visible recall is flat to slightly down (−4 %). So with a capable detector the
light field's contribution is *entirely* concentrated on animals the single view
cannot see, which is a cleaner version of the Part 1 result rather than a
contradiction of it.

Capacity does close part of the gap, as expected: YOLO's single-view visible
recall is 0.7774 against the small head's 0.4958. But it does **not** close the
gap on hidden animals — 0.0631 against 0.0739, i.e. no better. Capacity buys
performance on what is visible and buys nothing on what is not. That is the
strongest evidence so far that the multi-view effect is geometric rather than an
artifact of an underpowered head.

## Precision and localization

Two hypotheses for the original mAP gap, both tested and both refuted:

| arm | dets | TP | precision |
|---|---|---|---|
| raw / single | 896 | 718 | 0.8013 |
| raw / multi | 825 | 757 | **0.9176** |

| arm | R@0.50 | R@0.60 | R@0.70 | R@0.75 | R@0.80 | R@0.90 | mean IoU of found GT |
|---|---|---|---|---|---|---|---|
| raw / single | 0.6111 | 0.5573 | 0.4017 | 0.2812 | 0.1547 | 0.0274 | 0.7374 |
| raw / multi | 0.6385 | 0.6359 | 0.6197 | 0.5880 | 0.5248 | 0.1641 | **0.8527** |

The integral render is not blurring the boxes — it localizes *better*, and holds
recall at strict IoU where the single view collapses (0.5248 vs 0.1547 at 0.80).
Plausible reading: averaging 31 views suppresses the sensor noise and background
texture that make a thermal blob's extent ambiguous. Worth stating as a
hypothesis, not a finding; it was not tested directly.

## Other populations

Frame level (all boxes, grouped by whether the frame contains a hidden animal):

| arm | R clean frames | R occluded frames |
|---|---|---|
| raw / single | 0.8197 | 0.2896 |
| raw / multi | 0.8008 | 0.3197 |

Reviewed partial occlusion (Zenodo flags, n=219):

| arm | R clear | R occluded | ratio |
|---|---|---|---|
| raw / single | 0.8080 | 0.8311 | 1.03× |
| raw / multi | 0.7039 | 0.7900 | 1.12× |

Both reproduce Part 1's pattern: partial occlusion is nearly free, and the
multi-view advantage does not live there.

---

## What Part 2 changes

1. The risk Part 2 was built to check does not exist. Multi-view does not hurt a
   high-capacity detector; it doubles its mAP once both arms are scored against
   the same ground truth.
2. **Correction to the record.** "alfs_thermal 0.5230 was against merged GT,
   0.4018 is the like-for-like figure" is wrong in its second half. 0.4018 is
   alfs scored against *single-view* GT, which structurally penalises multi-view
   success. The like-for-like pair is 0.5308 (alfs) vs 0.2654 (ortho), both
   against merged GT. Corrected in `PLAN.md`.
3. Capacity and multi-view address different failure modes: capacity lifts
   visible recall (0.4958 -> 0.7774) and does nothing for hidden animals
   (0.0739 -> 0.0631). They are not substitutes.

## Caveats

* One checkpoint per arm. Everything here is a single draw, unlike Part 1's
  6 seeds. The 2× mAP gap is far too large to be seed noise, but the smaller
  differences (visible recall −4 %) are not separable.
* imgsz 1024 on 1024 px renders, so this does not speak to the 2048 question.
* The embedding row of the grid has no YOLO equivalent, so Part 2 upgrades the
  detector only on the raw row.
