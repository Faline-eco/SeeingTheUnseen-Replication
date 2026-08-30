# Part 1 — how the four cells perform on occluded vs visible animals (thermal)

Measured 2026-08-07. 6 seeds per cell, IoU 0.5, confidence >= 0.3, real alfspy
renders in the sensor row, 0.28 M CenterNet head throughout. Val = 233 labelled
central frames.

"Occluded" is not one thing, so this reports three populations rather than
picking the flattering one. They disagree, and the disagreement is the result.

---

## A. Fully hidden animals (the claim)

Animals in the aperture-merged GT with no central-frame counterpart, restricted
to those the central view **demonstrably imaged** (so wider-footprint recoveries
are excluded). n = 212 hidden, ~900 visible.

| cell | R visible | R hidden | retained |
|---|---|---|---|
| raw / single (realortho) | 0.4958 ± 0.0464 | 0.0739 ± 0.0279 | 15 % |
| raw / multi (realalfs) | 0.5955 ± 0.0686 | 0.1179 ± 0.0248 | 20 % |
| embed / single | 0.7761 ± 0.0230 | 0.1887 ± 0.0158 | 24 % |
| **embed / multi** | **0.8407 ± 0.0209** | **0.4088 ± 0.0537** | **49 %** |

The "retained" column is the point. Every cell is better on visible animals than
hidden ones — that is not news. But three of the four keep only 15–24 % of their
visible performance when the animal is hidden, while embed/multi keeps half. The
advantage is **specific to occlusion**, not a general accuracy offset that would
lift both columns equally.

Read as a 2×2 on hidden animals alone, the two axes are close to multiplicative:

| | single | multi | multi/single |
|---|---|---|---|
| raw | 0.0739 | 0.1179 | 1.60× |
| embeddings | 0.1887 | 0.4088 | 2.17× |
| embed/raw | 2.55× | 3.47× | |

Neither axis alone gets there: multi-view on pixels reaches 0.118, embeddings on
a single view reach 0.189, and only the combination reaches 0.409. The
super-multiplicative corner (2.17 > 1.60, 3.47 > 2.55) says the two help each
other rather than merely stacking.

For reference, the same cells with the **PyTorch integrator** instead of the real
renderer — kept because it is what the earlier conclusions were built on:

| cell | R visible | R hidden |
|---|---|---|
| sensor / single | 0.2341 ± 0.0326 | 0.0393 ± 0.0183 |
| sensor / multi | 0.3540 ± 0.0564 | 0.0464 ± 0.0205 |

Note it inverts the multi-view finding on hidden animals (1.18× vs the real
renderer's 1.60×, and not separable at this variance). The renderer substitution
was not cosmetic.

---

## B. Partially occluded animals (Zenodo reviewer flags)

Animals annotated centrally but marked partially hidden by a human reviewer.
672 clear, 219 occluded, 77 unlabelled (excluded).

| cell | R clear | R occluded | ratio |
|---|---|---|---|
| raw / single | 0.5040 ± 0.0374 | 0.6081 ± 0.0502 | 1.21× |
| raw / multi | 0.6074 ± 0.0834 | 0.6225 ± 0.0463 | 1.02× |
| embed / single | 0.8083 ± 0.0268 | 0.7747 ± 0.0255 | 0.96× |
| embed / multi | 0.8313 ± 0.0169 | 0.8181 ± 0.0326 | 0.98× |

**Partial occlusion costs essentially nothing, for anyone.** Every ratio is
within noise of 1.0, and the multi-view advantage that dominates population A is
absent here. This is the honest limit on the headline claim: the light field
compensates *total* occlusion, and there is nothing to compensate when the animal
is merely half-hidden — a thermal signature clipped by a branch is still a
thermal signature.

The >1.0 ratio for raw/single is worth flagging rather than explaining away:
reviewer-flagged animals are on average larger (they are the ones a human could
track through cover), so the flag partly selects for easy targets. Population A
does not have this confound, since membership there is geometric.

---

## C. Frame level — are scenes containing occluders harder?

All boxes of a frame grouped by whether the frame contains at least one hidden
animal. 373 boxes in occluder-containing frames, ~740 in clean ones.

| cell | R clean frames | R occluded frames | retained |
|---|---|---|---|
| raw / single | 0.5522 ± 0.0512 | 0.1439 ± 0.0296 | 26 % |
| raw / multi | 0.6538 ± 0.0713 | 0.2082 ± 0.0380 | 32 % |
| embed / single | 0.7836 ± 0.0274 | 0.4272 ± 0.0194 | 55 % |
| embed / multi | 0.8410 ± 0.0224 | 0.5947 ± 0.0358 | 71 % |

Frames containing occluders are harder *even for the animals that are plainly
visible in them* — dense canopy degrades everything in the scene, not just what
it hides. Compare the visible-animal recall in those frames against the overall
figure from A:

| cell | R visible, all frames | R visible, occluder frames | cost |
|---|---|---|---|
| raw / single | 0.4958 | 0.2758 | −44 % |
| raw / multi | 0.5955 | 0.3931 | −34 % |
| embed / single | 0.7761 | 0.7271 | −6 % |
| embed / multi | 0.8407 | 0.8215 | −2 % |

This is the strongest single result in Part 1. The pixel cells lose a third to a
half of their performance on animals they can see perfectly well, purely from
being in a cluttered scene; the embedding cells barely notice. DINOv3 features
are robust to the clutter itself, independently of whether they can see through
it.

---

## What Part 1 establishes

1. Multi-view + embeddings is the only cell that meaningfully recovers fully
   hidden animals (5.5× the single-view pixel baseline), and it does so
   specifically — it keeps 49 % of its visible-animal performance where the
   others keep 15–24 %.
2. The effect is confined to **total** occlusion. Partial occlusion is nearly
   free for every method, so any claim phrased as "handles occlusion" without
   that qualifier overstates it.
3. Clutter has a large, separate cost that embeddings absorb and pixels do not.
   Part of what looks like an occlusion result is really a robustness result.

## Caveats carried into Parts 2–4

* All of the above is a 0.28 M head. Whether a high-capacity detector shows the
  same pattern is Part 2, and it is a real risk: on aggregate mAP, multi-view
  *helps* this head and *hurts* YOLO.
* Thermal only. rgb is Part 3 and needs a real 2048 ortho render first.
* V-JEPA is not in these tables; it is scored in image space and on a slightly
  different population (243 of 268 reachable). Part 4.
* Recall at a fixed conf 0.3 throughout, so the cells are compared at one
  operating point rather than each at its own best. Precision is not in these
  tables at all.
