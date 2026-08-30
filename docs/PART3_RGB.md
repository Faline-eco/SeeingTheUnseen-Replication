# Part 3 — the full rgb grid

Measured 2026-08-08. 6 seeds per cell, IoU 0.5, conf >= 0.3, real alfspy renders
in the sensor row, 0.28 M head. Companion to `PART3_YOLO_RGB.md` (the
high-capacity half).

Completing this needed more than the plan assumed: `cell_rgb_realortho_single`
was empty *and* `geo-referenced2_2k` contained no rgb renders at all, so the
2048 px ortho render had to be run from scratch (9643 frames, all verified
readable and correctly sized), staged, built into a cell, and trained.

---

## The 2×2 on genuinely hidden animals (222 boxes)

| | single view | multi view | multi/single |
|---|---|---|---|
| raw (pixels, real renderer) | 0.1306 ± 0.0199 | 0.1119 ± 0.0234 | **0.86×** |
| embeddings (DINOv3) | 0.2590 ± 0.0358 | **0.4219 ± 0.0836** | 1.63× |
| embed / raw | 1.98× | 3.77× | |

Retention (hidden recall as a fraction of the same cell's visible recall):
27 % / 24 % / 34 % / **52 %**.

The embedding row behaves as thermal did — multi-view gives 1.63× (thermal
2.17×), and embed/multi is the only cell that retains over half its visible
performance.

## The raw row does not reproduce, and it disagrees with YOLO

**The pixel row shows no multi-view benefit in rgb: 0.86×, not separable.** On
thermal the same cells gave 1.60×.

This directly contradicts the YOLO half of Part 3, which found **2.22×** on the
same modality, same population, same renders:

| rgb, raw row, hidden animals | single | multi | ratio |
|---|---|---|---|
| 0.28 M head @128 px | 0.1306 | 0.1119 | 0.86× |
| YOLO26x @1024 px | 0.1729 | 0.3832 | 2.22× |

Both cannot be described as "multi-view helps rgb pixels". The consistent reading
is that in rgb the light-field signal in pixel space **requires capacity and
resolution to extract**: the integral render of a textured rgb scene is a
detailed image, and a 0.28 M head reading a 128×128 downsample cannot use it,
while YOLO at 1024 px can. Thermal does not have this problem because the signal
is a bright blob that survives aggressive downsampling.

This is a limitation of the small-head grid on rgb, not a contradiction of the
occlusion result — the embedding row still shows the effect (1.63×), and the
capacity-equipped pixel row shows it strongly (2.22×). But any statement of the
form "multi-view improves pixel-space rgb detection" must carry the capacity
qualifier, and the 0.86× cell should be reported, not quietly dropped.

## Frame level (412 boxes)

| cell | R clean frames | R occluded frames | retained |
|---|---|---|---|
| raw / single | 0.5073 ± 0.0337 | 0.2508 ± 0.0305 | 49 % |
| raw / multi | 0.5064 ± 0.0472 | 0.2100 ± 0.0268 | 41 % |
| embed / single | 0.7713 ± 0.0269 | 0.4826 ± 0.0325 | 63 % |
| embed / multi | 0.8237 ± 0.0395 | 0.5951 ± 0.0539 | 72 % |

Same ordering as thermal: clutter costs the pixel cells about half their
performance and the embedding cells about a third.

## Partial occlusion — refines the Part 3a claim

Reviewed (human-flagged) partial occlusion, 614 clear / 278 occluded:

| cell | R clear | R occluded | ratio |
|---|---|---|---|
| raw / single | 0.5898 ± 0.0388 | 0.4526 ± 0.0603 | 0.77× |
| raw / multi | 0.5079 ± 0.0324 | 0.4598 ± 0.0343 | 0.91× |
| embed / single | 0.7880 ± 0.0231 | 0.8118 ± 0.0273 | **1.03×** |
| embed / multi | 0.8078 ± 0.0191 | 0.8633 ± 0.0212 | **1.07×** |

`PART3_YOLO_RGB.md` concluded that partial occlusion costs ~30 % in rgb and
nothing in thermal, and read that as a property of the *modality*. With the
embedding cells added, that is too broad. The correct statement:

| | thermal | rgb |
|---|---|---|
| pixel cells | free (0.98–1.21×) | **costly (0.69–0.91×)** |
| embedding cells | free (0.96–0.98×) | **free (1.03–1.07×)** |

**Partial occlusion is costly only for pixel-space methods on rgb.** DINOv3
features do not pay it. So the mechanism is not "rgb appearance is fragile" in
general — it is that raw rgb pixels of a half-covered animal are ambiguous while
the learned features remain discriminative. The thermal/rgb asymmetry is real but
lives in the representation, not in the sensor.

---

## rgb vs thermal, embed/multi on hidden animals

| | thermal | rgb |
|---|---|---|
| embed / multi | 0.4088 ± 0.0537 | 0.4219 ± 0.0836 |
| retention | 49 % | 52 % |
| multi/single (embeddings) | 2.17× | 1.63× |

The headline cell replicates across modalities. The earlier concern recorded in
`RESULTS.md` §6.2 — that the occlusion-specific claim "does not replicate on
rgb" — is resolved: it does replicate, once the population is restricted to
genuinely hidden animals and the sensor row uses the real renderer.

## Caveats

* The rgb raw row's 0.86× is a null, not evidence of harm; ±0.023 on a 0.019
  difference.
* embed/multi rgb has the widest seed spread in the study (±0.0836), so its
  2× lead over embed/single is about 2σ — real but less firm than thermal's.
* Cell counts differ slightly by construction (9643 rgb ortho renders vs 9689
  realalfs); training uses only frames with labels, so the difference does not
  enter the comparison.
