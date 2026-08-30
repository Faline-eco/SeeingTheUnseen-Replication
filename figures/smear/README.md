# Motion smear under synthetic aperture

**Flight 192, frame 000640, thermal — fold 3.** The clearest example in the set of
what the aperture does to an animal that *moves* during the sweep.

| box | meaning |
|---|---|
| solid yellow | merged box — hull of the track over every contributing frame |
| dotted cyan | central box — the animal in the central frame alone |

| file | what it shows |
|---|---|
| `thermal_192_000640_alfs.png` | the 31-view ALFS integral, all four animals |
| `thermal_192_000640_alfs_zoom.png` | the two most smeared animals, 3× |
| `thermal_192_000640_ortho.png` | the same ground in the single ortho view |
| `thermal_192_000640_ortho_zoom.png` | the same zoom in the single ortho view |

## What the aperture does here

The aperture focuses on the DEM plane, so static ground detail — rocks, bushes,
the drainage line — sharpens as the 31 views are integrated. Anything that moved
during the ~9 m sweep does the opposite: its radiance is spread along its own
trajectory and averaged against background, so it integrates into a **dim, curved
streak**. In the ortho zoom the same four animals are compact, bright,
high-contrast blobs.

This frame is open ground with no canopy. There is nothing to see through, so
integration here only costs — it is the failure mode of the technique rather than
the case it is for, and worth showing alongside the frames where the aperture wins.

The annotation records the same motion independently. Merged boxes are 2.1–3.5×
the area of their central counterparts:

| animal | merged | central | area |
|---|---|---|---|
| 0 | 91 × 190 px | 63 × 79 px | ×3.47 |
| 1 | 90 × 85 px | 71 × 50 px | ×2.15 |
| 2 | 108 × 97 px | 66 × 66 px | ×2.40 |
| 3 | 91 × 58 px | 58 × 44 px | ×2.07 |

## Why this frame is also a caveat about the hidden population

All four animals are counted as **hidden** by the study's own definition, and all
four are plainly visible in the central view.

That is not a labelling error — it follows from how the population is built, in
`scripts/decompose_invisible.py`. A merged box is "invisible centrally" when it
matches no central box at **IoU 0.5**, and it is "hidden" when the single-view
coverage map nonetheless shows the central view covered that ground. The flag is a
**geometric coverage test, not a human judgement**. Here the merged boxes are
inflated enough by motion that all four fall under the threshold — 0.288, 0.464,
0.416, 0.484 — three of them only narrowly. The coverage test then correctly
reports that the ground was covered, and the animals enter the hidden set.

Across the whole dataset, splitting hidden boxes by their best IoU against the
central annotation:

| | thermal (10,046) | rgb (8,771) |
|---|---|---|
| no central box overlaps at all | 43.2 % | 45.8 % |
| overlaps below 0.3 | 30.8 % | 31.0 % |
| **overlaps 0.3–0.5** | **26.0 %** | **23.2 %** |

The last row is the affected group: an animal that *is* annotated in the central
frame, whose merged box misses the threshold. Median IoU in that band is 0.426,
and 920 thermal boxes sit above 0.45.

This matters for the headline result because every arm is scored against the
**merged** labels. A single-view detector that boxes such an animal correctly in
the central frame still fails IoU 0.5 against the inflated merged box and is
counted as a miss, while a multi-view detector trained on merged boxes predicts the
inflated box and matches. That would inflate the multi-view advantage on the hidden
population by a label-geometry effect rather than occlusion recovery.

**Unquantified.** How much of the reported ×1.70 (rgb) / ×2.04 (thermal) hidden-recall
ratio this accounts for has not been measured. The check is cheap — re-score with
the hidden mask restricted to boxes with no central overlap, which needs no
retraining, only a re-run of `occlusion_metrics.py` — and it should be done before
the ratio is quoted as occlusion recovery.

## Regenerating

```bash
python scripts/make_smear_figure.py \
  --flight 192 --frame 000640 --fold 3 --modality thermal \
  --out figures/smear
```
