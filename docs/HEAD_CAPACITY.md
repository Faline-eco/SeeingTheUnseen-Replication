# Detector-head capacity — does the embedding comparison depend on head size?

Status: **complete**. 30 runs, thermal DINOv3 only.

## 1. The objection being tested

Every embedding condition in this study — DINOv3 and V-JEPA alike — is read by
the **same 0.36 M** CenterNet head. Verified from the checkpoints: all 120
`cell_*_vjepa_*` runs and all 174 baseline `cell_*_embed_*` runs carry
`width=128, up=4, in_dim=128`. The encoder comparison in `PART4_VJEPA.md` is
therefore **matched by construction**, and no head-size correction is owed to it.

> An earlier version of this document, and of the paper, stated that V-JEPA used
> a 3.81 M head and that this ablation existed to match it. That was wrong. The
> figure came from `PLAN_VJEPA_ABLATION.md`, which described the head of the
> external V-JEPA project at planning time; the V-JEPA arms actually run for
> this study were trained through `train_embedding_detector.py` at its default
> width. `scripts/ckwidth.py` reads the widths back out of the checkpoints.

The question the ablation does answer is different and still worth asking: **is
0.36 M large enough to read a 128×128×128 embedding grid at all?** If the head
were the bottleneck, the 2×2 gaps would measure the detector rather than the
inputs, and the embedding row would be understated against an image row whose
inputs are far lower-dimensional.

The head scales with `--width` (stem → transposed-conv upsample ×4 → refine →
three 1×1 heads):

| `--width` | params |
|---|---|
| 128 (published) | 0.36 M |
| 256 | 1.12 M |
| 384 | 2.31 M |
| **512** | **3.90 M**, ten times the published head |

`embed_single` and `embed_multi` were rerun at `--width 512`, thermal, across
the same 5 scene-level folds × 3 seeds = **30 runs**. Everything else is held:
same cells, same labels, same folds, same seeds, same optimiser, same metrics.

Not run on the pixel-space row — YOLO26x at ~60 M already serves as its capacity
probe (`PART2_YOLO_THERMAL.md`).

## 2. Results

Ratios are 3.90 M ÷ 0.36 M, paired by fold. `won` counts folds in which the
wider head is ahead.

| cell | population | metric | 0.36 M | 3.90 M | ratio | t(4) | p | won |
|---|---|---|---|---|---|---|---|---|
| single | visible | recall | 0.6309 | 0.6648 | **1.054** | 9.66 | **< .01** | 5/5 |
| single | visible | precision | 0.6364 | 0.6394 | 1.005 | 0.27 | ns | 2/5 |
| single | visible | AP50 | 0.5946 | 0.6215 | **1.045** | 5.76 | **< .01** | 5/5 |
| single | visible | AP50-95 | 0.2372 | 0.2546 | 1.073 | 2.47 | < .1 | 4/5 |
| single | hidden | recall | 0.2125 | 0.2260 | 1.064 | 2.29 | < .1 | 4/5 |
| single | hidden | precision | 0.1911 | 0.1972 | 1.032 | 0.62 | ns | 2/5 |
| single | hidden | AP50 | 0.0738 | 0.0807 | 1.093 | 1.39 | ns | 3/5 |
| single | hidden | AP50-95 | 0.0165 | 0.0187 | 1.132 | 1.62 | ns | 4/5 |
| multi | visible | recall | 0.7107 | 0.7359 | 1.035 | 1.53 | ns | 3/5 |
| multi | visible | precision | 0.5925 | 0.6142 | **1.037** | 3.50 | **< .05** | 5/5 |
| multi | visible | AP50 | 0.6394 | 0.6835 | 1.069 | 2.53 | < .1 | 4/5 |
| multi | visible | AP50-95 | 0.2800 | 0.3055 | 1.091 | 1.58 | ns | 4/5 |
| multi | hidden | recall | 0.3192 | 0.3229 | 1.012 | 0.38 | ns | 3/5 |
| multi | hidden | precision | 0.2059 | 0.2224 | **1.080** | 3.00 | **< .05** | 5/5 |
| multi | hidden | AP50 | 0.1235 | 0.1334 | 1.080 | 1.46 | ns | 4/5 |
| multi | hidden | AP50-95 | 0.0334 | 0.0374 | 1.120 | 1.60 | ns | 4/5 |

## 3. Reading

**The wider head is better, and that is not in doubt.** All **16 of 16** ratios
are ≥ 1.005 — not one metric goes down — and four reach p < 0.05, three of them
winning 5/5 folds. This is a different picture from the debias study, where the
significant cells disagreed on sign. (The 16 comparisons are 2 cells × 8
correlated metrics, so a formal sign test would overstate the case; the point is
the uniformity of direction, not a p-value on it.)

**But it is better at the things the study does not rest on.** Sort the
significant results by what they measure:

| improves significantly | does not |
|---|---|
| visible recall (single), ×1.054 | **hidden recall (multi), ×1.012, ns, 3/5** |
| visible AP50 (single), ×1.045 | hidden recall (single), ×1.064, p < .1 |
| visible precision (multi), ×1.037 | every hidden AP metric |
| hidden precision (multi), ×1.080 | |

Capacity buys **visible-animal performance and precision**. The one number the
2×2 turns on — multi-view hidden recall — moves by 1.2 %, wins 3 of 5 folds, and
is the least significant comparison in the table. That is the same asymmetry
YOLO26x showed from the other side, where a 60 M detector left visible recall
unchanged at 0.997× and moved hidden recall by 2.04×: capacity and aperture act
on different populations.

**The view-axis gain survives the larger head.** Recomputing the multi ÷ single
hidden-recall ratio at each capacity:

| head | single | multi | multi/single |
|---|---|---|---|
| 0.36 M | 0.2125 | 0.3192 | **1.502** |
| 3.90 M | 0.2260 | 0.3229 | **1.428** |

The gain narrows from 1.50× to 1.43× because the single-view cell gains more
from capacity than the multi-view cell does (×1.064 against ×1.012) — the
single-view cell was the more capacity-limited of the two. It does not close,
and 1.43× still sits inside the 1.50–1.68× range the encoders span.

**What may be claimed.** The published 0.36 M numbers understate absolute
performance somewhat, so the 2×2 gaps are, if anything, conservative on the
visible population. The view-axis conclusion is not
explained by head capacity. What cannot be claimed is that head size is
irrelevant — it plainly is not, on 16 of 16 metrics.

## 4. Provenance

```
metrics/w512_thermal_f{0..4}.json         scored arms, 3 seeds each
embedding_runs_multiseed/
  cell_thermal_embed_{single,multi}_..._w512_f{0..4}_s{1337,42,7}/
scripts/cap_table.py                      reproduces the table above
```

Baselines are read from `metrics/thermal_f{0..4}.json`, the same files the main
2×2 is scored from, so no rescoring is involved. The metrics store calls the
hidden population `occluded`; `meta.hidden_mask` is what restricts it, and it is
the same population reported everywhere else.


## 5. Why the 0.36 M head stays in the primary 2x2

The wider head is uniformly better, so why not adopt it everywhere?

Not because of the encoder comparison — that is already matched, as §1 sets out.
The reason is the **representation axis**. It sets a 0.21 M image-space detector
against an embedding-space detector, and at 0.36 M the two sides sit within 1.7×
of each other. Give the embedding row 3.90 M and they are 18.6× apart, so
"embeddings help" becomes hard to separate from "a larger head helps" — an
effect this ablation has just shown is real, on 16 of 16 metrics.

So the small head is not legacy. It is what keeps the study's primary axis from
turning into a capacity comparison.

### The encoder ranking under a handicap

Because both encoders already share a head, the wider one can be given to
DINOv3 *alone*, putting V-JEPA at a tenfold parameter disadvantage. Thermal
hidden recall, V-JEPA ÷ DINOv3:

| | single | multi |
|---|---|---|
| at parity (both 0.36 M) — as published | 1.146 | 1.175 |
| DINOv3 handicapped up to 3.90 M | **1.078** | **1.162** |

V-JEPA still leads. About half its single-view margin, and almost none of its
multi-view margin, can be bought back with detector capacity.

### Scope

Thermal only. The question — is the embedding grid under-read at 0.36 M — is
answered by one modality, but the RGB conditions have not been retrained at
width 512, and that is a scope limit rather than a design choice.
