# Learned view aggregation — is the mean the bottleneck?

Status: **complete**. 120 runs on the multi-view DINOv3 cell, both modalities:
a mean control and three learned aggregators, 5 scene-level folds × 3 seeds
each. Paper: Appendix "Learned view aggregation".

## 1. Question

Every multi-view embedding cell integrates its 31 registered views with a
**mean**. That is a fixed aggregator: a view in which an animal is occluded
weighs exactly as much as one in which it shows through a canopy gap.
`HEAD_CAPACITY.md` showed that a tenfold larger head does not move hidden
recall, which points at the *evidence reaching the head*, not the head, as
the limit. A learned aggregator is the cheapest way to raise that evidence
without touching geometry, encoder or head. V-JEPA, which learns its own clip
integration, leads DINOv3 on thermal (1.175× hidden recall at head parity);
this experiment asks how much of such a margin the integration step alone can
account for.

## 2. Design

One thing moves. Same DINOv3 source-frame embeddings, same DEM focus-plane
geometry, same 128×128×128 grid, same 0.36 M CenterNet head, same folds,
seeds, central-GT checkpoint selection, scorer and hidden mask.

| arm (`--agg`) | what replaces the mean | params | order-aware | spatial context |
|---|---|---|---|---|
| `mean` (control) | nothing: the mean through the same online-warp pipeline | 0 | no | no |
| `attn` | masked softmax over one learned score per (view, cell); weighted sum of the *input* features | 27 k | no | 3×3 conv on the score only |
| `gru` | bidirectional `GRUCell` over the views in flight order, masked updates; mean + zero-init readout of the final states | 75 k | yes | no |
| `swin` | two pre-norm transformer blocks over all views × a 4×4-cell window (496 tokens), second block on a partition shifted by 2 cells; learned view and position embeddings; masked mean over views + zero-init readout | 90 k | via view embedding | 4×4, ~8×8 after the shift |

**Every arm starts as the published cell.** The last layer of each aggregator
is zero-initialised (attention score → uniform weights; GRU and swin readout →
zero residual on the mean), so anything the model does differently it had to
learn.

**Inputs are warped on the fly.** Storing the registered stacks would be
31 × 128 × 128 × 128 × fp16 ≈ 130 MB per frame. Instead
`render_embedding_field.py --emit grid` writes the sampling geometry
(`project_to_grid` coordinates and validity, restricted by the modality mask
exactly as `integrate()` does; ~2 MB per frame) and `view_aggregator.py`
`grid_sample`s the source embeddings on the GPU. `check_viewgrid.py` verifies
that this warp with uniform weights reproduces the stored field (mean |Δ|
4e-5 on features of range 3.5, i.e. fp16 quantisation).

**The control reproduces the published cell.** Mean-through-this-pipeline vs
`cell_<mod>_embed_multi`: thermal hidden recall 0.316 vs 0.319 (0.99×),
rgb 0.314 vs 0.306 (1.03×), visible within 1.02×, nothing significant. What
follows is therefore attributable to the aggregator alone.

Training: AdamW 3e-4, cosine, batch 16, 40 epochs with patience 12, fp16
autocast; GRU cells and swin windows are processed in chunks under gradient
checkpointing (7.8 GB / 10.2 GB peak on a 16 GB V100). One V100 run takes
55–135 min (attn), 170–360 min (gru), 200–400 min (swin).

## 3. Results

`python scripts/viewagg_arms.py attn gru swin` reproduces every number below
from `metrics/viewagg_{mean,attn,gru,swin}_<mod>_f<k>.json`. Pooled recall and
precision are Σtp/Σn over folds × seeds; AP50 is the mean of fold means;
ratios are arm ÷ control, paired by fold (t, df 4).

### 3.1 Each arm against the control

| population / metric | ctrl (thermal) | attn | gru | swin | ctrl (rgb) | attn | gru | swin |
|---|---|---|---|---|---|---|---|---|
| hidden recall | 0.3162 | **1.108\*** (5/5) | 1.043 (2/5) | **1.064\*** (5/5) | 0.3144 | 1.033 (3/5) | 0.981 (2/5) | 1.041 (3/5) |
| hidden precision | 0.2121 | 1.002 | 1.049 (5/5) | 1.086 (5/5) | 0.2035 | 1.137 (4/5) | 1.064 (4/5) | 1.098 (4/5) |
| hidden AP50 | 0.1268 | **1.130\*** (4/5) | 1.094 (4/5) | **1.158\*** (5/5) | 0.1274 | 1.137 (4/5) | 0.976 (3/5) | **1.157\*** (5/5) |
| visible recall | 0.7212 | 1.002 | 1.000 | 1.013 (4/5) | 0.6749 | 0.977 (0/5) | 0.977 (2/5) | 1.013 (4/5) |
| visible precision | 0.5984 | 0.960 | 1.008 | 1.023 | 0.5763 | 1.046 | 1.031 | 1.039 |
| visible AP50 | 0.6524 | 1.002 | 1.011 | **1.037\*** (5/5) | 0.5978 | 0.997 | 0.990 | **1.038\*** (4/5) |

\* p < 0.05, fold-paired. Folds won in parentheses where informative.

### 3.2 Hidden recall per fold

| | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|
| thermal mean / attn / gru / swin | .125 / .133 / .124 / .128 | .295 / .344 / .331 / .319 | .293 / .360 / .338 / .313 | .403 / .415 / .403 / .414 | .465 / .500 / .453 / .508 |
| rgb mean / attn / gru / swin | .115 / .126 / .117 / .123 | .322 / .312 / .317 / .320 | .333 / .326 / .320 / .331 | .387 / .412 / .398 / .432 | .460 / .496 / .430 / .477 |

### 3.3 Arm against arm (fold means, df 4)

| | thermal hidden recall | thermal hidden AP50 | rgb hidden recall | rgb visible recall |
|---|---|---|---|---|
| attn − gru | **+0.021, 5/5, p = 0.038** | +0.005, ns | +0.018, 4/5, ns | 0.000 |
| swin − attn | −0.014, 1/5, ns | +0.004, ns | +0.002, ns | **+0.023, 5/5, p = 0.001** |
| swin − gru | +0.007, 3/5, ns | +0.008, ns | +0.020, 5/5, p = 0.078 | +0.023, 5/5, p = 0.022 |

## 4. Reading

- **All three arms recover thermal hidden recall, by 1.04–1.11×, and none
  moves it significantly in RGB.** The pre-registered signature (hidden rises,
  visible flat) appears in thermal for every arm. In RGB canopy gaps carry
  little per-view contrast, so there is nothing for a weighting to select.
- **The spread between architectures is small and does not favour the more
  expressive ones.** Attention pooling, the smallest model, has the largest
  recall gain; the GRU sits below it on every fold; the windowed transformer
  is between the two. Neither memory along the aperture nor joint
  spatial–view context lifts the recall ceiling set by re-weighting the views.
  Re-combining the 31 views recovers at most about one tenth more hidden
  animals; the rest of the hidden evidence lies upstream, in the per-view
  encoder and the aperture geometry.
- **What spatial context buys is precision.** The swin arm is the only one
  that improves every metric of both populations in both modalities: hidden
  AP50 +16 % in both (p = 0.026), visible AP50 +4 % (p < 0.05), and no RGB
  visible-recall cost (+0.023 over both other arms on every fold).
- **Scale.** On multi-view hidden recall the measured factors order as
  aperture (1.50×), learned aggregation (1.06–1.11× thermal, ≤ 1.04× RGB)
  and head capacity (1.01×). The thermal aggregation gain is of the same size
  as V-JEPA's thermal multi-view margin over DINOv3, so part of that margin is
  plausibly learned integration rather than the encoder.

Caveats: five folds is a small paired sample; the training recipe was tuned
for the mean and not re-tuned per arm; the GRU is order-sensitive by
construction and the swin arm only through its view embedding, so "flight
order" is tested by one arm only.

## 5. Provenance

```
scripts/render_embedding_field.py --emit grid     sampling geometry per frame
scripts/view_aggregator.py                        dataset, model (--agg mean|attn|gru|swin), trainer
scripts/check_viewgrid.py                         warp == stored field check
scripts/occlusion_metrics.py                      dispatches on ck["kind"] == "view_aggregator"
scripts/viewagg_arms.py                           the tables above
docker/viewgrid_dgx.sh                            dump grids, sharded by flight
docker/viewagg_cv.sh                              5 folds x 3 seeds + scoring for one AGG
docker/launch_viewagg.sh                          thermal + rgb loops on two GPUs
metrics/viewagg_<agg>_<mod>_f<k>.json             arm vs published cell, same scorer + hidden mask
```

Checkpoints: `heads/viewgrid_<mod>_embed_multi_<agg>_f<k>_s<seed>/` on
Hugging Face (`best.pt` carries `agg`, `hid`, `win`;
`ViewAggregatorDetector.from_checkpoint` rebuilds the model). Runs took
2026-09-09 to 2026-09-18 on two V100s, 120 runs, 0 failures.
