# INSID3 positional debiasing — full technical record

Everything about the positional-debiasing ablation: what was tested, how it was
built, what it cost, what it produced, and what the numbers do and do not
support. Companion to `METHODS.md` (the 16-arm compendium) and `RESULTS.md`
(§6.10 carries the condensed version).

Status: **complete**. All four encoder×modality pairs scored, plus both
integration orders for thermal. 120 training runs, 64 paired comparisons.

---

## 1. The operation being tested

Dense transformer features carry a component that encodes **absolute patch
position** rather than image content. INSID3 (arXiv 2603.28480) proposes removing
it by estimating that subspace from content-free input and projecting features
onto its orthogonal complement:

```
F~ = F (I_D − B Bᵀ)
```

where `F` is `(N, D)` patch tokens and `B` is `(D, r)` — the leading `r`
directions of the positional subspace. The operation is a single linear
projection: no centring, no rescaling, no learned parameters.

**Terminology correction that matters.** "Gaussian-noise normalisation" describes
the *probe input* used to estimate `B`, not the operation applied to features.
Nothing is normalised. Calling it normalisation in the paper draft was wrong and
the sentence is flagged for deletion.

`r = 32` throughout. Rank was not swept — see §9.

---

## 2. Estimating the basis

`B` is estimated by pushing **Gaussian noise** through the encoder. Noise carries
no semantic content, so whatever structure survives in the token features is
positional (plus the encoder's mean response). The leading right singular vectors
of the noise feature matrix span it.

Both fitters accumulate the **Gram matrix** `MᵀM` rather than storing tokens —
the right singular vectors of `M` are the eigenvectors of `MᵀM`, and `D` is only
768–1280 while the token matrix would be ~10⁶ × D. Eigendecomposition via
`np.linalg.eigh`, eigenvalues clipped at 0, sorted descending. The **full** basis
and its singular values are stored, so rank can be chosen afterwards without
re-encoding.

**The basis is geometry-specific.** It describes a particular token grid, so a
basis fitted at one input size does not describe another. Both fitters take the
post-scaling edge length actually fed to the encoder.

### 2.1 DINOv3 — `scripts/fit_positional_basis_dino.py`

| parameter | value |
|---|---|
| model | DINOv3 ViT-H+/16, local HF dir |
| input | 64 Gaussian-noise **images**, 2048×2048 |
| noise | mean 127, std 60.0 in 0–255 pixel units, seed 0 |
| tokens | (2048/16)² = 16,384 per image → **1,048,576 total** |
| dimension | 1280 |
| output | `zenodo_labels/dino_positional_basis.npz` |

Spectrum (cumulative variance of the noise-feature subspace):

| rank | 1 | 32 |
|---|---|---|
| cumulative | **44.9 %** | **99.2 %** |

### 2.2 V-JEPA — `scripts/fit_positional_basis_vjepa.py` (written for this study)

The V-JEPA project already contained a `positional_basis.npz`. **It is unusable
here** and reusing it would have been a silent error: it was fitted for
`facebook/vjepa2-vitl-fpc64-256` — ViT-**L**, 1024 dims, clip length 64, 256×256
input, 8 clips, 65,536 tokens. Our cells use `2.1-vit-b-384`: ViT-**B**, 768
dims, clip length 30, 384×384 tiles. Wrong width *and* wrong token grid.

| parameter | value |
|---|---|
| model | V-JEPA 2.1 ViT-B-384 (86.8 M params), patch 16, tubelet 2 |
| input | 32 Gaussian-noise **clips**, 30 × 384 × 384 |
| noise | mean 127, std 60.0 (identical to the DINOv3 fit), seed 0 |
| preprocessing | ImageNet mean/std — the same constants `encode_vjepa_cells.py` uses |
| tokens | (30/2) × (384/16)² = 15 × 576 = 8,640 per clip → **276,480 total** |
| dimension | 768 |
| precision | fp32 weights + `torch.autocast` fp16 |
| runtime | **17.1 s** |
| output | `zenodo_labels/vjepa_positional_basis.npz` |

Spectrum:

| rank | 1 | 8 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| cumulative | **89.49 %** | 95.07 % | **97.91 %** | 98.70 % | 99.25 % | 99.65 % | 99.92 % |

**V-JEPA's positional subspace is far more concentrated than DINOv3's** — one
direction carries 89.5 % of the noise response versus 44.9 %. This drove the
prediction that V-JEPA would benefit most. It did not (§7.3).

Noise must go through *identical* preprocessing to the real encode. Estimating
the subspace for inputs the encoder never sees would describe a different
operating point.

---

## 3. The second premise measurement (DINOv3 only)

Beyond "is the subspace low-rank", the question is whether it holds content we
need. Measured on **real** patch tokens, split by whether a token falls inside an
animal box:

| tokens | energy in rank-32 subspace |
|---|---|
| inside animal boxes | **0.2673** |
| background | **0.4810** |
| ratio | **0.56** (below 1 at every rank tested) |

Animal tokens put markedly *less* energy in the positional subspace than
background does. The positional component is therefore disproportionately a
**background** component, and removing it should raise animal/background contrast
rather than cost discriminability.

This prediction did not survive contact with the detector (§8).

---

## 4. Applying the projection — DINOv3 path

The projection must be applied to the **native 1280-d features, before the PCA**.
The PCA basis is fitted in the debiased space, which is why a debiased run cannot
reuse the baseline `pca.pkl` — it describes a different linear space.

Driver: `docker/debias_probe.sh`. Four stages.

**Stage 1 — re-encode the per-frame store.** `scripts/encode_embeddings.py` with
`--debias-basis` / `--debias-rank`. `--fit-only` runs first in a single process
so every shard shares one PCA basis; sharded workers then refuse to fit their own
(a race here would put shards in different linear spaces — invisible downstream
and fatal to it). Three shards across GPUs 0–2.

| | thermal | rgb |
|---|---|---|
| frames re-encoded | **138,220** | **119,863** |
| PCA | 1280 → 128 | 1280 → 128 |
| explained variance | **87.3 %** | 87.3 % |
| baseline equivalent | 92.6 % | 92.6 % |
| wall time | ~10 h | ~8 h |

Debiasing costs ~5 points of retained variance, as expected — the removed
directions were among the highest-variance ones.

Output is **auto-suffixed** `srcframes_<mod>_debias32`. The suffix is derived
inside the encoder, not taken from `--out`, so a debiased run *structurally
cannot* overwrite the baseline store.

**Stage 2 — integrate onto the ortho canvas.** Via `docker/integrate_dgx.sh` for
both apertures (`single`, `full`), `--channels feat`, `--out-hw 128`.

| | thermal | rgb |
|---|---|---|
| flights with labels | **79** | **76** |
| fields per aperture | **10,663** | **9,221** |

Each field carries a `_cov.npy` sidecar, so raw file counts are 2×.

**Stage 3 — flatten.** Symlink each field into a flat cell directory the trainer
reads: `cell_<mod>_embed_{single,multi}_debias32`.

**Stage 4 — train.** `docker/cv_folds.sh` with
`CELLS="embed_single_debias32 embed_multi_debias32"`, 5 folds × 3 seeds = 30 runs.

---

## 5. Applying the projection — V-JEPA path

Structurally different: there is **no per-frame store to re-encode**. V-JEPA cells
are produced straight from projected view stacks, so the projection lives inside
`encode_vjepa_cells.py` and the *entire cell build* is redone — including
re-projecting the aperture views, because the batch driver deletes each batch's
stacks after encoding.

### 5.1 Cell geometry

| parameter | value |
|---|---|
| variant | `2.1-vit-b-384`, 768-d, patch 16, tubelet 2 |
| source render | 2048 × 2048 |
| tile | 384, stride 384 |
| tile offsets | 0, 384, 768, 1152, 1536, **1664** — 6 per axis, **36 tiles/frame** |
| cells per tile | 384/16 = 24 |
| output grid | 2048/16 = **128 × 128** |
| clip length | **30** |
| PCA | 768 → 128, output float16 |

The final tile offset is pulled back to `extent − tile` (1664, not 1728) rather
than padded, so every tile is real image. That makes it overlap its neighbour,
which `stitch()` resolves by **averaging** the overlap.

The aperture is 31 views but tubelet 2 needs an even `T`, so **one view is
dropped** rather than duplicating a view and weighting it twice.

Two cells:
- **single** — the central view replicated 30× to fill the clip. No parallax, so
  the temporal axis carries nothing; this isolates the representation.
- **multi** — the real ±45 / stride-3 aperture, ortho-projected. The clip axis
  carries parallax and V-JEPA integrates it itself.

### 5.2 Where the projection is applied

To the **stitched (128, 128, 768) map**, before the PCA — not to raw tokens.

This is mathematically identical and cheaper. The projection is linear; the
stitched map is a mean over the temporal axis and over overlapping tiles;
projecting the mean equals the mean of the projections. Cost: **one matmul per
frame instead of 36**.

Applied at *both* call sites — the PCA fit and the encode loop. Missing the fit
site would fit the basis in the undebiased space and quietly invalidate the arm.

### 5.3 PCA retention

| cell | debiased | baseline |
|---|---|---|
| single | **91.8 %** | 95.2 % |
| multi | **94.2 %** | 97.2 % |

### 5.4 Numerical precision

fp32 weights with `torch.autocast` fp16. Casting V-JEPA 2.1's weights directly to
fp16 **fails** inside the vendored attention (query/key stay fp32 while value
casts). Autocast casts per operation and is consistent by construction.
Measured: **5.0× fp32 throughput, cosine 0.999991** against the fp32 reference —
both faster and more faithful than bf16 (0.999735).

### 5.5 Cell counts

| | single | multi |
|---|---|---|
| thermal, debiased | **10,663** | **10,663** |
| thermal, baseline | 10,663 | 10,663 |
| rgb, baseline | 9,221 | 9,221 |

Thermal debiased matches baseline **exactly** — the strongest available check
that the rebuild covered the same frames.

---

## 6. Training and evaluation

Held identical to every other cell in the 2×2, so the only variable is the
representation.

### 6.1 Head

CenterNet-style, **0.36 M parameters** for 128-channel input. A 3×3 → 3×3 trunk
feeding three 1×1 convolutions: heatmap (1), width/height (2), offset (2).

```
loss = focal(hm) + 5.0 · L1(wh) + 1.0 · L1(off)
```

Penalty-reduced focal loss, CenterNet formulation.

### 6.2 Optimisation

| | |
|---|---|
| optimiser | AdamW, lr 3e-4 |
| schedule | cosine |
| batch | 16 |
| epochs | ≤ 60, early stop patience 12 |
| precision | AMP autocast |
| seeds | **1337, 7, 42** |

Typical best epoch ~13, so patience 12 is the binding constraint, not the epoch
cap.

### 6.3 Cross-validation

Occlusion is a property of terrain and canopy, so **scene-to-scene** variation is
the error term that matters. 72 flights present in both modalities, partitioned
into **5 folds**, balanced greedily on hidden-box count so each fold carries
**2008–2010 hidden thermal boxes**. Each fold held out in turn; no frame from a
validation flight appears in training. 3 seeds per fold separates seed variance
from scene variance.

**30 runs per encoder × modality pair.**

### 6.4 Metrics

`scripts/occlusion_metrics.py`. IoU 0.5, confidence 0.3, **one-to-one greedy
matching** (an earlier matcher let one detection satisfy multiple GT boxes,
inflating recall ~10 %).

Three populations, with the **COCO ignore convention** — out-of-population
detections are ignored, not counted as false positives, so population rows are
not comparable to `all`:

- **visible** — animals visible in the central frame
- **occluded / hidden** — invisible in the central frame **and** the central view
  imaged that ground (masks: `{mod}_hidden_mask_all.json`)
- **all**

Logged per population: recall, precision, f1, ap50, ap75, ap50_95 — mean and
spread over seeds, per fold.

**Baseline and debiased arms are scored in the same pass, on the same fold, with
the same mask, matcher and threshold.** The pairing is therefore exact, which is
what licenses the fold-paired statistics.

Outputs: `metrics/debias32_{thermal,rgb}_f{0..4}.json`,
`metrics/vjdebias32_thermal_f{0..4}.json`.

### 6.5 Statistics

Paired *t* over the **5 folds** (df = 4, two-sided), each fold value being a mean
over 3 seeds. Bonferroni threshold for the 16 comparisons per pair:
0.05/16 = **0.003**.

The 16 metrics are **not independent** — recall, precision and AP derive from the
same detections, and the two cells share data. A sign test across them therefore
overstates significance and is not used as a p-value.

---

## 7. Results

### 7.0 Summary — all four encoder × modality pairs

Every number is debiased ÷ baseline, so **1.000 means the projection changed
nothing**. Five folds × three seeds per cell, paired by fold.

| encoder | modality | cell | vis recall | hidden recall | hidden prec | hidden AP50 |
|---|---|---|---|---|---|---|
| DINOv3 | thermal | single | 1.025 | 1.032 | 0.937 | 0.944 |
| DINOv3 | thermal | multi  | 1.013 | 1.025 | **1.037** | 1.043 |
| DINOv3 | RGB | single | 1.003 | 1.030 | 1.023 | 1.055 |
| DINOv3 | RGB | multi  | 1.018 | 1.044 | 1.046 | **1.157** |
| V-JEPA | thermal | single | 0.965 | **0.900** | 1.018 | 0.950 |
| V-JEPA | thermal | multi  | 1.015 | 1.017 | 0.983 | 1.028 |
| V-JEPA | RGB | single | 0.985 | 0.963 | 0.964 | 0.880 |
| V-JEPA | RGB | multi  | 1.013 | 1.030 | 0.991 | 1.026 |

**Bold** marks a ratio significant at p < 0.05 in a fold-paired test. Across the
full set of 4 pairs × 2 cells × 8 metrics = **64 comparisons, 5 reach p < 0.05**
— against 3.2 expected by chance at α = 0.05 — and **none survives Bonferroni**
correction, which requires p < 0.00078 while the smallest observed is p < 0.01.

| where | metric | ratio | p | folds |
|---|---|---|---|---|
| DINOv3 / thermal / multi | hidden precision | 1.037 | < .02 | 5/5 |
| DINOv3 / RGB / multi | hidden AP50:95 | 1.168 | < .05 | 4/5 |
| V-JEPA / thermal / single | hidden recall | 0.900 | < .02 | 0/5 |
| V-JEPA / thermal / single | visible AP50:95 | 0.951 | < .02 | 0/5 |
| V-JEPA / thermal / single | visible precision | 1.030 | < .01 | 5/5 |

Three things are worth reading off this table.

**The projection is inert overall.** No arm is changed enough to alter a
conclusion, and the significant cells do not agree with each other on sign: two
are positive, three negative. All main results therefore use the original,
undebiased features.

**Recall barely moves, but AP does.** Seven of eight visible-recall ratios sit
within 3 % of 1.000. The AP metrics do not: DINOv3 / RGB / multi gains **+15.7 %**
hidden AP50 while V-JEPA / RGB / single loses **−12.0 %**. Debiasing perturbs
box *localisation* considerably more than it perturbs *detection*, which is
consistent with removing a smooth positional component the regression heads were
partly leaning on.

**The single-view cells are where the harm is.** Every one of the eight ratios
below 0.97 sits on a single-view cell — not one multi-view cell drops that far
on any metric — and the strongest effect in the whole study is V-JEPA / thermal /
single losing a tenth of its hidden recall on 0 of 5 folds. Multi-view cells
trend mildly positive almost everywhere. A
plausible reading is that averaging 31 views already suppresses much of the
positional component — it is common to every view, but so is whatever else
survives — so an explicit projection has less left to remove and mostly costs
signal.

Per-pair detail, including the metrics not shown above, follows.

### 7.1 DINOv3 / thermal

| cell | vis recall | hidden recall | hidden prec | hidden AP50 |
|---|---|---|---|---|
| single | 0.6309 ± 0.0540 | 0.2125 ± 0.0924 | 0.1911 | 0.0738 |
| single, debiased | 0.6467 ± 0.0608 | 0.2194 ± 0.0917 | 0.1791 | 0.0697 |
| multi | 0.7107 ± 0.0687 | 0.3192 ± 0.1436 | 0.2059 | 0.1235 |
| multi, debiased | 0.7198 ± 0.0434 | 0.3271 ± 0.1209 | 0.2136 | 0.1288 |

All 16 differences between −6 % and +5 % of baseline. One reaches p < 0.02:
multi-view hidden **precision +0.0077 ± 0.0043**, winning 5/5 folds — does not
survive Bonferroni. The single cell **gains recall and loses precision**
(visible −0.0263, hidden −0.0120, both p < 0.1).

### 7.2 DINOv3 / rgb

| cell | vis recall | hidden recall | hidden prec | hidden AP50 |
|---|---|---|---|---|
| single | 0.5756 ± 0.0467 | 0.2007 ± 0.0788 | 0.1947 | 0.0692 |
| single, debiased | 0.5774 ± 0.0440 | 0.2068 ± 0.0834 | 0.1991 | 0.0730 |
| multi | 0.6576 ± 0.0483 | 0.3143 ± 0.1262 | 0.2126 | 0.1169 |
| multi, debiased | **0.6693 ± 0.0329** | **0.3282 ± 0.1251** | **0.2224** | **0.1352** |

**15 of 16 comparisons positive.** The two largest sit on the multi-view cell's
localisation of hidden animals:

| metric | delta | ratio | t(4) | p | folds won |
|---|---|---|---|---|---|
| hidden AP50 | +0.0182 | **1.156** | 2.38 | < .1 | 4/5 |
| hidden AP50:95 | +0.0051 | **1.168** | 2.86 | **< .05** | 4/5 |

### 7.3 V-JEPA / thermal

| cell | vis recall | hidden recall | hidden prec | hidden AP50 |
|---|---|---|---|---|
| single | 0.6917 ± 0.0575 | 0.2436 ± 0.0935 | 0.2032 | 0.0907 |
| single, debiased | 0.6673 ± 0.0696 | 0.2192 ± 0.0848 | 0.2068 | 0.0862 |
| multi | 0.7503 ± 0.0652 | 0.3752 ± 0.1510 | 0.2566 | 0.1800 |
| multi, debiased | 0.7619 ± 0.0624 | 0.3816 ± 0.1327 | 0.2522 | 0.1850 |

**The multi cell shows nothing** — all eight comparisons p > 0.10.

**The single cell trades recall for precision, strongly:**

| metric | delta | ratio | t(4) | p | folds won |
|---|---|---|---|---|---|
| hidden recall | −0.0244 | 0.900 | −4.14 | **< .02** | **0/5** |
| visible AP50:95 | −0.0132 | 0.951 | −4.07 | **< .02** | 0/5 |
| visible recall | −0.0245 | 0.965 | −2.65 | < .1 | 1/5 |
| visible precision | **+0.0190** | 1.030 | **6.53** | **< .01** | **5/5** |

### 7.5 Integration order (thermal, DINOv3, 128 dims)

The first parity comparison of the two orders on the cross-validated split: same
grid, dims, head, folds, seeds and frames.

| arm | vis recall | hidden recall | hidden prec | hidden AP50 |
|---|---|---|---|---|
| encode->average (`embed_multi`) | **0.7107 ± 0.0687** | **0.3192 ± 0.1436** | 0.2059 | **0.1235** |
| average->encode (`alfsembed_multi`) | 0.6705 ± 0.0602 | 0.2815 ± 0.1091 | 0.2087 | 0.1158 |
| average->encode, debiased | 0.6810 ± 0.0653 | 0.2886 ± 0.1144 | 0.2062 | 0.1191 |

**The two orders are not separable at n = 5** — nothing reaches p < 0.10. The
direction favours encode-then-average on recall (visible ratio 0.943, t = -1.93;
hidden ratio 0.882, t = -1.42), while precision and AP50:95 marginally favour
average-then-encode at ratios 1.01-1.02.

The useful reading: the two approaches are **near-equivalent detectors**, so the
multi-view benefit reported throughout comes from aperture integration itself,
not from where in the pipeline that integration happens.

Debiasing the average-then-encode arm behaves like the other multi-view cells —
hidden recall x1.025, AP50 x1.028, AP50:95 x1.043, each winning 4/5 folds, none
reaching p < 0.10.

**Fold 0 misled again.** It showed average-then-encode winning every metric; over
five folds the recall direction reverses. That is the second time in this study a
fold-0 read would have produced the wrong headline.

### 7.4 V-JEPA / rgb

| cell | vis recall | hidden recall | hidden prec | hidden AP50 |
|---|---|---|---|---|
| single | 0.5291 ± 0.0238 | 0.1813 ± 0.0724 | 0.1935 | 0.0725 |
| single, debiased | 0.5213 ± 0.0524 | 0.1746 ± 0.0782 | 0.1865 | **0.0638** |
| multi | 0.6087 ± 0.0544 | 0.3041 ± 0.1287 | 0.2396 | 0.1423 |
| multi, debiased | 0.6164 ± 0.0445 | 0.3132 ± 0.1345 | 0.2375 | 0.1460 |

The single cell is **negative on all eight comparisons** -- hidden AP50 x0.880
(0/5 folds, p < 0.1), hidden AP50:95 x0.851 (0/5). The multi cell is mildly
positive on 6 of 8, nothing significant.

---

## 8. What the results support

**The projection is not a dependable gain**, and all reported numbers in the
paper use undebiased features. But the four pairs show a consistent asymmetry
that is worth stating precisely.

### 8.1 The direction split, over 64 comparisons

| cell type | comparisons positive |
|---|---|
| **multi-view** | **28 / 32** |
| **single-view** | **11 / 32** |

Every individually reliable *positive* sits in a multi-view cell:

- DINOv3 / rgb, hidden AP50:95 **+16.8 %** (p < 0.05)
- DINOv3 / thermal, hidden precision +0.0077 (p < 0.02, 5/5 folds)

Every individually reliable *negative* sits in a single-view cell:

- V-JEPA / thermal, hidden recall **-10.0 %** (p < 0.02, 0/5 folds)
- V-JEPA / rgb, hidden AP50 -12.0 % (p < 0.1, 0/5 folds)

So multi-view cells never reliably lose and occasionally reliably gain;
single-view cells sometimes reliably lose.

### 8.2 A hypothesis, refuted then partly restored

After the first two pairs the pattern looked like *multi-view cells benefit,
single-view cells do not* (16/16 positive across both DINOv3 pairs). The proposed
mechanism: the multi-view cell averages features from 31 views, and a world point
lands at a **different patch position in each view**, so the positional component
does not cancel under integration -- removing it beforehand should help precisely
where features from different grid positions get mixed.

V-JEPA/thermal appeared to falsify this: the encoder with the far sharper
positional subspace (rank 1 = 89.5 % vs 44.9 %) showed nothing significant in its
multi cell, and this record previously concluded the pattern "does not extend".
**That was too strong on three pairs.** With the fourth, the asymmetry holds in
direction (28/32 vs 11/32) and the mechanism remains the best available
explanation -- but the magnitudes do not support it as a claim.

### 8.3 Why it is still not adopted

Effect sizes are 1-3 %. Only **2 of 64** comparisons reach p < 0.05 and **none**
survives Bonferroni correction (0.05/16 = 0.003 per pair). The honest statement
is that the projection is harmless to multi-view cells and can mildly help them,
while sometimes hurting single-view ones, and is too small and too inconsistent
to be part of the method.

### 8.4 Why the premise measurements misled

A subspace that is sharply low-rank, and that animal tokens occupy only about
half as strongly as background tokens, still does not imply that deleting it
helps a detector. The head is trained on whatever representation it is given, and
a linear component this predictable is one it can already discount -- removing it
in advance saves an adjustment the head was evidently making for itself.

That makes **subspace energy** the third measurement in this project that looked
diagnostic and was not, alongside PCA reconstruction error and aggregate mAP.

---

## 9. Limits of this study

### 9.1 Integration order — now measured for thermal

Since the first version of this record, the missing order was built and
cross-validated. The study now covers **both** orders for thermal:

- **encode-then-integrate** — `cell_<mod>_embed_multi`, features averaged after
  encoding;
- **integrate-then-encode** — `alfs_2k_<mod>`, the ALFS render built first and
  embedded (see 7.5).

Two facts made the baseline arm free rather than a rebuild. PCA components are
nested, so the first 128 channels of the existing 256-dim store **are** the
128-dim PCA, and `train_embedding_detector.py` truncates on read via
`--use-dims` (recorded per run as `use_dims: 128, stored_dims: 256`, which
`occlusion_metrics.py` reads back so evaluation truncates identically). And both
stores cover the **same 52,295 labelled stems**; the 469 extra ALFS frames carry
no labels and never enter training or validation, so the two orders see
identical data without any stem filtering.

Still open: **rgb**, and the single-view analogue (`project -> encode`, i.e.
embedding the geo-referenced ortho render). `SOURCE_DIRS` maps `ortho` to the
1024 px tree only, so a 2k ortho source would have to be added for that.

Two names still invite the wrong reading: `realalfs_multi` and
`realortho_single` are **sensor** cells (128x128x**3**, the production
renderer's PNGs downsampled, no encoder involved), not embedded ALFS.

### 9.2 Other limits

- **Rank was not swept.** Only r = 32. The DINOv3 spectrum (99.2 % at 32) and
  V-JEPA's (97.9 %) motivated it, but a lower rank — especially r = 1 for V-JEPA,
  where a single direction holds 89.5 % — is untested and is the cheapest
  remaining experiment.
- **One PCA sample.** Both arms use the same `{mod}_train_stems.txt`, a single
  train/val split rather than per-fold refitting. Deliberate: a different PCA
  sample would be a *second* difference between arms. It is still an
  approximation carried over from the baseline.
- **n = 5.** Fold-paired tests on five points have little power; only large,
  consistent effects can register.
- **Flight 159** produces no stacks and is absent from both V-JEPA arms
  (baseline included), so it is excluded identically — consistent, but the cause
  was not diagnosed.
- **No V-JEPA energy probe.** The animal-vs-background subspace-energy
  measurement (§3) was only run for DINOv3.

---

## 10. Defects and incidents

Recorded because several were silent, and because two produced results that
looked like completions.

| # | defect | consequence | resolution |
|---|---|---|---|
| 1 | `render_embedding_field.py --flight-ids` defaults to a **single flight**; the probe called it directly instead of via `integrate_dgx.sh` | 292 fields instead of 10,663; all 30 runs died `no (embedding, label) pairs` | route through `integrate_dgx.sh`, which enumerates labelled flights; `EMB` made overridable |
| 2 | each stage logged `FAILED` and continued, so the driver printed **`probe finished`** after total failure | completion signal was meaningless | verify by artefact counts, never by the completion line |
| 3 | `core.autocrlf=true` checked `.sh` out with CRLF; `scp` deployed `\r`; bash read `set -euo pipefail\r` as an invalid option and `set -e` aborted at line 15 | `integrate_dgx.sh` — used by other pipelines — silently broken by my own deploy | CRs stripped, files normalised, **`.gitattributes` pins `*.sh` to `eol=lf`** |
| 4 | `ls dir/*.npy \| wc -l` overflows the argument list past ~20k files and returns **0** | twice read as "the run lost everything" | all counters use `find` |
| 5 | status scripts grepped the **whole** log for errors | historical failures reported as live; monitor fired every poll forever | error counts scoped to the current driver start; monitors fire on error *increase* |
| 6 | `vjepa_batch.py` decides what to skip by counting existing cells | pointed at the complete **baseline** dirs it would declare every flight done and encode nothing | the `_debias32` suffix is applied **driver-side**, not left to the encoder |
| 7 | a neighbour's `llama-server` claimed GPUs 4–7 (29–39 GB each) mid-run | CUDA OOM on flights 13, 130, 131 | build relocated to GPUs 0–2; flights reprocessed cleanly |
| 8 | container GPU renumbering | OOM reported as "GPU 0" when running on physical GPU 4/6/7 | the tell: the *failing process* held only 1 GB while the card was full |
| 9 | early probe attempts missing `--config` / `--splits-json` / `--model-dir` | `PCA fit FAILED` ×2 | `COMMON=(…)` array mirrors `encode_dgx.sh` exactly |

**The generalisable lesson from #1, #3 and #6:** every one came from hand-writing
an invocation instead of reusing the known-good path, or from deploying without
verifying what landed. Verification is by **artefact count**, not by exit codes
or log lines.

---

## 11. Cost

| stage | thermal | rgb |
|---|---|---|
| DINOv3 re-encode | ~10 h (138,220 frames, 3 GPUs) | ~8 h (119,863 frames) |
| DINOv3 integrate + flatten | ~1.5 h | ~1.5 h |
| DINOv3 training (30 runs) | ~1.5 h | ~1.5 h |
| V-JEPA cell build | ~33 h wall (65 flights, 11 batches, ~22–28 min/flight) | ~35 h (est.) |
| V-JEPA training (30 runs) | ~3.7 h (~45 min/fold) | est. ~3.5 h |
| scoring | ~15 min | ~15 min |

Peak staged-stack footprint ~52 GB (6 flights × ~8.7 GB); `/scratch` stayed above
4.3 TB free throughout.

Basis fitting is negligible: **17.1 s** for V-JEPA.

---

## 12. Artefacts

**Code (new or modified for this study)**

| path | role |
|---|---|
| `scripts/fit_positional_basis_vjepa.py` | **new** — V-JEPA basis fitter |
| `scripts/encode_vjepa_cells.py` | `--debias-basis` / `--debias-rank`, driver-side-safe suffix, width guard, meta records the basis |
| `scripts/vjepa_batch.py` | `--debias-basis` / `--debias-rank`, suffixed output dirs, `--proj-workers` / `--proj-threads` / `--batch` passthrough |
| `scripts/encode_embeddings.py` | `--debias-basis` / `--debias-rank`, `project_out`, auto-suffixed output |
| `docker/vjepa_debias.sh` | **new** — V-JEPA probe: PCA fit → cell build → train |
| `docker/debias_probe.sh` | DINOv3 probe; `SKIP_ENCODE=1` resume; integrate via `integrate_dgx.sh`; `find`-based counters |
| `docker/score_debias.sh` | **new** — pairs debiased and baseline arms into one JSON per fold; `CELLS` selects the encoder |
| `docker/integrate_dgx.sh` | `EMB` overridable |
| `.gitattributes` | **new** — `*.sh text eol=lf` |

**Data (nothing existing overwritten — every artefact carries `_debias32`)**

```
embeddings/srcframes_{thermal,rgb}_debias32/     re-encoded per-frame stores
field_{mod}_feat_{single,full}_debias32/         integrated fields
embeddings/cell_{mod}_embed_{single,multi}_debias32/
embeddings/cell_{mod}_vjepa_{single,multi}_debias32/
embedding_runs_multiseed/*_debias32_f{0..4}_s{1337,7,42}/
zenodo_labels/vjepa_positional_basis.npz         + .json metadata
metrics/debias32_{thermal,rgb}_f{0..4}.json
metrics/vjdebias32_thermal_f{0..4}.json
```

Metrics JSONs mirrored to `D:\PipelineStep2\metrics\`.

**Reproduction**

```bash
# DINOv3, either modality
RANK=32 MOD=thermal BAMBI_GPUS="0 1 2" bash docker/debias_probe.sh
MOD=thermal RANK=32 CELLS="embed_single embed_multi" bash docker/score_debias.sh

# V-JEPA — basis first, then the probe
python scripts/fit_positional_basis_vjepa.py \
    --out .../zenodo_labels/vjepa_positional_basis.npz
MOD=thermal RANK=32 BAMBI_GPUS="0 1 2" bash docker/vjepa_debias.sh
MOD=thermal RANK=32 CELLS="vjepa_single vjepa_multi" TAG=vjdebias32 \
    bash docker/score_debias.sh
```
