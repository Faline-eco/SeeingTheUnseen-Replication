# Microsoft OWL as an external baseline

An independently-developed, point-based aerial wildlife detector applied to the
rgb aperture axis. Companion to `RESULTS.md` §6.13.

Status: **complete** -- zero-shot, fine-tuned, and an unfrozen-backbone probe
that did not support the explanation we proposed for the fine-tuned result.

---

## 1. Why this baseline

Every detector in the 2x2 is ours: a CenterNet head we wrote, or YOLO26x we
fine-tuned. OWL (Overhead Wildlife Locator, arXiv 2606.13911, Microsoft AI for
Good, June 2026) is external, published, and trained by other people on other
data. Three released variants:

| model | backbone | file | notes |
|---|---|---|---|
| OWL-C | DLA-34 | `OWL-C.pth` (207 MB) | HerdNet detection branch; the baseline |
| OWL-T | DLA-34 + Swin | `OWL-T.pth` (339 MB) | windowed self-attention for cluttered scenes |
| **OWL-D** | **DINOv3 ViT-H+/16 + DPT decoder** | `OWL-D.pth` (3.3 GB) | ~840 M backbone |

**OWL-D uses the same frozen backbone as our embedding cells.** That makes it a
decoder comparison at fixed representation: their DPT-style density head against
our CenterNet head, on DINOv3 ViT-H+/16 in both cases.

MIT licence; checkpoints on Zenodo record 20802844. The general checkpoints are
trained on public overhead datasets, **not** BAMBI, so this is zero-shot.

---

## 2. Two traps found while wiring it up

**`tools/infer.py` is legacy and cannot load these checkpoints.** It is hardcoded
to a 7-class HerdNet (`assert num_classes == 7`, hardcoded class names and
ImageNet normalisation) and reads `model.cls_head.2.weight`. The OWL checkpoints
carry only `model.loc_head.*` with `classes = {1: 'animal'}`. The supported path
is `tools/test.py` with the registry configs, which is what Microsoft's own
`demo_owl_models.sh` uses.

**Detections are written at heatmap scale, not image scale.** The stitcher runs
with `down_ratio: 2, up: False`, so saved coordinates must be multiplied by 2.
Their evaluator hides this by downsampling the ground truth to match
(`DownSample(down_ratio=2)`). Scored raw against our boxes, recall collapses to
~0 and OWL would have looked simply broken on our imagery.

Verification: after the x2 mapping, 44.6% of GT points fall within 20 px of a
detection, matching the authors' own evaluator reporting recall 0.4385 on the
same images.

---

## 3. Protocol

Inference via `docker/owl_run.sh`: 3 models x 2 arms x 5 folds, each fold's own
validation set, one model per A100. Images are 1024x1024, tiled 512 with overlap
160 (3x3 = 9 tiles), LMDS peak detection (kernel 3x3, adapt_ts 0.3) -- all as
the authors' configs specify, identical across the three models.

**Scoring is point-in-box** (`scripts/owl_point_metrics.py`): a detection hits
when its point lies inside a ground-truth box, matched one-to-one and greedily by
descending confidence. Where a point falls in several boxes the smallest is
taken, so a tight box is not stranded by an enclosing one.

The population construction, hidden mask and COCO ignore rule are **imported**
from `occlusion_metrics.py` rather than reimplemented, so OWL and our detectors
see exactly the same ground truth.

> **Point-in-box is more permissive than IoU 0.5.** Numbers here are comparable
> with each other, *not* with the IoU-based tables elsewhere. The companion pass
> that re-scores our own detectors under this same rule (box centres as points)
> is still outstanding.

---

## 4. Zero-shot results (rgb, 5 folds)

| model / arm | vis recall | hidden recall | hidden prec | hidden F1 |
|---|---|---|---|---|
| OWL-C / ortho | 0.2452 ± 0.0821 | 0.1481 ± 0.0496 | 0.2542 | 0.1814 |
| OWL-C / alfs | 0.1840 ± 0.0937 | 0.0887 ± 0.0391 | 0.1836 | 0.1154 |
| OWL-T / ortho | 0.2131 ± 0.0624 | 0.1401 ± 0.0666 | 0.2589 | 0.1745 |
| OWL-T / alfs | 0.1536 ± 0.0795 | 0.0800 ± 0.0518 | 0.1626 | 0.1044 |
| **OWL-D / ortho** | **0.5128 ± 0.0574** | **0.3265 ± 0.1121** | **0.4246** | **0.3631** |
| OWL-D / alfs | 0.4788 ± 0.0823 | 0.2643 ± 0.0934 | 0.3317 | 0.2906 |

Paired by fold, ortho -> alfs:

| model | hidden recall | hidden precision | hidden F1 | folds won |
|---|---|---|---|---|
| OWL-C | x0.599 | x0.722 (p<.02) | x0.636 (p<.1) | 1/5, 0/5, 1/5 |
| OWL-T | x0.571 | x0.628 (p<.02) | x0.598 (p<.1) | 1/5, 0/5, 1/5 |
| OWL-D | x0.809 (p<.05) | x0.781 (p<.01) | x0.801 (p<.02) | **0/5, 0/5, 0/5** |

---

## 5. What it means

**The aperture hurts every zero-shot model**, consistently and significantly.
That is the opposite of our trained detectors, where ALFS gives **x1.70 hidden
recall winning 5/5 folds** (§6.12).

The two results together isolate the cause, and the control is internal: **the
same ALFS images that a zero-shot OWL does worse on give a trained YOLO a 70%
gain.** The imagery is therefore not degraded -- the integral image is simply out
of distribution for a detector trained on conventional aerial photographs, being
blurred and artefacted in ways ordinary drone imagery is not.

> **Synthetic-aperture integration is not free preprocessing that can be bolted
> onto an off-the-shelf detector. The benefit requires training on aperture
> imagery.**

That is a practically important caveat: it says *when* the method pays off, and
it explains why a practitioner who simply runs an existing detector over ALFS
renders would conclude the technique does not work.

**Second finding.** OWL-D is far the strongest zero-shot model -- 0.3265 hidden
recall against 0.148 and 0.140 for the DLA-34 variants. Its frozen DINOv3
ViT-H+/16 is the backbone our own embedding cells use, so an independent group's
results corroborate that representation choice.

**Limits.** Zero-shot, rgb only. Section 6 tests the explanation directly.

---

## 6. Fine-tuning OWL-D: the penalty goes, the benefit does not arrive

The zero-shot deficit was attributed to unfamiliarity. Fine-tuning is the direct
test: OWL-D was fine-tuned from the released checkpoint on each fold's training
split, both arms, backbone frozen exactly as published, then evaluated on full
1024 px frames with the same stitcher and the same point-in-box scorer.

Recipe changes from the authors' benchmark config, kept to three: `load_from`
set to `OWL-D.pth` (fine-tune their decoder rather than train one), the dataset
paths, and epochs 20 -> 10. Optimiser, LR, batch, loss, augmentations, frozen
backbone and evaluator are as published.

| arm | vis recall | hidden recall | hidden prec | hidden F1 |
|---|---|---|---|---|
| ortho | 0.5903 ± 0.0444 | 0.3038 ± 0.1368 | **0.5089** | 0.3610 |
| alfs | 0.6120 ± 0.0596 | 0.2992 ± 0.1263 | 0.3788 | 0.3250 |

### The three-point picture

| detector | what adapts | hidden recall alfs/ortho | p | folds won |
|---|---|---|---|---|
| OWL-D zero-shot | nothing | **x0.809** hurts | < .05 | 0/5 |
| OWL-D fine-tuned | decoder only, backbone frozen | **x0.985** neutral | ns | 2/5 |
| YOLO26x | end to end | **x1.696** helps | < .05 | 5/5 |

**Fine-tuning removes the penalty but does not deliver the benefit.** The
zero-shot deficit -- losing every fold -- disappears, so that part *was*
unfamiliarity. But the aperture gain never appears; x0.985 is exactly neutral.
Precision on hidden animals is significantly worse on ALFS (x0.744, p<0.01,
0/5), so the fine-tuned model produces more false positives on integral imagery.

### The explanation we proposed, and why we no longer lean on it

We froze the DINOv3 backbone as published, so only the DPT decoder adapted, while
YOLO26x adapts end to end. The natural reading was that extracting the aperture
benefit requires the **representation** to adapt, not just the head.

Section 6.1 tests that directly. It does not support it.

### 6.1 Unfrozen-backbone probe: does not support the explanation

Two folds (0 and 3) x two arms, backbone trainable, backbone LR x0.1. Batch had
to drop 16 -> 4 and epochs 10 -> 3: unfreezing 840 M parameters needs
\SI{34.8}{\gigayte} of a \SI{41}{\gigayte} card even at batch 4, and the
smaller batch quadruples the step count.

| fold | backbone | hidden ortho | hidden alfs | ratio |
|---|---|---|---|---|
| 0 | frozen | 0.0845 | 0.1012 | **1.198** |
| 0 | unfrozen | 0.0971 | 0.0956 | **0.984** |
| 3 | frozen | 0.3405 | 0.4433 | **1.302** |
| 3 | unfrozen | 0.4044 | 0.4265 | **1.055** |

Mean over the probe folds: frozen **1.281**, unfrozen **1.041**.

**Unfreezing made the aperture advantage smaller, not larger** -- the opposite of
the prediction. It improved the model overall (ortho hidden recall 0.2125 ->
0.2508) but improved the *ortho* arm more than the ALFS arm, closing the gap.

**Read this cautiously; it is a probe, not a measurement.**

- The unfrozen runs differ in batch size and epochs as well as the backbone, so
  they are undertrained relative to the frozen ones. "Unfrozen" is confounded
  with "less training". The clean control -- frozen at batch 4 for 3 epochs --
  was not run; we chose to stop spending time here.
- The two folds are unrepresentative. The frozen model's ratio on folds 0 and 3
  is 1.281, against 0.985 across all five. These are the two folds most
  favourable to ALFS, which is partly why they were picked.

What can be said: **there is no evidence that unfreezing recovers the YOLO-like
aperture gain, and the trend runs against it.** The frozen backbone is therefore
not demonstrably the reason fine-tuned OWL-D fails to benefit from the aperture.

Remaining untested candidates: point/density supervision versus box regression,
patch-based training versus full frames, and the DPT decoder itself.

Metrics: `metrics/owlunfroz_rgb_f{0,3}.json`.


### A gap this incidentally fills

`DEBIAS_STUDY.md` section 9.1 notes a missing `project -> encode` single-view
arm. Fine-tuned OWL-D is effectively that comparison -- frozen DINOv3 plus a
trained head, applied to ortho vs ALFS *pixel* images -- and it says the two are
equivalent for hidden recall.

Metrics: `metrics/owlft_rgb_f{0..4}.json`. Configs generated by
`scripts/owl_make_configs.py`; driver `docker/owl_finetune.sh`.

### Defects hit while wiring the fine-tune

| defect | consequence |
|---|---|
| `train.py` reads `cfg.model.freeze` **directly**, not via `getattr`, inside the `load_from` branch | all 10 runs died instantly; the from-scratch benchmark config we derived from has `load_from: null` and never enters that branch, so it lacks the key. Their *fine-tuning* configs carry `freeze: null` |
| `patcher.py` uses `PadIfNeeded.PositionType`, removed in albumentations 2.x (and `value` renamed `fill`) | patching produced zero patches; fixed locally, original kept as `patcher.py.orig` |
| our inference GT writes a sentinel point at (0,0) for empty frames | as a *training* target that is a phantom animal in the corner of every empty frame; `--no-empty` added for training GT |
| status probe grepped whole worker logs | the 10 failures of the first attempt counted as live and would have ended the monitor before training began |

---

## 7. Artefacts

```
/scratch/bambi/owl/                     environment, checkpoints, runs
  MegaDetector-Overhead/                upstream repo (MIT)
  ckpt/OWL-{C,T,D}.pth                  Zenodo record 20802844
  runs/<model>_<arm>_f<k>/detections.csv
metrics/owl_rgb_f{0..4}.json            point-in-box scores, zero-shot
metrics/owlft_rgb_f{0..4}.json          point-in-box scores, fine-tuned OWL-D
metrics/owlunfroz_rgb_f{0,3}.json       point-in-box scores, unfrozen probe
  ft_runs/<tag>/best_model.pth          fine-tuned checkpoints
  ft_eval/<tag>/detections.csv          full-frame evaluation
scripts/owl_make_gt.py                  YOLO labels -> OWL point CSV
scripts/owl_point_metrics.py            point-in-box scorer on our populations
docker/owl_run.sh                       inference driver
```
