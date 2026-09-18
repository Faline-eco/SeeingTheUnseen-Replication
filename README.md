# Seeing the Unseen — replication package

Detecting animals that are **not visible in any single aerial frame**, and
measuring what actually recovers them.

Aerial wildlife monitoring fails where it matters most: under canopy. This is a
controlled study of how far occlusion can be compensated, along two axes —
**single view vs multi view** and **image space vs embedding space** — over
matched RGB and thermal drone imagery from 72 flights.

This package contains what is needed to reproduce the paper and nothing else.
[`REPLICATION.md`](REPLICATION.md) is the step-by-step; start there.

## The design

Every cell holds grid, detector head, folds, seeds and frames fixed, so a cell
differs from its neighbour along exactly one axis.

| | single view | multi view (synthetic aperture) |
|---|---|---|
| **image space** | orthorectified central frame | ALFS integral over 31 views |
| **embedding space** | DINOv3 / V-JEPA on one view | integrated feature field |

The multi-view input is **ALFS** (Airborne Light Field Sampling): ±45 frames at
stride 3 — 31 views over a ~9 m baseline — reprojected onto the DEM focus plane
and integrated, so occluders above the plane blur away while the ground stays
sharp.

Evaluation is **scene-level** five-fold cross-validation over the 72 flights,
balanced on hidden-animal count, three seeds per fold. No frame from a validation
flight appears in training. Claims are made on two explicit populations —
animals visible in the central view, and animals **hidden** in it although the
central view imaged their ground.

## What the study found

- **Aperture integration recovers hidden animals.** With a trained
  high-capacity image-space detector, hidden-animal recall rises **1.70× on RGB
  and 2.04× on thermal**, five of five folds in both, while recall on centrally
  visible animals is unchanged (0.995× and 0.997×). The effect is specific to
  occlusion, not a general detector improvement.
- **It does not depend on the representation.** In embedding space the
  multi-view factor is **1.50–1.68×** across two architecturally unrelated
  encoders and both sensors, even though which encoder leads flips between them.
- **Embeddings add a further, largely independent gain**, so the
  embedding/multi-view corner is strongest overall.
- **But the benefit must be trained for.** Every zero-shot OWL detector gets
  *worse* on integrated imagery (0.809× for the strongest). Fine-tuning its
  decoder removes that penalty but never delivers the gain (0.985×), while an
  end-to-end detector on the same renders reaches 1.696×. Three levels of
  adaptation, three outcomes.
- **Integration order barely matters.** "Render the ALFS image, then embed" and
  "embed each view, then integrate" are not separable at five folds.
- **Positional debiasing (INSID3) is inert.** Five of 64 comparisons reach
  p < 0.05 against 3.2 expected by chance, none surviving correction —
  [`docs/DEBIAS_STUDY.md`](docs/DEBIAS_STUDY.md).
- **Detector capacity buys a different population.** A tenfold larger embedding
  head improves 16 of 16 metrics, but the gains land on visible animals and on
  precision; multi-view hidden recall moves 1.2 %, non-significant —
  [`docs/HEAD_CAPACITY.md`](docs/HEAD_CAPACITY.md).
- **Learned view aggregation has a ceiling.** Replacing the mean over the 31
  views by attention pooling, a bidirectional GRU or a windowed transformer
  (27–90 k parameters, all starting as the mean) recovers **1.04–1.11×**
  thermal hidden recall and nothing significant on RGB; three architectures
  agree, so the missing hidden animals are upstream of aggregation. What
  spatial context buys instead is precision: the windowed transformer raises
  hidden AP50 1.16× in both modalities with no cost anywhere —
  [`docs/VIEW_AGGREGATION.md`](docs/VIEW_AGGREGATION.md).

Numbers, protocol and caveats: [`docs/RESULTS.md`](docs/RESULTS.md),
[`docs/METHODS.md`](docs/METHODS.md), [`docs/YOLO_METRICS.md`](docs/YOLO_METRICS.md).

## Layout

```
run.py, config*.yaml, src/georef/   rendering: ortho + ALFS from raw flights
scripts/                            pipeline, scoring, table generation
docker/                             the container the study ran in, and its drivers
metrics/                            every scored result, the evidence behind each table
docs/                               protocol and the per-experiment records
figures/                            the qualitative figures in the paper
bambi_splits.json                   the cross-validation split as used
```

## Trained models

Every checkpoint the study produced is on Hugging Face:

**<https://huggingface.co/cpraschl/SeeingTheUnseen>**

510 cross-validated runs — the embedding heads, the 120 learned-aggregator runs
and the 60-run YOLO26x grid, each with the per-epoch log and configuration it
was trained with — plus the PCA and INSID3 bases every arm depends on. Directory
names are the run identifiers used throughout this repository, so they join
directly against `metrics/*.json` under the `arms` key:

```
heads/cell_thermal_embed_multi_embed_multi_f0_s1337/    <- fold 0, seed 1337
heads/viewgrid_thermal_embed_multi_swin_f0_s1337/       <- learned aggregator (mean|attn|gru|swin)
yolo/ortho_thermal_f0_s1337/
```

Pick the checkpoint **by fold** and never pool across folds, or the held-out
guarantee is lost. A name without `_f<fold>` is one of 24 retained single-split
runs, which back the ground-truth-choice and PCA-width results only.

Backbones are **not** included: DINOv3 and V-JEPA 2.1 are used frozen and
unmodified, and come from their original sources.

## Reading the metrics honestly

Two things in `metrics/` will mislead if taken at face value, and both are
documented where they arise:

**Strict-IoU metrics favour the multi-view arm for a reason that is not
detection quality.** Both arms are scored against merged boxes — the hull of a
track across the whole aperture — and the ALFS render shows the animal smeared
along exactly that hull while the single view shows it at one instant. Hidden
AP75 comes out at 13.6× on thermal. The tell is that *visible* AP75 rises 2.68×
while visible recall sits at 0.997×, which occlusion cannot explain. Quote
recall, precision and F1 at IoU 0.5.

**The hidden population is defined geometrically, not by review.** A merged box
counts as hidden when it matches no central box at IoU 0.5 and the central view
covered that ground. About a quarter of the population has a central annotation
overlapping in the 0.3–0.5 band, i.e. an animal that *is* annotated centrally
whose merged box misses the threshold. `scripts/audit_hidden_population.py`
quantifies it; it has not been re-scored with those excluded.

## Licence

**MIT** ([`LICENSE`](LICENSE)) for everything here, and for the detection heads,
PCA bases and INSID3 bases published on Hugging Face.

**One exception.** The YOLO26x weights under `yolo/` on Hugging Face are
fine-tuned from Ultralytics YOLO26x and inherit its **AGPL-3.0** terms. If you
use those checkpoints, AGPL-3.0 applies to them; it does not reach the rest.

Imagery derives from the public BAMBI dataset under its own terms. DINOv3 and
V-JEPA 2.1 are used frozen and unmodified and are not redistributed here.
