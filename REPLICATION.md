# Replication

Two levels, depending on how far back you want to start.

**From the scored metrics** (minutes, no GPU). Every number in the paper is
recomputed from `metrics/*.json`, which are committed here. Start at §3.

**From raw data** (weeks of GPU time). Render, encode, train, score. §1–§2.

---

## 0. What you need that is not in this repository

| | where | why |
|---|---|---|
| BAMBI matched + orthographic subsets | Zenodo, see `docs/METHODS.md` | the 72 flights |
| `alfspy` | <https://github.com/bambi-eco/alfs_py> | light-field projection |
| DINOv3 ViT-H+/16 | Meta release | frozen encoder |
| V-JEPA 2.1 ViT-B-384 | Meta release | frozen encoder |
| YOLO26x COCO weights | Ultralytics | image-space detector init |
| OWL-C / OWL-T / OWL-D | Microsoft AI for Good | external baseline |
| trained checkpoints | <https://huggingface.co/cpraschl/SeeingTheUnseen> | skip training |

Encoders are used **frozen and unmodified**; nothing here fine-tunes them.

Build the environment from `docker/Dockerfile` (`bambi-embed:1.5`). Rendering is
the exception: it uses ModernGL and must run on a machine with a real GL driver,
because software rasterisation in a container has produced artefacts. Only the
rendered PNGs move to the GPU host.

## 1. Render

```bash
python run.py --config config_2k.yaml         # 2048 px ortho + 31-view ALFS
python scripts/fetch_zenodo_labels.py         # labels only, not the 90 GB of frames
```

Produces `matched_dataset_new/{geo-referenced2_2k,alfs_2k}/<flight>/<mod>/*.png`.
`config.yaml` is the earlier 1024 px configuration, kept because the YOLO arms
read a 1024 px export (see §4).

## 2. Datasets, folds and populations

```bash
python scripts/build_yolo_datasets.py         # image datasets + merged labels
python scripts/make_central_gt_val.py         # central-frame GT, for the matched scoring
python scripts/make_folds.py                  # scene-level 5-fold split, balanced on hidden count
python scripts/make_yolo_folds.py             # same split for the image datasets
python scripts/decompose_invisible.py --modality thermal --out <mask>.json
python scripts/decompose_invisible.py --modality rgb     --out <mask>.json
```

`make_folds.py` writes `cv_folds.json`; the split used in the paper is committed
as `bambi_splits.json`. `decompose_invisible.py` writes the hidden masks that
define the **hidden** population — a merged box with no central counterpart at
IoU 0.5 *and* whose ground the central view actually imaged. Verify with:

```bash
python scripts/verify_hidden_population.py    # 73.4 % thermal, 77.8 % rgb
```

## 3. Encode

```bash
bash docker/encode_dgx.sh                     # DINOv3 per-view fields
bash docker/integrate_dgx.sh                  # aperture integration -> embed_single / embed_multi
bash docker/vjepa_cells.sh                    # V-JEPA single / multi cells
python scripts/make_realalfs_cell.py          # image-space cells (128x128x3)
python scripts/render_embedding_field.py      # average-then-encode arm (integration order)
bash docker/viewgrid_dgx.sh                   # --emit grid: sampling geometry for the learned aggregators
python scripts/fit_positional_basis_dino.py   # INSID3 bases, for the debias study
python scripts/fit_positional_basis_vjepa.py
```

## 4. Train

```bash
bash docker/multiseed_dgx.sh                  # embedding heads, 5 folds x 3 seeds
bash docker/vjepa_debias.sh                   # debiased variants
bash docker/alfsembed_cv.sh                   # integration-order arm
bash docker/launch_viewagg.sh                 # learned view aggregation, AGG=mean|attn|gru|swin (viewagg_cv.sh)
bash docker/yolo_cv.sh                        # YOLO26x, 60 runs
bash docker/owl_run.sh                        # OWL zero-shot
bash docker/owl_finetune.sh                   # OWL-D fine-tune
```

**Resolution differs by arm and this is deliberate.** The embedding arms encode
the 2048 px PNG renders; the YOLO arms read a 1024 px JPEG export. Both YOLO arms
use the same export, so their ratios are internally controlled, but absolute
values are not comparable across the two families. See `docs/YOLO_METRICS.md`.

## 5. Score

One scorer for every arm, so nothing differs but the arm:

```bash
python scripts/occlusion_metrics.py \
    --runs <runs> --embeddings <cells> --yolo-datasets <datasets> \
    --labels-dataset alfs_2k_thermal_f0 --modality thermal \
    --occlusion hidden --hidden-mask <mask>.json \
    --arms "<name>:<dataset>,..." --out metrics/thermal_f0.json
```

Detections are matched greedily in confidence order, one per ground-truth box,
at IoU 0.5 and confidence 0.3. Evaluating a population subset **ignores**
detections that match an out-of-population animal rather than charging them as
false positives (the COCO crowd convention), so population rows are not
comparable with `all`.

OWL is scored differently — a detection is correct if its point falls inside a
box — by `scripts/owl_point_metrics.py`. Only its ratios are comparable with the
rest.

## 6. Which table comes from which file

| paper table | metrics | regenerate |
|---|---|---|
| 2×2 hidden recall | `thermal_f*.json`, `rgb_f*.json` | `scripts/prec_table.py` |
| 2×2 visible recall | same | `scripts/prec_table.py` |
| precision and AP | same | `scripts/prec_table.py` |
| encoders (DINOv3 / V-JEPA) | `vjepa_{thermal,rgb}_f*.json` | — |
| YOLO26x cross-validation | `yolocv_{thermal,rgb}_f*.json` | `scripts/yolo_md.py` |
| integration order | `alfsembed_thermal_f*.json` | — |
| positional debiasing | `debias32_*`, `vjdebias32_*` | `docs/DEBIAS_STUDY.md` §7.0 |
| OWL baseline | `owl_rgb_f*.json`, `owlft_rgb_f*.json` | `docs/OWL_BASELINE.md` |
| ground-truth choice | `yolo_thermal.json`, `yolo_rgb.json` | single split, see `docs/PART2_YOLO_THERMAL.md` |
| head capacity | `w512_thermal_f*.json` | `scripts/cap_table.py` |
| learned view aggregation | `viewagg_{mean,attn,gru,swin}_{thermal,rgb}_f*.json` | `scripts/viewagg_arms.py attn gru swin` |

`scripts/ckwidth.py` reads head geometry back out of a checkpoint, which is how
the head-capacity claims are checked against the runs rather than against the
plan that produced them.

## 7. Figures

```bash
python scripts/make_example_figures.py --modality thermal --flight 70 --fold 1 \
    --stems 70_001360,70_001370,70_001640,70_001350 --out figures/examples
python scripts/make_smear_figure.py --flight 192 --frame 000640 --fold 3 \
    --modality thermal --out figures/smear
```

Frames must come from the chosen fold's **validation** split; a frame drawn with
a head that trained on it demonstrates nothing. Flight 70 is in fold 1, flight
192 in fold 3.

## 8. Known limitations of this package

- The single-split results behind the ground-truth table predate the
  cross-validation and use the published 4-flight split. They are reported as
  such and are not part of the main claims.
- The head-capacity ablation is thermal-only.
- Arm A (unregistered aperture into V-JEPA) is not included; its validation gate
  did not pass and no result is claimed from it.
- Roughly a quarter of the hidden population has a central annotation that
  overlaps at IoU 0.3–0.5, i.e. below the 0.5 threshold that defines the
  population. `scripts/audit_hidden_population.py` quantifies this. It has not
  been re-scored with those boxes excluded.
