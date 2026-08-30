#!/usr/bin/env bash
# Fine-tune OWL-D on our folds, then evaluate exactly as the zero-shot models
# were evaluated.
#
# Why this run exists: zero-shot OWL does WORSE on ALFS than on ortho, while our
# own trained detectors do 1.70x BETTER. The hypothesis is domain shift -- the
# integral image is unlike the aerial photographs OWL was trained on. Fine-tuning
# on aperture imagery is the direct test: if the aperture benefit returns, the
# zero-shot deficit was familiarity, not information.
#
# The backbone stays frozen (as published), so only the DPT decoder adapts. That
# also makes this the same design as our own embedding cells -- frozen DINOv3
# ViT-H+/16 plus a trained head -- differing only in decoder and target.
#
# Evaluation deliberately runs on FULL 1024px validation frames via the stitcher,
# not on patches, so the numbers sit alongside the zero-shot ones unchanged.
#
#   WORKER=0 GPU=0 bash owl_finetune.sh
set -uo pipefail

OWL=/scratch/bambi/owl
REPO=$OWL/MegaDetector-Overhead
PY=$OWL/venv/bin/python
DS=/scratch/bambi/datasets/alfs_embed
WORKER=${WORKER:-0}
NWORKERS=${NWORKERS:-3}
GPU=${GPU:-0}
CFG_PREFIX=${CFG_PREFIX:-bambi_owld_ft}   # hydra config name prefix
TAG_PREFIX=${TAG_PREFIX:-owld_ft}         # run/eval directory prefix
FOLDS=${FOLDS:-"0 1 2 3 4"}
LOG=$OWL/logs/${TAG_PREFIX}_w${WORKER}.log
mkdir -p "$OWL/logs" "$OWL/ft_runs" "$OWL/ft_eval"

say() { echo "== $(date +%H:%M:%S) [w$WORKER] $*" | tee -a "$LOG"; }

# The patch sets are built by patch_all.sh; starting before it finishes would
# train on a half-written directory.
while ! grep -q "patching finished" "$OWL/logs/patch_all.log" 2>/dev/null; do
  say "waiting for patching to finish"
  sleep 300
done
say "patches ready"

i=0
for arm in ortho_rgb alfs_rgb; do
  for k in $FOLDS; do
    if (( i % NWORKERS != WORKER )); then i=$((i+1)); continue; fi
    i=$((i+1))
    TAG=${TAG_PREFIX}_${arm}_f${k}
    RUN=$OWL/ft_runs/$TAG
    EVAL=$OWL/ft_eval/$TAG

    # ---- train ----------------------------------------------------------
    if [[ -f "$RUN/best_model.pth" ]]; then
      say "$TAG: checkpoint present, skipping train"
    else
      say "$TAG: training"
      rm -rf "$RUN"
      ( cd "$REPO" && WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$GPU \
        "$PY" tools/train.py "train=${CFG_PREFIX}_${arm}_f${k}" \
          ++hydra.run.dir="$RUN" ) >>"$LOG" 2>&1 \
        || { say "  TRAIN FAILED $TAG"; continue; }
      [[ -f "$RUN/best_model.pth" ]] || { say "  NO CHECKPOINT $TAG"; continue; }
      say "  trained"
    fi

    # ---- evaluate on full frames, same protocol as zero-shot -------------
    if [[ -f "$EVAL/detections.csv" ]]; then
      say "$TAG: detections present, skipping eval"; continue
    fi
    IMGS=$DS/yolo_images/${arm}_f${k}/images/val
    GT=$OWL/data/${arm}_f${k}_gt.csv          # inference GT, with sentinels
    [[ -f "$GT" ]] || { say "  MISSING inference GT $GT"; continue; }
    rm -rf "$EVAL"
    ( cd "$REPO" && OWL_DEMO_DATA=$OWL WANDB_MODE=disabled \
      CUDA_VISIBLE_DEVICES=$GPU "$PY" tools/test.py test=owld_caribou_demo \
        ++test.device_name=cuda \
        ++test.model.pth_file="$RUN/best_model.pth" \
        ++test.dataset.root_dir="$IMGS" \
        ++test.dataset.csv_file="$GT" \
        ++hydra.run.dir="$EVAL" ) >>"$LOG" 2>&1 \
      && say "  evaluated" || say "  EVAL FAILED $TAG"
  done
done

say "== worker $WORKER finished"
