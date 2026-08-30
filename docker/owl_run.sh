#!/usr/bin/env bash
# Zero-shot OWL baselines on the rgb aperture axis.
#
# Runs each released OWL model over both arms (single-view ortho vs multi-view
# ALFS) on every fold's validation set, so the aperture comparison is paired
# exactly as it is for our own detectors.
#
# The checkpoints are trained on public overhead datasets, NOT on BAMBI, so this
# is a zero-shot cross-domain baseline. That understates absolute performance
# but leaves the ortho-vs-alfs contrast intact -- which is the question.
#
# One model per GPU: OWL-D carries a ViT-H+/16 backbone and dominates wall time.
#
#   MODEL=owlc GPU=0 bash owl_run.sh
set -uo pipefail

ROOT=/scratch/bambi
OWL=$ROOT/owl
REPO=$OWL/MegaDetector-Overhead
PY=$OWL/venv/bin/python
DS=$ROOT/datasets/alfs_embed
MODEL=${MODEL:-owlc}
GPU=${GPU:-0}
ARMS=${ARMS:-"ortho_rgb alfs_rgb"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
LOG=$OWL/logs/${MODEL}.log
mkdir -p "$OWL/logs" "$OWL/data" "$OWL/runs"

case "$MODEL" in
  owlc) CFG=owlc_caribou_demo; PTH=$OWL/ckpt/OWL-C.pth ;;
  owlt) CFG=owlt_caribou_demo; PTH=$OWL/ckpt/OWL-T.pth ;;
  owld) CFG=owld_caribou_demo; PTH=$OWL/ckpt/OWL-D.pth ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

say() { echo "== $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
say "OWL $MODEL on GPU $GPU, arms: $ARMS"

for arm in $ARMS; do
  for k in $FOLDS; do
    IMGS=$DS/yolo_images/${arm}_f${k}/images/val
    LBLS=$DS/yolo_images/${arm}_f${k}/labels/val
    GT=$OWL/data/${arm}_f${k}_gt.csv
    OUT=$OWL/runs/${MODEL}_${arm}_f${k}

    [[ -d "$IMGS" ]] || { say "  MISSING $IMGS"; continue; }
    if [[ -f "$OUT/detections.csv" ]]; then
      say "  skip ${MODEL}_${arm}_f${k} (done)"; continue
    fi

    # GT is model-independent, so build once and reuse across the three models.
    if [[ ! -f "$GT" ]]; then
      "$PY" "$OWL/owl_make_gt.py" --images "$IMGS" --labels "$LBLS" --out "$GT" \
        >>"$LOG" 2>&1 || { say "  GT FAILED ${arm}_f${k}"; continue; }
    fi

    say "  ${arm}_f${k}: $(wc -l < "$GT") gt rows"
    rm -rf "$OUT"
    ( cd "$REPO" && OWL_DEMO_DATA=$OWL WANDB_MODE=disabled \
      CUDA_VISIBLE_DEVICES=$GPU "$PY" tools/test.py "test=$CFG" \
        ++test.device_name=cuda \
        ++test.model.pth_file="$PTH" \
        ++test.dataset.root_dir="$IMGS" \
        ++test.dataset.csv_file="$GT" \
        ++hydra.run.dir="$OUT" ) >>"$LOG" 2>&1 \
      && say "    done" || say "    FAILED ${MODEL}_${arm}_f${k}"
  done
done

say "== OWL $MODEL finished"
