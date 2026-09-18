#!/usr/bin/env bash
# Learned view aggregation, scene-level 5-fold CV, then paired scoring against
# the published mean-aggregated cell. One modality on one GPU, runs in series.
#
# Sources are read from the local RAID mirror (BAMBI_SRCFRAMES_ROOT); the grids,
# labels and outputs stay on /scratch so the scorer can run from either host.
#
#   bash viewagg_cv.sh thermal 6
set -uo pipefail

MOD=${1:-thermal}
GPU=${2:-0}
AGG=${AGG:-attn}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"1337 7 42"}
EPOCHS=${EPOCHS:-40}
HID=${HID:-32}
WIN=${WIN:-4}
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-8}
IMAGE=${IMAGE:-bambi-embed:1.5}
SRCROOT=${SRCROOT:-/raid/bambi/srcframes}
ROOT=/scratch/bambi
DS=$ROOT/datasets/alfs_embed
CODE=$ROOT/alfs_embed/code
OUTDIR=$DS/embedding_runs_multiseed
LOGDIR=$ROOT/alfs_embed/logs/viewagg
MASK=$DS/zenodo_labels/${MOD}_hidden_mask_all.json
mkdir -p "$LOGDIR" "$OUTDIR" "$DS/metrics"
LOG=$LOGDIR/viewagg_${MOD}.log
say() { echo "== $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

dock() {   # dock <args...>: one container run on our GPU with the RAID mirror
  docker run --rm --ipc=host --gpus "\"device=$GPU\"" \
    -v "$ROOT:$ROOT" -v "$SRCROOT:$SRCROOT" -w "$CODE" \
    -e PYTHONPATH="/opt/bambi:$CODE:$CODE/src:$CODE/alfspy_src" \
    -e BAMBI_SRCFRAMES_ROOT="$SRCROOT" \
    "$IMAGE" "$@"
}

say "modality $MOD gpu $GPU agg $AGG folds [$FOLDS] seeds [$SEEDS]"
fail=0
for k in $FOLDS; do
  for s in $SEEDS; do
    tag="embed_multi_${AGG}_f${k}_s${s}"
    run=$OUTDIR/viewgrid_${MOD}_${tag}
    if [[ -f "$run/summary.json" ]]; then say "skip $tag (done)"; continue; fi
    say "train $tag"
    dock python scripts/view_aggregator.py \
        --dataset "viewgrid_${MOD}" --labels-dataset "alfs_2k_${MOD}_f${k}" \
        --eval-labels "alfs_2k_${MOD}_f${k}_centralgt" \
        --embeddings "$DS/embeddings" --yolo-datasets "$DS/yolo_datasets" \
        --out "$OUTDIR" --agg "$AGG" --hid "$HID" --win "$WIN" --epochs "$EPOCHS" --batch "$BATCH" \
        --workers "$WORKERS" --seed "$s" --tag "$tag" \
        > "$LOGDIR/${MOD}_${tag}.log" 2>&1 \
      && say "  done $tag: $(python3 -c "import json;m=json.load(open('$run/summary.json'));print(f\"R {m['recall']:.4f} mAP50 {m['mAP50']:.4f} {m['minutes']} min\")")" \
      || { say "  FAILED $tag"; fail=1; }
  done
  say "score fold $k"
  arms="viewgrid_${MOD}_embed_multi_${AGG}_f${k}:viewgrid_${MOD}"
  arms+=",cell_${MOD}_embed_multi_embed_multi_f${k}:cell_${MOD}_embed_multi"
  dock python scripts/occlusion_metrics.py \
      --runs "$OUTDIR" --embeddings "$DS/embeddings" \
      --yolo-datasets "$DS/yolo_datasets" --labels-dataset "alfs_2k_${MOD}_f${k}" \
      --modality "$MOD" --occlusion hidden --hidden-mask "$MASK" \
      --arms "$arms" --out "$DS/metrics/viewagg_${AGG}_${MOD}_f${k}.json" \
      >> "$LOGDIR/score_${MOD}.log" 2>&1 \
    && say "  scored fold $k" || { say "  SCORING FAILED fold $k"; fail=1; }
done
say "finished (fail=$fail)"
exit $fail
