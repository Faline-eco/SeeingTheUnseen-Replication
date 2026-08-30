#!/usr/bin/env bash
# YOLO capacity controls on the scene-level CV folds.
#
# Why: the staged YOLO datasets use the published split (flights 10/211/212/213),
# which is ~70 % one flight. On the embedding grid that split reported a 5.53x
# effect where CV says 2.49x, so any YOLO number from it is inflated the same way
# and cannot sit in a table beside the cross-validated grid.
#
# 4 arms x 5 folds x 3 seeds = 60 runs. Measured 2.2-5.4 h per run (mean ~3.7 h),
# so roughly 74 h on three GPUs. Resumable: a run whose results.png exists is
# skipped, so an interrupted sweep can simply be relaunched.
#
#   BAMBI_GPUS="0 1 2" bash yolo_cv.sh
set -uo pipefail

ROOT=/scratch/bambi
IMG=$ROOT/datasets/alfs_embed/yolo_images
OUT=$ROOT/datasets/alfs_embed/yolo_runs_cv
ARMS=${ARMS:-"ortho_thermal alfs_thermal ortho_rgb alfs_rgb"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"1337 7 42"}
EPOCHS=${EPOCHS:-100}
IMGSZ=${IMGSZ:-1024}
BATCH=${BATCH:-4}
GPUS=${BAMBI_GPUS:-"0 1 2"}
LOG=$ROOT/alfs_embed/logs/yolo_cv.log
mkdir -p "$(dirname "$LOG")" "$OUT"

# Job order is FOLD-MAJOR on purpose. Ultralytics writes labels/*.cache into the
# dataset directory, so two concurrent runs sharing a dataset race to build it —
# that is exactly what killed ortho_rgb_s1337 in the previous sweep. Varying the
# fold fastest guarantees the three jobs in a wave always touch three different
# dataset directories.
JOBS=()
for s in $SEEDS; do for a in $ARMS; do for k in $FOLDS; do JOBS+=("$a:$k:$s"); done; done; done

GPU_ARR=($GPUS); NG=${#GPU_ARR[@]}
echo "== $(date +%F\ %T) ${#JOBS[@]} runs on GPUs $GPUS" | tee -a "$LOG"

i=0
while (( i < ${#JOBS[@]} )); do
  pids=(); tags=(); dsets=()
  for (( g=0; g<NG && i<${#JOBS[@]}; g++, i++ )); do
    IFS=: read -r arm k seed <<<"${JOBS[$i]}"
    tag="${arm}_f${k}_s${seed}"
    if [[ -f "$OUT/$tag/results.png" ]]; then
      echo "   skip $tag (already finished)" | tee -a "$LOG"; ((g--)); continue
    fi
    # Assert the wave's datasets are distinct; silence here would mean the race.
    ds="${arm}_f${k}"
    for d in "${dsets[@]:-}"; do
      [[ "$d" == "$ds" ]] && echo "   WARNING: $ds twice in one wave (cache race)" >&2
    done
    dsets+=("$ds")
    echo "   -> GPU ${GPU_ARR[$g]}: $tag" | tee -a "$LOG"
    docker run --rm --ipc=host --gpus "\"device=${GPU_ARR[$g]}\"" \
      -v "$ROOT:$ROOT" -w "$ROOT/alfs_embed/code" -e YOLO_CONFIG_DIR=/tmp \
      bambi-embed:1.5 yolo detect train \
        model=$ROOT/bambi_models/yolo26x.pt data=$IMG/$ds/data.yaml \
        epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH seed=$seed patience=10 \
        project=$OUT name=$tag exist_ok=True verbose=False \
      >> "$LOG" 2>&1 &
    pids+=($!); tags+=("$tag")
  done
  for j in "${!pids[@]}"; do
    wait "${pids[$j]}" || echo "   ${tags[$j]} FAILED" | tee -a "$LOG"
  done
done

echo "== $(date +%F\ %T) YOLO CV finished" | tee -a "$LOG"
