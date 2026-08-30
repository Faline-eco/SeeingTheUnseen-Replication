#!/usr/bin/env bash
# Score the cross-validated YOLO sweep, one JSON per fold.
#
# Both arms of a fold are scored in the same pass on the same labels, mask,
# matcher and threshold, so the single-vs-multi comparison is exactly paired --
# the same construction used for the embedding cells.
set -uo pipefail
ROOT=/scratch/bambi; DS=$ROOT/datasets/alfs_embed
MOD=${MOD:-rgb}
MASK=$DS/zenodo_labels/${MOD}_hidden_mask_all.json
LOG=$ROOT/alfs_embed/logs/score_yolo_cv_${MOD}.log
for k in 0 1 2 3 4; do
  arms="ortho_${MOD}_f${k}:ortho_${MOD}_f${k},alfs_${MOD}_f${k}:alfs_${MOD}_f${k}"
  echo "== fold $k" | tee -a "$LOG"
  docker run --rm --ipc=host --gpus "\"device=0\"" -v "$ROOT:$ROOT" \
    -w "$ROOT/alfs_embed/code" -e YOLO_CONFIG_DIR=/tmp \
    -e PYTHONPATH="/opt/bambi:$ROOT/alfs_embed/code:$ROOT/alfs_embed/code/src:$ROOT/alfs_embed/code/alfspy_src" \
    bambi-embed:1.5 python scripts/occlusion_metrics.py \
      --yolo-runs "$DS/yolo_runs_cv" --yolo-images "$DS/yolo_images" \
      --runs "$DS/embedding_runs_multiseed" --embeddings "$DS/embeddings" \
      --yolo-datasets "$DS/yolo_datasets" --labels-dataset "alfs_2k_${MOD}_f${k}" \
      --modality "$MOD" --occlusion hidden --hidden-mask "$MASK" \
      --arms "$arms" --out "$DS/metrics/yolocv_${MOD}_f${k}.json" 2>&1 | tee -a "$LOG"
done
echo "== scoring finished" | tee -a "$LOG"
