#!/usr/bin/env bash
# Score the integration-order comparison and its debias variant in one pass.
#
# Three arms per fold, same labels/mask/matcher/threshold, so both comparisons
# are exactly paired:
#   encode-then-average  (cell_<mod>_embed_multi)   <- the existing CV cell
#   average-then-encode  (alfs_2k_<mod>)            <- the arm being added
#   average-then-encode, debiased
set -uo pipefail
ROOT=/scratch/bambi; DS=$ROOT/datasets/alfs_embed
MOD=${MOD:-thermal}; RANK=${RANK:-32}
MASK=$DS/zenodo_labels/${MOD}_hidden_mask_all.json
LOG=$ROOT/alfs_embed/logs/score_alfsembed_${MOD}.log
for k in 0 1 2 3 4; do
  arms="alfs_2k_${MOD}_alfsembed_multi_f${k}:alfs_2k_${MOD}"
  arms+=",alfs_2k_${MOD}_debias${RANK}_alfsembed_multi_debias${RANK}_f${k}:alfs_2k_${MOD}_debias${RANK}"
  arms+=",cell_${MOD}_embed_multi_embed_multi_f${k}:cell_${MOD}_embed_multi"
  echo "== fold $k" | tee -a "$LOG"
  docker run --rm --ipc=host --gpus "\"device=0\"" -v "$ROOT:$ROOT" \
    -w "$ROOT/alfs_embed/code" \
    -e PYTHONPATH="/opt/bambi:$ROOT/alfs_embed/code:$ROOT/alfs_embed/code/src:$ROOT/alfs_embed/code/alfspy_src" \
    bambi-embed:1.5 python scripts/occlusion_metrics.py \
      --runs "$DS/embedding_runs_multiseed" --embeddings "$DS/embeddings" \
      --yolo-datasets "$DS/yolo_datasets" --labels-dataset "alfs_2k_${MOD}_f${k}" \
      --modality "$MOD" --occlusion hidden --hidden-mask "$MASK" \
      --arms "$arms" --out "$DS/metrics/alfsembed_${MOD}_f${k}.json" 2>&1 | tee -a "$LOG"
done
echo "== scoring finished" | tee -a "$LOG"
