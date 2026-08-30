#!/usr/bin/env bash
# Score the INSID3-debiased embedding cells against their undebiased baseline.
#
# Both arms go into one JSON per fold so the comparison is like-for-like: same
# fold, same labels, same hidden mask, same matcher, same conf.
#
# Output names are new (debias32_thermal_f*.json). Nothing under metrics/ is
# overwritten -- the baseline thermal_f*.json files stay as they are.
set -uo pipefail

ROOT=/scratch/bambi
DS=$ROOT/datasets/alfs_embed
MOD=${MOD:-thermal}
RANK=${RANK:-32}
# CELLS picks the encoder: embed_* are the DINOv3 cells, vjepa_* the V-JEPA ones.
CELLS=${CELLS:-"embed_single embed_multi"}
TAG=${TAG:-debias${RANK}}
MASK=$DS/zenodo_labels/${MOD}_hidden_mask_all.json
LOG=$ROOT/alfs_embed/logs/score_${TAG}_${MOD}.log

dock() {
  docker run --rm --ipc=host --gpus '"device=0"' -v "$ROOT:$ROOT" \
    -w "$ROOT/alfs_embed/code" \
    -e PYTHONPATH="/opt/bambi:$ROOT/alfs_embed/code:$ROOT/alfs_embed/code/src:$ROOT/alfs_embed/code/alfspy_src" \
    bambi-embed:1.5 "$@"
}

for k in 0 1 2 3 4; do
  arms=""
  for c in $CELLS; do
    d=${c}_debias${RANK}
    # arm:ds -- arm is the run-dir prefix cv_folds.sh produced, ds is the cell dir
    arms+="cell_${MOD}_${d}_${d}_f${k}:cell_${MOD}_${d},"
    arms+="cell_${MOD}_${c}_${c}_f${k}:cell_${MOD}_${c},"
  done
  echo "== fold $k" | tee -a "$LOG"
  dock python scripts/occlusion_metrics.py \
    --runs "$DS/embedding_runs_multiseed" \
    --embeddings "$DS/embeddings" \
    --yolo-datasets "$DS/yolo_datasets" \
    --labels-dataset "alfs_2k_${MOD}_f${k}" \
    --modality "$MOD" --occlusion hidden --hidden-mask "$MASK" \
    --arms "${arms%,}" \
    --out "$DS/metrics/${TAG}_${MOD}_f${k}.json" 2>&1 | tee -a "$LOG"
done
echo "== scoring finished" | tee -a "$LOG"
