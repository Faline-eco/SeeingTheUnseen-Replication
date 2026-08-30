#!/usr/bin/env bash
# The missing integration order: "build the ALFS render first, then embed it"
# (average-then-encode), cross-validated at parity with the encode-then-average
# cell, plus its INSID3-debiased variant.
#
# Why this arm did not exist in the CV grid: `alfs_<mod>` was only ever built on
# the superseded single split at 64x64 / 64 dims, and the 2048px store
# `alfs_2k_<mod>` sits at 256 PCA dims. Neither is comparable to
# cell_<mod>_embed_multi at 128x128 / 128 dims.
#
# Two facts make the baseline arm free:
#   * PCA components are nested, so the first 128 channels of the 256-dim store
#     ARE the 128-dim PCA -- and train_embedding_detector.py truncates on read
#     via --use-dims. No re-encoding.
#   * alfs_2k_<mod> and cell_<mod>_embed_multi cover the SAME 52,295 labelled
#     stems (checked). The 469 extra ALFS frames carry no labels, so they never
#     enter training or validation and the two orders see identical data.
#
# Only the debiased variant needs an encode, because the projection must be
# applied before the PCA and the PCA refitted in the debiased space.
#
#   MOD=thermal BAMBI_GPUS="0 1 2" bash alfsembed_cv.sh
set -uo pipefail

ROOT=/scratch/bambi
DS=$ROOT/datasets/alfs_embed
MOD=${MOD:-thermal}
RANK=${RANK:-32}
DIMS=${DIMS:-128}
GPUS=${BAMBI_GPUS:-"0 1 2"}
SEEDS=${SEEDS:-"1337 7 42"}
EPOCHS=${EPOCHS:-40}          # matches cv_folds.sh
FOLDS=${FOLDS:-"0 1 2 3 4"}
BASIS=$DS/zenodo_labels/dino_positional_basis.npz
SRC=alfs_2k
DEB=${SRC}_${MOD}_debias${RANK}
LOG=$ROOT/alfs_embed/logs/alfsembed_cv_${MOD}.log
mkdir -p "$(dirname "$LOG")"

say() { echo "== $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

dock() {   # dock <gpu|none> <cmd...>
  local g=$1; shift
  local ga=()
  [[ "$g" != "none" ]] && ga=(--gpus "\"device=$g\"")
  docker run --rm --ipc=host "${ga[@]}" -v "$ROOT:$ROOT" -w "$ROOT/alfs_embed/code" \
    -e PYTHONPATH="/opt/bambi:$ROOT/alfs_embed/code:$ROOT/alfs_embed/code/src:$ROOT/alfs_embed/code/alfspy_src" \
    -e YOLO_CONFIG_DIR=/tmp bambi-embed:1.5 "$@"
}

# Same arguments encode_dgx.sh/debias_probe.sh use, so the only differences from
# the baseline store are --source and the projection.
COMMON=(--config "$ROOT/alfs_embed/code/config_dgx.yaml"
        --source "$SRC" --modality "$MOD"
        --out "$DS/embeddings" --pca-dim "$DIMS"
        --splits-json "$ROOT/alfs_embed/code/bambi_splits_reference.json"
        --model-dir "$DS/DINOv3")

say "embedded-ALFS CV, modality $MOD, dims $DIMS, debias rank $RANK"
[[ -f "$BASIS" ]] || { say "missing basis $BASIS"; exit 1; }

want=$(find "$DS/embeddings/${SRC}_${MOD}" -name '*.npy' 2>/dev/null | wc -l)
say "baseline store ${SRC}_${MOD}: $want cells (truncated to $DIMS on read)"

# ---- 1. encode the debiased variant ---------------------------------------
have=$(find "$DS/embeddings/$DEB" -name '*.npy' 2>/dev/null | wc -l)
if (( have < want )); then
  say "encoding $DEB ($have/$want present)"
  if [[ ! -f "$DS/embeddings/$DEB/pca.pkl" ]]; then
    say "  fitting PCA on debiased features"
    dock "$(awk '{print $1}' <<<"$GPUS")" python scripts/encode_embeddings.py \
        "${COMMON[@]}" --debias-basis "$BASIS" --debias-rank "$RANK" --fit-only \
        >>"$LOG" 2>&1 || { say "  PCA fit FAILED"; exit 1; }
  fi
  n_gpu=$(wc -w <<<"$GPUS"); i=0
  for g in $GPUS; do
    say "  shard $i/$n_gpu on GPU $g"
    dock "$g" python scripts/encode_embeddings.py "${COMMON[@]}" \
        --debias-basis "$BASIS" --debias-rank "$RANK" \
        --shard "$i" --num-shards "$n_gpu" >>"$LOG" 2>&1 &
    i=$((i+1))
  done
  wait
  say "  encoded $(find "$DS/embeddings/$DEB" -name '*.npy' 2>/dev/null | wc -l)/$want"
else
  say "$DEB already complete ($have cells)"
fi

# ---- 2. train both orders over the same folds ------------------------------
# arm := <label>:<embedding dataset>:<use-dims>:<labels dataset>
for k in $FOLDS; do
  arms="alfsembed_multi_f${k}:${SRC}_${MOD}:${DIMS}:${SRC}_${MOD}_f${k}"
  arms+=" alfsembed_multi_debias${RANK}_f${k}:${DEB}:${DIMS}:${SRC}_${MOD}_f${k}"
  say "fold $k / $MOD"
  MOD=$MOD SEEDS="$SEEDS" EPOCHS=$EPOCHS BAMBI_GPUS="$GPUS" BASE_ARMS="$arms" \
    bash "$ROOT/alfs_embed/docker/multiseed_dgx.sh" >>"$LOG" 2>&1 \
    && say "  done fold $k" || say "  FAILED fold $k"
done

say "== embedded-ALFS CV finished"
