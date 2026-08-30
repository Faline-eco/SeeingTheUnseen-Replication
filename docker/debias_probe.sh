#!/usr/bin/env bash
# INSID3 spatial-debiasing probe for the DINOv3 embedding cells (thermal).
#
# Why both cells and not just the single-view one: cell_*_embed_single and
# cell_*_embed_multi are both derived from srcframes_<mod>, the per-frame DINOv3
# store, which is already PCA-reduced to 128 dims. The projection must be applied
# BEFORE that PCA, so the 138k-frame re-encode is unavoidable and is shared by
# both cells -- adding the multi cell costs one extra integration and training
# pass rather than a second re-encode.
#
# Nothing existing is overwritten: encode_embeddings.py derives
# srcframes_<mod>_debias<rank>, and the field and cell outputs carry the same
# suffix. The undebiased store is the baseline the probe is measured against.
#
#   RANK=32 bash debias_probe.sh
set -uo pipefail

ROOT=/scratch/bambi
DS=$ROOT/datasets/alfs_embed
MOD=${MOD:-thermal}
RANK=${RANK:-32}
GPUS=${BAMBI_GPUS:-"0 1 2"}
SEEDS=${SEEDS:-"1337 7 42"}
BASIS=$DS/zenodo_labels/dino_positional_basis.npz
LOG=$ROOT/alfs_embed/logs/debias_probe_${MOD}.log
SRC=srcframes_${MOD}_debias${RANK}

say() { echo "== $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

dock() {   # dock <gpu|none> <cmd...>
  local g=$1; shift
  local ga=()
  [[ "$g" != "none" ]] && ga=(--gpus "\"device=$g\"")
  docker run --rm --ipc=host "${ga[@]}" -v "$ROOT:$ROOT" -w "$ROOT/alfs_embed/code" \
    -e PYTHONPATH="/opt/bambi:$ROOT/alfs_embed/code:$ROOT/alfs_embed/code/src:$ROOT/alfs_embed/code/alfspy_src" \
    -e TORCH_HOME=$ROOT/vjepa2/torch_hub bambi-embed:1.5 "$@"
}

# Exactly the arguments encode_dgx.sh uses for the baseline store, so the only
# difference between this encode and the undebiased one is the projection.
# Omitting --config sent the encoder to the Windows paths in the default config,
# where it found no frames at all.
COMMON=(--config "$ROOT/alfs_embed/code/config_dgx.yaml"
        --source srcframes --modality "$MOD"
        --out "$DS/embeddings" --pca-dim 128
        --splits-json "$ROOT/alfs_embed/code/bambi_splits_reference.json"
        --model-dir "$DS/DINOv3")

say "INSID3 probe, rank $RANK, modality $MOD"
[[ -f "$BASIS" ]] || { say "missing basis $BASIS"; exit 1; }

# ---- 1. re-encode the per-frame store with the positional subspace removed --
# --fit-only first: every shard must share one PCA basis, and that basis has to
# be fitted on debiased features rather than reused from the undebiased store.
if [[ ! -f "$DS/embeddings/$SRC/pca.pkl" ]]; then
  say "fitting PCA on debiased features"
  dock 0 python scripts/encode_embeddings.py "${COMMON[@]}" \
      --debias-basis "$BASIS" --debias-rank "$RANK" --fit-only >>"$LOG" 2>&1 \
    || { say "PCA fit FAILED"; exit 1; }
fi

n_gpu=$(wc -w <<<"$GPUS")
# SKIP_ENCODE=1 resumes from a completed encode. The re-encode is ~10h and its
# output is verifiable by frame count, so a downstream failure should not cost it.
if [[ "${SKIP_ENCODE:-0}" == 1 ]]; then
  say "SKIP_ENCODE=1, reusing $(find "$DS/embeddings/$SRC" -name '*.npy' 2>/dev/null | wc -l) encoded frames"
else
  i=0
  for g in $GPUS; do
    say "encode shard $i of $n_gpu on GPU $g"
    dock "$g" python scripts/encode_embeddings.py "${COMMON[@]}" \
        --debias-basis "$BASIS" --debias-rank "$RANK" \
        --shard "$i" --num-shards "$n_gpu" >>"$LOG" 2>&1 &
    i=$((i+1))
  done
  wait
  say "encoded $(find "$DS/embeddings/$SRC" -name '*.npy' 2>/dev/null | wc -l) frames"
fi

# ---- 2. project onto the ortho canvas: single view, then the full aperture --
# Via integrate_dgx.sh rather than a direct render_embedding_field call: that
# script enumerates the labelled flights and passes --flight-ids. Calling the
# renderer directly leaves --flight-ids at its single-flight default, which
# produced 292 fields instead of 10663 and made the training stage fail on every
# fold with almost no data.
for ap in single full; do
  say "integrating aperture=$ap"
  EMB="$DS/embeddings/$SRC" OUT="$DS/field_${MOD}_feat_${ap}_debias${RANK}" \
  CHANNELS=feat APERTURE="$ap" \
    bash "$ROOT/alfs_embed/docker/integrate_dgx.sh" "$MOD" "$(wc -w <<<"$GPUS")" >>"$LOG" 2>&1 \
    || say "  integration $ap FAILED"
done

# ---- 3. flatten into cell directories the trainer can read -----------------
for pair in "single:embed_single" "full:embed_multi"; do
  ap=${pair%%:*}; cell=${pair##*:}
  dst=$DS/embeddings/cell_${MOD}_${cell}_debias${RANK}
  mkdir -p "$dst"
  find "$DS/field_${MOD}_feat_${ap}_debias${RANK}" -name '*.npy' ! -name '*_cov.npy' 2>/dev/null \
  | while read -r f; do
      fl=$(basename "$(dirname "$(dirname "$f")")")
      ln -sf "$f" "$dst/${fl}_$(basename "$f")"
    done
  say "$cell: $(find "$dst" -name '*.npy' 2>/dev/null | wc -l) cells"
done

# ---- 4. train both cells across the same folds -----------------------------
say "training"
cd "$ROOT/alfs_embed/docker" && BAMBI_GPUS="$GPUS" MODS="$MOD" SEEDS="$SEEDS" \
  CELLS="embed_single_debias${RANK} embed_multi_debias${RANK}" \
  bash cv_folds.sh >>"$LOG" 2>&1 && say "training done" || say "training FAILED"

say "== probe finished"
