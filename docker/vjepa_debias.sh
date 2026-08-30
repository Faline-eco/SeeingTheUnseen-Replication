#!/usr/bin/env bash
# INSID3 debias probe for the V-JEPA 2.1 cells.
#
# Unlike the DINOv3 path there is no per-frame store to re-encode: the V-JEPA
# cells are produced straight from the projected view stacks, so the projection
# is applied inside encode_vjepa_cells.py and the whole cell build is redone.
# That means re-projecting the aperture views as well, because vjepa_batch.py
# deletes each batch's stacks after encoding it.
#
# Three stages:
#   1. fit the PCA in the debiased space (workers refuse to fit their own, and
#      the debiased space is not the baseline's -- its pca.pkl cannot be reused)
#   2. build every cell via vjepa_batch.py
#   3. train both cells over the same folds and seeds
#
# Nothing existing is overwritten: every output goes to cell_<mod>_vjepa_<kind>_debias<rank>.
#
#   MOD=thermal BAMBI_GPUS="0 1 2" bash vjepa_debias.sh
set -uo pipefail

ROOT=/scratch/bambi
DS=$ROOT/datasets/alfs_embed
MOD=${MOD:-thermal}
RANK=${RANK:-32}
GPUS=${BAMBI_GPUS:-"0 1 2"}
SEEDS=${SEEDS:-"1337 7 42"}
VARIANT=${VARIANT:-2.1-vit-b-384}
SRC_PX=${SRC_PX:-2048}
# Projection is CPU-bound. The defaults (12x16 = 192 threads) are for a machine
# to ourselves; when another encode is running on the other GPUs, oversubscribing
# the cores starves its data loading and both jobs get slower.
PROJ_WORKERS=${PROJ_WORKERS:-12}
PROJ_THREADS=${PROJ_THREADS:-16}
BATCH=${BATCH:-6}
BASIS=$DS/zenodo_labels/vjepa_positional_basis.npz
STACK_ROOT=$DS/vjepa_stack_${MOD}_stack
STEMS=$DS/zenodo_labels/${MOD}_train_stems.txt
LOG=$ROOT/alfs_embed/logs/vjepa_debias_${MOD}.log
mkdir -p "$(dirname "$LOG")"

say() { echo "== $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

dock() {   # dock <gpu> <image> <cmd...>
  local g=$1 img=$2; shift 2
  docker run --rm --ipc=host --gpus "\"device=$g\"" -v "$ROOT:$ROOT" \
    -w "$ROOT/alfs_embed/code" \
    -e PYTHONPATH="/opt/bambi:$ROOT/alfs_embed/code:$ROOT/alfs_embed/code/src:$ROOT/alfs_embed/code/alfspy_src" \
    -e TORCH_HOME=$ROOT/vjepa2/torch_hub -e YOLO_CONFIG_DIR=/tmp "$img" "$@"
}

say "V-JEPA INSID3 probe, rank $RANK, modality $MOD"
[[ -f "$BASIS" ]] || { say "missing basis $BASIS -- run fit_positional_basis_vjepa.py"; exit 1; }
[[ -f "$STEMS" ]] || { say "missing train stems $STEMS"; exit 1; }

G0=$(awk '{print $1}' <<<"$GPUS")

# ---- 1. PCA basis in the debiased space ------------------------------------
# Fitted on training flights only, exactly as the baseline cells were. The same
# stems file is used deliberately: a different PCA sample would be a second
# difference between the debiased and baseline arms.
FIT_FLIGHTS=${FIT_FLIGHTS:-$(sed 's/_.*//' "$STEMS" | sort -u | head -3 | tr '\n' ' ')}
say "PCA fit flights: $FIT_FLIGHTS"

need_fit=0
for kind in single multi; do
  [[ -f "$DS/embeddings/cell_${MOD}_vjepa_${kind}_debias${RANK}/pca.pkl" ]] || need_fit=1
done

if (( need_fit )); then
  for fid in $FIT_FLIGHTS; do
    if [[ -d "$STACK_ROOT/$fid" ]]; then
      say "  stack for $fid already present"
      continue
    fi
    say "  projecting $fid for the fit"
    dock "$G0" bambi-embed:1.5 python scripts/render_embedding_field.py \
        --config config_dgx.yaml --flight-ids "$fid" --modality "$MOD" \
        --channels rgb --emit stack --out-hw "$SRC_PX" \
        --embeddings "$DS/embeddings" \
        --out "$DS/vjepa_stack_${MOD}" >>"$LOG" 2>&1 \
      || { say "  FAILED projection $fid"; exit 1; }
  done

  for kind in single multi; do
    say "  fitting PCA, kind=$kind"
    dock "$G0" bambi-vjepa:1.0 python scripts/encode_vjepa_cells.py \
        --stack-root "$STACK_ROOT" --kind "$kind" --modality "$MOD" \
        --variant "$VARIANT" --src-px "$SRC_PX" \
        --train-stems "$STEMS" \
        --debias-basis "$BASIS" --debias-rank "$RANK" \
        --fit-only --out "$DS/embeddings/cell_${MOD}_vjepa_${kind}" >>"$LOG" 2>&1 \
      || { say "  FAILED PCA fit $kind"; exit 1; }
  done

  # Dropped rather than kept: vjepa_batch.py re-projects each batch anyway, and
  # a stale stack directory is the difference between a bounded and an unbounded
  # disk footprint.
  for fid in $FIT_FLIGHTS; do rm -rf "${STACK_ROOT:?}/$fid"; done
fi
say "PCA ready"

# ---- 2. build every cell ---------------------------------------------------
say "building cells (this is the long stage)"
python3 "$ROOT/alfs_embed/code/scripts/vjepa_batch.py" \
    --modality "$MOD" --variant "$VARIANT" --src-px "$SRC_PX" \
    --gpus "$(tr ' ' ',' <<<"$GPUS")" --batch "$BATCH" \
    --proj-workers "$PROJ_WORKERS" --proj-threads "$PROJ_THREADS" \
    --debias-basis "$BASIS" --debias-rank "$RANK" >>"$LOG" 2>&1 \
  || say "  vjepa_batch reported failures"

for kind in single multi; do
  d=$DS/embeddings/cell_${MOD}_vjepa_${kind}_debias${RANK}
  say "  cell_${MOD}_vjepa_${kind}_debias${RANK}: $(find "$d" -name '*.npy' 2>/dev/null | wc -l) cells"
done

# ---- 3. train --------------------------------------------------------------
say "training"
cd "$ROOT/alfs_embed/docker" && BAMBI_GPUS="$GPUS" MODS="$MOD" SEEDS="$SEEDS" \
  CELLS="vjepa_single_debias${RANK} vjepa_multi_debias${RANK}" \
  bash cv_folds.sh >>"$LOG" 2>&1 && say "training done" || say "training FAILED"

say "== vjepa debias finished"
