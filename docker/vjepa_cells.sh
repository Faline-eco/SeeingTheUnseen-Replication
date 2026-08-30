#!/usr/bin/env bash
# Build the V-JEPA 2.1 cells (design C) one flight at a time.
#
# The projected aperture views are 130 MB/frame at 2048 px -- 1.36 TB for
# thermal. They are therefore materialised per batch, consumed, and deleted,
# never accumulated. Batching by flight rather than by a fixed count because
# render_embedding_field.py already works per flight; the transient peak is the
# largest flight (~507 frames, ~66 GB).
#
# Two stages in two containers on purpose: alfspy/embree does the projection,
# the V-JEPA container does the encoding. Fusing them into one process was the
# earlier plan and would have forced both dependency stacks together for no gain.
#
#   BAMBI_GPUS="0 1 2" bash vjepa_cells.sh              # both kinds, thermal
#   KINDS=multi FLIGHTS="10 211" bash vjepa_cells.sh    # subset
set -uo pipefail

ROOT=/scratch/bambi
DS=$ROOT/datasets/alfs_embed
MOD=${MOD:-thermal}
KINDS=${KINDS:-"single multi"}
VARIANT=${VARIANT:-2.1-vit-b-384}
GPU=${GPU:-0}
SRC_PX=${SRC_PX:-2048}
STACK_ROOT=$DS/vjepa_stack_${MOD}_stack      # _stack suffix is added by the renderer
LOG=$ROOT/alfs_embed/logs/vjepa_cells_${MOD}.log
mkdir -p "$(dirname "$LOG")"

say() { echo "== $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

FLIGHTS=${FLIGHTS:-$(ls $DS/matched_dataset_new/alfs_2k)}

dock() {   # dock <gpu> <cmd...>
  local g=$1; shift
  docker run --rm --ipc=host --gpus "\"device=$g\"" -v "$ROOT:$ROOT" \
    -w "$ROOT/alfs_embed/code" \
    -e PYTHONPATH="/opt/bambi:$ROOT/alfs_embed/code:$ROOT/alfs_embed/code/src:$ROOT/alfs_embed/code/alfspy_src" \
    -e YOLO_CONFIG_DIR=/tmp bambi-embed:1.5 "$@"
}

for fid in $FLIGHTS; do
  n_done=1
  for kind in $KINDS; do
    out=$DS/embeddings/cell_${MOD}_vjepa_${kind}
    # Resume: a flight whose cells all exist is skipped before any projection,
    # so a restart costs nothing rather than re-rendering 130 MB/frame.
    want=$(ls $DS/matched_dataset_new/alfs_2k/$fid/$MOD/*.txt 2>/dev/null | grep -vc _central || echo 0)
    have=$(ls $out/${fid}_*.npy 2>/dev/null | wc -l)
    (( want > 0 && have >= want )) && continue
    n_done=0
  done
  (( n_done )) && { say "skip flight $fid (cells present)"; continue; }

  say "flight $fid: projecting aperture views"
  dock "$GPU" python scripts/render_embedding_field.py \
      --config config_dgx.yaml --flight-ids "$fid" --modality "$MOD" \
      --channels rgb --emit stack --out-hw "$SRC_PX" \
      --embeddings "$DS/embeddings" \
      --out "$DS/vjepa_stack_${MOD}" >>"$LOG" 2>&1 \
    || { say "  FAILED projection $fid"; continue; }

  sz=$(du -sh "$STACK_ROOT/$fid" 2>/dev/null | cut -f1)
  say "  stack $sz; encoding"
  for kind in $KINDS; do
    dock "$GPU" python scripts/encode_vjepa_cells.py \
        --stack-root "$STACK_ROOT" --kind "$kind" --modality "$MOD" \
        --variant "$VARIANT" --src-px "$SRC_PX" \
        --train-stems "$DS/zenodo_labels/${MOD}_train_stems.txt" \
        --out "$DS/embeddings/cell_${MOD}_vjepa_${kind}" >>"$LOG" 2>&1 \
      || say "  FAILED encode $fid/$kind"
  done

  # Delete before the next flight, not at the end: the whole point of batching.
  rm -rf "${STACK_ROOT:?}/$fid"
  say "  flight $fid done, batch deleted"
done

say "== finished"
