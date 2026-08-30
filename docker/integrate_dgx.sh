#!/usr/bin/env bash
# Build the embedded light field: integrate DINOv3 features across each
# aperture, on several GPUs at once.
#
# This is the "encode-then-average" arm. The ALFS arm averages pixels and then
# encodes; this encodes each source frame and averages the features, with the
# same aperture, same virtual camera and same DEM. The integrator is GL-free
# (embree ray-casts + torch grid_sample), which is why it can run in this
# container at all — ModernGL here has produced renders with artifacts.
#
# Sharded by flight rather than by frame: each flight loads a DEM and builds a
# ray-cast mesh once, so splitting a flight across workers would repeat that.
#
#   ./integrate_dgx.sh thermal 4
set -euo pipefail

MOD=${1:-thermal}
BAMBI_MAX_GPUS=${BAMBI_MAX_GPUS:-1}   # shared machine: one GPU unless told otherwise
NGPU=${2:-$BAMBI_MAX_GPUS}
IMAGE=${IMAGE:-bambi-embed:1.5}
# The 2x2: CHANNELS picks sensor (rgb) vs embedding (feat) space, APERTURE
# picks single vs multi view. Everything else is held fixed, so a cell differs
# from its neighbour in exactly one axis.
CHANNELS=${CHANNELS:-feat}
APERTURE=${APERTURE:-full}
TAG=${TAG:-${CHANNELS}_${APERTURE}}
ROOT=/scratch/bambi
DS=$ROOT/datasets/alfs_embed
CODE=$ROOT/alfs_embed/code
# Overridable so a variant per-frame store (e.g. the INSID3-debiased one) can be
# integrated through this same script. Hand-writing a second render_embedding_field
# call instead is how the debias probe silently integrated one flight: the script
# defaults --flight-ids to a single flight, and only this path enumerates them.
EMB=${EMB:-$DS/embeddings/srcframes_${MOD}}
OUT=${OUT:-$DS/field_${MOD}_${TAG}}
LOGDIR=$ROOT/alfs_embed/logs
mkdir -p "$LOGDIR" "$OUT"
echo "== cell: modality=$MOD channels=$CHANNELS aperture=$APERTURE -> $OUT"

# Source embeddings are only read in feature mode; the sensor cells read images.
if [[ "$CHANNELS" == feat && ! -d "$EMB" ]]; then
  echo "ERROR: no source embeddings at $EMB" >&2; exit 1
fi

# Flights that actually have labels for this modality, in the reference split.
mapfile -t FLIGHTS < <(python3 - "$CODE/bambi_splits_reference.json" \
                                "$DS/matched_dataset_new/MOT/$MOD" "$MOD" <<'PY'
import json, sys
from pathlib import Path
ids = {str(f) for v in json.load(open(sys.argv[1])).values() for f in v}
mot = Path(sys.argv[2]); mod = sys.argv[3]
print("\n".join(sorted((f for f in ids if (mot / f"{f}_accepted_{mod}_mot.txt").exists()),
                       key=int)))
PY
)
(( ${#FLIGHTS[@]} )) || { echo "ERROR: no flights with $MOD labels" >&2; exit 1; }

mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                    | awk -F', ' '$2+0 < 1000 {print $1}')
(( ${#FREE[@]} >= NGPU )) || { echo "ERROR: wanted $NGPU free GPUs, ${#FREE[@]} free (${FREE[*]:-none})" >&2; exit 1; }
GPUS=("${FREE[@]:0:NGPU}")
echo "== ${#FLIGHTS[@]} flights over GPUs ${GPUS[*]}  (modality=$MOD)"

# Round-robin so the big flights spread across workers instead of landing in
# one contiguous block.
pids=(); labels=()
for g in "${!GPUS[@]}"; do
  group=""
  for i in "${!FLIGHTS[@]}"; do
    (( i % NGPU == g )) && group+="${FLIGHTS[$i]},"
  done
  group=${group%,}
  [[ -n "$group" ]] || continue
  logf=$LOGDIR/integrate_${MOD}_${TAG}_w${g}.log
  echo "   worker $g -> GPU ${GPUS[$g]}: $(tr -cd ',' <<<"$group" | wc -c) flights"
  docker run --rm --ipc=host --gpus "\"device=${GPUS[$g]}\"" \
    -v "$ROOT:$ROOT" -w "$CODE" \
    -e PYTHONPATH="/opt/bambi:$CODE:$CODE/src:$CODE/alfspy_src" \
    "$IMAGE" python scripts/render_embedding_field.py \
      --config "$CODE/config_dgx.yaml" --flight-ids "$group" \
      --modality "$MOD" --channels "$CHANNELS" --aperture "$APERTURE" \
      --embeddings "$EMB" --out "$OUT" --out-hw 128 \
      > "$logf" 2>&1 &
  pids+=($!); labels+=("w$g")
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "   ${labels[$i]} OK"
  else echo "   ${labels[$i]} FAILED" >&2; fail=1; fi
done

n=$(find "$OUT" -name '*.npy' ! -name '*_cov.npy' | wc -l)
echo "== embedded light fields written: $n"
du -sh "$OUT"
exit $fail
