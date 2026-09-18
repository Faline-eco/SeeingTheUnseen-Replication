#!/usr/bin/env bash
# Write the sampling geometry of every aperture (render_embedding_field.py
# --emit grid), then flatten it to <fid>_<frame>.npz under embeddings/, the
# layout view_aggregator.py reads.
#
# CPU-bound (embree ray-casts per frame); the GPU only samples the modality
# mask. Sharded by flight over NSHARDS containers pinned to ONE gpu, since each
# needs a few hundred MB of device memory at most.
#
#   bash viewgrid_dgx.sh thermal 6 8        # modality, gpu, shards
set -euo pipefail

MOD=${1:-thermal}
GPU=${2:-0}
NSHARDS=${3:-8}
IMAGE=${IMAGE:-bambi-embed:1.5}
FLIGHT_IDS=${FLIGHT_IDS:-}          # comma list to restrict (smoke test)
ROOT=/scratch/bambi
DS=$ROOT/datasets/alfs_embed
CODE=$ROOT/alfs_embed/code
EMB=$DS/embeddings/srcframes_${MOD}
OUT=$DS/viewgrid_${MOD}             # integrator appends _grid
FLAT=$DS/embeddings/viewgrid_${MOD}
LOGDIR=$ROOT/alfs_embed/logs
mkdir -p "$LOGDIR"

if [[ -n "$FLIGHT_IDS" ]]; then
  IFS=, read -r -a FLIGHTS <<<"$FLIGHT_IDS"
else
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
fi
(( ${#FLIGHTS[@]} )) || { echo "ERROR: no flights" >&2; exit 1; }
(( NSHARDS > ${#FLIGHTS[@]} )) && NSHARDS=${#FLIGHTS[@]}
echo "== ${#FLIGHTS[@]} flights, $NSHARDS shards on GPU $GPU (modality=$MOD)"

pids=()
for (( s=0; s<NSHARDS; s++ )); do
  group=""
  for i in "${!FLIGHTS[@]}"; do (( i % NSHARDS == s )) && group+="${FLIGHTS[$i]},"; done
  group=${group%,}
  [[ -n "$group" ]] || continue
  docker run --rm --ipc=host --gpus "\"device=$GPU\"" \
    -v "$ROOT:$ROOT" -w "$CODE" \
    -e PYTHONPATH="/opt/bambi:$CODE:$CODE/src:$CODE/alfspy_src" \
    "$IMAGE" python scripts/render_embedding_field.py \
      --config "$CODE/config_dgx.yaml" --flight-ids "$group" \
      --modality "$MOD" --channels feat --aperture full --emit grid \
      --embeddings "$EMB" --out "$OUT" --out-hw 128 \
      > "$LOGDIR/viewgrid_${MOD}_w${s}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail )) && echo "WARNING: a shard failed, see $LOGDIR/viewgrid_${MOD}_w*.log" >&2

mkdir -p "$FLAT"
for f in "${OUT}_grid"/*/"$MOD"/*.npz; do
  fl=$(basename "$(dirname "$(dirname "$f")")")
  [[ -e "$FLAT/${fl}_$(basename "$f")" ]] || ln -s "$f" "$FLAT/${fl}_$(basename "$f")"
done
echo "== grids: $(ls "$FLAT" | wc -l) frames -> $FLAT"
du -sh "${OUT}_grid"
exit $fail
