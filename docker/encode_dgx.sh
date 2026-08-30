#!/usr/bin/env bash
# Encode ALFS renders to DINOv3 embeddings across several GPUs.
#
# Enforces the one ordering that matters: the PCA basis is fitted ONCE, up
# front, and every shard then reuses it. Launching the shards straight away
# would have each of them fit its own basis on its own subsample and race to
# write pca.pkl — the shards' embeddings would end up in different linear
# spaces, which nothing downstream would notice and everything downstream
# would be wrong about. encode_embeddings.py refuses to fit when
# --num-shards > 1, and this script is the sanctioned way to satisfy it.
#
#   ./encode_dgx.sh thermal 4 128
set -euo pipefail

MOD=${1:-thermal}
BAMBI_MAX_GPUS=${BAMBI_MAX_GPUS:-1}   # shared machine: one GPU unless told otherwise
NGPU=${2:-$BAMBI_MAX_GPUS}
PCA_DIM=${3:-128}
SOURCE=${SOURCE:-alfs}
IMAGE=${IMAGE:-bambi-embed:1.5}
ROOT=/scratch/bambi
OUT=$ROOT/datasets/alfs_embed/embeddings
CFG=$ROOT/alfs_embed/code/config_dgx.yaml
LOGDIR=$ROOT/alfs_embed/logs
mkdir -p "$LOGDIR"

COMMON=(scripts/encode_embeddings.py
        --config "$CFG" --source "$SOURCE" --modality "$MOD"
        --out "$OUT" --pca-dim "$PCA_DIM"
        --splits-json "$ROOT/alfs_embed/code/bambi_splits_reference.json"
        --model-dir "$ROOT/datasets/alfs_embed/DINOv3")

# Claim the GPUs once, so the set does not change between the fit and the
# shards (another user could otherwise take one in between).
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                    | awk -F', ' '$2+0 < 1000 {print $1}')
if (( ${#FREE[@]} < NGPU )); then
  echo "ERROR: wanted $NGPU free GPUs, only ${#FREE[@]} free (${FREE[*]:-none})." >&2
  echo "Other users are on this machine - wait or lower the GPU count." >&2
  exit 1
fi
GPUS=("${FREE[@]:0:NGPU}")
echo "== using GPUs: ${GPUS[*]}  | modality=$MOD  pca=$PCA_DIM"

run_on() {           # run_on <gpu> <logfile> <args...>
  local gpu=$1 logf=$2; shift 2
  docker run --rm --ipc=host --gpus "\"device=$gpu\"" \
    -v "$ROOT:$ROOT" -w "$ROOT/alfs_embed/code" \
    -e PYTHONPATH="/opt/bambi:$ROOT/alfs_embed/code:$ROOT/alfs_embed/code/src:$ROOT/alfs_embed/code/alfspy_src" \
    "$IMAGE" python "$@" >"$logf" 2>&1
}

PCA_FILE=$OUT/${SOURCE}_${MOD}/pca.pkl
if [[ -f "$PCA_FILE" ]]; then
  echo "== PCA basis already present: $PCA_FILE"
else
  echo "== fitting PCA basis on GPU ${GPUS[0]} (single process)"
  run_on "${GPUS[0]}" "$LOGDIR/encode_${MOD}_fit.log" "${COMMON[@]}" --fit-only
  grep -E "explained variance|PCA 1280" "$LOGDIR/encode_${MOD}_fit.log" || true
fi
[[ -f "$PCA_FILE" ]] || { echo "ERROR: fit produced no pca.pkl; see $LOGDIR/encode_${MOD}_fit.log" >&2; exit 1; }

echo "== launching $NGPU shards"
pids=()
for i in "${!GPUS[@]}"; do
  logf=$LOGDIR/encode_${MOD}_shard${i}.log
  run_on "${GPUS[$i]}" "$logf" "${COMMON[@]}" --shard "$i" --num-shards "$NGPU" &
  pids+=($!)
  echo "   shard $i -> GPU ${GPUS[$i]}  ($logf)"
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "   shard $i OK"
  else echo "   shard $i FAILED (exit $?)" >&2; fail=1; fi
done

echo "== per-shard results"
for f in "$OUT/${SOURCE}_${MOD}"/meta.shard*.json; do
  [[ -e "$f" ]] || continue
  python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print('   %s: %d written, %d skipped, %d failed, grid %s'%(sys.argv[1].split('/')[-1],m['written'],m['skipped'],m['failed'],m['grid']))" "$f"
done
echo "== .npy on disk: $(find "$OUT/${SOURCE}_${MOD}" -name '*.npy' | wc -l)"
exit $fail
