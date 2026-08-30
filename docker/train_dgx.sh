#!/usr/bin/env bash
# Train the embedding-detector arms on the DGX, one GPU each, in parallel.
#
# The arms exist to separate two things the 2048px re-render changed at once:
# render resolution and PCA dimensionality. Because PCA components are nested,
# the 128- and 64-dim arms are truncations of the single 256-dim encode
# (--use-dims), so all three come from one pass and differ in exactly one
# variable. Resolution is isolated separately by the 1024px arm, which uses its
# own cached embeddings but the *same* labels and the same code.
#
#   ./train_dgx.sh            # the three dimensionality arms
#   ARMS="1024_64" ./train_dgx.sh
set -euo pipefail

MOD=${MOD:-thermal}
IMAGE=${IMAGE:-bambi-embed:1.5}
EPOCHS=${EPOCHS:-60}
ROOT=/scratch/bambi
DS=$ROOT/datasets/alfs_embed
LOGDIR=$ROOT/alfs_embed/logs
OUTDIR=$DS/embedding_runs
mkdir -p "$LOGDIR" "$OUTDIR"

# arm := <label>:<embedding dataset>:<use-dims>[:<labels dataset>]
#
# The labels field defaults to the embedding dataset and only differs for the
# resolution arm, whose embeddings sit under the 1024px name but which must be
# scored against the same 2048px-derived ground truth as everything else — the
# two label sets differ by ~1% of box size, enough to muddy the comparison the
# arm exists to make.
LBL=alfs_2k_${MOD}
ARMS=${ARMS:-"2k_256:alfs_2k_${MOD}:0 2k_128:alfs_2k_${MOD}:128 2k_64:alfs_2k_${MOD}:64 1024_64:alfs_${MOD}:0:$LBL"}

mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                    | awk -F', ' '$2+0 < 1000 {print $1}')
read -ra ARM_ARR <<<"$ARMS"
if (( ${#FREE[@]} < ${#ARM_ARR[@]} )); then
  echo "ERROR: ${#ARM_ARR[@]} arms but only ${#FREE[@]} free GPUs (${FREE[*]:-none})." >&2
  echo "Other users are on this machine - wait, or run fewer arms." >&2
  exit 1
fi
echo "== arms: ${#ARM_ARR[@]} | GPUs ${FREE[*]:0:${#ARM_ARR[@]}} | epochs $EPOCHS"

pids=(); names=()
for i in "${!ARM_ARR[@]}"; do
  IFS=: read -r label dset dims labds <<<"${ARM_ARR[$i]}"
  labds=${labds:-$dset}
  gpu=${FREE[$i]}
  logf=$LOGDIR/train_${MOD}_${label}.log
  echo "   $label (emb=$dset dims=${dims:-all} labels=$labds) -> GPU $gpu"
  docker run --rm --ipc=host --gpus "\"device=$gpu\"" \
    -v "$ROOT:$ROOT" -w "$ROOT/alfs_embed/code" \
    -e PYTHONPATH="/opt/bambi:$ROOT/alfs_embed/code:$ROOT/alfs_embed/code/src:$ROOT/alfs_embed/code/alfspy_src" \
    "$IMAGE" python scripts/train_embedding_detector.py \
      --dataset "$dset" \
      --labels-dataset "$labds" \
      --embeddings "$DS/embeddings" \
      --yolo-datasets "$DS/yolo_datasets" \
      --eval-labels "${labds}_centralgt" \
      --out "$OUTDIR" --epochs "$EPOCHS" --use-dims "$dims" --tag "$label" \
      >"$logf" 2>&1 &
  pids+=($!); names+=("$label")
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "   ${names[$i]} OK"
  else echo "   ${names[$i]} FAILED" >&2; fail=1; fi
done

echo "== results"
for n in "${names[@]}"; do
  f=$LOGDIR/train_${MOD}_${n}.log
  printf '   %-8s %s\n' "$n" "$(grep -E 'best|mAP50-95' "$f" 2>/dev/null | tail -1)"
done
exit $fail
