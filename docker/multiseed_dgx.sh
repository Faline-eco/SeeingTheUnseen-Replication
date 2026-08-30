#!/usr/bin/env bash
# Repeat the embedding-detector arms across seeds, to separate real effects
# from run-to-run variance.
#
# Why this exists: the single-seed sweep gave 64 -> 128 -> 256 dims as
# 0.3598 -> 0.3295 -> 0.3747 mAP50-95. A genuine dimensionality effect would be
# monotonic; a dip in the middle is what noise looks like. Without a variance
# estimate, any ranking read off those numbers is unfounded — and the same
# numbers are about to decide whether the embedded light field is encoded at
# 128 dims (~257 GB) or 64 (~128 GB), so the difference matters.
#
# Runs in waves sized to the free GPUs, since the machine is shared.
#
#   ./multiseed_dgx.sh              # 3 dims x 3 seeds, plus the 1024px control
set -euo pipefail

MOD=${MOD:-thermal}
IMAGE=${IMAGE:-bambi-embed:1.5}
EPOCHS=${EPOCHS:-40}
SEEDS=${SEEDS:-"1337 7 42"}
ROOT=/scratch/bambi
DS=$ROOT/datasets/alfs_embed
LOGDIR=$ROOT/alfs_embed/logs/multiseed
OUTDIR=$DS/embedding_runs_multiseed
mkdir -p "$LOGDIR" "$OUTDIR"
LBL=alfs_2k_${MOD}

# arm := <label>:<embedding dataset>:<use-dims>:<labels dataset>
BASE_ARMS=${BASE_ARMS:-"2k_256:alfs_2k_${MOD}:0:$LBL 2k_128:alfs_2k_${MOD}:128:$LBL 2k_64:alfs_2k_${MOD}:64:$LBL 1024_64:alfs_${MOD}:0:$LBL"}

JOBS=()
for s in $SEEDS; do
  for a in $BASE_ARMS; do JOBS+=("$a:$s"); done
done
echo "== ${#JOBS[@]} runs (${EPOCHS} epochs each)"

# BAMBI_GPUS pins an explicit allocation. Without it we take whatever is free,
# which on a shared machine drifts: "the first N free" is not the same as "the N
# we were allocated", and a neighbour finishing frees a GPU that is not ours.
if [[ -n "${BAMBI_GPUS:-}" ]]; then
  read -r -a FREE <<<"$BAMBI_GPUS"
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '$2+0 >= 1000 {print $1}')
  # An explicit allocation is an assertion by the operator, so occupancy is only
  # a warning. Failing hard here was wrong: our OWN short jobs (an evaluation
  # sharing an allocated GPU) legitimately occupy it, and that is precisely what
  # aborted four rgb folds once. nvidia-smi cannot tell us whose job it is.
  for g in "${FREE[@]}"; do
    grep -qx "$g" <<<"$busy" && echo "WARNING: allocated GPU $g already in use; proceeding." >&2
  done
else
  mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                      | awk -F', ' '$2+0 < 1000 {print $1}')
fi
(( ${#FREE[@]} )) || { echo "ERROR: no free GPUs; other users are on this machine." >&2; exit 1; }
# Cap the wave width: without this the run takes every free GPU, which is not
# ours to take on a shared machine.
BAMBI_MAX_GPUS=${BAMBI_MAX_GPUS:-${#FREE[@]}}
NG=${#FREE[@]}
(( NG > BAMBI_MAX_GPUS )) && NG=$BAMBI_MAX_GPUS
FREE=("${FREE[@]:0:NG}")
echo "== free GPUs: ${FREE[*]}"

run_job() {                       # run_job <gpu> <label:dset:dims:labds:seed>
  local gpu=$1
  IFS=: read -r label dset dims labds seed <<<"$2"
  # Which GT the val pass scores against -- i.e. what best.pt is SELECTED on.
  #   centralgt (default) : single-view GT. This penalises a multi-view arm for
  #     every hidden animal it correctly finds, so checkpoint selection and early
  #     stopping run against those arms' own interest, making their published
  #     results conservative rather than inflated.
  #   merged              : aperture GT, the neutral choice.
  # Head width. Default 128 (0.36 M params on 128-dim embeddings); 512 gives
  # 3.90 M, matching frozen V-JEPA's 3.81 M head, so the two foundation models
  # are compared at equal task-head capacity rather than a 10x difference.
  local WIDTH_ARG=""
  [[ -n "${WIDTH:-}" ]] && WIDTH_ARG="--width ${WIDTH}"
  local EVAL_ARG=""
  [[ "${EVAL_LABELS:-centralgt}" == "centralgt" ]] \
    && EVAL_ARG="--eval-labels ${labds}_centralgt"
  local tag="${label}_s${seed}"
  docker run --rm --ipc=host --gpus "\"device=$gpu\"" \
    -v "$ROOT:$ROOT" -w "$ROOT/alfs_embed/code" \
    -e PYTHONPATH="/opt/bambi:$ROOT/alfs_embed/code:$ROOT/alfs_embed/code/src:$ROOT/alfs_embed/code/alfspy_src" \
    "$IMAGE" python scripts/train_embedding_detector.py \
      --dataset "$dset" --labels-dataset "$labds" \
      --embeddings "$DS/embeddings" --yolo-datasets "$DS/yolo_datasets" \
      ${EVAL_ARG} \
      --out "$OUTDIR" --epochs "$EPOCHS" --use-dims "$dims" \
      --seed "$seed" --tag "$tag" ${WIDTH_ARG} \
      > "$LOGDIR/${MOD}_${tag}.log" 2>&1
}

fail=0; i=0
while (( i < ${#JOBS[@]} )); do
  pids=(); tags=()
  for (( g=0; g<NG && i<${#JOBS[@]}; g++, i++ )); do
    IFS=: read -r l _ _ _ s <<<"${JOBS[$i]}"
    echo "   -> GPU ${FREE[$g]}: ${l} seed ${s}"
    run_job "${FREE[$g]}" "${JOBS[$i]}" & pids+=($!); tags+=("${l}_s${s}")
  done
  for j in "${!pids[@]}"; do
    if ! wait "${pids[$j]}"; then echo "   ${tags[$j]} FAILED" >&2; fail=1; fi
  done
done

echo "== aggregating"
python3 - "$OUTDIR" <<'PY'
import json, sys, statistics as st
from pathlib import Path
runs = {}
for f in Path(sys.argv[1]).glob("*/summary.json"):
    m = json.loads(f.read_text())
    arm = m["name"].rsplit("_s", 1)[0]
    runs.setdefault(arm, []).append(m)
print(f"\n  {'arm':28s} {'n':>2}  {'mAP50-95':>18}  {'mAP50':>18}")
print("  " + "-" * 72)
for arm in sorted(runs):
    v = [r["mAP50-95"] for r in runs[arm]]
    w = [r["mAP50"] for r in runs[arm]]
    def f(x):
        return (f"{st.mean(x):.4f} +/- {st.stdev(x):.4f}" if len(x) > 1
                else f"{x[0]:.4f}")
    print(f"  {arm:28s} {len(v):2d}  {f(v):>18}  {f(w):>18}")
print("\n  A difference smaller than ~2x the stdev is not separable at this n.")
PY
exit $fail
