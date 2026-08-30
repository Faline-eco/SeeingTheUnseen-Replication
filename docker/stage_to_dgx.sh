#!/usr/bin/env bash
# Stage the data the DGX needs for embedding encoding + detector training.
#
# Copies ONLY what those two jobs read:
#   * the 2048px ALFS renders (the images to encode)
#   * the MOT label files (encoder decides which frames are labelled)
#   * the YOLO label sets (detector ground truth + train/val split)
#   * DINOv3 weights
#
# Deliberately NOT copied: the neighbour frame cache (~47 GB) and the source
# videos. Those are only needed for *rendering*, which stays on the workstation
# because ModernGL in a container has produced artifacts before.
#
# Everything lands under /scratch/bambi, the only permitted path on that host.
#
#   ./stage_to_dgx.sh thermal
set -euo pipefail
MOD=${1:-thermal}
REMOTE=dgxa100
DST=/scratch/bambi/datasets/alfs_embed
SRC_DS=/z/Hugo/matched_dataset_new

# Stream a directory to the remote over tar|ssh.
#
# GNU tar exits 1 for "file changed as we read it" — an mtime touched by a
# scanner is enough, and for these static trees it is benign. Under
# `set -euo pipefail` that warning aborted the whole staging run after the
# first label set, and because the caller piped this script into `tee`, the
# masked exit code reported success while 15 GB of renders never moved. So:
# tolerate tar's exit 1, treat 2+ as fatal, and check the remote end separately.
tar_stream() {                      # tar_stream <srcdir> <remote-dir> [tar args…]
  local src=$1 dst=$2; shift 2
  local st
  set +e
  tar -C "$src" -cf - "$@" . | ssh "$REMOTE" "mkdir -p '$dst' && tar -C '$dst' -xf -"
  st=("${PIPESTATUS[@]}")
  set -e
  if (( ${st[0]} > 1 )); then
    echo "ERROR: tar failed reading $src (exit ${st[0]})" >&2; return 1
  fi
  if (( ${st[1]} != 0 )); then
    echo "ERROR: remote extract into $dst failed (exit ${st[1]})" >&2; return 1
  fi
  (( ${st[0]} == 1 )) && echo "   (tar reported changed files; contents verified below)"
  return 0
}

echo "== staging '$MOD' to $REMOTE:$DST"
ssh "$REMOTE" "mkdir -p $DST/matched_dataset_new/{alfs_2k,MOT/$MOD} $DST/yolo_datasets $DST/DINOv3"

echo "-- DINOv3 weights (4.3 GB, skipped if already there)"
ssh "$REMOTE" "test -f $DST/DINOv3/model.safetensors" \
  && echo "   already present" \
  || scp -r "/d/DINOv3/." "$REMOTE:$DST/DINOv3/"

echo "-- MOT labels"
scp -q "$SRC_DS/MOT/$MOD/"*.txt "$REMOTE:$DST/matched_dataset_new/MOT/$MOD/"

echo "-- YOLO label sets (ground truth + split)"
# From yolo_datasets_2k, NOT yolo_datasets: the label projection rounds to
# integer pixels before normalising, so the 1024px set differs from the 2048px
# one by up to 1/1024 (measured: ~1.08% of mean box size). Scoring the
# resolution arms against different ground truth would confound exactly the
# comparison they exist to make, so every arm uses this one 2048-derived set.
for ds in "alfs_${MOD}" "alfs_${MOD}_centralgt"; do
  ssh "$REMOTE" "mkdir -p $DST/yolo_datasets/$ds/labels/{train,val}"
  for split in train val; do
    d="/d/yolo_datasets_2k/$ds/labels/$split"
    [ -d "$d" ] || continue
    tar_stream "$d" "$DST/yolo_datasets/$ds/labels/$split"
  done
done

echo "-- 2048px ALFS renders (tar-streamed; scp over thousands of files is slow)"
# alfs_2k holds every modality, so a plain tar of it would re-send the other
# modality's tens of GB that are already on the remote. Excluding them keeps a
# second staging run proportional to what actually changed.
OTHER=$([ "$MOD" = rgb ] && echo thermal || echo rgb)
tar_stream "$SRC_DS/alfs_2k" "$DST/matched_dataset_new/alfs_2k" \
  --exclude='*_utm.txt' --exclude="*/${OTHER}/*"

echo "-- config"
scp -q "/d/PipelineStep2/docker/config_dgx.yaml" "$REMOTE:/scratch/bambi/alfs_embed/code/config_dgx.yaml"
scp -q "/d/PipelineStep2/bambi_splits_reference.json" "$REMOTE:/scratch/bambi/alfs_embed/code/"

echo "== verifying (counts compared against the local source, not assumed)"
local_png=$(find "$SRC_DS/alfs_2k" -path "*/${MOD}/*" -name '*.png' | wc -l)
local_tr=$(ls "/d/yolo_datasets_2k/alfs_${MOD}/labels/train" | wc -l)
local_va=$(ls "/d/yolo_datasets_2k/alfs_${MOD}/labels/val" | wc -l)
read -r rem_png rem_tr rem_va <<<"$(ssh "$REMOTE" "
  echo \$(find $DST/matched_dataset_new/alfs_2k -path '*/${MOD}/*' -name '*.png' 2>/dev/null | wc -l) \
       \$(ls $DST/yolo_datasets/alfs_${MOD}/labels/train 2>/dev/null | wc -l) \
       \$(ls $DST/yolo_datasets/alfs_${MOD}/labels/val 2>/dev/null | wc -l)")"

# Compared in the current shell: assigning rc inside $( ) would set it in a
# subshell, so a MISMATCH would print while the script still exited 0.
rc=0
check() {                           # check <name> <local> <remote>
  local verdict=OK
  if [[ "$2" != "$3" ]]; then verdict=MISMATCH; rc=1; fi
  printf '   %-16s local %7s  remote %7s  %s\n' "$1" "$2" "$3" "$verdict"
}
check renders      "$local_png" "$rem_png"
check labels/train "$local_tr"  "$rem_tr"
check labels/val   "$local_va"  "$rem_va"

ssh "$REMOTE" "du -sh $DST/* 2>/dev/null"
if (( rc )); then echo "== STAGING INCOMPLETE" >&2; else echo "== staged and verified"; fi
exit $rc
