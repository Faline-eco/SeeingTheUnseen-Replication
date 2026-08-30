#!/usr/bin/env bash
# Scene-level 5-fold cross-validation of the 2x2 grid.
#
# Why: the published split (flights 10/211/212/213) yields 212 hidden thermal
# boxes, of which flight 10 contributes 149. Its occlusion estimate therefore
# rests on effectively one scene, and the +/- reported with it is seed variance,
# which cannot express that. Occlusion is a property of terrain and canopy, so
# scene-to-scene variance is the quantity that decides whether the 2x2 result
# generalises — and it has never been measured.
#
# The folds partition all 72 flights present in both modalities, balanced
# greedily on hidden-box count (2008-2010 thermal each, so no fold's estimate is
# noise-dominated). Embeddings already cover every frame, so no re-encoding is
# needed: only the label partition changes.
#
#   BAMBI_GPUS="0 1 2" bash cv_folds.sh
set -euo pipefail

ROOT=/scratch/bambi
MODS=${MODS:-"thermal rgb"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"1337 7 42"}
EPOCHS=${EPOCHS:-40}
CELLS=${CELLS:-"realortho_single realalfs_multi embed_single embed_multi"}
export BAMBI_GPUS=${BAMBI_GPUS:-"0 1 2"}
export WIDTH=${WIDTH:-}
TAGSUF=${TAGSUF:-${WIDTH:+_w$WIDTH}}
export BAMBI_MAX_GPUS=$(wc -w <<<"$BAMBI_GPUS")
LOG=$ROOT/alfs_embed/logs/cv_folds.log
mkdir -p "$(dirname "$LOG")"

say() { echo "== $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
say "GPUs $BAMBI_GPUS | folds $FOLDS | mods $MODS | seeds $SEEDS"
say "$(( $(wc -w <<<"$FOLDS") * $(wc -w <<<"$MODS") * $(wc -w <<<"$CELLS") * $(wc -w <<<"$SEEDS") )) runs total"

for mod in $MODS; do
  for k in $FOLDS; do
    arms=""
    for c in $CELLS; do
      # label carries the fold so run dirs stay distinct in the shared OUTDIR
      # WIDTH is folded into the label so wide-head runs get their own run
      # directories instead of silently overwriting the 128-wide ones.
      arms+="${c}${TAGSUF}_f${k}:cell_${mod}_${c}:0:alfs_2k_${mod}_f${k} "
    done
    say "fold $k / $mod"
    MOD=$mod SEEDS="$SEEDS" EPOCHS=$EPOCHS BASE_ARMS="${arms% }" \
      bash "$ROOT/alfs_embed/docker/multiseed_dgx.sh" >>"$LOG" 2>&1 \
      && say "  done fold $k / $mod" || say "  FAILED fold $k / $mod"
  done
done

say "== all folds finished"
