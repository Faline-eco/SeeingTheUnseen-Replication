#!/usr/bin/env bash
# Publish the 120 learned-aggregator runs (viewgrid_<mod>_embed_multi_<agg>_f<k>_s<seed>)
# to heads/ on Hugging Face, from inside the study container on dgx1.
# Token: read from $HOME/.hf_token (mode 600), never from the command line.
# The container image sets HF_HUB_OFFLINE=1 (frozen backbones are read from
# disk), so it is overridden here.
#
#   bash hf_upload_viewagg.sh --dry-run
#   bash hf_upload_viewagg.sh
set -euo pipefail
ROOT=/scratch/bambi
DS=$ROOT/datasets/alfs_embed
CODE=$ROOT/alfs_embed/code
TOK=$(cat "$HOME/.hf_token")
docker run --rm -v "$ROOT:$ROOT" -w "$CODE" -e HF_TOKEN="$TOK" \
  -e HF_HUB_DISABLE_PROGRESS_BARS=1 -e HF_HUB_OFFLINE=0 \
  bambi-embed:1.5 python scripts/upload_to_hf.py \
    --runs "$DS/embedding_runs_multiseed" --glob 'viewgrid_*' --prefix heads --batch 30 \
    --message "Add learned view-aggregation heads (mean control, attention pooling, GRU, windowed transformer)" "$@"
