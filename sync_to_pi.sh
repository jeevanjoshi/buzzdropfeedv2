#!/usr/bin/env bash
# ==============================================================================
# CSVG Project - Rsync Sync to Raspberry Pi 5
# Syncs the current codebase + config to the Pi 5 edge node.
#   * Excludes logs, media, caches and other runtime/unnecessary files.
#   * --delete: removes files on the Pi that no longer exist on OCI, so the Pi
#     is an exact mirror of the OCI working tree (excluding the exclusions).
#   * Excluded paths (logs/, venv/, etc.) are NEVER deleted from the Pi.
# ==============================================================================

set -uo pipefail

PI5_IP="100.108.116.100"
PI5_USER="jeevanjoshi"
PI5_TARGET_DIR="/home/jeevanjoshi/buzzdropfeedv2"

echo "Syncing codebase and .env to Raspberry Pi 5 (${PI5_IP}) as ${PI5_USER}..."

# Run rsync excluding unnecessary/runtime files; delete stale files on the Pi.
rsync -avz --delete \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '.git/' \
    --exclude 'logs/' \
    --exclude 'tmp/' \
    --exclude '*.mp3' \
    --exclude '*.wav' \
    --exclude '*.mp4' \
    --exclude '*.ass' \
    --exclude '*.onnx' \
    --exclude '*.bin' \
    --exclude '*.png' \
    --exclude 'auth_url.txt' \
    --exclude '.hf_cache/' \
    --exclude '.huggingface/' \
    --exclude 'sentence_transformers/' \
    -e ssh \
    ./ "${PI5_USER}@${PI5_IP}:${PI5_TARGET_DIR}/"

echo "Sync completed successfully!"
