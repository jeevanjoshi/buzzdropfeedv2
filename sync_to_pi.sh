#!/usr/bin/env bash
# ==============================================================================
# CSVG Project - Rsync Sync to Raspberry Pi 5
# Syncs current codebase, configs, and .env to the Pi 5 edge node.
# ==============================================================================

set -uo pipefail

PI5_IP="100.108.116.100"
PI5_USER="jeevanjoshi"
PI5_TARGET_DIR="/home/jeevanjoshi/buzzdropfeedv2"

echo "Syncing codebase and .env to Raspberry Pi 5 (${PI5_IP}) as ${PI5_USER}..."

# Run rsync excluding runtime/venv files
rsync -avz \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '.git/' \
    --exclude 'logs/*.log' \
    --exclude 'tmp/' \
    --exclude '*.mp3' \
    --exclude '*.wav' \
    --exclude '*.onnx' \
    --exclude '*.bin' \
    -e ssh \
    ./ "${PI5_USER}@${PI5_IP}:${PI5_TARGET_DIR}/"

echo "Sync completed successfully!"
