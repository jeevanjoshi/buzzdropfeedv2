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
    --exclude '.mypy_cache/' \
    --exclude '.ruff_cache/' \
    --exclude 'sentence_transformers/' \
    --exclude 'rust_dashboard/target/' \
    -e ssh \
    ./ "${PI5_USER}@${PI5_IP}:${PI5_TARGET_DIR}/"

echo "Sync completed successfully!"

# Rebuild Rust dashboard only when explicitly requested (to avoid killing dashboard during live runs)
# NOTE: index.html is embedded into the binary via include_str!, so ANY change to
# rust_dashboard/web/index.html OR rust_dashboard/src/*.rs (e.g. the JSON UTF-8
# parser fix) requires a rebuild — run `./sync_to_pi.sh --dashboard` to ship it.
if [[ " $* " == *" --dashboard "* ]] || [[ " $* " == *" --build "* ]] || [ "${CSVG_BUILD_DASHBOARD:-0}" = "1" ]; then
    echo "Rebuilding Rust dashboard and restarting services on Raspberry Pi 5..."
    ssh -o StrictHostKeyChecking=no "${PI5_USER}@${PI5_IP}" \
        'export PATH="$HOME/.cargo/bin:$PATH"; cd /home/jeevanjoshi/buzzdropfeedv2/rust_dashboard && cargo build --release && sudo systemctl restart csvg_rust_dashboard.service' 2>/dev/null && \
        echo "✓ Rust dashboard recompiled and csvg_rust_dashboard.service restarted on Pi." || \
        echo "Notice: Rust dashboard build skipped."
fi
