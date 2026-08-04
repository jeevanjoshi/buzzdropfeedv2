#!/usr/bin/env bash
# ==============================================================================
# CSVG Project - Rsync Sync From OCI to Pi 5
# Pulls the latest logs, published topics, and stats from the OCI server.
# ==============================================================================

set -uo pipefail

OCI_USER="ubuntu"
OCI_HOST="oci-prod"
OCI_DIR="/home/ubuntu/buzzdropfeedv2"

LOCAL_DIR="/home/jeevanjoshi/buzzdropfeedv2"

echo "🔄 Pulling logs, published topics, and stats from OCI (${OCI_HOST})..."

# Rsync logs and JSON configuration states from OCI to Pi 5
rsync -avz \
    --include='logs/' \
    --include='logs/**' \
    --include='published_topics.json' \
    --include='channel_stats.json' \
    --exclude='*' \
    -e ssh \
    "${OCI_USER}@${OCI_HOST}:${OCI_DIR}/" "${LOCAL_DIR}/"

echo "✅ Pull completed successfully! Dashboard updated."
