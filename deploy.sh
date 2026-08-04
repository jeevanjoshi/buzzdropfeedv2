#!/usr/bin/env bash
# ==============================================================================
# CSVG Autonomous YouTube Pipeline - Remote Multi-Node Deployment Script
# 
# Target Nodes:
# 1. Raspberry Pi 5 Edge Node: 172.198.1.30 (Path: /opt/csvg_edge)
# 2. OCI Cloud Instance: 'oci-prod' (Accessed via Raspberry Pi 5 SSH hop)
# ==============================================================================

set -e

PI5_IP="192.168.1.30"
PI5_USER="jeevanjoshi"
PI5_TARGET_DIR="/home/jeevanjoshi/buzzdropfeedv2"

OCI_HOST="oci-prod"
OCI_TARGET_DIR="/opt/csvg_pipeline"

REPO_URL="https://github.com/jeevanjoshi/buzzdropfeedv2.git"
BRANCH="main"

echo "========================================================================"
echo "🚀 STARTING CSVG AUTOMATED REMOTE MULTI-NODE DEPLOYMENT"
echo "========================================================================"
echo "📅 Timestamp: $(date -u)"
echo "📍 Raspberry Pi 5 Edge IP: ${PI5_IP}"
echo "☁️ OCI Cloud Host: ${OCI_HOST} (via Pi 5 SSH Hop)"
echo "📦 Repository: ${REPO_URL} (${BRANCH})"
echo "========================================================================"

# ------------------------------------------------------------------------------
# STEP 1: Deploy to Raspberry Pi 5 Edge Node via Git
# ------------------------------------------------------------------------------
echo ""
echo "📡 [1/2] Deploying Edge Audio & STT Services to Raspberry Pi 5 (${PI5_IP})..."

ssh -o StrictHostKeyChecking=no ${PI5_USER}@${PI5_IP} bash -s << EOF
    set -e
    REPO_URL="${REPO_URL}"
    BRANCH="${BRANCH}"
    TARGET_DIR="${PI5_TARGET_DIR}"

    echo "Ensuring directory \${TARGET_DIR} exists..."
    sudo mkdir -p \${TARGET_DIR}
    sudo chown -R ${PI5_USER}:${PI5_USER} \${TARGET_DIR}

    if [ ! -d "\${TARGET_DIR}/.git" ]; then
        echo "Cloning repository on Raspberry Pi 5..."
        git clone \${REPO_URL} \${TARGET_DIR}
        cd \${TARGET_DIR}
        git checkout \${BRANCH}
    else
        echo "Updating existing Git repository on Raspberry Pi 5..."
        cd \${TARGET_DIR}
        git fetch origin
        git reset --hard origin/\${BRANCH}
    fi

    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    # Systemd Service Verification & Restart
    if [ -f "mcp_servers/audio_edge/server.py" ]; then
        sudo systemctl restart kokoro_tts.service || echo "⚠️ systemctl restart kokoro_tts failed; check if service is enabled."
    fi
EOF

echo "✅ Raspberry Pi 5 Edge Node Deployment Completed Successfully!"

# ------------------------------------------------------------------------------
# STEP 2: Deploy to OCI Cloud Node via Git (via Raspberry Pi 5 SSH Hop to oci-prod)
# ------------------------------------------------------------------------------
echo ""
echo "☁️ [2/2] Deploying Master Pipeline & Cloud MCP Servers to OCI (${OCI_HOST})..."

ssh -o StrictHostKeyChecking=no -J ${PI5_USER}@${PI5_IP} ${OCI_HOST} bash -s << EOF
    set -e
    REPO_URL="${REPO_URL}"
    BRANCH="${BRANCH}"
    TARGET_DIR="${OCI_TARGET_DIR}"

    echo "Ensuring directory \${TARGET_DIR} exists..."
    sudo mkdir -p \${TARGET_DIR}
    sudo chown -R \$USER:\$USER \${TARGET_DIR}

    if [ ! -d "\${TARGET_DIR}/.git" ]; then
        echo "Cloning repository on OCI Cloud Node..."
        git clone \${REPO_URL} \${TARGET_DIR}
        cd \${TARGET_DIR}
        git checkout \${BRANCH}
    else
        echo "Updating existing Git repository on OCI Cloud Node..."
        cd \${TARGET_DIR}
        git fetch origin
        git reset --hard origin/\${BRANCH}
    fi

    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    # Verify Cron Setup
    (crontab -l 2>/dev/null | grep -q "main.py") || (crontab -l 2>/dev/null; echo "0 4 * * * /opt/csvg_pipeline/venv/bin/python3 /opt/csvg_pipeline/main.py >> /var/log/csvg.log 2>&1") | crontab -
EOF

echo "✅ OCI Cloud Node Deployment Completed Successfully!"

echo ""
echo "========================================================================"
echo "🎉 DEPLOYMENT FINISHED! ALL SERVICES CLONED & DEPLOYED VIA GIT"
echo "========================================================================"
