#!/usr/bin/env bash
# ==============================================================================
# CSVG Autonomous YouTube Pipeline - Remote Multi-Node Deployment Script
# 
# Target Nodes:
# 1. Raspberry Pi 5 Edge Node: 172.198.1.30 (Path: /opt/csvg_edge)
# 2. OCI Cloud Instance: 'oci-prod' (Accessed via Raspberry Pi 5 SSH hop)
# ==============================================================================

set -e

PI5_IP="172.198.1.30"
PI5_USER="pi"
PI5_TARGET_DIR="/opt/csvg_edge"

OCI_HOST="oci-prod"
OCI_TARGET_DIR="/opt/csvg_pipeline"

echo "========================================================================"
echo "🚀 STARTING CSVG AUTOMATED REMOTE MULTI-NODE DEPLOYMENT"
echo "========================================================================"
echo "📅 Timestamp: $(date -u)"
echo "📍 Raspberry Pi 5 Edge IP: ${PI5_IP}"
echo "☁️ OCI Cloud Host: ${OCI_HOST} (via Pi 5 SSH Hop)"
echo "========================================================================"

# ------------------------------------------------------------------------------
# STEP 1: Deploy to Raspberry Pi 5 Edge Node
# ------------------------------------------------------------------------------
echo ""
echo "📡 [1/2] Deploying Edge Audio & STT Services to Raspberry Pi 5 (${PI5_IP})..."

ssh -o StrictHostKeyChecking=no ${PI5_USER}@${PI5_IP} "sudo mkdir -p ${PI5_TARGET_DIR} && sudo chown -R ${PI5_USER}:${PI5_USER} ${PI5_TARGET_DIR}"

rsync -avz --exclude='venv' --exclude='.git' --exclude='logs' --exclude='tmp' \
    ./ ${PI5_USER}@${PI5_IP}:${PI5_TARGET_DIR}/

ssh -o StrictHostKeyChecking=no ${PI5_USER}@${PI5_IP} bash << 'EOF'
    set -e
    cd /opt/csvg_edge
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
# STEP 2: Deploy to OCI Cloud Node (via Raspberry Pi 5 SSH Hop to oci-prod)
# ------------------------------------------------------------------------------
echo ""
echo "☁️ [2/2] Deploying Master Pipeline & Cloud MCP Servers to OCI (${OCI_HOST})..."

# Use SSH Jump Host (-J) to route directly through Raspberry Pi 5 to oci-prod
ssh -o StrictHostKeyChecking=no -J ${PI5_USER}@${PI5_IP} ${OCI_HOST} "sudo mkdir -p ${OCI_TARGET_DIR} && sudo chown -R \$USER:\$USER ${OCI_TARGET_DIR}" || \
ssh -o StrictHostKeyChecking=no ${PI5_USER}@${PI5_IP} "ssh ${OCI_HOST} 'sudo mkdir -p ${OCI_TARGET_DIR} && sudo chown -R \$USER:\$USER ${OCI_TARGET_DIR}'"

rsync -avz -e "ssh -J ${PI5_USER}@${PI5_IP}" --exclude='venv' --exclude='.git' --exclude='logs' --exclude='tmp' \
    ./ ${OCI_HOST}:${OCI_TARGET_DIR}/ || \
    (echo "Falling back to rsync via Pi 5..." && \
     rsync -avz --exclude='venv' --exclude='.git' --exclude='logs' ./ ${PI5_USER}@${PI5_IP}:/tmp/csvg_sync/ && \
     ssh ${PI5_USER}@${PI5_IP} "rsync -avz /tmp/csvg_sync/ ${OCI_HOST}:${OCI_TARGET_DIR}/ && rm -rf /tmp/csvg_sync")

ssh -o StrictHostKeyChecking=no -J ${PI5_USER}@${PI5_IP} ${OCI_HOST} bash << 'EOF' || \
ssh ${PI5_USER}@${PI5_IP} "ssh ${OCI_HOST} 'cd /opt/csvg_pipeline && if [ ! -d venv ]; then python3 -m venv venv; fi && source venv/bin/activate && pip install -r requirements.txt'"
    set -e
    cd /opt/csvg_pipeline
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
echo "🎉 DEPLOYMENT FINISHED! ALL SERVICES DEPLOYED TO RASPBERRY PI 5 & OCI"
echo "========================================================================"
