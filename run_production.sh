#!/usr/bin/env bash
# ==============================================================================
# CSVG Autonomous YouTube Pipeline - Production Background Execution Wrapper
# Runs the main pipeline in the background, syncs logs to Pi 5, and sends emails.
# ==============================================================================

set -uo pipefail

# Resolve script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
LOG_FILE="logs/pipeline_run.log"
PI5_IP="192.168.1.30"
PI5_USER="jeevanjoshi"
PI5_TARGET_DIR="/home/jeevanjoshi/buzzdropfeedv2"

# Check if we should detach into the background
if [ "${1:-}" != "--no-detach" ]; then
    echo "🚀 Launching CSVG Production Pipeline in the background..."
    # Launch itself with --no-detach in the background, fully decoupled from current terminal
    nohup "$0" --no-detach > /dev/null 2>&1 &
    PID=$!
    echo "✅ Pipeline successfully spawned in background (PID: ${PID})."
    echo "📝 Real-time logs: tail -f ${LOG_FILE}"
    echo "📧 Notification will be sent to jeevan.z.joshi@gmail.com upon completion."
    exit 0
fi

# Actual pipeline execution (runs in the background detached shell)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "========================================================================" >> "${LOG_FILE}"
echo "🚀 STARTING PIPELINE RUN: ${TIMESTAMP}" >> "${LOG_FILE}"
echo "========================================================================" >> "${LOG_FILE}"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️ Warning: venv directory not found! Running with system python." >> "${LOG_FILE}"
fi

# Run the pipeline
python3 main.py --global >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?

echo "" >> "${LOG_FILE}"
echo "🏁 PIPELINE RUN FINISHED WITH EXIT CODE: ${EXIT_CODE}" >> "${LOG_FILE}"
echo "========================================================================" >> "${LOG_FILE}"

# Sync logs and status back to the local Raspberry Pi 5 edge server
echo "🔄 Rsyncing logs and status back to Raspberry Pi 5..." >> "${LOG_FILE}"
rsync -avz --include='logs/' --include='logs/**' --include='published_topics.json' --include='channel_stats.json' --exclude='*' -e ssh ./ "${PI5_USER}@${PI5_IP}:${PI5_TARGET_DIR}/" >> "${LOG_FILE}" 2>&1

# Send email notification
STATUS="success"
if [ ${EXIT_CODE} -ne 0 ]; then
    STATUS="failure"
fi

python3 send_pipeline_email.py --status "${STATUS}" --log_file "${LOG_FILE}" >> "${LOG_FILE}" 2>&1

exit ${EXIT_CODE}
