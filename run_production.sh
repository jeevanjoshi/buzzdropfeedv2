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
PI5_IP="100.108.116.100"
PI5_USER="jeevanjoshi"
PI5_TARGET_DIR="/home/jeevanjoshi/buzzdropfeedv2"

# Extract a --renderer moviepy|ffmpeg switch if supplied by the caller so it
# survives the background detach and is passed to the pipeline.
RENDERER_ARG=""
if [[ "$*" == *"--renderer"* ]]; then
    for ((i=1; i<=$#; i++)); do
        if [[ "${!i}" == "--renderer" ]]; then
            j=$((i+1))
            RENDERER_ARG="--renderer ${!j}"
            break
        fi
    done
fi

# Check if we should detach into the background
if [ "${1:-}" != "--no-detach" ]; then
    LATEST_STATE=$(ls -t logs/state_*.json 2>/dev/null | head -n 1)
    RESUME_ARG=""
    if [ -n "${LATEST_STATE}" ]; then
        PIPELINE_ID=$(grep -o '"pipeline_id": "[^"]*' "${LATEST_STATE}" | head -n 1 | cut -d'"' -f4)
        STAGE=$(grep -o '"execution_stage": "[^"]*' "${LATEST_STATE}" | head -n 1 | cut -d'"' -f4)
        
        if [ "${STAGE}" != "PUBLISHED_SUCCESS" ]; then
            echo -e "\n[CHECKPOINT] Found incomplete pipeline checkpoint: ${PIPELINE_ID} (Stage: ${STAGE})"
            echo -n "Would you like to resume this run? (y/N) [Defaulting to N in 30s]: "
            if read -t 30 response && [[ "${response}" =~ ^[Yy](es)?$ ]]; then
                echo "[RESUMING] Resuming run ${PIPELINE_ID}..."
                RESUME_ARG="--resume ${PIPELINE_ID}"
            else
                echo -e "\n[STARTING] Starting a fresh pipeline run..."
            fi
        fi
    fi

    echo "[LAUNCH] Launching CSVG Production Pipeline in the background..."
    # Launch itself with --no-detach in the background, forwarding resume + renderer arguments
    nohup "$0" --no-detach ${RESUME_ARG} ${RENDERER_ARG} > /dev/null 2>&1 &
    PID=$!
    echo "[SUCCESS] Pipeline successfully spawned in background (PID: ${PID})."
    echo "[LOGS] Real-time logs: tail -f ${LOG_FILE}"
    echo "[EMAIL] Notification will be sent to jeevan.z.joshi@gmail.com upon completion."
    exit 0
fi

# Actual pipeline execution (runs in the background detached shell)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "========================================================================" >> "${LOG_FILE}"
echo "[STARTING] STARTING PIPELINE RUN: ${TIMESTAMP}" >> "${LOG_FILE}"
echo "========================================================================" >> "${LOG_FILE}"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "[WARNING] Warning: venv directory not found! Running with system python." >> "${LOG_FILE}"
fi

# Run the pipeline
# --renderer is already present in "$@" (survived the detach), so just forward.
if [[ "$*" == *"--resume"* ]]; then
    python3 main.py "$@" >> "${LOG_FILE}" 2>&1
else
    python3 main.py --global "$@" >> "${LOG_FILE}" 2>&1
fi
EXIT_CODE=$?

echo "" >> "${LOG_FILE}"
echo "[FINISHED] PIPELINE RUN FINISHED WITH EXIT CODE: ${EXIT_CODE}" >> "${LOG_FILE}"
echo "========================================================================" >> "${LOG_FILE}"


# Send email notification
STATUS="success"
if [ ${EXIT_CODE} -ne 0 ]; then
    STATUS="failure"
fi

python3 send_pipeline_email.py --status "${STATUS}" --log_file "${LOG_FILE}" >> "${LOG_FILE}" 2>&1

exit ${EXIT_CODE}
