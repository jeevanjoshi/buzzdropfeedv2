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

# Help: show usage and exit BEFORE any sync/detach/run side effects.
if [[ " $* " == *" --help "* ]] || [[ " $* " == *" -h "* ]]; then
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run the CSVG pipeline in the background, sync code + logs to the Raspberry
Pi edge node, and email the result when it finishes.

Options:
  --no-detach            Run in the foreground; keep all flags; skip Pi sync
  --rag <MODE>           Research mode: grounded|scraper (Default: 'scraper')
  --renderer <ENGINE>    Renderer: ffmpeg|moviepy (Default: 'ffmpeg')
  --crossfade <SEC>      Crossfade duration in seconds (float)
  --resume               Auto-resume the latest unfinished checkpoint
  -h, --help             Display this help menu and exit

Notes:
  Only --no-detach, --resume, --renderer, --crossfade, and --rag survive
  the detach. Use --no-detach to keep flags like --till-upload or --offline.

Examples:
  $(basename "$0")
  $(basename "$0") --rag grounded
  $(basename "$0") --no-detach --rag scraper --till-upload
EOF
    exit 0
fi

# Configuration
# Each run gets its own timestamped log file. LOG_FILE is inherited (exported) from
# the detaching parent so the background child writes to the exact same file.
LOG_FILE="${LOG_FILE:-logs/pipeline_run_$(date +'%Y%m%d-%H%M%S').log}"
export LOG_FILE
# Keep a canonical pipeline_run.log so the Pi dashboard always points at the
# latest run (symlink target is in the same logs/ dir).
ln -sfn "$(basename "$LOG_FILE")" "logs/pipeline_run.log"
PI5_IP="100.108.116.100"
PI5_USER="jeevanjoshi"
PI5_TARGET_DIR="/home/jeevanjoshi/buzzdropfeedv2"

# First step: push the latest codebase/config to the Pi edge node BEFORE the run.
# Runs once in the parent (skipped in the detached --no-detach child). Delegates
# to sync_to_pi.sh (excludes logs/media/caches and deletes stale Pi files).
if [ "${1:-}" != "--no-detach" ] && command -v rsync >/dev/null 2>&1; then
    echo "[SYNC] Pushing codebase to Raspberry Pi 5 (${PI5_IP})..."
    bash sync_to_pi.sh || echo "[WARN] Code sync to Pi failed; continuing with run anyway."
    echo "[SYNC] Codebase sync to Pi complete."
fi

# Extract switches that must survive the background detach and reach the pipeline.
RAG_ARG=""
RENDERER_ARG=""
CROSSFADE_ARG=""
if [[ "$*" == *"--rag"* ]]; then
    for ((i=1; i<=$#; i++)); do
        if [[ "${!i}" == "--rag" ]]; then
            j=$((i+1)); RAG_ARG="--rag ${!j}"; break
        fi
    done
fi
if [[ "$*" == *"--renderer"* ]]; then
    for ((i=1; i<=$#; i++)); do
        if [[ "${!i}" == "--renderer" ]]; then
            j=$((i+1)); RENDERER_ARG="--renderer ${!j}"; break
        fi
    done
fi
if [[ "$*" == *"--crossfade"* ]]; then
    for ((i=1; i<=$#; i++)); do
        if [[ "${!i}" == "--crossfade" ]]; then
            j=$((i+1)); CROSSFADE_ARG="--crossfade ${!j}"; break
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
    # Launch itself with --no-detach in the background, forwarding resume + renderer + crossfade + rag
    nohup "$0" --no-detach ${RESUME_ARG} ${RENDERER_ARG} ${CROSSFADE_ARG} ${RAG_ARG} > /dev/null 2>&1 &
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

# Heartbeat + periodic log-sync to the Pi dashboard. The pipeline runs here (OCI)
# but the dashboard runs on the Pi, so a local `ps` check there is always wrong.
# Instead we write a heartbeat file and rsync it (plus logs + state) to the Pi on
# an interval while the pipeline is alive; the Pi dashboard infers "running" from
# how fresh the heartbeat is.
HB_FILE="logs/pipeline_heartbeat.json"
SYNC_LOOP_PID=""
if command -v rsync >/dev/null 2>&1; then
    sync_loop() {
        while true; do
            printf '{"running":true,"ts":"%s"}\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "${HB_FILE}"
            rsync -az --exclude '*.onnx' --exclude '*.bin' --exclude '*.mp3' --exclude '*.wav' \
                -e ssh \
                logs/ "${PI5_USER}@${PI5_IP}:${PI5_TARGET_DIR}/logs/" 2>/dev/null
            sleep 15
        done
    }
    sync_loop &
    SYNC_LOOP_PID=$!
fi

# Run the pipeline
# --renderer is already present in "$@" (survived the detach), so just forward.
if [[ "$*" == *"--resume"* ]]; then
    python3 main.py "$@" >> "${LOG_FILE}" 2>&1
else
    python3 main.py --global "$@" >> "${LOG_FILE}" 2>&1
fi
EXIT_CODE=$?

# Mark the pipeline as no longer running (so the dashboard shows Idle).
printf '{"running":false,"ts":"%s","exit_code":%s}\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "${EXIT_CODE}" > "${HB_FILE}"
if [ -n "${SYNC_LOOP_PID}" ]; then
    kill "${SYNC_LOOP_PID}" 2>/dev/null
fi

echo "" >> "${LOG_FILE}"
echo "[FINISHED] PIPELINE RUN FINISHED WITH EXIT CODE: ${EXIT_CODE}" >> "${LOG_FILE}"
echo "========================================================================" >> "${LOG_FILE}"


# Send email notification
STATUS="success"
if [ ${EXIT_CODE} -ne 0 ]; then
    STATUS="failure"
fi

python3 send_pipeline_email.py --status "${STATUS}" --log_file "${LOG_FILE}" >> "${LOG_FILE}" 2>&1

# Push run logs + state to the Pi so the on-Pi dashboard reflects the latest run.
# The canonical logs/pipeline_run.log symlink travels too, so the dashboard's
# /api/logs (which reads that fixed name) always points at the newest run.
if command -v rsync >/dev/null 2>&1; then
    mkdir -p logs
    rsync -az \
        --exclude '*.onnx' --exclude '*.bin' --exclude '*.mp3' --exclude '*.wav' \
        -e ssh \
        logs/ "${PI5_USER}@${PI5_IP}:${PI5_TARGET_DIR}/logs/" 2>/dev/null \
        || echo "[WARN] Failed to push logs to Pi" >> "${LOG_FILE}"
fi

exit ${EXIT_CODE}
