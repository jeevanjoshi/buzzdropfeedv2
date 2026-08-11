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
  --rag <MODE>           Research mode: grounded|hybrid|scraper (Default: 'scraper')
  --renderer <ENGINE>    Renderer: ffmpeg|moviepy (Default: 'ffmpeg')
  --crossfade <SEC>      Crossfade duration in seconds (float)
  --tail <SEC>           Video-only hold after each shot's narration (float, default 1.2)
  --resume               Auto-resume the latest unfinished checkpoint
  --skip-health-check    Skip the pre-flight health gate (LLM/YT/Pi) — not recommended
  -h, --help             Display this help menu and exit

Notes:
  Only --no-detach, --resume, --renderer, --crossfade, --tail, and --rag survive
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

# ── Pre-Flight Health Check ──────────────────────────────────────────────
# Gate the run on LLM availability, YouTube quota+auth, Pi audio-edge
# reachability, grounding (when --rag grounded is requested), and required
# binaries/.env keys BEFORE any sync or run cost. Aborts (exit 1) if any
# REQUIRED check fails. Pass --skip-health-check to bypass (dangerous: a broken
# run will just fail mid-pipeline).
if [[ " $* " != *" --skip-health-check "* ]]; then
    echo "[HEALTH] Running production pre-flight health check..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    # Forward the requested RAG research mode so the grounding gate can FAIL
    # when --rag grounded is asked for but Vertex is not configured.
    HEALTH_RAG=""
    if [[ "$*" == *"--rag"* ]]; then
        for ((i=1; i<=$#; i++)); do
            if [[ "${!i}" == "--rag" ]]; then
                j=$((i+1)); HEALTH_RAG="--rag ${!j}"; break
            fi
        done
    fi
    if python healthcheck.py ${HEALTH_RAG}; then
        echo "[HEALTH] All checks green — proceeding with production run."
    else
        echo "[HEALTH] FAILED — aborting production run before launch." >&2
        echo "[HEALTH] See logs/health_check.log for the audit trail." >&2
        exit 1
    fi
else
    echo "[HEALTH] Skipping health check (--skip-health-check given)."
fi

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
TAIL_ARG=""
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
if [[ "$*" == *"--tail"* ]]; then
    for ((i=1; i<=$#; i++)); do
        if [[ "${!i}" == "--tail" ]]; then
            j=$((i+1)); TAIL_ARG="--tail ${!j}"; break
        fi
    done
fi

# ── Live Progress Monitor ──────────────────────────────────────────────────
# Renders a single-line progress bar driven by the pipeline's PERSISTED
# execution_stage (read from the newest logs/state_*.json), so the operator can
# see, at a glance, the overall completion % and the current stage of the run.
# The monitor only reads local files; the detached pipeline + Pi dashboard
# keep working regardless. Ctrl+C stops the monitor, NOT the pipeline.
stage_to_pct() {
    case "$1" in
        PUBLISHED_SUCCESS|"DONE") echo 100 ;;
        QUALITY_VERIFIED)        echo 92 ;;
        MEDIA_PRODUCED)          echo 85 ;;
        SCRIPT_APPROVED)         echo 50 ;;
        SCRIPT_REVISION_REQUIRED)echo 40 ;;
        SCRIPT_GENERATED)        echo 35 ;;
        TOPIC_SELECTED)          echo 20 ;;
        INITIALIZATION|"")       echo 5  ;;
        *)                       echo 30 ;;
    esac
}

# Latest (most recently written) persisted stage, or "" if none yet.
current_stage() {
    local f state
    f="$(ls -t logs/state_*.json 2>/dev/null | head -n 1)"
    if [ -n "$f" ]; then
        state="$(grep -o '"execution_stage": "[^"]*' "${f}" 2>/dev/null | head -n 1 | cut -d'"' -f4)"
        if [ -n "$state" ] && [ "$state" != "INITIALIZATION" ]; then
            echo "$state"; return
        fi
    fi
    # Fallback: infer the furthest stage mentioned in the run log.
    for s in PUBLISHED_SUCCESS QUALITY_VERIFIED MEDIA_PRODUCED SCRIPT_APPROVED \
             SCRIPT_REVISION_REQUIRED SCRIPT_GENERATED TOPIC_SELECTED; do
        if grep -q "${s}" "${LOG_FILE}" 2>/dev/null; then echo "$s"; return; fi
    done
    echo ""
}

stage_label() {
    case "$1" in
        PUBLISHED_SUCCESS)       echo "Publishing / done" ;;
        QUALITY_VERIFIED)        echo "Quality gates passed" ;;
        MEDIA_PRODUCED)          echo "Rendering media" ;;
        SCRIPT_APPROVED)         echo "Script approved" ;;
        SCRIPT_REVISION_REQUIRED)echo "Revising script" ;;
        SCRIPT_GENERATED)        echo "Writing script" ;;
        TOPIC_SELECTED)          echo "Topic + RAG" ;;
        INITIALIZATION|"")       echo "Starting" ;;
        *)                       echo "Working" ;;
    esac
}

render_bar() {
    local pct=$1 label=$2 elapsed=$3 width=40 i filled bar=""
    filled=$(( pct * width / 100 ))
    for ((i=0;i<width;i++)); do
        if [ "$i" -lt "$filled" ]; then bar+="#"; else bar+="-"; fi
    done
    printf '\r[%s] %3d%%  %-22s %02d:%02d' "${bar}" "$pct" "$label" \
        $((elapsed/60)) $((elapsed%60))
}

show_progress() {
    local pid=$1 start=$SECONDS stage last=""
    trap 'echo ""; echo "[PROGRESS] Monitor stopped; pipeline continues in the background."; exit 0' INT
    while kill -0 "${pid}" 2>/dev/null; do
        stage="$(current_stage)"
        if [ -t 1 ]; then
            render_bar "$(stage_to_pct "$stage")" "$(stage_label "$stage")" "$((SECONDS-start))"
        elif [ "$stage" != "$last" ]; then
            printf '[PROGRESS] %-22s %3d%% (%02d:%02d)\n' "$(stage_label "$stage")" \
                "$(stage_to_pct "$stage")" $((SECONDS/60)) $((SECONDS%60))
        fi
        last="$stage"
        sleep 3
    done
    if [ -t 1 ]; then render_bar 100 "Finished" "$((SECONDS-start))"; echo ""; fi
    echo "[PROGRESS] Pipeline process finished with a reported exit code (see emails/logs)."
}

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
    # Launch itself with --no-detach in the background, forwarding resume + renderer + crossfade + tail + rag.
    # `setsid` fully detaches the child from this shell's session/controlling terminal so a closed
    # SSH/tmux session can NEVER kill it mid/finish (the cause of "pipeline published but no final
    # exit-code line and no email" — the child was reaped before its post-run steps).
    # The region override (--india/--global) is forwarded so an explicit CLI
    # override survives the detach; with no override the child runs dynamic
    # region (fact_retriever picks the market from day/time/topic/events).
    REGION_FWD=""
    if [[ "$*" == *"--india"* ]]; then
        REGION_FWD="--india"
    elif [[ "$*" == *"--global"* ]]; then
        REGION_FWD="--global"
    fi
    # Use an ABSOLUTE path for the detached child: under cron $PATH does NOT
    # contain the repo dir, and $0 arrives as a bare filename (cron_publish.sh
    # runs us via `bash run_production.sh`). `nohup` resolves a bare name via
    # $PATH (execvp), so `"$0"` silently died with exit 127 and NO log — the
    # child never launched. Also keep the child's early stderr APPENDED to the
    # log file instead of /dev/null so a spawn failure is always visible.
    setsid nohup "${SCRIPT_DIR}/$(basename "$0")" --no-detach ${RESUME_ARG} ${RENDERER_ARG} ${CROSSFADE_ARG} ${TAIL_ARG} ${RAG_ARG} ${REGION_FWD} \
        < /dev/null >> "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "[SUCCESS] Pipeline successfully spawned in background (PID: ${PID})."
    echo "[LOGS] Real-time logs: tail -f ${LOG_FILE}"
    echo "[EMAIL] Notification will be sent to jeevan.z.joshi@gmail.com upon completion."
    # Live progress bar so the operator can see what's happening in real time.
    # (Only attach in an interactive terminal; never emit ANSI junk into scripts.)
    if [ -t 1 ]; then
        echo "[PROGRESS] Monitoring pipeline progress — (Ctrl+C stops the monitor only)..."
        show_progress "${PID}"
    else
        echo "[PROGRESS] Non-interactive shell — progress bar skipped."
    fi
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

# ── Guaranteed finalization ────────────────────────────────────────────────
# Runs the heartbeat-stop, exit-code log line, notification email and final Pi
# log push. A trap on EXIT calls it so these happen even if the normal flow is
# interrupted after main.py returns — fixing the "published but no exit code and
# no email" failure where the child was reaped before its post-run steps.
FINALIZED=0
finalize() {
    local ec=$1
    [ "${FINALIZED}" -eq 1 ] && return "$ec"
    FINALIZED=1
    printf '{"running":false,"ts":"%s","exit_code":%s}\n' \
        "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$ec" > "${HB_FILE}" 2>/dev/null || true
    if [ -n "${SYNC_LOOP_PID}" ]; then kill "${SYNC_LOOP_PID}" 2>/dev/null; fi
    {
        echo "" >> "${LOG_FILE}"
        echo "[FINISHED] PIPELINE RUN FINISHED WITH EXIT CODE: ${ec}" >> "${LOG_FILE}"
        echo "========================================================================" >> "${LOG_FILE}"
        local st=success; [ "$ec" -ne 0 ] && st=failure
        echo "[EMAIL] Sending notification email (status=${st})..." >> "${LOG_FILE}"
        python3 send_pipeline_email.py --status "${st}" --log_file "${LOG_FILE}" >> "${LOG_FILE}" 2>&1
        echo "[EMAIL] Notification step done." >> "${LOG_FILE}"
        if command -v rsync >/dev/null 2>&1; then
            mkdir -p logs
            rsync -az --exclude '*.onnx' --exclude '*.bin' --exclude '*.mp3' --exclude '*.wav' \
                -e ssh \
                logs/ "${PI5_USER}@${PI5_IP}:${PI5_TARGET_DIR}/logs/" 2>/dev/null \
                || echo "[WARN] Failed to push logs to Pi" >> "${LOG_FILE}"
            echo "[SYNC] Final logs pushed to Pi." >> "${LOG_FILE}"
        fi
    }
    return "$ec"
}
trap 'finalize $? >/dev/null 2>&1' EXIT

# Run the pipeline
# --renderer is already present in "$@" (survived the detach), so just forward.
# Region: default is DYNAMIC (fact_retriever picks the best market from
# day/time/topic/events and sets state.region). An explicit --india/--global
# passthrough pins it (override) — otherwise NO fixed --global is forced.
REGION_FLAG=""
if [[ "$*" == *"--india"* ]]; then
    REGION_FLAG="--india"
elif [[ "$*" == *"--global"* ]]; then
    REGION_FLAG="--global"
fi
if [[ "$*" == *"--resume"* ]]; then
    python3 main.py "$@" >> "${LOG_FILE}" 2>&1
elif [ -n "${REGION_FLAG}" ]; then
    python3 main.py ${REGION_FLAG} "$@" >> "${LOG_FILE}" 2>&1
else
    python3 main.py "$@" >> "${LOG_FILE}" 2>&1   # dynamic region (default)
fi
EXIT_CODE=$?

finalize ${EXIT_CODE}
exit ${EXIT_CODE}
