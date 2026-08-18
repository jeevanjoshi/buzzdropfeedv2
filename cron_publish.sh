#!/usr/bin/env bash
# ==============================================================================
# cron_publish.sh — Publication scheduler (invoked by cron).
#
# Schedules up to TWO production runs per day (TARGET_DAILY_PUBLISHES=2, the
# cadence behind the $2,000/month ad-revenue goal), back-timing each launch out
# of the MEASURED pipeline runtime (launch -> publish ≈ 1h50m) so the YouTube
# publish lands inside a peak viewing window. The region/market is decided
# DYNAMICALLY inside the pipeline (region_intelligence in the fact retriever
# picks the best market from day/time + topic affinity + per-market RPM +
# events), NOT here.
#
#   Launch windows (UTC) — back-time a publish into some peak window:
#     [11:00, 12:20]   -> publish ~13:10 UTC / ~6:40pm IST (India/Asia window)
#     [13:30, 14:30]   -> publish ~15:40 UTC / ~11:40am ET (US/UK/CA/AU window)
#
# Opt-in override: `--region india|global` pins the region (skips dynamic).
#
# Guards (all logged to logs/cron_publish.log, all skippable via env):
#   * never doubles a live run     (pgrep + heartbeat freshness)
#   * daily publish cap            (CSVG_MAX_DAILY_PUBLISHES, default 4; YT quota headroom)
#   * cooldown between launches    (CSVG_CRON_COOLDOWN_MIN, default 90) -> up to 2/day
#     (the two slots are 2.5h apart, so 90 min lets both fire — aligns the
#     per-video revenue gate $33.33 = $2,000 / (2 × 30) with the cadence)
#
# The actual run goes through run_production.sh, so the pre-flight health check,
# log sync to Pi, background detach and completion email still apply.
# ==============================================================================

set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs
LOG="logs/cron_publish.log"

RUNTIME_EST="${CSVG_RUNTIME_EST_MIN:-110}"     # measured launch->publish (1h09m/1h46m/1h59m)
MAX_DAILY="${CSVG_MAX_DAILY_PUBLISHES:-4}"     # published-per-day cap (YouTube quota ~4)
COOLDOWN="${CSVG_CRON_COOLDOWN_MIN:-90}"       # min seconds-age of cron_last_launch to fire again

# ---- arg parsing (--region india|global | --dry-run) ----
REGION_OVERRIDE=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --region) REGION_OVERRIDE="${2:-}"; shift 2 ;;
        --region=*) REGION_OVERRIDE="${1#--region=}"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        india|global) REGION_OVERRIDE="$1"; shift ;;
        *) shift ;;
    esac
done

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }

# ----------------------------  Periodic competitor benchmark  -------------------------
# Keyless competitor scraper (competitor_scraper.py): benchmarks competitor
# titles / cadence / engagement against THIS pipeline and writes
# logs/competitor_data.json + a gap report. Runs on its own cooldown (default
# weekly) with a lock so we never hammer YouTube or run two scrapes at once.
# It is non-fatal: a failure is logged but never blocks a publish.
BENCH_COOLDOWN_DAYS="${CSVG_COMPETITOR_BENCHMARK_COOLDOWN_DAYS:-7}"
BENCH_CHANNELS="${CSVG_COMPETITOR_CHANNELS:-FinanceBureauOfficial,AIUncovered,economicsexplained}"
BENCH_LOCK="logs/.competitor_benchmark.lock"
BENCH_LAST="logs/competitor_last_benchmark"

run_competitor_benchmark() {
    local age=9999
    if [ -f "$BENCH_LAST" ]; then
        age=$(( ($(date +%s) - $(stat -c %Y "$BENCH_LAST")) / 86400 ))
    fi
    if [ "$age" -lt "$BENCH_COOLDOWN_DAYS" ]; then
        log "BENCHMARK skip (last ${age}d < ${BENCH_COOLDOWN_DAYS}d cooldown)"
        return 0
    fi
    if [ -e "$BENCH_LOCK" ]; then
        log "BENCHMARK skip (lock present, concurrent run)"
        return 0
    fi
    touch "$BENCH_LOCK"
    log "BENCHMARK start channels=${BENCH_CHANNELS}"
    if ./venv/bin/python competitor_scraper.py --channels "$BENCH_CHANNELS" --limit 15 --no-script \
            >> "logs/competitor_benchmark.log" 2>&1; then
        log "BENCHMARK ok -> logs/competitor_data.json"
    else
        log "BENCHMARK failed (see logs/competitor_benchmark.log)"
    fi
    rm -f "$BENCH_LOCK"
    touch "$BENCH_LAST"
}
run_competitor_benchmark

# ----------------------------  Guard 1: already running  ------------------------------
if pgrep -f "[m]ain.py" >/dev/null 2>&1; then
    log "SKIP pipeline already running (main.py present)"
    exit 0
fi
if [ -f "logs/pipeline_heartbeat.json" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "logs/pipeline_heartbeat.json") ))
    if [ "$age" -lt 600 ]; then
        log "SKIP heartbeat fresh ${age}s (a run just finished)"
        exit 0
    fi
fi

# ----------------------------  Guard 2: daily publish cap  ----------------------------
DCOUNT=$(python3 - <<'PY'
import json, datetime
try:
    data = json.load(open("published_topics.json"))
except Exception:
    data = []
if isinstance(data, dict):
    data = data.get("published_topics") or data.get("topics") or []
today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
n = 0
for t in data:
    if isinstance(t, dict):
        ts = t.get("published_at") or t.get("timestamp") or t.get("date") or ""
    else:
        ts = getattr(t, "published_at", "") or ""
    if str(ts).startswith(today):
        n += 1
print(n)
PY
)
DCOUNT="${DCOUNT:-0}"
if [ "$DCOUNT" -ge "$MAX_DAILY" ]; then
    log "SKIP daily publish cap reached (${DCOUNT}/${MAX_DAILY})"
    exit 0
fi

# ----------------------------  Guard 3: cooldown  ----------------------------
if [ -f "logs/cron_last_launch" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "logs/cron_last_launch") ))
    if [ "$age" -lt "$COOLDOWN" ]; then
        log "SKIP cooldown ${age}s < ${COOLDOWN}s (last launch $(date -u -d @$(stat -c %Y logs/cron_last_launch) +%H:%M))"
        exit 0
    fi
fi

# ----------------------------  Launch window check  ----------------------------
tod=$(( 10#$(date -u +%H) * 60 + 10#$(date -u +%M) )) # UTC minutes-of-day

# These are the two windows that back-time a publish into a peak viewing window.
INDIAPM_SLOT=0; GLOBALS_SLOT=0
if [ "$tod" -ge 660 ] && [ "$tod" -le 740 ]; then INDIAPM_SLOT=1; fi    # 11:00-12:20
if [ "$tod" -ge 810 ] && [ "$tod" -le 870 ]; then GLOBALS_SLOT=1; fi    # 13:30-14:30

if [ "$INDIAPM_SLOT" -ne 1 ] && [ "$GLOBALS_SLOT" -ne 1 ]; then
    log "SKIP outside launch windows (now ${tod}m UTC; back-time +${RUNTIME_EST}m into peak window)"
    exit 0
fi

# Region is dynamic (chosen inside the pipeline). Only an explicit --region
# override is forwarded.
LAUNCH_ARGS=""
if [ -n "$REGION_OVERRIDE" ]; then
    LAUNCH_ARGS="--${REGION_OVERRIDE}"
fi

log "LAUNCH slot=$( (( INDIAPM_SLOT==1 )) && echo india-pm || echo global-am ) now=${tod}m published_today=${DCOUNT} (dynamic region, override='${REGION_OVERRIDE}')"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "WOULD LAUNCH: bash run_production.sh ${LAUNCH_ARGS}"
    exit 0
fi

touch "logs/cron_last_launch"
# run_production.sh detaches by itself; the cron job stays short and cheap.
# Region is decided by the pipeline unless --region was explicitly given.
nohup bash run_production.sh ${LAUNCH_ARGS} >> "$LOG" 2>&1 &
log "spawned run_production.sh ${LAUNCH_ARGS} pid=$!"
exit 0