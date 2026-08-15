"""
Channel Phase Manager — reads YouTube Analytics API daily to determine
which operating phase the channel is in, and auto-transitions between:
  GROWTH  → optimise for subscribers + watch time (pre-YPP)
  REVENUE → optimise for RPM × Views × MidRoll (post-YPP)
  SCALE   → optimise for RPM + brand deals (10K+ subs)
"""
import os
import json
import datetime
from typing import Dict, Any

from src.schemas.state import (
    ChannelStats,
    CHANNEL_PHASE_GROWTH,
    CHANNEL_PHASE_REVENUE,
    CHANNEL_PHASE_SCALE,
    YPP_SUBS_THRESHOLD,
    YPP_WATCH_HOURS_THRESHOLD,
    YPP_SCALE_SUBS_THRESHOLD,
)

# Path to persisted channel stats (updated daily by this manager)
_STATS_FILE = os.path.join(os.path.dirname(__file__), "../../channel_stats.json")
_STATS_FILE = os.path.abspath(_STATS_FILE)

# ── Revenue goal (single source of truth) ───────────────────────────────────
# Channel goal: $2,000 / month from ad revenue. Per-video minimums are DERIVED,
# never hardcoded, so the gate, the competitor-volume filter and the YPP
# watch-time estimate all agree with the actual publish cadence:
#   per-video gate = MONTHLY_REVENUE_TARGET_USD / (TARGET_DAILY_PUBLISHES × 30)
# e.g. $2,000 / (2 × 30) = $33.33/video at 2 publishes/day (the cron cadence).
MONTHLY_REVENUE_TARGET_USD = max(0.0, float(os.getenv("CSVG_MONTHLY_REVENUE_TARGET_USD", "2000.0")))
TARGET_DAILY_PUBLISHES = max(1, int(os.getenv("CSVG_TARGET_DAILY_PUBLISHES", "2")))

# Revenue minimum gate per video (USD) — only enforced in REVENUE/SCALE phase.
# Derived from the monthly goal ÷ cadence, aligned with cron_publish.sh (2 slots/day).
REVENUE_GATE_MIN_USD = round(MONTHLY_REVENUE_TARGET_USD / (TARGET_DAILY_PUBLISHES * 30.0), 2)

# Daily publish launch slots in UTC — MUST match cron_publish.sh's two launch
# bands ([11:00-12:20] and [13:30-14:30] UTC back-timed from the ~1h50m runtime):
#   india window -> launch ~11:20 UTC   (publish ~13:10 UTC / ~6:40pm IST)
#   global window -> launch ~13:50 UTC  (publish ~15:40 UTC / ~11:40am ET)
DAILY_PUBLISH_SLOTS_UTC = ["11:20", "13:50"]

# Niche allocation targets for the 2-publish/day schedule (cron slot → preferred niche).
# Slot index 0-1 maps to DAILY_PUBLISH_SLOTS_UTC; dynamic region still overrides.
SLOT_NICHE_MAP = {
    0: {"niche": "Technology & Artificial Intelligence", "audience_type": "tech"},
    1: {"niche": "Personal Finance & Investing",         "audience_type": "investor"},
}


def _determine_phase(subscribers: int, watch_hours: int) -> str:
    """Pure function: determine channel phase from subscriber + watch hour counts."""
    if subscribers >= YPP_SCALE_SUBS_THRESHOLD:
        return CHANNEL_PHASE_SCALE
    elif subscribers >= YPP_SUBS_THRESHOLD and watch_hours >= YPP_WATCH_HOURS_THRESHOLD:
        return CHANNEL_PHASE_REVENUE
    else:
        return CHANNEL_PHASE_GROWTH


def _load_stats_from_disk() -> Dict[str, Any]:
    """Load persisted channel stats from JSON file."""
    if os.path.exists(_STATS_FILE):
        try:
            with open(_STATS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_stats_to_disk(stats: Dict[str, Any]) -> None:
    """Persist channel stats to JSON file."""
    try:
        os.makedirs(os.path.dirname(_STATS_FILE), exist_ok=True)
        with open(_STATS_FILE, "w") as f:
            json.dump(stats, f, indent=2)
    except IOError:
        pass


def _fetch_from_youtube_analytics() -> Dict[str, Any]:
    """
    Fetches subscriber count, total views, and video count from YouTube Data API.
    Falls back to environment variables if API is unavailable.
    Watch hours are NOT available from Data API v3 — uses CHANNEL_WATCH_HOURS env.
    """
    # 1. Try YouTube Data API v3 (channel statistics)
    try:
        import googleapiclient.discovery
        api_key = os.getenv("YOUTUBE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        channel_id = os.getenv("YOUTUBE_CHANNEL_ID")
        if api_key and channel_id:
            yt = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
            resp = yt.channels().list(
                part="statistics",
                id=channel_id
            ).execute()
            items = resp.get("items", [])
            if items:
                stats = items[0]["statistics"]
                subs = int(stats.get("subscriberCount", 0))
                views = int(stats.get("viewCount", 0))
                videos = int(stats.get("videoCount", 0))
                # Watch hours not in Data API v3 — use Analytics API or env override
                watch_hours = int(os.getenv("CHANNEL_WATCH_HOURS", "0"))
                return {
                    "subscribers": subs,
                    "total_watch_hours": watch_hours,
                    "total_views": views,
                    "total_videos": videos,
                }
    except Exception:
        pass

    # 2. Fall back to environment variable overrides (set manually from YouTube Studio)
    subs = int(os.getenv("CHANNEL_SUBSCRIBERS", "0"))
    watch_hours = int(os.getenv("CHANNEL_WATCH_HOURS", "0"))
    views = int(os.getenv("CHANNEL_VIEWS", "0"))
    videos = int(os.getenv("CHANNEL_VIDEOS", "0"))
    return {
        "subscribers": subs,
        "total_watch_hours": watch_hours,
        "total_views": views,
        "total_videos": videos,
    }


def get_channel_stats(force_refresh: bool = False) -> ChannelStats:
    """
    Returns the current ChannelStats, refreshing from YouTube API if:
    - force_refresh=True, OR
    - stats file is missing or older than 24 hours
    """
    cached = _load_stats_from_disk()
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Check if cache is fresh (< 24 hours old)
    is_stale = True
    if cached.get("last_updated"):
        try:
            last = datetime.datetime.fromisoformat(cached["last_updated"])
            last = last.replace(tzinfo=datetime.timezone.utc) if last.tzinfo is None else last
            is_stale = (datetime.datetime.now(datetime.timezone.utc) - last).total_seconds() > 86400
        except (ValueError, TypeError):
            is_stale = True

    if force_refresh or is_stale:
        live = _fetch_from_youtube_analytics()
        subs = live["subscribers"]
        watch_hours = live["total_watch_hours"]
        views = live.get("total_views", 0)
        videos = live.get("total_videos", 0)
        phase = _determine_phase(subs, watch_hours)
        ypp_unlocked = (subs >= YPP_SUBS_THRESHOLD and watch_hours >= YPP_WATCH_HOURS_THRESHOLD)

        data = {
            "subscribers": subs,
            "total_watch_hours": watch_hours,
            "total_views": views,
            "total_videos": videos,
            "channel_phase": phase,
            "last_updated": now_utc,
            "ypp_unlocked": ypp_unlocked,
        }
        _save_stats_to_disk(data)
        cached = data

    return ChannelStats(
        subscribers=cached.get("subscribers", 0),
        total_watch_hours=cached.get("total_watch_hours", 0),
        total_views=cached.get("total_views", 0),
        total_videos=cached.get("total_videos", 0),
        channel_phase=cached.get("channel_phase", CHANNEL_PHASE_GROWTH),
        last_updated=cached.get("last_updated", now_utc),
        ypp_unlocked=cached.get("ypp_unlocked", False),
    )


def get_topsis_weights(channel_phase: str) -> list:
    """Returns the correct TOPSIS weight vector for the current channel phase.
    Single source of truth is ``src.engine.topic_topsis`` (the 8-criterion,
    revenue-led vectors); this is a thin delegate so nothing drifts."""
    from src.engine.topic_topsis import (
        TOPSIS_WEIGHTS_GROWTH, TOPSIS_WEIGHTS_REVENUE, TOPSIS_WEIGHTS_SCALE,
    )
    if channel_phase == CHANNEL_PHASE_SCALE:
        return TOPSIS_WEIGHTS_SCALE
    elif channel_phase == CHANNEL_PHASE_REVENUE:
        return TOPSIS_WEIGHTS_REVENUE
    else:
        return TOPSIS_WEIGHTS_GROWTH


def get_next_publish_time_utc(slot_index: int = 0) -> str:
    """
    Returns the next upcoming UTC publish datetime string for the given slot index (0-3).
    Used by Publisher to schedule optimal upload times targeting US/UK peak hours.
    """
    slot_time_str = DAILY_PUBLISH_SLOTS_UTC[slot_index % len(DAILY_PUBLISH_SLOTS_UTC)]
    h, m = map(int, slot_time_str.split(":"))
    now = datetime.datetime.now(datetime.timezone.utc)
    candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if candidate <= now:
        candidate += datetime.timedelta(days=1)
    return candidate.isoformat()


def get_ypp_progress_report(stats: ChannelStats) -> Dict[str, Any]:
    """
    Returns a human-readable progress report toward YPP unlock.
    Useful for logging and monitoring during GROWTH phase.
    """
    subs_pct = min(100.0, stats.subscribers / YPP_SUBS_THRESHOLD * 100)
    hrs_pct  = min(100.0, stats.total_watch_hours / YPP_WATCH_HOURS_THRESHOLD * 100)

    # At TARGET_DAILY_PUBLISHES/day, 13 min each → estimated watch-mins/day from a
    # conservative new-channel view estimate (100 views/video).
    views_per_day_estimate = TARGET_DAILY_PUBLISHES * 100
    watch_mins_per_day = views_per_day_estimate * 13.0 * 0.42
    hrs_remaining = max(0, YPP_WATCH_HOURS_THRESHOLD - stats.total_watch_hours)
    days_to_4k_hrs = (hrs_remaining * 60) / max(1, watch_mins_per_day)

    return {
        "channel_phase": stats.channel_phase,
        "ypp_unlocked": stats.ypp_unlocked,
        "subscribers": stats.subscribers,
        "subs_progress_pct": round(subs_pct, 1),
        "subs_remaining": max(0, YPP_SUBS_THRESHOLD - stats.subscribers),
        "watch_hours": stats.total_watch_hours,
        "watch_hours_progress_pct": round(hrs_pct, 1),
        "watch_hours_remaining": hrs_remaining,
        "estimated_days_to_ypp": round(days_to_4k_hrs, 0),
    }


# Singleton for easy import
channel_phase_manager = type("ChannelPhaseManager", (), {
    "get_channel_stats": staticmethod(get_channel_stats),
    "get_topsis_weights": staticmethod(get_topsis_weights),
    "get_next_publish_time_utc": staticmethod(get_next_publish_time_utc),
    "get_ypp_progress_report": staticmethod(get_ypp_progress_report),
    "REVENUE_GATE_MIN_USD": REVENUE_GATE_MIN_USD,
    "MONTHLY_REVENUE_TARGET_USD": MONTHLY_REVENUE_TARGET_USD,
    "TARGET_DAILY_PUBLISHES": TARGET_DAILY_PUBLISHES,
    "SLOT_NICHE_MAP": SLOT_NICHE_MAP,
    "DAILY_PUBLISH_SLOTS_UTC": DAILY_PUBLISH_SLOTS_UTC,
})()
