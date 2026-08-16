"""
Analytics Feedback Loop — "top growth drivers" (vidIQ tactic).

Pulls per-video performance metrics from the YouTube Analytics API v2 for every
video the pipeline has published (correlated back to the topic that produced it
via the run's state checkpoint), persists them to ``logs/analytics_feedback.json``,
and exposes a *niche signal* — which audience/niche historically converts best
(subscriber-gain + retention weighted) — so the topic-selection layer can "double
down on what works" (FactRetriever applies it as a soft TOPSIS bias).

Design rules (mirrors channel_phase_manager / feedback_memory):
  - Thread-safe + atomic writes (os.replace).
  - Failsafe: never raises or crashes the pipeline — on any failure the store
    is left unchanged and ``get_audience_bias`` returns 1.0 (no bias).
  - Refresh is rate-limited (default every 6h) so a per-publish refresh can't
    burn Analytics quota; ``force=True`` bypasses (CLI / dashboard refresh).
  - Only REAL video ids are ever queried (``_is_real_video_id`` semantics:
    ``demo_*`` / ``hermetic-*`` are never treated as published).

Files:
  - ``logs/analytics_feedback.json`` — persisted per-video + niche-signal store.
"""
import os
import json
import glob
import datetime
import threading
from typing import Dict, Any, List, Optional


_FEEDBACK_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/analytics_feedback.json"))
_STATES_GLOB = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/state_*.json"))

# Only refresh if the store is older than this many seconds (default 6h).
_REFRESH_MIN_AGE_SEC = float(os.getenv("CSVG_ANALYTICS_REFRESH_MIN_AGE_SEC", "21600").strip() or 0.0)
# Max videos to pull per refresh (bounds Analytics API quota per run).
_MAX_REFRESH_VIDEOS = int(os.getenv("CSVG_ANALYTICS_MAX_VIDEOS", "30").strip() or 30)

# The metrics we ask YouTube Analytics for. ``averageViewPercentage`` (audience
# retention) and ``subscribersGained`` are the "hook retention" + "top growth
# drivers" signals from the vidIQ playbook.
_METRICS = [
    "views",
    "estimatedMinutesWatched",
    "averageViewDuration",
    "averageViewPercentage",
    "subscribersGained",
]


def _load_credentials():
    """Load + refresh the channel-owner OAuth credentials from token.json.
    Returns None when unavailable (never raises)."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        token_path = os.getenv("YOUTUBE_TOKEN_FILE", "token.json")
        if not os.path.exists(token_path):
            return None
        creds = Credentials.from_authorized_user_file(token_path)
        if creds is None:
            return None
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if not creds.valid:
            return None
        return creds
    except Exception:  # noqa: BLE001
        return None


def _load_raw() -> Dict[str, Any]:
    if not os.path.exists(_FEEDBACK_FILE):
        return {"schema": "analytics_feedback/v1", "videos": [], "niche_signal": {}, "captured_at": ""}
    try:
        with open(_FEEDBACK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"schema": "analytics_feedback/v1", "videos": [], "niche_signal": {}, "captured_at": ""}
        data.setdefault("videos", [])
        data.setdefault("niche_signal", {})
        return data
    except (json.JSONDecodeError, IOError):
        return {"schema": "analytics_feedback/v1", "videos": [], "niche_signal": {}, "captured_at": ""}


def _atomic_write(data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(_FEEDBACK_FILE), exist_ok=True)
    tmp_file = _FEEDBACK_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_file, _FEEDBACK_FILE)


def _is_real_video_id(video_id: Optional[str]) -> bool:
    if not video_id:
        return False
    vid = str(video_id).strip()
    if not vid or len(vid) > 64:
        return False
    low = vid.lower()
    return not low.startswith("demo_") and low != "demo_id" and not low.startswith("hermetic-")


def _discover_videos() -> List[Dict[str, Any]]:
    """Scan ``logs/state_*.json`` checkpoints for real published video ids and
    correlate them back to the topic (headline / audience_type / niche) and
    publish time. Returns a list of {video_id, headline, audience_type,
    niche_category, published_at} records (deduped by video_id)."""
    out: Dict[str, Dict[str, Any]] = {}
    for state_path in glob.glob(_STATES_GLOB):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        if not isinstance(state, dict):
            continue
        um = state.get("upload_metadata") or {}
        vid = (um or {}).get("video_id")
        if not _is_real_video_id(vid):
            continue
        topic = state.get("selected_topic") or {}
        out[vid] = {
            "video_id": vid,
            "headline": (topic or {}).get("headline", ""),
            "audience_type": (topic or {}).get("audience_type", ""),
            "niche_category": (topic or {}).get("niche_category", ""),
            "published_at": state.get("timestamp", ""),
        }
    return list(out.values())


def refresh(max_videos: int = _MAX_REFRESH_VIDEOS, force: bool = False) -> Dict[str, Any]:
    """
    Pull per-video analytics for the channel's published videos and persist them
    to ``logs/analytics_feedback.json``.

    Non-fatal and rate-limited: if the store was captured within
    ``CSVG_ANALYTICS_REFRESH_MIN_AGE_SEC`` seconds (default 6h) and ``force`` is
    false, this is a no-op that returns the existing store. Each video is queried
    independently; a single video's failure is skipped without aborting.

    Returns the (possibly unchanged) store dict.
    """
    store = _load_raw()
    captured = store.get("captured_at", "")
    if not force and captured:
        try:
            last = datetime.datetime.fromisoformat(captured)
            last = last.replace(tzinfo=datetime.timezone.utc) if last.tzinfo is None else last
            age = (datetime.datetime.now(datetime.timezone.utc) - last).total_seconds()
            if age < _REFRESH_MIN_AGE_SEC:
                return store
        except (ValueError, TypeError):
            pass

    creds = _load_credentials()
    if creds is None:
        return store

    videos = _discover_videos()
    if not videos:
        return store

    from googleapiclient.discovery import build

    try:
        ya = build("youtubeAnalytics", "v2", credentials=creds)
    except Exception:  # noqa: BLE001
        return store

    ids = "channel==MINE"
    today = datetime.date.today().isoformat()
    # Pull the most recently published videos first, bounded to max_videos.
    videos.sort(key=lambda v: v.get("published_at", ""), reverse=True)
    videos = videos[:max(1, int(max_videos))]

    existing = {v.get("video_id") for v in store.get("videos", [])}
    updated = []
    for v in videos:
        vid = v["video_id"]
        try:
            row = ya.reports().query(
                ids=ids,
                startDate="2005-01-01",
                endDate=today,
                metrics=",".join(_METRICS),
                filters=f"video=={vid}",
            ).execute()
            rows = row.get("rows") or []
            if not rows:
                continue
            vals = rows[0]
            record = {
                "video_id": vid,
                "headline": v.get("headline", ""),
                "audience_type": v.get("audience_type", ""),
                "niche_category": v.get("niche_category", ""),
                "published_at": v.get("published_at", ""),
                "views": int(vals[0] or 0),
                "watch_minutes": int(vals[1] or 0),
                "avg_view_duration_sec": float(vals[2] or 0.0),
                "avg_view_percentage": float(vals[3] or 0.0),
                "subscribers_gained": int(vals[4] or 0),
                "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        except Exception as e:  # noqa: BLE001
            print(f"[AnalyticsFeedback] Skip video {vid}: {e}")
            continue

        if vid in existing:
            _replace_record(store["videos"], vid, record)
        else:
            store["videos"].append(record)
        updated.append(vid)

    if not updated:
        return store

    store["niche_signal"] = _compute_niche_signal(store["videos"])
    store["captured_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _atomic_write(store)
    print(f"[AnalyticsFeedback] Refreshed analytics for {len(updated)} video(s); "
          f"{len(store['videos'])} total in store.")
    return store


def _replace_record(videos: List[Dict[str, Any]], video_id: str, record: Dict[str, Any]) -> None:
    for i, v in enumerate(videos):
        if v.get("video_id") == video_id:
            videos[i] = record
            return


def _compute_niche_signal(videos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-video metrics by audience_type into a normalized signal:
    which niches drive subscribers + retention (the "top growth drivers" signal).
    Returns {audience_type: {videos, views, subscribers_gained, avg_view_percentage,
    retention, signal}} where ``signal`` ∈ [0, 1] normalised across niches."""
    agg: Dict[str, Dict[str, Any]] = {}
    for v in videos:
        aud = (v.get("audience_type") or "general").strip() or "general"
        bucket = agg.setdefault(aud, {
            "videos": 0, "views": 0, "subscribers_gained": 0,
            "avg_view_percentage_sum": 0.0, "retention": 0.0, "signal": 0.0,
        })
        bucket["videos"] += 1
        bucket["views"] += int(v.get("views") or 0)
        bucket["subscribers_gained"] += int(v.get("subscribers_gained") or 0)
        bucket["avg_view_percentage_sum"] += float(v.get("avg_view_percentage") or 0.0)
    if not agg:
        return {}

    # Mean retention per niche (0..100).
    for aud, b in agg.items():
        n = max(1, b["videos"])
        b["avg_view_percentage"] = round(b["avg_view_percentage_sum"] / n, 2)
        b["retention"] = b["avg_view_percentage"]

    # Normalise the subscriber-gain signal (0..1) across niches; retention is
    # already a 0..100 fraction. Signal = 0.6 * sub_norm + 0.4 * retention/100.
    max_subs = max((b["subscribers_gained"] for b in agg.values()), default=1)
    max_subs = max(1, max_subs)
    for aud, b in agg.items():
        sub_norm = b["subscribers_gained"] / max_subs
        b["signal"] = round(0.6 * sub_norm + 0.4 * (b["avg_view_percentage"] / 100.0), 4)
    return agg


def get_niche_signal() -> Dict[str, Any]:
    """Return the cached niche signal (no API call). Empty dict when no data."""
    return _load_raw().get("niche_signal", {}) or {}


def get_audience_bias(audience_type: str, max_bias: float = 0.30) -> float:
    """
    Return a soft multiplier in ``[1.0, 1.0 + max_bias]`` favouring an audience
    type that historically converts (subscriber-gain + retention signal) — the
    FactRetriever's "double down on growth drivers" tie-break. Returns 1.0 (no
    bias) when the audience is unknown or the store has no signal.
    """
    if not audience_type:
        return 1.0
    aud = str(audience_type).strip().lower()
    signal = get_niche_signal().get(aud)
    if not signal or not isinstance(signal, dict):
        return 1.0
    # A niche with zero subscriber gain carries no positive signal to boost.
    if int(signal.get("subscribers_gained") or 0) <= 0:
        return 1.0
    s = float(signal.get("signal") or 0.0)
    return round(1.0 + max(0.0, min(max_bias, s * max_bias)), 4)


class AnalyticsFeedbackStore:
    """Thin object wrapper exposing the module functions as a singleton-like API
    (mirrors the codebase's ``topic_deduplicator`` / ``youtube_engagement`` pattern)."""
    def __init__(self):
        self._lock = threading.Lock()

    def refresh(self, max_videos: int = _MAX_REFRESH_VIDEOS, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            return refresh(max_videos=max_videos, force=force)

    def get_audience_bias(self, audience_type: str) -> float:
        return get_audience_bias(audience_type)

    def get_niche_signal(self) -> Dict[str, Any]:
        return get_niche_signal()

    @staticmethod
    def _discover_videos() -> List[Dict[str, Any]]:
        return _discover_videos()


analytics_feedback = AnalyticsFeedbackStore()