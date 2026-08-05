import os
import json
import datetime
import threading
from typing import Optional, Dict, Any
from src.engine.logger import logger


class YouTubeTopicDemand:
    """
    Fetches REAL, forward-looking topic demand from public YouTube Data API v3
    (search.list + videos.list statistics). Used to replace the fake vph_score
    proxy in RSS ingestion with genuine competitor view volume.

    Only public metadata is used (no OAuth required) via a developer API key
    (YOUTUBE_API_KEY / GOOGLE_API_KEY). All calls are cached per query.

    A persistent DAILY search-call budget (YT_SEARCH_DAILY_BUDGET, default 30)
    prevents a single run/session from exhausting the whole 10k-unit/day free
    quota (each search.list = 100 units). When the budget is used up, calls are
    skipped and the pipeline falls back to the proxy — silently and non-fatally.
    All messages go through the structured logger, not raw stdout.
    """

    QUOTA_FILE = "yt_demand_quota.json"

    def __init__(self):
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._client = None
        self._quota = self._load_quota()
        # API keys in priority order: primary, then fallback(s). Used to rotate
        # when the primary hits a daily quota (429) error.
        self._keys = [
            k for k in (
                os.getenv("YOUTUBE_API_KEY"),
                os.getenv("GOOGLE_API_KEY"),
                os.getenv("YOUTUBE_API_KEY_FALLBACK"),
            ) if k
        ]
        self._key_idx = 0

    # ── quota budget ─────────────────────────────────────────────────────────
    def _load_quota(self) -> dict:
        try:
            if os.path.exists(self.QUOTA_FILE):
                with open(self.QUOTA_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            pass
        return {}

    def _save_quota(self) -> None:
        try:
            with open(self.QUOTA_FILE, "w", encoding="utf-8") as f:
                json.dump(self._quota, f, indent=2)
        except (IOError, OSError):
            pass

    def _can_search(self) -> bool:
        budget = int(os.getenv("YT_SEARCH_DAILY_BUDGET", "30"))
        key = datetime.date.today().isoformat()
        used = int(self._quota.get(key, 0))
        if used >= budget:
            logger.warning(
                "YT_DEMAND",
                f"YouTube search daily budget reached ({used}/{budget}); competitor feed paused for today (proxy fallback).",
                component="YOUTUBE_DEMAND",
            )
            return False
        self._quota[key] = used + 1
        self._save_quota()
        return True

    # ── client ───────────────────────────────────────────────────────────────
    def _get_client(self, key: Optional[str] = None):
        key = key or (self._keys[self._key_idx] if self._keys else None)
        if not key:
            return None
        try:
            import googleapiclient.discovery
            import httplib2
            http = httplib2.Http(timeout=8)
            return googleapiclient.discovery.build(
                "youtube", "v3", developerKey=key, http=http
            )
        except Exception as e:
            logger.warning("YT_DEMAND", f"Failed to build YouTube client: {e}", component="YOUTUBE_DEMAND")
            return None

    def fetch_topic_demand(self, query: str, max_videos: int = 5) -> Optional[Dict[str, Any]]:
        """Cached fetch of competitor view volume for a topic query (budget-gated)."""
        cache_key = query.strip().lower()
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        if not self._can_search():
            with self._lock:
                self._cache[cache_key] = None
            return None
        result = self._fetch(query, max_videos)
        with self._lock:
            self._cache[cache_key] = result
        return result

    def _fetch(self, query: str, max_videos: int) -> Optional[Dict[str, Any]]:
        if not self._keys:
            return None
        n = len(self._keys)
        last_err = None
        for _ in range(n):
            key_idx = self._key_idx
            client = self._get_client(self._keys[key_idx])
            if client is None:
                self._key_idx = (key_idx + 1) % n
                continue
            try:
                return self._fetch_with_client(client, query, max_videos)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # Only rotate keys on quota/rate-limit (429/403); other errors are fatal per-query
                if "quota" in msg or "429" in msg or "rateLimitExceeded" in msg or "403" in msg:
                    logger.warning(
                        "YT_DEMAND",
                        f"Key {key_idx + 1}/{n} quota/403 on '{query}'; trying next key.",
                        component="YOUTUBE_DEMAND",
                    )
                    self._key_idx = (key_idx + 1) % n  # exhaust this key for the process
                    continue
                logger.warning("YT_DEMAND", f"Fetch failed for '{query}': {e}", component="YOUTUBE_DEMAND")
                return None
        logger.warning(
            "YT_DEMAND",
            f"All {n} YouTube key(s) failed/quota-exceeded for '{query}': {last_err}",
            component="YOUTUBE_DEMAND",
        )
        return None

    def _fetch_with_client(self, client, query: str, max_videos: int) -> Optional[Dict[str, Any]]:
        search_resp = client.search().list(
            part="id",
            type="video",
            q=query,
            maxResults=max_videos,
            order="viewCount",
            relevanceLanguage="en",
            safeSearch="none",
        ).execute()

        ids = [
            it["id"]["videoId"]
            for it in search_resp.get("items", [])
            if it.get("id", {}).get("kind") == "youtube#video"
        ]
        if not ids:
            return None

        vids = client.videos().list(
            part="statistics,snippet", id=",".join(ids)
        ).execute()

        now = datetime.datetime.now(datetime.timezone.utc)
        total_views = 0
        total_age_days = 0.0
        counted = 0

        for item in vids.get("items", []):
            vc = int(item.get("statistics", {}).get("viewCount", 0) or 0)
            total_views += vc
            pub = item.get("snippet", {}).get("publishedAt")
            age_days = 30.0
            if pub:
                try:
                    dt = datetime.datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    age_days = max(0.5, (now - dt).total_seconds() / 86400.0)
                except Exception:
                    age_days = 30.0
            total_age_days += age_days
            counted += 1

        if counted == 0 or total_age_days <= 0:
            return None

        hourly = (total_views / total_age_days) / 24.0
        return {
            "query": query,
            "video_count": counted,
            "total_views": total_views,
            "avg_views_per_video": round(total_views / counted, 2),
            "views_per_hour": round(hourly, 2),
            "competitor_30d_avg_views": round((total_views / total_age_days) * 30.0, 2),
        }


youtube_topic_demand = YouTubeTopicDemand()
