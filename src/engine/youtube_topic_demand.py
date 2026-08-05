import os
import json
import datetime
import threading
from typing import Optional, Dict, Any, List
from src.engine.logger import logger

# Niche -> broad seed keyword used ONCE per niche per refresh to discover a pool
# of competitor video IDs. This keeps search.list usage low (the scarce quota),
# while per-topic demand is served mostly via cheap, high-limit videos.list batch
# stats (VideoBatchGetStats metric).
_NICHE_SEED = {
    "Personal Finance & Investing": "personal finance investing",
    "Global Economics & Finance": "economics news",
    "Technology & Artificial Intelligence": "technology news",
    "Business & Entrepreneurship": "business news",
    "Health & Science": "health news",
    "Health & Wellness": "health wellness",
    "Legal & Law": "legal news",
    "Real Estate": "real estate news",
    "Space & Scientific Innovation": "space science news",
    "Geopolitics & World Affairs": "geopolitics news",
    "Global Trends & Infotainment": "trending news",
    "Global Trends & Cultural Infotainment": "trending news",
}
_DEFAULT_NICHE_KEY = "Trending"
POOLS_FILE = "yt_demand_pools.json"
QUOTA_FILE = "yt_demand_quota.json"
REFRESH_DAYS = 2          # re-seed a niche pool after this many days
MAX_POOL_IDS = 50         # videos.list batch limit per call


class YouTubeTopicDemand:
    """
    Fetches REAL, forward-looking competitor view demand from public YouTube Data
    API v3 using minimal search.list (scarce quota) + cheap, high-limit videos.list
    batch statistics (VideoBatchGetStats).

    Strategy:
      - Each niche is seeded (a handful of search.list calls/day) into a persistent
        pool of competitor video IDs.
      - Per-topic demand is served by a single videos.list(id=<pool ids up to 50>)
        call (~1 unit, high daily limit) — nothing is re-searched per topic.

    Falls back to the proxy (returns None) non-fatally on any failure/quota cap,
    and rotates across multiple API keys on 429/403.
    """

    def __init__(self):
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._pools: Dict[str, dict] = self._load_pools()
        self._quota = self._load_quota()
        self._keys = [
            k for k in (
                os.getenv("YOUTUBE_API_KEY"),
                os.getenv("GOOGLE_API_KEY"),
                os.getenv("YOUTUBE_API_KEY_FALLBACK"),
                os.getenv("YOUTUBE_API_KEY_FALLBACK2"),
            ) if k
        ]
        self._key_idx = 0

    # ── persistence ───────────────────────────────────────────────────────────
    def _load_pools(self) -> dict:
        try:
            if os.path.exists(POOLS_FILE):
                with open(POOLS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            pass
        return {}

    def _save_pools(self) -> None:
        try:
            with open(POOLS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._pools, f, indent=2)
        except (IOError, OSError):
            pass

    def _load_quota(self) -> dict:
        try:
            if os.path.exists(QUOTA_FILE):
                with open(QUOTA_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            pass
        return {}

    def _save_quota(self) -> None:
        try:
            with open(QUOTA_FILE, "w", encoding="utf-8") as f:
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
                f"YouTube search daily budget reached ({used}/{budget}); using niche-pool batch stats only.",
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

    # ── niche mapping ─────────────────────────────────────────────────────────
    @staticmethod
    def _niche_key(niche_category: str, query: str) -> str:
        if niche_category and niche_category in _NICHE_SEED:
            return niche_category
        if niche_category:
            return niche_category
        # fall back to first word of the query as a coarse niche tag
        words = [w for w in (query or "").split() if len(w) > 3][:2]
        return " ".join(words) if words else _DEFAULT_NICHE_KEY

    # ── pool lifecycle ────────────────────────────────────────────────────────
    def _ensure_pool(self, niche_key: str) -> List[str]:
        """Returns a fresh-enough pool of video IDs for a niche; seeds it (1 search.list)
        if missing/stale. Returns [] if it can't seed right now (budget/api)."""
        pool = self._pools.get(niche_key) or {}
        seeded = pool.get("seeded_date", "")
        try:
            fresh = seeded >= (datetime.date.today() - datetime.timedelta(days=REFRESH_DAYS)).isoformat()
        except Exception:
            fresh = False
        if pool.get("ids") and fresh:
            return pool["ids"]

        if not self._can_search():
            return (pool or {}).get("ids", []) or []

        seed_q = _NICHE_SEED.get(niche_key, niche_key.replace("_", " "))
        ids = self._search_ids(seed_q, 10)
        if not ids:
            return (pool or {}).get("ids", []) or []
        self._pools[niche_key] = {
            "seeded_date": datetime.date.today().isoformat(),
            "seed_query": seed_q,
            "ids": ids,
        }
        self._save_pools()
        return ids

    def _search_ids(self, query, max_results: int = 10) -> List[str]:
        n = len(self._keys)
        for _ in range(max(1, n)) if n else range(0):
            key_idx = self._key_idx
            client = self._get_client(self._keys[key_idx]) if n else None
            if client is None:
                if n: self._key_idx = (key_idx + 1) % n
                continue
            try:
                resp = client.search().list(
                    part="id", type="video", q=query,
                    maxResults=max_results, order="viewCount",
                    relevanceLanguage="en", safeSearch="none",
                ).execute()
                return [
                    it["id"]["videoId"]
                    for it in resp.get("items", [])
                    if it.get("id", {}).get("kind") == "youtube#video"
                ]
            except Exception as e:
                msg = str(e).lower()
                if "quota" in msg or "429" in msg or "403" in msg or "rateLimitExceeded" in msg:
                    self._key_idx = (key_idx + 1) % n
                    continue
                logger.warning("YT_DEMAND", f"search.list failed for '{query}': {e}", component="YOUTUBE_DEMAND")
                return []
        return []

    # ── demand compute (cheap videos.list batch) ─────────────────────────────
    def _stats_for_ids(self, ids: List[str], max_videos: int = 5) -> Optional[Dict[str, Any]]:
        if not ids:
            return None
        n = len(self._keys)
        for _ in range(max(1, n)) if n else range(0):
            key_idx = self._key_idx
            client = self._get_client(self._keys[key_idx]) if n else None
            if client is None:
                if n: self._key_idx = (key_idx + 1) % n
                continue
            try:
                vids = client.videos().list(
                    part="statistics,snippet", id=",".join(ids[:MAX_POOL_IDS])
                ).execute()
                return self._demand_from_items(vids.get("items", []), max_videos)
            except Exception as e:
                msg = str(e).lower()
                if "quota" in msg or "429" in msg or "403" in msg or "rateLimitExceeded" in msg:
                    self._key_idx = (key_idx + 1) % n
                    continue
                logger.warning("YT_DEMAND", f"videos.list failed: {e}", component="YOUTUBE_DEMAND")
                return None
        return None

    @staticmethod
    def _demand_from_items(items, max_videos: int = 5) -> Optional[Dict[str, Any]]:
        now = datetime.datetime.now(datetime.timezone.utc)
        total_views = 0
        total_age_days = 0.0
        counted = 0
        for item in items[:max_videos]:
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
            "video_count": counted,
            "total_views": total_views,
            "avg_views_per_video": round(total_views / counted, 2),
            "views_per_hour": round(hourly, 2),
            "competitor_30d_avg_views": round((total_views / total_age_days) * 30.0, 2),
        }

    # ── public entry ──────────────────────────────────────────────────────────
    def fetch_topic_demand(self, query: str, niche_category: str = "", max_videos: int = 5) -> Optional[Dict[str, Any]]:
        """Returns competitor demand for a topic using the niche's video-ID pool
        (cheap batch videos.list), seeding the pool with minimal search.list."""
        cache_key = (query or "").strip().lower()
        with self._lock:
            if cache_key and cache_key in self._cache:
                return self._cache[cache_key]

        niche_key = self._niche_key(niche_category, query)
        pool_ids = self._ensure_pool(niche_key)
        result = self._stats_for_ids(pool_ids, max_videos)

        with self._lock:
            if cache_key:
                self._cache[cache_key] = result
        return result


youtube_topic_demand = YouTubeTopicDemand()
