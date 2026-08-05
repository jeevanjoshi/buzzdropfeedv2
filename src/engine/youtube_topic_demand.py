import os
import datetime
import threading
from typing import Optional, Dict, Any


class YouTubeTopicDemand:
    """
    Fetches REAL, forward-looking topic demand from public YouTube Data API v3
    (search.list + videos.list statistics). Used to replace the fake vph_score
    proxy in RSS ingestion with genuine competitor view volume.

    Only public metadata is used (no OAuth required) via a developer API key
    (YOUTUBE_API_KEY / GOOGLE_API_KEY). All calls are cached per query and
    gracefully fall back to None on any failure so the pipeline never crashes
    or hangs when the API is unavailable or quota-limited.
    """

    def __init__(self):
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.getenv("YOUTUBE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        try:
            import googleapiclient.discovery
            import httplib2
            http = httplib2.Http(timeout=8)
            self._client = googleapiclient.discovery.build(
                "youtube", "v3", developerKey=api_key, http=http
            )
            return self._client
        except Exception as e:
            print(f"[YTDemand] Failed to build YouTube client: {e}")
            return None

    def fetch_topic_demand(self, query: str, max_videos: int = 5) -> Optional[Dict[str, Any]]:
        """Cached fetch of competitor view volume for a topic query."""
        cache_key = query.strip().lower()
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        result = self._fetch(query, max_videos)
        with self._lock:
            self._cache[cache_key] = result
        return result

    def _fetch(self, query: str, max_videos: int) -> Optional[Dict[str, Any]]:
        client = self._get_client()
        if client is None:
            return None
        try:
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
        except Exception as e:
            print(f"[YTDemand] Fetch failed for '{query}': {e}")
            return None


youtube_topic_demand = YouTubeTopicDemand()
