import os
import time
import json
import logging
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional

logger = logging.getLogger("CSVG_PIPELINE")

PUBLIC_UA = os.getenv(
    "REDDIT_PUBLIC_USER_AGENT",
    "linux:buzzdropfeed-discovery:v2.0 (read-only public JSON)",
)


class RedditJsonClient:
    """
    Read-only Reddit client using Reddit's public JSON endpoints (.json suffix).
    Requires NO OAuth credentials. Used for thread discovery and comment context.
    Limited to ~10 requests/min unauthenticated; callers must pace.
    """

    BASE = "https://www.reddit.com"

    def __init__(self, min_interval: float = 7.0):
        self._last_call = 0.0
        self._min_interval = min_interval

    def _throttle(self):
        now = time.time()
        wait = self._min_interval - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _get(self, path: str, params: Optional[Dict[str, str]] = None) -> Optional[Any]:
        url = self.BASE + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        self._throttle()
        req = urllib.request.Request(url, headers={"User-Agent": PUBLIC_UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"[RedditJsonClient] GET {url} failed: {e}")
            return None

    def search_active_threads(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        data = self._get(
            "/search.json",
            {"q": query, "sort": "relevance", "t": "day", "limit": str(limit)},
        )
        results = []
        if not data or "data" not in data or "children" not in data["data"]:
            return results
        for child in data["data"]["children"]:
            post = (child or {}).get("data") or {}
            if post.get("locked") or post.get("archived") or post.get("over_18"):
                continue
            results.append({
                "id": post.get("id"),
                "fullname": post.get("name"),
                "title": post.get("title"),
                "selftext": post.get("selftext") or "",
                "subreddit": post.get("subreddit"),
                "url": post.get("url"),
                "permalink": post.get("permalink"),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
            })
        return results

    def get_comments_context(self, subreddit: str, thread_id: str, max_comments: int = 3) -> str:
        data = self._get(f"/r/{subreddit}/comments/{thread_id}.json")
        if not data or len(data) < 2:
            return ""
        comments = data[1].get("data", {}).get("children", [])
        lines = []
        for child in comments:
            comment = (child or {}).get("data") or {}
            author = comment.get("author") or "[deleted]"
            body = (comment.get("body") or "")[:300]
            lines.append(f"- Comment by u/{author}: {body}")
            if len(lines) >= max_comments:
                break
        return "\n".join(lines)