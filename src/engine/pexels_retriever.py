import os
from typing import Optional
import requests
from typing import List, Dict, Any


class PexelsMediaRetriever:
    """
    Pexels Media Retriever for free stock photos and videos.
    Returns results shaped like Pixabay's retriever (largeImageURL / video_url)
    so the producer's free-visual path can use either source interchangeably.
    Requires PEXELS_API_KEY (Authorization header).
    """

    def __init__(self):
        pass

    @property
    def api_key(self) -> Optional[str]:
        # Resolve at call time so load_dotenv() in main.py is honored even if this
        # module was imported before the .env was loaded.
        return os.getenv("PEXELS_API_KEY")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": self.api_key} if self.api_key else {}

    def search_images(self, query: str, image_type: str = "photo", limit: int = 3) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            res = requests.get(
                "https://api.pexels.com/v1/search",
                params={"query": query, "per_page": max(1, int(limit)), "orientation": "landscape"},
                headers=self._headers(), timeout=6,
            )
            if res.status_code == 200:
                out = []
                for p in res.json().get("photos", []):
                    src = p.get("src", {})
                    out.append({
                        "id": p.get("id"),
                        "pageURL": p.get("url"),
                        "largeImageURL": src.get("large") or src.get("original"),
                        "user": p.get("photographer"),
                    })
                return out
        except Exception as e:
            print(f"[Pexels] Image search error: {e}")
        return []

    def search_videos(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            res = requests.get(
                "https://api.pexels.com/videos/search",
                params={"query": query, "per_page": max(1, int(limit)), "orientation": "landscape"},
                headers=self._headers(), timeout=8,
            )
            if res.status_code == 200:
                out = []
                for v in res.json().get("videos", []):
                    files = [f for f in v.get("video_files", []) if f.get("link")]
                    if not files:
                        continue
                    files.sort(key=lambda f: (f.get("height") or 0))
                    hd = [f for f in files if (f.get("height") or 0) >= 720]
                    chosen = (hd or files)[0]
                    out.append({
                        "id": v.get("id"),
                        "video_url": chosen.get("link"),
                        "duration": v.get("duration"),
                    })
                return out
        except Exception as e:
            print(f"[Pexels] Video search error: {e}")
        return []


pexels_retriever = PexelsMediaRetriever()
