import os
import requests
from typing import List, Dict, Any, Optional

class PixabayMediaRetriever:
    """
    Pixabay Media Retriever for free stock images, vector SVGs, and video clips.
    """
    def __init__(self):
        self.api_key = os.getenv("PIXABAY_API_KEY", "57001932-5bafbb571b0450127b8dd7ca6")

    def search_images(self, query: str, image_type: str = "photo", limit: int = 3) -> List[Dict[str, Any]]:
        """
        Searches Pixabay for images (photo, illustration, vector).
        """
        url = "https://pixabay.com/api/"
        # Pixabay API requires 3 <= per_page <= 200; clamp so limit=1 still works.
        params = {
            "key": self.api_key,
            "q": query,
            "image_type": image_type,
            "per_page": max(3, min(int(limit), 200))
        }
        try:
            res = requests.get(url, params=params, timeout=5)
            if res.status_code == 200:
                hits = res.json().get("hits", [])
                results = []
                for hit in hits:
                    results.append({
                        "id": hit.get("id"),
                        "pageURL": hit.get("pageURL"),
                        "previewURL": hit.get("previewURL"),
                        "webformatURL": hit.get("webformatURL"),
                        "largeImageURL": hit.get("largeImageURL"),
                        "user": hit.get("user")
                    })
                return results
        except Exception as e:
            print(f"[Pixabay] Image search error: {e}")
        return []

    def search_videos(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Searches Pixabay for stock videos.
        """
        url = "https://pixabay.com/api/videos/"
        params = {
            "key": self.api_key,
            "q": query,
            "per_page": max(3, min(int(limit), 200))
        }
        try:
            res = requests.get(url, params=params, timeout=5)
            if res.status_code == 200:
                hits = res.json().get("hits", [])
                results = []
                for hit in hits:
                    videos_dict = hit.get("videos", {})
                    # Prefer medium (usually 960x540 or 1280x720) or large
                    video_url = (
                        videos_dict.get("medium", {}).get("url") or 
                        videos_dict.get("large", {}).get("url") or 
                        videos_dict.get("small", {}).get("url")
                    )
                    if video_url:
                        results.append({
                            "id": hit.get("id"),
                            "pageURL": hit.get("pageURL"),
                            "video_url": video_url,
                            "duration": hit.get("duration"),
                            "user": hit.get("user")
                        })
                return results
        except Exception as e:
            print(f"[Pixabay] Video search error: {e}")
        return []

pixabay_retriever = PixabayMediaRetriever()
