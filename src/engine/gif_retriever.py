import os
import requests
from typing import List, Dict, Any, Optional


class GIFMediaRetriever:
    """
    GIF & Meme Reaction Media Retriever integrating GIPHY, Tenor (Google), and Imgur APIs
    for high-engagement comedic visual inserts in YouTube Infotainment videos.
    """

    def __init__(self):
        self.giphy_key = os.getenv("GIPHY_API_KEY", "dc6zaTOxFJmzC")  # GIPHY public beta key
        self.tenor_key = os.getenv("TENOR_API_KEY")

    def search_giphy_reaction(self, query: str = "shocked reaction", limit: int = 3) -> List[Dict[str, str]]:
        """
        Queries GIPHY API for trending comedic reaction GIFs and mp4 video clips.
        """
        url = f"https://api.giphy.com/v1/gifs/search?api_key={self.giphy_key}&q={query}&limit={limit}&rating=g"
        try:
            res = requests.get(url, timeout=4)
            if res.status_code == 200:
                data = res.json().get("data", [])
                clips = []
                for item in data:
                    images = item.get("images", {})
                    mp4_url = images.get("original_mp4", {}).get("mp4") or images.get("fixed_height", {}).get("mp4")
                    gif_url = images.get("original", {}).get("url")
                    if mp4_url or gif_url:
                        clips.append({
                            "title": item.get("title", query),
                            "mp4_url": mp4_url or gif_url,
                            "source": "GIPHY"
                        })
                return clips
        except Exception:
            pass
        return []

    def search_tenor_gif(self, query: str = "money rain", limit: int = 3) -> List[Dict[str, str]]:
        """
        Queries Tenor GIF API (Google) for category-tagged comedic reaction clips.
        """
        if not self.tenor_key:
            return []

        url = f"https://tenor.googleapis.com/v2/search?q={query}&key={self.tenor_key}&limit={limit}"
        try:
            res = requests.get(url, timeout=4)
            if res.status_code == 200:
                results = res.json().get("results", [])
                clips = []
                for item in results:
                    media = item.get("media_formats", {}).get("mp4", {})
                    mp4_url = media.get("url")
                    if mp4_url:
                        clips.append({
                            "title": item.get("title", query),
                            "mp4_url": mp4_url,
                            "source": "Tenor (Google)"
                        })
                return clips
        except Exception:
            pass
        return []


# Global GIF Retriever Instance
gif_retriever = GIFMediaRetriever()
