import requests
import re
from typing import Optional, List, Dict


class NASARetriever:
    """
    Retriever for NASA's Image and Video Library (https://images-api.nasa.gov/).
    Free, public domain, and requires no API key.
    Provides verified, real astronomical and spacecraft photography for Space/Astronomy topics.
    """
    BASE_URL = "https://images-api.nasa.gov/search"

    def search_image(self, query: str) -> Optional[str]:
        """
        Searches NASA Image & Video Archive and returns the highest resolution JPEG URL.
        Returns None if no relevant image is found.
        """
        if not query or not query.strip():
            return None
        # Clean query: strip non-alphanumeric noise
        clean_q = re.sub(r"[^\w\s-]", " ", query).strip()
        clean_q = re.sub(r"\s+", " ", clean_q)
        if not clean_q:
            return None

        try:
            params = {
                "q": clean_q[:60],
                "media_type": "image"
            }
            resp = requests.get(self.BASE_URL, params=params, headers={"User-Agent": "CSVG-Pipeline/2.0"}, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("collection", {}).get("items", [])
                if items:
                    # Look for high-res direct image links
                    for item in items[:3]:
                        links = item.get("links", [])
                        for link in links:
                            href = link.get("href", "")
                            if href and (href.endswith(".jpg") or href.endswith(".png") or link.get("render") == "image"):
                                return href
        except Exception:
            pass
        return None


nasa_retriever = NASARetriever()
