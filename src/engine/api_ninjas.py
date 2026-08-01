import os
import requests
from typing import List, Dict, Any, Optional
from src.schemas.state import VerifiedFact


class APINinjasRetriever:
    """
    API Ninjas Data Retriever enriching RAG Ground-Truth facts with real-time financial,
    market index, and inflation data (https://api-ninjas.com).
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("API_NINJAS_KEY")
        self.base_url = "https://api.api-ninjas.com/v1"

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 5)

    def fetch_market_news(self, category: str = "business", limit: int = 5) -> List[VerifiedFact]:
        """
        Fetches breaking business & tech news facts from API Ninjas News endpoint.
        """
        if not self.is_available():
            return []

        headers = {"X-Api-Key": self.api_key}
        url = f"{self.base_url}/news?category={category}"

        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                articles = res.json()
                facts = []
                for idx, item in enumerate(articles[:limit]):
                    facts.append(VerifiedFact(
                        source_id=f"ninjas-news-{idx+1}",
                        headline=item.get("title", "Market Update"),
                        summary=item.get("description", item.get("text", "")[:200]),
                        url=item.get("url", "https://api-ninjas.com")
                    ))
                return facts
        except Exception:
            pass

        return []

    def fetch_interesting_facts(self, limit: int = 3) -> List[str]:
        """
        Fetches general trivia facts from API Ninjas /v1/facts endpoint.
        """
        if not self.is_available():
            return []

        headers = {"X-Api-Key": self.api_key}
        url = f"{self.base_url}/facts?limit={limit}"

        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                return [item.get("fact", "") for item in res.json() if "fact" in item]
        except Exception:
            pass

        return []
