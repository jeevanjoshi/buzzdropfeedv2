import feedparser
import re
from typing import List, Dict, Any


GLOBAL_FINANCE_TECH_FEEDS = [
    "https://search.cnbc.com/rs/search/combinedrender?source=yahoo&partnerId=2001&type=news&id=10000664",
    "https://feeds.feedburner.com/TechCrunch/",
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
]

INDIA_ENGLISH_FINANCE_TECH_FEEDS = [
    "https://economictimes.indiatimes.com/rssfeedstopstories.cms",      # Economic Times Top Stories
    "https://www.livemint.com/rss/news",                               # Livemint Business & Markets
    "https://www.moneycontrol.com/rss/MCtopnews.xml",                   # Moneycontrol Markets
    "https://www.business-standard.com/rss/latest-news-155.rss",        # Business Standard News
    "https://yourstory.com/feed",                                      # YourStory (Indian Startups)
    "https://inc42.com/feed/",                                         # Inc42 (Indian Unicorns & Tech)
]


def extract_keywords(text: str) -> List[str]:
    """
    Extracts key entity words (length > 3) from text.
    """
    words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
    stopwords = {
        "with", "this", "that", "from", "they", "have", "been", "will", "more", "about", 
        "their", "which", "over", "after", "india", "indian", "news", "today", "report"
    }
    keywords = [w for w in set(words) if w not in stopwords]
    return keywords[:10]


def fetch_live_rss_feeds(
    region: str = "all", feed_urls: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Ingests live RSS news feeds from Global or India-specific English sources,
    parsing headlines, summaries, URLs, and extracting keywords.
    
    Regions supported: "global", "india", "all"
    """
    if not feed_urls:
        if region == "india":
            feed_urls = INDIA_ENGLISH_FINANCE_TECH_FEEDS
        elif region == "global":
            feed_urls = GLOBAL_FINANCE_TECH_FEEDS
        else:
            feed_urls = GLOBAL_FINANCE_TECH_FEEDS + INDIA_ENGLISH_FINANCE_TECH_FEEDS

    parsed_items = []
    
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:  # Top 5 stories per feed
                headline = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                link = entry.get("link", url)

                # Clean HTML tags from summary if present
                clean_summary = re.sub(r'<[^>]+>', '', summary)

                if headline and len(headline) > 10:
                    keywords = extract_keywords(f"{headline} {clean_summary}")
                    
                    parsed_items.append({
                        "headline": headline,
                        "summary": clean_summary[:300],
                        "url": link,
                        "keywords": keywords,
                        # Baseline 7-day search trend trajectory for newly discovered news
                        "search_history": [30.0, 38.0, 45.0, 60.0, 75.0, 90.0, 100.0],
                        "sentiment_variance": 0.85,
                        "competing_video_count": 2
                    })
        except Exception:
            continue

    return parsed_items
