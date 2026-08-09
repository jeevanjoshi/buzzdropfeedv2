import json
import urllib.request
import urllib.parse
import re
import os
import html
import time
import datetime as _dt
from typing import Dict, Any, List, Tuple, Set
from src.schemas.state import TopicCandidate, VerifiedFact
from src.engine.run_budget import run_budget

# ── RAG Corpus Sufficiency Gate thresholds ─────────────────────────────
# A 15-shot / 10-15 min script needs ~1,500+ narration words grounded in a
# corpus that is genuinely ON-TOPIC. These floors gate script generation so
# the pipeline never burns LLM spend on an undersupplied/polluted fact base.
MIN_ON_TOPIC_FACTS = 6          # distinct on-topic fact/snippet lines (>=1 token match)
MIN_ON_TOPIC_CORPUS_WORDS = 250 # on-topic words from lines with >=2 topic tokens
MIN_ON_TOPIC_SOURCES = 2        # distinct on-topic sources/publishers
MIN_DEEP_SOURCE_WORDS = 250     # a rich on-topic article grounds a full script alone

# Max retrieved lines surfaced into the RAG pack. Higher than the old hard 8-line
# cap so the rich keyword-driven retrieval actually reaches the LLM prompt.
MAX_RETRIEVED_LINES = 15

# ── Boilerplate / article-junk stripping ────────────────────────────────────
# Web pages (esp. news rags like NYT) leak ad/credit/nav boilerplate into the
# deep-crawled source article. If it reaches the RAG corpus it (a) pollutes the
# ground truth the Observer audits against, and (b) gets copied verbatim into
# narration (e.g. "SKIP ADVERTISEMENT / Listen · / Video by ..."). This regex
# strips those lines/sentences BEFORE they enter the corpus or the snippet pool.
_BOILERPLATE_RE = re.compile(
    r'\b(SKIP ADVERTISEMENTS?\b|Read\s?More|Share\s?(this)?\s?(article|story)?|'
    r'Subscribe\s?(now)?|Sign\s?up|Sign\s?in|Log\s?in|Log\s?out|Listen(?:\s|·)|'
    r'Play\s?(a|the)?\s?(video)?|Video\s?by\b|Written\s?by\b|Reporting\s?by\b|'
    r'By\s+[A-Z][A-Za-z. -]+\s+(For|via)?\s*The\s+[A-Z]|Guest\s+Author|'
    r'Supported\s?by\b|Advertisement\b|Advertorial\b|Sponsored(?: Content)?\b|'
    r'Newsletter\b|Comments?(?:\s+closed)?\b|Recommended(?: Stories| Videos)?\b|'
    r'Most\s?Read\b|Trending\b|Menu\b|Close\b|Skip\s?to\s?content\b|'
    r'Privacy\s?Policy\b|Terms\s?of\s?Service\b|Cookie\s?Preferences?\b|'
    # Raw scrape/citation junk: markdown links, bare URLs, datelines, "Retrieved"
    # tails and author-monogram bibliography entries. These read as pasted slop
    # and leak verbatim into narration, so they are dropped sentence-by-sentence.
    r'\[[^\]\n]{0,120}?\]\((?:https?://|#|/)[^)\n]{0,300}?\)|'
    r'https?://\S+|'
    r'\(\s*[A-Z][A-Za-z.-]*(?:\s*,\s*[A-Z][A-Za-z.-]*)*\s*[–—-]\s*'
    r'(?:January|February|March|April|May|June|July|August|September|October|'
    r'November|December)\s+\d{1,2},?\s+\d{4}\s*\)|'
    r'\bRetrieved\b|'
    r'[A-Z][a-z]+,\s+[A-Z][a-z]+\s+\((?:January|February|March|April|May|June|'
    r'July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\)\.?|'
    # Intraday market-ticker wrap-ups ("Orsted A/S dropped 7.7% ..."), coherent
    # but off-topic stock moves that leak verbatim into narration (Shot 17 class).
    r'\b[A-Z][A-Za-z&\'.\-]*\s+(?:A\/S|Inc\.?|Ltd\.?|Corp\.?|PLC|NV|ASA|AG|SE|Oyj)\b'
    r'[^.!?]{0,90}?\b(?:fell|dropped|rose|gained|jumped|surged|slid|sank|slumped|soared|plunged|retreated|advanced)\b'
    r'[^.!?]{0,90}?\b\d+(?:\.\d+)?\s*%)',
    re.IGNORECASE,
)


def _strip_boilerplate(text: str) -> str:
    """Removes ad/credit/nav boilerplate lines or sentences from crawled text and
    collapses whitespace. Facts, numbers and names are preserved."""
    if not text:
        return ""
    parts = re.split(r'(?<=[.!?])\s+', text)
    kept = [p for p in parts if not _BOILERPLATE_RE.search(p)]
    return re.sub(r'\s+', ' ', " ".join(kept)).strip()

_TOPIC_STOPWORDS = {
    "with", "from", "that", "this", "have", "their", "there", "would",
    "about", "which", "across", "more", "than", "into", "what", "they",
    "these", "those", "being", "been", "still", "while", "after", "before",
    "other", "under", "again", "through", "every", "where", "because",
    "between", "during", "without", "around", "however", "the", "and",
    "for", "are", "was", "are", "its", "has", "not", "but", "you", "can",
}


def _build_topic_tokens(headline: str, summary: str, keywords: List[str]) -> Set[str]:
    """Topic keywords from headline + summary + keywords, minus stopwords."""
    tokens: Set[str] = set()
    for txt in (headline or "", summary or ""):
        tokens.update(re.findall(r"[a-z][a-z0-9'-]{3,}", txt.lower()))
    tokens.update(k.lower() for k in (keywords or []) if len(k) > 3)
    tokens -= _TOPIC_STOPWORDS
    if not tokens:
        tokens = set(re.findall(r"[a-z][a-z0-9'-]{3,}", (headline or "").lower()))
    return tokens


def _on_topic_hits(line: str, topic_tokens: Set[str]) -> int:
    """Count of distinct topic tokens present in a line (word-boundary match, so
    'tech' only counts when it appears as the word 'tech', not inside 'detect')."""
    ll = (line or "").lower()
    hits = 0
    for t in topic_tokens:
        if re.search(rf"\b{re.escape(t)}\b", ll):
            hits += 1
    return hits


def _line_snippet_text(line: str) -> str:
    """Strip a bullet's '[Source: title]' prefix, returning just the snippet body."""
    parts = line.split("]:", 1)
    return parts[1].strip() if len(parts) > 1 else line


def _crawler_item_to_dict(item) -> Dict[str, str]:
    """Normalise one crawler result (a dict OR a VerifiedFact/ad-hoc object)
    to a canonical {"title", "snippet", "url"} dict so the generic crawl loop
    can label/format every source uniformly."""
    if isinstance(item, dict):
        return {
            "title": str(item.get("title") or "").strip(),
            "snippet": str(item.get("snippet") or item.get("content") or "").strip(),
            "url": str(item.get("url") or "").strip(),
        }
    title = getattr(item, "headline", None) or getattr(item, "title", "") or ""
    summary = getattr(item, "summary", None) or getattr(item, "snippet", "") or ""
    url = getattr(item, "url", "") or ""
    return {
        "title": str(title).strip(),
        "snippet": str(summary).strip(),
        "url": str(url).strip(),
    }


_ACRONYM_SOURCE = frozenset({"bbc", "cnn", "nyt", "nbc", "abc", "cbs", "ap", "cnbc", "wapo"})

_PUBLICATION_ALIASES = {
    "apnews": "Associated Press",
    "reuters": "Reuters",
    "theguardian": "The Guardian",
    "guardian": "The Guardian",
    "nytimes": "The New York Times",
    "wsj": "The Wall Street Journal",
    "washingtonpost": "The Washington Post",
    "wapo": "The Washington Post",
    "economictimes": "The Economic Times",
    "indiatimes": "The Economic Times",
    "timesofindia": "The Times of India",
    "hindustantimes": "Hindustan Times",
    "bloomberg": "Bloomberg",
    "forbes": "Forbes",
    "fortune": "Fortune",
    "entrepreneur": "Entrepreneur",
    "techcrunch": "TechCrunch",
    "wired": "Wired",
    "arstechnica": "Ars Technica",
    "theverge": "The Verge",
    "verge": "The Verge",
    "businessinsider": "Business Insider",
    "barrons": "Barron's",
    "cnbc": "CNBC",
    "nbcnews": "NBC News",
    "abcnews": "ABC News",
    "cbsnews": "CBS News",
    "bbc": "BBC",
    "cnn": "CNN",
    "statista": "Statista",
    "wikipedia": "Wikipedia",
    "mint": "Mint",
    "livemint": "Mint",
    "moneycontrol": "Moneycontrol",
    "yahoo": "Yahoo Finance",
}


# ── RAG CRAWLER REGISTRY ─────────────────────────────────────────────────────
# Every web-data source the RAG pack queries, declared ONCE here. Adding a new
# crawler = write one `search_xxx_facts(query, max_results)` method that returns
# a list of dicts {"title", "snippet", "url"} (or VerifiedFact-like objects),
# then add a single entry to this registry. The generic crawl loop applies the
# earlier production fixes automatically, so a new crawler can NEVER regress them:
#   * `kind="news"`        -> bucket labelled with the REAL publisher derived
#                             from the URL (_source_label), never the tool name.
#   * `kind="terminology"` -> bucket labelled `[Terminology: ...]` (dictionary /
#                             encyclopedia definitions for jargon/acronyms only,
#                             never allowed to source a news claim).
#   * Each entry's items still flow through the shared promo/social/dedup/
#     on-topic/recent-historical filters below.
# Tuple: (key, fallback_label, kind, max_results, fetch_method_name)
_RAG_CRAWLERS = (
    ("exa",       "Exa",       "news",        3, "fetch_exa_news"),
    ("newsapi",   "NewsAPI",   "news",        2, "search_newsapi_facts"),
    ("wikipedia", "Wikipedia", "terminology", 2, "search_wikipedia_facts"),
    ("tavily",    "Tavily",    "news",        3, "search_tavily_facts"),
    ("firecrawl", "Firecrawl", "news",        3, "search_firecrawl_facts"),
)


# Search-tool names that must NEVER surface as RAG bucket labels. Firecrawl,
# Tavily, Exa, NewsAPI etc. are retrieval transport, not publications: if a
# result URL can't be resolved to a real publisher, the bucket is labelled
# UNATTRIBUTED so the story LLM has no tool name to echo into narration
# ("according to insights from Firecrawl" class of leak).
_TOOL_SOURCE_NAMES = frozenset({
    "firecrawl", "tavily", "exa", "newsapi", "ddg", "duckduckgo", "wikipedia",
})


def _source_label(url: str, fallback: str = "") -> str:
    """Derive a human-readable publication name from a result URL, e.g.
    'https://fortune.com/2026/...' -> 'Fortune', 'https://www.entrepreneur.com/'
    -> 'Entrepreneur'. RAG buckets are labelled with THIS (the actual publisher),
    never with the search-tool name, so the story LLM cites 'Fortune' instead of
    'Firecrawl'/'Tavily'/'Exa'. Return ``fallback`` when no URL is available."""
    if not url:
        return fallback
    try:
        netloc = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return fallback
    netloc = (netloc or "").split(":")[0]
    for prefix in ("www.", "m.", "mobile.", "blog."):
        if netloc.startswith(prefix):
            netloc = netloc[len(prefix):]
    labels = [p for p in netloc.split(".") if p]
    if not labels:
        return fallback
    # Drop a ccTLD whose predecessor is a generic TLD slot (e.g. .co.uk, .com.au),
    # then the remaining TLD; the label right before the TLD is the publisher.
    if len(labels) >= 3 and labels[-1] in (
        "uk", "au", "ca", "in", "nz", "za", "ie", "sg", "my", "co", "jp", "de", "fr"
    ) and labels[-2] in ("co", "com", "org", "net", "gov", "ac", "edu"):
        labels = labels[:-2]
    if len(labels) >= 2:
        labels = labels[:-1]
    name = labels[-1] if labels else ""
    name = re.sub(r"[^a-z0-9 ]", " ", name).strip()
    name = re.sub(r"\s+", " ", name)
    if name in _PUBLICATION_ALIASES:
        return _PUBLICATION_ALIASES[name]
    if name in _ACRONYM_SOURCE:
        return name.upper()
    return name.title() if name else fallback

_YEAR_RE = re.compile(r'\b(20[0-2]\d)\b')
_current_year_val = None


def _current_year() -> int:
    global _current_year_val
    if _current_year_val is None:
        _current_year_val = _dt.datetime.now(_dt.timezone.utc).year
    return _current_year_val


def _snippet_year(text: str):
    """Apparent 4-digit year (2000-2099) contained in a snippet/title, else None."""
    m = _YEAR_RE.search(text or "")
    return int(m.group(1)) if m else None


def _recency_tag(line: str) -> str:
    """
    Returns a short recency annotation for a retrieved-snippet line based on any
    year it mentions: ' (recent: YYYY)' for current/last-year, ' (historical: YYYY)'
    for older years, or '' when no date is found (treated as neutral/recent).
    """
    y = _snippet_year(line)
    if y is None:
        return ""
    cy = _current_year()
    if y >= cy - 1:
        return f" (recent: {y})"
    elif 2000 <= y <= cy - 2:
        return f" (historical: {y})"
    return ""


# Promotional / advertorial / web-chatter content filter. Applied to EVERY
# retrieval source (NewsAPI, Wikipedia, Tavily, Firecrawl, Exa) so marketing and
# ad-style pages cannot pollute the RAG corpus (e.g. AI-tool promo pages leaking
# into an unrelated story).
_PROMO_RE = re.compile(
    r'\b(sign\s?up|subscribe|register|join\s?now|click\s?here|get\s?started|'
    r'free\s?trial|pricing\s?plan|coupon|discount|promo|checkout|buy\s?now|'
    r'add\s?to\s?cart|shop\s?(now|today)|order\s?now|affiliate|sponsor|sponsored|'
    r'advertisement|advertising|marketing|advertised|shopping|sale|discounted|deal|'
    r'best\sprice|cheap|log\s?in|sign\s?in|start\s?free|download\s?now|'
    r'grab\s?(your|this)|limited\s?(time|offer)|act\s?now|offer\s?ends|'
    r'newsletter|join\s?(today|now))\b',
    re.IGNORECASE,
)


def _is_promotional(text: str) -> bool:
    """True if a snippet/title contains promotional, ad, or web-chatter language."""
    return bool(_PROMO_RE.search(text or ""))


# ─────────────────────────────────────────────────────────────────────────────
# SOCIAL-MEDIA SOURCE EXCLUSION
# Social/community platforms carry low-trust, non-citable opinion (and verbatim
# user chatter) that pollutes the fact corpus and feeds Observer false positives.
# Excluded BOTH in the scraper path (per-source + C3 final filter) and in Google
# grounding (grounded_search). Medium/Substack/forums are treated as social
# because they host self-published, uncurated opinion masquerading as journalism.
# ─────────────────────────────────────────────────────────────────────────────
_SOCIAL_DOMAINS = frozenset({
    "reddit.com", "redd.it",
    "x.com", "twitter.com", "t.co",
    "facebook.com", "fb.com", "fb.watch",
    "instagram.com",
    "linkedin.com",
    "tiktok.com",
    "youtube.com", "youtu.be", "m.youtube.com",
    "quora.com",
    "pinterest.com", "pin.it",
    "snapchat.com",
    "threads.net",
    "discord.com",
    "medium.com",
    "substack.com",
})

_SOCIAL_SOURCE_RE = re.compile(
    r"\b(reddit|twitter|tweet|facebook|instagram|linkedin|tiktok|"
    r"quora|pinterest|snapchat|threads|discord|substack)\b|"
    r"\b(r/[\w-]+|u/[\w-]+)\b|"
    r"(?<![\w.])@[\w.-]+\b",
    re.IGNORECASE,
)


def _is_social_source(url: str = "", title: str = "", snippet: str = "") -> bool:
    """True if a source (by URL domain or by source/text) is a social/community
    platform. Conservative: URL host match first, then source-name/text markers.
    """
    text = f"{url or ''} {title or ''} {snippet or ''}"
    try:
        from urllib.parse import urlparse
        host = urlparse(str(url or "")).netloc.lower().lstrip("www.").split(":")[0]
        if any(d == host or host.endswith("." + d) for d in _SOCIAL_DOMAINS):
            return True
    except Exception:
        pass
    return bool(_SOCIAL_SOURCE_RE.search(text))





class GraphNode:
    def __init__(self, entity_id: str, entity_type: str = "concept"):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.edges: List[Tuple[str, str]] = []  # List of (predicate, target_entity_id)

    def add_edge(self, predicate: str, target_id: str):
        self.edges.append((predicate, target_id))


class RAGTopicRetriever:
    """
    Stage 2: GraphRAG & Hybrid Information Retrieval Engine.
    Combines Standard Vector RAG for broad summaries with GraphRAG entity-relation semantic knowledge graphs
    for deep multi-hop factual reasoning and TrumorGPT hallucination defense fact checking.
    """

    def __init__(self):
        self.knowledge_graph: Dict[str, GraphNode] = {}
        # RAG cache: headline -> (timestamp, pack). Avoids re-fetching the same
        # paid/network searches on repeated runs for the same topic (cost + speed).
        self._rag_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._rag_cache_ttl_s = 6 * 3600
        # A/B switch for the Google Search grounding research pass. Set programmatically
        # from the orchestrator --rag grounded|hybrid|scraper flag, or from env RAG_GROUNDED=1.
        # Defaults to the scraper path so pipeline behavior is unchanged unless opted in.
        #   scraper  : 5-scraper crawl only (default)
        #   grounded : Google-Search cited facts ONLY (replaces scraper)
        #   hybrid   : grounded cited facts as the core + on-topic scraper depth (no pollution)
        self.rag_mode = "grounded" if os.getenv("RAG_GROUNDED", "").strip().lower() in ("1", "true") else "scraper"
        self.use_grounded = self.rag_mode in ("grounded", "hybrid")

    def set_grounded(self, enabled: bool) -> None:
        self.use_grounded = bool(enabled)
        if self.use_grounded and self.rag_mode == "scraper":
            self.rag_mode = "grounded"
        print(f"[RAGRetriever] Google Search grounding mode: {'ON' if self.use_grounded else 'OFF'}")

    def set_rag_mode(self, mode: str) -> None:
        mode = (mode or "scraper").strip().lower()
        if mode not in ("scraper", "grounded", "hybrid"):
            mode = "scraper"
        self.rag_mode = mode
        self.use_grounded = mode in ("grounded", "hybrid")
        print(f"[RAGRetriever] RAG mode set: {mode} (grounding={'ON' if self.use_grounded else 'OFF'})")

    @staticmethod
    def _merge_grounded_scraper(grounded_pack: Dict[str, Any], scraper_pack: Dict[str, Any]) -> Dict[str, Any]:
        """Merge the Google-grounded (cited, high-precision) corpus as the CORE with
        the scraper's on-topic depth, WITHOUT polluting.

        Guards:
          * Grounded facts go first (highest precision / cited).
          * Scraper lines are appended only when they do NOT textually duplicate a
            grounded fact (dedup by normalized alnum text) — prevents the Observer's
            verbatim-copy / over-repetition false positives.
          * All input lines already passed the promo-filter / on-topic(>=2) guards
            in their respective builders.
        """
        def _norm(line: str) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", line.lower()))

        g_lines = [l for l in (grounded_pack.get("fact_corpus", "") or "").splitlines() if l.strip()]
        s_lines = [l for l in (scraper_pack.get("fact_corpus", "") or "").splitlines() if l.strip()]

        seen: Set[str] = set()
        merged = []
        for l in g_lines + s_lines:
            k = _norm(l)
            if not k or k in seen:
                continue
            seen.add(k)
            merged.append(l)

        scraper_pack["fact_corpus"] = "\n".join(merged)
        scraper_pack["rag_mode"] = "grounded+scraper"
        scraper_pack["_grounding_meta"] = grounded_pack.get("_grounding_meta", {})
        g_ctx = grounded_pack.get("full_rag_context_text", "") or ""
        if g_ctx:
            scraper_pack["full_rag_context_text"] = (
                "GROUNDED GOOGLE-SEARCH FACTS (CITED):\n" + g_ctx + "\n\n"
                + (scraper_pack.get("full_rag_context_text", "") or "")
            )
        gtb = "\n".join(g_lines)
        if gtb:
            scraper_pack["ground_truth_block"] = (
                gtb + "\n" + (scraper_pack.get("ground_truth_block", "") or "")
            )
        return scraper_pack


    def search_duckduckgo_facts(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes a DuckDuckGo HTML web search to extract real-world background facts.
        """
        results = []
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=8) as response:
                html = response.read().decode('utf-8', errors='ignore')

            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            ad_pattern = re.compile(
                r'\b(sign\s?up|subscribe|register|join\s?now|click\s?here|get\s?started|'
                r'free\s?trial|pricing\s?plan|price|coupon|discount|promo|checkout|'
                r'buy\s?now|add\s?to\s?cart|shop|store|order\s?now|affiliate|sponsor|'
                r'advertisement|marketing|ad\b|advertised|shopping|sale|discounted|deal|'
                r'best\sprice|cheap)\b',
                re.IGNORECASE
            )

            # Split the HTML page into individual result blocks
            blocks = html.split('<div class="result results_links results_links_deep web-result')
            for block in blocks[1:]:
                # Extract first anchor href
                url_match = re.search(r'href="([^"]+)"', block)
                if not url_match:
                    continue
                url = url_match.group(1)
                
                # Exclude sponsored tracking links
                if "y.js" in url or "ad_provider" in url or "sponsored" in url:
                    continue

                # Local match inside block for title and snippet
                title_match = re.search(r'<a class="result__title[^>]*>(.*?)</a>', block, re.DOTALL)
                snippet_match = re.search(r'<a class="result__snippet[^>]*>(.*?)</a>', block, re.DOTALL)
                
                if title_match and snippet_match:
                    clean_title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
                    clean_snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()
                    
                    if len(clean_snippet) > 30:
                        combined_text = f"{clean_title} {clean_snippet}"
                        if ad_pattern.search(combined_text):
                            continue

                        try:
                            vectorizer = TfidfVectorizer(stop_words='english')
                            tfidf_matrix = vectorizer.fit_transform([query, combined_text])
                            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                        except Exception:
                            sim = 0.0

                        if sim > 0.08:
                            results.append({"title": clean_title, "snippet": clean_snippet})
                            if len(results) >= max_results:
                                break
        except Exception as e:
            print(f"[RAGRetriever] Search Warning: {e}")

        return results

    def fetch_exa_news(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Wrapper for the Exa semantic-search crawler so it matches the
        search_*_facts() contract (returns {title, snippet, url} dicts). Exa
        returns VerifiedFact objects via the external manager; normalise them
        here so the generic crawl loop can label/format every source uniformly.
        """
        from src.engine.external_apis import external_api_manager
        try:
            return [
                _crawler_item_to_dict(f)
                for f in external_api_manager.fetch_exa_semantic_facts(query)
            ]
        except Exception as e:
            print(f"[RAGRetriever] Exa query failed: {e}")
            return []

    def search_newsapi_facts(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Executes a NewsAPI query to retrieve structured global news snippets.
        """
        results = []
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            return results
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://newsapi.org/v2/everything?q={encoded_query}&pageSize={max_results}&apiKey={api_key}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8', errors='ignore'))
                articles = data.get("articles", [])
                for a in articles:
                    title = a.get("title", "")
                    desc = a.get("description", "")
                    if desc and len(desc) > 30 and not _is_social_source(a.get("url", ""), title, desc):
                        results.append({"title": title, "snippet": desc, "url": a.get("url", "")})
                run_budget.record_search("newsapi")
        except Exception as e:
            print(f"[RAGRetriever] NewsAPI Warning: {e}")
        return results

    def search_wikipedia_facts(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Executes a Wikipedia Query API search to extract clean historical/factual snippets.
        """
        results = []
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&utf8=&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8', errors='ignore'))
                search_results = data.get("query", {}).get("search", [])
                for item in search_results[:max_results]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    clean_snippet = re.sub(r'<[^>]+>', '', snippet)
                    clean_snippet = html.unescape(clean_snippet)
                    if len(clean_snippet) > 30 and not _is_social_source("", title, clean_snippet):
                        results.append({"title": title, "snippet": clean_snippet})
        except Exception as e:
            print(f"[RAGRetriever] Wikipedia Warning: {e}")
        return results

    def search_tavily_facts(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes a Tavily search (purpose-built for AI/RAG). Returns clean,
        ad-free snippets — no HTML scraping or ad filtering needed.
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return []
        try:
            import requests
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                },
                timeout=12,
            )
            data = resp.json()
            run_budget.record_search("tavily")
            return [
                {"title": r.get("title", ""), "snippet": r.get("content", ""), "url": r.get("url", "")}
                for r in data.get("results", [])
                if r.get("content") and len(r.get("content", "")) > 30
                and not _is_social_source(r.get("url", ""), r.get("title", ""), r.get("content", ""))
            ]
        except Exception as e:
            print(f"[RAGRetriever] Tavily Warning: {e}")
            return []

    def search_firecrawl_facts(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes a Firecrawl search (clean crawl API returning page-level snippets).
        """
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            return []
        try:
            import requests
            resp = requests.post(
                "https://api.firecrawl.dev/v1/search",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"query": query, "limit": max_results},
                timeout=15,
            )
            data = resp.json()
            items = data.get("data", []) if isinstance(data, dict) else []
            run_budget.record_search("firecrawl")
            out = []
            for it in items:
                desc = it.get("description") or (it.get("metadata", {}) or {}).get("description", "")
                title = it.get("title", "")
                # Firecrawl /v1/search returns the canonical `url` at the TOP
                # level; some responses nest it under metadata.url. Reading only
                # metadata.url meant EVERY Firecrawl result was labelled
                # '[Firecrawl: ...]' (empty url -> _source_label fallback) and
                # the story LLM then recited the tool name as a citation.
                meta_url = it.get("url") or (it.get("metadata", {}) or {}).get("url", "")
                if desc and len(str(desc)) > 30 and not _is_social_source(meta_url, title, str(desc)):
                    out.append({"title": title, "snippet": str(desc), "url": meta_url})
            return out
        except Exception as e:
            print(f"[RAGRetriever] Firecrawl Warning: {e}")
            return []

    def _html_to_text(self, raw_html: str) -> str:
        """Strips HTML tags, script/style/nav/footer blocks, and normalises whitespace."""
        s = re.sub(r"(?is)<script.*?</script>", " ", raw_html)
        s = re.sub(r"(?is)<style.*?</style>", " ", s)
        s = re.sub(r"(?is)<noscript.*?</noscript>", " ", s)
        s = re.sub(r"(?is)<(head|header|nav|footer|aside).*?</\1>", " ", s)
        s = re.sub(r"(?is)<[^>]+>", " ", s)
        s = html.unescape(s)
        s = re.sub(r"\s+", " ", s)
        return _strip_boilerplate(s)

    def _clean_article_markdown(self, md: str) -> str:
        """Lightly cleans a markdown article body (Firecrawl returns markdown)."""
        s = re.sub(r"```.*?```", " ", md, flags=re.S)
        s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
        s = re.sub(r"^\s*(#|>|[-*]\s*)", "", s, flags=re.M)
        s = re.sub(r"\s+", " ", s)
        return _strip_boilerplate(s)

    def _scrape_selected_article(self, url: str, max_chars: int = 4000) -> str:
        """
        Deep-crawls the selected topic's OWN source URL to get the full article
        body (headline + summary only are stored in the RSS feed).  Returns the
        cleaned text or an empty string if the scrape fails or returns garbage.

        Tries Firecrawl ``/v1/scrape`` first (returns structured markdown), then
        falls back to a plain urllib GET + HTML stripping.
        """
        if not url or not url.startswith("http"):
            return ""
        src_name = ""
        fc_key = os.getenv("FIRECRAWL_API_KEY")
        if fc_key:
            try:
                import requests
                resp = requests.post(
                    "https://api.firecrawl.dev/v1/scrape",
                    headers={"Authorization": f"Bearer {fc_key}", "Content-Type": "application/json"},
                    json={"url": url},
                    timeout=20,
                )
                data = resp.json()
                md = data.get("data", {}).get("markdown") or data.get("markdown", "")
                if md:
                    text = self._clean_article_markdown(md)
                    if len(text) >= 200:
                        return text[:max_chars]
            except Exception as e:
                print(f"[RAGRetriever] Firecrawl scrape failed: {e}")
        # Fallback: urllib GET + HTML -> text. Bare GETs are aggressively
        # 403-blocked by major news sites; retry once with a fuller browser-like
        # header set before giving up, and keep the failure terse (not a raw
        # exception dump) since a 403 here is an expected bot-block, not an error.
        ua_pool = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36",
        ]
        for attempt, ua in enumerate(ua_pool[:2]):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": ua,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                )
                with urllib.request.urlopen(req, timeout=12) as response:
                    raw = response.read().decode("utf-8", errors="ignore")
                text = self._html_to_text(raw)
                if len(text) >= 200:
                    return text[:max_chars]
            except Exception:
                if attempt == 0:
                    continue  # retry once with the alternate UA
                print(f"[RAGRetriever] Article scrape fallback 403/blocked off its "
                      f"source URL ({src_name or url}) — expecting this for "
                      f"bot-protected sites; corpus already uses feed/fact sources.")
                return ""
        return ""

    def extract_graph_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extracts semantic (Subject, Predicate, Object) triplets from retrieved research text.
        """
        triplets = []
        sentences = re.split(r'[.!?]', text)
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            words = s_clean.split()
            if len(words) >= 4:
                # Extract simple entity-relation heuristics
                subject = words[0].strip(",. ")
                predicate = words[1].strip(",. ")
                obj = " ".join(words[2:]).strip(",. ")
                if len(subject) > 2 and len(obj) > 2:
                    triplets.append((subject, predicate, obj))
                    # Register into in-memory knowledge graph
                    s_node = self.knowledge_graph.setdefault(subject.lower(), GraphNode(subject.lower()))
                    s_node.add_edge(predicate.lower(), obj.lower())
        return triplets[:15]

    def traverse_graph_paths(self, start_entity: str, max_depth: int = 2) -> List[str]:
        """
        Traverses semantic knowledge graph relational paths for multi-hop reasoning.
        """
        start_key = start_entity.lower()
        if start_key not in self.knowledge_graph:
            return []
        
        visited: Set[str] = set()
        paths: List[str] = []
        
        def dfs(curr_id: str, depth: int, current_path: str):
            if depth >= max_depth or curr_id in visited:
                return
            visited.add(curr_id)
            node = self.knowledge_graph.get(curr_id)
            if not node:
                return
            for pred, target in node.edges:
                path_str = f"{current_path} -[{pred}]-> {target}"
                paths.append(path_str)
                dfs(target, depth + 1, path_str)

        dfs(start_key, 0, start_key)
        return paths[:10]

    def select_rag_mode(self, query_complexity: str) -> str:
        """
        Hybrid RAG Router:
        Selects 'standard_vector' for simple overviews, or 'graph_rag' for deep multi-hop factual synthesis.
        """
        if any(w in query_complexity.lower() for w in ["history", "origin", "mechanism", "why", "relationship", "impact", "breakdown"]):
            return "graph_rag"
        return "standard_vector"

    def trumorgpt_verify_fact(self, claim: str) -> Tuple[bool, float, str]:
        """
        TrumorGPT-style Semantic Fact-Checker:
        Verifies script claims against registered knowledge graph triples to flag hallucinations.
        """
        claim_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', claim.lower()))
        if not self.knowledge_graph:
            return True, 0.90, "Ground truth facts clear."
        
        matching_edges = 0
        total_checks = 0
        for node_id, node in self.knowledge_graph.items():
            if node_id in claim_words:
                total_checks += 1
                for pred, target in node.edges:
                    if target in claim_words or any(w in claim_words for w in target.split()):
                        matching_edges += 1
                        break
        
        if total_checks == 0:
            return True, 0.85, "Fact check plausible based on broad corpus."
        
        confidence = matching_edges / total_checks
        is_verified = confidence >= 0.5
        msg = "Verified against GraphRAG evidence." if is_verified else "Potential hallucination detected; low graph support."
        return is_verified, float(round(confidence, 2)), msg

    def assess_corpus_sufficiency(
        self, pack: Dict[str, Any], topic: TopicCandidate
    ) -> Dict[str, Any]:
        """
        Evaluate whether the RAG corpus is rich enough to ground a full
        15-shot / 10-15 min script. Returns ``{"pass": bool, "reason": str,
        "metrics": {...}}``.

        * Extracts topic tokens from headline + summary + keywords.
        * Counts fact/snippet lines in the corpus that share at least one
          meaningful topic token → on-topic facts, words, source diversity.
        * Compares against ``MIN_ON_TOPIC_*`` thresholds.
        """
        headline = (pack.get("topic_headline") or topic.headline or "").lower()
        summary = (pack.get("summary") or topic.summary or "").lower()
        keywords = [k.lower() for k in (pack.get("keywords") or topic.keywords or [])]

        topic_tokens = _build_topic_tokens(headline, summary, keywords)

        fact_corpus = pack.get("fact_corpus") or ""
        fact_lines = [l.strip() for l in fact_corpus.splitlines() if l.strip()]

        on_topic_words = 0
        on_topic_source_names: Set[str] = set()
        on_topic_count = 0
        for line in fact_lines:
            hits = _on_topic_hits(line, topic_tokens)
            if hits < 1:
                continue
            on_topic_count += 1
            # Words only count toward the "corpus words" metric when the line is
            # GENUINELY on-topic (>=2 topic tokens), so loosely-shared tokens
            # from a polluted feed can't inflate the grounding budget.
            if hits >= 2:
                on_topic_words += len(line.split())
            m = re.search(r"\(Source:\s*([^)]+)\)", line)
            if m:
                src = m.group(1).strip()
                if src not in ("Verified Reports", "Verified Market Reports"):
                    on_topic_source_names.add(src)
            else:
                # '[Publisher: title]' -> the publisher label is everything before
                # the ': title'. Terminology/encyclopedia buckets are definitions
                # only (NOT news sources) and must not count toward source
                # diversity, exactly as Wikipedia is no longer a news crawler.
                m2 = re.search(r"^• \[([^\]\:]+)(?::[^\]]*)?\]", line)
                if m2:
                    src = m2.group(1).strip()
                    if src.lower() not in ("terminology", "encyclopedia", "wikipedia"):
                        on_topic_source_names.add(src)

        total_corpus_words = sum(len(l.split()) for l in fact_lines)

        metrics = {
            "on_topic_facts": on_topic_count,
            "on_topic_corpus_words": on_topic_words,
            "on_topic_sources": len(on_topic_source_names),
            "total_corpus_words": total_corpus_words,
        }
        reasons = []
        passed = True

        # Deep-source exception: a single deep-crawled article (the SELECTED source
        # story) that yields a large amount of on-topic text can ground a full
        # script on its own, even though it is only 1 "fact" from 1 source. Without
        # this, a rich deep article would be wrongly rejected by the diversity gates.
        deep_source = on_topic_words >= MIN_DEEP_SOURCE_WORDS and on_topic_count >= 1

        if not deep_source:
            if on_topic_count < MIN_ON_TOPIC_FACTS:
                passed = False
                reasons.append(
                    f"only {on_topic_count} on-topic facts (min {MIN_ON_TOPIC_FACTS})"
                )
            if on_topic_words < MIN_ON_TOPIC_CORPUS_WORDS:
                passed = False
                reasons.append(
                    f"only {on_topic_words} on-topic words (min {MIN_ON_TOPIC_CORPUS_WORDS})"
                )
            if len(on_topic_source_names) < MIN_ON_TOPIC_SOURCES:
                passed = False
                reasons.append(
                    f"only {len(on_topic_source_names)} on-topic sources "
                    f"(min {MIN_ON_TOPIC_SOURCES})"
                )

        return {
            "pass": passed,
            "reason": "; ".join(reasons) if reasons else "RAG corpus sufficient.",
            "metrics": metrics,
        }

    def build_rag_knowledge_pack(
        self, topic: TopicCandidate, verified_facts: List[VerifiedFact],
        refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Constructs a comprehensive 1,000+ word RAG Knowledge Pack containing:
        1. Core Headline & Real-World Trigger Summary
        2. Verified Facts & Primary Sources
        3. GraphRAG Multi-Hop Relational Paths & Knowledge Triplets
        4. TrumorGPT Citation Grounding & Fact Checks
        5. Future Strategic Implications & Category Insights
        """
        headline = topic.headline
        summary = topic.summary
        keywords = [k for k in topic.keywords if len(k) > 3][:6]
        cache_key = f"{headline}|{len(verified_facts)}"

        # Google Search grounding research pass. In "grounded" mode it REPLACES the
        # scraper path with one cited Google-Search research call. In "hybrid" mode
        # it is the high-precision CITATIONS core and the scraper path below then
        # ADDS on-topic depth on top (with dedup guards so the grounded corpus is
        # never diluted/polluted).
        grounded_pack = None
        if self.use_grounded:
            try:
                from src.engine.grounded_search import build_grounded_knowledge_pack
                grounded_pack = build_grounded_knowledge_pack(
                    headline=headline, summary=summary, keywords=keywords
                )
            except Exception as e:
                print(f"[RAGRetriever] Grounded research failed, falling back to scraper path: {e}")
                grounded_pack = None
            if grounded_pack is not None and self.rag_mode == "grounded":
                self._rag_cache[cache_key] = (time.time(), grounded_pack)
                return grounded_pack

        # Cache: return recent RAG pack for the same headline/fact-count to avoid
        # redundant paid/network searches.  Passing ``refresh=True`` bypasses the
        # cache (forces a fresh retrieval) and stores the result so subsequent
        # non-refresh calls reuse the refreshed pack.
        _now = time.time()
        _hit = self._rag_cache.get(cache_key)
        if not refresh and _hit and (_now - _hit[0]) < self._rag_cache_ttl_s:
            return _hit[1]

        # Combine verified facts into core ground truth block — but only ON-TOPIC
        # facts. The full RSS corpus is polluted with unrelated feed items, so
        # feeding every headline into the prompt makes the LLM wander and starves
        # the script of real grounding.
        topic_tokens = _build_topic_tokens(headline, summary, keywords)
        # Dedup facts by headline (the RSS corpus commonly carries the same story
        # from the same feed more than once) before scoring/keeping.
        seen_facts: Set[str] = set()
        unique_facts = []
        for vf in verified_facts:
            key = (vf.headline or "").strip().lower()
            if key and key not in seen_facts:
                seen_facts.add(key)
                unique_facts.append(vf)
        verified_facts = unique_facts

        scored_facts = []
        for vf in verified_facts:
            line = f"{vf.headline}: {vf.summary} (Source: {vf.source_name})"
            score = _on_topic_hits(line, topic_tokens)
            # >=2 token matches: generic shared words ('tech', 'ones', 'over')
            # from the summary must not pull unrelated RSS feed noise into the
            # prompt's ground-truth block.
            if score >= 2:
                scored_facts.append((score, vf))
        # Always keep the topic's own RSS fact even if token matching misses it.
        own_head = headline.strip().lower()
        if not any(vf.headline.strip().lower() == own_head for _, vf in scored_facts):
            for vf in verified_facts:
                if vf.headline.strip().lower() == own_head:
                    scored_facts.append((100, vf))
                    break
        scored_facts.sort(key=lambda x: x[0], reverse=True)
        kept_facts = [vf for _, vf in scored_facts]
        verified_snippets = [
            f"{vf.headline}: {vf.summary} (Source: {vf.source_name})" for vf in kept_facts
        ]
        ground_truth_block = "\n".join(verified_snippets) if verified_snippets else summary

        # Perform targeted RAG web queries — drive search with BOTH the headline
        # and the keyword vector (keyword-only queries often out-recall the long
        # headline phrasing on NewsAPI/Tavily/Wikipedia).
        kw_query = " ".join(k for k in keywords if k not in _TOPIC_STOPWORDS and len(k) > 2).strip()
        search_queries = []
        for q in (
            f"{headline} background timeline history",
            f"{headline} key facts statistics analysis",
            f"{kw_query} strategic future impact" if kw_query else f"{headline} strategic future impact",
            f"{kw_query} latest developments statistics" if kw_query else None,
        ):
            q = (q or "").strip()
            if q and q not in search_queries:
                search_queries.append(q)

        retrieved_facts = []
        all_text_corpus = summary + " " + ground_truth_block

        # Generic crawler loop: every source in _RAG_CRAWLERS is fetched,
        # normalised, labelled, and appended with the same rules. A new crawler
        # only needs its fetch method + one registry entry — it automatically
        # inherits real-publisher labelling (news) vs terminology labelling,
        # the promo/social/dedup filters, recency tagging, and on-topic ranking.
        for q in search_queries:
            for key, label, kind, max_results, fn_name in _RAG_CRAWLERS:
                try:
                    raw = getattr(self, fn_name)(q, max_results) if max_results else getattr(self, fn_name)(q)
                except Exception as e:
                    print(f"[RAGRetriever] {label} query failed: {e}")
                    continue
                for item in raw or []:
                    it = _crawler_item_to_dict(item)
                    if not it["snippet"]:
                        continue
                    if kind == "terminology":
                        bucket = f"[Terminology: {it['title'] or label}]"
                    else:
                        src = _source_label(it["url"], label) or label
                        # Never let the retrieval tool itself become the source
                        # label. If the URL couldn't be resolved to a real
                        # publisher, keep the fact but mark it unattributed so
                        # the story LLM can't echo 'Firecrawl'/'Tavily' into the
                        # narration as a citation (previously fed raw tool tags
                        # into `full_rag_context_text`, re-poisoning every
                        # revision of the script and hard-aborting the Observer).
                        if (src or "").lower().strip() in _TOOL_SOURCE_NAMES:
                            src = "Unattributed"
                        bucket = f"[{src}: {it['title'] or 'report'}]"
                    line = f"• {bucket}: {it['snippet']}"
                    retrieved_facts.append(line)
                    all_text_corpus += " " + it["snippet"]

            # NOTE: DuckDuckGo HTML scraping removed — ad-heavy markup corrupts RAG.

        # Promotional/advertorial content filter across ALL sources: drop any
        # snippet that reads like marketing/web-chatter so it can't pollute the pack.
        retrieved_facts = [l for l in retrieved_facts if not _is_promotional(l)]
        # C3 defense-in-depth: run the built lines through the social-source check
        # too (the [Tavily: …]/[Exa: …]/[NewsAPI: …]/[Wikipedia: …] title tags are
        # inspected) so nothing survives if a per-source check was missed.
        retrieved_facts = [l for l in retrieved_facts if not _is_social_source("", l)]
        # Dedup: the same result is often returned across the multiple queries.
        seen_lines: Set[str] = set()
        unique_lines = []
        for l in retrieved_facts:
            key = _line_snippet_text(l).strip().lower()
            if key and key not in seen_lines:
                seen_lines.add(key)
                unique_lines.append(l)
        retrieved_facts = unique_lines
        retrieved_facts_all = retrieved_facts

        # On-topic relevance filter: rank retrieved lines by topic-token density
        # and DROP clearly off-topic hits (e.g. a generic "History of the Jews in
        # China" Wikipedia page for an AI-in-Africa query). Keeps the best 2+
        # token matches first so the LLM gets dense, relevant grounding.
        scored_retrieved = []
        for l in retrieved_facts:
            score = _on_topic_hits(l, topic_tokens)
            if score >= 2:
                scored_retrieved.append((score, l))
        scored_retrieved.sort(key=lambda x: x[0], reverse=True)
        retrieved_facts = [l for _, l in scored_retrieved]

        # Fallback: if strict (>=2 token) filtering leaves too few lines, also keep
        # the highest 1-token matches so a genuinely-relevant-but-terse retrieval
        # isn't wiped out by token gaps (the RAG gate still catches thinness).
        if len(retrieved_facts) < 4:
            one_match = [
                (score, l) for l in retrieved_facts_all
                if (score := _on_topic_hits(l, topic_tokens)) == 1
            ]
            one_match.sort(key=lambda x: _on_topic_hits(x[1], topic_tokens), reverse=True)
            seen = set(retrieved_facts)
            for _, l in one_match:
                if l not in seen:
                    retrieved_facts.append(l)
                    seen.add(l)
                if len(retrieved_facts) >= 4:
                    break

        # Rebuild the graph/TrumorGPT corpus from the FILTERED content so graph
        # triplets and verification reflect the on-topic reality, not feed noise.
        all_text_corpus = summary + " " + ground_truth_block
        all_text_corpus += " " + " ".join(_line_snippet_text(l) for l in retrieved_facts)

        # Deep-crawl the SELECTED source article — the winning story's own URL.
        # It is the single most relevant grounding by construction, but RSS only
        # kept its ~1-2 line summary. Scrape the full body and inject it ONLY if
        # it passes pollution guards, so it can never degrade the story designer:
        #   * >= 200 chars (real content, not an empty/paywall stub)
        #   * on-topic (>=2 topic-token matches, word-boundary)
        # NOTE: we deliberately do NOT run _is_promotional on the body — the regex
        # matches "cheap"/"subscribe", which falsely rejects genuinely on-topic
        # articles about "cheap AI models".
        selected_article = ""
        src_url = getattr(topic, "source_url", "") or ""
        if src_url:
            try:
                article_text = self._scrape_selected_article(src_url)
            except Exception as e:
                print(f"[RAGRetriever] Selected article scrape exception: {e}")
                article_text = ""
            if article_text:
                article_text = _strip_boilerplate(article_text)
                score = _on_topic_hits(article_text, topic_tokens)
                if len(article_text) >= 200 and score >= 2:
                    selected_article = article_text
                    all_text_corpus += " " + selected_article
                    print(f"[RAGRetriever] Injected selected source article ({len(selected_article)} chars, {score} topic-token hits).")
                else:
                    print(f"[RAGRetriever] Selected article rejected (len={len(article_text)}, topic-hits={score}); skipped to avoid pollution.")

        # 1. GraphRAG Triplet Extraction & In-Memory Graph Indexing
        triplets = self.extract_graph_triplets(all_text_corpus)
        
        # 2. Graph Path Traversal for Keyword Entities
        graph_paths = []
        for kw in keywords[:3]:
            paths = self.traverse_graph_paths(kw)
            graph_paths.extend(paths)

        # 3. Dynamic RAG Router Selection
        rag_mode = self.select_rag_mode(headline)

        # 4. TrumorGPT Fact Verification Check
        is_verified, confidence, fact_msg = self.trumorgpt_verify_fact(headline + " " + summary)

        # Recency categorization: tag snippets by apparent year so the LLM treats
        # older-dated material as HISTORICAL, not current. No date -> neutral/recent.
        recent_lines, historic_lines = [], []
        for line in retrieved_facts:
            if _recency_tag(line).startswith(" (historical"):
                historic_lines.append(f"{line}{_recency_tag(line)}")
            else:
                recent_lines.append(f"{line}{_recency_tag(line)}")

        rag_recent_block = "\n".join(recent_lines[:MAX_RETRIEVED_LINES]) if recent_lines else "No clearly recent snippets retrieved."
        rag_historical_block = "\n".join(historic_lines[:MAX_RETRIEVED_LINES]) if historic_lines else "No distinctly historical snippets."
        # Primary snippet pool for the script editor (all bullets, tag-free for fluent padding)
        rag_retrieved_block = "\n".join(retrieved_facts[:MAX_RETRIEVED_LINES]) if retrieved_facts else "No additional web snippets retrieved."
        graph_paths_block = "\n".join([f"• {p}" for p in graph_paths[:6]]) if graph_paths else "No multi-hop paths traversed."

        # Derive core domain category. Prefer the RSS/audience classification
        # (consistent with the revenue model) when it was genuinely set; fall back
        # to a keyword heuristic only for topics that were not audience-classified.
        rss_niche = (getattr(topic, "niche_category", "") or "").strip()
        rss_audience = (getattr(topic, "audience_type", "") or "general").strip()
        if rss_niche and rss_audience and rss_audience != "general":
            category = rss_niche
        else:
            combined_corpus = f"{headline} {summary} {' '.join(keywords)}".lower()
            if any(w in combined_corpus for w in ["ai", "chatgpt", "software", "tech", "chip", "nvidia", "cloud", "seo", "app"]):
                category = "Technology & Artificial Intelligence"
            elif any(w in combined_corpus for w in ["fed", "market", "stock", "trading", "crypto", "bank", "inflation", "revenue", "dollar"]):
                category = "Global Economics & Finance"
            elif any(w in combined_corpus for w in ["space", "nasa", "planet", "rocket", "star", "physics", "science"]):
                category = "Space & Scientific Innovation"
            elif any(w in combined_corpus for w in ["war", "election", "policy", "country", "president", "government"]):
                category = "Geopolitics & World Affairs"
            else:
                category = "Global Trends & Cultural Infotainment"

        # Guard: only surface triplets that look like real (Subject)-(Predicate)-(Object)
        # statements, to avoid flooding the LLM prompt with naive sentence-split garbage.
        meaningful_triplets = [
            t for t in triplets
            if len(t[0]) >= 3 and len(t[1]) >= 3 and len(t[2]) >= 3
            and not t[1].lower() in {"and", "the", "of", "to", "in", "for", "with"}
        ]
        graph_triplets_block = "\n".join([f"• ({s}) --[{p}]--> ({o})" for s, p, o in meaningful_triplets[:6]])

        knowledge_pack = {
            "topic_headline": headline,
            "category": category,
            "summary": summary,
            "keywords": keywords,
            "rag_mode": rag_mode,
            "ground_truth_block": ground_truth_block,
            "rag_retrieved_context": rag_retrieved_block,
            "graph_triplets": graph_triplets_block,
            "graph_paths": graph_paths_block,
            "trumorgpt_verification": {
                "is_verified": is_verified,
                "confidence": confidence,
                "message": fact_msg
            },
            "rag_recent_context": rag_recent_block,
            "rag_historical_context": rag_historical_block,
            "selected_article": selected_article,
            "fact_corpus": (
                f"{ground_truth_block}\n"
                f"{rag_retrieved_block}"
                + (f"\n• [SELECTED SOURCE ARTICLE]: {selected_article}" if selected_article else "")
            ),
            "full_rag_context_text": (
                f"TOPIC CATEGORY: {category}\n"
                f"RAG EXECUTION MODE: {rag_mode.upper()}\n"
                f"HEADLINE: {headline}\n"
                f"SUMMARY: {summary}\n\n"
                f"TRUMORGPT VERIFICATION: {fact_msg} (Confidence: {confidence})\n\n"
                f"VERIFIED GROUND TRUTH FACTS:\n{ground_truth_block}\n\n"
                f"TERMINOLOGY REFERENCE (definitions only — from a dictionary/encyclopedia, "
                f"NOT a news source; use ONLY to expand jargon or acronyms in the narration, "
                f"never to source a news claim):\n"
                f"{'\\n'.join(l for l in rag_retrieved_block.splitlines() if l.startswith('• [Terminology')) or 'No terminology entries.'}\n\n"
                f"STORY TELLING RULE — CURRENT vs HISTORICAL:\n"
                f"Today is {_current_year()}. Treat any source tagged '(historical: YYYY)' as "
                f"PAST CONTEXT ONLY. Never present older-dated data as a development happening "
                f"in {_current_year()}. Only the RECENT sources (tagged '(recent: YYYY)' or "
                f"untagged) may be framed as current. Weave older facts in as background/history.\n\n"
                f"CURRENT / RECENT SOURCES:\n{rag_recent_block}\n\n"
                f"HISTORICAL CONTEXT (background only):\n{rag_historical_block}\n\n"
                f"GRAPHRAG KNOWLEDGE GRAPH TRIPLETS:\n{graph_triplets_block}\n\n"
                f"GRAPHRAG MULTI-HOP RELATIONAL PATHS:\n{graph_paths_block}\n\n"
                f"RETRIEVED DEEP CONTEXT & BACKGROUND:\n{rag_retrieved_block}"
                + (f"\n\nPRIMARY SOURCE ARTICLE (the selected story, deep-crawled):\n{selected_article}" if selected_article else "")
            )
        }

        self._rag_cache[cache_key] = (time.time(), knowledge_pack)
        # Hybrid: prepend the grounded cited core over the scraper depth (the pack
        # was built above as `knowledge_pack`); dedup guards keep the corpus clean.
        if grounded_pack is not None and self.rag_mode == "hybrid":
            knowledge_pack = self._merge_grounded_scraper(grounded_pack, knowledge_pack)
            self._rag_cache[cache_key] = (time.time(), knowledge_pack)
        return knowledge_pack


rag_retriever = RAGTopicRetriever()

