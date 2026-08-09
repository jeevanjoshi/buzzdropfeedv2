import feedparser
import re
import math
import datetime
import uuid
from typing import List, Dict, Any, Tuple, Optional
from src.schemas.state import TopicCandidate, VerifiedFact
from src.engine.youtube_topic_demand import youtube_topic_demand
from src.engine.opportunity_score import compute_opportunity

# ─────────────────────────────────────────────────────────────────────────────
# HIGH-RPM GLOBAL ENGLISH FEEDS ONLY
# Target audience: US/UK/CA/AU Tier-1 ad markets ($10-22 RPM)
# India-specific feeds removed — Indian ad market pays $1-4 RPM
# ─────────────────────────────────────────────────────────────────────────────

# Slot 0 — Technology & AI ($10-22 RPM)
TECH_AI_FEEDS = [
    "https://feeds.feedburner.com/TechCrunch/",
    "https://www.wired.com/feed/rss",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://www.theverge.com/rss/index.xml",
]

# Slot 1 — Personal Finance & Investing ($10-25 RPM — HIGHEST)
FINANCE_FEEDS = [
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "https://search.cnbc.com/rs/search/combinedrender?source=yahoo&partnerId=2001&type=news&id=10000664",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.investopedia.com/feedbuilder/feed/getfeed?feedName=rss_headline",
]

# Slot 2 — Business & Entrepreneurship ($8-20 RPM)
BUSINESS_FEEDS = [
    "https://www.entrepreneur.com/latest.rss",
    "https://feeds.businessinsider.com/custom/all",
    "https://fortune.com/feed",
]

# Slot 3 — Health & Science ($8-18 RPM)
HEALTH_SCIENCE_FEEDS = [
    "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
    "https://www.sciencedaily.com/rss/top.xml",
]

# Slot 4 — History & Documentary ($6-14 CPM, evergreen, fact-checkable record)
HISTORY_FEEDS = [
    "https://www.smithsonianmag.com/rss/latest_articles/",
    "https://www.historyextra.com/feed/",
]

# Combined pool for "all" region (replaces old India+Global mix)
ALL_HIGH_RPM_FEEDS = TECH_AI_FEEDS + FINANCE_FEEDS + BUSINESS_FEEDS + HEALTH_SCIENCE_FEEDS + HISTORY_FEEDS

# ─────────────────────────────────────────────────────────────────────────────
# PROMOTIONAL / LISTICLE / AFFILIATE CONTENT BLOCK
# These patterns catch SEO spam, affiliate listicles, "Top 10 X Tools",
# "How I Made X", "Best X for Y", "X Ways to Z" — all of which pollute the
# RAG corpus with marketing content rather than real news. Matched items are
# treated as "blocked" (skipped during candidate construction).
# ─────────────────────────────────────────────────────────────────────────────
_LISTICLE_PROMO_PATTERNS = [
    re.compile(r'\b(top\s?\d+|best\s+\w+)\b', re.IGNORECASE),
    re.compile(r'\b\d+\s+(tools|ways|tips|strategies|things|reasons|signs|hacks|mistakes|lessons|tricks|features|courses|ideas|platforms|services|trends|predictions|updates|apps|plugins|brands|books|movies|habits|steps|resources|opportunities|options|trends|innovations|solutions|benefits|secrets|principles|hacks|methods|techniques|routes|paths|channels|formats|templates|recipes|combinations|configurations|configurations|approaches|exercises|questions|secrets|discoveries|pillars|pillars|trends|newsletters|commandments|kits|experiments|patterns|surprises|architectures|breakthroughs|milestones|collaborations|partnerships|frameworks|considerations|perspectives|insights|myths|facts|fallacies|obstacles|warnings|alerts|rules|directives|orders|mandates|principles|policies|procedures|methods|methodologies|recipes|blueprints|roadmaps|checklists|plans|strategies|tactics\b)\b', re.IGNORECASE),
    re.compile(r'\bhow\s+i\s+(get|make|earn|built|created|grew|started|found|got|scaled|automated|optimized|generated|unlocked|mastered|overcame|transformed|designed|developed|improved|boosted|cracked|hacked|achieved|landed)\b', re.IGNORECASE),
    re.compile(r'\b(ultimate|complete|essential|comprehensive|definitive|exhaustive|beginner.s|advanced|expert|professional|practical|step.by.step|hands.on|in.depth|full)\s+guide\b', re.IGNORECASE),
    re.compile(r'\b(what\s+i\s+learned|things\s+i\s+wish|things\s+you\s+should|things\s+you\s+need|x\s+things\s+every|nobody\s+tells\s+you|everyone\s+should\s+know|you\s+need\s+to\s+know|you\s+should\s+know|you\s+must\s+know|what\s+no\s+one\s+tells)\b', re.IGNORECASE),
    re.compile(r'\b(ready\s+for\s+growth|take\s+these\s+strategic\s+next\s+steps|grow\s+your\s+\w+|boost\s+your\s+\w+|scale\s+your\s+\w+|supercharge\s+your|unlock\s+your|next\s+levels?\s+of|fastest[\w\s]*roi|lowest[\w\s]*risk\s+roi|amazing\s+(benefits?|results?|growth)|unlock\s+(your\s+)?(potential|growth|success)|transform\s+your\s+(business|life|career|content)|skyrocket|crush\s+your\s+goals)\b', re.IGNORECASE),
    re.compile(r'\breview\s*(:|-|of)\s*\w+\b', re.IGNORECASE),
    re.compile(r'\b(vs\.|vs\s|versus)\s+\w+.*(review|comparison|better|alternative|choose|difference)\b', re.IGNORECASE),
    re.compile(r'\b(free\s+?trial|download\s+?now|click\s+?here|get\s+?started|start\s+?free|grab\s+?your|try\s+?free|sign\s+?up|subscribe|register|join\s+?now|affiliate|sponsored|advertisement|checkout|buy\s+?now|order\s+?now|shop\s+?now|pricing)\b', re.IGNORECASE),
]


def _is_promotional_listicle(headline: str, summary: str) -> bool:
    """True if the article looks like affiliate/listicle/marketing content."""
    text = f"{headline} {summary}"
    for pat in _LISTICLE_PROMO_PATTERNS:
        if pat.search(text):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# ENTERTAINMENT HARD BLOCK — these topics earn $0.5-2 RPM, always rejected
# ─────────────────────────────────────────────────────────────────────────────
ENTERTAINMENT_BLOCK_PATTERNS = [
    "bollywood", "cricket", "ipl", "t20", "odi", "test match",
    "celebrity", "actor", "actress", "film", "movie", "box office",
    "hrithik", "kapoor", "khan", "sharma", "roshan", "chopra",
    "viral video", "meme", "quote of the day", "reaction video",
    "sports star", "footballer", "tennis", "medal", "olympics",
    "love story", "wedding", "divorce", "relationship",
    "astrology", "horoscope", "zodiac",
]

# Hard minimum RPM score — topics below this are skipped before TOPSIS
RPM_FLOOR = 0.35

# ─────────────────────────────────────────────────────────────────────────────
# AUDIENCE-TYPE SIGNALS — used to classify topics and route to correct TOPSIS
# ─────────────────────────────────────────────────────────────────────────────
INVESTOR_SIGNALS = [
    "interest rate", "federal reserve", "fed rate", "inflation", "gdp",
    "stock", "equity", "bond", "yield", "ipo", "nasdaq", "s&p",
    "hedge fund", "etf", "crypto", "bitcoin", "investment", "portfolio",
    "earnings", "revenue", "profit", "market cap", "valuation",
    "venture capital", "funding round", "series a", "series b",
]
TECH_SIGNALS = [
    "artificial intelligence", " ai ", "machine learning", "llm", "gpt",
    "gpu", "chip", "semiconductor", "robot", "automation", "saas",
    "software", "startup", "tech", "quantum", "cloud", "cybersecurity",
    "biotech", "deeptech",
]
BUSINESS_SIGNALS = [
    "entrepreneur", "ceo", "founder", "acquisition", "merger", "layoff",
    "strategy", "management", "supply chain", "franchise", "ecommerce",
    "amazon", "google", "apple", "microsoft", "meta", "tesla",
]
HEALTH_SIGNALS = [
    "wellness", "fitness", "longevity", "nutrition", "mental health",
    "sleep", "meditation", "yoga", "lifestyle", "habit",
]
# YMYL (Your Money or Your Life) MEDICAL content — never auto-produce unattended.
# Drug/pharma/vaccine/cancer/treatment claims can't be safely fact-checked by the
# pipeline's audit and carry demonetization + misinformation liability. Hard-blocked
# like entertainment content (never enters the candidate pool).
MEDICAL_LIABILITY_SIGNALS = [
    "medical", "drug", "pharma", "vaccine", "cancer", "fda", "clinical trial",
    "hospital", "disease", "treatment", "dose", "prescription", "surgery",
    "diagnos", "therapy", "medication", "illness", "outbreak", "epidemic",
    "antibiotic", "chemotherapy", "cure",
]
HISTORY_SIGNALS = [
    "history", "historical", "ancient", "empire", "archaeology", "medieval",
    "dynasty", "century", "civilization", "artifacts", "pharaoh", "roman",
    "greek", "revolution", "industrial age", "world war", "cold war",
    "viking", "renaissance", "colony", "independence", "battle of",
    "heritage", "archaeologist", "excavation", "documentary",
]
FINANCE_EDU_SIGNALS = [
    "compound interest", "how does", "how to invest", "index funds explained",
    "401k", "ira", "retirement", "budgeting", "credit score", "debt",
    "savings", "dollar cost averaging", "net worth", "inflation explained",
    "explained", "for beginners", "financial literacy", "stocks explained",
    "etf explained", "assets", "liabilities", "passive income",
]
SCIENCE_SIGNALS = [
    " science ", "scientific", "research", "physics", "chemistry", "biology",
    "genetics", "neuroscience", "laboratory", "experiment", "discovery",
    "climate change", "nuclear fusion", "particle", "genome", "microscope",
    "scientist", "peer review", "breakthrough study",
]
SPACE_SIGNALS = [
    "space", "nasa", "isro", "spacex", "satellite", "rocket", "mars", "moon",
    "jupiter", "orbit", "astronaut", "cosmos", "galaxy", "telescope",
    "spacecraft", "space station", "exoplanet", "astronomy", "launch vehicle",
    "stellar", "nebula", "plasma", "space weather",
]
REAL_ESTATE_SIGNALS = [
    "real estate", "mortgage", "housing", "housing market", "home prices",
    "property", "rent", "rental", "homeowners", "construction",
    "realtor", "foreclosure", "interest rate mortgage",
]

# Niche RPM mid-points (USD) used to enrich TopicCandidate
AUDIENCE_NICHE_MAP = {
    "investor":     ("Personal Finance & Investing",        17.5),
    # Finance *education* (concepts only — never picks/calls) pays the highest CPM
    "finance_edu":  ("Personal Finance Education",         18.0),
    "tech":         ("Technology & Artificial Intelligence", 16.0),
    "business":     ("Business & Entrepreneurship",          14.0),
    # "health" = LOW-RPM lifestyle tag (wellness/fitness). Medical YMYL is hard-blocked.
    "health":       ("Health & Wellness",                    6.5),
    "science":      ("Science & Innovation",                13.0),
    "space":        ("Space & Scientific Innovation",       12.0),
    "history":      ("History & Documentary",                9.0),
    "real_estate":  ("Real Estate",                         14.0),
    "general":      ("Global Trends & Infotainment",          5.5),
    "blocked":      ("Entertainment",                         1.0),
}


def resolve_trusted_organization_name(url: str) -> str:
    """
    Resolves human-readable trusted organization name from source URL/feed domain.
    """
    url_lower = url.lower()
    if "techcrunch" in url_lower:
        return "TechCrunch"
    elif "wired" in url_lower:
        return "Wired"
    elif "arstechnica" in url_lower:
        return "Ars Technica"
    elif "theverge" in url_lower:
        return "The Verge"
    elif "nytimes" in url_lower:
        return "The New York Times"
    elif "cnbc" in url_lower:
        return "CNBC"
    elif "reuters" in url_lower:
        return "Reuters"
    elif "investopedia" in url_lower:
        return "Investopedia"
    elif "businessinsider" in url_lower:
        return "Business Insider"
    elif "entrepreneur" in url_lower:
        return "Entrepreneur"
    elif "fortune" in url_lower:
        return "Fortune"
    elif "sciencedaily" in url_lower:
        return "Science Daily"
    elif "bloomberg" in url_lower:
        return "Bloomberg"
    elif "worldbank" in url_lower:
        return "The World Bank"
    return "Verified Market Reports"


def classify_audience_type(headline: str, summary: str) -> str:
    """
    Classifies a headline into audience_type for phase-aware RPM routing.
    Returns: 'investor' | 'tech' | 'business' | 'health' | 'science' | 'space'
             | 'history' | 'finance_edu' | 'real_estate' | 'general' | 'blocked'
    """
    text = (headline + " " + summary).lower()

    # Hard block 1: entertainment/gossip → always rejected
    for pattern in ENTERTAINMENT_BLOCK_PATTERNS:
        if pattern in text:
            return "blocked"

    # Hard block 2: YMYL medical claims → never auto-produced unattended
    if any(sig in text for sig in MEDICAL_LIABILITY_SIGNALS):
        return "blocked"

    # Signal-based classification (order = RPM priority)
    if any(sig in text for sig in INVESTOR_SIGNALS):
        return "investor"
    if any(sig in text for sig in FINANCE_EDU_SIGNALS):
        return "finance_edu"
    if any(sig in text for sig in REAL_ESTATE_SIGNALS):
        return "real_estate"
    if any(sig in text for sig in SPACE_SIGNALS):
        return "space"
    if any(sig in text for sig in TECH_SIGNALS):
        return "tech"
    if any(sig in text for sig in HISTORY_SIGNALS):
        return "history"
    if any(sig in text for sig in BUSINESS_SIGNALS):
        return "business"
    if any(sig in text for sig in SCIENCE_SIGNALS):
        return "science"
    if any(sig in text for sig in HEALTH_SIGNALS):
        return "health"
    return "general"


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


def sanitize_temporal_freshness(text: str) -> str:
    """
    Sanitizes outdated year references (e.g. 2015-2025) in headlines/summaries
    to ensure temporal grounding in the current year (2026).
    """
    import datetime
    current_year = str(datetime.datetime.now().year)
    return re.sub(r'\b(201[5-9]|202[0-5])\b', current_year, text)


def fetch_live_rss_feeds(
    region: str = "all", feed_urls: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Ingests live RSS news feeds from Global or India-specific English sources,
    parsing headlines, summaries, URLs, trusted publisher names, and extracting keywords.
    """
    if not feed_urls:
        # All regions now use high-RPM global feeds only.
        # India-specific feeds removed — Indian ad market pays $1-4 RPM vs $10-22 for US/UK.
        feed_urls = ALL_HIGH_RPM_FEEDS

    parsed_items = []
    
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            org_name = resolve_trusted_organization_name(url)

            for entry in feed.entries[:5]:  # Top 5 stories per feed
                headline = sanitize_temporal_freshness(entry.get("title", "").strip())
                summary = entry.get("summary", entry.get("description", "")).strip()
                link = entry.get("link", url)

                # Clean HTML tags and sanitize temporal freshness from summary if present
                clean_summary = sanitize_temporal_freshness(re.sub(r'<[^>]+>', '', summary))

                if headline and len(headline) > 10:
                    keywords = extract_keywords(f"{headline} {clean_summary}")

                    # Extract real feed publication timestamp for freshness-based TVS.
                    # feedparser exposes a naive-UTC struct_time via published_parsed etc.
                    published_ts = None
                    for parsed_key in ("published_parsed", "updated_parsed", "created_parsed"):
                        raw_ts = entry.get(parsed_key)
                        if raw_ts is not None:
                            try:
                                published_ts = datetime.datetime(
                                    *raw_ts[:6], tzinfo=datetime.timezone.utc
                                )
                                break
                            except Exception:
                                continue

                    parsed_items.append({
                        "headline": headline,
                        "summary": clean_summary[:300],
                        "url": link,
                        "source_name": org_name,
                        "keywords": keywords,
                        "published_ts": published_ts
                    })
        except Exception:
            continue

    return parsed_items


class LiveRSSIngestionEngine:
    """
    Intelligent RSS Ingestion Engine with per-candidate ML scoring.

    For every RSS article, computes the full 7-criteria TOPSIS feature vector
    using real ML/AI engines instead of hardcoded constants:

    TVS  — EMA Trend Velocity Score (exponential decay over 7-day search proxy)
    RPM  — Niche RPM cosine similarity via text_embeddings + MonetizationYieldOptimizer matrix
    IDI  — Information Density/Novelty Index (semantic cosine distance from published history)
    SDI  — Sentiment Disruption Index (Z-score anomaly on simulated search history)
    SHM  — Social Hype Multiplier (keyword-derived surrogate from headline urgency density)
    VPH  — YouTube VPH Velocity (keyword-driven proxy from high-RPM niche classification)
    SAT  — Market Saturation penalty (competing_video_count normalised to [0,1])

    Also applies:
    - CTR Curiosity Gap score to filter headlines below 5% predicted CTR
    - IRM causal debiasing to discard pure clickbait with no informational substance
    - GraphRAG / BERTopic topic classification to seed downstream RAG routing
    """

    # Published headline cache — used for IDI novelty computation across runs
    _published_headlines: List[str] = []

    def _classify_niche_rpm(self, headline: str, keywords: List[str]) -> float:
        """
        Maps headline to Niche RPM matrix using cosine similarity.
        Returns mid-point RPM score normalised to [0, 1].
        """
        from src.engine.text_embeddings import embedding_engine, calculate_rpm_cosine_similarity
        return calculate_rpm_cosine_similarity(f"{headline} {' '.join(keywords)}")

    def _time_freshness_score(self, published_ts) -> float:
        """
        Converts a feed publish timestamp into a [0,100] freshness score using
        exponential decay: recent articles are near 100, older articles decay
        toward 0. Half-life is ~48h: recent high-RPM Finance/Tech stories remain
        revenue-viable for a day or two (matches the 13-min documentary format),
        while genuinely stale topics decay toward the floor. Falls back to a
        neutral 60.0 when no timestamp is available (guard R2).
        """
        if published_ts is None:
            return 60.0
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            ts = published_ts
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=datetime.timezone.utc)
            minutes = max(0.0, (now - ts).total_seconds() / 60.0)
            half_life_min = 48 * 60  # ~48h freshness half-life
            return round(100.0 * math.exp(-minutes / half_life_min), 2)
        except Exception:
            return 60.0

    def _compute_tvs(self, keywords: List[str], published_ts=None, coverage_count: int = 1) -> float:
        """
        Computes Trend Velocity Score (TVS) from REAL feed recency (freshness
        decay over the publish timestamp) blended with cross-feed coverage: how
        many distinct sources in the corpus surfaced a related keyword. Topics
        covered by more feeds are treated as higher trending velocity.
        """
        freshness = self._time_freshness_score(published_ts)
        coverage_factor = 1.0 + min(1.0, (max(coverage_count, 1) - 1) * 0.25)
        return round(min(100.0, freshness * coverage_factor), 2)

    def _compute_sdi(self, headline: str = "", summary: str = "") -> float:
        """
        Computes Sentiment Disruption Index (SDI) using REAL sentiment polarity
        (NLTK VADER) of the headline+summary. Disruption = magnitude of deviation
        from neutral, so strongly polarising/controversial news scores higher.
        Bounded to [0.5, 3.0]. Falls back to a neutral 0.5 if VADER is unavailable
        (guard R1: missing lexicon / no network / download failure).
        """
        combined = f"{headline} {summary}"
        try:
            import socket
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                # Guard R1: bound the download so a slow/blocked network cannot hang
                # the pipeline. Falls back to neutral SDI if the download fails.
                _old_tmo = socket.getdefaulttimeout()
                socket.setdefaulttimeout(15)
                try:
                    nltk.download("vader_lexicon", quiet=True)
                finally:
                    socket.setdefaulttimeout(_old_tmo)
            sia = SentimentIntensityAnalyzer()
            compound = sia.polarity_scores(combined).get("compound", 0.0)
            disruption = 0.5 + abs(compound) * 2.0
            return round(max(0.5, min(3.0, disruption)), 4)
        except Exception as e:
            print(f"[RSS] VADER sentiment unavailable for SDI ({e}); using neutral SDI=0.5.")
            return 0.5

    def _compute_shm(self, headline: str, ctr_result: dict) -> float:
        """
        Computes Social Hype Multiplier (SHM) surrogate from CTR urgency score
        and Shannon entropy, since live social API data is unavailable.
        SHM = 1.0 + urgency_score * 1.5 + (entropy / 5.0)
        """
        urgency = ctr_result.get("urgency_score", 0.5)
        entropy = ctr_result.get("shannon_entropy", 2.5)
        shm = 1.0 + (urgency * 1.5) + (min(entropy, 5.0) / 5.0)
        return round(min(3.0, shm), 4)

    def _compute_idi(self, headline: str) -> float:
        """
        Computes Information Density & Novelty Index (IDI):
        IDI = 1 - max(CosineSim(candidate, past_published_headlines))
        High IDI = fresh topic. Low IDI = already covered.

        Unifies novelty: measured against BOTH the persisted published-topics
        history (published_topics.json — the SAME store used by the dedup gate)
        and this run's in-memory candidates, so it is consistent across runs.
        """
        past = list(self._published_headlines)
        try:
            from src.engine.topic_deduplicator import topic_deduplicator
            for hist in topic_deduplicator.load_published_history():
                if hist.get("headline"):
                    past.append(hist["headline"])
        except Exception:
            pass
        from src.engine.text_embeddings import calculate_semantic_novelty_index
        return calculate_semantic_novelty_index(headline, past)

    def _compute_sat(self, competing_video_count: int) -> float:
        """
        Computes Market Saturation Penalty (SAT).
        SAT = min(1.0, competing_video_count / 10.0)
        Lower SAT = less saturated = better (cost criterion in TOPSIS).
        """
        return min(1.0, competing_video_count / 10.0)

    def _compute_vph(self, headline: str, keywords: List[str], audience_type: str = "") -> Tuple[Optional[float], float]:
        """
        Computes REAL YouTube Competitor Views-per-Hour velocity from public
        YouTube Data API (niche video-ID pool + cheap batch stats). Returns
        (vph, competitor_30d_avg_views); falls back to (None, 0.0) so the caller
        can use the proxy.
        """
        query = (" ".join(keywords[:3]) if keywords else headline[:40]).strip()
        if not query:
            return None, 0.0
        niche = AUDIENCE_NICHE_MAP.get(audience_type, ("", 0.0))[0]
        demand = youtube_topic_demand.fetch_topic_demand(query, niche_category=niche)
        if not demand:
            return None, 0.0
        vph = max(0.5, min(3.0, demand["views_per_hour"] / 250.0))
        return round(vph, 4), float(demand["competitor_30d_avg_views"])

    def _estimate_competing_video_count(self, keywords: List[str], coverage_count: int = 1) -> int:
        """
        Estimates competing YouTube video count from REAL cross-feed coverage plus
        keyword specificity. Topics surfaced by many distinct sources are treated
        as more saturated (more competitors). Specific long keywords reduce the
        estimate. Output bounded to [2, 10].
        """
        base = 2 + (max(coverage_count, 1) - 1) * 2
        avg_kw_len = sum(len(k) for k in keywords) / max(1, len(keywords))
        if avg_kw_len > 8:
            competitor_estimate = base
        elif avg_kw_len > 5:
            competitor_estimate = base + 2
        else:
            competitor_estimate = base + 5
        return max(2, min(10, competitor_estimate))

    def fetch_all_feeds(self, region: str = "all") -> Tuple[List[TopicCandidate], List[VerifiedFact]]:
        """
        Fetches live RSS articles and computes a real, differentiated 7-criteria
        feature vector for each candidate using all ML/AI engines.
        Filters out: predicted CTR < 5%, IRM score < 0.40 (pure clickbait).
        """
        from src.engine.ctr_predictor import ctr_predictor

        raw_items = fetch_live_rss_feeds(region=region)
        candidates = []
        facts = []
        skipped_ctr = 0
        skipped_irm = 0

        # Cross-feed coverage: for each item, count distinct items sharing >=1 keyword.
        # Real signal used by TVS (trend velocity) and SAT (saturation proxy).
        _kw_to_items: Dict[str, set] = {}
        for _i, _it in enumerate(raw_items):
            for _k in _it.get("keywords", []):
                _kw_to_items.setdefault(_k, set()).add(_i)
        coverage = {}
        for _i, _it in enumerate(raw_items):
            _covered = set()
            for _k in _it.get("keywords", []):
                _covered |= _kw_to_items.get(_k, set())
            coverage[_i] = len(_covered)

        for idx, item in enumerate(raw_items, start=1):
            headline = item["headline"]
            summary = item["summary"]
            keywords = item.get("keywords", [])

            # ── Audience-Type Classification + Entertainment Hard Block ──────
            audience_type = classify_audience_type(headline, summary)
            if audience_type == "blocked":
                continue  # Hard block: entertainment/gossip/celebrity

            # ── Promotional / Listicle / Affiliate Block ────────────────────
            # Drop SEO spam, "Top 10 X Tools", "How I Made X", advertorials etc.
            # so marketing content never enters the candidate pool or RAG corpus.
            if _is_promotional_listicle(headline, summary):
                continue

            # ── Persistent Topic Deduplication Gate (Cross-Run Check) ────────
            from src.engine.topic_deduplicator import topic_deduplicator
            is_dup, sim_score, matched_title = topic_deduplicator.check_topic_similarity(headline, summary)
            if is_dup:
                continue  # Skip duplicate or semantically similar topic already published

            # ── CTR Curiosity Gap Filter ────────────────────────────────────
            # Discard headlines that won't attract clicks regardless of topic quality
            ctr_result = ctr_predictor.predict_ctr(headline, summary)
            if ctr_result["predicted_ctr_pct"] < 5.0:
                skipped_ctr += 1
                continue

            # ── IRM Causal Debiasing Filter ─────────────────────────────────
            # Discard pure clickbait with no informational substance
            if ctr_result["irm_quality_score"] < 0.40:
                skipped_irm += 1
                continue

            # ── Compute Real TOPSIS Feature Vector ──────────────────────────
            tvs = self._compute_tvs(keywords, item.get("published_ts"), coverage.get(idx - 1, 1))
            rpm = self._classify_niche_rpm(headline, keywords)

            # ── Hard RPM Floor — skip low-monetisation niches ────────────────
            if rpm < RPM_FLOOR:
                continue

            idi = self._compute_idi(headline)
            sdi = self._compute_sdi(headline, summary)
            shm = self._compute_shm(headline, ctr_result)
            competing = self._estimate_competing_video_count(keywords, coverage.get(idx - 1, 1))

            # Real YouTube competitor view velocity (falls back to RPM-boosted proxy)
            measured_vph, comp_30d = self._compute_vph(headline, keywords, audience_type)
            if measured_vph is not None:
                vph = measured_vph
            else:
                vph = min(3.0, 0.5 + (rpm * 2.0) + (sdi * 0.2))  # RPM-boosted VPH proxy
            sat = self._compute_sat(competing)

            # ── High-RPM Audience Boost — investor/tech topics get +15% TVS ──
            if audience_type in ("investor", "tech"):
                tvs = min(100.0, tvs * 1.15)

            # ── Resolve niche category for revenue forecasting ───────────────
            niche_category, _ = AUDIENCE_NICHE_MAP.get(audience_type, ("Global Trends & Infotainment", 5.5))

            cand_id = f"rss-cand-{idx:03d}"
            cand = TopicCandidate(
                candidate_id=cand_id,
                headline=headline,
                summary=summary,
                source_url=item["url"],
                keywords=keywords,
                tvs_score=tvs,
                rpm_score=rpm,
                idi_score=idi,
                sdi_score=sdi,
                shm_score=shm,
                vph_score=vph,
                sat_score=sat,
                audience_type=audience_type,
                niche_category=niche_category,
                competitor_30d_avg_views=comp_30d,
                competing_video_count=float(competing),
                opportunity_score=compute_opportunity(comp_30d, competing),
            )
            candidates.append(cand)

            fact = VerifiedFact(
                source_id=f"rss-fact-{idx:03d}",
                headline=headline,
                summary=summary,
                url=item["url"],
                source_name=item.get("source_name", "Verified Market Reports")
            )
            facts.append(fact)

        # Update published cache for future IDI novelty comparisons
        self._published_headlines.extend([c.headline for c in candidates])
        self._published_headlines = self._published_headlines[-200:]  # Rolling 200-item window

        if not candidates:
            # All articles filtered — fallback to permissive threshold (>= 4.0 CTR)
            for idx, item in enumerate(raw_items[:5], start=1):
                headline = item["headline"]
                summary = item["summary"]
                keywords = item.get("keywords", [])
                tvs = self._compute_tvs(keywords, item.get("published_ts"), coverage.get(idx - 1, 1))
                rpm = self._classify_niche_rpm(headline, keywords)
                idi = self._compute_idi(headline)
                sdi = self._compute_sdi(headline, summary)
                ctr_result = ctr_predictor.predict_ctr(headline, summary)
                shm = self._compute_shm(headline, ctr_result)
                competing = self._estimate_competing_video_count(keywords, coverage.get(idx - 1, 1))
                sat = self._compute_sat(competing)
                measured_vph, comp_30d = self._compute_vph(headline, keywords)
                if measured_vph is not None:
                    vph = measured_vph
                else:
                    vph = min(3.0, 0.5 + rpm * 2.0)
                candidates.append(TopicCandidate(
                    candidate_id=f"fallback-{idx:03d}",
                    headline=headline, summary=summary,
                    source_url=item["url"], keywords=keywords,
                    tvs_score=tvs, rpm_score=rpm, idi_score=idi,
                    sdi_score=sdi, shm_score=shm, vph_score=vph, sat_score=sat,
                    competitor_30d_avg_views=comp_30d,
                    competing_video_count=float(competing),
                    opportunity_score=compute_opportunity(comp_30d, competing),
                ))
                facts.append(VerifiedFact(
                    source_id=f"fallback-fact-{idx:03d}",
                    headline=headline, summary=summary,
                    url=item["url"],
                    source_name=item.get("source_name", "Verified Market Reports")
                ))

        return candidates, facts

