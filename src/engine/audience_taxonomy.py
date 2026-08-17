"""Single source of truth for the audience/niche taxonomy.

Previously the audience type -> niche -> RPM -> playlist mapping was split across
three places that could drift apart:

* ``AUDIENCE_NICHE_MAP`` + ``classify_audience_type`` (rss_ingestion.py)
* ``NICHE_RPM_MATRIX`` (monetization_optimizer.py)
* ``_OUTCOME_PLAYLISTS`` (publisher.py)

This module consolidates the *classification* half (audience_type -> niche, rpm,
playlist, description, and the keyword signals used to detect it). The monetization
RPM bands live in ``monetization_optimizer.NICHE_RPM_MATRIX`` (keyed by the same
niche string) and the playlist routing reads ``playlist_for`` here, so every
consumer agrees on one definition.

Priority is ascending: lower numbers are checked first, so more specific /
higher-RPM audiences win ties. ``geopolitics`` is intentionally placed *after*
``investor``/``finance_edu`` so genuine finance stories keep their top-RPM bucket,
while international-relations stories stop being misrouted into
"Personal Finance Education" (the bug that tagged an Iran nuclear-claim story as
finance_edu). The earlier misroute was caused by weak finance_edu signals
("explained", "assets", "liabilities", "passive income"); those have been removed
so only explicit finance-education terms qualify.
"""

from typing import Any, Dict, List, Optional, Tuple

# audience_type -> definition. ``signals`` are lower-cased substring matchers.
AUDIENCE_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "investor": {
        "niche": "Personal Finance & Investing",
        "rpm": 17.5,
        "priority": 10,
        "playlist": ("Finance, Markets & Wealth Stories",
                     "Documentary storytelling on markets, investing, money and the people behind them."),
        "description": "Stocks, bonds, markets, investing, macro-economy, personal investing strategy.",
        "signals": [
            "interest rate", "federal reserve", "fed rate", "inflation", "gdp",
            "stock", "equity", "bond", "yield", "ipo", "nasdaq", "s&p",
            "hedge fund", "etf", "crypto", "bitcoin", "investment", "portfolio",
            "earnings", "revenue", "profit", "market cap", "valuation",
            "venture capital", "funding round", "series a", "series b",
        ],
    },
    "finance_edu": {
        # Finance *education* (concepts only) pays the highest CPM of the group.
        "niche": "Personal Finance Education",
        "rpm": 18.0,
        "priority": 20,
        "playlist": ("Finance, Markets & Wealth Stories",
                     "Documentary storytelling on markets, investing, money and the people behind them."),
        "description": "Personal-finance concepts explained: saving, budgeting, retirement, literacy.",
        # Tightened: only explicit finance-education terms. Generic words like
        # "explained", "assets", "liabilities", "passive income" were removed
        # because they misrouted unrelated stories (e.g. geopolitics) into this
        # top-RPM bucket.
        "signals": [
            "compound interest", "how does", "how to invest", "index funds explained",
            "401k", "roth ira", "retirement", "budgeting", "credit score", "debt",
            "savings", "dollar cost averaging", "net worth", "inflation explained",
            "financial literacy", "stocks explained", "etf explained",
        ],
    },
    "geopolitics": {
        "niche": "World News & Geopolitics",
        "rpm": 11.0,
        "priority": 25,
        "playlist": ("Global Trends & Infotainment",
                     "Global trends, culture and infotainment documentaries."),
        "description": "International relations, war, diplomacy, nuclear/security, elections, misinformation campaigns.",
        "signals": [
            "geopolitic", "diplomacy", "diplomatic", "foreign policy", "sanction",
            "military", "nuclear weapon", "nuclear power", "nuclear threat",
            "nuclear strike", "nuclear program", "missile", "war ", "troops",
            "election", "treaty", "border", "summit", "nato", "united nations",
            "washington", "iran", "israel", "ukraine", "russia", "china",
            "false claim", "misinformation", "disinformation", "embassy",
            "prime minister",
        ],
    },
    "real_estate": {
        "niche": "Real Estate",
        "rpm": 14.0,
        "priority": 30,
        "playlist": ("Finance, Markets & Wealth Stories",
                     "Documentary storytelling on markets, investing, money and the people behind them."),
        "description": "Property, mortgages, housing market, real-estate investment.",
        "signals": [
            "real estate", "mortgage", "housing", "housing market", "home prices",
            "property", "rent", "rental", "homeowners", "construction",
            "realtor", "foreclosure",
        ],
    },
    "space": {
        "niche": "Space & Scientific Innovation",
        "rpm": 12.0,
        "priority": 40,
        "playlist": ("Space, Cosmology & Economic History",
                     "Documentary deep-dives on space, the cosmos and economic history."),
        "description": "Astronomy, space exploration, cosmology.",
        "signals": [
            "space", "nasa", "isro", "spacex", "satellite", "rocket", "mars", "moon",
            "jupiter", "orbit", "astronaut", "cosmos", "galaxy", "telescope",
            "spacecraft", "space station", "exoplanet", "astronomy", "launch vehicle",
            "stellar", "nebula", "plasma", "space weather",
        ],
    },
    "tech": {
        "niche": "Technology & Artificial Intelligence",
        "rpm": 16.0,
        "priority": 50,
        "playlist": ("AI, Tech & Innovation Deep-Dives",
                     "In-depth documentaries on AI, technology, business and scientific breakthrough stories."),
        "description": "AI, software, hardware, startups, consumer tech.",
        "signals": [
            "artificial intelligence", " ai ", "machine learning", "llm", "gpt",
            "gpu", "chip", "semiconductor", "robot", "automation", "saas",
            "software", "startup", "tech", "quantum", "cloud", "cybersecurity",
            "biotech", "deeptech",
        ],
    },
    "business": {
        "niche": "Business & Entrepreneurship",
        "rpm": 14.0,
        "priority": 60,
        "playlist": ("AI, Tech & Innovation Deep-Dives",
                     "In-depth documentaries on AI, technology, business and scientific breakthrough stories."),
        "description": "Companies, founders, M&A, strategy, commerce.",
        "signals": [
            "entrepreneur", "ceo", "founder", "acquisition", "merger", "layoff",
            "strategy", "management", "supply chain", "franchise", "ecommerce",
            "amazon", "google", "apple", "microsoft", "meta", "tesla",
        ],
    },
    "science": {
        "niche": "Science & Innovation",
        "rpm": 13.0,
        "priority": 70,
        "playlist": ("AI, Tech & Innovation Deep-Dives",
                     "In-depth documentaries on AI, technology, business and scientific breakthrough stories."),
        "description": "Research, physics, biology, discovery.",
        "signals": [
            " science ", "scientific", "research", "physics", "chemistry", "biology",
            "genetics", "neuroscience", "laboratory", "experiment", "discovery",
            "climate change", "nuclear fusion", "particle", "genome", "microscope",
            "scientist", "peer review", "breakthrough study",
        ],
    },
    "history": {
        "niche": "History & Documentary",
        "rpm": 9.0,
        "priority": 80,
        "playlist": ("Space, Cosmology & Economic History",
                     "Documentary deep-dives on space, the cosmos and economic history."),
        "description": "Historical events, archaeology, civilizations.",
        "signals": [
            "history", "historical", "ancient", "empire", "archaeology", "medieval",
            "dynasty", "century", "civilization", "artifacts", "pharaoh", "roman",
            "greek", "revolution", "industrial age", "world war", "cold war",
            "viking", "renaissance", "colony", "independence", "battle of",
            "heritage", "archaeologist", "excavation", "documentary",
        ],
    },
    "health": {
        "niche": "Health & Wellness",
        "rpm": 6.5,
        "priority": 90,
        "playlist": ("AI, Tech & Innovation Deep-Dives",
                     "In-depth documentaries on AI, technology, business and scientific breakthrough stories."),
        "description": "Lifestyle wellness, fitness, habits (NOT YMYL medical).",
        "signals": [
            "wellness", "fitness", "longevity", "nutrition", "mental health",
            "sleep", "meditation", "yoga", "lifestyle", "habit",
        ],
    },
    "general": {
        "niche": "Global Trends & Infotainment",
        "rpm": 5.5,
        "priority": 100,
        "playlist": ("Global Trends & Infotainment",
                     "Global trends, culture and infotainment documentaries."),
        "description": "Catch-all for stories that fit no specialized niche.",
        "signals": [],
    },
    "blocked": {
        "niche": "Entertainment",
        "rpm": 1.0,
        "priority": 999,
        "playlist": None,
        "description": "Entertainment/gossip — hard-blocked from production.",
        "signals": [],
    },
}


def _ordered_types() -> List[str]:
    """Audience types sorted by ascending priority (most specific/highest-RPM first)."""
    return sorted(AUDIENCE_TAXONOMY.keys(), key=lambda k: AUDIENCE_TAXONOMY[k]["priority"])


# Backwards-compatible mapping: audience_type -> (niche, rpm) used by callers
# that only need the niche string + planning RPM (e.g. rss_ingestion, fact_retriever).
AUDIENCE_NICHE_MAP: Dict[str, Tuple[str, float]] = {
    k: (v["niche"], v["rpm"]) for k, v in AUDIENCE_TAXONOMY.items()
}


def classify_audience_by_signals(text: str) -> str:
    """
    Classify ``text`` (headline + summary, lower-cased) into an audience_type
    using the taxonomy's keyword signals. Returns "general" if nothing matches.
    Hard blocks (entertainment / YMYL medical) are handled by the caller
    (rss_ingestion.classify_audience_type) before this is invoked.
    """
    lowered = (text or "").lower()
    for aud in _ordered_types():
        signals = AUDIENCE_TAXONOMY[aud].get("signals") or []
        if any(sig in lowered for sig in signals):
            return aud
    return "general"


def niche_for(audience_type: str) -> str:
    return AUDIENCE_TAXONOMY.get(audience_type, AUDIENCE_TAXONOMY["general"])["niche"]


def rpm_for(audience_type: str) -> float:
    return AUDIENCE_TAXONOMY.get(audience_type, AUDIENCE_TAXONOMY["general"])["rpm"]


def description_for(audience_type: str) -> str:
    return AUDIENCE_TAXONOMY.get(audience_type, AUDIENCE_TAXONOMY["general"])["description"]


def playlist_for(audience_type: str) -> Optional[Tuple[str, str]]:
    """Return (title, description) for the themed playlist, or None for blocked."""
    entry = AUDIENCE_TAXONOMY.get(str(audience_type).strip().lower())
    if not entry:
        return None
    return entry.get("playlist")
