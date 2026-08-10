"""Dynamic market/region intelligence for topic selection.

Single source of truth for *which market a run should target* and a per-market
RPM baseline consumed by the revenue forecast. RSS ingestion + TOPSIS scoring
are region-aware: a topic can surface specifically because it is the highest
yield bet for its audience (US vs UK vs Canada vs Australia vs EU vs India) at
this day/time/event — even when it is not #1 for the US alone.

Why per-market (user requirement): the US is the largest ad market, but plenty
of topics/events pay better per-view in the UK, Canada or Australia at a given
moment (RBA/BoE rate decisions, The Ashes/Big Bash, Rio Tinto/BHP, Shopify/
TSX moves, Nifty/Sensex sessions...). The old single scalar region
("global"=US-led 1.0 for everything and "india"=0.25) could never surface those.

Design (deterministic, zero extra LLM/network cost):
  * Markets with a base "locale RPM" multiplier + the L2 region label the rest
    of the pipeline understands ("global" | "india").
  * Topic→market affinity estimated from headline/summary/keywords text tokens
    (country/city names, companies, regulators, currencies, sports/events) plus
    the publisher's ccTLD (source_url domain).
  * Day/time factor: how close the projected publish time falls to each market's
    peak viewing window (UTC minutes-of-day, local-evening / workday daytime).
  * score(market) = locale_rpm * (1 + text_affinity) * (1 + domain_bonus) * window_factor
"""

import os
from typing import Dict, List, Optional, Tuple, Any

# Estimated pipeline launch → YouTube-publish latency (minutes). The region
# scorer projects the publish time from the current clock using this so the
# day/time window factor targets the window the video will ACTUALLY hit.
DEFAULT_RUNTIME_MIN = 120

# Selection-weight policy: "ad revenue in the region has the HIGHEST weight".
# The (topic, market) selection score is:
#     score = α·region_revenue_norm + β·fit + γ·window
# with α > β > γ (defaults 0.55 / 0.30 / 0.15). Tune α via env
# REGION_WEIGHT_REVENUE (0..1); β,γ are derived as the rest of the mass.
REVENUE_WEIGHT = float(os.getenv("REGION_WEIGHT_REVENUE", "0.55"))
ALPHA = max(0.1, min(0.9, REVENUE_WEIGHT))
BETA = round((1.0 - ALPHA) * (2.0 / 3.0), 3)
GAMMA = round((1.0 - ALPHA) / 3.0, 3)
# Reference net-RPM used to normalise region_revenue to [0,1] (max niche rpm ≈ 22).
_REV_REFERENCE = float(os.getenv("REGION_REVENUE_REFERENCE_USD", "16.0"))

# Market table. window = (start, end) UTC minutes-of-day (0..1440); end may
# wrap past 1440 (e.g. AU 22:00-26:00 = 8am-12pm AEST).
MARKETS: Dict[str, Dict[str, Any]] = {
    "us":    {"name": "United States",  "rpm": 1.00, "window": ( 840, 1080), "l2": "global"},
    "uk":    {"name": "United Kingdom", "rpm": 0.97, "window": ( 720,  960), "l2": "global"},
    "ca":    {"name": "Canada",         "rpm": 0.96, "window": ( 870, 1110), "l2": "global"},
    "au":    {"name": "Australia",      "rpm": 0.94, "window": (1320, 1560), "l2": "global"},
    "eu":    {"name": "Continental Europe", "rpm": 0.90, "window": (720, 990), "l2": "global"},
    "india": {"name": "India",          "rpm": 0.25, "window": ( 810,  960), "l2": "india"},
}

# Human readable hint tokens per market (lower-cased substrings). A hit adds
# affinity; multiple distinct hits compound. Company/regulator/currency/sport/
# event references carry most of the "event-driven RPM" signal.
MARKET_HINTS: Dict[str, List[str]] = {
    "us": [
        "united states", "usa", "america", "american", "washington", "wall street",
        "s&p 500", "sp500", "nasdaq", "dow jones", "federal reserve", "the fed",
        "us treasury", "senate", "congress", "white house", "biden", "trump",
        "apple", "microsoft", "windows", "copilot", "nvidia", "google", "alphabet",
        "meta", "facebook", "amazon", "aws", "tesla", "openai", "netflix", "intel",
        "amd", "jpmorgan", "goldman sachs", "bank of america", "citigroup",
        "super bowl", "nfl", "nba", "mlb", "nhl", "silicon valley", "walmart",
        "coca-cola", "pfizer", "mcdonald", "boeing", "starlink", "usd", "dollar",
    ],
    "uk": [
        "united kingdom", "uk ", "britain", "british", "england", "london",
        "scotland", "wales", "ftse", "pound", "sterling", "gbp", "bank of england",
        "boe", "starmer", "westminster", "bbc", "premier league", "arsenal",
        "liverpool", "chelsea", "manchester united", "hsbc", "bp ", "shell",
        "astrazeneca", "glaxosmithkline", "barclays", "lloyds", "rolls-royce",
        "vodafone", "easyjet", "arm holdings", "tesco", "sainsbury", "natwest",
        "ashes", "test cricket", "england cricket", "lords", "county cricket",
        "the open", "wimbledon", "formula one", " grand prix", "british gp",
        "queen elizabeth", "king charles", "duke of edinburgh",
    ],
    "ca": [
        "canada", "canadian", "ottawa", "toronto", "quebec", "alberta", "vancouver",
        "tsx", "tsx venture", "trudeau", "cbc", "rbc", "royal bank of canada",
        "toronto-dominion", "td bank", "scotiabank", "bombardier", "shopify",
        "lululemon", "blackberry", "bank of canada", "hockey", "nhl", "loonie",
        "canadian dollar", "cad", "enbridge", "bc ferries", "stanley cup",
    ],
    "au": [
        "australia", "australian", "sydney", "melbourne", "brisbane", "perth",
        "canberra", "asx", "rba", "reserve bank of australia", "australian dollar",
        "aud", "albanese", "rio tinto", "bhp", "commonwealth bank", "westpac",
        "nab ", "qantas", "coles", "woolworths", "the ashes", "big bash", "bbl",
        "cricket australia", "afl", "nrl", "australian open", "asx 200",
        "telstra", "fortescue", "woodside", "scg", "mcg ", "cricket world cup",
        "t20 world cup", "grand slam", "hobart", "adelaide",
    ],
    "eu": [
        "europe", "european", "eurozone", "frankfurt", "paris", "berlin",
        "amsterdam", "brussels", "dax", "euro stoxx", "euro", "ecb",
        "european central bank", "bundesbank", "germany", "german", "france",
        "french", "netherlands", "dutch", "spain", "italy", "bayer", "siemens",
        "volkswagen", "lvmh", "airbus", "ing group", "arcelormittal", "allianz",
        "daimler", "bmw", "sap ", "hertz",
    ],
    "india": [
        "india", "indian", "mumbai", "delhi", "bengaluru", "new delhi",
        "sensex", "nifty", "nifty 50", "modi", "rupee", "inr", "reliance",
        "jio", "tata", "adani", "infosys", "tcs", "wipro", "hdfc", "icici",
        "sbi", "state bank of india", "ipl", "cricket world cup", "bollywood",
        "sebi", "rbi", "reserve bank of india", "gst", "zomato", "swiggy",
        "paytm", "upi", "vedanta", "bharti airtel", "lic ",
    ],
}

# Publisher ccTLD -> market (source_url host suffix).
TLD_MARKET: Dict[str, str] = {
    "us": "us", "uk": "uk", "ca": "ca", "au": "au", "in": "india",
    "de": "eu", "fr": "eu", "nl": "eu", "es": "eu", "it": "eu", "se": "eu",
    "com": "us", "org": "us", "io": "us", "ai": "us", "net": "us", "gov": "us",
}
# Two-letter second-level domains that are really their own markets.
_COUNTRY_SLD: Dict[str, str] = {
    "co.uk": "uk", "com.au": "au", "com.in": "india", "co.in": "india",
    "org.uk": "uk", "com.br": "eu", "co.za": "india",
}


def _candidate_text(candidate) -> str:
    words = [
        str(getattr(candidate, "headline", "") or ""),
        str(getattr(candidate, "summary", "") or ""),
    ]
    kws = getattr(candidate, "keywords", None) or []
    if isinstance(kws, list):
        words.extend(str(k) for k in kws)
    else:
        words.append(str(kws))
    return " ".join(words).lower()


def _domain_market(candidate) -> Optional[str]:
    url = str(getattr(candidate, "source_url", "") or "").lower()
    if not url:
        return None
    host = url.split("//")[-1].split("/")[0].split("?")[0].split("#")[0]
    labels = host.split(".")
    if len(labels) >= 2:
        sld = ".".join(labels[-2:])
        if sld in _COUNTRY_SLD:
            return _COUNTRY_SLD[sld]
    if labels:
        return TLD_MARKET.get(labels[-1])
    return None


def _text_affinity(candidate, market: str) -> float:
    text = _candidate_text(candidate)
    hits = 0
    for hint in MARKET_HINTS.get(market, []):
        if hint in text:
            hits += 1
    return float(hits)


def window_factor(market: str, projected_utc_minutes: Optional[float] = None) -> float:
    """1.0 when the projected publish time is inside the market's peak window,
    decaying linearly to 0 over ±4h outside it (handles windows crossing
    midnight)."""
    if projected_utc_minutes is None:
        return 1.0
    start, end = MARKETS[market]["window"]
    total = 1440.0
    t = float(projected_utc_minutes) % total
    s = start % total
    e = end % total
    if s < e:
        inside = s <= t <= e
    else:
        inside = (t >= s) or (t <= e)     # window wraps past midnight
    if inside:
        return 1.0
    d = min((t - s) % total, (s - t) % total,
            (t - e) % total, (e - t) % total)
    span = (end - start) % total
    d = min(d, total - span)              # can't be farther than the complement
    return max(0.0, 1.0 - d / 240.0)


def market_scores(
    candidate,
    projected_utc_minutes: Optional[float] = None,
    net_rpm: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per-market selection scores for one candidate. Returns:
    {market: {"score": float, "net_rpm": float, "locale_rpm": float,
              "region_revenue_norm": float, "affinity": float, "domain": bool,
              "window": float}}

    Selection score (revenue-led, per the converged plan):
      fit            = min(1.0, affinity/4) + 0.35·domain      (0..~1, cap 1)
      effective_rev  = net_rpm · locale_rpm · (0.25 + 0.75·fit)  ← market potential
      region_rev_norm= clamp(effective_rev / REGION_REVENUE_REFERENCE_USD, 0, 1)
      score          = ALPHA·region_rev_norm + BETA·fit + GAMMA·window

    Regional ad revenue has the HIGHEST weight (ALPHA > BETA > GAMMA). The market
    affinity feeds the revenue term (a Sensex/RBA story earns almost nothing from
    US viewers, so its effective US revenue collapses and its own market wins
    despite a lower locale RPM). The peak-window factor is deliberately a small,
    SEPARATE nudge (GAMMA) — it must never zero the revenue of the topic's own
    market just because the run started a few hours off-peak."""
    domain = _domain_market(candidate)
    if net_rpm is None:
        net_rpm = _net_rpm_usd(candidate)
    out: Dict[str, Dict[str, Any]] = {}
    for market, info in MARKETS.items():
        affinity = _text_affinity(candidate, market)
        dom = 1.0 if domain == market else 0.0
        wf = window_factor(market, projected_utc_minutes)
        fit = min(1.0, min(1.0, affinity / 4.0) + 0.35 * dom)
        rev_usd = net_rpm * info["rpm"] * (0.25 + 0.75 * fit)
        rev_norm = max(0.0, min(1.0, rev_usd / _REV_REFERENCE))
        score = ALPHA * rev_norm + BETA * fit + GAMMA * wf
        out[market] = {
            "score": round(score, 6),
            "net_rpm": net_rpm,
            "locale_rpm": info["rpm"],
            "region_revenue_norm": round(rev_norm, 6),
            "affinity": affinity,
            "domain": dom == 1.0,
            "window": round(wf, 3),
        }
    return out


def best_market(candidate, projected_utc_minutes: Optional[float] = None, net_rpm: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
    """(winning market, winning MarketRow) for a candidate given the projected
    publish time; the US is the tie-breaker default when everything ties at 0."""
    scores = market_scores(candidate, projected_utc_minutes, net_rpm=net_rpm)
    market = max(scores, key=lambda m: (scores[m]["score"], -_market_rank(m)))
    return market, scores[market]


def candidate_region_profile(
    candidate,
    projected_utc_minutes: Optional[float] = None,
) -> Dict[str, Any]:
    """Full region decision for one candidate: winning market, the revenue-led
    selection score, the REAL expected ad revenue for that market (full forecast:
    views/1000 · net_rpm · locale_rpm · midroll · seasonal — the value TOPSIS's
    8th criterion uses), the L2 region label and a reason. Never raises; falls
    back to a US/global profile when revenue is unavailable."""
    try:
        net_rpm = _net_rpm_usd(candidate)
    except Exception:
        net_rpm = 6.0
    market, row = best_market(candidate, projected_utc_minutes, net_rpm=net_rpm)
    revenue_usd = 0.0
    try:
        from src.engine.monetization_optimizer import monetization_optimizer
        revenue_usd = float(
            monetization_optimizer
            .calculate_revenue_yield(candidate, estimated_runtime_mins=13.0, region=market)
            .get("total_expected_revenue_usd", 0.0)
        )
    except Exception:
        pass
    info = MARKETS[market]
    reason = (
        f"best market {market.upper()} ({info['name']}, locale x{info['rpm']}) "
        f"rev=${revenue_usd:.2f} rpm={net_rpm:.2f} score={row['score']:.3f} "
        f"aff={row['affinity']:.1f} window={row['window']:.2f}"
    )
    return {
        "winner": candidate,
        "market": market,
        "l2_region": info["l2"],
        "score": row["score"],
        "region_revenue_usd": round(revenue_usd, 4),
        "net_rpm": net_rpm,
        "reason": reason,
    }


def _net_rpm_usd(candidate) -> float:
    from src.engine.monetization_optimizer import monetization_optimizer
    return float(monetization_optimizer._net_rpm_usd(candidate))


def _market_rank(market: str) -> int:
    order = {"us": 0, "uk": 1, "ca": 2, "au": 3, "eu": 4, "india": 5}
    return order.get(market, 9)


def select_region_by_day(candidates, projected_utc_minutes: Optional[float] = None) -> Dict[str, Any]:
    """Pick the (topic, market) pair with the best expected yield among the
    candidates — the whole point of region-aware topic selection. Returns:
    {winner, market, l2_region, score, reason, per_candidate}"""
    best = None
    best_key = (-1.0, None)
    for cand in candidates:
        try:
            market, row = best_market(cand, projected_utc_minutes)
        except Exception:
            continue
        key = (row["score"], market)
        if best is None or key > best_key:
            best = (cand, market, row)
            best_key = key
    if best is None:
        cand = candidates[0] if candidates else None
        return {
            "winner": cand, "market": "us", "l2_region": "global",
            "score": 0.0, "reason": "no candidates",
        }
    cand, market, row = best
    info = MARKETS[market]
    reason = (
        f"best market {market.upper()} ({info['name']}, rpm x{info['rpm']}) "
        f"score={row['score']:.3f} affinity={row['affinity']:.1f} "
        f"window={row['window']:.2f} domain={row['domain']}"
    )
    return {
        "winner": cand,
        "market": market,
        "l2_region": info["l2"],
        "score": row["score"],
        "reason": reason,
    }