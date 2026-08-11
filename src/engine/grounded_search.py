"""Google Search Grounding research pass.

POC for replacing the fragile 5-scraper RAG crawl layer in
``rag_retriever.build_rag_knowledge_pack`` with one Gemini call that uses the
``googleSearch`` tool to hit Google's live index and returns machine-readable
citations (``groundingMetadata`` / ``groundingChunks[].web.uri``). This is the
documented two-stage integration:

  1. A grounded research pass emits a JSON fact list with real source URLs.
  2. The existing story_designer / Observer consume that corpus unchanged.

Only enabled when ``GOOGLE_CLOUD_PROJECT`` + ADC credentials are present; the
OpenRouter / scraper path in ``rag_retriever`` remains the non-grounded
fallback. See the "Grounded search" notes in the source history.
"""

import os
import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from src.engine.run_budget import run_budget

load_dotenv()

# Social/community platforms excluded from grounding chunks/facts. These host
# low-trust, non-citable opinion that pollutes the corpus. Mirrors
# rag_retriever._SOCIAL_DOMAINS so the grounded path drops the same domains.
_SOCIAL_DOMAINS = frozenset({
    "reddit.com", "redd.it", "x.com", "twitter.com", "t.co", "facebook.com",
    "fb.com", "instagram.com", "linkedin.com", "tiktok.com", "youtube.com",
    "youtu.be", "quora.com", "pinterest.com", "snapchat.com", "threads.net",
    "discord.com", "medium.com", "substack.com",
})


def _is_social_domain(domain: str = "") -> bool:
    """True if a grounding chunk's domain (or a URL's host) is a social platform."""
    host = (domain or "").strip().lower().lstrip("www.").split(":")[0]
    if host:
        if any(d == host or host.endswith("." + d) for d in _SOCIAL_DOMAINS):
            return True
    return False


def _is_social_url(url: str = "") -> bool:
    if not url:
        return False
    try:
        return _is_social_domain(urlparse(url).netloc)
    except Exception:
        return False

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_OK = True
except Exception:  # pragma: no cover - dependency not installed
    _GENAI_OK = False

RESEARCH_PROMPT = """\
You are a strict news researcher. Using Google Search grounding, research the
following topic and return a JSON object in EXACTLY this shape (no prose, no
markdown fences):

{{
  "facts": [
    {{
      "fact": "one verifiable statement with exact figures as published",
      "source_url": "https://... original URL backing this fact",
      "source_name": "publisher/domain, e.g. Reuters",
      "year": 2025
    }}
  ]
}}

Rules:
- Facts about the topic must be grounded in the retrieved search results. Do
  NOT invent numbers, percentages, names, or dates.
- Preserve EXACT figures and units as published (e.g. "US$55.11 billion",
  "19.29%") -- never truncate or round into a changed value.
- Record "year" as the publication year of the source article.
- Aim for 10-18 diverse facts from DIFFERENT sources, not a single outlet.
- 3-5 of the facts should be narrowly on-topic; the rest can be broader
  background as long as they are still relevant to the topic.

Topic to research:
{{
  "headline": "{headline}",
  "summary": "{summary}",
  "keywords": {keywords}
}}
"""


def _client() -> Optional[Any]:
    """Build a genai client for the ``google_search`` grounding tool.

    Two supported auth paths (either grounds the ``google_search=GoogleSearch()``
    tool):
      * Gemini Enterprise Agent Platform (the path at
        docs.cloud.google.com/gemini-enterprise-agent-platform/.../grounding-with-google-search):
        requires GOOGLE_GENAI_USE_ENTERPRISE=True + GOOGLE_CLOUD_PROJECT + ADC,
        built with ``Client(http_options=HttpOptions(api_version="v1"))`` (NOT
        project=/location= args).
      * Gemini Developer API (ai.google.dev): requires GEMINI_API_KEY / GOOGLE_API_KEY.
    Enterprise is preferred when configured (matches this repo's .env); api key is
    the fallback. Returns None when no auth is configured.
    """
    if not _GENAI_OK:
        print("[GroundedSearch] google-genai SDK not installed (python -m pip install google-genai).")
        return None

    enterprise = os.getenv("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in ("1", "true", "yes")
    if enterprise:
        if not os.getenv("GOOGLE_CLOUD_PROJECT"):
            print("[GroundedSearch] Enterprise mode set but GOOGLE_CLOUD_PROJECT missing; grounding unavailable.")
            return None
        try:
            from google.genai import types as _t
            return genai.Client(http_options=_t.HttpOptions(api_version="v1"))
        except Exception as e:
            print(f"[GroundedSearch] Failed to init enterprise client: {e}")
            return None

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            return genai.Client(api_key=api_key)
        except Exception as e:
            print(f"[GroundedSearch] Failed to init api-key client: {e}")
            return None

    print("[GroundedSearch] No grounding auth configured "
          "(need GOOGLE_GENAI_USE_ENTERPRISE+project+ADC, or GEMINI_API_KEY); grounding unavailable.")
    return None


def _facets(headline: str, summary: str, keywords: Optional[List[str]] = None,
            category: str = "") -> List[str]:
    """Build a small set of specific, answerable research questions (facets) from
    the topic. Keeping each query SHORT and direct is what makes Google attach
    ``grounding_chunks`` to the answer (a single 10-18-fact synthesis prompt
    fires searches but returns no inline citations).

    The facet set is CATEGORY-AWARE: market/competitor/forecast questions only
    make sense for business/finance topics. For History, Science, Health,
    Geopolitics etc. those generic finance facets make Google grounding return
    off-topic "Contract Research Organizations market $X" snippets that pollute
    the corpus, so they are replaced with domain-appropriate questions.
    """
    kw = list(keywords or [])
    primary = kw[0].strip() if kw else "the topic"
    topic = headline.strip().strip(".").strip()[:90] or primary
    summary_short = summary.strip().strip(".").strip()[:120] or topic

    # Core facets are universal: latest facts, developments, background, risks.
    facets = [
        f"Give the latest key facts and figures about: {topic}.",
        f"Summarize the latest developments behind: {summary_short}. Give specific facts.",
        f"What is the recent historical background of {topic}? Give several facts.",
        f"What are the main risks, concerns, or criticisms about {topic}? Give several facts.",
    ]

    # Finance/business topics get market+competitor+forecast depth; everything
    # else gets domain-appropriate questions instead (never "market cap/outlook").
    _BUSINESS = ("Finance", "Trading", "Economics", "Business", "Startup")
    if any(b.lower() in category.lower() for b in _BUSINESS):
        facets += [
            f"What are the exact financial figures (revenue, market cap, growth) for {primary} as of now?",
            f"What products, models, or launches did {primary} recently announce? Give specific details.",
            f"Which companies or competitors are most affected by {topic}? Give specific facts.",
            f"What is the near-term outlook or forecast for {topic}? Give several facts.",
        ]
    else:
        facets += [
            f"What are the key events, people, or findings central to {topic}? Give specific, dated facts.",
            f"What evidence or studies support the claims about {topic}? Cite specific sources and numbers.",
        ]
    return facets


def _split_fact_sentences(text: str, min_words: int = 6) -> List[str]:
    """Split a grounded answer into standalone factual sentences, dropping the
    model's own instructional preamble (e.g. "Here are 4-6 detailed factual
    points:") so it never leaks into the corpus as a bogus fact."""
    text = (text or "").strip()
    if not text:
        return []
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    out = []
    for s in sents:
        s = re.sub(r'^[-*\d.\s]+', '', s).strip()
        low = s.lower().lstrip(":")
        if re.match(r'^(here are|here\'s|sure[,!]?|okay[,!]?|certainly[,!]?|'
                    r'using google search|certainly[,!]?|let me research|i will research|'
                    r'as of my|based on google search results,[^.]{0,40}here)',
                    low, re.IGNORECASE):
            continue
        if len(s.split()) >= min_words:
            out.append(s)
    return out


def _collect_chunks(gm) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    if gm is None:
        return chunks
    for c in (getattr(gm, "grounding_chunks", None) or []):
        w = getattr(c, "web", None)
        if w is not None:
            domain = getattr(w, "domain", "") or ""
            if _is_social_domain(domain):
                continue
            chunks.append({
                "title": getattr(w, "title", ""),
                "uri": getattr(w, "uri", ""),
                "domain": domain,
            })
    return chunks


def grounded_research(
    headline: str,
    summary: str,
    keywords: Optional[List[str]] = None,
    model: Optional[str] = None,
    category: str = "",
) -> Optional[Dict[str, Any]]:
    """Run a per-facet Google-Search-grounded research pass.

    Each facet is a SHORT, direct grounded question (per the ADK/docs guidance,
    search grounding returns ``grounding_chunks`` per cited answer; a big JSON
    synthesis prompt does not). Each facet's answer is split into facts that are
    attributed to the citation(s) that facet actually returned.

    Returns ``{"facts": [...], "grounding_chunks": [...], "web_search_queries":
    [...], "raw_text": str, "model": str}`` — where every ``fact`` carries a real
    ``source_url``/``source_name`` — or ``None`` if grounding is not configured.
    """
    if not _GENAI_OK:
        print("[GroundedSearch] google-genai SDK not installed (python -m pip install google-genai).")
        return None
    client = _client()
    if client is None:
        return None
    model = model or os.getenv("GROUNDING_MODEL", "gemini-2.5-flash")

    all_chunks: List[Dict[str, str]] = []
    web_queries: List[str] = []
    facts: List[Dict[str, Any]] = []
    texts: List[str] = []
    seen_facts: set = set()
    topic_terms = ", ".join((keywords or [])[:6]) or headline[:40]

    for q in _facets(headline, summary, keywords, category):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=(
                    "Using Google Search, produce 4-6 DETAILED factual points about the topic. "
                    "Each point: 1-2 sentences, include an exact number or figure where possible, "
                    "and explicitly reference the topic. Use the key terms: "
                    f"{topic_terms}. Write as concise news statements.\n\n"
                ) + q,
                config=genai_types.GenerateContentConfig(
                    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                    temperature=1.0,
                ),
            )
        except Exception as e:
            print(f"[GroundedSearch] facet call failed: {str(e)[:200]}")
            continue

        run_budget.record_grounding()
        raw_text = getattr(resp, "text", "") or ""
        texts.append(raw_text)
        gm = None
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            gm = getattr(candidates[0], "grounding_metadata", None)
        if gm is not None:
            web_queries += list(getattr(gm, "web_search_queries", None) or [])
        fchunks = _collect_chunks(gm)
        all_chunks += fchunks

        # One cited fact per sentence; cycle through the facet's real citations so
        # multiple facts per facet are kept (each still attributed to a source the
        # facet's search actually returned). Dedupe by fact text, not URL.
        for j, sent in enumerate(_split_fact_sentences(raw_text)):
            src = fchunks[j % len(fchunks)] if fchunks else {}
            if _is_social_url(src.get("uri", "")):
                continue
            key = sent.lower()[:90]
            if key in seen_facts:
                continue
            seen_facts.add(key)
            facts.append({
                "fact": sent,
                "source_url": src.get("uri", ""),
                "source_name": src.get("domain") or src.get("title", ""),
                "year": None,
            })
        if len(facts) >= 26:
            break

    print(f"[GroundedSearch] {model}: {len(facts)} facts, {len(all_chunks)} chunks, {len(web_queries)} queries.")
    return {
        "facts": facts,
        "grounding_chunks": all_chunks,
        "web_search_queries": web_queries,
        "raw_text": "\n".join(texts),
        "model": model,
    }


def _extract_facts(raw_text: str) -> List[Dict[str, Any]]:
    """Parse the researcher's JSON fact list defensively."""
    if not raw_text:
        return []
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text, strict=False)
    except Exception:
        import re
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0), strict=False)
        except Exception:
            return []
    facts = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(facts, list):
        return []
    cleaned = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        fact = (f.get("fact") or "").strip()
        if not fact:
            continue
        cleaned.append({
            "fact": fact,
            "source_url": (f.get("source_url") or "").strip(),
            "source_name": (f.get("source_name") or "").strip(),
            "year": f.get("year"),
        })
    return cleaned


def corpus_from_facts(facts: List[Dict[str, Any]], max_lines: int = 22) -> str:
    """Build fact_corpus lines in the same ``Source:`` format the Observer and
    ``assess_corpus_sufficiency`` already parse for source diversity."""
    lines = []
    for f in facts[:max_lines]:
        src = f.get("source_name") or ""
        url = f.get("source_url") or ""
        source_tag = src if src else (url.split("/")[2] if url.startswith("http") else src)
        year = f.get("year")
        suffix = f" (Source: {source_tag})" + (f" (year: {year})" if year else "")
        lines.append(f"• {f['fact']}{suffix}")
    return "\n".join(lines)


def build_grounded_knowledge_pack(
    headline: str,
    summary: str,
    keywords: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Stage-1 research pass that yields a RAG-pack-shaped dict for the existing
    ``story_designer`` / ``assess_corpus_sufficiency`` / Observer to consume.

    Returns ``None`` when grounding is unavailable so callers fall back to the
    non-grounded scraper path.
    """
    category = _category_guess(headline, summary, keywords)
    result = grounded_research(headline, summary, keywords, model=model, category=category)
    if not result or not result["facts"]:
        print("[GroundedSearch] No grounded facts produced; falling back to scraper path.")
        return None
    if not result.get("grounding_chunks"):
        # Refuse to ship a "grounded" corpus that has no citations from the search
        # tool — otherwise the LLM's JSON facts are ungrounded and could be
        # fabricated. Fail closed to the scraper path loudly instead.
        print("[GroundedSearch] Facts produced but ZERO grounding chunks (googleSearch not "
              "returning citations) — refusing ungrounded output; falling back to scraper path.")
        return None
    fact_corpus = corpus_from_facts(result["facts"])
    fact_lines = [f for f in fact_corpus.splitlines() if f.strip()]
    sources = sorted({f.get("source_name") or "" for f in result["facts"] if f.get("source_name")})
    pack = {
        "topic_headline": headline,
        "category": category,
        "summary": summary,
        "keywords": keywords or [],
        "rag_mode": "google_search_grounded",
        "ground_truth_block": fact_corpus,
        "rag_retrieved_context": fact_corpus,
        "selected_article": "",
        "fact_corpus": fact_corpus,
        "full_rag_context_text": (
            f"TOPIC CATEGORY: {category}\n"
            f"RAG EXECUTION MODE: GOOGLE_SEARCH_GROUNDED\n"
            f"HEADLINE: {headline}\n"
            f"SUMMARY: {summary}\n\n"
            f"GROUNDED FACTS (googled + cited):\n{fact_corpus}\n\n"
            f"GROUNDING CHUNKS (for fact-audit verification):\n"
            + ("\n".join(f"— {c['title']} ({c['domain']}) {c['uri']}" for c in result["grounding_chunks"]))
            + "\n\nWEB SEARCH QUERIES: " + ", ".join(result["web_search_queries"])
        ),
    }
    pack["_grounding_meta"] = {
        "sources": sources,
        "grounding_chunks": result["grounding_chunks"],
        "web_search_queries": result["web_search_queries"],
        "model": result["model"],
        "fact_count": len(fact_lines),
    }
    return pack


def _category_guess(headline: str, summary: str, keywords: Optional[List[str]]) -> str:
    combined = f"{headline} {summary} {' '.join(keywords or [])}".lower()
    if any(w in combined for w in ["compound interest", "401k", "ira", "retirement", "budgeting", "credit score", "financial literacy", "how to invest", "index fund"]):
        return "Personal Finance Education"
    if any(w in combined for w in ["history", "ancient", "empire", "archaeology", "medieval", "civilization", "pharaoh", "roman", "viking", "renaissance", "world war"]):
        return "History & Documentary"
    if any(w in combined for w in ["space", "nasa", "planet", "rocket", "star", "physics", "science"]):
        return "Space & Scientific Innovation"
    # Finance beats technology: "Nvidia stock surges / crypto market / bank revenue"
    # is an economics topic even when it names a tech company.  Failing to route
    # these to the finance branch would strip the market/forecast facets they need.
    if any(w in combined for w in ["fed", "market", "stock", "trading", "crypto", "bank", "inflation", "revenue", "dollar", "earnings", "growth"]):
        return "Global Economics & Finance"
    if any(w in combined for w in ["ai", "chatgpt", "software", "tech", "chip", "nvidia", "cloud", "seo", "app"]):
        return "Technology & Artificial Intelligence"
    if any(w in combined for w in ["war", "election", "policy", "country", "president", "government"]):
        return "Geopolitics & World Affairs"
    return "Global Trends & Cultural Infotainment"


def is_grounding_available() -> bool:
    """True when the google-genai SDK and a Vertex project are configured."""
    return _GENAI_OK and bool(os.getenv("GOOGLE_CLOUD_PROJECT"))