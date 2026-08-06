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
fallback. See http://.../grounding-with-google-search and
``debugging_060820260057.md`` (section "EXPLORED 2026-08-06").
"""

import os
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

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
    """Build a Vertex/Enterprise genai client from ADC, or None if absent."""
    if not _GENAI_OK:
        return None
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        print("[GroundedSearch] GOOGLE_CLOUD_PROJECT not set; grounding unavailable.")
        return None
    try:
        return genai.Client(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        )
    except Exception as e:
        print(f"[GroundedSearch] Failed to init genai client: {e}")
        return None


def grounded_research(
    headline: str,
    summary: str,
    keywords: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run one Google-Search-grounded research pass.

    Returns ``{"facts": [...], "grounding_chunks": [...], "web_search_queries":
    [...], "raw_text": str, "model": str}`` or ``None`` if grounding is not
    configured / the call or JSON parse fails.
    """
    if not _GENAI_OK:
        print("[GroundedSearch] google-genai SDK not installed (python -m pip install google-genai).")
        return None
    client = _client()
    if client is None:
        return None
    model = model or os.getenv("GROUNDING_MODEL", "gemini-2.5-flash")
    kws = json.dumps(keywords or [])
    prompt = RESEARCH_PROMPT.format(headline=headline, summary=summary, keywords=kws)
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                temperature=1.0,
            ),
        )
    except Exception as e:
        print(f"[GroundedSearch] generate_content failed: {str(e)[:300]}")
        return None

    raw_text = getattr(response, "text", "") or ""
    gm = None
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        gm = getattr(candidates[0], "grounding_metadata", None)

    chunks: List[Dict[str, str]] = []
    web_queries: List[str] = []
    if gm is not None:
        web_queries = list(getattr(gm, "web_search_queries", None) or [])
        for c in (getattr(gm, "grounding_chunks", None) or []):
            w = getattr(c, "web", None)
            if w is not None:
                chunks.append({
                    "title": getattr(w, "title", ""),
                    "uri": getattr(w, "uri", ""),
                    "domain": getattr(w, "domain", ""),
                })
    print(f"[GroundedSearch] {model}: {len(chunks)} chunks, {len(web_queries)} queries.")

    facts = _extract_facts(raw_text)
    return {
        "facts": facts,
        "grounding_chunks": chunks,
        "web_search_queries": web_queries,
        "raw_text": raw_text,
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


def corpus_from_facts(facts: List[Dict[str, Any]], max_lines: int = 12) -> str:
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
    result = grounded_research(headline, summary, keywords, model=model)
    if not result or not result["facts"]:
        print("[GroundedSearch] No grounded facts produced; falling back to scraper path.")
        return None
    fact_corpus = corpus_from_facts(result["facts"])
    fact_lines = [f for f in fact_corpus.splitlines() if f.strip()]
    sources = sorted({f.get("source_name") or "" for f in result["facts"] if f.get("source_name")})
    pack = {
        "topic_headline": headline,
        "category": _category_guess(headline, summary, keywords),
        "summary": summary,
        "keywords": keywords or [],
        "rag_mode": "google_search_grounded",
        "ground_truth_block": fact_corpus,
        "rag_retrieved_context": fact_corpus,
        "selected_article": "",
        "fact_corpus": fact_corpus,
        "full_rag_context_text": (
            f"TOPIC CATEGORY: {_category_guess(headline, summary, keywords)}\n"
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
    if any(w in combined for w in ["ai", "chatgpt", "software", "tech", "chip", "nvidia", "cloud", "seo", "app"]):
        return "Technology & Artificial Intelligence"
    if any(w in combined for w in ["fed", "market", "stock", "trading", "crypto", "bank", "inflation", "revenue", "dollar"]):
        return "Global Economics & Finance"
    if any(w in combined for w in ["space", "nasa", "planet", "rocket", "star", "physics", "science"]):
        return "Space & Scientific Innovation"
    if any(w in combined for w in ["war", "election", "policy", "country", "president", "government"]):
        return "Geopolitics & World Affairs"
    return "Global Trends & Cultural Infotainment"


def is_grounding_available() -> bool:
    """True when the google-genai SDK and a Vertex project are configured."""
    return _GENAI_OK and bool(os.getenv("GOOGLE_CLOUD_PROJECT"))