"""Documentary / investigative storability gate for topic selection.

Rule: a news item is ONLY selectable when OUR automation can build a 6-act
documentary or investigative story out of it. Single-event "direct news" blips —
press-release announcements, price ticks, game scores, Q-result drops — are
culled before they ever reach TOPSIS ranking.

Deterministic-first: the verdict is computed from cheap, offline text signals so
the gate ALWAYS works (no LLM, no network). An optional batched LLM cross-check
(``refine_with_llm``) may fine-tune near-boundary topics ONLY — it can never
resurrect a topic the deterministic gate culled, so LLM drift cannot ship direct
news.

Evergreen synthesized tool topics (``demand_query`` set / ``narrow-synth-*``
ids) pass through untouched: they are not news, and they already survive the
strict demand/opportunity gate in the Fact Retriever.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Conservative default floor: only CLEAR direct-news blips are culled; a
# filmable niche story with weak signals still passes. env DOCUMENTARY_POTENTIAL_FLOOR
FLOOR = float(os.getenv("DOCUMENTARY_POTENTIAL_FLOOR", "0.35"))

# Penalty per direct-news pattern hit and bonus per depth pattern hit.
_DIRECT_PENALTY = 0.18
_DEPTH_BONUS = 0.12
_DATA_BONUS = 0.10

# ── Direct-news / press-release / blip signals ──────────────────────────────
_DIRECT_NEWS_RES = [
    re.compile(r"\b(announc\w*|unveil\w*|launch\w*|introduc\w*|debut\w*|reveal\w*)\b"),
    re.compile(r"\b(opens?\b|inaugurat\w*|premi\w* yes|ships?\b)"),
    re.compile(r"\b(scores?|wins?|beat\w*|defeat\w*|record\b(?! against))\b"),
    re.compile(r"\bq[1-4]\b|\bquarterly (revenue|results|earnings|report)"),
    re.compile(r"\b(rises?|falls?|drops?|gains?|declines?|slips?|surges?|plunges?)"
               r"[^.!?]{0,24}\d+(?:\.\d+)?\s?(?:%|points|basis points)"),
    re.compile(r"\b(?:stock|shares?|price|mkt cap)(?:s?)[^.!?]{0,20}\b(?:up|down|higher|lower|surge|drop)\b"),
    re.compile(r"\bprice target|upgrad\w* to|downgrad\w* to|buy rating|outperform\b"),
    re.compile(r"\b(?:plans? to|set to|will) (?:raise|cut|open|launch|acquire|invest|list)\b"),
    re.compile(r"\b(today announced|announced today|will be available|goes on sale|now shipping|"
               r"available this month|from \$\d|starting at)"),
    # Photo gallery / listicle blips (cannot build 6-act documentary, prevents synthetic photo clickbait)
    re.compile(r"\b(see\s+photos?|magnificent\s+photos?|rare\s+photos?|gallery|photo\s+essay|pictures?\s+of)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s+(photos?|pictures?|images?)\s+of\b", re.IGNORECASE),
]

# ── Documentary / investigative depth signals ───────────────────────────────
_DEPTH_RES = [
    re.compile(r"\b(investigat\w*|probe|inquir\w*|inquest|whistleblow\w*|leak\w*|explosive)\b"),
    re.compile(r"\b(lawsuit|sue\w*|sued|litigation|court|appeal\w*|verdict|ruling|indict\w*)"),
    re.compile(r"\b(fraud|scandal|corruption|embezzle\w*|criminal|tax evas\w*|money laundering)"),
    re.compile(r"\b(regulator\w*|scrutin\w*|antitrust|monopol\w*|compliance|sanction\w*)"),
    re.compile(r"\b(sec |rbi |sebi |boe |bank of england|federal reserve|ecb |the fed|ftc |accia|accc )"),
    re.compile(r"\b(hazard|risk|danger|threat\w*|fallout|backlash|controvers\w*|debat\w*|undermin\w*)"),
    re.compile(r"\b(history|historical|origin\w*|evolution|decades?|timeline|roots|legacy)"),
    re.compile(r"\b(how it works|behind|inside|under the hood|what really|why)\b"),
    re.compile(r"\b(expert\w* warn|analyst\w* warn|study finds|report finds|data shows|research shows)"),
    re.compile(r"\b(supply chain|infrastructure|semiconductor|chip\w*|quantum|electric grid|network effect)"),
    re.compile(r"\b(transition|shift\w*|transformation|revolution|disrupt\w*|reset)"),
]

_NUMERIC_RE = re.compile(
    r"\d+(?:\.\d+)?\s?(?:%|million|billion|trillion|crore|bn|mn)|\$?\d[\d,.]*\s?(?:bn|mn|billion|million)"
)


def _candidate_text(candidate) -> str:
    parts = [
        str(getattr(candidate, "headline", "") or ""),
        str(getattr(candidate, "summary", "") or ""),
    ]
    kws = getattr(candidate, "keywords", None) or []
    if isinstance(kws, list):
        parts.extend(str(k) for k in kws)
    return " ".join(parts).lower()


def _is_evergreen_topic(candidate) -> bool:
    """Synthesized evergreen tool topics pass the storability gate by design."""
    return bool(getattr(candidate, "demand_query", "")) or \
        str(getattr(candidate, "candidate_id", "")).startswith("narrow-synth-")


def score_documentary_potential(candidate) -> Dict[str, Any]:
    """
    Deterministic storability score in [0, 1] for a candidate. Returns
    {score, verdict, direct_hits, depth_hits, has_data, reason}. ``verdict`` is
    "direct_news" when below the floor, else "documentary". Never raises.
    """
    text = _candidate_text(candidate)
    direct_hits = sum(1 for rx in _DIRECT_NEWS_RES if rx.search(text))
    depth_hits = sum(1 for rx in _DEPTH_RES if rx.search(text))

    score = 0.5
    score -= min(4, direct_hits) * _DIRECT_PENALTY
    score += min(6, depth_hits) * _DEPTH_BONUS

    data_hits = len(_NUMERIC_RE.findall(text))
    has_data = data_hits >= 2
    if has_data:
        score += _DATA_BONUS

    words = len(text.split())
    if words < 30:
        score -= 0.10          # thin / shallow item
    elif words >= 120:
        score += 0.10          # rich source material

    score = max(0.0, min(1.0, score))
    verdict = "direct_news" if score < FLOOR else "documentary"
    return {
        "score": round(score, 4),
        "verdict": verdict,
        "direct_hits": direct_hits,
        "depth_hits": depth_hits,
        "has_data": has_data,
        "reason": (
            f"score={score:.2f} vs floor={FLOOR:.2f} "
            f"(direct={direct_hits}, depth={depth_hits}, data={has_data})"
        ),
    }


def gate_candidates(
    candidates: List[Any],
    floor: float = FLOOR,
) -> Tuple[List[Any], List[Tuple[Any, Dict[str, Any]]]]:
    """
    Hard cull of direct-news candidates. Evergreen synthesized topics always
    pass. Returns (kept, culled) where culled is [(candidate, audit), ...].
    All candidates without usable text pass (never crashes the run).
    """
    kept: List[Any] = []
    culled: List[Tuple[Any, Dict[str, Any]]] = []
    for cand in candidates:
        if _is_evergreen_topic(cand):
            kept.append(cand)
            continue
        try:
            audit = score_documentary_potential(cand)
        except Exception:
            kept.append(cand)          # gate must never crash selection
            continue
        if audit["verdict"] == "documentary" or audit["score"] >= floor:
            kept.append(cand)
        else:
            culled.append((cand, audit))
    return kept, culled


# ── Best-effort batched LLM cross-check (opt-in) ────────────────────────────
# Env DOCUMENTARY_LLM_CROSSCHECK=1 enables ONE batched call over the surviving
# shortlist. It may only fine-tune NEAR-BOUNDARY topics; a deterministically
# culled topic is never resurrected. Off by default (keeps runs deterministic).
def refine_with_llm(
    candidates: List[Any],
    floor: float = FLOOR,
) -> Dict[str, str]:
    """Return {candidate_id: "cull"} for near-boundary topics the LLM judges
    clearly direct-news. Deterministically-culled topics are not re-admitted.
    Empty dict on any failure. NEVER raises."""
    if os.getenv("DOCUMENTARY_LLM_CROSSCHECK", "0").strip().lower() not in ("1", "true", "yes"):
        return {}
    try:
        from src.engine.llm_client import LLMClient
        lines = []
        for c in candidates:
            if _is_evergreen_topic(c):
                continue
            audit = score_documentary_potential(c)
            if audit["score"] >= floor and audit["score"] < min(1.0, floor + 0.20):
                lines.append(
                    f"- id={getattr(c, 'candidate_id', '?')} | {str(getattr(c, 'headline', ''))[:120]}")
        if not lines:
            return {}
        client = LLMClient()
        if not client.is_available():
            return {}
        prompt = (
            "You are a documentary story editor. For EACH item, decide whether our "
            "automation can build a deep 6-act DOCUMENTARY or INVESTIGATIVE story "
            "(history, technical depth, stakeholders, risk, data, timeline) OR whether "
            "it is DIRECT NEWS only (single announcement / press release / price tick / "
            "result) with no story underneath. Answer ONLY with a JSON object "
            '{"cull": ["<id>", ...]} listing ids that are clearly direct-news.\n\n'
            + "\n".join(lines)
        )
        result = client.generate_json(
            prompt,
            "You classify story potential. Strict JSON only.",
            route="generate",
        )
        if not isinstance(result, dict):
            return {}
        cull_ids = result.get("cull") or []
        return {str(i): "cull" for i in cull_ids if str(i)}
    except Exception:
        return {}