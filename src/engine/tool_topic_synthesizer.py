"""Narrow tool-topic synthesis for the Fact Retriever.

The RSS pipeline surfaces what is *happening in the news*. Non-news, evergreen,
high-search-demand topics — "[Tool A] vs [Tool B] for [task]", "How to [task]
using [AI tool]", individual tool deep-dives, enterprise/developer tooling
comparisons — never enter the pool through news feeds alone. This module runs an
LLM pass over the day's fresh RSS corpus and emits a small number of such
narrow, target-specific candidate topics, each carrying an explicit
``demand_query`` that is evaluated (YouTube Data API, per-candidate) by the Fact
Retriever.

EVERGREEN STRATEGY / lifecycle:
  * "[Tool A] vs [Tool B] for [specific task]"  — evergreen until a tool dies
  * "How to [specific technical task] using [AI tool]" — tutorial, high volume
  * "[Tool] in-depth review"                     — bread-and-butter evergreen
  * "Enterprise/developer tooling comparison"    — B2B-adjacent, high RPM

The synthesizer only *proposes* topics. Selection economics (demand, TI,
opportunity) are decided by the Fact Retriever's TOPSIS + opportunity gate, so a
proposal with zero real, measurable YouTube demand is dropped (see
``fact_retriever``), not surfaced in the episode.

No network is touched in this module itself except config-free local imports;
the LLM call is made through the shared ``LLMClient``.
"""

import os
import re
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from src.engine.run_budget import run_budget

load_dotenv()

# Curated anchor inventory of high-interest AI/dev tools. The LLM is told to tie
# its proposals to these + the day's corpus, so candidates stay grounded in
# real, citable tools rather than invented names. Kept deliberately small.
TOOL_INVENTORY: List[str] = [
    "ChatGPT", "Claude", "Gemini", "Copilot", "Cursor", "Windsurf",
    "DeepSeek", "Llama", "Mistral", "n8n", "Zapier", "Make",
    "LangChain", "LangGraph", "CrewAI", "AutoGen", "Dify", "Replit",
    "Midjourney", "Runway", "ElevenLabs", "Sora", "Krea",
    "Perplexity", "NotebookLM", "Airtable", "Notion AI",
]

DEFAULT_MAX_SYNTH = int(os.getenv("TOOL_TOPIC_MAX", "4"))

# Archetype templates fed to the LLM (kept crisp so headlines don't drift into
# clickbait/listicle territory that the pipeline's promo filter would reject).
_ARCHETYPES = """\
  A. "[Tool A] vs [Tool B] for [specific task]"  — evergreen side-by-side
  B. "How to [specific technical task] using [AI tool]" — tutorial, high volume
  C. "[Tool] in-depth review" — single-tool deep dive
  D. "Enterprise/developer tooling comparison" — B2B/system-level tradeoffs
"""

_SYSTEM_PROMPT = """\
You are the topic strategist for a tech/AI documentary channel. From the day's
fresh RSS headlines you must propose a small set of NARROW, evergreen,
search-demand-driven topics that news feeds do not surface. Each topic must be
ONE of the four archetypes below (exactly one per topic):
""" + _ARCHETYPES + """
Rules:
- Use real tool names from the provided inventory when relevant, or tools that
  appear in the RSS corpus.
- Topics must be specific + narrow, not "AI in 2026" vibes.
- Choose tasks that people actually search for (coding, data analysis, video,
  automation, content, agents, image generation, enterprise ops).
- Do NOT invent tools, numbers, or facts; stay grounded in tool names + the
  corpus you are given.
- Output STRICT JSON only, no prose, exactly this schema:
{
  "topics": [
    {
      "headline": "2-12 word, CTR-friendly headline appropriate for a produced video",
      "summary": "1-2 sentence summary a documentary could ground narration on",
      "keywords": ["4-8 lowercase keywords for retrieval"],
      "demand_query": "the exact phrase a user would type into YouTube search"
    }
  ]
}
The demand_query must be a natural search phrase (e.g. "claude vs chatgpt for
coding") — never a full sentence you'd use as a headline.
"""


def _truncate_items(items: List[str], n: int = 12) -> str:
    return "\n".join(f"- {s.strip()[:160]}" for s in (items or [])[:n] if s.strip())


def build_prompt(rss_headlines: List[str]) -> str:
    """Assemble the synthesis prompt from the day's RSS corpus + tool inventory."""
    return (
        "Today's fresh RSS headlines (use these as topical grounding):\n"
        f"{_truncate_items(rss_headlines)}\n\n"
        "Tool inventory (prefer these when naming tools):\n"
        + ", ".join(TOOL_INVENTORY)
        + "\n\nPropose up to " + str(DEFAULT_MAX_SYNTH)
        + " narrow topics now."
    )


# A4-messy-quote scrub: the LLM occasionally wraps JSON in ```json fences or
# prepends prose. Shared with the other generators; kept local + defensive.
def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _clean_field(v: Any, default: str = "") -> str:
    return (str(v or default)).strip() or default


def synthesize_tool_topics(
    rss_headlines: List[str],
    llm=None,
    max_topics: int = DEFAULT_MAX_SYNTH,
) -> List[Dict[str, str]]:
    """LLM-propose narrow tool topics from the RSS corpus.

    Returns a list of ``{headline, summary, keywords, demand_query}`` dicts with
    no network/LLM interaction of its own — pure text shaping. Any parse or LLM
    failure returns ``[]`` (caller decides whether to abort or fall through).
    ``llm`` injectable for hermetic tests; defaults to the shared LLMClient.
    """
    if not rss_headlines:
        return []
    prompt = build_prompt(rss_headlines)
    if llm is None:
        from src.engine.llm_client import LLMClient
        llm = LLMClient()
    try:
        data = llm.generate_json(prompt=prompt, system_prompt=_SYSTEM_PROMPT)
    except Exception as e:
        print(f"[ToolTopicSynth] LLM synthesis failed: {e}")
        return []
    if not isinstance(data, dict):
        return []
    topics = data.get("topics") if isinstance(data.get("topics"), list) else []
    out: List[Dict[str, str]] = []
    for t in topics:
        if not isinstance(t, dict):
            continue
        headline = _clean_field(t.get("headline"))
        summary = _clean_field(t.get("summary"))
        demand_query = _clean_field(t.get("demand_query"))
        if not headline or not demand_query:
            continue
        keywords = [str(k).strip().lower() for k in (t.get("keywords") or []) if str(k).strip()]
        if len(keywords) < 2:
            keywords = [w for w in re.split(r"\W+", headline.lower()) if w]
        out.append({
            "headline": headline,
            "summary": summary or f"A focused look at: {headline}.",
            "keywords": keywords,
            "demand_query": demand_query,
        })
        if len(out) >= max_topics:
            break
    return out