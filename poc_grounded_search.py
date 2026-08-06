#!/usr/bin/env python3
"""POC: Google Search Grounded research pass for CSVG.

Runs the two-stage design from ``debugging_060820260057.md`` (section
"EXPLORED 2026-08-06"): a Gemini ``googleSearch``-grounded research call that
emits a cited fact list (stage 1), then feeds a RAG-pack-shaped corpus
(stage 2) that the existing ``assess_corpus_sufficiency`` / ``story_designer``
can consume unchanged.

Usage (from repo root, venv active):
    python poc_grounded_search.py                      # default China-AI-Africa topic
    python poc_grounded_search.py --query "..."        # custom query
    python poc_grounded_search.py --model gemini-2.5-flash
    python poc_grounded_search.py --write-logs         # also dump JSON to logs/

Requires ADC + GOOGLE_CLOUD_PROJECT for Vertex grounding
(see debugging md re: gcloud auth application-default login).

This script is a POC and does NOT touch the production pipeline.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.engine.grounded_search import (  # noqa: E402
    build_grounded_knowledge_pack,
    _GENAI_OK,
)

DEFAULT_HEADLINE = "How China's A.I. Is Surging Across Africa"
DEFAULT_SUMMARY = (
    "Chinese AI companies are expanding rapidly across African markets, "
    "deploying open-source models, data centers, and smartphone-based AI tools "
    "to capture a young, mobile-first user base amid US export controls."
)
DEFAULT_KEYWORDS = ["China", "AI", "Africa", "open-source models", "technology", "surge"]


def main():
    ap = argparse.ArgumentParser(description="Google Search Grounding POC")
    ap.add_argument("--query", help="Input prompt for the research pass.")
    ap.add_argument("--model", default=os.getenv("GROUNDING_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--write-logs", action="store_true", help="Dump result JSON to logs/")
    args = ap.parse_args()

    if not _GENAI_OK:
        print("FATAL: google-genai SDK not installed. Run: pip install 'google-genai'")
        sys.exit(1)
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        print("FATAL: GOOGLE_CLOUD_PROJECT not set. Set it and run: gcloud auth application-default login")
        sys.exit(1)

    print(f"[POC] Model: {args.model}")
    print(f"[POC] GOOGLE_CLOUD_PROJECT={os.getenv('GOOGLE_CLOUD_PROJECT')} "
          f"LOCATION={os.getenv('GOOGLE_CLOUD_LOCATION') or 'global'}")
    print("-" * 70)

    if args.query:
        print("[POC] Running raw grounded query (no schema):")
        from google import genai
        from google.genai import types
        client = genai.Client(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        )
        resp = client.models.generate_content(
            model=args.model,
            contents=args.query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=1.0,
            ),
        )
        print("\n--- RESPONSE ---")
        print(resp.text)
        gm = getattr(getattr(resp, "candidates", [None])[0], "grounding_metadata", None)
        if gm:
            print("\n--- GROUNDING CHUNKS ---")
            for c in (gm.grounding_chunks or [])[:10]:
                print(f"  - {getattr(c.web,'title','')} | {getattr(c.web,'uri','')} | {getattr(c.web,'domain','')}")
        return

    print(f"[POC] Headline: {DEFAULT_HEADLINE}")
    pack = build_grounded_knowledge_pack(
        headline=DEFAULT_HEADLINE,
        summary=DEFAULT_SUMMARY,
        keywords=DEFAULT_KEYWORDS,
        model=args.model,
    )
    if not pack:
        print("FATAL: grounded research produced no facts (see errors above).")
        sys.exit(1)

    print("\n===== FACT CORPUS (= Stage-2 RAG pack) =====")
    print(pack["fact_corpus"])
    print("\n===== SOURCE DIVERSITY =====")
    meta = pack["_grounding_meta"]
    print(f"  {meta['fact_count']} facts from {len(meta['sources'])} sources:")
    for s in sorted(meta["sources"]):
        print(f"    - {s}")
    print(f"\n===== GROUNDING CHUNKS (audit URLs) =====")
    for c in meta["grounding_chunks"][:12]:
        print(f"  - {c['title']} | {c['domain']} | {c['uri']}")
    print(f"\n===== WEB SEARCH QUERIES =====")
    for q in meta["web_search_queries"]:
        print(f"  * {q}")

    if args.write_logs:
        logs = REPO_ROOT / "logs"
        logs.mkdir(exist_ok=True)
        out = logs / "poc_grounded_research.json"
        out.write_text(json.dumps(pack, indent=2, default=str))
        print(f"\n[POC] Wrote {out}")

    print("\n[POC] SUCCESS")


if __name__ == "__main__":
    main()