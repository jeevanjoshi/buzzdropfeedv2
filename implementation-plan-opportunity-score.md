# Implementation Plan — Opportunity Score, BGM Ducking & Social-Media RAG Exclusion

Consolidated implementation plan for `buzzdropfeedv2` (CSVG pipeline). Confirmed decisions at the end.

## A. Opportunity Score (topic selection)

### A1. `src/schemas/state.py` — `TopicCandidate`
Add two **defaulted** fields (backward-compatible: offline fixtures, tests, checkpoint resume unaffected):
```python
competing_video_count: float = Field(default=0.0, description="Estimated # of competing videos on this topic")
opportunity_score: float = Field(default=0.0, description="Views-per-competitor opportunity = competitor_30d_avg_views / max(1, competing_video_count)")
```
Auto-surfaces in `logs/state_<id>.json` via `tracer.py:66` and `winner.model_dump()` (fact_retriever.py:221).

### A2. New `src/engine/opportunity_score.py`
```python
def compute_opportunity(comp_30d_avg_views: float, competing_count: float) -> float:
    raw = comp_30d_avg_views / max(1.0, competing_count)   # views-per-competitor
    return round(1.0 - 1.0 / (1.0 + math.log1p(raw)), 4)   # monotonic, log-tamed
```
Hermetic unit test (0-competitor edge case, monotonicity).

### A3. `src/engine/rss_ingestion.py`
Persist `competing_video_count` (pass the `_estimate_competing_video_count(...)` result, rss_ingestion.py:446-461) and set `opportunity_score = compute_opportunity(comp_30d, competing)` in **both** candidate paths — main (554-571) and fallback (604-611). `competitor_30d_avg_views` is already stored.

### A4. `src/agents/fact_retriever.py`
- **Hard gate** (REVENUE/SCALE only, fires only when measured — preserves "unknown → TOPSIS decides"):
  ```python
  if cand.opportunity_score > 0 and cand.opportunity_score < OPPORTUNITY_MIN_SCORE:
      continue
  ```
  `OPPORTUNITY_MIN_SCORE` module constant, start ~0.5, tune via calibration (A6).
- **B1 shortlist precise check (first pass):** after TOPSIS, take top-3 → for each, one on-topic `search.list` (100 units) + `videos.list` batch → true avg-views/count → recompute `opportunity_score` → re-rank top-3. Guarded by multi-key rotation + `_can_search` budget (youtube_topic_demand.py:97-110, 167-193) — **silent fall-through** on cap, never fails the run.

### A5. `src/engine/youtube_topic_demand.py`
Sharper `_NICHE_SEED` queries (13-25) so on-topic competitor IDs land in the pools → more meaningful `competitor_30d_avg_views` at **zero extra** search budget (still 1 `search.list`/niche/`REFRESH_DAYS`).

### A6. Calibration script (mirror `test_live_topic_engine.py`)
Fetch a live batch of candidates, print `comp_30d / competing / opportunity / TOPSIS` to tune the hard-gate floor before committing `OPPORTUNITY_MIN_SCORE`.

## B. BGM Loudness — sidechain ducking (recommended)

### B1. `mcp_servers/media_cloud/server.py` (ffmpeg default renderer)
Replace flat volume + loudnorm on the mix (752, 819) with sidechain ducking so music rides down under speech. Extract a shared filter string so the inline (819) and parts-list (751-753) paths can't drift:
```
[1:a]volume=0.5[bgm]
[0:a]asplit=2[voice][sc]
[bgm][sc]sidechaincompress=threshold=0.02:ratio=12:attack=150:release=1200[duck]
[voice][duck]amix=inputs=2:duration=first,alimiter=limit=0.9:level=false,loudnorm=I=-14:TP=-1.5:LRA=11[a]
```

### B2. `src/agents/media_producer.py` (moviepy fallback)
- `BGM_VOLUME = 0.12 → 0.08` (:31).
- MoviePy ducking if feasible; else flat-gain reduction + note that ffmpeg is the ducked path.

### B3. Verifiability
- Expose duck params via env (`BGM_VOLUME`, optional `BGM_SIDECHAIN_THRESHOLD`) so tuning needs no code edits.
- Render a real `--till-upload` clip; confirm narration dominates during speech, music audible only in pauses.

## C. Social-Media RAG Exclusion (both scraper + grounding)

### C1. `src/engine/rag_retriever.py` — blocklist helper
- `_SOCIAL_DOMAINS` set = Reddit, X/Twitter, Facebook, Instagram, LinkedIn, TikTok, YouTube (comments), Quora, Pinterest, Snapchat, Threads, Discord, **Medium, Substack**, forums/BBS.
- `_SOCIAL_SOURCE_RE` regex on source names (e.g. "Reddit · r/…", "X post", "@user", "medium.com/@…").
- `_is_social_source(url, title="", snippet="") -> bool` = domain match **or** source-name/text match.

### C2. Apply at each fetch site
- `search_tavily_facts` — **also return the result `url`** (add to dict, rag_retriever.py:392-395) so it can be domain-filtered too.
- `search_firecrawl_facts`, `search_newsapi_facts` — drop flagged.
- Exa (`external_apis.py`) — drop flagged (has `url`).
- **Google grounding (`grounded_search.py`)** — drop grounding chunks/facts whose `domain`/`source_url`/`source_name` is social (chunk collect 155-167, fact build 241-246).

### C3. Defense-in-depth final filter
At rag_retriever.py:847, run the built `retrieved_facts` lines through `_is_social_source(line)` too (the `[Tavily: …]`/`[Exa: …]` title tags are checked) so nothing survives if a per-source check is missed.

### C4. Verify
- Hermetic unit test: fake search result with `reddit.com` URL, a bare "r/wallstreetbets" title, `medium.com/@x`, and a genuine news domain → assert social excluded, news kept.
- `python run_tests.py` + `python tests/test_smoke.py`; live `--rag scraper` and `--rag hybrid`/`--rag grounded` runs to confirm corpus carries no social sources.

## D. Doc rewrite — `ai-youtube-2000usd-plan.md`
- Drop **ElevenLabs** (§4.10, §5) → free Kokoro TTS on Pi (`audio_edge`).
- Drop **Antigravity CLI** (§4.11, §5) → existing `src/agents/` orchestrator.
- **Scheduling (§6, §4.15)** → marked out-of-scope for now.
- Replace the unimplementable per-topic "avg views ÷ #competing" with the implemented `opportunity_score` (pool-derived) + B1 shortlist check + quota math (10k units/day; search=100, upload=1600).
- §2 templates: keep, flagged aspirational (they hit the promo block, rss_ingestion.py:66-67).
- Add **audio-mix note** (sidechain ducking + tuning) and **source-policy note** (social platforms excluded from RAG, both scraper and grounded).

---

## Files touched
| File | Change |
|---|---|
| `src/schemas/state.py` | +2 defaulted fields on `TopicCandidate` |
| `src/engine/opportunity_score.py` | **new** — `compute_opportunity` |
| `src/engine/rss_ingestion.py` | persist competing count; compute+set opportunity on both candidate paths |
| `src/agents/fact_retriever.py` | pre-TOPSIS opportunity hard gate; B1 shortlist precise check |
| `src/engine/youtube_topic_demand.py` | sharper `_NICHE_SEED` queries |
| `src/engine/rag_retriever.py` | social blocklist helper + per-source filters + C3 final filter + Tavily URL capture |
| `src/engine/external_apis.py` | Exa social filter |
| `src/engine/grounded_search.py` | drop social grounding chunks/facts |
| `mcp_servers/media_cloud/server.py` | sidechain-ducated BGM mix |
| `src/agents/media_producer.py` | `BGM_VOLUME` reduction (+ MoviePy ducking if feasible) |
| `ai-youtube-2000usd-plan.md` | doc corrections (drop Antigravity/ElevenLabs/scheduling; document real discovery + opportunity score + audio/source notes) |
| `tests/` | new hermetic unit tests (`compute_opportunity`, social exclusion) |

## Sequence
Schema → opportunity fn + test → rss wiring → pool seeds → hard gate → B1 shortlist → calibration → BGM ducking → social exclusion (C1-C3) → social tests → doc rewrite → smoke tests + sample render + ruff.

## Confirmed decisions
- **`OPPORTUNITY_MIN_SCORE` = hard gate** (REVENUE/SCALE only, only when measured).
- **B1 shortlist precise check in the first pass.**
- **BGM: sidechain ducking** (recommended) in ffmpeg; flat `BGM_VOLUME` reduction for moviepy.
- **Block Medium/Substack/forums** in addition to core social platforms.
- **Social exclusion applies to both scraper path and Google grounding (`--rag grounded`/`hybrid`).**
