# HY3 Improvements Suggestion (code audit)

Scope: full audit of src/, mcp_servers/, tests/, and top-level scripts.
Enforced gate (AGENTS.md) `ruff --select F821,F822,F823` PASSES; hermetic
suite `run_tests.py` PASSES (27/27). Findings below are beyond that gate.

## 1. CRITICAL — Security

### 1.1 SSH command injection on the Pi  (src/agents/publisher.py:113-121)
`title_json = json.dumps(title).replace('"','\\"')` only escapes double
quotes. The whole remote command is a single `ssh user@host "<cmd>"` arg, so
a title containing `$`, backticks, `$(...)`, `;`, `|`, `&&` yields arbitrary
command execution on the Pi as PI5_USER. `title` is LLM/RSS-derived (external).
FIX: `shlex.quote(title_json)` (or pass `--title` as a separate argv element
to `ssh`, avoiding shell interpretation).

## 2. HIGH — Robustness / logic

### 2.1 ffprobe has no timeout  (src/engine/quality_verifier.py:322)
`subprocess.run(cmd, capture_output=True, text=True, check=True)` with no
`timeout`, while sibling calls (lines 381/404/455/495) all set `timeout=60`.
A stalled/corrupt media file hangs the whole run. FIX: add `timeout=60`.

### 2.2 yt-dlp calls have no timeout  (src/engine/youtube_video_verifier.py:87, 434)
Both `subprocess.run` calls can hang indefinitely in an unattended pipeline.
FIX: add `timeout=` (e.g. 120-300s).

### 2.3 SSRF in article scraper  (src/engine/rag_retriever.py:788)
`_scrape_selected_article` only checks `url.startswith("http")` then
`urllib.request.urlopen(req, timeout=12)`. Source URLs are RSS-controlled, so
the pipeline will fetch internal hosts / cloud metadata (169.254.169.254,
localhost). FIX: https-only scheme allowlist + block private/link-local IPs.

### 2.4 API keys embedded in URL query strings
- src/engine/external_apis.py:53  `api_token={self.marketaux_key}`
- src/engine/external_apis.py:108,130  `apikey={self.alpha_vantage_key}`
- src/engine/rag_retriever.py:652  `apiKey={api_key}` (NEWSAPI_KEY)
Query-string secrets are captured in upstream access logs/proxies. FIX: move
to `Authorization`/header-based auth or POST body (the code already does this
for Exa/Firecrawl/Tavily).

### 2.5 Concurrency: unguarded module-level refresh + logger
- src/engine/analytics_feedback.py:141  module-level `refresh()` has no lock
  (only `AnalyticsFeedbackStore.refresh()` locks). Races the publisher's
  background `analytics_feedback.refresh(...)` (publisher.py:450) on
  logs/analytics_feedback.json.
- src/engine/logger.py:100  every `logger.log` opens the log file concurrently
  from multiple threads (LLM client, analytics thread, budget thread) -> torn
  JSON lines. FIX: add a lock or a `QueueHandler`.

## 3. MEDIUM — Dead code / orphaned modules (never imported)

### 3.1 Orphaned whole modules (confirmed zero external references)
- src/engine/linucb_bandit.py  (LinUCBContextualBandit + all methods)
- src/engine/script_pacing_engine.py  (ScriptPacingEngine + methods)
- src/engine/social_signals.py  (calculate_social_hype_multiplier,
  calculate_youtube_vph_velocity)
- src/engine/trend_velocity.py  (calculate_ema_trend_velocity,
  calculate_zscore_anomaly)

### 3.2 Dead functions/methods inside used modules
- src/engine/quality_verifier.py:78 verify_gate2_script_to_tts (gate never called)
- src/engine/quality_verifier.py:98 verify_gate3_tts_to_subtitles
- src/engine/rag_retriever.py:546 search_duckduckgo_facts (removed from registry)
- src/engine/rag_retriever.py:487 set_grounded
- src/engine/region_intelligence.py:295 select_region_by_day
- src/engine/space_cinema_apis.py:21 fetch_nasa_space_imagery
- src/engine/space_cinema_apis.py:67 fetch_tmdb_movie_box_office
- src/engine/api_ninjas.py:48 fetch_interesting_facts
- src/engine/gif_retriever.py:41 search_tenor_gif
- src/engine/bertopic_engine.py:91 track_temporal_sentiment_shift
- src/engine/api_usage.py:346 run_costs ; :394 live_total_usd
- src/engine/media_budget.py:67 reset_month
- src/engine/monetization_optimizer.py:215 rank_candidates_by_revenue_pareto
- src/engine/nano_banana.py:1001 generate_video_thumbnail_art
- src/engine/grounded_search.py:319 _extract_facts ; :450 is_grounding_available
- src/engine/tool_topic_synthesizer.py:107 _extract_json_object
- src/agents/media_producer.py:80 _has_numeric_claim
- src/schemas/seed_distribution.py:27 DiscordEmbedField ; :32 DiscordWebhookPayload
- src/schemas/state.py:194 GlobalState.is_monetised

### 3.3 Deprecated / orphaned files
- dashboard_server.py  (header: "# DEPRECATED.", replaced by Rust dashboard;
  nothing imports it) — safe to delete.
- index.html  (legacy dashboard HTML referenced only by dashboard_server.py).

## 4. LOW — Doc / code mismatches (AGENTS.md & docs/PIPELINE_CONVERGED_PLAN.md)

- `--rag` default: RESOLVED — AGENTS.md, main.py, run_production.sh,
  healthcheck.py, and the dashboard dropdown now all agree the default is
  `hybrid` (main.py:76). orchestrator.run_pipeline still defaults its own
  `rag_mode` param to `scraper`, but main.py overrides it with `hybrid`.
- `--help`: AGENTS.md says "silently ignored"; main.py:65-66 explicitly
  handles `--help`/`-h` (prints usage + exits). FIX doc.
- `--probe-llm`/`--probe-yt`: AGENTS.md attributes them to run_production.sh;
  they are healthcheck.py flags (healthcheck.py:41,43). run_production.sh only
  forwards `--rag`. FIX doc.
- `FakeLLM`: AGENTS.md names it; actual test class is `FakeLLMClient`
  (tests/test_hermetic_e2e.py:67). FIX doc.
- File-index: `active_thread_seeder.py`, `reddit_browser_poster.py`,
  `reddit_json_client.py` listed at repo root but live under src/engine/.
- `upsert_playlist_add_video` is defined in mcp_servers/youtube_cloud/server.py:426
  (publisher.py only calls it).
- `fetch_caption_transcript_oauth` is defined in
  mcp_servers/youtube_cloud/server.py:277 (not nano_banana.py).
- docs/PIPELINE_CONVERGED_PLAN.md:172 references `cleanup_pi.py`; the real
  implementation is cleanup.py.
- USE_SEMANTIC: AGENTS.md lists `USE_SEMANTIC_GATES=1` alongside
  `ALLOW_SOFT_APPROVAL=1`; code defaults semantic gates OFF
  (text_embeddings.py:21 `os.getenv("USE_SEMANTIC_GATES","")`), soft approval ON.

## 5. STYLE — large but mechanical (enforced gate excludes these)

Full `ruff check src/ mcp_servers/ tests/ *.py` = 1653 findings (59% auto-fixable).
Notably unused imports (F401) that the F821/F822/F823 gate misses:
- src/agents/orchestrator.py:4 `uuid`, :7 `asyncio`, :10 `A2AMessage/AgentRole`
- src/agents/media_producer.py:7 `ScriptData`, :18 `extract_numeric_chart_spec`
- src/agents/observer.py:9 `semantic_max_similarity`
- src/engine/tool_topic_synthesizer.py:33 `run_budget`
- src/engine/external_apis.py:5 `TopicCandidate`
- src/engine/rss_ingestion.py:5 `uuid`, :373 `embedding_engine`
- src/engine/topic_deduplicator.py:11 `compute_keyword_vector/embedding_engine`
- mcp_servers/youtube_cloud/server.py:76 duplicate import (also at :71)
- ~60 more across mcp_servers/, scripts, and top-level .py
Other common classes: import sorting (I001), bare `except` (BLE001/S110),
line-length. Suggested: extend the hermetic gate to F401/F811 + selectively
I001/BLE001, then `ruff --fix`.

## 6. Verified NOT a bug
- fact_retriever.py:50 `if candidates and facts:` — every appended candidate
  also appends a matching VerifiedFact (rss_ingestion.py:625-634), so the guard
  is effectively `if candidates` and does not discard live candidates.
- All subprocess calls use list args (no shell=True) -> no local shell injection.
- No eval/exec/os.system/os.popen/pickle/yaml.load found.
- Secrets are placed in HTTP headers / credential objects, not printed/logged.

## Suggested priority order
1) 1.1 SSH injection (security, external RCE)
2) 2.1/2.2 subprocess timeouts (hang risk)
3) 2.3 SSRF + 2.4 URL keys (security hygiene)
4) 2.5 concurrency guards
5) 4 doc fixes (zero-risk)
6) 3 dead-code cleanup (after confirming no dynamic imports)
7) 5 style pass (mechanical, opt-in)
