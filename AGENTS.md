# AGENTS.md

Autonomous 8-stage YouTube storytelling video pipeline (CSVG): RSS -> topic TOPSIS selection -> RAG-grounded 6-act script -> LLM-edited/observed -> TTS + AI visuals + ffmpeg assembly -> YouTube publish. Plain Python scripts, no packaging, no build step. Run everything from the repo root (relative `logs/` paths and `from src...` imports depend on it).

**Design, architecture decisions and feature status (shipped/planned/not-feasible) live in `docs/PIPELINE_CONVERGED_PLAN.md` — the single source of truth.** This file is the operational quick-reference only.

## Where to look (file index)

Need to change something? Start here, not with a global grep.

| Concern | File(s) |
|---|---|
| Pipeline entrypoint / flags / resume | `main.py` |
| Run orchestration, phase wiring, soft approval | `src/agents/orchestrator.py` (single entrypoint `run_pipeline()`) |
| Production wrapper (health check, detach, email, auto-resume) | `run_production.sh`, `healthcheck.py`, `send_pipeline_email.py` |
| Cron scheduling / daily publish cadence | `cron_publish.sh` |
| Deploy / sync to Pi | `deploy.sh`, `sync_to_pi.sh`, `pull_from_oci.sh` |
| Agents (sequential A2A) | `src/agents/fact_retriever.py`, `story_designer.py`, `observer.py`, `media_producer.py`, `publisher.py` |
| Pydantic models / checkpoint schema | `src/schemas/state.py`, `a2a.py`, `seed_distribution.py` |
| LLM client + per-role routing | `src/engine/llm_client.py` |
| RAG (scrapers + grounder) | `src/engine/rag_retriever.py`, `grounded_search.py`, `external_apis.py` |
| RSS ingestion / news source | `src/engine/rss_ingestion.py` |
| Topic selection (TOPSIS, demand, dedup, doc-gate, region) | `src/engine/topic_topsis.py`, `opportunity_score.py`, `youtube_topic_demand.py`, `topic_deduplicator.py`, `documentary_potential.py`, `trend_velocity.py`, `social_signals.py`, `tool_topic_synthesizer.py` |
| Region / market / revenue selection | `src/engine/region_intelligence.py`, `monetization_optimizer.py`, `channel_phase_manager.py` |
| Script quality (semantic gates, embeddings, chapters, pacing, term register) | `src/engine/quality_verifier.py`, `text_embeddings.py`, `chapters.py`, `script_pacing_engine.py`, `term_register.py`, `feedback_memory.py` |
| Media production (audio/visuals/ffmpeg/orchestration) | `src/agents/media_producer.py`, `src/engine/micro_content_producer.py`, `media_budget.py`, `pexels_retriever.py`, `pixabay_retriever.py`, `gif_retriever.py`, `video_quality_metrics.py` |
| Grounded-search POC | `poc_grounded_search.py` |
| MCP servers (external service endpoints) | `mcp_servers/audio_edge/server.py`, `media_cloud/server.py`, `youtube_cloud/server.py` |
| Video/quality audit of an uploaded video | `run_video_verifier.py`, `src/engine/youtube_video_verifier.py` |
| Post-publish distribution (seed, Reddit, Shorts) | `src/engine/seed_distributor.py`, `active_thread_seeder.py`, `reddit_browser_poster.py`, `reddit_json_client.py`, `reddit_link_seeder.py`, `reddit_warmup.py`, `post_reddit_links.py`, `cleanup_pi.py`, `src/schemas/seed_distribution.py` |
| Budget / quota ledgers | `src/engine/run_budget.py` |
| Dashboard (Rust) | `rust_dashboard/` |
| Tests | `run_tests.py`, `tests/test_hermetic_e2e.py` |
| OAuth token for YouTube | `get_youtube_token.py` (token.json + client_secret.json) |
| Deprecated (do not use) | `dashboard_server.py` (replaced by Rust dashboard) |

## Commands
- Activate venv first: `source venv/bin/activate`
- `--offline` uses canned topic candidates instead of live RSS — it is NOT a full dry-run. It does NOT skip RAG (Tavily/Firecrawl), the LLM (aborts if unavailable), visuals (fal/Replicate), Pi TTS, or publish. Those each need their own flag or a mock: `--dummy-frames` (synthetic visuals, no fal/Replicate), `--till-upload` (no publish). The truly hermetic/no-network path is `tests/test_hermetic_e2e.py` (stub agents mock LLM/RAG/TTS/visuals/publisher).
- Real run without publishing: `python main.py --global --till-upload` (alias `--no-upload`)
- Production run (recommended for real publishes): `./run_production.sh` — **runs a pre-flight health check** (`healthcheck.py`: env keys, ffmpeg/ffprobe, LLM availability, Pi audio-edge reachability, YouTube upload quota + competitor-demand budget, RAG fact-source keys, BGM + disk space) and **aborts before launch if any required check fails**; then syncs code to Pi, runs in background, logs to `logs/`, emails result. `--no-detach` blocks; `--skip-health-check` bypasses the gate; optional `--probe-llm`/`--probe-yt` do a real 1-token LLM call and token-refresh `channels.list`. Auto-resumes from latest `logs/state_*.json` checkpoint if a previous run didn't reach `PUBLISHED_SUCCESS`.
- Resume a specific run: `python main.py --resume <pipeline_id>` (reads `logs/state_<pipeline_id>.json`).
- Region-optimized cron scheduler: `./cron_publish.sh [--region india|global] [--dry-run]` — schedules up to TWO publishes/day (the cadence behind the $2,000/month goal), back-timing launch windows [11:00-12:20] and [13:30-14:30] UTC out of the measured ~1h50m runtime so the publish lands in a peak window. The MARKET/region itself is decided DYNAMICALLY inside the pipeline (fact_retriever via `region_intelligence`, see converged plan); `--region` only pins an explicit override. Guards: never duplicates a live run (pgrep + heartbeat), caps daily publishes (`CSVG_MAX_DAILY_PUBLISHES`, default 4), enforces a cooldown (`CSVG_CRON_COOLDOWN_MIN`, default 90). Installed in crontab at `20 11` and `50 13` UTC daily; all decisions/skips journal to `logs/cron_publish.log`. `run_production.sh` passes `--global`/`--india` through as an override and otherwise runs dynamic.
- **Storability gate + revenue-weighted dynamic region + semantic gates + seed suite + fail-fast TTS & voice rotation** → see `docs/PIPELINE_CONVERGED_PLAN.md` (SHIPPED & VERIFIED).

### Flags (main.py)
`--offline`, `--india` / `--global` (region), `--till-upload`/`--no-upload`, `--dummy-frames` (synthetic visuals, skips fal/Replicate), `--renderer ffmpeg|moviepy`, `--crossfade <seconds>`, `--tail <seconds>` (video-only hold after each shot's narration, default 1.2; env `CSVG_PAD_AFTER_NARRATION`), `--resume <id>`, `--rag grounded|hybrid|scraper` (A/B: Google Search grounding research pass vs the 5-scraper RAG path, with `hybrid` = grounded core + scraper depth; default = scraper), `--outline-first` (A/B: validate an 18-shot beat outline + per-act narration before falling back to the monolithic generation; env `CSVG_OUTLINE_FIRST=1`).
- There is NO `--help`; `main.py` scans `sys.argv` (no argparse) so unknown flags (incl. `--help`) are silently ignored. Semantic gates are env-only (`USE_SEMANTIC_GATES=1`), not a CLI flag.

### Tests
- Suite runner: `python run_tests.py` (custom wrapper, not pytest) imports and runs the single hermetic E2E module.
- The ONLY test file is `tests/test_hermetic_e2e.py` — a fully HERMETIC, self-sufficient end-to-end test of the **real** orchestrator with the **real** StoryDesigner + Observer agents (only genuinely-external boundaries are stubbed: RAG scrapers, channel stats, YouTube publish, TTS/visual/ffmpeg gates, and a scripted FakeLLM as `LLMClient`). It covers the happy path through publish, the surgical per-shot revision loop (state_hash enforcement, non-target shots bit-identical), stale-REVISE rejection, the outline-first path, and per-agent model routing. **No network, no Pi, no media generation, no `.env`** — a failure is a real regression, never a flaky mock. It self-boots its own `sys.path` so it runs from any CWD.
- Run it: `python run_tests.py` or `python tests/test_hermetic_e2e.py`. Both are `py_compile`-clean.
- Lint/format: ruff is used informally (only `.ruff_cache/` exists, no committed config, no CI). No enforced lint/formatter.

## Env & secrets
- `.env` is gitignored; copy `example.env` -> `.env` and fill real keys. NEVER commit `.env`, `token.json`, or `client_secret.json` (YouTube OAuth).
- LLM fallback chain: primary model -> `LLM_FALLBACK_MODEL` -> `LLM_FALLBACK_MODEL2`. There's NO template/boilerplate fallback — if the LLM is unavailable the run **aborts** rather than emit canned content.
- LLM default provider config lives in `.env` (`PREFERRED_LLM_PROVIDER`, `LLM_MODEL`, keys).
- Semantic-gate env: `USE_SEMANTIC_GATES=1`, `ALLOW_SOFT_APPROVAL=1`, `HF_HOME`/`TRANSFORMERS_CACHE` -> repo-local `.hf_cache/`. Heavy ML deps live in `requirements-master.txt` (master ONLY, never the Pi; the Pi keeps TF-IDF fallback).

## Narrow tool-topic synthesis (fact_retriever stage 1b)
- `src/engine/tool_topic_synthesizer.py` + `fact_retriever.py`: after RSS ingestion, an LLM pass over the day's fresh corpus proposes narrow, evergreen, search-demand topics RSS news never surfaces — `"[Tool A] vs [Tool B] for [task]"`, `"How to [task] using [AI tool]"`, individual tool deep-dives, enterprise/developer tooling comparisons. Toggle: `TOOL_TOPIC_SYNTHESIS=1` (default on), count `TOOL_TOPIC_MAX` (default 4).
- Each proposed topic is precision-measured (`precise_topic_demand` on its exact `demand_query`) and passes a STRICT gate (all channel phases): unmeasured ⇒ **culled** (synthetic topics get no RSS "presumption of relevance"), measured-but `< OPPORTUNITY_MIN_SCORE` (0.5) ⇒ **culled**. Kept synthetics enter TOPSIS first (evergreen > news), budget-bounded by `YT_SEARCH_DAILY_BUDGET`.

## Architecture (where things live)
- `src/agents/` — sequential A2A agents wired by `orchestrator.py`: `fact_retriever` -> `story_designer` -> `observer` (quality gates + bounded 3-revision **surgical per-shot repair** loop, driving the Observer's `REVISE_SCRIPT` message with state_hash enforcement) -> `media_producer` -> `publisher`. `orchestrator.run_pipeline()` is the single entrypoint.
- `src/engine/` — stateless/stateful helpers (RAG, TOPSIS, quality_verifier, llm_client, channel_phase_manager, etc.). See the "Where to look" table above for which file owns which concern; otherwise random leaf files; no surprises here.
- `src/schemas/` — pydantic models (`state.py`, `a2a.py`); `GlobalState` is the checkpoint schema.
- `mcp_servers/` — three standalone FastAPI apps: `audio_edge` (Kokoro TTS + Whisper .ass, on the Pi), `media_cloud` (fal/flux visuals + ffmpeg, port 8001), `youtube_cloud` (upload/quota, port 8002). Only `audio_edge` has a committed systemd unit (`kokoro_tts.service`, port 8000, Pi); `media_cloud`/`youtube_cloud` run via `uvicorn.run` (no committed unit; `deploy.sh` only restarts `kokoro_tts`). Add new model/http endpoints here, not in `src/engine`.

## Dashboard (Rust)
- `rust_dashboard/` — zero-dependency (std-only) Rust web dashboard that **replaces the deprecated Python `dashboard_server.py` + legacy `index.html`**. Built with `cargo build --release`; run as `csvg_rust_dashboard.service` (port 8080, `CSVG_ROOT` env points at the repo dir holding `logs/` + `*.json`). Serves a self-contained SPA (`rust_dashboard/web/index.html`, embedded via `include_str!`) plus JSON endpoints: `/api/status` (heartbeat freshness + latest stage + channel stats), `/api/logs` (`pipeline_run.log` tail + `csvg_execution.log`), `/api/published` (`published_topics.json`), `/api/budget` (per-run + monthly ledger from `logs/run_budget.json`), `/api/runs` (recent `logs/state_*.json` checkpoints), `/health`. No JSON crate — `src/json.rs` is a minimal parser/serializer; build the binary per-node (the OCI x86 build is NOT pushed to the Pi via `sync_to_pi.sh`, which excludes `rust_dashboard/target/`; `deploy.sh` builds it on the Pi and switches the service).

## Distributed layout (this is not a single-host repo)
- Master pipeline (agents, media_cloud, youtube_cloud) runs on the OCI cloud host.
- Audio/TTS (audio_edge, Kokoro, Whisper) runs on a Raspberry Pi 5 edge node, reached via `AUDIO_EDGE_URL` and `LLAMA_CPP_URL`.
- `deploy.sh` git-clones/reset-hards the repo to both nodes over SSH and restarts services; it also builds the Rust dashboard on the Pi and fires up `csvg_rust_dashboard.service` (stopping the deprecated Python one); `sync_to_pi.sh` rsyncs the working tree to the Pi (excludes logs/media/venv/`rust_dashboard/target/`).
- `run_production.sh` pushes logs/state/heartbeat to the Pi for a dashboard there; the Pi dashboard reads the fixed `logs/pipeline_run.log` name and heartbeat freshness to infer "running".

## Runtime/generated files (all gitignored, auto-created per run)
- `logs/state_<pipeline_id>.json` — checkpoint/resume source.
- `channel_stats.json` (channel phase GROWTH/REVENUE/SCALE), `published_topics.json` (dedup), `yt_demand_pools.json` + `yt_demand_quota.json` (competitor-demand quota budget).
- Large fixed assets `kokoro-v0.19.onnx` (~325MB) and `voices.bin` are gitignored and locally present; don't expect them in a fresh clone and don't commit them.

## Conventions & gotchas
- New hosts/agents must not duplicate functionality already in `src/engine`; follow the existing A2A message + `tracer.record_step(state, ...)` + structured `logger` pattern used by every agent.
- Blocked/promo/advertorial content is filtered out of RSS and RAG candidates (covered by the smoke test's RAG/publisher path); keep this intact — it's a monetization/quality invariant.
- Publish path enforces YouTube quota (max ~4 uploads/day) and EU AI Act synthetic-content disclosure tags; don't bypass in tests or new features.
- Story requires live LLM and quality gates (Observer + Gates 1/3b/4/5/6/7) pass before publish. Fact/temporal/revenue/audience violations and quality gates 1/3b/4/5/6/7 are hard abort conditions — never warnings. Observer style-class violations are soft (see `docs/PIPELINE_CONVERGED_PLAN.md` §1.7).