# AGENTS.md

Autonomous 8-stage YouTube storytelling video pipeline (CSVG): RSS -> topic TOPSIS selection -> RAG-grounded 6-act script -> LLM-edited/observed -> TTS + AI visuals + ffmpeg assembly -> YouTube publish. Plain Python scripts, no packaging, no build step. Run everything from the repo root (relative `logs/` paths and `from src...` imports depend on it).

## Commands
- Activate venv first: `source venv/bin/activate`
- `--offline` uses canned topic candidates instead of live RSS — it is NOT a full dry-run. It does NOT skip RAG (Tavily/Firecrawl), the LLM (aborts if unavailable), visuals (fal/Replicate), Pi TTS, or publish. Those each need their own flag or a mock: `--dummy-frames` (synthetic visuals, no fal/Replicate), `--till-upload` (no publish). The truly hermetic/no-network path is `tests/test_hermetic_e2e.py` (stub agents mock LLM/RAG/TTS/visuals/publisher).
- Real run without publishing: `python main.py --global --till-upload` (alias `--no-upload`)
- Production run (recommended for real publishes): `./run_production.sh` — **runs a pre-flight health check** (`healthcheck.py`: env keys, ffmpeg/ffprobe, LLM availability, Pi audio-edge reachability, YouTube upload quota + competitor-demand budget, RAG fact-source keys, BGM + disk space) and **aborts before launch if any required check fails**; then syncs code to Pi, runs in background, logs to `logs/`, emails result. `--no-detach` blocks; `--skip-health-check` bypasses the gate; optional `--probe-llm`/`--probe-yt` do a real 1-token LLM call and token-refresh `channels.list`. Auto-resumes from latest `logs/state_*.json` checkpoint if a previous run didn't reach `PUBLISHED_SUCCESS`.
- Resume a specific run: `python main.py --resume <pipeline_id>` (reads `logs/state_<pipeline_id>.json`).
- Region-optimized cron scheduler: `./cron_publish.sh [--region india|global] [--dry-run]` — schedules up to TWO publishes/day (the cadence behind the $2,000/month goal), back-timing launch windows [11:00-12:20] and [13:30-14:30] UTC out of the measured ~1h50m runtime so the publish lands in a peak window. The MARKET/region itself is decided DYNAMICALLY inside the pipeline (fact_retriever via `region_intelligence`, see below); `--region` only pins an explicit override. Guards: never duplicates a live run (pgrep + heartbeat), caps daily publishes (`CSVG_MAX_DAILY_PUBLISHES`, default 4), enforces a cooldown (`CSVG_CRON_COOLDOWN_MIN`, default 90 — the 2.5h-apart slots are ~2/day when both fire). Installed in crontab at `20 11` and `50 13` UTC daily; all decisions/skips journal to `logs/cron_publish.log`. `run_production.sh` passes `--global`/`--india` through as an override and otherwise runs dynamic (no forced `--global`).
- **Storability gate + revenue-weighted dynamic region** (single design doc: `docs/PIPELINE_CONVERGED_PLAN.md`). (1) `src/engine/documentary_potential.py` hard-culls **direct news** — announcement/press-release/Q-results/price-tick blips that our automation can't turn into a 6-act documentary — before TOPSIS (env `DOCUMENTARY_POTENTIAL_FLOOR`, default 0.35; evergreen synthesized tool topics pass through; all-culled ⇒ ship best-with-warning; optional LLM cross-check `DOCUMENTARY_LLM_CROSSCHECK=1` can only cull, never resurrect). (2) **Ad revenue in the region has the HIGHEST weight**: `region_intelligence` scores each market as `α·region_revenue + β·fit + γ·window` (α default 0.55 = env `REGION_WEIGHT_REVENUE`) and `topic_topsis` gained an **8th criterion `REGIONAL_REV`** as the highest single weight in REVENUE/SCALE (GROWTH is discovery-led TVS+IDI so it unlocks YPP faster). (3) With no `--global`/`--india`, `fact_retriever.py` selects the (topic, market) with the best regional revenue — Sensex→india, RBA/Ashes→au, BoE→uk, Shopify→ca, Meta→us — from topic tokens + publisher ccTLD + per-market RPM + projected publish window. The decision is persisted to `state.region` (l2 "global"|"india") + `state.region_market`/`region_reason` and flows to story_designer, media_producer, and the monetization forecast (shared per-market table, `monetization_optimizer._locale_multiplier`). Best-effort: any failure falls back to a US/global run.

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
- Semantic-gate env (see section below): `USE_SEMANTIC_GATES=1`, `ALLOW_SOFT_APPROVAL=1`, `HF_HOME`/`TRANSFORMERS_CACHE` -> repo-local `.hf_cache/`. Heavy ML deps live in `requirements-master.txt` (master ONLY, never the Pi; the Pi keeps TF-IDF fallback).

## Narrow tool-topic synthesis (fact_retriever stage 1b)
- `src/engine/tool_topic_synthesizer.py` + `fact_retriever.py`: after RSS ingestion, an LLM pass over the day's fresh corpus proposes narrow, evergreen, search-demand topics RSS news never surfaces — `"[Tool A] vs [Tool B] for [task]"`, `"How to [task] using [AI tool]"`, individual tool deep-dives, enterprise/developer tooling comparisons. Toggle: `TOOL_TOPIC_SYNTHESIS=1` (default on), count `TOOL_TOPIC_MAX` (default 4).
- Each proposed topic is precision-measured (`precise_topic_demand` on its exact `demand_query`) and passes a STRICT gate (all channel phases): unmeasured ⇒ **culled** (synthetic topics get no RSS "presumption of relevance"), measured-but `< OPPORTUNITY_MIN_SCORE` (0.5) ⇒ **culled**. Kept synthetics enter TOPSIS first (evergreen > news), budget-bounded by `YT_SEARCH_DAILY_BUDGET`.

## Architecture (where things live)
- `src/agents/` — sequential A2A agents wired by `orchestrator.py`: `fact_retriever` -> `story_designer` -> `observer` (quality gates + bounded 3-revision **surgical per-shot repair** loop, driving the Observer's `REVISE_SCRIPT` message with state_hash enforcement) -> `media_producer` -> `publisher`. `orchestrator.run_pipeline()` is the single entrypoint.
- `src/engine/` — stateless/stateful helpers (RAG, TOPSIS, quality_verifier, llm_client, channel_phase_manager, etc.). Random leaf files; no surprises here.
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

## Semantic quality gates (Observer MiniLM backend)
A frozen `all-MiniLM-L6-v2` sentence-encoder (torch/transformers, ~0.8 GB peak RAM) behind `src/engine/text_embeddings.py`'s existing API, enabled via `USE_SEMANTIC_GATES=1`. On the Pi, or when deps are absent, every gate falls back to the original TF-IDF/NLTK logic — no regressions.

### Install & enable (master only)
```bash
source venv/bin/activate && pip install -r requirements-master.txt
# .env: USE_SEMANTIC_GATES=1, HF_HOME=/path/to/buzzdropfeedv2/.hf_cache
```
The model cache stays in `.hf_cache/` (gitignored, excluded from `sync_to_pi.sh`). Never install `requirements-master.txt` on the Pi.

### What it replaces (observer.py)
| Gate | Current (fallback) | Semantic (when ON) |
|---|---|---|
| Verbatim/paraphrase copy | ≥12-word substring match | MiniLM cosine sim ≥ 0.94 vs clean corpus = whole-sentence meaning copy. The 0.82–0.93 band is deliberately NOT flagged: fact-dense narration that must preserve names/numbers/dates can't paraphrase below ~0.85, so flagging it was a false positive (calibrated live: 27→13 flags) |
| Keyword over-repetition | Exact-token blacklist | Semantic topic-membership: token vs topic anchor words (keywords + headline + summary); `chinese~china` = 0.68, `gadget~china` = 0.31; threshold 0.50 |
| Sentence monotony / duplication | TF-IDF cos ≥ 0.82 | MiniLM pairwise similarity ≥ 0.82; one-batch-encode, precomputed matrix |
| Quality score | N/A | Paraphrase diversity = 1 − mean pairwise sentence sim; used for best-draft retention across revisions |

### Soft approval (orchestrator.py)
After the bounded 3-revision loop, style-class violations (keyword over-repetition, verbatim copy, sentence repetition, visual prompt, narration too long) are **non-blocking** when `ALLOW_SOFT_APPROVAL=1` (default). Source diversity checks are bypassed for the narration since attributions are description-only. Hard invariants (fact/temporal audit, revenue/audience gate, runtime, shot count, quality gates 1–7) still abort. Set `ALLOW_SOFT_APPROVAL=0` to restore all-or-nothing.

### Model persistence
The model loads lazily on first use and stays **resident** for the process lifetime. `release()` frees the model weights (~50 MB) but torch stays imported in-process. Reloading is expensive (~10–20 s) and unnecessary for 4–6 runs/day — resident is the default.

### Clean RAG (rag_retriever.py + story_designer.py)
`_strip_boilerplate()` strips ad/credit/nav junk (SKIP ADVERTISEMENT, Video by, Listen ·, Subscribe, etc.) from deep-crawled articles before they enter the fact corpus, crawled_content, and the snippet padding pool. The same junk filter (`_SNIPPET_JUNK_RE`) protects the story designer's RAG snippets. This removes the "verbatim NYT boilerplate in Shot #15" class of false positives at the source.

### Thresholds (shared: text_embeddings.py, consumed by observer.py + story_designer.py)
- `COPY_SEMANTIC_HARD_THRESHOLD = 0.94` — narration sentence ~== clean corpus sentence (whole-sentence meaning copy; the verbatim gate)
- `COPY_SEMANTIC_THRESHOLD = 0.82` — legacy lower bound, kept only for API compatibility (no longer used by the gate)
- `TOPIC_MEMBER_SEMANTIC_THRESHOLD = 0.50` — token ~== topic anchor word (keyword/headline/summary)

### Anti-verbatim writer (story_designer.py)
The writer self-corrects copies BEFORE the Observer audits, in two layers:
1. **LLM polish prompt** (`_polish_script`, `process()` passes `corpus_sents`): the polish prompt detects narration sentences that are whole-sentence meaning copies (sim ≥ 0.94) and lists them explicitly, demanding a *structurally different* rewrite (restructure clause order, split/merge) — not just synonym substitution. A standing ANTI-VERBATIM rule in the prompt forbids mirroring a source sentence's wording.
2. **Local WordNet dissolve** (`_dissolve_verbatim_copies`, run after polish + word-floor): a deterministic, no-LLM, offline pass that swaps local NLTK WordNet synonyms in any remaining ≥0.94 sentence until it drops below threshold, protecting proper nouns/numbers/dates/currency. Requires `wordnet` + `omw-1.4` data in NLTK (installed under `/home/ubuntu/nltk_data/corpora/`); a no-op when WordNet or the semantic backend is unavailable (never fails/hangs, never hits the network).

Net effect: the validator stops flagging fact-dense rephrases (27→13 on live data) and the writer dissolves the true copies, so the 3-revision loop converges instead of growing (was 25→27→32).

### Not learning (frozen encoder)
The MiniLM model is a **frozen pretrained encoder** — no gradient updates, no RL, no weight adaptation across runs. It generalises to `china→chinese` because it was trained on billions of text pairs, not because it learns from your pipeline. If you want learned gates, future options (in order of complexity):

1. **Online threshold calibration** — record `{scores, verdict}` per run; nudge thresholds so false rejections trend to zero. Deterministic, no training infra.
2. **Supervised reward classifier** — accumulate accepted-vs-rejected scripts, fine-tune a small DistilBERT to predict pass/fail. Needs ~hundreds of labeled samples.
3. **RLHF/DPO on the writer** — treat the Observer as a reward signal and fine-tune the generative model. Requires an open model (Gemini is not fine-tunable); heavy, out of scope for 2-vCPU/12 GB.

## Seed Traffic Seeding & Distribution Pipeline
An automated post-publish distribution pipeline designed to seed early traffic, boost CTR, and jumpstart YouTube's recommendation metrics.

### Components
1. **Instant Pinned Comment:** Posts an engaging question thread via `/tools/insert_pinned_comment` right after publication to drive early comments (GROWTH-phase variant pushes subscribe + watch-time).
2. **Micro-Content Clips:** The `MicroContentProducer` extracts key 30-60s acts from the master video, crops them to 9:16 vertical crop layout (`crop=ih*9/16:ih`), and renders them as independent Clips for Shorts/Reels/TikTok. In GROWTH phase the clips are **published as YouTube Shorts** (`/tools/upload_short`, `#Shorts`) — the #1 discovery lever pre-YPP.
3. **Seeding Assistant (`SeedDistributor`):** Creates tailored draft templates for Reddit, Hacker News, LinkedIn, X, TikTok, Instagram, Pinterest, Telegram, and Medium based on the video's summary. It dispatches a rich embed to Slack/Discord webhooks containing copyable markdown blocks. Takeaways are grounded in the run's verified facts (never canned filler).
4. **Intelligent Semantic Relevance:** Computes cosine similarity between thread topics and video content using the **resident `MiniLM`** sentence-transformer (`seed_distributor._seed_embedder`, default on, cached under `.hf_cache`); only subreddits scoring >= `SEED_RELEVANCE_THRESHOLD` (default **0.50**, measured floor; raise toward 0.75 via env for a stricter gate) are seeded. When MiniLM is unavailable (`SEED_RELEVANCE_SEMANTIC=0` / no torch) it falls back to TF-IDF char-n-gram cosine at its own floor (0.30). The active backend + threshold are logged per run.

## Conventions & gotchas
- New hosts/agents must not duplicate functionality already in `src/engine`; follow the existing A2A message + `tracer.record_step(state, ...)` + structured `logger` pattern used by every agent.
- Blocked/promo/advertorial content is filtered out of RSS and RAG candidates (covered by the smoke test's RAG/publisher path); keep this intact — it's a monetization/quality invariant.
- Publish path enforces YouTube quota (max ~4 uploads/day) and EU AI Act synthetic-content disclosure tags; don't bypass in tests or new features.
- Story requires live LLM and quality gates (Observer + Gates 1/3b/4/5/6/7) pass before publish. Fact/temporal/revenue/audience violations and quality gates 1/3b/4/5/6/7 are hard abort conditions — never warnings. Observer style-class violations are soft (see Semantic quality gates section).

