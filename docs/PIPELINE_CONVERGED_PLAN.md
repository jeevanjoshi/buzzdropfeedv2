# CSVG Converged Implementation Plan — Single Source of Truth

_Status: ratified · supersedes the earlier fragmented plans (opportunity-score, RAG-alignment,
publish-integrity, script-quality, stabilization, media-quality upgrades). Last updated 2026-08-13.
This document is the **single source of truth** for architecture, design decisions and feature
status. Legend: **SHIPPED & VERIFIED** / **PLANNED — NEXT PASS** / **NOT FEASIBLE / DEFERRED**.
Operational quick-reference (commands, flags, tests, env, conventions) lives in `AGENTS.md`._

## 0. Goal

1. **Never publish direct news.** A topic is only selectable when our automation can build a
   6-act documentary / investigative story out of it. Announcement / press-release / price-tick
   blips are culled before they ever reach ranking.
2. **Ad revenue in the region has the highest weight.** Regional ad revenue dominates both the
   TOPSIS ranking (8th criterion, highest single weight) and the region/market selection score.
3. **Region is dynamic.** The market a run targets is decided at selection time from day/time +
   topic affinity + per-market RPM + events, then passed downstream to story design, visuals and
   the revenue forecast (`state.region` / `state.region_market` / `state.region_reason`).

---

## 1. SHIPPED & VERIFIED

### 1.1 Storability gate (documentary potential)

`src/engine/documentary_potential.py` (deterministic-first, offline).

- `score_documentary_potential(candidate) -> {score 0..1, verdict, direct_hits, depth_hits, has_data, reason}`:
  - baseline 0.5; −0.18 per **direct-news** pattern (announce/unveil/launch/introduce, wins/scores/Q-results, price ticks, price-target/rating notes, "today announced / will be available" phrases); −0.10 for thin text (<30 words), +0.10 for rich text.
  - +0.12 per **depth** pattern (probe/inquiry/whistleblower/leak, lawsuit/court/charges, fraud/scandal/criminal, regulator bodies, risk/fallout/backlash, history/timeline/origins, how-it-works/why, expert-warn/study-finds, supply-chain/chip/quantum, transition/revolution); +0.10 when ≥2 numeric data points exist (chartable).
  - `verdict = direct_news` when `score < DOCUMENTARY_POTENTIAL_FLOOR` (default **0.35**, conservative — only clear blips culled).
- `gate_candidates(candidates)` → hard cull; evergreen synthesized tool topics (have `demand_query` / `narrow-synth-*` id) pass through.
- `refine_with_llm(candidates)` — **opt-in** (`DOCUMENTARY_LLM_CROSSCHECK=1`), one batched LLM call over the surviving shortlist; may only cull near-boundary topics, **never resurrects** a deterministic cull (LLM drift cannot ship direct news).

Wiring (fact retriever): run right after RSS + topic synthesis, before the opportunity gate and
TOPSIS. Culled topics are logged (`[FactRetriever] CULLED (direct news, no doc potential): …`).
**All-culled → SHIP BEST-WITH-WARNING**: the highest documentary-scoring survivor is retained with
a loud log line (no abort; the downstream RAG-sufficiency gate still guards factual depth).

### 1.2 Revenue-weighted dynamic region

`src/engine/region_intelligence.py` — selection score per (topic, market):

```
fit             = min(1.0, affinity/4) + 0.35·domain                 # 0..1
effective_rev   = net_rpm_usd · locale_rpm · (0.25 + 0.75·fit)       # market potential
region_rev_norm = clamp(effective_rev / REGION_REVENUE_REFERENCE_USD, 0, 1)
score           = ALPHA·region_rev_norm + BETA·fit + GAMMA·window
```

- **ALPHA (regional ad revenue) is the highest weight** — default 0.55 (env `REGION_WEIGHT_REVENUE`);
  `BETA = (1-α)·2/3`, `GAMMA = (1-α)/3`. `net_rpm_usd` and `locale_rpm` come from the shared market
  table used by `monetization_optimizer._locale_multiplier` (no drift).
- Affinity/domain feed the revenue term so a market's own topics win despite a lower locale RPM
  (Sensex→india, RBA→au, BoE→uk, Shopify→ca, Meta→us), while otherwise-close markets are resolved
  toward the one that pays the most now; `window` is a small, separate timing nudge that never zeros
  a topic's own-market revenue.
- `candidate_region_profile(candidate, projected_min)` returns `{market, l2_region, score,
  region_revenue_usd (full forecast), reason}`; used to fill the TOPSIS 8th criterion and
  `state.region*`.

`src/engine/topic_topsis.py` — **8th criterion `REGIONAL_REV`** (benefit). In the MONETISED
phases it holds the single highest weight: REVENUE `[0.14,0.20,0.12,0.05,0.08,0.08,0.05,0.28]`,
SCALE `[0.15,0.18,0.12,0.05,0.10,0.07,0.05,0.28]`. GROWTH is discovery-led (TVS+IDI lead, revenue
secondary): `[0.25,0.05,0.25,0.10,0.10,0.05,0.05,0.15]`.
`TopicCandidate.regional_revenue_usd` carries each candidate's best-market forecast (or the
fixed-region forecast in `--global`/`--india` mode).

### 1.3 Selection order (fact retriever)

1. RSS fetch (+ World Bank enrichment).
2. Tool-topic synthesis + strict demand gate.
3. **Storability gate** (cull direct news; ship-best-with-warning if all culled; optional LLM refine).
4. Dynamic region profiles (projected publish = now + `DEFAULT_RUNTIME_MIN` 120).
5. Fill `regional_revenue_usd` for all candidates.
6. Opportunity hard gate (REVENUE/SCALE).
7. TOPSIS (8 criteria, revenue-led).
8. Precise shortlist; then audience/niche/revenue/competitor gates → winner.
9. Persist `state.region` (l2) / `state.region_market` / `state.region_reason`; forecast for the
   winner's market; A2A payload includes `regional_revenue_usd`.

### 1.4 Revenue-goal alignment ($2,000/month — single source of truth)

Everything scales from one model in `channel_phase_manager.py` (derived, never
hardcoded):

```
MONTHLY_REVENUE_TARGET_USD = env CSVG_MONTHLY_REVENUE_TARGET_USD  (default 2000)
TARGET_DAILY_PUBLISHES     = env CSVG_TARGET_DAILY_PUBLISHES      (default 2, = cron slots)
REVENUE_GATE_MIN_USD       = 2000 / (2 × 30) = $33.33 per video    (REVENUE/SCALE phases)
```

| Constant | Before (conflict) | After (aligned) |
|---|---|---|
| `channel_phase_manager.REVENUE_GATE_MIN_USD` | `16.67` = $2000/120 videos (4/day) — but cron did ~1/day | `$33.33` derived from $2000/(2×30), matches the 2/day cron cadence |
| `monetization_optimizer.filter_by_competitor_volume` default | `2450.0` (arbitrary) | the derived per-video gate (`REVENUE_GATE_MIN_USD`) |
| Stale `TOPSIS_WEIGHTS_*` in channel_phase_manager | 7-criteria duplicates (RPM low in GROWTH) conflicting with the 8-criterion revenue-led vectors | removed; `get_topsis_weights` delegates to `topic_topsis` (single source) |
| `DAILY_PUBLISH_SLOTS_UTC` | 4 slots (01:30/06:30/10:30/14:30) | the two real cron slots (`11:20`, `13:50`) |
| `cron_publish.sh` cooldown | 480 min → ~1 run/day | 90 min → up to 2/day |
| YPP watch-time estimate | assumed 4 videos/day × 100 views | `TARGET_DAILY_PUBLISHES × 100` views/day |
| Observer Gate 0b revenue check | evaluated with default region (US 1.0) | evaluated against the topic's selected market (`state.region_market`) |

Note: the per-video revenue gate only binds in REVENUE/SCALE (post-YPP). During
GROWTH the pipeline optimises watch-time/novelty with revenue still wired as the
strongest TOPSIS criterion, so it is YPP-ready the moment the channel unlocks.

### 1.5 GROWTH phase (pre-YPP)

The channel goal is $2,000/month (REVENUE phase), but GROWTH is the step before:
it must convert viewers into subscribers + watch-hours to unlock YPP.

- **Shorts published** — `micro_content_producer` cuts 9:16 vertical 30-45s clips;
  `mcp_servers/youtube_cloud` `/tools/upload_short` (same OAuth resumable insert +
  `#Shorts` tag + EU-AI-act disclosure) uploads them in GROWTH phase (non-fatal,
  quota-shared) — the #1 discovery/subscriber lever pre-YPP (`UploadMetadata.shorts_video_id`).
- **Phase-aware pinned comment** — GROWTH pushes subscribe + watch-time; later phases keep
  the pure engagement question.
- **Subscribe CTA in description** (GROWTH) — appends a subscribe/bell + daily-series line.
- **GROWTH is discovery-led.** Pre-YPP the channel earns $0 in ads, so the GROWTH TOPSIS
  vector leads with novelty + trend (IDI 0.25 / TVS 0.25) for watch-time/subscriber growth,
  regional revenue secondary (0.15); REVENUE/SCALE keep regional revenue highest. Unlocks
  YPP faster → monetizes sooner.
- **Region-optimized cron** (`cron_publish.sh`) — up to TWO publishes/day behind the $2,000/month
  goal; launch bands [11:00-12:20] & [13:30-14:30] UTC backed out of the ~1h50m runtime; guards
  (no double-run, `CSVG_MAX_DAILY_PUBLISHES=4`, cooldown 90) → up to 2/day; journal
  `logs/cron_publish.log`. Installed in crontab at `20 11` / `50 13` UTC.
- **Contextual chapters** (`src/engine/chapters.py`) — per-act chapter labels derived from
  shot narrations (LLM + deterministic fallback), stored on `SEOMetadata.act_titles`, reused
  at publish.
- **Niche thumbnails + stock art** (`media_producer.py`) — company-aware ticker resolution,
  private-company guard, narration-grounded fallback moves (no fabricated prices), clean chart
  titles, topic-fused thumbnail scene.

### 1.6 Seed Traffic Seeding & Distribution

Post-publish distribution to seed early traffic, boost CTR and jumpstart YT recommendation.

1. **Instant Pinned Comment** — `/tools/insert_pinned_comment` right after publication
   (GROWTH variant pushes subscribe + watch-time).
2. **Micro-Content Clips** — extracts 30-60s acts, crops to 9:16
   (`crop=ih*9/16:ih`), renders independent Clips for Shorts/Reels/TikTok; in GROWTH phase
   published as YouTube Shorts (`/tools/upload_short`, `#Shorts`).
3. **SeedDistributor** — tailored draft templates for Reddit, HN, LinkedIn, X, TikTok, Instagram,
   Pinterest, Telegram, Medium grounded in the run's verified facts (never canned filler);
   dispatches a rich embed to Slack/Discord webhooks with copyable markdown.
4. **Semantic relevance** — cosine similarity between thread topics and video content via the
   resident MiniLM (`seed_distributor._seed_embedder`, cached under `.hf_cache`); only
   subreddits ≥ `SEED_RELEVANCE_THRESHOLD` (default 0.50) are seeded; TF-IDF char-n-gram
   fallback at floor 0.30 when MiniLM unavailable.
5. **Active Thread Reply Bot** (`ActiveThreadSeeder`) — injects helpful comments into highly
   active discussions from the last 24h; LLM writes context-aware replies containing verified
   facts + a natural YouTube citation at the end. Backend chain: PRAW (`REDDIT_CLIENT_ID` set)
   → Playwright browser poster → read-only `.json` client. Soft warm-up toggle
   (`ACTIVE_SEEDER_WARMUP=1`).
6. **Browser-driven Reddit engagement** (`RedditBrowserPoster`) — Playwright Chromium poster
   for the Pi that works where `.json`/PRAW are IP-blocked; real Chrome-derived binary
   (`REDDIT_CHROMIUM_PATH`), persisted login sessions (`logs/reddit_sessions/`), rotated
   gitignored account pool (`reddit_accounts.json`), RAM/process resource guards so bursts
   can't OOM the Pi's 4GB, old.reddit form posting, best-effort visibility verification,
   learns per-subreddit permissiveness, retires shadowbanned accounts
   (`REDDIT_RETIRE_AFTER_UNVERIFIED`, default 3). State in `logs/reddit_rotation_state.json`.
   `RedditJsonClient` is a throttled, OAuth-free fallback for discovery/comment context only.
7. **Pi edge warm-up on publish** (`_trigger_pi_warmup`) — fire-and-forget SSH to the Pi to run
   `reddit_warmup.py`: low-volume, NO-LINK informative comments from per-domain banks to build
   account trust. Env: `REDDIT_PI_WARMUP_ON_PUBLISH` (default 1), `REDDIT_WARMUP_COUNT` (default 3).
   Non-fatal.
8. **Generic link seeder** (`reddit_link_seeder.py`) — reads ALL published videos from
   `logs/state_*.json` (minus `exclude_video_ids` in `seed_campaigns.json`), discovers active
   threads per topic, matches each to the most relevant published video (keyword overlap +
   learned sub permissiveness + sub size), posts genuine on-topic link comments into niche
   subreddits (cap `REDDIT_LINK_MAX_PER_RUN` default 1; skips subs > `niche_max_subscribers`
   default 150k). `post_reddit_links.py` is a one-off ops variant; `cleanup_pi.py` prunes old
   logs/media/state dirs on the Pi (`CSVG_KEEP_MEDIA_RUNS`, `CSVG_KEEP_STATE_FILES`).

### 1.7 Semantic quality gates (Observer MiniLM backend)

A frozen `all-MiniLM-L6-v2` sentence-encoder (torch/transformers, ~0.8 GB peak RAM) behind
`src/engine/text_embeddings.py`, enabled via `USE_SEMANTIC_GATES=1`. On the Pi, or when deps
are absent, every gate falls back to the original TF-IDF/NLTK logic — no regressions.

- **Install & enable (master only):** `pip install -r requirements-master.txt`;
  `.env`: `USE_SEMANTIC_GATES=1`, `HF_HOME`/`TRANSFORMERS_CACHE` → repo-local `.hf_cache/`
  (gitignored, excluded from `sync_to_pi.sh`). Never install `requirements-master.txt` on the Pi.

| Gate | Current (fallback) | Semantic (when ON) |
|---|---|---|
| Verbatim/paraphrase copy | ≥12-word substring match | MiniLM cosine sim ≥ 0.94 vs clean corpus = whole-sentence meaning copy. The 0.82–0.93 band is deliberately NOT flagged: fact-dense narration that must preserve names/numbers/dates can't paraphrase below ~0.85, so flagging it was a false positive (calibrated live: 27→13 flags) |
| Keyword over-repetition | Exact-token blacklist | Semantic topic-membership: token vs topic anchor words (keywords + headline + summary); `chinese~china` = 0.68, `gadget~china` = 0.31; threshold 0.50 |
| Sentence monotony / duplication | TF-IDF cos ≥ 0.82 | MiniLM pairwise similarity ≥ 0.82; one-batch-encode, precomputed matrix |
| Quality score | N/A | Paraphrase diversity = 1 − mean pairwise sentence sim; used for best-draft retention across revisions |

- **Soft approval** (`orchestrator.py`) — after the bounded 3-revision loop, style-class
  violations are **non-blocking** when `ALLOW_SOFT_APPROVAL=1` (default). Hard invariants
  (fact/temporal audit, revenue/audience gate, runtime, shot count, quality gates 1–7) still
  abort. Set `ALLOW_SOFT_APPROVAL=0` to restore all-or-nothing.
- **Model persistence** — loads lazily on first use, stays resident for process lifetime;
  `release()` frees weights (~50 MB) but torch stays imported. Resident is the default
  (4–6 runs/day).
- **Clean RAG** (`rag_retriever.py` + `story_designer.py`) — `_strip_boilerplate()` strips
  ad/credit/nav junk (SKIP ADVERTISEMENT, Video by, Listen ·, Subscribe, etc.) from
  deep-crawled articles before they enter the fact corpus and snippet padding pool; the same
  junk filter (`_SNIPPET_JUNK_RE`) protects the story designer's RAG snippets. Removes the
  "verbatim NYT boilerplate in Shot #15" class of false positives at the source.
- **Thresholds** (shared `text_embeddings.py`): `COPY_SEMANTIC_HARD_THRESHOLD = 0.94`
  (whole-sentence meaning copy; the verbatim gate); `COPY_SEMANTIC_THRESHOLD = 0.82` (legacy
  lower bound, API compatibility only); `TOPIC_MEMBER_SEMANTIC_THRESHOLD = 0.50`.
- **Anti-verbatim writer** (`story_designer.py`) — the writer self-corrects BEFORE the Observer
  audits, in two layers: (1) **LLM polish prompt** (`_polish_script`) detects narration
  sentences that are whole-sentence meaning copies (sim ≥ 0.94) and demands a structurally
  different rewrite; a standing ANTI-VERBATIM rule forbids mirroring source wording. (2)
  **Local WordNet dissolve** (`_dissolve_verbatim_copies`) — deterministic, no-LLM, offline
  pass that swaps local NLTK WordNet synonyms in any remaining ≥0.94 sentence until below
  threshold, protecting proper nouns/numbers/dates/currency; a no-op when WordNet or the
  semantic backend is unavailable (never fails/hangs, never hits the network).
  Net effect: the validator stops flagging fact-dense rephrases (27→13 live) and true copies
  dissolve, so the 3-revision loop converges (was 25→27→32).

### 1.8 Fail-fast TTS & subtitle integrity

When the Pi edge (`AUDIO_EDGE_URL`) is unreachable **and** no real neural TTS is available
locally, the run **aborts immediately on the first shot** instead of silently shipping
placeholder audio (`src/agents/media_producer.py`):

- **Toner wav** (`engine == "synthetic_wav_fallback"` from either the remote edge response or
  the local `synthesize_tts` fallback) → `RuntimeError`. The tone (1%-amplitude repeating beep)
  is NOT speech and used to pass Gate 2 (wav >1KB) silently; now it fails fast so the checkpoint
  stays resume-safe.
- **Whisper-degraded subtitles** — a single dummy `NARRATION` `Dialogue:` line (written when
  Whisper alignment fails with no real word timestamps) → Gate 3 early `RuntimeError`.
- **Resume safety** — the run keeps its `SCRIPT_APPROVED` checkpoint, so once the Pi is back you
  can `--resume` the same pipeline_id to regenerate media; only a *real* local Kokoro
  (`kokoro_onnx`) fallback is ever allowed to continue offline.
- **Dynamic Voice Rotation** — To break vocal monotony and improve average view percentage (AVP), the pipeline dynamically rotates Kokoro TTS voices across the script's 6 acts. For global/US markets, it assigns a primary female narrator (`af_bella` for Act 1/6), a deep, serious male co-host (`am_adam` for Act 2/5), a technical analyst (`am_michael` for Act 3), and a bright female analyst (`af_nicole` for Act 4). For the India market, it rotates between a primary female Indian voice (`af_sarah`) and serious male analytical segments (`am_adam`).

### 1.9 Ops & reliability

- **Detached-child absolute-path launch** (`run_production.sh`) — the detached child is launched
  via `${SCRIPT_DIR}/$(basename "$0")`, NOT `$0`: under cron `$0` is a bare filename and `$PATH`
  has no repo dir, so `nohup`'s execvp silently `exit 127`'d (no log, no process; child stderr
  used to go to `/dev/null`). Child stderr now appends to its `LOG_FILE` so any spawn failure is
  always visible.
- **OAuth scope regularization** (`get_youtube_token.py`) — grants
  `youtube` / `youtube.upload` / `youtube.readonly` so `channels.list` probes and future
  playlist/comment writes work; YouTube health-check probe passes.
- **Pre-flight health check** (`run_production.sh` → `healthcheck.py`) — env keys, ffmpeg/ffprobe,
  LLM availability, Pi audio-edge reachability, YouTube upload quota + competitor-demand budget,
  RAG fact-source keys, BGM + disk space; **aborts before launch** if any required check fails.
  Optional `--probe-llm`/`--probe-yt` do a real 1-token LLM call and token-refresh `channels.list`.

### 1.10 Architecture & hosts

- `src/agents/` — sequential A2A agents wired by `orchestrator.py`: `fact_retriever` →
  `story_designer` → `observer` (quality gates + bounded 3-revision **surgical per-shot repair**
  loop, state_hash enforcement) → `media_producer` → `publisher`. `orchestrator.run_pipeline()`
  is the single entrypoint.
- `src/engine/` — stateless/stateful helpers (RAG, TOPSIS, quality_verifier, llm_client,
  channel_phase_manager, documentary_potential, region_intelligence, text_embeddings…).
- `src/schemas/` — pydantic models (`state.py`, `a2a.py`); `GlobalState` is the checkpoint schema.
- `mcp_servers/` — three FastAPI apps: `audio_edge` (Kokoro TTS + Whisper .ass, on the Pi, port
  8000, committed `kokoro_tts.service`), `media_cloud` (fal/flux visuals + ffmpeg, port 8001),
  `youtube_cloud` (upload/quota/comment, port 8002). Add new model/http endpoints here, not in
  `src/engine`.
- **Distributed layout:** master pipeline (agents, media_cloud, youtube_cloud) on the OCI cloud
  host; audio/TTS (`audio_edge`, Kokoro, Whisper) on a Raspberry Pi 5 edge node via
  `AUDIO_EDGE_URL`/`LLAMA_CPP_URL`; `deploy.sh` git-clones/reset-hards to both nodes and restarts
  services; `sync_to_pi.sh` rsyncs the working tree (excludes logs/media/venv/`rust_dashboard/
  target/`).
- **Dashboard (Rust)** — `rust_dashboard/` (std-only, `cargo build --release`), run as
  `csvg_rust_dashboard.service` (port 8080, `CSVG_ROOT` env). Self-contained SPA
  (`web/index.html` via `include_str!`) + `/api/status`, `/api/logs`, `/api/published`,
  `/api/budget`, `/api/runs`, `/health`. Minimal JSON parser in `src/json.rs`; built per-node
  (OCI x86 build not pushed to Pi; `deploy.sh` builds on the Pi). Replaces the deprecated Python
  `dashboard_server.py`.
- **Runtime/generated files** (gitignored, auto-created per run): `logs/state_<pipeline_id>.json`
  (checkpoint/resume), `channel_stats.json` (phase GROWTH/REVENUE/SCALE), `published_topics.json`
  (dedup), `yt_demand_pools.json` + `yt_demand_quota.json` (competitor-demand budget). Large fixed
  assets `kokoro-v0.19.onnx` (~325MB) and `voices.bin` are gitignored and locally present.
- **LLM Client Routing & Fallbacks:** The `LLMClient` supports native Google Cloud Vertex AI (using ADC), native Gemini AI Studio (using `GEMINI_API_KEY`), cloud OpenRouter (using `OPENROUTER_API_KEY`), and local `llama.cpp` (using `LLAMA_CPP_URL`). The provider is configured via `PREFERRED_LLM_PROVIDER` in `.env` (values: `google`/`vertex`, `gemini`, `cloud`, or `local`). If a provider fails or isn't configured, the client automatically cascades through the fallback chain (Vertex AI → Gemini AI Studio → OpenRouter → Local llama.cpp) to guarantee run execution without using template fallbacks. The client dynamically respects `LLM_MODEL` configured in `.env`.

### 1.11 Validation

- `python run_tests.py` — **20 hermetic cases** incl. `STORABILITY_GATE` (probe/scandal
  documentary, press-release blip culled, evergreen pass-through, all-culled→best-warning),
  `REGION_REVENUE_DOMINATES` (rba→au, sensex→india, meta→us; ALPHA>BETA>GAMMA; revenue is the
  highest TOPSIS weight in REVENUE & GROWTH), `REVENUE_GOAL_ALIGNMENT` (monthly=$2000, daily=2,
  gate=$33.33, slots 11:20/13:50), `SYNTHETIC_TOPIC_DEDUPLICATION` (ensuring synthesized tool-topics
  undergo full similarity gates), GROWTH-path Shorts publishing, surgical revision loop,
  stale-REVISE rejection, outline-first, routing, A2A alignment, SEO source filter, junk scrub,
  term register, synonym guard, chapter timestamps, fake-upload abort.
- `bash -n run_production.sh cron_publish.sh`.

### 1.12 Media Polish: Audios, Subtitles, and Modern Charts

- **Subtitle Wrapping for Shorts Aspect Ratio:** Restricts subtitle text boundaries to the center 600px safe zone (`MarginL=660, MarginR=660` relative to `PlayResX=1920`) in both `mcp_servers/audio_edge/server.py` and `src/agents/media_producer.py` so they wrap properly and don't clip when center-cropped to vertical 9:16 Shorts.
- **Intro Audio Delay & Subtitle Sync:** Prepends a 0.5s start delay (`adelay=500` filter) to the first shot's audio and shifts its subtitle timestamps by 0.5s in the master subtitle merge function to prevent an abrupt voice jump on play, keeping narration aligned with visuals.
- **Audio Fade In & Out:** Applies a 0.3s audio fade-in and a 1.5s audio fade-out to the final output video mix (both BGM and narration streams) in `mcp_servers/media_cloud/server.py` to ensure smooth audio transitions at the start and end of the video.
- **Modern Minimalist Matplotlib Charts:** Completely restyled visual data charts inside `mcp_servers/media_cloud/server.py` using a premium dark background (`#090d16`), clean annotations, a uniform sky-cyan (`#00e5ff`) color scheme (no cluttering rainbow colors), area glow fill under line trends, smart label prefix/suffix formatting, and 20-degree x-axis label rotation to prevent overlapping.
- **Static Outro & Chart Frames:** Added `disable_motion` to `KenBurnsRequest` so the final outro screen and all dynamic data charts render as high-quality static looped video clips instead of applying the Ken Burns pan-and-zoom effect, improving readability of text and graphs.
- **TTS Decimal Pronunciation:** Expanded decimal points in numbers (e.g. `3.5` -> `3 point 5`) inside `sanitize_tts_text` before passing text to Kokoro TTS to ensure natural fluid speech.

### 1.13 Semantic Topic Deduplication

- **High-Quality Semantic Novelty Checks:** `calculate_semantic_novelty_index` in `src/engine/text_embeddings.py` utilizes the active MiniLM SentenceTransformer semantic embedding backend when `USE_SEMANTIC_GATES=1` is configured. This provides a deep semantic cosine-similarity comparison against past published topics, falling back gracefully to a stop-word-filtered TF-IDF vectorizer when the neural backend is unavailable.
- **Synthesized Topic Gating:** The topic deduplication similarity check is applied directly inside `_measure_and_gate_synthetic` within `src/agents/fact_retriever.py` before querying the YouTube API for demand stats. This prevents synthesized evergreen tool-topics from bypassing the deduplication checks and ensures they are culled if they are semantically similar to any topic in `published_topics.json`.

---

## 2. PLANNED — NEXT PASS

API-verified growth features (pre-YPP: drive watch-time + subs). All follow the existing
non-fatal, quota-aware, env-gated patterns.

### 2.1 Auto-playlist chaining (watch-time amplifier)

- **`mcp_servers/youtube_cloud/server.py`** — `POST /tools/upsert_playlist_add_video`:
  `playlists.list(mine=true, part=snippet)` (the API has **no title filter** → match
  `snippet.title` client-side against env `YOUTUBE_PLAYLIST_TITLE`, default
  `LumenLoop AI Documentaries`); if missing, `playlists.insert` (body requires
  `snippet.title`); then `playlistItems.insert` with `snippet.resourceId`
  (`kind=youtube#video`, `videoId`) — **do NOT set `snippet.position`** (throws
  `manualSortRequired` unless the playlist uses manual sort). Scopes `youtube` /
  `youtube.force-ssl` (not `youtube.upload`). Quota: list 1 + insert 50 + item 50.
- **`src/schemas/state.py`** — add `playlist_id` / `playlist_url` to `UploadMetadata`.
- **`src/agents/publisher.py`** — after the real-video-id guard, non-fatal
  `await upsert_playlist_add_video(...)`; env gate `YOUTUBE_AUTO_PLAYLIST` (default 1).
- **Tests** — playlist side-effect gated off `demo_*` ids (like FAKE_UPLOAD_ABORTS); called
  once on a real-id path.

### 2.2 Burned-in subscribe end-card (subscriber conversion at peak intent)

- **`src/agents/media_producer.py`** — in the ffmpeg assembly path (moviepy path skipped),
  append a rendered subscribe clip to the master concat list: 1920x1080 ffmpeg `drawtext`
  ("SUBSCRIBE 🔔 new documentary every day") over a color fade; duration from env
  `CSVG_END_CARD_SECONDS` (default 6, `0` disables). Zero API cost / no quota.
- **Tests** — concat list includes the end-card entry when enabled; total ≈ sum(shots) + card.

### 2.3 YouTube comment-reply bot (comment velocity → dwell)

- **`mcp_servers/youtube_cloud/server.py`** — `POST /tools/list_comments`
  (`commentThreads.list` on `videoId`, `textFormat=plainText`, quota 1) and
  `POST /tools/reply_comment` (`comments.insert` with body `snippet.textOriginal` +
  `snippet.parentId` — **replies use `comments.insert`, NOT `commentThreads.insert`**,
  quota 50). Scope `youtube.force-ssl`.
- **`src/engine/youtube_engagement.py`** (mirrors `active_thread_seeder.py`) — fetch recent
  top-level comments, filter out own channel, cap `YOUTUBE_REPLY_MAX` (default 3), LLM-grounded
  replies from `state.verified_facts` (reuse the reply-generation prompt pattern).
- **`src/agents/publisher.py`** — non-fatal `await reply_to_viewers(...)` after the pinned-comment
  block; env gate `YOUTUBE_REPLY_BOT` (default 1).
- **Tests** — ≤ cap replies; self-comments skipped; gated off `demo_*` ids.

*Note:* the on-disk `token.json` scopes (`youtube`, `youtube.upload`, `youtube.readonly`)
satisfy these writes via the broad `youtube` scope; new endpoints must load credentials with the
`youtube.force-ssl` scope string (as `insert_pinned_comment` already does), not the
`youtube.upload`-scoped uploads loader.

### 2.4 Targeted discussion-injection seeding (Tier 1–3)

The growth goal is maximum views + subscribers → YPP → $2,000/month ad revenue. Broadcasting
multi-social posts into feeds yields low-RPM views and dilutes the viewer segment pre-YPP; the
high-leverage move is injecting value into **live, on-topic discussions/groups** the video is
intended for. The pipeline already does this on Reddit (`ActiveThreadSeeder` +
`SeedDistributor` semantic gate); this section extends that pattern. **100% automation — no
manual posting, no bots/channels set up by hand** — all email/password auth uses the Reddit
account-pool + Playwright pattern.

**Decisions:** Reddit scoping is **soft by default** (prefer relevant groups; global fallback
only as last resort) — `ACTIVE_SEEDER_STRICT_SCOPE=1` makes it strict-skip. All tiers planned
now; code lands in a later pass. Per-run budget = ≤1 post per platform.

**Shared foundation**
- `src/engine/seed_account_pool.py` (new) — generalize `RedditRotationState` +
  `reddit_accounts.json`: `{username/email, password, daily_cap, warmup, extra:{}}` +
  `settings {min_delay_seconds, max_delay_seconds, retire_on_shadowban, visibility_check_enabled}`.
  Pools live in the gitignored `seed_accounts/` dir (`reddit.json`, `quora.json`,
  `telegram.json`; keep `reddit_accounts.json` working via back-compat). Per-platform
  `logs/<platform>_rotation_state.json` (daily caps, retirement, per-target permissiveness,
  posted-URL dedup). Shared memory/process resource guards so concurrent seeds can't OOM the Pi.
- `BasePlatformSeeder` (`active_thread_seeder.py:12`) stays the interface: extend with
  `post_answer(...)` / `post_message(...)` where needed.
- Fan-out: single `seed_all_platforms(state, youtube_url)` in `publisher.py` (all non-fatal,
  ≤1 post each): Reddit active-reply (scoped) → Quora answer (gated) → Telegram group post
  (gated) → existing Reddit link-seeder + warmup.
- Account onboarding: `get_seed_account.py` (new) verifies login by automation, persists the
  session, runs a NO-LINK warm-up (mirror `reddit_warmup.py`). Creation of accounts (captcha/2FA)
  stays manual; everything after credential entry is automated.

**Tier 1 — Reddit scoping (soft; high value, low effort)**
- `seed_active_discussions(state, youtube_url, target_subreddits=None)`; publisher computes
  `seed_distributor.select_target_subreddits(state)` once and passes it.
- Search **in scope first**: PRAW `reddit.subreddit("+".join(subs)).search(query,
  time_filter="day")`; `RedditJsonClient.search_active_threads` filters `subreddit in scoped_subs`.
- **Thread-level relevance ranking:** score each candidate thread `title+selftext` vs video text
  via `seed_distributor._semantic_relevance` (MiniLM ≥ 0.50 / TF-IDF fallback ≥ 0.30); sort by
  score desc, tie-break `num_comments`. Pick the top scoped thread.
- **Soft fallback:** no qualifying scoped thread → loud warning + today's global
  highest-comments pick (preserves current behavior). `ACTIVE_SEEDER_STRICT_SCOPE=1` → skip.
- Tests (hermetic): off-topic 500-comment thread loses to on-topic 40-comment scoped thread;
  soft mode falls back globally when nothing qualifies; strict mode posts nothing; `_post_reply`
  never out-of-scope in strict.

**Tier 2 — Quora (medium effort)**
- `src/engine/quora_seeder.py` — `QuoraSeeder(BasePlatformSeeder)`:
  `search_active_questions(topic)` (Quora search, title + keywords),
  `get_question_context(question)` (question + top answers so the LLM won't repeat them),
  MiniLM relevance gate (question vs video text), `_build_answer(...)` → LLM (route `generate`)
  answering with `state.verified_facts[:3]` + one natural credit line
  (`*covered the data + visuals in a video here: <link>*`), `QUORA_REPLY_MAX=1`.
- `src/engine/quora_browser_poster.py` — clone of `RedditBrowserPoster` driving
  `seed_accounts/quora.json`; Playwright → www.quora.com, email/password login, DOM answer
  post, visibility check, retire on ban. Env `QUORA_CHROMIUM_PATH` (default `/usr/bin/chromium`).
- Gated by `QUORA_SEEDER_ENABLED` (default 0 until an account is onboarded).
- Tests: gated off → 0 posts; ≤1 answer; answer contains a verified fact + link; off-topic
  question skipped; account-pool shape equals `reddit_accounts.json`.

**Tier 3 — Telegram on-topic groups (low-medium; no bots, no manual setup)**
- `src/engine/telegram_seeder.py` — `TelegramSeeder(BasePlatformSeeder)`, **browser-only**
  (web.telegram.org; no Bot API, no BotFather, no manual channel invites):
  `search_active_groups(topic)` (public-group/channel search matching keywords, recent context
  where visible), MiniLM gate vs video text, `_build_message(...)` reuses
  `seed_distributor.telegram_post` (shortened, verified fact + YT link), `TELEGRAM_REPLY_MAX=1`.
- `src/engine/telegram_browser_poster.py` — Reddit-pattern browser poster;
  `seed_accounts/telegram.json` login; group-join via public link then post; session persistence;
  daily caps + retirement.
- Gated by `TELEGRAM_SEEDER_ENABLED` (default 0).
- **Risk note:** Telegram Web login is phone/QR-based in many regions — email/password parity may
  not hold on every account. Fallback: persist an already-logged-in session in the pool
  (`session` field, refreshed by automation) so zero manual per-post action is still required.
  Verify login realism at implementation time with a real account.
- Tests: gated; ≤1 message; content carries verified fact + link; account-pool parity.

**Env additions (`example.env`):** `ACTIVE_SEEDER_STRICT_SCOPE` (default `0`),
`QUORA_SEEDER_ENABLED` / `TELEGRAM_SEEDER_ENABLED` (default `0`), `QUORA_REPLY_MAX` /
`TELEGRAM_REPLY_MAX` (default `1`), `QUORA_CHROMIUM_PATH`.

**Verification:** full suite (19 + new cases) green, `bash -n`, no dangling config refs.
Execution order: T1 ships alone first (no accounts needed) → T2 → T3.

---

## 3. NOT FEASIBLE / DEFERRED

### Not feasible

- **RLHF/DPO on the writer** — treating the Observer as a reward signal to fine-tune the
  generative model requires an open model; Gemini is **not fine-tunable** and training is too
  heavy for 2-vCPU/12 GB.
- **Feed-set changes** (replacing global high-RPM feeds) — a monetization/quality invariant;
  blocked/promo/advertorial content must stay filtered out of RSS and RAG candidates.

### Deferred

- **Supervised reward classifier** — accumulate accepted-vs-rejected scripts and fine-tune a
  small DistilBERT to predict pass/fail; needs ~hundreds of labeled samples (not enough run data
  yet). Revisit after more runs.
- **Online threshold calibration for semantic gates** — record `{scores, verdict}` per run and
  nudge thresholds so false rejections trend to zero; deterministic, no training infra — cheap
  once more run data accumulates.
- **Channel trailer rotation** — set the newest documentary as the channel trailer via
  `channels.update(brandingSettings.channel.trailer)`; low yield at 0 subs, add post-GROWTH.

### Already exists (no action)

- **RAG-sufficiency gate** — guards factual depth before story design.
- **Cron mechanics** — `cron_publish.sh` + guards built and installed (`20 11` / `50 13` UTC).