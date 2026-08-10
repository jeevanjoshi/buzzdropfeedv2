# CSVG Converged Implementation Plan — Documentary gate + revenue-weighted dynamic region

_Status: ratified · supersedes the earlier fragmented plans (opportunity-score, RAG-alignment,
publish-integrity, script-quality, stabilization, media-quality upgrades). Updated after each
implementation pass so it stays the single source of truth._

## 0. Goal

1. **Never publish direct news.** A topic is only selectable when our automation can build a
   6-act documentary / investigative story out of it. Announcement / press-release / price-tick
   blips are culled before they ever reach ranking.
2. **Ad revenue in the region has the highest weight.** Regional ad revenue dominates both the
   TOPSIS ranking (8th criterion, highest single weight) and the region/market selection score.
3. **Region is dynamic.** The market a run targets is decided at selection time from day/time +
   topic affinity + per-market RPM + events, then passed downstream to story design, visuals and
   the revenue forecast (`state.region` / `state.region_market` / `state.region_reason`).

## 1. Shipped & verified (part of the converged plan)

| Area | Where | What |
|---|---|---|
| Contextual chapters | `src/engine/chapters.py`, `src/agents/story_designer.py`, `.publisher.py` | Per-act chapter labels derived from the shot narrations (LLM + deterministic fallback), stored on `SEOMetadata.act_titles`, reused at publish. |
| Niche thumbnails + correct stock art | `src/agents/media_producer.py` | Company-aware ticker resolution (Meta→META…), private-company guard, narration-grounded fallback moves (no fabricated prices), clean chart titles, topic-fused thumbnail scene. |
| Region-optimized cron | `cron_publish.sh` | Up to TWO publishes/day behind the $2,000/month goal; launch bands [11:00-12:20] & [13:30-14:30] UTC backed out of the ~1h50m runtime; guards (no double-run, daily cap `CSVG_MAX_DAILY_PUBLISHES=4`, cooldown 90m) → up to 2/day; journal `logs/cron_publish.log`. |
| Dynamic region plumbing | `src/engine/region_intelligence.py`, `fact_retriever.py`, `monetization_optimizer.py`, `orchestrator.py` | (topic, market) picked in the fact retriever; `state.region/region_market/region_reason` flow downstream; `run_production.sh`/`main.py` default to dynamic. |

### GROWTH phase (pre-YPP) — gap closures

The channel goal is $2,000/month (REVENUE phase), but GROWTH is the step before:
it must convert viewers into subscribers + watch-hours to unlock YPP. Closed gaps:

- **Shorts were generated but never published** — `micro_content_producer` already cut
  9:16 vertical 30-45s clips, but nothing uploaded them (`UploadMetadata.shorts_video_id`
  was never set). Added `mcp_servers/youtube_cloud` `/tools/upload_short` (same OAuth
  resumable insert + `#Shorts` tag + EU-AI-act disclosure), and the publisher now **uploads
  the Shorts in GROWTH phase** (non-fatal, quota-shared) — the #1 discovery/subscriber lever
  for a pre-YPP channel while the long-form master stays the monetized asset.
- **Pinned comment was phase-agnostic** — GROWTH now pushes subscribe + watch-time in the
  pinned comment; later phases keep the pure engagement question.
- **Description had no subscribe CTA in GROWTH** — appends a subscribe/bell + daily-series
  line so discovery metadata converts viewers into subs.
- **Resolved — GROWTH is discovery-led.** Pre-YPP the channel earns $0 in ads, so the
  GROWTH TOPSIS vector leads with novelty + trend (IDI 0.25 / TVS 0.25) for watch-time and
  subscriber growth, with regional revenue secondary (0.15); REVENUE/SCALE keep regional
  revenue at the single highest weight. Unlocks YPP faster → monetizes sooner.

## 2. Storability gate (this change)

New module `src/engine/documentary_potential.py` (deterministic-first, offline).

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

## 3. Revenue-weighted region (this change)

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

## 4. Selection order (fact retriever)

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

## 5. Configuration (env, defaults)

`DOCUMENTARY_POTENTIAL_FLOOR=0.35` · `DOCUMENTARY_LLM_CROSSCHECK=0` · `REGION_WEIGHT_REVENUE=0.55` ·
`REGION_REVENUE_REFERENCE_USD=16`.

### Revenue-goal alignment ($2,000/month — single source of truth)

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

## 6. Validation

- `python run_tests.py` — 18 hermetic cases incl. `STORABILITY_GATE` (probe/scandal documentary,
  press-release blip culled, evergreen pass-through, all-culled→best-warning) and
  `REGION_REVENUE_DOMINATES` (rba→au, sensex→india, meta→us; ALPHA>BETA>GAMMA; revenue is the
  highest TOPSIS weight in REVENUE & GROWTH; top-ranked by regional revenue).
- `bash -n run_production.sh cron_publish.sh`.

## 7. Out of scope

Feed changes (global high-RPM feeds stay), RAG-sufficiency gate (exists), cron mechanics (built).