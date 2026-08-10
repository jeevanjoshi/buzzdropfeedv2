# CSVG — Publish Integrity & Narration/Sources Quality — Implementation Plan

**Date:** 2026-08-10 (rev 3 — issues #1/#2/#3 IMPLEMENTED & tested 12/12 green)
**Scope:** Fix the 6 issues found in the post-publish review of the ostrich run
(`csvg-exec-20260809-194005`, real upload `E1T5IiXSl3E`) and the Meta run
(`csvg-exec-20260810-113142`, `video_id=demo_451614aa`).
**Priority ordering** follows the review: a masked fake-upload first, then
description sources, then narration junk, then off-topic stat leakage, then the
two polish items.

### Implementation status (rev 3)
| Issue | Status |
|---|---|
| #1 demo_/fake upload ids ABORT (server, publisher, resume guard) | ✅ done (11 tests green) |
| #2 SEO description sources filtered on-topic | ✅ done (SEO_SOURCE_FILTER) |
| #3 byline/follow/hashtag junk scrubbed + hard-gated | ✅ done (NARRATION_JUNK_SCRUB) |
| #4 off-topic market-stat leakage — fixed at RAG ingestion (IDF self-maintaining register) | ✅ done (SIGNATURE_INGESTION_FILTER + TERM_REGISTER_GROWTH) |
| #5 `_best_synonym` quality guard (`eld`/`powerfulness`) | ✅ done (BEST_SYNONYM_GUARD) |
| #6 chapter timestamps — shared source of truth (`compute_act_chapters`) | ✅ done (CHAPTER_TIMESTAMPS) |

### Rev 2 delta (what upstream `f01c3c4` changed, and how the plan adapts)
- Upstream added the Seed Traffic Seeding & Distribution Pipeline (`publisher.py`,
  `seed_distributor.py`, `micro_content_producer.py`, `insert_pinned_comment` in
  `youtube_cloud/server.py`).
- **Issue #1 is now WORSE:** after upload, `publisher.py:145` takes
  `video_id = upload_res.get("video_id", "demo_id")`, then posts the pinned
  engagement comment, generates shorts and dispatches the seed package — **all
  against the fake `demo_*` id** — and finally records `PUBLISHED_SUCCESS` +
  `topic_deduplicator.record_published_topic(...)` (so the topic is permanently
  marked published even though no video exists). The upload mock in
  `youtube_cloud/server.py` and the new `insert_pinned_comment` mock use the same
  swallow-and-return-`demo_*` pattern. Every downstream step must be gated on a
  **real** video id, and the click-path is the single check.
- **Issue #6 (chapters) is MOSTLY fixed upstream:** `publisher.py` now computes act
  start-times from `asset_paths.measured_durations` + crossfade and rewrites the
  published description chapters + `seo.chapter_timestamps` (replacing the static
  block). The hanging work is (a) `story_designer.generate_seo_metadata` still
  emits the static `chapter_timestamps`/chapter block (story_designer.py:1532-1538,
  1560-1563) that publisher trusts as `old_chapters_pattern`, and (b) the SEO
  state is patched AFTER `publish_video` starts, so the description is updated
  only via the replace-regex path, not computed at design time. Scope the SEO-side
  computation to match publisher's so there is one source of truth.
- Verified no overlap conflicts: upstream's `publisher.py`/`youtube_cloud/server.py`
  edits don't collide with the plan's defense-in-depth layers (they add code paths,
  the plan's `demo_*` reject wraps the same upload result).

### Shorts/seed hang-over to include
`publisher.publish_video` step order must become: **real-id check FIRST** → upload →
pin comment → shorts → seed dispatch. If the id is fake, abort before any side
effect and before dedup recording (currently dedup fires unconditionally).

---

## 1. CRITICAL — `demo_*` upload IDs must ABORT, not fake PUBLISHED_SUCCESS

### Root cause
- `mcp_servers/youtube_cloud/server.py:123-134` wraps the entire real upload in
  `try/except Exception: pass` and ALWAYS returns `status: "success"` with a
  fabricated `mock_video_id = f"demo_{os.urandom(4).hex()}"` (line 127). Any real
  failure (expired token, 403, quota, network) is silently swallowed → mock id.
- `src/agents/publisher.py:57-61` then writes `UploadMetadata(video_id=demo_..., status="PUBLISHED")`
  and `execution_stage="PUBLISHED_SUCCESS"`. The pipeline logs
  *"Pipeline Run Completed Successfully! Video ID: demo_451614aa"* — nothing was uploaded.
- Observation that proved it: the Meta run (13:17, expired/unrefreshable token)
  produced `demo_451614aa`, while the ostrich run (14:16) produced a real `E1T5IiXSl3E`.
  The healthcheck at 13:54 still shows `WARN insufficient auth scopes`.
- Worse: `orchestrator.py:581` resume-guard only skips re-publish when
  `video_id != "demo_id"` (the exact literal), so a `demo_<hex>` id is treated as
  "already published" on resume → a fix/retry can never re-upload.

### Fix (defense in depth, 3 layers)
1. **`youtube_cloud/server.py` — stop lying (upload AND comment endpoints).**
   - Upload endpoint (lines ~123-134): never `return` the mock block as success.
     On real-upload failure, log the actual exception via `logger`/`print` (it is
     currently invisible) and either `raise HTTPException(502, ...)` or return
     `{"status": "mock", "video_id": "demo_...", "reason": str(e)}`.
   - Keep the mock path ONLY behind an explicit opt-in (e.g. env
     `YOUTUBE_UPLOAD_MOCK=1` or a real `--offline`-style flag) so hermetic/local
     tests can still simulate, but production never silently mocks.
   - Fix the secondary leak at line ~105: `response.get("id", "uploaded_demo_id")`
     must NOT default to a fake id — if `id` is missing treat it as failure.
   - New `insert_pinned_comment` (lines ~143-190): same swallow-and-mock pattern
     (`comment_<hex>` fake id). The comment insert is an OPTIONAL engagement
     side-effect: on failure it may warn, but it must not gate publish success.
     Apply the same explicit-mock-only rule as the upload endpoint.
2. **`publisher.py:145` — check the id BEFORE any side effect.**
   - Immediately after `upload_youtube_resumable`, reject any `video_id` that is
     falsy, equals `"demo_id"`, equals `"uploaded_demo_id"`, or starts with `demo_`:
     raise `RuntimeError("YouTube upload fell back to mock (video_id=...); not publishing.")`
     so the orchestrator aborts (existing exception path at orchestrator.py:597).
   - Move the pinned-comment (step 3), shorts (step 4) and seed-dispatch (step 5)
     AFTER this check so a fake id never triggers side effects, and keep
     `record_published_topic` after the real-id check too (it currently fires
     unconditionally at the end) — a mock must NOT poison the dedup history.
   - Only write `PUBLISHED_SUCCESS` after a real id is returned.
3. **`orchestrator.py:581` — resume guard must use the same predicate.**
   - Change the resume check from `video_id != "demo_id"` to a shared helper
     `_is_real_video_id(id)` = id is set AND not `demo_id` AND not `demo_*` AND not `uploaded_demo_id`.
     Place the helper in `publisher.py` (or `youtube_cloud/server.py`) and import it,
     so fix/retry runs correctly re-publish a previously-mocked run.

### Alignment with existing invariants
Publish path already enforces quota + EU AI Act synthetic disclosure (AGENTS.md);
we are hardening *truthfulness of the success signal*, not the content rules.

### Tests
- `tests/test_hermetic_e2e.py`: StubPublisher returns `hellermetic-video-123` today (line 295-301).
  Add a variant stub/asynchronous branch that returns a `demo_*` id and assert the
  orchestrator raises (pipeline does NOT reach `PUBLISHED_SUCCESS`).
- Unit-test `_is_real_video_id` across `demo_id`, `demo_<hex>`, `uploaded_demo_id`, real ids.

---

## 2. HIGH — SEO description "Sources & Data Grounding" polluted by off-topic facts

### Root cause
`story_designer.generate_seo_metadata` (story_designer.py:1506-1558) iterates
**every** `state.verified_facts` entry (line 1520-1525) and dumps each URL into the
published description. `verified_facts` = the whole RSS/enrichment corpus (~35 facts,
mostly unrelated: Mars rover, hurricanes, satellites, Unitree IPO, SpaceX moon
crash, a museum housecoat, World Bank/Wikipedia placeholder links). The Meta run
published a description listing all of them.
(The ostrich run's description only shows the clean primary source because it was
generated before HEAD `79cfe48`; the current code is the regression risk.)

### Fix
Filter + cap the fact list to topic-relevant facts before building `source_links`:
1. Extract a reusable on-topic scorer from `rag_retriever._on_topic_hits`
   (rag_retriever.py:93) + `_build_topic_tokens` (rag_retriever.py:81). Add a public
   helper `rag_retriever.filter_facts_for_topic(verified_facts, topic,
   min_hits=2, max_facts=6)`.
   - Keep at minimum the topic's own story (source_url / matching headline).
   - Drop placeholder/generic sources (`The World Bank`, `Wikipedia Historical
     Archives`, `NASA Open APIs`, `Verified Reports`, `Verified Market Reports`,
     generic `wikipedia.org`/`data.worldbank.org` roots).
   - Sort by hit-count desc, cap at `max_facts` so descriptions stay tight.
2. Call it inside `generate_seo_metadata` (replace the raw loop at 1520-1525) so it
   never regresses, without touching the RAG pack that uses a stricter
   threshold already.
3. Keep the existing dedup (`seen_urls`) and the singular primary-source line.

### Acceptance
Re-generating SEO metadata for the Meta topic yields ≤ ~6 sources, all containing
`meta`/`zuckerberg`/`muse`/`spark` topic tokens (or the primary source), no
World Bank/Wikipedia/`historyextra` noise.

---

## 3. HIGH — raw scraped BYLINE/SOCIAL junk in narration outro (Meta Shot 18)

### Root cause
Meta Shot 18 narration ends with raw article chrome:
`# Meta launches new artificial intelligence model as Zuckerberg champions
open-weight push. Tech Meta launches Muse Glimmer model as Zuckerberg champions
artificial intelligence for 'everyone' ByTom Carter You're currently following
this author! # Meta launches Muse Spark artificial intelligence model...`
`_SNIPPET_JUNK_RE` (story_designer.py:255-266), `_RAW_JUNK_RE` (53-56) and the
Observer's `_RAW_JUNK_IN_NARR_RE` (observer.py:48-57) do NOT match this block —
the byline pattern `By<FirstName> <LastName>` (no space after "By"), the
"following this author" follow-prompt and the standalone `#`-hashtag lines are
not in any junk corpus. It entered via the snippet-expansion path
(`expand_narration_with_semantic_facts`, story_designer.py:637-685) whose
`_append`/`_paraphrase_padding` only strip connectors, not junk.

### Fix
1. Extend the junk regexes (all three sites, kept in sync):
   - byline w/o space: `\bBy[A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+` and `You'?re currently following this author`.
   - remove stray `# Name ...` tag lines from narration (they read as hashtag slop).
   - `Read? More`, `Subscribe to continue`, `Sign in to`, `loading...`, `View: `,
     `SHARE: ` variants.
2. Re-apply `_snippet_is_junk` inside `_append`/`_paraphrase_padding` (before
   appending) so a bad snippet can never become narration padding, matching the
   corpus-side guard in `parse_raw_snippets` (story_designer.py:448).
3. Observer: add the new patterns to `_RAW_JUNK_IN_NARR_RE` (hard gate, NOT
   soft-approvable) so this class of leak always aborts even if the writer regresses.
4. Consider a trailing-slop sweep in `_clean_narration` (story_designer.py:103-122):
   drop the clause after the last sentence-level junk trigger as a final net.

### Acceptance
The exact Shot 18 tail from the Meta run is rejected as junk by
`_snippet_is_junk`; a re-run cannot append it as narration. Hermetic test adds a
narration containing the observed bytes and asserts the Observer hard-flags it.

---

## 4. MEDIUM — off-topic market/organisation stats narrated + charted (ostrich Shot 11) — SOLVED AT SOURCE

> **Note (rev 3):** per the operator's direction ("fix at source so the issue doesn't
> occur right first time — not a per-violation gate"), this issue was solved at
> RAG **ingestion** rather than by adding another detector. No market-stat gate.
>
> **Note (rev 4):** the generic-token definition is now **corpus-driven and
> self-maintaining** (`src/engine/term_register.py`, IDF-style), replacing the
> hardcoded `_CITATION_GENERIC_TOKENS` denylist (removed — no more manual
> word-curation).

### Root cause
Ostrich Shot 11 narrates *"The Contract Research Organizations market ... was
estimated at 69.56 billion USD ... predicted to grow to 74.37B ... PrecedenceResearch.com"*
and renders a matching `matplotlib_chart`. That CRO stat leaked in from
`rdworldonline.com`/`precedenceresearch.com` contamination in the fact corpus and
passed the on-topic filter `_on_topic_hits >= 2` because it shares **generic**
tokens (`researchers`, `research`) with the topic's own summary.

### Fix (right-first-time — root cause, not a gate)
Centralise a **signature-token set** — `_topic_signature_tokens()` in
`rag_retriever.py` — that strips high-frequency niche/news words from the topic
tokens, and use it at EVERY ingestion/scoring site so an off-topic line scores 0
and never enters the corpus in the first place:
1. `build_rag_knowledge_pack` verified-facts ground-truth scoring (was line ~1079).
2. `build_rag_knowledge_pack` retrieved-line on-topic filter (was line ~1177) —
   **this is where the CRO line died**.
3. selected-article injection gate (was ~1224).
4. `assess_corpus_sufficiency` (was ~940) — so a polluted-but-large corpus can't
   pass the budget check on generic word counts.
5. `filter_facts_for_topic` (the SEO citation list, issue #2) now reuses the same
   helper — one definition everywhere.

Because the corpus/verified-facts/snippets are now clean, the chart builder (which
reads numbers off verified_facts) has nothing off-topic to render, and the
Observer/`_clean_narration` never see the line. No new gate needed.

### What "generic" means now (self-maintaining IDF)
`src/engine/term_register.py` — `TermRegister` accumulates **document frequency**
over the corpus and reports a token as generic when it appears in a high fraction
of background documents (`DF_RATIO_THRESHOLD = 0.18`):
- **In-run:** `build_rag_knowledge_pack` observes the day's WHOLE feed pool
  (all candidate topics' facts, not just the winner); a circulating leak word is
  generic that run.
- **Across runs:** counts roll up to `logs/term_register.json` (gitignored via
  `logs/*.json`); a historically-generic word stays excluded even on a quiet day.
- **Growth/self-maintenance:** the register grows on its own from normal runs —
  no hand-curated list, no per-word edits.
- **Known limit (transparency):** frequency catches recurring/ubiquitous leak
  vocabulary only. A word that leaks exactly ONCE, appearing ONLY in the winning
  topic's own snippets, will not yet have high df — IDF cannot know it was a
  mistake. That one-shot novel case is explicitly out of scope for a frequency
  model (it would require explicit failure feedback, a separate future hook).
- **Offline/empty safety net:** a small `_BASELINE_GENERIC` stopword set (the
  former tuned list, preserved) guarantees sane behavior on a fresh register.

### Acceptance
`SIGNATURE_INGESTION_FILTER` + `TERM_REGISTER_GROWTH` hermetic cases:
- the exact CRO line scores 0 on the archaeology topic's signature set (dropped);
  a genuinely on-topic line (ostrich eggshell + geometric) still passes (`>=2`);
- a word circulating through many background docs becomes generic **automatically**;
  a rare/distinctive word (`ostrich`) and a single occurrence (1-of-100) do NOT;
- learned generic words **persist** across register reloads.
Verified on the live ostrich corpus: `research`/`researchers`/`model`/`found`/
`study`/`scientist` auto-detected generic; `ostrich`/`eggshell`/`engravings`/
`geometric`/`zuckerberg`/`muse`/`spark` stay distinctive. Sufficiency still passes
(44 facts / 1804 words / 24 sources).

---

## 5. LOW — stilted WordNet dissolve substitutions (`eld`, `powerfulness`, `capableness`) — DONE

### Root cause
`_best_synonym` (story_designer.py:572) returned the **first** WordNet lemma,
which can be archaic/rare (observed `years→eld`, `ability→powerfulness`,
`capableness`). `_dissolve_verbatim_copies` then happily swapped them in.

### Fix (synonym-quality guard in `_best_synonym`)
- Two-pass selection that PREFERS COMMON words:
  1. Prefer a lemma whose WordNet synset lists **multiple lemmas** (popularity
     proxy — a well-used word) and that passes all guards.
  2. Fall back to any acceptable single-lemma-synset alternative.
- Guards added/extended (single word, ≤14 chars preserved):
  - `_STILTED_SYNONYMS` denylist of 28 archaic/over-formal words (`eld`,
    `powerfulness`, `capableness`, `aforementioned`, `wherewithal`, `whilst`,
    `thou`/`thy`/`thee`, ...) plus a blanket reject of bare `-ness`/`-ment`
    nominalisations.
- If no acceptable synonym exists, `_best_synonym` returns `None` → the dissolve
  **leaves the word unchanged** instead of injecting a rare word (bounded loop
  then ends). No regression in the surgical/verbatim flows.

### Acceptance
`BEST_SYNONYM_GUARD` hermetic case (deterministic fake WordNet): `years→age`
(not `eld`), `ability→power` (not `powerfulness`), `capableness→capability`,
`powerfulness→power`; a token whose only candidate is stilted yields `None`.
Verified against real NLTK WordNet: same outcomes; `capableness`→`capability`.

---

## 6. LOW — hardcoded chapter timestamps don't match real runtime — DONE (shared source of truth)

### Root cause
`generate_seo_metadata` hardcoded `0:00/2:15/4:30/6:45/9:00/11:15` chapters
(story_designer.py:1532-1538, 1560-1563) regardless of actual runtime — the meat
video is 13:23 (803.4s). The `chapter_timestamps` schema field (state.py:28) was
also hardcoded there. Upstream `f01c3c4` added a *separate* computation in
`publisher.publish_video` (from `measured_durations` + crossfade) that rewrote the
description via a brittle regex, but left `generate_seo_metadata` static — so the
design-time `chapter_timestamps` (dashboards/notifications) and the static
description block were still wrong, and any shape drift broke the replace.

### Fix
New shared helper `src/engine/chapters.py:compute_act_chapters(shots,
measured_durations, crossfade, act_names)` — ONE definition for the CHAPTERS
block + `chapter_timestamps` list:
- `story_designer.generate_seo_metadata` calls it with `script.shots`
  (`duration_estimate` fallback) → design-time timestamps are now computed, not
  static, and the description block is emitted from the helper.
- `publisher.publish_video` calls the SAME helper with `measured_durations` +
  `crossfade` → replaces the desc block & `seo.chapter_timestamps`. Same code
  path both times, so no regex drift; the old static text is still tolerated as a
  replacement target but is no longer the source.
- Static `0:00/2:15/...` values remain ONLY as a no-shot fallback (legacy layout).

### Acceptance
`CHAPTER_TIMESTAMPS` hermetic case: computed times advance per actual shot
duration (Act 2@0:45, Act 6@3:45 for 6×45s), Act 1 anchored at 0:00, crossfade
overlap subtracts (2×45s − 5s crossfade ⇒ Act 2@0:40), and the static fallback
still emits the legacy block. Verified on the live Meta script: design-time
chapters now read 2:15/4:25/6:33/8:39/10:45 (computed) instead of static 11:15.

---

## Out of scope (explicitly not in this plan)
- Re-uploading the already-published Meta `demo_` run (it was never
  uploaded; re-publish needs the fixed code + a healthy OAuth token, which is
  `get_youtube_token.py` re-auth — ops, not code).
- Retraining/learned gates, scheduling, RAG caching (see
  `improvement_20260810_074500.md` §5).
- The `healthcheck.py` uncommitted WARN-level softening (separate change; not
  reverted here, but note: an upload mock must be a FAIL at the publisher layer
  regardless of healthcheck verdicts).

## Files touched (summary)
| File | Change |
|---|---|
| `mcp_servers/youtube_cloud/server.py` | stop swallowing upload/comment exceptions; opt-in mock (`YOUTUBE_UPLOAD_MOCK`); no fake `id` default |
| `src/agents/publisher.py` | reject `demo_*`/fake ids BEFORE comment/shorts/seed/dedup side effects → raise; expose `_is_real_video_id`; chapters via shared `compute_act_chapters` |
| `src/agents/orchestrator.py` | resume guard uses `_is_real_video_id` |
| `src/engine/rag_retriever.py` | `filter_facts_for_topic`, `_topic_signature_tokens`, IDF-driven generic detection at all ingestion sites; removed hardcoded denylist |
| `src/engine/term_register.py` | NEW: corpus-driven, persistent generic-token detector (self-maintaining IDF) |
| `src/engine/chapters.py` | NEW: shared `compute_act_chapters` (single source of truth for CHAPTERS + timestamps) |
| `src/agents/story_designer.py` | SEO source filter; byline/follow/hashtag junk; `_best_synonym` quality guard; design-time chapters via helper |
| `src/agents/observer.py` | extend `_RAW_JUNK_IN_NARR_RE` (hard) |
| `tests/test_hermetic_e2e.py` | demo-mock publish aborts; junk/stat/byline fixtures; SEO source filter; ingestion filter; term-register growth; synonym guard; chapter fixture |

## Acceptance gate
`python run_tests.py` passes (16/16) and `py_compile` stays clean on every touched file.