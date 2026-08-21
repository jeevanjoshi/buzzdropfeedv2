# Robustness Review & Remediation Plan — `buzzdropfeedv2`

**Scope:** entire codebase · **Severity:** all High + Medium + Low · **Includes:** error handling, logging, retries, resume/checkpoint · **Logging:** normalize `print`→structured `logger` + `exc_info=True`.

## Verification gate (run after every phase — must stay green)
- `ruff check src/ mcp_servers/ tests/ *.py --select F821,F822,F823`
- `python run_tests.py` (30/30 hermetic)
- `python healthcheck.py` (pre-flight)

---

## 1. Orchestrator / Resume / Checkpoint

**H1 — Non-atomic checkpoint write + unsafe load** (`src/engine/tracer.py:63-66`, `main.py:131`)
- Write via `tempfile.NamedTemporaryFile` → `os.fsync` → `os.replace` (atomic). Never overwrite the prior good file until the new one is durable.
- `main.py:131` `GlobalState.model_validate_json(f.read())`: wrap in try/except; on corrupt/partial file log clearly and fall back to most-recent valid checkpoint (or start fresh for `demo_`/test pids) — never crash the entrypoint.

**H2 — Resume stage-gating bug** (`src/agents/orchestrator.py:281-285`, `504-513`)
- `APPROVED_SCRIPT_STAGES` includes dead `"MEDIA_READY"` and omits the real `"MEDIA_PRODUCED"` (set at `media_producer.py:1608`). Fix: drop `MEDIA_READY`, add `MEDIA_PRODUCED`.
- When resume forces script regeneration, also invalidate `final_video`/`asset_paths` so a stale on-disk video is not reused against a new script (prevents wrong-narration publish + Gate 3b mismatch).

**H3 — Unintended publish on resume** (`src/schemas/state.py`, `orchestrator.py:604-608`, `main.py:71`)
- Add `publish_intent: Literal["publish","till_upload"]` to `GlobalState`; persist it. On resume honor the checkpointed intent instead of defaulting `publish=True`. A `QUALITY_VERIFIED` checkpoint (set by `--till-upload`) must NOT publish on resume unless explicitly re-requested.

**M4 — Concurrent-run guard** (`orchestrator.py`, `run_production.sh`)
- `pipeline_id`-keyed `flock`/pidfile at `run_pipeline` start; abort if another live process holds it (prevents double media-prod + duplicate upload).

**M5 — Region overwrite on resume** (`orchestrator.py:120`)
- On resume, don't clobber checkpointed `state.region`/`region_market`/`region_reason` unless an explicit `--india`/`--global` override is passed.

**M6 — No per-stage retry/backoff in orchestrator** (`orchestrator.py:166-645`)
- Top-level try/except wraps all stages; acceptable by design (rely on resume), but document the contract. Optional: add bounded per-stage retry for `media_producer.process` / `publisher.process` transient faults.

**M7 — No global timeout/watchdog** (`orchestrator.py` `run_pipeline`)
- Add an optional stage deadline + watchdog that kills a wedged run (stuck Pi TTS / hung SSH rsync) so autonomic cron misses the window loudly instead of hanging forever.

**L8 — `print` in entrypoint/tracer** (`main.py:128-156`, `tracer.py:69`): move to `logger` (with `component="ORCHESTRATOR"`/`CHECKPOINT`).

**L9 — `finally` swallows budget-save errors** (`orchestrator.py:667-668`): log the swallowed `Exception` with `exc_info=True` (keep from masking the real error, but record cost-ledger loss).

**L10 — Weak `compute_state_hash`** (`src/schemas/a2a.py:37-63`): hash also `duration_estimate`, `chart_spec`, and shot count/ordering so materially different scripts can't collide and weaken the stale-REVISE protection.

**L11 — Corrupt-checkpoint UX** (`main.py:131`): friendly message + fallback (covered by H1 load fix).

**L12 — Write-amplification** (`tracer.py:66`): re-serializes full `GlobalState` every step. Optional later: size cap / rotation / diff checkpoint.

---

## 2. LLM Client (`src/engine/llm_client.py`)

**H1/H2/H3 — JSON-retry regression + dead code (one root cause)**
- In `json_mode`, validate `content` parses (or is non-empty) before returning. On parse failure / `finish_reason == "length"` (truncation), run the **currently-dead** warning blocks at `363-366` / `436-440` / `505-509` and `continue` to next `retry_idx` / next model instead of `return content`. Restores lost malformed-JSON retry + removes 3 unreachable blocks.
- Inspect `finish_reason == "length"` in Vertex (`380-446`) and Gemini (`448-515`) branches (currently never checked) → treat as retryable truncation.

**L1 — Return annotation mismatch:** `try_local_llama_cpp`/`try_cloud_api`/`try_vertex_api`/`try_gemini_api` annotated `-> Optional[Dict]` but return `str`. Change to `-> Optional[str]`.

**L3 — Inconsistent logging:** dispatch/entry points + several error bodies use `print` (`219,233,239,252,258,303,387,456,473,…`); normalize to `logger`.

**L4 — Local llama.cpp empty content:** `try_local_llama_cpp` returns `""` on 200-with-no-body; validate non-empty before returning (low impact, last-resort path).

---

## 3. Media Production (ffmpeg/async hang + abort guards)

**H1 — `media_producer.py:1425`** merge subprocess: add `timeout=300` + `-nostdin`, wrap in `try/except` (currently can hang forever and leak `temp_specialized_path`).

**H2 — `mcp_servers/media_cloud/server.py:1110,1152`** final assembly: add `timeout` + `-nostdin` (the `apply_ken_burns_motion` at `:918` already does this — make consistent).

**H3 — `media_cloud/server.py:297,311,398,781`** per-shot GIF/SVG/chart renders: add `timeout=120-180`.

**H4 — `media_producer.py:1390`** `apply_ken_burns_motion` unguarded (raises `HTTPException(500)` on ffmpeg error → aborts whole run). Wrap and fall back to a static hold.

**M1 — Blocking I/O in async:** `media_producer` calls `media_cloud` route handlers in-process with `await`; those do sync `fal_client.submit().get()`, `requests.get()`, `subprocess.run` up to 300s on the event loop. Move blocking work into `asyncio.to_thread` (or call over HTTP as designed) so timeouts are enforceable.

**M2 — `print`→`logger`** in `media_producer.py` + `nano_banana.py` (matches `micro_content_producer.py`'s structured logger).

**M3 — Local TTS fallback unguarded** (`media_producer.py:1138-1146`): wrap `synthesize_tts(...)` so unexpected raises fall back gracefully instead of aborting.

**M4 — `generate_image` retry too narrow** (`nano_banana.py:213-231`): only retries quota/429; broaden to transient network errors; fix misleading "gave up" log on early `break`.

**M5 — `produce_all_media` not idempotent:** add "skip if `final_video` + all shot assets exist" (cheap resume optimization; only the paid-image visual cache is currently reused).

**L1 — `reference` bytes unvalidated** (`nano_banana.py:197-206`): size-cap/resize before `types.Part.from_bytes` to avoid avoidable API rejection.

**L2 — Redundant `config` reassignment** (`nano_banana.py:197-201`): the `if reference:` block duplicates `:193-196`; simplify.

**L3 — `media_budget.py:36-41`** read-modify-write not atomic: use `tmp + os.replace`.

**L4 — Swallowed `except Exception` in `nano_banana`** (`add_thumbnail_text:448`, `craft_ctr_hook:1133`, `comply_thumbnail:526`, `fetch_link_video_metadata:983`): add `exc_info=True`.

**L5 — Fragile `"ydl" in dir()`** (`nano_banana.extract_video_frames:914`): capture filename explicitly inside the `with` block.

**L6 — Missing `-nostdin`** on `media_producer.py:459,497,1524` ffmpeg probes: add for consistency.

---

## 4. YouTube / Reddit External Calls

**H1 — Infinite upload retry loop** (`mcp_servers/youtube_cloud/server.py:323-334` `_resumable_upload`): cap attempts (e.g. 12) with backoff + elapsed-deadline; re-raise after cap.

**H2 — No Google API socket timeout** (`youtube_cloud/server.py` `build()` at `107,176,218,290,389,443,533,580`; `verifier.py:370`): pass `http=httplib2.Http(timeout=60)` at every `build()`.

**H3 — yt-dlp / subprocess no timeout** (`youtube_video_verifier.py:87,131,434`): add `timeout=300` + `TimeoutExpired` handling.

**M1 — Duplicate live upload on kill-between-upload-and-checkpoint** (`publisher.py:294-379`): persist `upload_metadata.video_id` to a durable marker immediately after successful upload, before side effects.

**M2 — Quota ignores insert/comment/playlist cost** (`publisher.py:208`, `youtube_cloud:139`): track `thumbnails.set`/`playlistItems`/`commentThreads`/`comments` units (not just 1600/upload) or query live quota.

**M3 — Verifier temp-file leak** (`verifier.py:64-92,115-137`): clean `/tmp/yt_verifier/*` audio/VTT via `try/finally`/`tempfile`.

**M4 — Reddit context leak** (`reddit_browser_poster.py:418-465`): `ctx.close()` in `try/finally`.

**M5 — No inter-post pacing in seeding** (`active_thread_seeder.py:193-306`): add 30-90s sleep between thread comments (scaled by account age) to avoid AutoMod/rate-limit.

**M6 — Engagement no quota/pacing** (`youtube_engagement.py:25-102`): small `asyncio.sleep` between replies + consult quota.

**L1 — yt-dlp binary resolver inconsistency** (`verifier.py:433` hardcodes `./venv/bin/yt-dlp` vs `:76-78` resolver): unify.

**L2 — `WhisperModel` re-loaded per call** (`verifier.py:185`): cache on `self`.

**L3 — `print`→`logger`** in `youtube_video_verifier.py` + `publisher.py` (dashboard consumes structured logs).

**L4 — Webhook dispatch no retry** (`seed_distributor.py:342-390`): add short retry/backoff + explicit `ClientTimeout`.

**L5 — `reddit_rotation_state.json` non-atomic** (`reddit_browser_poster.py:50-53`): use `tmp + os.replace` (like `api_usage._atomic_dump`).

**L6 — Redundant double click** (`reddit_browser_poster.py:295-296`): drop duplicate `Log In` click.

**L7 — `reddit_link_seeder.py:207` `force=True` re-post risk:** guard against rotation-state reset (tie to L5 atomicity).

---

## 5. RAG / Ingestion Data Integrity

**M1/M2 — Fabricated fallback data + silent swallows** (`src/engine/external_apis.py`):
- `fetch_world_bank_gdp_inflation` (`71-99`) returns hardcoded `{"gdp_growth":"6.8%","inflation":"5.1%"}` on any exception.
- `fetch_alpha_vantage_stock_quote` (`101-122`) returns `{"price":"$125.40","change":"+3.45%"}`.
- `fetch_alpha_vantage_stock_history` (`124-206`) fabricates synthetic 5-point trend.
- `fetch_marketaux_*` / `fetch_exa_*` use bare `except Exception: pass`.
- Fix: return empty/None on failure; add bounded retry + logged `warning`; never inject fabricated numbers into `verified_facts`.

**M3 — No retry on RAG retrieval** (`rag_retriever.py` `search_*_facts` `546-758`; `grounded_search.py:259-275`): add single bounded retry/backoff with logged 429 handling per facet/crawler.

**M4 — Unguarded `nltk.download`** (`observer.py:223-224`): wrap `punkt`/`averaged_perceptron_tagger` downloads in try/except (mirror `rss_ingestion._compute_sdi:344-354`) so a missing corpus + no network can't abort the audit hot path.

**L2 — Misleading dead assignment** (`story_designer.py:833` `self.last_llm_source = "FALLBACK_GROUNDED_TEMPLATE"`): remove (no fallback path exists; both branches `raise`).

---

## 6. Cross-cutting Logging Normalization (per request)
- `print(...)` → structured `logger` (with level + `component=`) in `media_producer.py`, `nano_banana.py`, `llm_client.py` (remaining spots), `youtube_video_verifier.py`, `publisher.py`, `main.py`, `tracer.py`.
- Add `exc_info=True` to all intentional "never abort" `except Exception` swallow sites for debuggability.

---

## Implementation order
1. **Phase 1** Orchestrator/Resume/Checkpoint (correctness-critical) — then run tests.
2. **Phase 2** LLM client (regression + dead code).
3. **Phase 3** Media production timeouts/guards + logging.
4. **Phase 4** YouTube/Reddit external calls (hang loops, timeouts, leaks, pacing).
5. **Phase 5** RAG/ingestion data integrity.
6. **Phase 6** Logging normalization sweep.
7. Final full verification gate.

---

## Deferred-by-default notes (folded back into scope per request — all Low items above are INCLUDED)
- `compute_state_hash` broadening (L10), `media_budget` file-lock (L3), verifier `WhisperModel` caching (L2), `run_pipeline` global watchdog (M7), reddit link-seeder `force` re-post (L7), redundant double-click (L6) are all listed above and are part of the plan.
