# Dedicated Professional Shorts Producer — Design & Scaffold

**Status:** Scaffolded + wired (env-gated). Core render path implemented; advanced
"pro-level" polish marked TODO and lands incrementally.

**Owner section of AGENTS.md:** "current shorts videos are cuts from main video, need
a professional producer to create pro level shorts."

---

## 1. The problem

Today a YouTube Short is a **post-publish crop of the 16:9 master**:

`src/engine/micro_content_producer.generate_shorts()` seeks into
`state.asset_paths.final_video` and runs
`ffmpeg -vf crop=ih*9/16:ih,scale=1080:1920` on time windows pulled from
`state.asset_paths.measured_durations` + `state.script_data.shots`.

Consequences of "cut from main":
- The vertical frame is a **center-crop** of a landscape frame — the subject (framed
  for 16:9) is routinely sliced off.
- The audio/captions are the **long-form narration re-timed**, not written for mobile
  retention (no hook/payoff arc, no punchy cadence).
- No short-specific music bed, no short-specific caption styling, no pattern-interrupt
  opening beyond a 1s cover frame.

A "pro" Short must be a **first-class, short-native deliverable**.

---

## 2. Target architecture

```
FactRetriever → StoryDesigner → Observer → MediaProducer(master 16:9)
   → ShortProducer(ONE short-native 9:16 script + media)   ← NEW
   → Quality Gates → Publisher(publishes short-native clip)
```

One Short per run (per product decision): a single, retention-optimised 9:16 video
with its own hook → payoff arc, independent of the master timeline. This is enforced
both in `ShortProducerAgent.produce_short_media` (`assets.shorts = [final_path]` — the
`ShortBeat`s assemble into ONE clip, not N clips) and in the publisher fallback
(`micro_content_producer.generate_shorts(state, max_shorts=1)`). Cleanup is already
covered: the short output lives under `logs/media/<pid>/shorts/`, pruned by
`cleanup.py` as a Tier-B per-run dir (`logs/media` → `run_dirs`), so no extra cleanup
step is required.

### New schema (`src/schemas/state.py`)
- `ShortBeat` — vertical-native beat: `is_hook`, `narration_text` (≤18 words),
  `visual_prompt` (9:16 brief), `duration_estimate`, `caption_style`.
- `ShortScript` — `title`, `hook_line`, `beats[]`, `estimated_runtime_seconds`,
  `music_bed`.
- `GlobalState.short_script: Optional[ShortScript]` and
  `GlobalState.short_asset_paths: AssetPaths`.

### New agent (`src/agents/short_producer.py`)
`ShortProducerAgent` with two stages:
1. **`author_short_script(state)`** — LLM (`route="generate"`) authors a short-native
   script grounded in the same verified facts / topic, reframed for a standalone
   mobile viewer. Pure parser `parse_short_script()` is deterministic and unit-tested;
   a deterministic `_fallback_short_script()` covers the no-LLM case.
2. **`produce_short_media(state, short_script)`** — renders the vertical clip:
   per-beat TTS (`synthesize_tts`, speed `SHORT_TTS_SPEED=1.06`),
   **9:16 Ken-Burns** (`_render_vertical_clip` — a vertical-aware ffmpeg zoompan,
   because the shared `apply_ken_burns_motion` hardcodes 1920×1080), mobile captions
   (`align_subtitles_whisper` + `merge_ass_subtitle_files`), a hook cover prepended as
   the opening frame (`_make_hook_cover` / `_image_to_video`), assembled via
   `assemble_ffmpeg_timeline` at 1080×1920.

### Wiring
- `orchestrator.run_pipeline()` — `ShortProducerAgent` injected; after quality gates,
  calls `await self.short_producer.process(state, dummy_frames=...)` **ON by default**
  (set `CSVG_SHORTS_PRODUCER=0` to disable and fall back to the legacy crop path),
  wrapped in try/except (non-fatal: a Short failure never blocks the long-form publish).
- `publisher.publish()` — prefers `state.asset_paths.shorts` (the short-native clip);
  falls back to `micro_content_producer.generate_shorts()` (legacy crop) only when the
  new producer didn't run. Backwards-compatible.

---

## 3. What is implemented vs. TODO (incremental after review)

**Implemented & tested (hermetic):**
- Short-native LLM script authoring + deterministic fallback.
- Vertical 9:16 renderer reusing the existing TTS / Whisper / ffmpeg toolchain.
- Hook-cover opening frame; mobile-caption merge; 1080×1920 assembly.
- Orchestrator wiring (env-gated, non-fatal) + publisher short-native preference.
- Tests: `StubShortProducer`, `case_short_producer_helpers`,
  `case_short_producer_authoring`, `case_short_producer_pipeline` (all green).
- `ruff --select F821,F822,F823` clean on all touched files.

**TODO (pro-level polish, next increments):**
1. ✅ **Portrait visuals + subject reframe (DONE)** — `produce_short_media` now
   requests a real 9:16 FLUX still per beat (budget-gated via `media_budget`, with a
   free synthetic 9:16 fallback) and a `reframe_16_9_to_9_16` crop guards landscape
   inputs. `ImageGenRequest.aspect` ("9:16"/"16:9") was added to
   `mcp_servers/media_cloud/server.py` (backward-compatible; master still lands
   landscape). True ML subject-tracking (face/object detection) remains a future
   refinement, but the center-crop-from-16:9 problem is eliminated.
2. ✅ **Short-native music bed (DONE)** — `_short_music_bed(work, mood)` now picks
   an **optimal track per Short** via `_classify_short_mood` (maps the Short's hook/
   title/niche to a mood: upbeat_discovery / inspiring / tense / curiosity / calm /
   epic). Resolution order: curated local library `resources/shorts_music/<mood>.mp3`
   → download from operator-configured `CSVG_SHORTS_MUSIC_BASE_URL/<mood>.mp3` (cached,
   **no hard-coded URLs**) → fallback to the master `resources/bgm.mp3` (never silence).
   Short-specific ducking (`BGM_TEMPO` etc. in `media_cloud`) is still a future tweak.
 3. ✅ **Short-specific caption styling (DONE)** — `_style_mobile_ass` restyles the
    merged 16:9 master `.ass` into a 9:16 mobile-safe track: `PlayRes 1080x1920`, large
    bold white `Montserrat` (Fontsize 20, Outline 3, Shadow 2), bottom-center
    `Alignment=2` with portrait-safe margins. Applied in `produce_short_media` when any
    beat's `caption_style == "mobile"` (the `ShortBeat` default).
 4. ✅ **A/B variant set (DONE)** — with `CSVG_SHORTS_VARIANTS=1`, `process` produces
    a **primary Short + 3 experiment variants** (hook-led / hero-object / text-led,
    reusing the same script) and writes `logs/media/<pid>/shorts/shorts_variants.json`.
    Each variant stresses a different retention lever: hook-led = longer hook cover;
    hero-object = leads the cover with the hero beat's subject line; text-led = tighter
    caption margin (more on-screen text). The **primary (index 0) is what gets
    published**; the rest are finished as YouTube Studio "test & compare" experiments
    (like `CSVG_THUMBNAIL_VARIANTS`). `produce_short_media` takes `spec`/`suffix` so
    each variant renders to a distinct file without clobbering the primary.
 5. ✅ **Per-beat visual diversity / motion (DONE)** — `_beat_motion(beat_id, is_opener)`
    deterministically varies the Ken-Burns pan (center/up/down/left/right, cycling on
    `beat_id`) and zoom direction (in on odd beats, out on even) so every beat drifts
    differently; `_build_ken_burns_vf` turns that into a vertical-aware ffmpeg zoompan.
    The **opener beat** (first real beat after the hook cover) gets a `punch` (1.0→1.6
    aggressive zoom) as a pattern-interrupt that jolts attention. Wired into the beat
    render loop in `produce_short_media`.

---

## 4. How to enable / verify

```bash
# ON by default — just run normally:
python main.py --global --till-upload        # short-native clip built, not published
python main.py --global                        # short-native clip published as the Short

# Disable (fall back to legacy crop):
export CSVG_SHORTS_PRODUCER=0

# A/B "test & compare" variant set (primary + 3 experiment Shorts):
export CSVG_SHORTS_VARIANTS=1

# Hermetic (no network/ffmpeg/TTS) verification of the scaffold + wiring:
python run_tests.py        # includes SHORT_PRODUCER_HELPERS / _AUTHORING / _PIPELINE
```

Run cost: the new producer is a separate render pass (extra TTS + ffmpeg assembly),
so it adds ~1 short's worth of compute per run when enabled. The legacy crop path is
fully preserved as the fallback when `CSVG_SHORTS_PRODUCER=0`.
