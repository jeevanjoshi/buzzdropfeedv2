# CSVG Media Quality Upgrade — Plan

> Consolidated implementation plan for the media/production quality fixes identified from the
> `csvg-exec-20260806-191241` run (`logs/pipeline_run_20260806-191227.log`).
>
> Goal: **quality up, cost flat-or-down, effort down.** Priority legend:
> - **P0** = shipped-quality bug that must be fixed
> - **P1** = low-risk quality improvement
> - **P2** = the revenue lever

---

## Evidence from `2026-08-06` published run

| Asset | Finding |
|---|---|
| 15 shot MP4s | All 1080p/25fps, all have audio, durations 45.9–65.5s (sync to TTS) ✓ |
| 13/15 visuals | Free Pixabay stock — only shots 1,8,12,15 are premium FLUX candidates + thumbnail used Fal; this run ALL in-video visuals were free |
| Frozen frames | Gate 7 flagged **every shot** "suspect frozen frame" — stock stills only get a subtle Ken Burns pan |
| Synthetic stat image | One shot showed the **PIL placeholder** (`generate_synthetic_png`) instead of a real chart |
| Audio | **Shots 2,3,7,13 clip at -0.0 dB** (full-scale) at the TTS/WAV stage; all shots mean ≈ -21 dB (quiet) |
| Bitrate | Final video **487 kbps at 1080p** (low/soft) + 199 kbps AAC |
| Shorts | 3 vertical 9:16 × 45s, 1080×1920 ✓ |
| Subs | Per-shot `.ass` + merged master, synced to measured audio ✓ |
| Thumbnail | Fal-generated 494 KB ✓ |
| Runtime | **847s (14:07)** vs 11.93 min design target (2 min over) |

### Root causes
1. **Frozen/off-topic stock**: `media_producer.py` builds the stock search query from the **first 4 three-letter+ words** of the visual prompt. Since every prompt starts *"Cinematic 16:9 widescreen shot: ..."*, searches collapse to generic "Cinematic widescreen shot ..." → irrelevant B-roll.
2. **Synthetic stat image**: statistics shots are all `standard_image`; the chart path (`MATPLOTLIB_CHART`) only fires on explicit `"[chart:"`/`"stock chart"`/`"market graph"` tokens. When free-stock fetch fails it silently emits the PIL placeholder.
3. **Audio clipping**: originates at the TTS **WAV** stage (measured `max_volume = -0.0 dB` in `audio/shot_2.wav`).
4. **Low bitrate**: `-crf 22` + no cap yields soft 1080p.

---

## Phase 1 — Real, accurate stat charts (fixes synthetic placeholder) — **P0, +Rs.0**

**Goal**: statistic shots render real annotated matplotlib charts from grounded RAG data, never the PIL placeholder.

### 1.1 Add chart-spec fields to schema — `src/schemas/state.py:83-89`
Add to `ShotData`:
```python
chart_spec: Optional[Dict[str, Any]] = None   # {title, labels, values, unit, chart_type}
```
`VisualType` already has `MATPLOTLIB_CHART` (line 80). Backward-compatible (default None).

### 1.2 LLM emits chart specs — `src/agents/story_designer.py`
- In `generate_6act_script` (line 289) prompt block, instruct the LLM: for any narration sentence containing a numeric/statistical claim, set `visual_type=matplotlib_chart` and output a `chart_spec` with real labels/values/unit derived **only** from RAG facts (verbatim numbers).
- Parse into `ShotData.chart_spec`; add `extract_numeric_chart_spec(fact)` helper that pulls cited values like "33% YoY" from `state.verified_facts`.

### 1.3 Route stat shots to the chart path — `src/agents/media_producer.py:453-533`
- Add a **Check 0** before the GIF/Chart/SVG checks: if `shot.chart_spec` is present OR narration has ≥2 numeric patterns (`\d+\s*%|¥|\$|₹|Billion|crore|million|year-on-year`), build a `ChartRequest` from `chart_spec` (fallback: derive from top cited fact) and call `generate_dynamic_chart`.
- Pass chart data through so labels/values come from the script, not the hardcoded 5-point market tick.

### 1.4 Robust `generate_dynamic_chart` — `mcp_servers/media_cloud/server.py:421`
- Support `chart_type`: `line` (current) and `bar` (better for % and discrete comparisons).
- Keep matplotlib; **remove the silent `"DUMMY_CHART_MP4"` fallback** (line 470-471) — raise so the caller falls through to a visible placeholder caught by the new Gate 7 check.

### 1.5 Never ship the synthetic silently — `mcp_servers/media_cloud/server.py`
- `generate_synthetic_png` (line 66) stays as last resort, but add a **Gate 7 post-check**: scan each `shot_*.mp4` first frame for the PIL-placeholder signature (uniform `#0f141e` background + cyan border). If any shot is still a placeholder → **abort before publish** (or banner + auto-retry chart once).

**Acceptance**: the statistics shot now shows a real animated chart with correct numbers; no placeholder reaches YouTube.

---

## Phase 2 — Fix frozen / off-topic stock visuals — **P1, ≈ +Rs.5/run**

**Goal**: every visual is on-topic and moving; kill the Gate-7 frozen-frame warnings.

### 2.1 Fix stock query relevance — `src/agents/media_producer.py:191-192`
Replace "first 4 words" `clean_query` with a **semantic keyword extractor**:
- Drop camera/lighting noise (`cinematic, widescreen, sweeping, archival, 8k, photorealistic, golden, shot, slow, dolly, glowing...`).
- Keep the topical subject words after `:`.
- If <2 topical keywords remain, use **topic anchor words** (keywords + headline) from `state.selected_topic`.

### 2.2 Generate ALL in-video visuals with Flux (budget allows) — `media_producer.py:543-557`
- Flux-schnell ≈ $0.003/img → 16 imgs ≈ $0.05/run, well under the $24/mo cap. Change so **every standard-image shot attempts `generate_flux_image` first**; use free-stock only on Flux failure (inverted from today's default-free).
- Keep `enrich_visual_prompt` cinematic tags.
- Verify `media_budget.json` isn't silently in `economy_mode()` (this run was all-free, budget likely exhausted/blocked); confirm `FAL_KEY` set in `.env`.

**Acceptance**: no more unrelated visuals; Gate 7 frozen-frame count → ~0.

### 2.3 Stronger motion (optional) — `media_producer.py:596-602`
Increase Ken Burns zoom amplitude (`min(zoom+0.0015,1.15)` → `1.25`); alternate directions already present (line 451).

---

## Phase 3 — Audio: kill clipping + normalize loudness — **P0/P1, +Rs.0**

**Goal**: no full-scale clip; consistent -14 LUFS (YouTube target).

### 3.1 Peak limiter at TTS synth
- In `mcp_servers/audio_edge/server.py` post-synth (or the WAV merge in `media_producer.py:614-625`), apply ffmpeg `alimiter=limit=0.89` (≈ -1.0 dB) so shots never hit -0.0 dB.
- Clipping originates at the **WAV** stage, so fix at synth/merge, not just final.

### 3.2 Loudness normalize final mix — `mcp_servers/media_cloud/server.py:683`
In the `amix` filter chain (line 684) append `loudnorm=I=-14:TP=-1.5:LRA=11` after voice+BGM mix. Also acts as a safety limiter.

**Acceptance**: `max_volume ≤ -1.0 dB` on all shots; measured LUFS ≈ -14.

---

## Phase 4 — Render quality: bitrate up — **P1, +Rs.0**

**Goal**: crisp 1080p (was 487 kbps).

### 4.1 Raise encode quality
- `-crf 22` → `-crf 18` in **three** places: `media_producer.py:624` and `:888`, and `server.py:691`/`:892` (assemble paths).
- Optionally `-maxrate 6M -bufsize 12M` for ~5 Mbps target.

**Acceptance**: final bitrate ≥ ~3.5–5 Mbps at 1080p; no macroblocking on text.

---

## Phase 5 — Duration + retention (the revenue lever) — **P2, +Rs.0**

**Goal**: hit ~12 min target (was 14:07); denser = better retention.

- Reduce `target_shots` / tighten narration ceiling (`story_designer._enforce_narration_ceiling`, line 71).
- Fix repeated stats (e.g. "~50% vs <25%" in shots 2 AND 5) and trim off-topic tangents flagged in the Observer audit (Taiwan GDP, US leasing, US jobs).

---

## Phase 6 — Verification & regression
- Run `python run_tests.py` (hermetic smoke) after each Phase — stubs mock agents, no network.
- Then `python main.py --offline --till-upload --dummy-frames` to validate the full chain incl. chart routing.
- Optional GPU (skip for now): only revisit Lumino/CloudPe L4 (RTX 4090 ≈ $1.05/hr, L4 ≈ $0.69/hr) if later adding **LTX motion clips** — not required for these fixes.

---

## Cost & impact summary (per run)

| Phase | Cost | Quality impact | Revenue impact |
|---|---|---|---|
| 1. Real charts | +Rs.0 | High (worst→best shot) | Indirect (accuracy/trust) |
| 2. On-topic Flux visuals | +Rs.5 | High (frozen→relevant) | Low-moderate (retention) |
| 3. Audio (limit+loudness) | +Rs.0 | Medium | Low (less churn/annoyance) |
| 4. Bitrate | +Rs.0 | Medium (sharpness) | Low |
| 5. Retention/duration | +Rs.0 | High | **Highest** (watch-time) |

**Suggested delivery order**: **1 → 3 → 4 → 2 → 5**.
Phases 1, 3, 4 are low-risk P0/P1 quality guards; 2 is the visual overhaul; 5 is optional but the only one that materially moves the $2k/mo ad-revenue goal.

---

## Key file map

| File | Lines | Responsibility |
|---|---|---|
| `src/schemas/state.py` | 77-89 | `VisualType`, `ShotData` (add `chart_spec`) |
| `src/agents/story_designer.py` | 71, 289, 511 | narration ceiling, chart-spec emission, polish |
| `src/agents/media_producer.py` | 21, 150-234, 453-602, 624, 888 | premium IDs, stock query, routing, Ken Burns, bitrate |
| `mcp_servers/media_cloud/server.py` | 66, 421-475, 683-695, 892 | synthetic fallback, chart gen, final mix, bitrate |
| `mcp_servers/audio_edge/server.py` | synth path | TTS limiter |
| `src/engine/media_budget.py` | all | FLUX budget guard ($24/mo cap) |
