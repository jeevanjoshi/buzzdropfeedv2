# Competitor Gap Analysis — Fix List

_Generated: 2026-08-18 11:22 UTC_

Auto-consolidated from competitor benchmarks. Use the checkboxes as a later-fix backlog; each item points at the exact code location.

- Light benchmark: `logs/competitor_data.json`
- Deep analysis:   `logs/competitor_deep.json`

## Summary

- **HIGH**: 1
- **MEDIUM**: 3
- **LOW**: 1
- **INFO**: 1

## Action Items (Later Fixes)

### HIGH

- [ ] **Title angle (tension/contrarian framing)** — Top faceless finance competitor (Finance Bureau) uses tension/contrarian framing in ~47% of titles; pipeline previously defaulted to a generic 'The Hidden Truth Behind {headline}...' template. Tension titles consistently outperform neutral ones on CTR.
  - Code: `src/agents/story_designer.py:999,1452`
  - Fix: Prefer _generate_tension_title (shipped) for contrarian hook generation; keep CTR fallback only as no-LLM safety net.
  - id: `GAP-01` · source: light

### MEDIUM

- [ ] **Thumbnail brightness / pop** — Competitor thumbnails measured ~61 mean_lum vs our nano-banana target mean_lum >=100. Brighter thumbs pop more in the subscription feed.
  - Code: `src/engine/nano_banana.py (comply_thumbnail brightness lift)`
  - Fix: Confirm brightness-lift target is actually applied at render time; consider raising target to ~110.
  - id: `GAP-04` · source: light
- [ ] **Narration pace (wpm)** — Some faceless competitors narrate faster than our hardcoded 150 wpm assumption; a faster pace can improve retention on data/explainer content.
  - Code: `src/agents/story_designer.py:1157,1455 (150 wpm -> seconds)`
  - Fix: Make narration wpm niche-aware instead of hardcoded 150; measure from a real deep run.
  - id: `GAP-02` · source: deep
- [ ] **Shot density / pacing** — Competitor explainers often cut faster (more shots) than our ~15-18 shot target. Faster cuts hold attention on static topics.
  - Code: `src/agents/story_designer.py (target_shots, --tail padding)`
  - Fix: Tune target_shots / --tail hold per niche to match competitor retention pacing; verify via a real deep run.
  - id: `GAP-03` · source: deep

### LOW

- [ ] **Cadence / upload frequency** — Competitors publish ~1.1-1.4 videos/week; pipeline targets up to 2/day (cron max 4/day). Volume is not our bottleneck — quality/retention is.
  - Code: `cron_publish.sh (CSVG_MAX_DAILY_PUBLISHES=4)`
  - Fix: Keep cadence; do not chase competitor frequency — focus on per-video retention.
  - id: `GAP-05` · source: light

### INFO

- [ ] **Background music / SFX** — Many faceless competitors run a continuous music bed under narration; our BGM sidechain duck may be too conservative on some niches.
  - Code: `mcp_servers/media_cloud/server.py (BGM sidechain)`
  - Fix: If competitors are music-heavy, raise BGM_VOLUME / lower duck threshold; verify via a real deep run.
  - id: `GAP-06` · source: deep
