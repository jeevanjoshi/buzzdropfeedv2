# Script-Quality Fix — Implementation Plan (surgical revision, outline-first, per-agent routing)

Status: IMPLEMENTED (Points 2, 3, 4, 5; verified by the single hermetic E2E test — `python run_tests.py`)
Date: 2026-08-09
Context: Run `csvg-exec-20260809-135331` aborted after a 3-revision loop that grew
violations 6 → 13 → 15 and hard-aborted on a "RAG tool-name citation leak"
(already fixed at source in `rag_retriever.py`; see "Firecrawl corpus poisoning" below).

## Problem

The story-revision loop regenerates the entire 18-shot script to fix a handful of
shot-scoped violations. Because the whole script is re-sampled, violations grow
non-monotonically across revisions (6 → 13 → 15) and ~15 LLM calls are wasted before
a hard abort. Root causes:

- Monolithic generation (`story_designer.generate_6act_script`) with ~20 competing
  constraints in one prompt.
- Full re-gen on every revision (`orchestrator.py:306`) → cross-shot Semantic
  Observer re-embeds the entire 18-shot corpus (`observer.py:568-579`) and flags the
  new parallel phrasing the rebuild inevitably creates.
- Best-draft retention picks by paraphrase-diversity, not fewest hard violations
  (`orchestrator.py:309`).

### Firecrawl corpus poisoning (fixed upstream, 2026-08-09)

`rag_retriever.search_firecrawl_facts` read the result URL from `metadata.url`, but
Firecrawl `/v1/search` returns it at the top level. Every Firecrawl result therefore
had an empty URL → `_source_label(url, "Firecrawl")` fell back to the tool name →
the RAG corpus passed to the LLM contained literal `[Firecrawl: ...]:` lines. The LLM
read "Firecrawl" as a publication and wrote "according to insights from Firecrawl",
which the Observer (correctly) treated as a hard abort.

Fixed by:
1. `search_firecrawl_facts` now reads `it.get("url")` first.
2. A `_TOOL_SOURCE_NAMES` guard relabels any residual tool-name bucket to
   `"Unattributed"` so a tool name can never be a citation source.

## What is / is not fixed by this plan

| Point | Fixes | Generic? |
|---|---|---|
| 2. Surgical per-shot revision | non-convergence, wasted LLM spend | Yes |
| 3. Outline-first beat artifact | fact-assignment/temporal/source-diversity errors + verbatim copies, cheap validation before prose | Yes |
| 4. Per-agent model/thinking routing | cost of the loop + critic precision (does NOT fix quality alone; enables 2 & 3) | Yes |
| 1. A2A crew (tighten, not refactor) | removes architecture/behaviour mismatch (dead intents, fake targets); the revision loop becomes the one real routed message | Yes (enables correct control flow for 2 & 3) |

## Point 2 — Surgical per-shot revision loop

### 2a. Bucket violations by shot id

Violation strings already embed `Shot #N` deterministically
(`observer.py:518,584,588,590,597,605`). Non-shot strings (`Source Diversity`,
`Keyword over-repetition`, `Runtime`, `Revenue Gate`) go to a global bucket.

```python
# orchestrator.py (new helper)
_RE_SHOT = re.compile(r"Shot #(\d+)")

def _bucket_violations(violations):
    by_shot, global_ = {}, []
    for v in violations:
        m = _RE_SHOT.search(v or "")
        if m:
            by_shot.setdefault(int(m.group(1)), []).append(v)
        else:
            global_.append(v)
    return by_shot, global_
```

### 2b. `StoryDesigner.repair_shots(...)` — new method

```python
def repair_shots(self, script, state, by_shot, global_violations):
    """Per-shot LLM repair. Reuses the _polish_script contract
    ({shot_id, narration_text} JSON) so the deterministic passes below
    (_clean_narration, ceiling, dissolve, acronyms) keep working verbatim."""
    corpus = state.crawled_content or ""
    corpus_sents = [...]  # same tokenization as process() at :1224
    targets = frozenset(by_shot)
    to_repair = [
        {"shot_id": s.shot_id, "act_index": s.act_index,
         "narration_text": s.narration_text, "visual_prompt": s.visual_prompt}
        for s in script.shots if s.shot_id in targets
    ]
    # one focused, low-thinking prompt for the failing shots only
    res = self.llm_client.generate_json(
        _REPAIR_PROMPT(topic=headline, violations=by_shot,
                       neighbors=neighbor_narration(script, targets)),
        system_prompt="You are a documentary editor. Return valid JSON only. Rewrite ONLY the listed shots.",
        route="shot_repair", thinking="low")
    narr_by_id = {s.shot_id: clean(s.narration_text) for s in res["shots"]}
    rebuilt = []
    for s in script.shots:
        if s.shot_id in narr_by_id:
            narr = _enforce_narration_ceiling(
                _expand_acronyms(self._dissolve_verbatim_copies(narr_by_id[s.shot_id], corpus_sents)))
            s = s.model_copy(update={"narration_text": narr,
                "duration_estimate": max(42.0, round(len(narr.split())/2.2, 1))})
        rebuilt.append(s)
    ...
```

Anti-regression guarantees:

- `targets` frozen → non-failing shots are copied through **bit-identical**.
- Neighbor narration (prev+next shot) is fed to the repair prompt
  ("must not repeat / must transition from X").
- Only the existing deterministic chain runs on repaired shots:
  `_clean_narration` → dissolve verbatim → `_expand_acronyms` → ceiling
  (`story_designer.py:1259-1272`).
- Malformed `res` → keep the shot as-is (never blank it).

### 2c. New revision loop in the orchestrator

Replaces the generate-all loop at `orchestrator.py:294-321`. Repairs are dispatched by
sending the Observer's `REVISE_SCRIPT` message to `story_designer.repair_shots`, so the
A2A message (not a second ad-hoc call) is the control path — see Point 5:

```python
MAX_REVISIONS = 3
best = (None, 10**9, -1.0)          # (script, hard_count, quality_score)
prev_hard = None
for attempt in range(1, MAX_REVISIONS + 1):
    by_shot, global_ = _bucket_violations(msg_obs.payload.get("violations", []))
    if global_ and not by_shot:
        # nothing shot-scoped to fix -> whole-script polish (existing _polish_script)
        msg_script = self.story_designer._polish_script(state.script_data, ...)
    elif by_shot:
        # dispatch the REVISE_SCRIPT message for real (it already exists at :800)
        msg_script = self.story_designer.repair_shots(
            state.script_data, state, by_shot, global_, msg_obs)
    msg_obs = self.observer.process(state)
    quality_pass, quality_error = run_script_quality_checks()
    violations = msg_obs.payload.get("violations", [])
    hard = [v for v in violations if not is_soft_violation(v)]
    # best-draft = FEWEST HARD, then paraphrase-diversity (was diversity-only, :309)
    if (len(hard), -observer_quality_score(state.script_data)) < (best[1], -best[2]):
        best = (copy.deepcopy(state.script_data), len(hard), observer_quality_score(state.script_data))
    if msg_obs.intent != AgentIntent.REVISE_SCRIPT and quality_pass:
        revision_ok = True; break
    # EARLY EXIT: a straight/rising hard count means the corpus is the problem
    if attempt > 1 and len(hard) >= prev_hard:
        logger.warning(..., "hard violations non-decreasing; aborting revision loop.")
        break
    prev_hard = len(hard)
```

## Point 3 — Outline-first beat artifact

### 3a. New schema (backward compatible)

```python
class ShotBeat(BaseModel):
    shot_id: int
    act_index: int            # 1..6
    beat_summary: str         # what this shot must accomplish
    facts_to_use: List[str]   # fact ids / short fact text to cite (from verified_facts)
    publisher: str            # THE publication to attribute (never a tool name)
    visual_type: VisualType
    chart_notes: Optional[str]
```

### 3b. `StoryDesigner.generate_outline(state) -> Optional[List[ShotBeat]]`

One cheap LLM call producing the 18 beats with no prose. Then a deterministic
validator (cheap, no MiniLM):

```python
def _validate_outline(beats, verified_facts, topic):
    errs = []
    if Counter(b.act_index for b in beats) != {1:3,2:3,3:3,4:3,5:3,6:3}:
        errs.append("act coverage")
    for b in beats:
        if not b.facts_to_use:
            errs.append(f"beat {b.shot_id} has no grounded facts")
        if b.publisher.lower() in _TOOL_SOURCE_NAMES:
            errs.append(f"beat {b.shot_id} cites a tool")
    srcs = [b.publisher for b in beats if b.publisher]
    if srcs and len(set(srcs)) < 2:
        errs.append("source diversity")
    # temporal: any facts_to_use carrying '(historical: YYYY)' must not be act 1
    ...
    return errs
```

If `errs` → fix deterministically (relabel tool → nearest real publisher, reassign an
ungrounded beat to a fact with topic-token hits) or regenerate the whole outline once.
**Zero prose burned on a bad plan.**

### 3c. Narrate against beats

```python
def narrate_from_outline(self, outline, state):
    # per-act calls (6 calls) or per-shot (18 calls) — default per-act for speed
    # each call gets: ITS beats + the exact facts_to_use slices + prev-act narration
    # identical downstream: polish -> dissolve -> acronyms -> ceiling (reuse process())
```

Opt-in: `main.py` flag `--outline-first`, plus env `CSVG_OUTLINE_FIRST=1` (mirroring
`USE_SEMANTIC_GATES`). Default off = current monolith, so it is A/B-able like `--rag`.

This is the point that cuts verbatim-copy violations: the narrator is told which fact
to convey in its own words for each shot rather than shown 4,000 chars of raw corpus
to copy from (sim 1.00 copies in the aborted run).

## Point 4 — Per-agent model / thinking routing

### 4a. `LLMClient.generate_json(..., route="default")`

Resolve a route → model before the fallback chain at `llm_client.py:161`:

```python
_ROUTES = {
    "generate":  os.getenv("LLM_ROUTE_GENERATE"),
    "polish":    os.getenv("LLM_ROUTE_POLISH"),
    "repair":    os.getenv("LLM_ROUTE_REPAIR",  "deepseek/deepseek-v4-flash-0731"),
    "critic":    os.getenv("LLM_ROUTE_CRITIC"),
}

def _model_chain(model, route=None):
    routed = _ROUTES.get(route or "default")
    chain = [routed or model,
             os.getenv("LLM_FALLBACK_MODEL") or "deepseek/deepseek-v4-flash-0731",
             os.getenv("LLM_FALLBACK_MODEL2")]
    seen = set()
    return [m for m in chain if m and m not in seen]
```

Optional thinking knob (env-gated; no-op for models without OpenRouter `reasoning`):

```python
if os.getenv("LLM_THINKING_EFFORT") and route == "critic":
    payload["reasoning"] = {"effort": "high"}
```

### 4b. Call sites

- `story_designer.generate_6act_script` → `route="generate"` (default = `LLM_MODEL`)
- `_polish_script` (`story_designer.py:1004`) → `route="polish"`
- `repair_shots` (new) → `route="repair"` (cheap flash model, low thinking)
- `observer.py:453` critic call → `route="critic"` (higher precision for the
  "verify 1 flagged claim" 5-min call)
- `tool_topic_synthesizer`, `youtube_video_verifier` stay on default

No `.env` changes required (all routes fall back to today's chain).

## Point 1 — A2A architecture: review + align (tighten, NOT refactor)

The crew is in place (`fact_retriever → story_designer → observer → media_producer →
publisher`, wired by `orchestrator.run_pipeline`). An end-to-end wiring review
(2026-08-09) found it is **a sequential workflow wearing A2A-shaped telemetry**, not
a message bus — and that is fine, because this pipeline's invariants are determinism +
checkpoint-resumability + hard gates before publish. The review found real defects to
tighten (see Point 5). The GOOD news for script quality:

- The **revision loop is the ONE place where messages genuinely matter**, and it is
  already coupled correctly: `observer` emits `REVISE_SCRIPT` → `STORY_DESIGNER`
  (`observer.py:800-812`), which the orchestrator reads at `orchestrator.py:274-274`.
  Point 2's surgical loop should be **expressed as that message**, not a second,
  ad-hoc direct call (keeps one control path, one API).
- `GlobalState` (checkpoint `logs/state_*.json`) is the real transport; A2A messages
  augment it with structured hand-off records for tracing/resume. This stays as-is.

So Point 1 = "keep the sequential crew; align the message layer to reality". Correlates
with 2 (revision routes through `REVISE_SCRIPT` payload) and 3 (outline approval can be
a message) — see Point 5.

## Point 5 — A2A alignment pass (correlates Points 1–3)

A full-wiring review (2026-08-09) confirmed the crew is a **sequential workflow with
A2A telemetry**, not a message bus — keep it that way. But it surfaced concrete defects
to align, and two of them touch script quality directly:

### What the review found
1. **Messages are never routed.** Agents are called directly
   (`orchestrator.py:240,399,488`); `msg_script`/`msg_media`/`msg_pub` are dropped into
   `tracer.record_step` as telemetry. The ONLY message-driven branch is the
   `REVISE_SCRIPT`/`APPROVE_SCRIPT` check (`orchestrator.py:274,319`).
2. **Targets don't match reality.** `story_designer` sends `target=OBSERVER`
   (`story_designer.py:1285`) but the orchestrator calls them both directly;
   `observer` sends `target=STORY_DESIGNER` (`observer.py:803`) yet never delivers it.
3. **4 of 10 intents are dead.** `FETCH_TOPIC`, `PRODUCE_MEDIA`, `PUBLISH_VIDEO`,
   `FAILURE_REPORT` are declared in `a2a.py:15-24` but never emitted.
4. **True transport is `GlobalState` + `logs/state_*.json`, not messages.** Resume works
   by stage, not message replay — correct, keep it.

### Alignments (small, zero architecture risk)
- `a2a.py`: delete the 4 dead intents, or document them `# reserved`. Prefer delete —
  dead enums invite agents to be written against non-existent flows.
- Fix `sender/target` so they reflect who actually calls whom: all agents are invoked by
  `ORCHESTRATOR`; the only true cross-edge is `OBSERVER → STORY_DESIGNER` for
  `REVISE_SCRIPT`. Set targets accordingly in `story_designer.py:1285`, `observer.py:803`.
- **Make `REVISE_SCRIPT` real (the quality tie-in).** Instead of the orchestrator
  reading `msg_obs.payload["violations"]` and calling `repair_shots` ad hoc, send
  `msg_obs` to `story_designer.repair_shots(..., msg_obs)` (Point 2c above). One control
  path, one API — and it exercises the message layer on exactly the loop that matters
  (violation count). Add `msg_obs.state_hash` (already in `a2a.py:34`) so the repair
  handler can verify it is fixing the same state the observer audited.
- `FAILURE_REPORT`: wire it OR remove it. Recommend remove from `a2a.py` and let the
  existing exception → `tracer.record_step(..., status="ERROR")` (`orchestrator.py:522`)
  stay the single failure path.

### Out of scope (deliberately NOT planned)
- No message inbox/router, no agent-to-agent autonomy, no per-shot parallelism. These
  would erode the determinism + checkpoint + publish-gate invariants. CrewAI-style
  autonomy is a regression risk for a pipeline that must not publish garbage.

## Sequencing, risk, gates

| Step | Touches | Risk | Gate |
|---|---|---|---|
| 2. Surgical revision | `orchestrator.py`, `story_designer.py` (new method + helper) | Medium (revision path only; happy path untouched) | smoke suite + resume of `csvg-exec-20260809-135331` |
| 5. A2A alignment | `a2a.py`, `story_designer.py:1285`, `observer.py:803`, `orchestrator.py` REVISE dispatch | Low–Medium (behaviour-neutral except the REVISE edge, which 2 makes real) | smoke suite |
| 4. Routing | `llm_client.py` + 4 call sites | Low (pure config passthrough) | smoke suite (stubs unaffected) |
| 3. Outline-first | `state.py`, `story_designer.py`, `main.py` | Medium (new A/B path, off by default) | smoke suite + `--outline-first` on the laundromat topic |

Ordered so each step stays green before the next. 2 and 5 are coupled (2c dispatches
`REVISE_SCRIPT`), so implement them together; 4 and 3 afterwards.

## Tests — single hermetic E2E (`tests/test_hermetic_e2e.py`, self-sufficient)

The old `tests/test_smoke.py` (16 cases) was REMOVED and replaced by ONE fully
self-sufficient hermetic module. It drives the REAL orchestrator + REAL
StoryDesigner + REAL Observer with a scripted FakeLLM and boundary stubs only. It
covers (7 assertions):

1. Happy path through publish (REAL pipeline, fake LLM + stub media/publisher).
2. Surgical revision actually repairs: real Observer flags Shot #4 (>155 words),
   `repair_shots` fixes it via `route='repair'`, non-target shots bit-identical.
3. Stale `state_hash` on `REVISE_SCRIPT` is rejected by `repair_shots`.
4. Outline-first path (`CSVG_OUTLINE_FIRST=1`): `generate_outline` + per-act
   `narrate_from_outline` produce a valid 18-shot script.
5. Per-agent route→model resolution + fallback chain (`LLM_ROUTE_REPAIR` etc).
6. A2A alignment: dead intents removed; `state_hash` stable across equal states.
7. (implicit) A failure in any case is a real regression — no network/Pi/media/.env.

Run it: `python run_tests.py` or `python tests/test_hermetic_e2e.py`.

## Reference — related sources reviewed

- DeepMind Dramatron (arxiv 2209.14958) via "Generating narratives" (Medium, R. Layte) —
  outline-as-artifact + per-scene narration with previous-scene context.
- "AI-Driven Storytelling with Multi-Agent LLMs - Part III" (apiad.net, A. Piad / R.
  Fuentes) — blueprint/DAG as source of truth; Dependency Manager ≈ our Observer.
- "Automating a Newsroom Crew with AI Agents and CrewAI" (Medium, J. Uddståhl) —
  per-agent model routing; crew anatomy already covered by our A2A chain.
- "Prompt Engineering for Gemini 3.5 Flash" (Blockchain Council) — task-first,
  schema-first, staged prompts, thinking levels, single objective per call.