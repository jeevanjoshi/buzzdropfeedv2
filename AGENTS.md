# AGENTS.md

Autonomous 8-stage YouTube storytelling video pipeline (CSVG): RSS -> topic TOPSIS selection -> RAG-grounded 6-act script -> LLM-edited/observed -> TTS + AI visuals + ffmpeg assembly -> YouTube publish. Plain Python scripts, no packaging, no build step. Run everything from the repo root (relative `logs/` paths and `from src...` imports depend on it).

## Commands
- Activate venv first: `source venv/bin/activate`
- `--offline` uses canned topic candidates instead of live RSS — it is NOT a full dry-run. It does NOT skip RAG (Tavily/Firecrawl), the LLM (aborts if unavailable), visuals (fal/Replicate), Pi TTS, or publish. Those each need their own flag or a mock: `--dummy-frames` (synthetic visuals, no fal/Replicate), `--till-upload` (no publish). The truly hermetic/no-network path is `tests/test_smoke.py` (stub agents mock LLM/RAG/TTS/visuals/publisher).
- Real run without publishing: `python main.py --global --till-upload` (alias `--no-upload`)
- Production run (recommended for real publishes): `./run_production.sh` — syncs code to Pi, runs in background, logs to `logs/`, emails result. `--no-detach` blocks; auto-resumes from latest `logs/state_*.json` checkpoint if a previous run didn't reach `PUBLISHED_SUCCESS`.
- Resume a specific run: `python main.py --resume <pipeline_id>` (reads `logs/state_<pipeline_id>.json`).

### Flags (main.py)
`--offline`, `--india` / `--global` (region), `--till-upload`/`--no-upload`, `--dummy-frames` (synthetic visuals, skips fal/Replicate), `--renderer ffmpeg|moviepy`, `--crossfade <seconds>`, `--resume <id>`, `--rag grounded|scraper` (A/B: Google Search grounding research pass vs the 5-scraper RAG path; default = scraper).
- There is NO `--help`; `main.py` scans `sys.argv` (no argparse) so unknown flags (incl. `--help`) are silently ignored. Semantic gates are env-only (`USE_SEMANTIC_GATES=1`), not a CLI flag.

### Tests
- Suite runner: `python run_tests.py` (custom wrapper, not pytest) imports and runs the single smoke module.
- The ONLY test file is `tests/test_smoke.py` — a fully HERMETIC end-to-end smoke test covering POSITIVE / NEGATIVE / EDGE cases across the whole orchestrator (topic -> RAG -> script -> observer -> media -> gates -> publisher). It runs with **NO live/network calls, NO Raspberry Pi interaction, and NO media generation** (stub agents + controlled monkeypatching of module singletons). Safe to run without env/services and without `.env`.
- Run it: `python run_tests.py` or `python tests/test_smoke.py`. Both are `py_compile`-clean.
- Lint/format: ruff is used informally (only `.ruff_cache/` exists, no committed config, no CI). No enforced lint/formatter.

## Env & secrets
- `.env` is gitignored; copy `example.env` -> `.env` and fill real keys. NEVER commit `.env`, `token.json`, or `client_secret.json` (YouTube OAuth).
- LLM fallback chain: primary model -> `LLM_FALLBACK_MODEL` -> `LLM_FALLBACK_MODEL2`. There's NO template/boilerplate fallback — if the LLM is unavailable the run **aborts** rather than emit canned content.
- LLM default provider config lives in `.env` (`PREFERRED_LLM_PROVIDER`, `LLM_MODEL`, keys).
- Semantic-gate env (see section below): `USE_SEMANTIC_GATES=1`, `ALLOW_SOFT_APPROVAL=1`, `HF_HOME`/`TRANSFORMERS_CACHE` -> repo-local `.hf_cache/`. Heavy ML deps live in `requirements-master.txt` (master ONLY, never the Pi; the Pi keeps TF-IDF fallback).

## Architecture (where things live)
- `src/agents/` — sequential A2A agents wired by `orchestrator.py`: `fact_retriever` -> `story_designer` -> `observer` (quality gates + bounded 3-revision loop) -> `media_producer` -> `publisher`. `orchestrator.run_pipeline()` is the single entrypoint.
- `src/engine/` — stateless/stateful helpers (RAG, TOPSIS, quality_verifier, llm_client, channel_phase_manager, etc.). Random leaf files; no surprises here.
- `src/schemas/` — pydantic models (`state.py`, `a2a.py`); `GlobalState` is the checkpoint schema.
- `mcp_servers/` — three standalone FastAPI apps: `audio_edge` (Kokoro TTS + Whisper .ass, on the Pi), `media_cloud` (fal/flux visuals + ffmpeg, port 8001), `youtube_cloud` (upload/quota, port 8002). Only `audio_edge` has a committed systemd unit (`kokoro_tts.service`, port 8000, Pi); `csvg_dashboard.service` runs the dashboard; `media_cloud`/`youtube_cloud` run via `uvicorn.run` (no committed unit; `deploy.sh` only restarts `kokoro_tts`). Add new model/http endpoints here, not in `src/engine`.

## Distributed layout (this is not a single-host repo)
- Master pipeline (agents, media_cloud, youtube_cloud) runs on the OCI cloud host.
- Audio/TTS (audio_edge, Kokoro, Whisper) runs on a Raspberry Pi 5 edge node, reached via `AUDIO_EDGE_URL` and `LLAMA_CPP_URL`.
- `deploy.sh` git-clones/reset-hards the repo to both nodes over SSH and restarts services; `sync_to_pi.sh` rsyncs the working tree to the Pi (excludes logs/media/venv).
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
| Verbatim/paraphrase copy | ≥12-word substring match | MiniLM cosine sim ≥ 0.82 vs clean corpus; catches lightly-rewritten slop |
| Keyword over-repetition | Exact-token blacklist | Semantic topic-membership: token vs topic anchor words (keywords + headline + summary); `chinese~china` = 0.68, `gadget~china` = 0.31; threshold 0.50 |
| Sentence monotony / duplication | TF-IDF cos ≥ 0.82 | MiniLM pairwise similarity ≥ 0.82; one-batch-encode, precomputed matrix |
| Quality score | N/A | Paraphrase diversity = 1 − mean pairwise sentence sim; used for best-draft retention across revisions |

### Soft approval (orchestrator.py)
After the bounded 3-revision loop, style-class violations (keyword over-repetition, verbatim copy, sentence repetition, source diversity, visual prompt, narration too long) are **non-blocking** when `ALLOW_SOFT_APPROVAL=1` (default). Hard invariants (fact/temporal audit, revenue/audience gate, runtime, shot count, quality gates 1–7) still abort. Set `ALLOW_SOFT_APPROVAL=0` to restore all-or-nothing.

### Model persistence
The model loads lazily on first use and stays **resident** for the process lifetime. `release()` frees the model weights (~50 MB) but torch stays imported in-process. Reloading is expensive (~10–20 s) and unnecessary for 4–6 runs/day — resident is the default.

### Clean RAG (rag_retriever.py + story_designer.py)
`_strip_boilerplate()` strips ad/credit/nav junk (SKIP ADVERTISEMENT, Video by, Listen ·, Subscribe, etc.) from deep-crawled articles before they enter the fact corpus, crawled_content, and the snippet padding pool. The same junk filter (`_SNIPPET_JUNK_RE`) protects the story designer's RAG snippets. This removes the "verbatim NYT boilerplate in Shot #15" class of false positives at the source.

### Thresholds (observer.py module-level constants)
- `COPY_SEMANTIC_THRESHOLD = 0.82` — narration sentence ~== clean corpus sentence
- `TOPIC_MEMBER_SEMANTIC_THRESHOLD = 0.50` — token ~== topic anchor word (keyword/headline/summary)

### Not learning (frozen encoder)
The MiniLM model is a **frozen pretrained encoder** — no gradient updates, no RL, no weight adaptation across runs. It generalises to `china→chinese` because it was trained on billions of text pairs, not because it learns from your pipeline. If you want learned gates, future options (in order of complexity):

1. **Online threshold calibration** — record `{scores, verdict}` per run; nudge thresholds so false rejections trend to zero. Deterministic, no training infra.
2. **Supervised reward classifier** — accumulate accepted-vs-rejected scripts, fine-tune a small DistilBERT to predict pass/fail. Needs ~hundreds of labeled samples.
3. **RLHF/DPO on the writer** — treat the Observer as a reward signal and fine-tune the generative model. Requires an open model (Gemini is not fine-tunable); heavy, out of scope for 2-vCPU/12 GB.

## Conventions & gotchas
- New hosts/agents must not duplicate functionality already in `src/engine`; follow the existing A2A message + `tracer.record_step(state, ...)` + structured `logger` pattern used by every agent.
- Blocked/promo/advertorial content is filtered out of RSS and RAG candidates (covered by the smoke test's RAG/publisher path); keep this intact — it's a monetization/quality invariant.
- Publish path enforces YouTube quota (max ~4 uploads/day) and EU AI Act synthetic-content disclosure tags; don't bypass in tests or new features.
- Story requires live LLM and quality gates (Observer + Gates 1/3b/4/5/6/7) pass before publish. Fact/temporal/revenue/audience violations and quality gates 1/3b/4/5/6/7 are hard abort conditions — never warnings. Observer style-class violations are soft (see Semantic quality gates section).
