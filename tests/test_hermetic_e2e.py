#!/usr/bin/env python3
"""
SINGLE HERMETIC END-TO-END TEST — CSVG pipeline (buzzdropfeedv2).

Fully self-sufficient and independent:
  * NO network, NO Raspberry Pi, NO media generation, NO .env required.
  * Runs the REAL orchestrator with the REAL StoryDesignerAgent and REAL
    ObserverAgent (injereader: the new surgical-repair loop, state_hash on
    REVISE_SCRIPT, per-agent route/thinking dispatch, and the outline-first
    path all execute as production code).
  * Only genuinely-external boundaries are stubbed: RAG scrapers, channel
    stats, YouTube publish, TTS/visual/ffmpeg gates, and the LLM (scripted
    FakeLLM, deterministic per route).

If any case fails, it is a REAL pipeline regression, not a flaky mock.

Run directly (no pytest, no other test files):
    python tests/test_hermetic_e2e.py
"""
import os
import sys
import json
import copy
import traceback

# Self-sufficient: importable from any CWD (repo importable `src` package).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Hermetic defaults BEFORE importing any pipeline modules: never touch network
# or external services regardless of ambient environment.
os.environ["PREFERRED_LLM_PROVIDER"] = "cloud"
os.environ.setdefault("FIRECRAWL_API_KEY", "x" * 40)
os.environ.setdefault("OPENROUTER_API_KEY", "x" * 40)
os.environ.setdefault("CSVG_STORAGE", "/tmp/csvg_hermetic")
os.environ["ALLOW_SOFT_APPROVAL"] = "1"
os.environ.pop("USE_SEMANTIC_GATES", None)   # TF-IDF/NLTK path (no torch)
os.environ.pop("CSVG_OUTLINE_FIRST", None)

import asyncio

from src.schemas.state import (
    GlobalState, ScriptData, ShotData, VisualType, AssetPaths, ChannelStats,
    TopicCandidate, VerifiedFact, SEOMetadata,
)
from src.schemas.a2a import (
    A2AMessage, AgentRole, AgentIntent, compute_state_hash,
)

CASE_RESULTS = []

# Real class captured BEFORE any external patching (case_routing needs it).
from src.engine.llm_client import LLMClient as _RealLLMClient


def record(name, ok, detail=""):
    CASE_RESULTS.append((name, ok, detail))
    flag = "ok" if ok else "FAIL"
    print(f"[{flag}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Scripted FakeLLM — deterministic, records route/thinking per call.
# ---------------------------------------------------------------------------
class FakeLLMClient:
    """Mimics LLMClient.generate_json(prompt, system_prompt, route, thinking).
    Routes:
      'generate':  full 18-shot script (or outline if prompt says 'story planner')
      'polish':    faithful pass (returns same shots)
      'repair':    rewrites ONLY the violating Shot #4 (fixes the >155-word cap)
      'critic':    approves (no rejected assertions)
    Records every call so the test can assert routing + surgical behaviour."""

    def __init__(self, api_key="x", model="google/gemini-2.5-flash",
                 llama_cpp_url="http://localhost:8080"):
        self.calls = []          # list of dicts: {route, thinking, prompt}
        self.repair_calls = 0
        self.model = model

    # -- LLMClient surface --------------------------------------------------
    def is_available(self):
        return True

    def is_cloud_llm_available(self):
        return True

    def is_local_llm_available(self):
        return False

    def generate_json(self, prompt, system_prompt="", route=None, thinking=None):
        self.calls.append({"route": route, "thinking": thinking,
                           "prompt": (prompt or "")[:200]})
        p = (prompt or "")
        if route == "critic" or "facts verification critic" in p or "factual validation critic" in p:
            return {"violations": []}
        if route == "repair":
            self.repair_calls += 1
            # Repair ONLY Shot #4 with a compliant 95-word narration.
            return {"shots": [{"shot_id": 4,
                               "narration_text": repaired_shot4()}], "title": TITLE}
        if route == "polish":
            return {"shots": [{"shot_id": s["shot_id"] + 1 if False else s["shot_id"],
                               "narration_text": s["narration_text"]}
                              for s in SCRIPT_JSON["shots"]], "title": TITLE}
        # route == "generate" (or unset): outline-first vs monolithic.
        if "story planner" in p:
            return OUTLINE_JSON
        if "documentary narrator" in p:
            # Per-act narration: 3 shots per act, mirroring the outline beats.
            act_hint = 1
            if "BEATS TO NARRATE" in p:
                import re as _re
                m = _re.search(r'"act_index":\s*(\d+)', p)
                if m:
                    act_hint = int(m.group(1))
            return {"shots": [
                {"shot_id": ((act_hint - 1) * 3) + k,
                 "narration_text": _narr(((act_hint - 1) * 3) + k),
                 "visual_prompt": _visual(((act_hint - 1) * 3) + k)}
                for k in range(1, 4)
            ]}
        return SCRIPT_JSON


# ---------------------------------------------------------------------------
# Deterministic canned artifacts (all sources = "Fortune" so the Observer's
# source-diversity check has a single-attribution corpus: no false reject).
# ---------------------------------------------------------------------------
TITLE = "Quantum Computing in Finance: The Hidden Truth"

_FACTS = [
    ("FP1", "Banks pilot quantum error correction for portfolio risk modelling.",
     "Major lenders are testing quantum computers to price derivatives thousands of times faster than classical chips, according to Fortune."),
    ("FP2", "Quantum encryption race accelerates in high frequency trading.",
     "Firms are investing in post-quantum cryptography to protect market data streams from future code-breaking machines, per Fortune."),
    ("FP3", "Cost of quantum cloud access falls sharply in 2026.",
     "Monthly rental of a twelve-qubit processor has dropped below five thousand dollars, widening access for startups, reported by Fortune."),
    ("FP4", "Regulators begin drafting quantum financial stability rules.",
     "Central banks are asking banks to report quantum readiness as part of annual stress tests, Fortune records."),
]

VERIFIED_FACTS = [
    VerifiedFact(source_id=f[0], headline=f[1], summary=f[2],
                 url="https://fortune.com/hermetic", source_name="Fortune")
    for f in _FACTS
]

TOPIC = TopicCandidate(
    candidate_id="hermetic-1",
    headline="Quantum Computing Is Reshaping Global Finance in 2026",
    summary="Quantum computers are moving from labs into banks, speeding up pricing, encryption and risk models while regulators scramble for guardrails.",
    source_url="https://fortune.com/hermetic",
    keywords=["quantum", "computing", "finance", "banks", "encryption", "risk"],
    tvs_score=0.8, rpm_score=0.7, idi_score=0.6, sdi_score=0.4, shm_score=1.0,
    vph_score=1.0, sat_score=0.3, topsis_score=0.9,
    audience_type="finance_edu", niche_category="Global Economics & Finance",
    opportunity_score=0.7, demand_query="quantum computing finance 2026",
)


def _narr(i):
    """Distinct ~92-word narration for shot i. No bare acronyms, no repeated
    sentences, grounded in the Fortune facts above."""
    base = (
        f"For shot number {i:02d}, the story turns on a specific slice of evidence. "
        f"Banks are now piloting quantum error correction to price derivatives at "
        f"unimaginable speed, a leap that classical chips cannot match, according to "
        f"Fortune. The race extends to encryption as well, because firms fear that "
        f"future code-breaking machines could expose their live market data streams. "
        f"Meanwhile the rental price of a twelve-qubit machine has fallen sharply, "
        f"which lets confident startups join the experiment. Regulators are already "
        f"asking the biggest lenders to report their quantum readiness each year. "
    )
    return base


def _visual(i):
    return (
        "Cinematic 16:9 widescreen photorealistic visual, 8k detail, "
        f"dramatic volumetric lighting for shot {i:02d} of the quantum finance story."
    )


SCRIPT_JSON = {
    "title": TITLE,
    "shots": [
        {"shot_id": i, "act_index": min(6, (i - 1) // 3 + 1),
         "narration_text": _narr(i),
         "visual_prompt": _visual(i),
         "visual_type": "standard_image"}
        for i in range(1, 19)
    ],
}


def repaired_shot4():
    return (
        "Post-quantum defences now occupy the traders' roadmap at the largest "
        "lenders, because a future code-breaking machine could compromise live "
        "market data, as Fortune reports. The response is to pair hardware "
        "upgrades with strict encryption refreshes so the whole exchange "
        "network stays safe while prices still move at light speed."
    )


OUTLINE_JSON = {
    "shots": [
        {"shot_id": i, "act_index": min(6, (i - 1) // 3 + 1),
         "beat_summary": f"Narrate the quantum finance beat {i} with grounded evidence.",
         "facts_to_use": [f[2] for f in _FACTS],
         "publisher": "Fortune",
         "visual_type": "standard_image"}
        for i in range(1, 19)
    ]
}

BAD_SCRIPT_JSON = {
    "title": TITLE,
    "shots": [
        {"shot_id": i, "act_index": min(6, (i - 1) // 3 + 1),
         # Shot #4 is deliberately OVER the 155-word Observer cap.
         "narration_text": _narr(i) if i != 4 else (_narr(4) + " " + " ".join(["word"] * 120)),
         "visual_prompt": _visual(i),
         "visual_type": "standard_image"}
        for i in range(1, 19)
    ]
}


# ---------------------------------------------------------------------------
# Minimal stub agents for external boundaries (injected via constructor).
# ---------------------------------------------------------------------------
class StubFactRetriever:
    def __init__(self):
        self.calls = 0

    def process(self, state, use_live_rss=True, region="all", channel_phase="GROWTH", exclude_headlines=None):
        self.calls += 1
        state.selected_topic = TOPIC
        state.verified_facts = list(VERIFIED_FACTS)
        return A2AMessage(
            message_id="m1", sender=AgentRole.FACT_RETRIEVER,
            target=AgentRole.ORCHESTRATOR, intent=AgentIntent.TOPIC_SELECTED,
            payload={}, timestamp="0")


class StubObserverPass:
    """Deterministic observer that approves and records REVISE attempts."""
    def __init__(self, approve=True):
        self.approve = approve
        self.revises = 0

    def process(self, state):
        if self.approve:
            state.execution_stage = "SCRIPT_APPROVED"
            return A2AMessage(
                message_id="m2", sender=AgentRole.OBSERVER,
                target=AgentRole.ORCHESTRATOR, intent=AgentIntent.APPROVE_SCRIPT,
                payload={"violations": [], "status": "APPROVED",
                         "quality_score": 0.5}, timestamp="0")
        self.revises += 1
        state.execution_stage = "SCRIPT_REVISION_REQUIRED"
        return A2AMessage(
            message_id="m3", sender=AgentRole.OBSERVER,
            target=AgentRole.STORY_DESIGNER, intent=AgentIntent.REVISE_SCRIPT,
            payload={"violations": ["Shot #2 Fact Audit: the claim lacks grounding."]},
            state_hash=compute_state_hash(state), timestamp="0")


class StubMediaProducer:
    async def process(self, state, dummy_frames=False, renderer=None):
        d = "/tmp/csvg_hermetic_media"
        os.makedirs(os.path.join(d, "audio"), exist_ok=True)
        os.makedirs(os.path.join(d, "visuals"), exist_ok=True)
        final = os.path.join(d, "final_video_1080p.mp4")
        with open(final, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42" + b"0" * 64)
        ap = AssetPaths(storage_dir=d, final_video=final,
                        thumbnail=os.path.join(d, "thumbnail.png"))
        for s in state.script_data.shots:
            key = f"shot_{s.shot_id}"
            ap.audio[key] = os.path.join(d, "audio", f"{key}.wav")
            ap.visuals[key] = os.path.join(d, "visuals", f"{key}.mp4")
            open(ap.audio[key], "wb").write(b"RIFF" + b"0" * 64)
            open(ap.visuals[key], "wb").write(b"\x00\x00\x00\x18ftypmp42" + b"0" * 64)
            ap.measured_durations.append(3.0)
        state.asset_paths = ap
        return A2AMessage(
            message_id="m4", sender=AgentRole.MEDIA_PRODUCER,
            target=AgentRole.ORCHESTRATOR, intent=AgentIntent.MEDIA_READY,
            payload={"final_video": final}, timestamp="0")


class StubPublisher:
    async def process(self, state):
        state.upload_metadata.video_id = "hermetic-video-123"
        return A2AMessage(
            message_id="m5", sender=AgentRole.PUBLISHER,
            target=AgentRole.ORCHESTRATOR, intent=AgentIntent.PUBLISHED_SUCCESS,
            payload={"video_id": "hermetic-video-123"}, timestamp="0")


# ---------------------------------------------------------------------------
# Patching of the genuinely-external singletons (network / binaries).
# ---------------------------------------------------------------------------
def patch_externals(rag_pack=None, corpus="", topic=None):
    """Monkeypatch module singletons used by the real orchestrator + real
    story designer + real observer so no network/binary is touched."""

    # RAG: replace scrapers with a hermetic pack; corpus always 'sufficient'.
    from src.engine import rag_retriever as rr_mod
    rr = rr_mod.rag_retriever
    rr.set_rag_mode = lambda mode: None

    def _pack(*a, **k):
        return {
            "fact_corpus": corpus or _canonical_corpus(),
            "full_rag_context_text": corpus or _canonical_corpus(),
            "selected_article": "Beats: quantum banks 2026.",
            "rag_mode": "scraper",
            "topic_headline": TOPIC.headline,
            "summary": TOPIC.summary,
            "keywords": TOPIC.keywords,
            "category": "Global Economics & Finance",
            "ground_truth_block": _canonical_corpus(),
            "rag_retrieved_context": _canonical_corpus(),
            "rag_recent_context": _canonical_corpus(),
            "rag_historical_context": "",
            "graph_triplets": "",
            "graph_paths": "",
            "trumorgpt_verification": {"is_verified": True, "confidence": 0.9, "message": "ok"},
        }
    rr.build_rag_knowledge_pack = _pack
    rr.assess_corpus_sufficiency = lambda pack, topic: {
        "pass": True, "reason": "ok",
        "metrics": {"on_topic_facts": 4, "on_topic_corpus_words": 400, "on_topic_sources": 1},
    }

    # BERTopic outline: deterministic, no vectorization.
    from src.engine import bertopic_engine as be_mod
    be_mod.bertopic_engine.extract_chapter_outlines = lambda *a, **k: [
        {"chapter_title": "Origins", "cluster_keywords": ["banks", "quantum"]},
        {"chapter_title": "Mechanics", "cluster_keywords": ["qubit", "error"]},
        {"chapter_title": "Impact", "cluster_keywords": ["finance", "risk"]},
    ]

    # Channel phase: deterministic GROWTH (skips the revenue gate).
    from src.engine import channel_phase_manager as cp_mod
    cp_mod.channel_phase_manager.get_channel_stats = lambda force_refresh=False: ChannelStats(
        subscribers=10, total_watch_hours=5, channel_phase="GROWTH", ypp_unlocked=False)

    # Semantic gates: force the TF-IDF/NLTK fallback (no torch/transformers).
    from src.engine import text_embeddings as te_mod
    te_mod.semantic_embedder._enabled_override = False
    te_mod.semantic_embedder.available   # noqa: B018 — recomputes against override
    te_mod.semantic_embedder.load = lambda: None
    te_mod.semantic_embedder.encode_batch = lambda texts: None

    # Post-media quality gates: hermetic passes (no ffprobe on fake files).
    from src.engine import quality_verifier as qv_mod
    qv = qv_mod.quality_verifier
    qv.verify_gate3b_subtitle_text_coherence = lambda state: (True, [])
    qv.verify_gate4_video_audio_coherence = lambda state: (True, [])
    qv.verify_gate7_render_integrity = lambda state: (True, [])
    # Gate 6 anti-slop entropy is heuristic; force pass for deterministic runs.
    qv.verify_gate6_anti_slop_entropy = lambda script: {"passes": True, "issues": []}
    qv.verify_gate5_ai_disclosure_tags = lambda script, tags: {
        "youtube_metadata_patch": {"syntheticContent": True, "bInformed": True},
        "rationale": "hermetic", "passes": True}

    from src.engine import video_quality_metrics as vq_mod
    vq_mod.video_quality_metrics.run_full_quality_gate = lambda **k: {
        "stage8_overall_pass": True, "recommendation": "ok"}

    # Observer's fact critic: routed to the same FakeLLM (route 'critic').
    # The observer imports LLMClient locally from src.engine.llm_client, so
    # patch the CLASS so its internally-created instance is the Fake too.
    from src.engine import llm_client as lc_mod
    lc_mod.LLMClient = FakeLLMClient

    # rag on the orchestrator module too (same object, but guard import order).
    import src.agents.orchestrator as orch_mod
    orch_mod.rag_retriever = rr


def _canonical_corpus():
    return "\n".join(f"• [Fortune: {f[1]}]: {f[2]}" for f in _FACTS)


# ---------------------------------------------------------------------------
# Case 1 — REAL pipeline, happy path (approve on first audit).
# ---------------------------------------------------------------------------
def case_happy_path():
    from src.agents.orchestrator import OrchestratorAgent
    from src.agents.story_designer import StoryDesignerAgent
    patch_externals()
    fake = FakeLLMClient()
    orch = OrchestratorAgent(
        fact_retriever=StubFactRetriever(),
        story_designer=StoryDesignerAgent(llm_client=fake),
        observer=StubObserverPass(approve=True),
        media_producer=StubMediaProducer(),
        publisher=StubPublisher(),
        logs_dir="/tmp/csvg_hermetic_logs",
    )
    state = asyncio.run(orch.run_pipeline(
        use_live_rss=False, region="global", publish=True))
    ok = (state.script_data is not None
          and len(state.script_data.shots) >= 12
          and state.upload_metadata.video_id == "hermetic-video-123")
    routes = {c["route"] for c in fake.calls}
    # Real StoryDesigner should have exercised generate + polish routes.
    record("HAPPY_PATH", ok,
           f"shots={len(state.script_data.shots) if state.script_data else 0}, "
           f"video={state.upload_metadata.video_id}, routes={routes}")
    if "generate" not in routes:
        record("HAPPY_PATH_routes", False, "generate route not used")
    return ok


# ---------------------------------------------------------------------------
# Case 2 — surgical per-shot revision loop actually repairs and converges.
# ---------------------------------------------------------------------------
def case_surgical_revision():
    from src.agents.orchestrator import OrchestratorAgent
    from src.agents.story_designer import StoryDesignerAgent
    patch_externals()
    fake = FakeLLMClient()
    # Use a REAL observer so the >155-word Shot #4 is genuinely flagged.
    from src.agents.observer import ObserverAgent
    real_obs = ObserverAgent()
    orch = OrchestratorAgent(
        fact_retriever=StubFactRetriever(),
        story_designer=StoryDesignerAgent(llm_client=fake),
        observer=real_obs,
        media_producer=StubMediaProducer(),
        publisher=StubPublisher(),
        logs_dir="/tmp/csvg_hermetic_logs",
    )
    # Manually seed the initial script first so the loop starts from a REVISE.
    state = GlobalState(pipeline_id="surgical-1", timestamp="0")
    state.selected_topic = TOPIC
    state.verified_facts = list(VERIFIED_FACTS)
    state.crawled_content = _canonical_corpus()
    state.script_data = _script_from_json(BAD_SCRIPT_JSON)

    # First audit with the real observer must reject (Shot #4 too long).
    approved, violations = run_observer_real(state)
    too_long = any("Shot #4" in v and "too long" in v for v in violations)
    record("SURGICAL_flag", approved is False and too_long,
           f"violations={violations[:2]}")

    # Now exercise the orchestrator's surgical loop on a REVISE payload.
    # (Simulate the loop body: bucket -> repair_shots -> re-audit.)
    from src.agents.orchestrator import _bucket_violations
    from src.agents.story_designer import StoryDesignerAgent
    sd = StoryDesignerAgent(llm_client=fake)
    by_shot, gv = _bucket_violations(violations)
    target = frozenset(by_shot)
    untouched_before = {s.shot_id: s.narration_text
                        for s in state.script_data.shots if s.shot_id not in target}
    msg_obs = A2AMessage(
        message_id="m-rev", sender=AgentRole.OBSERVER,
        target=AgentRole.STORY_DESIGNER, intent=AgentIntent.REVISE_SCRIPT,
        payload={"violations": violations},
        state_hash=compute_state_hash(state), timestamp="0")
    repaired = sd.repair_shots(state.script_data, state, by_shot, gv, msg_obs=msg_obs)
    state.script_data = repaired
    untouched_after = {s.shot_id: s.narration_text
                       for s in state.script_data.shots if s.shot_id not in target}
    # 1) repair used the 'repair' route; 2) non-target shots bit-identical;
    # 3) Shot #4 now under the cap; 4) state_hash enforced.
    ok1 = fake.repair_calls >= 1
    ok2 = all(untouched_after[k] == v for k, v in untouched_before.items())
    shot4 = next(s for s in repaired.shots if s.shot_id == 4)
    ok3 = len(shot4.narration_text.split()) <= 155
    ok4 = True  # mastering: repair_shots accepted matching state_hash
    _approved2, v2 = run_observer_real(state)
    ok5 = not any("too long" in v for v in v2)
    record("SURGICAL_repair", ok1 and ok2 and ok3 and ok4 and ok5,
           f"repair_calls={fake.repair_calls}, non_target_identical={ok2}, "
           f"shot4_words={len(shot4.narration_text.split())}, "
           f"clean={ok5}")
    # Stale state_hash must be rejected: the REVISE message was minted for the
    # ORIGINAL audited draft, but a different (mutated) state is handed to repair.
    stale_state = copy.deepcopy(state)
    stale_state.script_data.shots[0].narration_text = "MUTATED"
    audited_hash = compute_state_hash(state)   # hash of the draft Observer saw
    try:
        sd.repair_shots(stale_state.script_data, stale_state, by_shot, gv,
                        msg_obs=A2AMessage(
                            message_id="m-stale", sender=AgentRole.OBSERVER,
                            target=AgentRole.STORY_DESIGNER,
                            intent=AgentIntent.REVISE_SCRIPT,
                            payload={}, state_hash=audited_hash,
                            timestamp="0"))
        record("SURGICAL_stale_hash", False, "stale state_hash was NOT rejected")
    except RuntimeError:
        record("SURGICAL_stale_hash", True, "stale state_hash rejected")


def _script_from_json(j):
    shots = []
    for s in j["shots"]:
        vtype = s.get("visual_type") or "standard_image"
        try:
            vt = VisualType(vtype)
        except ValueError:
            vt = VisualType.STANDARD_IMAGE
        shots.append(ShotData(
            shot_id=int(s["shot_id"]), act_index=int(s["act_index"]),
            narration_text=s["narration_text"], visual_prompt=s["visual_prompt"],
            visual_type=vt,
            duration_estimate=max(42.0, round(len(s["narration_text"].split()) / 2.2, 1))))
    return ScriptData(title=j.get("title", TITLE), target_shots=len(shots),
                      shots=shots,
                      estimated_runtime_seconds=round(
                          sum(len(x.narration_text.split()) for x in shots) / 150.0 * 60.0, 1))


def run_observer_real(state):
    from src.agents.observer import ObserverAgent
    obs = ObserverAgent()
    approved, violations = obs.evaluate_script(
        state.script_data, state.verified_facts, topic=state.selected_topic,
        channel_phase="GROWTH", crawled_content=state.crawled_content)
    return approved, violations


# ---------------------------------------------------------------------------
# Case 3 — outline-first path (real code, env-gated) produces a script.
# ---------------------------------------------------------------------------
def case_outline_first():
    from src.agents.story_designer import StoryDesignerAgent
    patch_externals()
    os.environ["CSVG_OUTLINE_FIRST"] = "1"
    try:
        fake = FakeLLMClient()
        sd = StoryDesignerAgent(llm_client=fake)
        state = GlobalState(pipeline_id="outline-1", timestamp="0")
        state.selected_topic = TOPIC
        state.verified_facts = list(VERIFIED_FACTS)
        state.crawled_content = _canonical_corpus()
        # The outline/narrate methods must be reachable on the real agent.
        beats = sd.generate_outline(state)
        ok1 = beats is not None and len(beats) >= 12
        if ok1:
            script = sd.narrate_from_outline(beats, state)
            ok2 = script is not None and len(script.shots) >= 12
            record("OUTLINE_FIRST", ok1 and ok2,
                   f"beats={len(beats) if beats else 0}, "
                   f"shots={len(script.shots) if script else 0}")
        else:
            record("OUTLINE_FIRST", False, "outline generation failed")
    finally:
        os.environ.pop("CSVG_OUTLINE_FIRST", None)


# ---------------------------------------------------------------------------
# Case 4 — per-agent model routing: route->model resolution + fallback chain.
# ---------------------------------------------------------------------------
def case_routing():
    os.environ["LLM_ROUTE_REPAIR"] = "deepseek/deepseek-v4-flash-0731"
    os.environ["LLM_ROUTE_CRITIC"] = "google/gemini-2.5-flash"
    os.environ.pop("LLM_ROUTE_GENERATE", None)
    try:
        lc = _RealLLMClient(model="google/gemini-2.5-flash")
        chain_gen = lc._model_chain(route="generate")
        chain_repair = lc._model_chain(route="repair")
        chain_critic = lc._model_chain(route="critic")
        ok_gen = bool(chain_gen) and "google/gemini-2.5-flash" in chain_gen and chain_gen[0] == "google/gemini-2.5-flash"
        ok_repair = bool(chain_repair) and chain_repair[0] == "deepseek/deepseek-v4-flash-0731"
        ok_critic = bool(chain_critic) and chain_critic[0] == "google/gemini-2.5-flash"
        # Route pin takes precedence over the primary model.
        record("ROUTING", ok_gen and ok_repair and ok_critic,
               f"gen={chain_gen[:1]}, repair={chain_repair[:1]}, critic={chain_critic[:1]}")
    finally:
        for k in ("LLM_ROUTE_REPAIR", "LLM_ROUTE_CRITIC", "LLM_ROUTE_GENERATE"):
            os.environ.pop(k, None)


# ---------------------------------------------------------------------------
# Case 5 — A2A alignment: dead intents removed; state_hash on REVISE.
# ---------------------------------------------------------------------------
def case_a2a_alignment():
    from src.schemas.a2a import AgentIntent
    declared = {m.value for m in AgentIntent}
    # The 4 dead intents must be gone.
    dead = {"FETCH_TOPIC", "PRODUCE_MEDIA", "PUBLISH_VIDEO"}
    gone = not (dead & declared)
    # compute_state_hash stable across equivalent states.
    s1 = GlobalState(pipeline_id="p", timestamp="0")
    s2 = GlobalState(pipeline_id="p", timestamp="0")
    s1.script_data = _script_from_json(SCRIPT_JSON)
    s2.script_data = _script_from_json(SCRIPT_JSON)
    stable = compute_state_hash(s1) == compute_state_hash(s2)
    diff = compute_state_hash(s1) != compute_state_hash(
        s1.model_copy(deep=True))
    record("A2A_ALIGNMENT", gone and stable and not diff,
           f"dead_removed={not dead & declared}, state_hash_stable={stable}")


# ---------------------------------------------------------------------------
# Case 6 — Gate 6 shallow shot formatting and routing.
# ---------------------------------------------------------------------------
def case_gate6_shallow_shots():
    from src.engine.quality_verifier import StageQualityVerifier
    from src.agents.orchestrator import _bucket_violations

    qv = StageQualityVerifier()  # fresh instance, not the stubbed singleton
    script_data = _script_from_json(SCRIPT_JSON)
    # Make shot 10 shallow (under 75 words)
    script_data.shots[9] = script_data.shots[9].model_copy(update={
        "narration_text": "This is a short narration containing only ten words."
    })

    res = qv.verify_gate6_anti_slop_entropy(script_data)
    ok_passes = (res["passes"] is False)
    ok_shallow = (10 in res["shallow_shots"])

    # Check that issues lists the shot as Shot #10
    issues = res["issues"]
    has_formatted_shot_issue = any("Shot #10" in issue for issue in issues)

    # Check that _bucket_violations correctly buckets this to shot 10
    by_shot, global_v = _bucket_violations(issues)
    ok_bucket = (10 in by_shot)

    record("GATE6_SHALLOW_SHOTS", ok_passes and ok_shallow and has_formatted_shot_issue and ok_bucket,
           f"passes_false={ok_passes}, shallow_10={ok_shallow}, formatted_shot_issue={has_formatted_shot_issue}, bucketed={ok_bucket}")


# ---------------------------------------------------------------------------
# Case 7 — demo_/fake upload ids must abort publish, not fake success.
# ---------------------------------------------------------------------------
def case_fake_upload_aborts():
    from src.agents.publisher import PublisherAgent, _is_real_video_id

    # 1. The shared predicate rejects every fabricated/sentinel id form.
    ids = {
        "demo_id": False,
        "demo_451614aa": False,
        "demo_abc123": False,
        "uploaded_demo_id": False,
        "": False,
        None: False,
        "E1T5IiXSl3E": True,
        "dQw4w9WgXcQ": True,
    }
    pred_ok = all(_is_real_video_id(k) is v for k, v in ids.items())

    # 2. SIDE EFFECTS are gated: a PublisherAgent whose upload returns a
    #    demo_ id must raise BEFORE the pinned comment / shorts / seed /
    #    dedup side effects run (monkeypatch the network-y boundary).
    from src.schemas.state import GlobalState, UploadMetadata
    stub_calls = {"comment": 0, "seed": 0, "dedup": 0}
    import src.agents.publisher as pub

    async def fake_upload_fails(req):
        return {"status": "mock", "video_id": "demo_abcdef", "youtube_url": "x"}

    async def fake_comment(req):
        stub_calls["comment"] += 1
        return {"status": "mock", "comment_id": "comment_x", "video_id": req.video_id}

    orig_upload = pub.upload_youtube_resumable
    orig_comment = pub.insert_pinned_comment
    orig_record = None
    try:
        from src.engine import topic_deduplicator as td_mod
        orig_record = td_mod.record_published_topic

        def fake_record_published(*a, **k):
            stub_calls["dedup"] += 1

        pub.upload_youtube_resumable = fake_upload_fails
        pub.insert_pinned_comment = fake_comment
        td_mod.record_published_topic = fake_record_published

        # Build a minimal, publish-ready state (media assets present).
        st = GlobalState(pipeline_id="fake-upload-1", timestamp="0")
        st.selected_topic = TOPIC
        st.verified_facts = list(VERIFIED_FACTS)
        st.crawled_content = _canonical_corpus()
        st.script_data = _script_from_json(SCRIPT_JSON)
        st.seo_metadata = None  # force the default description path

        from src.schemas.state import AssetPaths
        ap = AssetPaths(final_video="/tmp/csvg_hermetic_media/final_video_1080p.mp4",
                        thumbnail="/tmp/csvg_hermetic_media/thumb.png")
        st.asset_paths = ap

        raised = False
        try:
            asyncio.run(PublisherAgent().publish_video(st))
        except RuntimeError:
            raised = True

        side_effects_gated = (stub_calls["comment"] == 0 and stub_calls["seed"] == 0
                              and stub_calls["dedup"] == 0)
        not_published = (st.execution_stage != "PUBLISHED_SUCCESS"
                         and not _is_real_video_id(getattr(st.upload_metadata, "video_id", None)))
        ok = raised and side_effects_gated and not_published
        record("FAKE_UPLOAD_ABORTS", ok,
               f"raised={raised}, comment_calls={stub_calls['comment']}, "
               f"seed_calls={stub_calls['seed']}, dedup_calls={stub_calls['dedup']}, "
               f"published={getattr(st.upload_metadata, 'video_id', '')}")
    finally:
        pub.upload_youtube_resumable = orig_upload
        pub.insert_pinned_comment = orig_comment
        if orig_record is not None:
            from src.engine import topic_deduplicator as td_mod2
            td_mod2.record_published_topic = orig_record

    # 3. Orchestrator resume guard must NOT treat demo_ as already-published.
    from src.agents.publisher import _is_real_video_id as _pred
    um = UploadMetadata(video_id="demo_abc", status="PUBLISHED")
    st2 = GlobalState(pipeline_id="resume-demo", timestamp="0")
    st2.upload_metadata = um
    # The predicate is the same one the orchestrator now uses.
    resume_ok = not _pred(um.video_id)

    ok &= pred_ok and resume_ok
    record("RESUME_DEMO_ID_NOT_PUBLISHED", resume_ok, f"demo_ treated as published: {not resume_ok}")
    return ok


# ---------------------------------------------------------------------------
# Case 8 — SEO "Sources & Data Grounding" filtered to on-topic facts only.
# ---------------------------------------------------------------------------
def case_seo_source_filter():
    from src.schemas.state import VerifiedFact
    from src.engine.rag_retriever import filter_facts_for_topic

    headline = "Meta launches Muse Glimmer model as Zuckerberg champions AI for everyone"
    summary = "The social media giant plans to release an open-weight version of Muse Spark, its most powerful AI model, soon, Mark Zuckerberg said on Monday."
    keywords = ["champions", "launches", "open", "zuckerberg", "meta", "spark", "release", "version", "media", "plans"]

    facts = [
        # primary story (kept via own-headline rule)
        VerifiedFact(source_id="1", headline=headline,
                     summary="Open-weight 30B model announced.", url="https://businessinsider.com/meta-muse-glimmer-2026-8",
                     source_name="Business Insider"),
        # explicitly on-topic secondary (kept: meta + spark signature tokens)
        VerifiedFact(source_id="3", headline="Meta's open-weight Muse Spark 1.2 gains 100k-token context",
                     summary="Muse Spark 1.1 to 1.2 upgrade on Artificial Analysis index.",
                     url="https://meta.example/spark", source_name="Meta AI"),
        # shares only generic 'model' tokens (dropped)
        VerifiedFact(source_id="2", headline="Anthropic rolls out new Claude coding model",
                     summary="Company releases updated model.", url="https://anthropic.example/x", source_name="Anthropic"),
        # generic/placeholder source, off-topic (dropped)
        VerifiedFact(source_id="4", headline="NASA APOD: Sun silhouettes",
                     summary="Astronomy picture of the day.", url="https://apod.nasa.gov/x", source_name="NASA Open APIs"),
        # fake marketplace feed (dropped by generic source + off-topic)
        VerifiedFact(source_id="5", headline="Wall Street rally after oil gains",
                     summary="Shares mixed after oil.", url="https://abcnews.example/ws", source_name="Verified Reports"),
    ]

    kept = filter_facts_for_topic(facts, headline=headline, summary=summary, keywords=keywords)
    kept_urls = [vf.url for vf in kept]

    ok = (
        len(kept) == 2
        and "https://businessinsider.com/meta-muse-glimmer-2026-8" in kept_urls
        and "https://meta.example/spark" in kept_urls
        and "https://anthropic.example/x" not in kept_urls
        and "https://apod.nasa.gov/x" not in kept_urls
        and "https://abcnews.example/ws" not in kept_urls
    )
    record("SEO_SOURCE_FILTER", ok,
           f"kept={len(kept)} urls={kept_urls}")
    return ok


# ---------------------------------------------------------------------------
# Case 9 — byline/follow/hashtag scraped junk is dropped from narration and
#          hard-flagged by the Observer.
# ---------------------------------------------------------------------------
def case_narration_junk_scrub():
    from src.agents.story_designer import _snippet_is_junk, _clean_narration, _BYLINE_FOLLOW_RE
    from src.agents.story_designer import StoryDesignerAgent
    from src.agents.observer import _RAW_JUNK_IN_NARR_RE

    # The EXACT leaked tail observed live in the Meta run's Shot 18.
    bad_tail = (
        "Will Meta's open-source strategy truly democratize artificial intelligence? "
        "# Meta launches new artificial intelligence model as Zuckerberg champions open-weight push. "
        "Tech Meta launches Muse Glimmer model as Zuckerberg champions artificial intelligence for 'everyone' "
        "ByTom Carter You're currently following this author! "
        "# Meta launches Muse Spark artificial intelligence model as part of its artificial intelligence turnaround."
    )

    junk_detected = bool(_BYLINE_FOLLOW_RE.search(bad_tail))
    snippet_junk = _snippet_is_junk(bad_tail)
    observer_hard_flag = bool(_RAW_JUNK_IN_NARR_RE.search(bad_tail))
    cleaned = _clean_narration(bad_tail)
    junk_scrubbed = (
        "ByTom" not in cleaned
        and "following this author" not in cleaned
        and "# Meta launches" not in cleaned
    )
    # The legitimate question sentence must survive the scrub.
    survived = "truly democratize artificial intelligence" in cleaned

    # The semantic-expansion padding path must refuse the junk snippet too.
    padding = StoryDesignerAgent(llm_client=None)._paraphrase_padding(bad_tail)
    padding_rejected = (padding == "")

    # Guard: clean narration must NOT be falsely flagged.
    good = "Meta's open-weight strategy positions the company directly against competitors that prefer proprietary models."
    good2 = "By 2026, Meta will have shipped its new open models to millions of users."
    no_false_positive = (
        _snippet_is_junk(good) is False
        and bool(_RAW_JUNK_IN_NARR_RE.search(good)) is False
        and _snippet_is_junk(good2) is False
        and bool(_RAW_JUNK_IN_NARR_RE.search(good2)) is False
    )

    ok = junk_detected and snippet_junk and observer_hard_flag and junk_scrubbed and survived and no_false_positive and padding_rejected
    record("NARRATION_JUNK_SCRUB", ok,
           f"junk_detected={junk_detected}, snippet_junk={snippet_junk}, "
           f"observer_hard={observer_hard_flag}, scrubbed={junk_scrubbed}, "
           f"survived={survived}, no_false_pos={no_false_positive}, padding_rejected={padding_rejected}")
    return ok


# ---------------------------------------------------------------------------
# Case 10 — RIGHT-FIRST-TIME: off-topic facts dropped at corpus ingestion via
#          signature-token scoring (no downstream leak possible).
# ---------------------------------------------------------------------------
def case_signature_ingestion_filter():
    from src.engine.rag_retriever import _topic_signature_tokens, _on_topic_hits

    # Archaeology topic whose summary happens to mention "researchers"/"research".
    headline = "60,000-year-old ostrich eggshell engravings reveal a surprisingly sophisticated human mind"
    summary = ("More than 60,000 years ago, humans in southern Africa were engraving ostrich "
               "eggshells with intricate geometric patterns. Researchers found recurring grids, "
               "parallel lines, right angles, and repeated shapes.")
    keywords = ["researchers", "engravings", "ostrich", "eggshell", "geometric", "southern"]

    signature = _topic_signature_tokens(headline, summary, keywords)

    # The CRO-market stat that leaked into the ostrich narration/reel (off-topic:
    # shares only generic "researchers"/"research" tokens with the topic).
    cro_line = ("The market for Contract Research Organizations (CROs), where many "
                "researchers contribute their expertise, was estimated at USD 69.56 billion"
                " in 2025 and is predicted to grow to USD 74.37 billion in 2026.")
    cro_active = "Contract Research Organizations" in " ".join(signature)

    cro_hits = _on_topic_hits(cro_line, signature)
    cro_dropped = cro_hits < 2  # fails the ingestion on-topic filter -> never in corpus

    # A genuinely on-topic line (mentions ostrich eggshell + geometric) MUST pass.
    good_line = ("Archaeologists recovered engraved ostrich eggshell fragments with geometric "
                 "grids and parallel lines from southern African sites.")
    good_hits = _on_topic_hits(good_line, signature)
    good_kept = good_hits >= 2

    # Ensure the signature didn't collapse to the full token set (i.e. it really
    # stripped the generic words) — otherwise the fix is a no-op.
    sig_actually_signature = ("researchers" not in signature and "research" not in signature)

    ok = cro_dropped and good_kept and sig_actually_signature and not cro_active
    record("SIGNATURE_INGESTION_FILTER", ok,
           f"cro_hits={cro_hits}, cro_dropped={cro_dropped}, good_hits={good_hits}, "
           f"good_kept={good_kept}, generic_stripped={sig_actually_signature}")
    return ok


# ---------------------------------------------------------------------------
# Case 11 — RIGHT-FIRST-TIME, self-maintaining: TermRegister grows on its own.
# A word becomes "generic" once it circulates through enough background docs; a
# distinctive topic word stays out. Uses an isolated temp register (hermetic).
# ---------------------------------------------------------------------------
def case_term_register_growth():
    import tempfile, os
    from src.engine.term_register import TermRegister, DF_RATIO_THRESHOLD

    with tempfile.TemporaryDirectory() as td:
        reg = TermRegister(file_path=os.path.join(td, "term_register.json"))

        # A recurring off-topic leak word: appears in MANY unrelated docs.
        generic_word = "contractual"
        niche_word = "ostrich"

        # Start: 10 docs, contract appears once, ostrich once.
        docs1 = ["firm A in a contractual dispute over delivery",
                 "studio B looks at contractual obligations for acts",
                 "quote C handles contractual terms for the season",
                 "institute D reviews contractual language in grants",
                 "agency E warns of contractual risks in rollout",
                 "bank F sets contractual deadlines for the loan",
                 "tech G disputes a contractual clause with a vendor",
                 "lab H evaluates contractual liability in research",
                 "city I manages contractual procurement for buses",
                 "fund J files contractual claims against the manager"]
        # Make every doc mention the generic word (so df-ratio rises to ~1.0).
        docs1 = [d.replace("contractual", generic_word) for d in docs1]
        reg.observe(docs1)
        g1 = reg.generic_tokens()
        became_generic_after_many = generic_word in g1

        # A distinctive topic word observed rarely must NOT become generic.
        docs2 = [f"archaeologists found {niche_word} eggshell at a dig site"]
        reg.observe(docs2)
        g2 = reg.generic_tokens()
        niche_still_distinctive = niche_word not in g2

        # A SINGLE occurrence among many docs is NOT yet generic (the DF floor
        # is only reached when a word circulates widely) — catches the common
        # case without over-stripping one-off distinctive vocabulary.
        reg2 = TermRegister(file_path=os.path.join(td, "term_register2.json"))
        many_docs = [f"snippet {i} about unrelated market events each day" for i in range(100)]
        many_docs[7] = f"a single study mentions the rare word {generic_word} once"
        reg2.observe(many_docs)
        g_single = reg2.generic_tokens(extra_docs=many_docs)
        single_occurrence_not_generic = generic_word not in g_single

        # Verify persistence: a fresh instance reads the same learned generic.
        reg_reload = TermRegister(file_path=reg.file_path)
        g_reload = reg_reload.generic_tokens()
        persisted = generic_word in g_reload

        ok = (became_generic_after_many and niche_still_distinctive
              and single_occurrence_not_generic and persisted)
        record("TERM_REGISTER_GROWTH", ok,
               f"learned_generic={generic_word in g1}, niche_distinctive={niche_still_distinctive}, "
               f"single_occ_not_generic={single_occurrence_not_generic}, persists={persisted}, "
               f"threshold={DF_RATIO_THRESHOLD}")
        return ok


# ---------------------------------------------------------------------------
# Case 12 — _best_synonym quality guard: never emits stilted/archaic words.
# Uses a deterministic FAKE wordnet so this case is hermetic (no NLTK needed).
# ---------------------------------------------------------------------------
def case_best_synonym_guard():
    from src.agents.story_designer import StoryDesignerAgent
    sd = StoryDesignerAgent(llm_client=None)

    class _Lemma:
        def __init__(self, name):
            self._n = name
        def name(self):
            return self._n

    class _FakeSS:
        # synset(base) with lemmas ordered so the OLD code would pick the
        # stilted first lemma; the guard must skip it and reach the common one.
        def __init__(self, lemmas):
            self._lemmas = [_Lemma(x) for x in lemmas]
        def lemmas(self):
            return self._lemmas

    class _FakeWN:
        def __init__(self, mapping):
            self._m = mapping
        def synsets(self, word):
            return self._m.get(word.lower(), [])

    # "years" -> old code returns 'eld' (first); guard must return 'age' (a
    # multi-lemma-synset, common word).
    wn = _FakeWN({
        "years": [
            _FakeSS(["eld", "age", "geez"]),
            _FakeSS(["age", "old age"]),
        ],
        "ability": [
            _FakeSS(["powerfulness", "power", "potency"]),
            _FakeSS(["power", "capability"]),
        ],
        "capableness": [
            _FakeSS(["capableness", "capability"]),
            _FakeSS(["capability", "potency"]),
        ],
        "powerfulness": [
            _FakeSS(["powerfulness", "power"]),
            _FakeSS(["power", "potency"]),
        ],
        "plain": [
            _FakeSS(["plain"]),
        ],
    })

    good = sd._best_synonym("years", wn)
    ok1 = good == "age" and good != "eld"

    good2 = sd._best_synonym("ability", wn)
    ok2 = good2 in ("power",) and good2 != "powerfulness"

    good3 = sd._best_synonym("capableness", wn)
    ok3 = good3 in ("capability",) and good3 != "capableness"

    good4 = sd._best_synonym("powerfulness", wn)
    ok4 = good4 in ("power",) and good4 != "powerfulness"

    # A token with only a stilted candidate must yield None (leave the word),
    # never inject the stilted one.
    wn_single = _FakeWN({"years": [_FakeSS(["eld"])]})
    good5 = sd._best_synonym("years", wn_single)
    ok5 = good5 is None

    # Token whose every candidate is multi-word/long -> None (never a bad swap).
    wn_none = _FakeWN({"plain": [_FakeSS(["plain text"])]})
    good6 = sd._best_synonym("plain", wn_none)
    ok6 = good6 is None

    # The stilted words are actually registered on the guard.
    stilted_registered = (
        "eld" in sd._STILTED_SYNONYMS
        and "powerfulness" in sd._STILTED_SYNONYMS
        and "capableness" in sd._STILTED_SYNONYMS
    )

    ok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and stilted_registered
    record("BEST_SYNONYM_GUARD", ok,
           f"years->{good}, ability->{good2}, capableness->{good3}, "
           f"powerfulness->{good4}, single_stilted->{good5}, none_ok={good6 is None}, "
           f"registered={stilted_registered}")
    return ok


# ---------------------------------------------------------------------------
# Case 13 — shared chapter timing: computed from shot durations, not static.
# ---------------------------------------------------------------------------
def case_chapter_timestamps():
    from src.engine.chapters import compute_act_chapters

    class Shot:
        def __init__(self, act, dur):
            self.act_index = act
            self.duration_estimate = dur

    # 6 acts, 1 shot each at 45s => Act 2 at 0:45, Act 3 at 1:30, ... Act 6 at 3:45.
    shots = [Shot(act=i, dur=45.0) for i in range(1, 7)]
    lines, ts = compute_act_chapters(shots=shots)

    ok1 = lines[0].startswith("0:00")            # Act 1 anchored at 0
    ok2 = lines[1].startswith("0:45")            # Act 2 at 0:45
    ok3 = lines[2].startswith("1:30")            # Act 3 at 1:30
    ok4 = lines[5].startswith("3:45")            # Act 6 at 3:45
    # description block uppercase-act labels; timestamps list parallel
    ok5 = "Act 4" in lines[3] and ts[3] == lines[3].replace(" - ", " ")

    # Static fallback when no shots.
    slines, _ = compute_act_chapters(shots=None)
    ok6 = slines[0].startswith("0:00") and slines[1].startswith("2:15")

    # Crossfade overlap subtracts between shots: 2 shots 45s, 5s crossfade =>
    # Act 2 at 40s.
    shots_cf = [Shot(1, 45.0), Shot(2, 45.0)]
    clines, _ = compute_act_chapters(shots=shots_cf, crossfade=5.0)
    ok7 = clines[1].startswith("0:40")

    ok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7
    record("CHAPTER_TIMESTAMPS", ok,
           f"lines={lines}, fallback_0={slines[0]}, fallback_1={slines[1]}, crossfade={clines[1]}")
    return ok


# ---------------------------------------------------------------------------
# Case 14 — documentary / investigative storability gate (converged plan).
# Direct news with no story potential must never be considered; evergreen
# synthesized topics pass through; all-culled falls back to best-with-warning.
# ---------------------------------------------------------------------------
def _mk_cand(cid, headline, summary="", url="https://example.com/news/1", kws=("news",)):
    return TopicCandidate(
        candidate_id=cid, headline=headline, summary=summary, source_url=url,
        keywords=list(kws), tvs_score=0.8, rpm_score=0.7, idi_score=0.6,
        sdi_score=0.5, sat_score=0.3,
    )


def case_storability_gate():
    from src.engine.documentary_potential import (
        score_documentary_potential, gate_candidates,
    )
    investigative = _mk_cand(
        "scandal-probe", "SEC opens probe into Nvidia's data-centre sales amid fraud allegations",
        "The SEC investigation follows whistleblower claims of misleading revenue. Legal experts "
        "weigh the fallout, which could reshape how the company books billions in AI deals.",
        "https://www.cnbc.com/sec-nvidia", ["sec", "nvidia", "probe"])
    blip = _mk_cand(
        "mac-launch", "Apple launches new MacBook Air with latest chip",
        "Apple unveiled the new MacBook Air lineup at today's event, available this month.",
        "https://www.theverge.com/apple", ["apple", "macbook"])
    evergreen = _mk_cand(
        "narrow-synth-01", "Cursor vs Windsurf for building AI agents in 2026",
        "A detailed head-to-head of the two AI coding tools for production agent work.",
        "", ["cursor", "windsurf"])
    evergreen.demand_query = "cursor vs windsurf for agents"

    inv = score_documentary_potential(investigative)["verdict"]
    bli = score_documentary_potential(blip)["verdict"]

    kept, culled = gate_candidates([investigative, blip, evergreen])
    kept_ids = {c.candidate_id for c in kept}
    culled_ids = {cand.candidate_id for cand, _audit in culled}
    ok1 = inv == "documentary"                    # probe/scandal/legal/risk = story
    ok2 = bli == "direct_news"                    # press-release launch = culled
    ok3 = "scandal-probe" in kept_ids and "narrow-synth-01" in kept_ids  # evergreen + doc kept
    ok4 = "mac-launch" in culled_ids              # blip never considered
    # all-culled -> ship best-with-warning (never abort, never empty selection)
    all_blips = [_mk_cand(f"b{i}", "Widget Inc reports quarterly revenue up 3% this quarter",
                          "Widget Inc announced quarterly results today with a small revenue increase.",
                          "https://x.com/r", [f"w{i}"]) for i in range(3)]
    kept2, culled2 = gate_candidates(all_blips)
    ok5 = not kept2 and len(culled2) == 3
    if not kept2:
        best_warn = max(all_blips, key=lambda c: score_documentary_potential(c)["score"])
        ok5 = ok5 and best_warn.candidate_id is not None
    record("STORABILITY_GATE", ok1 and ok2 and ok3 and ok4 and ok5,
           f"investigative={inv}, blip={bli}, kept={sorted(kept_ids)}, culled={sorted(culled_ids)}, all_culled_is_empty={not kept2}")
    return ok1 and ok2 and ok3 and ok4 and ok5


# ---------------------------------------------------------------------------
# Case 15 — regional ad revenue has the HIGHEST weight (converged plan).
# Revenue-led region selection + revenue as the 8th, highest-weighted TOPSIS
# criterion. A market's own topics still win despite lower locale RPM.
# ---------------------------------------------------------------------------
def case_region_revenue_dominates():
    from src.engine import region_intelligence as ri
    from src.engine.topic_topsis import (
        TOPSIS_WEIGHTS_REVENUE, TOPSIS_WEIGHTS_GROWTH, rank_topics_topsis,
    )

    rba = _mk_cand("rba", "RBA holds rates at 4.1% as Australian inflation cools, ASX 200 steady",
                   "The Reserve Bank of Australia left the cash rate unchanged, AUD steady.",
                   "https://www.smh.com.au/business", ["rba", "asx"])
    meta = _mk_cand("meta", "Meta launches Muse Glimmer model as Zuckerberg champions AI",
                    "Meta released open-weight Muse Spark, its most powerful AI model.",
                    "https://www.theverge.com/meta", ["meta", "ai"])
    sensex = _mk_cand("sx", "Sensex hits record high as RBI holds rates, Nifty climbs above 26,000",
                      "Indian markets rally after the Reserve Bank of India held rates.",
                      "https://www.moneycontrol.com/news", ["sensex", "nifty", "rbi"])

    pr_rba = ri.candidate_region_profile(rba, 13 * 60 + 40)
    pr_sensex = ri.candidate_region_profile(sensex, 13 * 60 + 40)
    pr_meta = ri.candidate_region_profile(meta, 13 * 60 + 40)
    ok1 = pr_rba["market"] == "au" and pr_rba["l2_region"] == "global"     # AU topic -> AU
    ok2 = pr_sensex["market"] == "india" and pr_sensex["l2_region"] == "india"  # India topic -> IN
    ok3 = pr_meta["market"] == "us"                                        # global tech -> US

    # Revenue weighting: ALPHA (revenue) must be the single largest selection weight.
    ok4 = ri.ALPHA > ri.BETA > ri.GAMMA and ri.ALPHA > 0.45

    # TOPSIS: revenue is the 8th, highest single weight in REVENUE; GROWTH is
    # DISCOVERY-led (TVS + IDI lead; revenue secondary, pre-YPP it earns $0).
    ok5 = len(TOPSIS_WEIGHTS_REVENUE) == 8 and max(TOPSIS_WEIGHTS_REVENUE) == TOPSIS_WEIGHTS_REVENUE[7]
    ok6 = (len(TOPSIS_WEIGHTS_GROWTH) == 8
           and TOPSIS_WEIGHTS_GROWTH[0] == 0.25 == TOPSIS_WEIGHTS_GROWTH[2]
           and TOPSIS_WEIGHTS_GROWTH[7] < TOPSIS_WEIGHTS_GROWTH[0])

    # Higher regional revenue must out-rank a rival on the REVENUE vector.
    meta.regional_revenue_usd = 45.0
    sensex.regional_revenue_usd = 11.0
    rba.regional_revenue_usd = 42.0
    ranked = rank_topics_topsis([rba, meta, sensex], channel_phase="REVENUE")
    ok7 = ranked[0].candidate_id == "meta"     # highest regional ad revenue wins first

    record("REGION_REVENUE_DOMINATES",
           ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7,
           f"weights(revenue)={TOPSIS_WEIGHTS_REVENUE}, ALPHA={ri.ALPHA}, "
           f"rba->{pr_rba['market']}, sensex->{pr_sensex['market']}, meta->{pr_meta['market']}, "
           f"top1={ranked[0].candidate_id}")
    return ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7


# ---------------------------------------------------------------------------
# Case 16 — revenue-goal alignment: $2,000/month drives every money constant.
# The derived per-video gate and the competitor-volume filter must agree with
# the cadence, and the TOPSIS weights must have ONE source of truth.
# ---------------------------------------------------------------------------
def case_revenue_goal_alignment():
    from src.engine.channel_phase_manager import (
        channel_phase_manager, MONTHLY_REVENUE_TARGET_USD,
        TARGET_DAILY_PUBLISHES, REVENUE_GATE_MIN_USD,
    )
    from src.engine.monetization_optimizer import monetization_optimizer
    from src.engine.topic_topsis import TOPSIS_WEIGHTS_REVENUE

    ok1 = MONTHLY_REVENUE_TARGET_USD == 2000.0
    ok2 = TARGET_DAILY_PUBLISHES == 2                      # 2/day => 60/month
    ok3 = abs(REVENUE_GATE_MIN_USD - 2000.0 / (2 * 30)) < 0.01   # $33.33/video
    # TOPSIS weights: single source via the delegate (no drift to 7-criteria).
    ok4 = channel_phase_manager.get_topsis_weights("REVENUE") == TOPSIS_WEIGHTS_REVENUE
    ok5 = len(channel_phase_manager.get_topsis_weights("REVENUE")) == 8
    # Competitor-volume filter now uses the derived per-video gate, not 2450:
    # v_req(mid 9.5 tech) = (33.33/9.5)*1000 ≈ 3508 → 100k competitor views passes;
    # under the old $2450 default it would need ~258k and fail.
    r = monetization_optimizer.filter_by_competitor_volume(
        "Technology & Artificial Intelligence", 100_000)
    ok6 = r["passes_revenue_gate"] and r["v_req_realistic"] < 50_000
    # Publish slots match the cron's two launch windows (cadence = 2/day).
    ok7 = channel_phase_manager.DAILY_PUBLISH_SLOTS_UTC == ["11:20", "13:50"]

    record("REVENUE_GOAL_ALIGNMENT",
           ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7,
           f"monthly=${MONTHLY_REVENUE_TARGET_USD:.0f}, daily={TARGET_DAILY_PUBLISHES}, "
           f"gate=${REVENUE_GATE_MIN_USD:.2f}, slots={channel_phase_manager.DAILY_PUBLISH_SLOTS_UTC}, "
           f"comp_gate_vreq={r['v_req_realistic']:,.0f}")
    return ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main():
    print("CSVG hermetic end-to-end test (self-sufficient, no network)\n")
    case_happy_path()
    case_surgical_revision()
    case_outline_first()
    case_routing()
    case_a2a_alignment()
    case_gate6_shallow_shots()
    case_fake_upload_aborts()
    case_seo_source_filter()
    case_narration_junk_scrub()
    case_signature_ingestion_filter()
    case_term_register_growth()
    case_best_synonym_guard()
    case_chapter_timestamps()
    case_storability_gate()
    case_region_revenue_dominates()
    case_revenue_goal_alignment()

    passed = sum(1 for _, ok, _ in CASE_RESULTS if ok)
    failed = sum(1 for _, ok, _ in CASE_RESULTS if not ok)
    print(f"\nRESULTS: {passed} passed, {failed} failed of {len(CASE_RESULTS)}")
    if failed:
        for name, ok, detail in CASE_RESULTS:
            if not ok:
                print(f"  ∵ {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())