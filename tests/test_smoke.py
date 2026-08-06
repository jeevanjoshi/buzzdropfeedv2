"""
Single hermetic SMOKE test for the CSVG pipeline.

Runs the full orchestrator end-to-end (topic select -> RAG gate ->
StoryDesigner -> Observer -> MediaProducer -> Publisher) using dependency-
injected STUB agents and controlled monkeypatching of the module singletons
the orchestrator calls directly.

Guarantees (asserted at runtime where possible):
  * NO live/network calls   -> RSS, RAG, LLM, market, visuals, TTS, publisher
                               are all mocked. No sockets/http opened.
  * NO Pi interaction       -> AUDIO_EDGE_URL / LLAMA_CPP_URL forced blank; the
                               media agent is fully stubbed so nothing reaches
                               the Pi edge node.
  * NO media generation     -> the MediaProducer is stubbed to emit synthetic
                               asset paths (no ffmpeg / Ken Burns / TTS / fal).

Covers POSITIVE, NEGATIVE and EDGE cases so every stage branch is exercised:
  POSITIVE : full pipeline reaches PUBLISHED_SUCCESS (publish on)
  EDGE     : publish off -> QUALITY_VERIFIED (till-upload)
  EDGE     : resume from a pre-populated state (skips already-done stages)
  NEGATIVE : no topic selected                  -> RuntimeError
  NEGATIVE : RAG corpus insufficient for all    -> RuntimeError
  NEGATIVE : script generation fails            -> RuntimeError
  NEGATIVE : Observer hard rejection persists    -> RuntimeError
  NEGATIVE : media production fails             -> RuntimeError
  NEGATIVE : quality gate (Gate 3b/4/7) fails    -> RuntimeError

Run standalone (from repo root):
    python tests/test_smoke.py
"""
import os
import sys
import copy
import asyncio
import datetime
import tempfile

# ---------------------------------------------------------------------------
# Hard-block every possible external / Pi channel before importing app modules.
# ---------------------------------------------------------------------------
os.environ["AUDIO_EDGE_URL"] = ""
os.environ["LLAMA_CPP_URL"] = ""
os.environ["PINECONE_API_KEY"] = ""
os.environ["API_NINJAS_KEY"] = ""
os.environ["EXA_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""
os.environ["FIRECRAWL_API_KEY"] = ""
os.environ["NEWSAPI_KEY"] = ""
os.environ["ALPHA_VANTAGE_KEY"] = ""
os.environ["PIXABAY_API_KEY"] = ""
os.environ["PEXELS_API_KEY"] = ""
os.environ["GIPHY_API_KEY"] = ""
os.environ["FAL_KEY"] = ""
os.environ["REPLICATE_API_TOKEN"] = ""
os.environ["USE_SEMANTIC_GATES"] = "0"   # force TF-IDF fallback, no torch model
os.environ["ALLOW_SOFT_APPROVAL"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schemas.state import (
    GlobalState, ScriptData, ShotData, SEOMetadata, VisualType, TopicCandidate,
    VerifiedFact, AssetPaths, ChannelStats, UploadMetadata,
)
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent
from src.agents.orchestrator import OrchestratorAgent


def _msg(sender: AgentRole, target: AgentRole, intent: AgentIntent, payload=None) -> A2AMessage:
    return A2AMessage(
        message_id="m",
        sender=sender,
        target=target,
        intent=intent,
        payload=payload or {},
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def _topic(tid="cand-001") -> TopicCandidate:
    return TopicCandidate(
        candidate_id=tid,
        headline="Nvidia Unveils Next-Gen AI Microchip Architecture Shaking Tech Valuation",
        summary="Nvidia announced groundbreaking GPU architecture driving AI data center efficiency.",
        source_url="https://example.invalid/tech/nvidia-ai-chip",
        keywords=["nvidia", "ai", "chips", "gpu", "tech"],
        tvs_score=92.5, rpm_score=0.95, idi_score=0.88, sdi_score=1.4,
        shm_score=1.8, vph_score=2.2, sat_score=0.8,
        topsis_score=0.9121,
    )


def _fact() -> VerifiedFact:
    return VerifiedFact(
        source_id="fact-101",
        headline="Nvidia Market Cap Surge",
        summary="Nvidia market capitalisation reached record highs following the new architecture announcement.",
        url="https://example.invalid/tech/nvidia",
        source_name="TechCrunch",
    )


def _script() -> ScriptData:
    shots = [
        ShotData(
            shot_id=i,
            act_index=((i - 1) % 6) + 1,
            narration_text=f"Synthetic smoke narration shot {i}. Watch the numbers move now.",
            visual_prompt="Cinematic 16:9 widescreen synthetic market scene, warm lighting.",
            visual_type=VisualType.STANDARD_IMAGE,
            duration_estimate=3.0,
        )
        for i in range(1, 16)
    ]
    return ScriptData(title="Synthetic Smoke Market Shift", target_shots=15,
                      shots=shots, estimated_runtime_seconds=60.0)


def _seo() -> SEOMetadata:
    return SEOMetadata(
        title="Synthetic Smoke Market Shift 2026",
        description=("Deep-dive synthetic smoke description for the publisher path. ") * 4,
        tags=["synthetic", "smoke", "finance"],
        thumbnail_brief="MARKET SHIFT 2026",
        chapter_timestamps=["0:00 Intro", "1:00 The Shift"],
    )


# ---------------------------------------------------------------------------
# Stub agents (dependency-injected into the OrchestratorAgent)
# ---------------------------------------------------------------------------
class StubFactRetriever:
    """Controllable: picks a topic, or simulates 'no topic found'."""
    def __init__(self, ok=True, alt_topics=2):
        self.ok = ok
        self.alt_topics = alt_topics
        self.calls = 0

    def process(self, state, use_live_rss=True, region="all", channel_phase="REVENUE", exclude_headlines=None):
        self.calls += 1
        if not self.ok:
            return _msg(AgentRole.FACT_RETRIEVER, AgentRole.ORCHESTRATOR, AgentIntent.FAILURE_REPORT)
        state.selected_topic = _topic(f"cand-{self.calls}")
        state.verified_facts = [_fact()]
        return _msg(AgentRole.FACT_RETRIEVER, AgentRole.ORCHESTRATOR, AgentIntent.TOPIC_SELECTED)


class StubStoryDesigner:
    """Controllable: creates a script, or simulates script failure."""
    def __init__(self, ok=True):
        self.ok = ok

    def process(self, state, revision_violations=None):
        if not self.ok:
            return _msg(AgentRole.STORY_DESIGNER, AgentRole.ORCHESTRATOR, AgentIntent.FAILURE_REPORT)
        state.script_data = _script()
        state.seo_metadata = _seo()
        return _msg(AgentRole.STORY_DESIGNER, AgentRole.ORCHESTRATOR, AgentIntent.GENERATE_SCRIPT)


class StubObserver:
    """Controllable: approves, or persistently REVISEs (hard reject)."""
    def __init__(self, approve=True):
        self.approve = approve

    def process(self, state):
        intent = AgentIntent.APPROVE_SCRIPT if self.approve else AgentIntent.REVISE_SCRIPT
        payload = {} if self.approve else {"violations": [
            "Shot #2 Fact Audit: The claim lacks verified grounding in source facts."
        ]}
        return _msg(AgentRole.OBSERVER, AgentRole.ORCHESTRATOR, intent, payload)


class StubMediaProducer:
    """No media produced: sets synthetic asset paths only (no ffmpeg/TTS)."""
    def __init__(self, ok=True):
        self.ok = ok

    async def process(self, state, dummy_frames=False, renderer=None):
        if not self.ok:
            return _msg(AgentRole.MEDIA_PRODUCER, AgentRole.ORCHESTRATOR, AgentIntent.FAILURE_REPORT)
        d = tempfile.mkdtemp(prefix="csvg_smoke_")
        final = os.path.join(d, "final_video_1080p.mp4")
        with open(final, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42" + b"0" * 64)
        ap = AssetPaths(
            storage_dir=d,
            final_video=final,
            thumbnail=os.path.join(d, "thumbnail.png"),
        )
        for i in range(1, 16):
            key = f"shot_{i}"
            ap.audio[key] = os.path.join(d, "audio", f"{key}.wav")
            ap.visuals[key] = os.path.join(d, "visuals", f"{key}.mp4")
            ap.measured_durations.append(3.0)
        state.asset_paths = ap
        state.execution_stage = "MEDIA_PRODUCED"
        return _msg(AgentRole.MEDIA_PRODUCER, AgentRole.ORCHESTRATOR, AgentIntent.MEDIA_READY, {"status": "SUCCESS"})


class StubPublisher:
    """Controllable: publishes successfully, or fails."""
    def __init__(self, ok=True):
        self.ok = ok

    async def process(self, state, daily_uploads=0):
        if not self.ok:
            raise RuntimeError("publisher upload failed (mocked)")
        state.upload_metadata = UploadMetadata(
            video_id="smoke-001", status="PUBLISHED", synthetic_content_flag=True)
        state.execution_stage = "PUBLISHED_SUCCESS"
        return _msg(AgentRole.PUBLISHER, AgentRole.ORCHESTRATOR, AgentIntent.PUBLISHED_SUCCESS)


# ---------------------------------------------------------------------------
# Harness: build an orchestrator wired with stubs + controlled gates/RAG.
# ---------------------------------------------------------------------------
def _patch_singletons(gates_pass=True, rag_pass=True):
    """Monkeypatch the module singletons the orchestrator calls directly."""
    import unittest.mock as mock

    # channel phase: fix the singleton the orchestrator uses (no YT API refresh).
    import src.engine.channel_phase_manager as cpm
    cpm.channel_phase_manager.get_channel_stats = (lambda force_refresh=False: ChannelStats(
        subscribers=5000, total_watch_hours=5000,
        channel_phase="REVENUE", ypp_unlocked=True,
        last_updated="2026-01-01T00:00:00+00:00",
    ))

    # RAG: controlled build + assess (no network).
    import src.engine.rag_retriever as rr
    def _fake_build(self, topic, verified_facts, refresh=False):
        return {
            "topic_headline": topic.headline,
            "summary": topic.summary,
            "keywords": topic.keywords,
            "fact_corpus": (
                "Nvidia unveiled a next generation GPU architecture boosting AI data center "
                "efficiency and adding one hundred fifty billion dollars. (Source: TechCrunch)\n"
                "Intel delayed its seven nanometre nodes causing foundry customers to shift "
                "orders to TSMC and Nvidia. (Source: The Wall Street Journal)\n"
            ),
            "full_rag_context_text": "Nvidia GPU architecture AI data center efficiency. (Source: TechCrunch)",
            "ground_truth_block": "Nvidia GPU architecture AI data center efficiency.",
        }
    def _fake_assess(self, pack, topic):
        return {"pass": rag_pass,
                "reason": "RAG corpus sufficient (mocked)." if rag_pass else "undersupplied (mocked)",
                "metrics": {"on_topic_facts": 2, "on_topic_corpus_words": 40,
                            "on_topic_sources": 2, "total_corpus_words": 40}}
    rr.RAGTopicRetriever.build_rag_knowledge_pack = _fake_build
    rr.RAGTopicRetriever.assess_corpus_sufficiency = _fake_assess

    # Quality verifier: gates the orchestrator calls directly.
    import src.engine.quality_verifier as qv
    ok = (True, []) if gates_pass else (False, ["Gate failed (mocked)"])
    qv.StageQualityVerifier.verify_gate1_topic_to_script = lambda self, topic, script: ok
    qv.StageQualityVerifier.verify_gate3b_subtitle_text_coherence = lambda self, state: ok
    qv.StageQualityVerifier.verify_gate4_video_audio_coherence = lambda self, state: ok
    qv.StageQualityVerifier.verify_gate7_render_integrity = lambda self, state: ok
    qv.StageQualityVerifier.verify_gate6_anti_slop_entropy = lambda self, script: (
        {"passes": gates_pass, "issues": [] if gates_pass else ["Anti-slop (mocked)"]}
    )
    # Gate 5 stays real (pure metadata patch) — but ensure it never needs network.
    return mock


def _build_orchestrator(fr=None, sd=None, ob=None, mp=None, pu=None, gates_pass=True, rag_pass=True):
    _patch_singletons(gates_pass=gates_pass, rag_pass=rag_pass)
    return OrchestratorAgent(
        fact_retriever=fr or StubFactRetriever(),
        story_designer=sd or StubStoryDesigner(),
        observer=ob or StubObserver(),
        media_producer=mp or StubMediaProducer(),
        publisher=pu or StubPublisher(),
        logs_dir=tempfile.mkdtemp(prefix="csvg_logs_"),
    )


# ---------------------------------------------------------------------------
# POSITIVE / EDGE cases
# ---------------------------------------------------------------------------
def test_positive_full_pipeline_publishes():
    orch = _build_orchestrator()
    state = asyncio.run(orch.run_pipeline(
        pipeline_id="smoke-pos", use_live_rss=False, region="global",
        publish=True, dummy_frames=True, renderer="ffmpeg", crossfade=0.5,
    ))
    assert state.selected_topic is not None
    assert state.script_data is not None and len(state.script_data.shots) == 15
    assert state.execution_stage == "PUBLISHED_SUCCESS", state.execution_stage
    assert state.upload_metadata and state.upload_metadata.video_id == "smoke-001"
    print("✓ POSITIVE: full pipeline reached PUBLISHED_SUCCESS (all stages wired)")


def test_edge_publish_off_till_upload():
    orch = _build_orchestrator()
    state = asyncio.run(orch.run_pipeline(
        pipeline_id="smoke-edge", use_live_rss=False, region="global",
        publish=False, dummy_frames=True, renderer="ffmpeg", crossfade=0.0,
    ))
    assert state.execution_stage == "QUALITY_VERIFIED", state.execution_stage
    assert state.upload_metadata.video_id is None  # publish skipped
    print("✓ EDGE: publish=False -> QUALITY_VERIFIED (till-upload)")


def test_edge_resume_from_state():
    """Pre-populated state skips the stages already completed."""
    fr = StubFactRetriever(ok=True)
    prev = GlobalState(pipeline_id="smoke-resume", timestamp="x")
    prev.execution_stage = "SCRIPT_APPROVED"
    prev.selected_topic = _topic()
    prev.verified_facts = [_fact()]
    prev.script_data = _script()
    prev.seo_metadata = _seo()
    orch = _build_orchestrator(fr=fr)
    state = asyncio.run(orch.run_pipeline(
        pipeline_id="smoke-resume", use_live_rss=False, region="global",
        publish=False, state=prev, dummy_frames=True, renderer="ffmpeg", crossfade=0.5,
    ))
    assert state.execution_stage == "QUALITY_VERIFIED"
    assert fr.calls == 0  # fact retriever must NOT re-run (resumed after topic)
    print("✓ EDGE: resume pre-populated state skips completed stages")


# ---------------------------------------------------------------------------
# NEGATIVE / error-handling cases
# ---------------------------------------------------------------------------
def test_negative_no_topic_selected():
    orch = _build_orchestrator(fr=StubFactRetriever(ok=False))
    try:
        asyncio.run(orch.run_pipeline(pipeline_id="smoke-neg", use_live_rss=False,
                                      region="global", publish=True, dummy_frames=True))
    except RuntimeError as e:
        assert "No suitable topic" in str(e), e
        print("✓ NEGATIVE: no topic -> abort with clear error")
        return
    raise AssertionError("expected RuntimeError when no topic is selected")


def test_negative_rag_insufficient():
    # RAG always fails -> orchestrator switches topics, then aborts.
    orch = _build_orchestrator(rag_pass=False)
    try:
        asyncio.run(orch.run_pipeline(pipeline_id="smoke-neg-rag", use_live_rss=False,
                                      region="global", publish=True, dummy_frames=True))
    except RuntimeError as e:
        assert "RAG corpus insufficient" in str(e), e
        print("✓ NEGATIVE: RAG insufficient for all topics -> abort")
        return
    raise AssertionError("expected RuntimeError when RAG is undersupplied")


def test_negative_script_generation_fails():
    orch = _build_orchestrator(sd=StubStoryDesigner(ok=False))
    try:
        asyncio.run(orch.run_pipeline(pipeline_id="smoke-neg-sd", use_live_rss=False,
                                      region="global", publish=True, dummy_frames=True))
    except RuntimeError as e:
        assert "Script generation failed" in str(e), e
        print("✓ NEGATIVE: script generation fails -> abort")
        return
    raise AssertionError("expected RuntimeError when script generation fails")


def test_negative_observer_hard_reject_persists():
    # Observer always REVISEs with a hard fact-audit violation + gates pass.
    # No soft-approval rescue because the violation is a hard invariant.
    os.environ["ALLOW_SOFT_APPROVAL"] = "1"
    orch = _build_orchestrator(ob=StubObserver(approve=False), gates_pass=True)
    try:
        asyncio.run(orch.run_pipeline(pipeline_id="smoke-neg-ob", use_live_rss=False,
                                      region="global", publish=True, dummy_frames=True))
    except RuntimeError as e:
        assert "failed validation" in str(e) or "Fact Audit" in str(e), e
        print("✓ NEGATIVE: persistent hard observer rejection -> abort")
        return
    raise AssertionError("expected RuntimeError on persistent hard rejection")


def test_negative_media_production_fails():
    orch = _build_orchestrator(mp=StubMediaProducer(ok=False))
    try:
        asyncio.run(orch.run_pipeline(pipeline_id="smoke-neg-media", use_live_rss=False,
                                      region="global", publish=True, dummy_frames=True))
    except RuntimeError as e:
        assert "final video output path is empty" in str(e).lower() or "Media production failed" in str(e), e
        print("✓ NEGATIVE: media production fails -> abort")
        return
    raise AssertionError("expected RuntimeError when media production fails")


def test_negative_quality_gate_fails():
    orch = _build_orchestrator(gates_pass=False)
    try:
        asyncio.run(orch.run_pipeline(pipeline_id="smoke-neg-gates", use_live_rss=False,
                                      region="global", publish=True, dummy_frames=True))
    except RuntimeError as e:
        print(f"✓ NEGATIVE: quality gate failure -> abort ({str(e)[:60]}...)")
        return
    raise AssertionError("expected RuntimeError when a quality gate fails")


if __name__ == "__main__":
    tests = [
        ("POSITIVE full pipeline publishes", test_positive_full_pipeline_publishes),
        ("EDGE publish off -> QUALITY_VERIFIED", test_edge_publish_off_till_upload),
        ("EDGE resume from state", test_edge_resume_from_state),
        ("NEGATIVE no topic selected", test_negative_no_topic_selected),
        ("NEGATIVE RAG insufficient", test_negative_rag_insufficient),
        ("NEGATIVE script generation fails", test_negative_script_generation_fails),
        ("NEGATIVE observer hard reject", test_negative_observer_hard_reject_persists),
        ("NEGATIVE media production fails", test_negative_media_production_fails),
        ("NEGATIVE quality gate fails", test_negative_quality_gate_fails),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            failures += 1
            print(f"FAIL [{name}]: {type(e).__name__}: {e}")
    total = len(tests)
    print(f"\n{'='*60}")
    print(f"HERMETIC SMOKE SUMMARY: {total - failures}/{total} cases passed. "
          f"(No media, no live calls, no Pi interaction.)")
    sys.exit(1 if failures else 0)
