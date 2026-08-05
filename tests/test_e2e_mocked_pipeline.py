"""
Full end-to-end pipeline test with MOCKED data/media, using the REAL
Raspberry Pi 5 for TTS + Whisper subtitle alignment (via AUDIO_EDGE_URL).

Data flow simulated end-to-end:
  FactRetriever (offline canned candidates/facts)
  -> StoryDesigner (mocked 15-shot script + SEO metadata, no LLM/network)
  -> Observer (mocked approval)
  -> MediaProducer (REAL TTS + Whisper on Pi, REAL ffmpeg Ken Burns/concat,
                    synthetic visuals, dynamic market numbers mocked)
  -> Quality gates (REAL Gate 4 + Gate 7 run on the produced media)
  -> Publisher (mocked YouTube quota check + upload)

No external APIs are called: LLM, RSS, fal/Replicate, GIPHY, Playwright,
YouTube upload and market quote are all mocked/fallback. Only the Pi TTS
service is real (falls back to local synthetic WAV if the Pi is unreachable).
Run standalone:  python tests/test_e2e_mocked_pipeline.py
"""
import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schemas.state import (
    GlobalState, ScriptData, ShotData, SEOMetadata, VisualType, TopicCandidate,
)
from src.agents.orchestrator import OrchestratorAgent


def _canned_script() -> ScriptData:
    """Builds a synthetic 15-shot 6-Act script mixing every visual type.
    Narration is kept short (~8-10 words) so the produced master is ~1 minute."""
    types = [VisualType.STANDARD_IMAGE] * 15
    types[2] = VisualType.MATPLOTLIB_CHART   # shot 3: dynamic chart
    types[6] = VisualType.SVG_TICKER          # shot 7: ticker
    types[8] = VisualType.GIF_MEME            # shot 9: reaction gif
    shots = [
        ShotData(
            shot_id=i,
            act_index=((i - 1) % 6) + 1,
            narration_text=f"Synthetic market shift {i}. Watch the numbers move now.",
            visual_prompt="Cinematic 16:9 widescreen synthetic market scene with dynamic lighting.",
            visual_type=types[i - 1],
            duration_estimate=4.0,
        )
        for i in range(1, 16)
    ]
    return ScriptData(
        title="Synthetic E2E Market Shift",
        target_shots=15,
        shots=shots,
        estimated_runtime_seconds=60.0,
    )


def _canned_seo() -> SEOMetadata:
    return SEOMetadata(
        title="Synthetic E2E Market Shift 2026",
        description=(
            "Deep-dive synthetic storytelling test. This description is long enough "
            "to satisfy the publisher metadata path without any external API calls."
        ) * 3,
        tags=["synthetic", "e2e", "market", "test", "finance"],
        thumbnail_brief="MARKET SHIFT 2026",
        chapter_timestamps=["0:00 Intro", "1:00 The Shift"],
    )


def _apply_mocks():
    """Install all mocks BEFORE run_pipeline executes."""
    # --- StoryDesigner: no LLM / no RAG network ---
    import src.agents.story_designer as sd
    sd.StoryDesignerAgent.generate_6act_script = (
        lambda self, topic, facts, region="all", revision_violations=None, state=None: _canned_script()
    )
    sd.StoryDesignerAgent._polish_script = lambda self, script, headline, niche: None
    sd.StoryDesignerAgent.generate_seo_metadata = lambda self, topic, script: _canned_seo()

    # --- Observer: deterministic approval (no LLM critic) ---
    import src.agents.observer as ob
    ob.ObserverAgent.evaluate_script = (
        lambda self, script, facts, topic=None, channel_phase=None, crawled_content=None: (True, [])
    )
    import src.engine.llm_client as llmc
    llmc.LLMClient.is_available = lambda self: False

    # --- Quality verifier: keep Gate 4 + Gate 7 REAL; mock text/LLM gates ---
    import src.engine.quality_verifier as qv
    qv.StageQualityVerifier.verify_gate1_topic_to_script = lambda self, topic, script: (True, [])
    qv.StageQualityVerifier.verify_gate6_anti_slop_entropy = (
        lambda self, script: {"passes": True, "issues": []}
    )
    qv.StageQualityVerifier.verify_gate3b_subtitle_text_coherence = lambda self, state: (True, [])

    # --- Market quote for dynamic chart/ticker numbers (no Alpha Vantage API) ---
    import src.engine.external_apis as ea
    ea.ExternalAPIManager.fetch_alpha_vantage_stock_quote = (
        lambda self, symbol: {"symbol": symbol, "price": "$650.00", "change": "+3.50%"}
    )

    # --- Thumbnail: no fal/Replicate network ---
    import mcp_servers.media_cloud.server as mc
    def _fake_thumbnail(req):
        d = os.path.dirname(os.path.abspath(req.output_thumbnail_path)) or "."
        os.makedirs(d, exist_ok=True)
        with open(req.output_thumbnail_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
        return {"status": "success", "path": req.output_thumbnail_path}
    mc.generate_thumbnail = _fake_thumbnail

    # --- GIF retriever: force the fast synthetic fallback (no GIPHY network hang) ---
    import src.engine.gif_retriever as gr
    def _no_giphy(*a, **k):
        raise RuntimeError("mocked: GIPHY disabled for hermetic e2e")
    gr.gif_retriever.search_giphy_reaction = _no_giphy

    # --- Publisher: mock YouTube quota + upload (no real upload) ---
    import src.agents.publisher as pub
    async def _fake_quota(req):
        return {"is_safe": True, "used_units": 0, "quota_limit": 10000}
    async def _fake_upload(req):
        return {"video_id": "e2e-mocked-001", "status": "success"}
    pub.check_quota_available = _fake_quota
    pub.upload_youtube_resumable = _fake_upload


def test_full_pipeline_mocked_e2e() -> GlobalState:
    # Opt-in: this hits the real Raspberry Pi TTS (slow) and can hang in CI.
    # Run with:  CSVG_E2E_PI=1 python tests/test_e2e_mocked_pipeline.py
    if os.environ.get("CSVG_E2E_PI") != "1":
        print("SKIP test_full_pipeline_mocked_e2e (set CSVG_E2E_PI=1 to run against the Pi TTS)")
        return None
    _apply_mocks()
    orchestrator = OrchestratorAgent()
    state = asyncio.run(orchestrator.run_pipeline(
        pipeline_id="e2e-mocked",
        use_live_rss=False,      # canned offline candidates (no RSS network)
        region="global",
        publish=True,            # exercises the (mocked) publisher path
        dummy_frames=True,       # synthetic visuals (no fal/Replicate network)
        renderer="ffmpeg",
        crossfade=0.5,
    ))

    # Stage 1: topic selected
    assert state.selected_topic is not None
    assert state.execution_stage in ("PUBLISHED_SUCCESS", "QUALITY_VERIFIED"), state.execution_stage
    # Stage 2: script
    assert state.script_data is not None and len(state.script_data.shots) == 15
    # Stage 3: media produced with real TTS on the Pi (or synthetic fallback)
    assert state.asset_paths.final_video is not None and os.path.exists(state.asset_paths.final_video)
    assert len(state.asset_paths.visuals) == 15
    assert len(state.asset_paths.audio) == 15
    assert len(state.asset_paths.measured_durations) == 15
    # Stage 4: publisher (mocked upload)
    if state.upload_metadata:
        assert state.upload_metadata.status in ("PUBLISHED", "PENDING")
        assert state.upload_metadata.video_id

    print("✓ E2E mocked pipeline passed: topic -> script -> observer -> media (Pi TTS) -> gates -> publisher")
    print(f"  final_video={state.asset_paths.final_video}")
    print(f"  measured durations={[round(d,1) for d in state.asset_paths.measured_durations][:5]}...")
    print(f"  final stage={state.execution_stage} upload={state.upload_metadata.video_id if state.upload_metadata else 'n/a'}")
    return state


if __name__ == "__main__":
    os.environ["CSVG_E2E_PI"] = os.environ.get("CSVG_E2E_PI", "1")  # standalone always runs against Pi
