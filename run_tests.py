import unittest
from tests.test_topsis_math import test_topsis_ranking_correctness
from tests.test_trend_velocity import (
    test_ema_trend_velocity,
    test_zscore_anomaly_detection,
    test_rpm_cosine_similarity,
    test_semantic_novelty_index
)
from tests.test_script_designer import (
    test_story_designer_script_generation,
    test_observer_approval,
    test_anti_hallucination_audit_rejection,
    test_temporal_anchor_rejection
)
from tests.test_phase3_mcp import test_phase3_media_producer_mcp_workflow
from tests.test_end_to_end import test_full_pipeline_end_to_end
from tests.test_logger_tracer import test_logger_and_tracer_diagnostics
from tests.test_external_apis import test_world_bank_gdp_inflation_api, test_alpha_vantage_fallback_quote
from tests.test_space_cinema_apis import test_nasa_image_library_search, test_tmdb_movie_fallback_data, test_wikipedia_on_this_day_history
from tests.test_gif_retriever import test_giphy_reaction_search
from tests.test_media_sync import (
    test_scene_cue_parsing,
    test_visual_prompt_enrichment,
    test_subtitle_merge_offsets,
    test_subtitle_merge_crossfade,
    test_probe_wav_duration,
    test_market_symbol_picker,
    test_market_quote_parse,
    test_tts_text_sanitize,
    test_ass_to_srt,
    test_producer_renderer_params,
    test_subtitle_merge_empty_edge,
)
from tests.test_stage_quality import (
    test_gate7_passes_good,
    test_gate7_black_frames,
    test_gate7_mute_audio,
    test_gate7_duration_drift,
    test_gate7_crossfade_range,
    test_gate7_dark_but_valid_no_false_positive,
    test_gate7_partial_black_tolerated,
    test_gate7_empty_assets_edge,
    test_gate4_resolution,
    test_crossfade_filtergraph_math,
    test_crossfade_single_clip_edge,
)
from tests.test_e2e_mocked_pipeline import test_full_pipeline_mocked_e2e
from src.agents.fact_retriever import FactRetrieverAgent
from src.agents.story_designer import StoryDesignerAgent
from src.agents.observer import ObserverAgent
from src.agents.media_producer import MediaProducerAgent
from src.agents.orchestrator import OrchestratorAgent
from src.schemas.state import GlobalState


class TestCSVGPipeline(unittest.TestCase):

    def test_topsis(self):
        test_topsis_ranking_correctness()
        print("✓ TOPSIS Math Ranking Test Passed")

    def test_ema(self):
        test_ema_trend_velocity()
        print("✓ EMA Trend Velocity Test Passed")

    def test_zscore(self):
        test_zscore_anomaly_detection()
        print("✓ Z-Score Anomaly Detection Test Passed")

    def test_rpm_similarity(self):
        test_rpm_cosine_similarity()
        print("✓ RPM Cosine Similarity Test Passed")

    def test_novelty(self):
        test_semantic_novelty_index()
        print("✓ Semantic Novelty Index Test Passed")

    def test_fact_retriever_agent(self):
        agent = FactRetrieverAgent()
        state = GlobalState(pipeline_id="csvg-test-001", timestamp="2026-08-02T00:00:00Z")
        msg = agent.process(state)
        self.assertIsNotNone(state.selected_topic)
        self.assertEqual(state.execution_stage, "TOPIC_SELECTED")
        print(f"✓ FactRetrieverAgent Test Passed: Selected '{state.selected_topic.headline}' (TOPSIS Score: {state.selected_topic.topsis_score})")

    def test_phase2_script_designer(self):
        test_story_designer_script_generation()
        print("✓ StoryDesignerAgent 6-Act 10-15 Min Script Test Passed")

    def test_phase2_observer_approval(self):
        test_observer_approval()
        print("✓ ObserverAgent Script Validation & Approval Test Passed")

    def test_phase2_anti_hallucination_audit(self):
        test_anti_hallucination_audit_rejection()
        print("✓ ObserverAgent Anti-Hallucination & Fact Audit Test Passed")

    def test_phase2_temporal_anchor_rejection(self):
        test_temporal_anchor_rejection()
        print("✓ ObserverAgent Temporal Anchor Audit Test Passed")

    def test_external_api_manager(self):
        test_world_bank_gdp_inflation_api()
        test_alpha_vantage_fallback_quote()
        print("✓ ExternalAPIManager World Bank & Stock Quote Test Passed")

    def test_space_cinema_history_apis(self):
        test_nasa_image_library_search()
        test_tmdb_movie_fallback_data()
        test_wikipedia_on_this_day_history()
        print("✓ SpaceCinemaHistoryAPIManager NASA, TMDB & Wikipedia History Test Passed")

    def test_gif_retriever(self):
        test_giphy_reaction_search()
        print("✓ GIFMediaRetriever GIPHY & Tenor Reaction Media Test Passed")

    def test_phase3_mcp_media_producer(self):
        test_phase3_media_producer_mcp_workflow()
        print("✓ MediaProducerAgent Phase 3 Edge & Cloud MCP Workflow Test Passed")

    def test_logger_tracer_diagnostics(self):
        test_logger_and_tracer_diagnostics()
        print("✓ PipelineLogger & PipelineTracer Diagnostics Test Passed")

    def test_full_pipeline(self):
        test_full_pipeline_end_to_end()
        print("✓ Full End-to-End Orchestrator Pipeline Execution Test Passed")

    def test_media_sync_scenarios(self):
        test_scene_cue_parsing()
        test_visual_prompt_enrichment()
        test_subtitle_merge_offsets()
        test_subtitle_merge_crossfade()
        test_probe_wav_duration()
        print("✓ MediaProducer sync helpers (scene cues, prompts, subtitle offsets)")

    def test_media_dynamic_data(self):
        test_market_symbol_picker()
        test_market_quote_parse()
        print("✓ MediaProducer dynamic market data (symbol picker + quote parsing)")

    def test_media_text_and_wiring(self):
        test_tts_text_sanitize()
        test_ass_to_srt()
        test_producer_renderer_params()
        test_subtitle_merge_empty_edge()
        print("✓ MediaProducer TTS sanitization, ass->srt, renderer/crossfade wiring, empty-merge edge")

    def test_gate7_render_integrity(self):
        test_gate7_passes_good()
        test_gate7_black_frames()
        test_gate7_mute_audio()
        print("✓ Gate 7 render integrity (pass / black frames / mute audio)")

    def test_gate7_duration_and_crossfade(self):
        test_gate7_duration_drift()
        test_gate7_crossfade_range()
        test_crossfade_filtergraph_math()
        test_crossfade_single_clip_edge()
        print("✓ Gate 7 duration drift / crossfade range + filtergraph + single-clip edge")

    def test_gate7_false_positive_and_edge(self):
        test_gate7_dark_but_valid_no_false_positive()
        test_gate7_partial_black_tolerated()
        test_gate7_empty_assets_edge()
        print("✓ Gate 7 false-positive protection + edge cases")

    def test_gate4_resolution(self):
        test_gate4_resolution()
        print("✓ Gate 4 resolution + measured-duration skip")

    def test_full_pipeline_mocked_e2e(self):
        test_full_pipeline_mocked_e2e()
        print("✓ Full E2E mocked pipeline (real Pi TTS, mocked upstream data + upload)")


if __name__ == "__main__":
    unittest.main()
