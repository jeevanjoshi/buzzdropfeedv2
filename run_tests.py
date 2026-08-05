import unittest
from tests.test_topsis_math import test_topsis_ranking_correctness
from tests.test_trend_velocity import (
    test_ema_trend_velocity,
    test_zscore_anomaly_detection,
    test_rpm_cosine_similarity,
    test_semantic_novelty_index
)
from tests.test_logger_tracer import test_logger_and_tracer_diagnostics
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

    def test_logger_tracer_diagnostics(self):
        test_logger_and_tracer_diagnostics()
        print("✓ PipelineLogger & PipelineTracer Diagnostics Test Passed")

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

    def test_observer_fixes(self):
        from tests.test_observer_fixes import (
            test_keyword_repetition_excludes_topic_state,
            test_keyword_repetition_still_flags_real_slop,
            test_source_diversity_lenient_for_tiny_cites,
            test_source_diversity_enforced_at_scale,
            test_topic_entities_from_summary_excluded,
            test_revision_loop_cap_structure,
        )
        test_keyword_repetition_excludes_topic_state()
        test_keyword_repetition_still_flags_real_slop()
        test_source_diversity_lenient_for_tiny_cites()
        test_source_diversity_enforced_at_scale()
        test_topic_entities_from_summary_excluded()
        test_revision_loop_cap_structure()
        print("✓ Observer fixes (state/entity exclusion, source-diversity leniency)")

    def test_promo_filter(self):
        from tests.test_promo_filter import (
            test_blocks_top_n_listicle,
            test_blocks_how_i_make_style,
            test_blocks_review_affiliate,
            test_blocks_advertorial_growth_hype,
            test_blocks_direct_cta,
            test_allows_real_news,
        )
        test_blocks_top_n_listicle()
        test_blocks_how_i_make_style()
        test_blocks_review_affiliate()
        test_blocks_advertorial_growth_hype()
        test_blocks_direct_cta()
        test_allows_real_news()
        print("✓ Promo/listicle/affiliate content filter")

    def test_full_pipeline_mocked_e2e(self):
        test_full_pipeline_mocked_e2e()
        print("✓ Full E2E mocked pipeline (real Pi TTS, mocked upstream data + upload)")


if __name__ == "__main__":
    unittest.main()
