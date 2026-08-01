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

    def test_phase3_mcp_media_producer(self):
        test_phase3_media_producer_mcp_workflow()
        print("✓ MediaProducerAgent Phase 3 Edge & Cloud MCP Workflow Test Passed")

    def test_logger_tracer_diagnostics(self):
        test_logger_and_tracer_diagnostics()
        print("✓ PipelineLogger & PipelineTracer Diagnostics Test Passed")

    def test_full_pipeline(self):
        test_full_pipeline_end_to_end()
        print("✓ Full End-to-End Orchestrator Pipeline Execution Test Passed")


if __name__ == "__main__":
    unittest.main()
