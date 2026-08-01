import asyncio
from src.agents.orchestrator import OrchestratorAgent


def test_full_pipeline_end_to_end():
    orchestrator = OrchestratorAgent()
    state = asyncio.run(orchestrator.run_pipeline(pipeline_id="test-e2e-001", use_live_rss=False))

    assert state.selected_topic is not None
    assert state.script_data is not None
    assert len(state.script_data.shots) == 15
    assert state.asset_paths.final_video is not None
    assert state.upload_metadata.status == "PUBLISHED"
    assert state.execution_stage == "PUBLISHED_SUCCESS"
