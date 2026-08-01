import os
import asyncio
from src.schemas.state import GlobalState, TopicCandidate, VerifiedFact
from src.agents.story_designer import StoryDesignerAgent
from src.agents.media_producer import MediaProducerAgent


def test_phase3_media_producer_mcp_workflow():
    topic = TopicCandidate(
        candidate_id="p3-001",
        headline="Phase 3 Test Headline",
        summary="Phase 3 test summary",
        source_url="https://example.com/p3",
        keywords=["test"],
        tvs_score=50.0, rpm_score=0.5, idi_score=0.5, sdi_score=0.5, sat_score=1.0
    )

    facts = [VerifiedFact(source_id="f1", headline="H1", summary="S1", url="http://u1")]
    state = GlobalState(pipeline_id="p3-pipe-01", timestamp="2026-08-02T00:00:00Z", selected_topic=topic, verified_facts=facts)

    designer = StoryDesignerAgent()
    msg_design = designer.process(state)

    producer = MediaProducerAgent(storage_dir="/tmp/csvg_test_media")
    msg_prod = asyncio.run(producer.process(state))

    assert msg_prod.intent.value == "MEDIA_READY"
    assert state.execution_stage == "MEDIA_PRODUCED"
    assert state.asset_paths.final_video is not None
    assert len(state.asset_paths.audio) == 15
    assert len(state.asset_paths.visuals) == 15
