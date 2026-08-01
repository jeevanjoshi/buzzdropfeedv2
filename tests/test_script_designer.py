from src.schemas.state import TopicCandidate, GlobalState, VerifiedFact, ScriptData, ShotData
from src.agents.story_designer import StoryDesignerAgent
from src.agents.observer import ObserverAgent


def test_story_designer_script_generation():
    topic = TopicCandidate(
        candidate_id="test-001",
        headline="How Intel Lost the Microchip Monopoly to TSMC and Nvidia",
        summary="Inside the strategic missteps and corporate downfall that shifted a $1 trillion empire.",
        source_url="https://example.com/test",
        keywords=["intel", "tsmc", "chips"],
        tvs_score=85.0,
        rpm_score=0.90,
        idi_score=0.95,
        sdi_score=1.2,
        sat_score=1.0
    )

    facts = [
        VerifiedFact(
            source_id="f1",
            headline="Intel Market Cap Drops",
            summary="Intel lost market dominance to TSMC and Nvidia in 2026.",
            url="https://example.com/f1"
        )
    ]

    designer = StoryDesignerAgent()
    script = designer.generate_6act_script(topic, facts)

    assert len(script.shots) == 15
    assert script.estimated_runtime_seconds >= 600.0  # >= 10 minutes
    assert script.estimated_runtime_seconds <= 900.0  # <= 15 minutes
    
    # Check 6-Act coverage
    acts_found = set([shot.act_index for shot in script.shots])
    assert acts_found == {1, 2, 3, 4, 5, 6}


def test_observer_approval():
    topic = TopicCandidate(
        candidate_id="test-002",
        headline="Sensex, Nifty Volatility Shifts Indian Markets",
        summary="SBI, Tata Motors, and Hind Zinc lead active volume.",
        source_url="https://example.com/test2",
        keywords=["sensex", "nifty", "markets"],
        tvs_score=80.0,
        rpm_score=0.85,
        idi_score=0.90,
        sdi_score=1.1,
        sat_score=1.0
    )

    facts = [
        VerifiedFact(
            source_id="f2",
            headline="Sensex Nifty Volatility",
            summary="SBI and Tata Motors lead trading volume.",
            url="https://example.com/f2"
        )
    ]

    state = GlobalState(
        pipeline_id="p2-test-01",
        timestamp="2026-08-02T00:00:00Z",
        selected_topic=topic,
        verified_facts=facts
    )
    designer = StoryDesignerAgent()
    observer = ObserverAgent()

    msg_design = designer.process(state)
    assert state.script_data is not None

    msg_obs = observer.process(state)
    assert msg_obs.intent.value == "APPROVE_SCRIPT"
    assert state.execution_stage == "SCRIPT_APPROVED"


def test_anti_hallucination_audit_rejection():
    topic = TopicCandidate(
        candidate_id="test-003",
        headline="Market Shift Story",
        summary="Verified summary text.",
        source_url="https://example.com/t3",
        keywords=["market"],
        tvs_score=50.0, rpm_score=0.5, idi_score=0.5, sdi_score=0.5, sat_score=1.0
    )

    facts = [
        VerifiedFact(
            source_id="f3",
            headline="Market Story",
            summary="Verified facts contain $50B market shift.",
            url="https://example.com/f3"
        )
    ]

    # Create a script fixture containing an invented/hallucinated number 999.8b
    bad_shot = ShotData(
        shot_id=1, act_index=1,
        narration_text="Invented story claims company lost 999.8b in a single second.",
        visual_prompt="Cinematic 16:9 widescreen dark lighting, 8k resolution.",
        duration_estimate=45.0
    )
    bad_script = ScriptData(title="Bad Script", target_shots=12, shots=[bad_shot]*12, estimated_runtime_seconds=650.0)

    observer = ObserverAgent()
    is_approved, violations = observer.evaluate_script(bad_script, facts)

    assert is_approved is False
    assert any("unverified numerical claim" in v.lower() for v in violations)
