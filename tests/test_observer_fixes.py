"""
Hermetic unit tests for Observer audit fixes:
  1. Topic state/entities (e.g. "jersey" from NJ) are excluded from keyword over-repetition.
  2. Source-diversity over-citation is only enforced with enough citations (>=3).
  3. Bounded revision-loop logic respects MAX_REVISIONS (integration-level helper assertions).
The LLM critic is mocked to keep the suite fully offline.
"""
import unittest.mock as mock

from src.schemas.state import TopicCandidate, VerifiedFact, ScriptData, ShotData
from src.agents.observer import ObserverAgent


def _topic(headline="In Lawsuit, NJ Accuses Amazon of Suppressing Pay for Delivery Drivers"):
    return TopicCandidate(
        candidate_id="t-obs-001",
        headline=headline,
        summary="The state attorney general said in an antitrust lawsuit filed on Tuesday that the company was abusing its market power to keep its delivery costs low.",
        source_url="https://example.com/t",
        keywords=["drivers", "market", "company", "power", "said", "general", "accuses", "tuesday", "lawsuit", "amazon"],
        tvs_score=85.0, rpm_score=0.9, idi_score=0.95, sdi_score=1.2, sat_score=1.0,
    )


def _script(narration):
    return ScriptData(
        title="T", target_shots=len(narration),
        shots=[ShotData(shot_id=i + 1, act_index=1, narration_text=n, visual_prompt="Cinematic 16:9 widescreen dark lighting, 8k resolution.", duration_estimate=45.0) for i, n in enumerate(narration)],
        estimated_runtime_seconds=600.0,
    )


def _fact(name="Entrepreneur"):
    return VerifiedFact(source_id=f"f-{name}", headline=f"{name} Report", summary=f"Reported by {name}.", url="https://example.com/f", source_name=name)


def _evaluate(obs, narration, facts=None, topic=None):
    """Run evaluate_script with the LLM critic force-disabled (hermetic)."""
    with mock.patch("src.engine.llm_client.LLMClient") as llm_cls:
        llm_cls.return_value.is_available.return_value = False
        return obs.evaluate_script(_script(narration), facts or [_fact()], topic=topic or _topic())


def test_keyword_repetition_excludes_topic_state():
    """'jersey' (from NJ in the headline) must NOT be flagged as over-repeated filler."""
    obs = ObserverAgent()
    narration = [
        "New Jersey regulators opened a probe this week into the delivery giant.",
        "New Jersey lawmakers scheduled hearings on Monday.",
        "New Jersey officials declined to comment on the ongoing case.",
        "New Jersey workers described harsh conditions in warehouses.",
        "New Jersey courts heard arguments on the motion to dismiss.",
        "New Jersey drivers filed additional complaints with the state.",
        "New Jersey residents watched the proceedings closely.",
        "New Jersey attorneys joined the case on behalf of workers.",
        "New Jersey groups demanded transparency from the company.",
        "New Jersey agencies shared documents with the investigators.",
        "New Jersey unions scheduled a rally in support of the drivers.",
    ] + ["A separate analysis focused on national patterns."] * 4
    is_approved, violations = _evaluate(obs, narration)
    kw_violations = [v for v in violations if "Keyword over-repetition" in v]
    assert not kw_violations, f"Expected no keyword violation, got: {kw_violations}"


def test_keyword_repetition_still_flags_real_slop():
    """A non-topic filler word hammered in most shots is still caught."""
    obs = ObserverAgent()
    narration = [
        "The gadget whirred and spun in the lab today.",
        "The gadget hummed while technicians ran diagnostics.",
        "The gadget beeped during the evening calibration run.",
        "The gadget paused as the software updated itself.",
        "The gadget resumed once the new firmware loaded.",
        "The gadget flashed a green light after the reset.",
        "The gadget vibrated on the bench all afternoon.",
        "The gadget steamed slightly under the heat lamp.",
        "The gadget clicked as relays switched positions.",
        "The gadget cooled down before the next test cycle.",
        "The gadget warmed up again for the final trial.",
        "The gadget held steady through the stress test.",
    ] + ["Scientists gathered the final dataset."] * 3
    is_approved, violations = _evaluate(obs, narration)
    kw_violations = [v for v in violations if "Keyword over-repetition" in v]
    assert kw_violations, "Expected a keyword over-repetition violation for filler 'gadget'"


def test_source_diversity_lenient_for_tiny_cites():
    """1/2 citation split (50%) must NOT trip the over-citation rule."""
    obs = ObserverAgent()
    narration = [
        "Entrepreneur magazine documented the initial filing this week.",
        "Reuters carried the reaction from the company afterward.",
    ]
    is_approved, violations = _evaluate(obs, narration, facts=[_fact("Entrepreneur"), _fact("Reuters")])
    div_violations = [v for v in violations if "Source Diversity" in v]
    assert not div_violations, f"Expected no source-diversity violation at 1/2 cites, got: {div_violations}"


def test_source_diversity_enforced_at_scale():
    """Heavy single-source over-attribution with many citations is still caught."""
    obs = ObserverAgent()
    narration = [
        "Entrepreneur reported the story and added context on the ruling.",
        "Entrepreneur followed up with details on the decision.",
        "Entrepreneur interviewed the lead attorney about the outcome.",
        "Reuters carried a brief note on the same development.",
    ]
    is_approved, violations = _evaluate(obs, narration, facts=[_fact("Entrepreneur"), _fact("Reuters")])
    div_violations = [v for v in violations if "Source Diversity" in v]
    assert div_violations, "Expected over-citation violation at 4/5 cites"


def test_topic_entities_from_summary_excluded():
    """Central summary entities are excluded from keyword over-repetition."""
    obs = ObserverAgent()
    narration = [
        "The attorney general pressed the company over market power this week.",
        "The attorney general filed the antitrust suit on Tuesday.",
        "The attorney general accused the firm of abusing delivery costs.",
        "The attorney general cited drivers who faced low pay.",
        "The attorney general reviewed documents from the warehouses.",
        "The attorney general asked the court for a full hearing.",
        "The attorney general met with worker advocates on Monday.",
        "The attorney general said the case would proceed quickly.",
        "The attorney general demanded answers from executives.",
        "The attorney general gathered evidence over several months.",
        "The attorney general outlined the legal theory in detail.",
        "The attorney general promised a thorough investigation.",
    ] + ["Analysts noted broader implications for the sector."] * 3
    is_approved, violations = _evaluate(obs, narration)
    kw_violations = [v for v in violations if "Keyword over-repetition" in v]
    assert not kw_violations, f"Expected no keyword violation for summary entity, got: {kw_violations}"


def test_revision_loop_cap_structure():
    """The orchestrator's revision loop allows up to 3 bounded retries."""
    import inspect
    import src.agents.orchestrator as orch
    source = inspect.getsource(orch.OrchestratorAgent.run_pipeline)
    assert "MAX_REVISIONS = 3" in source, "Expected bounded MAX_REVISIONS = 3 revision cap"
    assert "for attempt in range(1, MAX_REVISIONS + 1):" in source, "Expected a bounded revision loop"
