from src.schemas.state import TopicCandidate
from src.engine.topic_topsis import rank_topics_topsis


def test_topsis_ranking_correctness():
    c1 = TopicCandidate(
        candidate_id="cand-001",
        headline="High Trend High RPM High Novelty Story",
        summary="Major tech corporate collapse story with high engagement.",
        source_url="https://example.com/1",
        keywords=["tech", "ai", "finance"],
        tvs_score=95.0,  # High trend
        rpm_score=0.95,  # High RPM
        idi_score=0.90,  # High novelty
        sdi_score=1.5,   # High controversy
        sat_score=1.0    # Low saturation (GOOD)
    )

    c2 = TopicCandidate(
        candidate_id="cand-002",
        headline="Low Trend High Saturation Story",
        summary="Generic finance tip covered by everyone.",
        source_url="https://example.com/2",
        keywords=["grocery", "coupons"],
        tvs_score=10.0,  # Low trend
        rpm_score=0.30,  # Low RPM
        idi_score=0.10,  # Low novelty
        sdi_score=0.1,   # Low controversy
        sat_score=50.0   # High saturation (BAD)
    )

    ranked = rank_topics_topsis([c1, c2])

    assert len(ranked) == 2
    assert ranked[0].candidate_id == "cand-001"
    assert ranked[0].topsis_score > ranked[1].topsis_score
    assert ranked[0].topsis_score >= 0.80
