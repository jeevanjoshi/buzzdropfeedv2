import uuid
import datetime
from typing import List, Dict, Any, Optional
from src.schemas.state import TopicCandidate, VerifiedFact, GlobalState
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent
from src.engine.trend_velocity import calculate_ema_trend_velocity, calculate_zscore_anomaly
from src.engine.text_embeddings import calculate_rpm_cosine_similarity, calculate_semantic_novelty_index
from src.engine.social_signals import calculate_social_hype_multiplier, calculate_youtube_vph_velocity
from src.engine.topic_topsis import rank_topics_topsis
from src.engine.rss_ingestion import fetch_live_rss_feeds


class FactRetrieverAgent:
    """
    Fact Retriever Agent responsible for topic ingestion, mathematical trend velocity calculation,
    novelty verification, social hype tracking, YouTube VPH velocity, and TOPSIS multi-criteria ranking.
    """

    def __init__(self, name: str = "FactRetriever"):
        self.name = name

    def fetch_candidate_topics(
        self, sample_feeds: Optional[List[Dict[str, Any]]] = None, use_live_rss: bool = False, region: str = "all"
    ) -> List[TopicCandidate]:
        """
        Ingests financial/tech stories and computes initial quantitative metrics.
        """
        if use_live_rss:
            sample_feeds = fetch_live_rss_feeds(region=region)

        if not sample_feeds:
            # High-retention seed stories for Infotainment
            sample_feeds = [
                {
                    "headline": "Federal Reserve Benchmark Decision Triggers $500B Tech Volatility",
                    "summary": "Central bank keeps interest rates steady while signaling persistent inflation, causing massive shifts in chip stock valuations and bond yields.",
                    "url": "https://example.com/finance/fed-rates-volatility",
                    "keywords": ["fed", "interest", "rates", "inflation", "bonds", "tech", "volatility"],
                    "search_history": [45.0, 52.0, 58.0, 64.0, 72.0, 85.0, 98.0],
                    "sentiment_variance": 0.85,
                    "competing_video_count": 3,
                    "posts_6h": 450,
                    "posts_24h_avg": 1200.0,
                    "recent_views": [12000, 25000, 18000],
                    "recent_hours": [4.0, 8.0, 6.0]
                },
                {
                    "headline": "How Intel Lost the Microchip Monopoly to TSMC and Nvidia",
                    "summary": "Inside the strategic missteps and corporate downfall that shifted a $1 trillion semiconductor empire to Taiwan and Silicon Valley competitors.",
                    "url": "https://example.com/tech/intel-tsmc-downfall",
                    "keywords": ["intel", "semiconductor", "tsmc", "nvidia", "chips", "monopoly", "downfall"],
                    "search_history": [30.0, 35.0, 42.0, 55.0, 70.0, 88.0, 100.0],
                    "sentiment_variance": 0.92,
                    "competing_video_count": 1,
                    "posts_6h": 890,
                    "posts_24h_avg": 1500.0,
                    "recent_views": [45000, 80000],
                    "recent_hours": [5.0, 10.0]
                },
                {
                    "headline": "Generic Tips to Save Money on Grocery Bills in 2026",
                    "summary": "A basic overview of budget shopping techniques and coupon usage for family households.",
                    "url": "https://example.com/finance/save-grocery-money",
                    "keywords": ["grocery", "money", "save", "coupons", "shopping"],
                    "search_history": [10.0, 12.0, 11.0, 10.0, 11.0, 12.0, 11.0],
                    "sentiment_variance": 0.10,
                    "competing_video_count": 45,
                    "posts_6h": 10,
                    "posts_24h_avg": 100.0,
                    "recent_views": [500],
                    "recent_hours": [24.0]
                }
            ]

        candidates = []
        for idx, feed in enumerate(sample_feeds):
            tvs = calculate_ema_trend_velocity(feed["search_history"])
            z_score, is_spike = calculate_zscore_anomaly(feed["search_history"])
            rpm = calculate_rpm_cosine_similarity(f"{feed['headline']} {feed['summary']}")
            idi = calculate_semantic_novelty_index(f"{feed['headline']} {feed['summary']}", [])
            sdi = float(feed.get("sentiment_variance", 0.5) * 1.5)
            
            # Extract Social Hype & YouTube VPH Velocity
            posts_6h = feed.get("posts_6h", 100)
            posts_24h = feed.get("posts_24h_avg", 400.0)
            shm = calculate_social_hype_multiplier(posts_6h, posts_24h)
            
            recent_views = feed.get("recent_views", [5000])
            recent_hours = feed.get("recent_hours", [5.0])
            vph = calculate_youtube_vph_velocity(recent_views, recent_hours)

            sat = float(feed.get("competing_video_count", 0))

            cand = TopicCandidate(
                candidate_id=f"cand-{idx+1:03d}",
                headline=feed["headline"],
                summary=feed["summary"],
                source_url=feed["url"],
                keywords=feed["keywords"],
                tvs_score=tvs,
                rpm_score=rpm,
                idi_score=idi,
                sdi_score=sdi,
                shm_score=shm,
                vph_score=vph,
                sat_score=sat
            )
            candidates.append(cand)

        return candidates

    def process(self, state: GlobalState, use_live_rss: bool = False, region: str = "all") -> A2AMessage:
        """
        Executes Fact Retriever workflow:
        1. Ingest candidate stories
        2. Apply 7-criteria TOPSIS ranking algorithm
        3. Select top candidate
        4. Return A2AMessage to Orchestrator
        """
        raw_candidates = self.fetch_candidate_topics(use_live_rss=use_live_rss, region=region)
        ranked_candidates = rank_topics_topsis(raw_candidates)

        selected = ranked_candidates[0] if ranked_candidates else None
        state.selected_topic = selected
        state.execution_stage = "TOPIC_SELECTED"

        if selected:
            state.verified_facts.append(
                VerifiedFact(
                    source_id=selected.candidate_id,
                    headline=selected.headline,
                    summary=selected.summary,
                    url=selected.source_url,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                )
            )

        msg = A2AMessage(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            sender=AgentRole.FACT_RETRIEVER,
            target=AgentRole.ORCHESTRATOR,
            intent=AgentIntent.TOPIC_SELECTED,
            payload={
                "selected_candidate": selected.model_dump() if selected else None,
                "ranked_candidates_count": len(ranked_candidates),
                "top_topsis_score": selected.topsis_score if selected else 0.0
            },
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        return msg
