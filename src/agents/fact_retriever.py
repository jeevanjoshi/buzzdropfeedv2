import uuid
import datetime
from typing import List, Dict, Any, Optional
from src.schemas.state import GlobalState, TopicCandidate, VerifiedFact
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent
from src.engine.rss_ingestion import LiveRSSIngestionEngine
from src.engine.topic_topsis import TopicTOPSISEngine
from src.engine.api_ninjas import APINinjasRetriever
from src.engine.external_apis import external_api_manager


class FactRetrieverAgent:
    """
    Fact Retriever Agent responsible for Phase 1:
    1. Ingesting live RSS feeds from Global & Indian English news sources.
    2. Enriching facts via World Bank, Marketaux, API Ninjas, and Exa Search APIs.
    3. Evaluating candidates using 7-Criteria TOPSIS Decision Engine.
    4. Selecting #1 topic candidate and populating GlobalState.
    """

    def __init__(
        self,
        name: str = "FactRetriever",
        rss_engine: Optional[LiveRSSIngestionEngine] = None,
        topsis_engine: Optional[TopicTOPSISEngine] = None,
        ninjas_retriever: Optional[APINinjasRetriever] = None
    ):
        self.name = name
        self.rss_engine = rss_engine or LiveRSSIngestionEngine()
        self.topsis_engine = topsis_engine or TopicTOPSISEngine()
        self.ninjas_retriever = ninjas_retriever or APINinjasRetriever()

    def fetch_candidates_and_facts(
        self, use_live_rss: bool = True, region: str = "all"
    ) -> tuple[List[TopicCandidate], List[VerifiedFact]]:
        """
        Fetches topic candidates and verified facts from RSS feeds, World Bank Data API, Marketaux, and API Ninjas.
        """
        if use_live_rss:
            candidates, facts = self.rss_engine.fetch_all_feeds(region=region)
            if candidates and facts:
                # 1. Enrich with World Bank Macro Economic Indicators (GDP & Inflation)
                country_code = "IND" if region == "india" else "USA"
                wb_data = external_api_manager.fetch_world_bank_gdp_inflation(country_code=country_code)
                facts.append(VerifiedFact(
                    source_id="wb-macro-01",
                    headline=f"World Bank Official Economic Data ({country_code})",
                    summary=f"Official World Bank indicators record GDP Growth at {wb_data['gdp_growth']} and Inflation rate at {wb_data['inflation']}.",
                    url="https://data.worldbank.org"
                ))

                # 2. Enrich with Marketaux financial news if API token present
                m_facts = external_api_manager.fetch_marketaux_sentiment_news()
                facts.extend(m_facts)

                # 3. Enrich with API Ninjas facts if API key present
                if self.ninjas_retriever.is_available():
                    ninja_facts = self.ninjas_retriever.fetch_market_news()
                    facts.extend(ninja_facts)

                return candidates, facts

        # Fallback Offline Fixture Data for deterministic testing
        c1 = TopicCandidate(
            candidate_id="cand-001",
            headline="Nvidia Unveils Next-Gen AI Microchip Architecture Shaking Tech Valuation",
            summary="Nvidia announced groundbreaking GPU architecture driving AI data center efficiency, boosting market cap by $150B.",
            source_url="https://example.com/tech/nvidia-ai-chip",
            keywords=["nvidia", "ai", "chips", "gpu", "tech", "semiconductor"],
            tvs_score=92.5,
            rpm_score=0.95,
            idi_score=0.88,
            sdi_score=1.4,
            shm_score=1.8,
            vph_score=2.2,
            sat_score=0.8
        )

        c2 = TopicCandidate(
            candidate_id="cand-002",
            headline="How Intel Lost the Microchip Monopoly to TSMC and Nvidia",
            summary="Inside the strategic missteps and corporate downfall that shifted a $1 trillion semiconductor empire to Taiwan and Silicon Valley competitors.",
            source_url="https://example.com/tech/intel-tsmc-downfall",
            keywords=["intel", "semiconductor", "tsmc", "nvidia", "chips", "monopoly", "downfall"],
            tvs_score=76.3,
            rpm_score=0.92,
            idi_score=0.96,
            sdi_score=1.6,
            shm_score=2.1,
            vph_score=2.5,
            sat_score=0.7
        )

        f1 = VerifiedFact(
            source_id="fact-101",
            headline="Nvidia Market Cap Surge",
            summary="Nvidia market capitalization reached record highs following new architecture announcement, boosting market cap by $150B.",
            url="https://example.com/tech/nvidia-ai-chip"
        )
        f2 = VerifiedFact(
            source_id="fact-102",
            headline="Intel Process Node Delays",
            summary="Intel delayed 7nm process nodes causing key foundry customers to transition order volume to TSMC.",
            url="https://example.com/tech/intel-tsmc-downfall"
        )

        return [c1, c2], [f1, f2]

    def process(self, state: GlobalState, use_live_rss: bool = True, region: str = "all") -> A2AMessage:
        """
        Executes Fact Retriever workflow:
        1. Ingests news candidates and facts.
        2. Evaluates candidates using TOPSIS.
        3. Updates state.selected_topic, state.verified_facts, state.execution_stage.
        4. Emits TOPIC_SELECTED A2AMessage.
        """
        candidates, facts = self.fetch_candidates_and_facts(use_live_rss=use_live_rss, region=region)
        
        if not candidates:
            raise ValueError("No topic candidates available for selection.")

        # Rank candidates using TOPSIS Decision Engine
        ranked_candidates = self.topsis_engine.rank_candidates(candidates)
        winner = ranked_candidates[0]

        # Update Global State
        state.selected_topic = winner
        state.verified_facts = facts
        state.execution_stage = "TOPIC_SELECTED"

        msg = A2AMessage(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            sender=AgentRole.FACT_RETRIEVER,
            target=AgentRole.ORCHESTRATOR,
            intent=AgentIntent.TOPIC_SELECTED,
            payload={
                "selected_candidate": winner.model_dump(),
                "topsis_score": winner.topsis_score,
                "verified_fact_count": len(facts),
                "region": region
            },
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        return msg
