import os
import uuid
import datetime
from typing import List, Dict, Any, Optional
from src.schemas.state import GlobalState, TopicCandidate, VerifiedFact, RevenueForecast
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent
from src.engine.rss_ingestion import LiveRSSIngestionEngine
from src.engine.topic_topsis import TopicTOPSISEngine
from src.engine.api_ninjas import APINinjasRetriever
from src.engine.external_apis import external_api_manager
from src.engine.space_cinema_apis import space_cinema_api_manager
from src.engine.monetization_optimizer import monetization_optimizer
from src.engine.opportunity_score import compute_opportunity
from src.engine.youtube_topic_demand import youtube_topic_demand

# Floor for the opportunity (views-per-competitor) hard gate. Tune via env
# (calibration script) without code edits.
OPPORTUNITY_MIN_SCORE = float(os.getenv("OPPORTUNITY_MIN_SCORE", "0.5"))


class FactRetrieverAgent:
    """
    Fact Retriever Agent responsible for Phase 1:
    1. Ingesting live RSS feeds from Global & Indian English news sources.
    2. Enriching facts via World Bank, Marketaux, API Ninjas, NASA Open APIs, TMDB Cinema, and Wikipedia Historical Archives.
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
        Fetches topic candidates and verified facts from RSS feeds, World Bank Data API, Marketaux, NASA, and History APIs.
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
                    url="https://data.worldbank.org",
                    source_name="The World Bank"
                ))

                # 2. Enrich with Historical Date Archives (On This Day)
                hist_facts = space_cinema_api_manager.fetch_on_this_day_history()
                facts.extend(hist_facts)

                # 3. Enrich with NASA APOD Space Telemetry Fact
                nasa_fact = space_cinema_api_manager.fetch_nasa_apod()
                if nasa_fact:
                    facts.append(nasa_fact)

                # 4. Enrich with Marketaux financial news if API token present
                m_facts = external_api_manager.fetch_marketaux_sentiment_news()
                facts.extend(m_facts)

                # 5. Enrich with API Ninjas facts if API key present
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
            url="https://example.com/tech/nvidia-ai-chip",
            source_name="TechCrunch"
        )
        f2 = VerifiedFact(
            source_id="fact-102",
            headline="Intel Process Node Delays",
            summary="Intel delayed 7nm process nodes causing key foundry customers to transition order volume to TSMC.",
            url="https://example.com/tech/intel-tsmc-downfall",
            source_name="The Wall Street Journal"
        )

        return [c1, c2], [f1, f2]

    def _apply_precise_shortlist(self, ranked: List[TopicCandidate]) -> List[TopicCandidate]:
        """Re-order the TOPSIS top-3 by precise per-topic opportunity.

        Runs one on-topic search + batch stats per candidate and recomputes the
        ``opportunity_score``. On any failure the candidate keeps its measured-or-
        unknown score and ordering. Only the top-3 are touched; the rest are
        returned in original order.
        """
        if not ranked:
            return ranked
        top = list(ranked[:3])
        rest = list(ranked[3:])
        for c in top:
            query = (" ".join(c.keywords[:3]) if getattr(c, "keywords", None)
                     else (c.headline or "")[:40]).strip() or (c.headline or "")
            demand = None
            try:
                demand = youtube_topic_demand.precise_topic_demand(query)
            except Exception:
                demand = None
            if demand and demand.get("competitor_30d_avg_views"):
                count = float(demand.get("video_count") or 0)
                avg_views = float(demand["competitor_30d_avg_views"])
                c.competitor_30d_avg_views = avg_views
                c.competing_video_count = count
                c.opportunity_score = compute_opportunity(avg_views, count)
        top.sort(key=lambda x: x.opportunity_score, reverse=True)
        return top + rest

    def process(self, state: GlobalState, use_live_rss: bool = True, region: str = "all",
                channel_phase: str = "REVENUE",
                exclude_headlines: Optional[List[str]] = None) -> A2AMessage:
        """
        Executes Fact Retriever workflow:
        1. Ingests news candidates and facts.
        2. Evaluates candidates using phase-aware TOPSIS.
        3. Updates state.selected_topic, state.verified_facts, state.revenue_forecast.
        4. Emits TOPIC_SELECTED A2AMessage.

        ``exclude_headlines`` filters out previously-tried topics (matched on
        headline, case-insensitive) so a re-run picks a DIFFERENT candidate
        instead of re-selecting the same topic that failed the RAG gate.
        """
        candidates, facts = self.fetch_candidates_and_facts(use_live_rss=use_live_rss, region=region)

        if not candidates:
            raise ValueError("No topic candidates available for selection.")

        # Pre-TOPSIS opportunity hard gate (REVENUE/SCALE only, measured only).
        # A topic with a MEASURED but low views-per-competitor opportunity is a
        # poor bet; cull it before TOPSIS so it can't win. Unmeasured (score 0)
        # passes through so "unknown -> TOPSIS decides". If every candidate is
        # culled in a monetised phase, abort the run rather than ship a bad topic.
        if channel_phase in ("REVENUE", "SCALE"):
            gated = [
                c for c in candidates
                if not (c.opportunity_score > 0 and c.opportunity_score < OPPORTUNITY_MIN_SCORE)
            ]
            if not gated:
                raise ValueError(
                    f"All {len(candidates)} candidate topics failed the opportunity hard gate "
                    f"(OPPORTUNITY_MIN_SCORE={OPPORTUNITY_MIN_SCORE}). "
                    "Refusing to select a low-opportunity topic."
                )
            candidates = gated

        # Rank candidates using phase-aware TOPSIS Decision Engine
        ranked_candidates = self.topsis_engine.rank_candidates(candidates, channel_phase=channel_phase)

        # Drop previously-tried topics (e.g. topics whose RAG corpus was
        # undersupplied) so a retry picks the next-best candidate.
        if exclude_headlines:
            excluded = {h.strip().lower() for h in exclude_headlines if h}
            ranked_candidates = [
                c for c in ranked_candidates
                if c.headline.strip().lower() not in excluded
            ]
            if not ranked_candidates:
                raise ValueError(
                    "All topic candidates were excluded (previously failed RAG "
                    "quality gate). No alternative topic available."
                )

        # B1 shortlist precise check (first pass): for the TOPSIS top-3, measure
        # TRUE on-topic competition and re-rank just those three by opportunity
        # so an underserved-but-strong topic can surface first. Silent
        # fall-through on quota/API failure keeps the TOPSIS ordering.
        ranked_candidates = self._apply_precise_shortlist(ranked_candidates)

        # Apply pre-filtering using Audience and Revenue Gates
        # High-RPM enforcement: if any genuinely-classified (non-general, non-blocked)
        # niche candidate exists, do not let a low-value 'general' lifestyle topic win,
        # even in GROWTH phase (protects the revenue orientation of the channel).
        has_specialized = any(
            getattr(c, "audience_type", "") not in ("general", "blocked") for c in ranked_candidates
        )
        winner = None
        for cand in ranked_candidates:
            # 1. Audience Gate (block 'blocked' categories like gossip/entertainment)
            if getattr(cand, "audience_type", "") == "blocked":
                continue

            # 1b. Prefer high-RPM niches over generic 'general' content when available
            if has_specialized and getattr(cand, "audience_type", "") == "general":
                continue
            
            # 2. Revenue Gate (in REVENUE / SCALE phases, expected revenue must be >= MIN)
            if channel_phase in ("REVENUE", "SCALE"):
                rev = monetization_optimizer.calculate_revenue_yield(cand, estimated_runtime_mins=13.0, region=region)
                from src.engine.channel_phase_manager import channel_phase_manager
                if rev["total_expected_revenue_usd"] < channel_phase_manager.REVENUE_GATE_MIN_USD:
                    continue

            # 3. Competitor View-Volume Gate (REVENUE / SCALE only, only when measured)
            if channel_phase in ("REVENUE", "SCALE"):
                comp_views = getattr(cand, "competitor_30d_avg_views", 0.0)
                if comp_views > 0:
                    gate = monetization_optimizer.filter_by_competitor_volume(
                        cand.niche_category, comp_views
                    )
                    if not gate["passes_revenue_gate"]:
                        continue
            winner = cand
            break

        if not winner:
            winner = ranked_candidates[0]

        # Compute revenue forecast for the winning topic
        rev = monetization_optimizer.calculate_revenue_yield(winner, estimated_runtime_mins=13.0, region=region)
        state.revenue_forecast = RevenueForecast(
            predicted_views=rev["predicted_views"],
            estimated_rpm_usd=rev["estimated_rpm_usd"],
            midroll_multiplier=rev["midroll_multiplier"],
            base_ad_revenue_usd=rev["base_ad_revenue_usd"],
            total_expected_revenue_usd=rev["total_expected_revenue_usd"],
            audience_type=getattr(winner, "audience_type", "general"),
            niche_category=getattr(winner, "niche_category", "Technology & Artificial Intelligence"),
        )

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
                "region": region,
                "channel_phase": channel_phase,
                "revenue_forecast_usd": rev["total_expected_revenue_usd"],
                "audience_type": getattr(winner, "audience_type", "general"),
            },
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        return msg
