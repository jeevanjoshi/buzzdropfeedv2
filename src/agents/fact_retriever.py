import os
import uuid
import datetime
from typing import List, Dict, Any, Optional
from src.schemas.state import GlobalState, TopicCandidate, VerifiedFact, RevenueForecast
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent, compute_state_hash
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

# Pre-YPP saturation floor (GROWTH phase only). A candidate with
# ``sat_score >= SATURATION_FLOOR`` is in a fully-saturated niche (>=10 competing
# videos) that a brand-new, unmonetized channel cannot realistically win. Culled
# before ranking; the run aborts if the entire pool is oversaturated rather than
# ship a topic with no path to traction. Tune via env.
SATURATION_FLOOR = float(os.getenv("SATURATION_FLOOR", "1.0"))

# Absolute TOPSIS quality floor. TOPSIS C* is RELATIVE (0.5 = mediocre
# compromise, 1.0 = ideal profile). A winner below this means the whole candidate
# pool is weak; refuse to publish a low-quality compromise. Tune via env.
TOPSIS_MIN_SCORE = float(os.getenv("TOPSIS_MIN_SCORE", "0.6"))


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
                country_code = self._wb_country_code(region)
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

    def _synthesize_narrow_topics(
        self, corpus_headlines: List[str]
    ) -> List[Dict[str, Any]]:
        """
        LLM-propose narrow, evergreen, search-demand topics (tool-vs-tool,
        how-to, tool deep-dives, enterprise tooling) grounded in the day's RSS
        corpus. Returns raw ``{headline, summary, keywords, demand_query}`` specs
        or ``[]`` when synthesis is unavailable/fails (caller decides the gate).
        """
        if not corpus_headlines:
            return []
        try:
            from src.engine.tool_topic_synthesizer import synthesize_tool_topics
            return synthesize_tool_topics(corpus_headlines)
        except Exception as e:
            print(f"[FactRetriever] Tool-topic synthesis skipped ({e}); continuing with RSS candidates.")
            return []

    def _score_synthetic_candidate(self, spec: Dict[str, Any], idx: int) -> TopicCandidate:
        """
        Convert an LLM tool-topic spec into a full TopicCandidate with the same
        phase-aware scoring path the RSS engine uses, so it competes fairly in
        TOPSIS. ``demand_query`` is retained on the candidate for the caller's
        precise per-topic demand measurement.
        """
        from src.engine.rss_ingestion import (
            classify_audience_type, AUDIENCE_NICHE_MAP,
        )
        headline = (spec.get("headline") or "").strip()
        summary = (spec.get("summary") or "").strip()
        demand_query = (spec.get("demand_query") or headline)[:120]
        keywords = list(spec.get("keywords") or [])
        if not keywords:
            import re as _re
            keywords = [w for w in _re.split(r"\W+", headline.lower()) if w][:8]
        audience = classify_audience_type(headline, summary)
        if audience == "blocked":
            audience = "tech"  # never let a synthesized tool topic be blocked
        niche_category, _ = AUDIENCE_NICHE_MAP.get(
            audience, ("Technology & Artificial Intelligence", 16.0))

        # Reuse the RSS scoring helpers so the 7-criteria vector is consistent.
        r = self.rss_engine
        rpm = r._classify_niche_rpm(headline, keywords)
        if rpm < 0.35:  # hard RPM floor, matches RSS ingestion
            rpm = 0.45  # tool/dev is inherently monetisable; keep candidate viable
        idi = r._compute_idi(headline)
        sdi = r._compute_sdi(headline, summary)
        competing = r._estimate_competing_video_count(keywords, 1)

        return TopicCandidate(
            candidate_id=f"narrow-synth-{idx:02d}",
            headline=headline,
            summary=summary,
            source_url="",
            keywords=keywords,
            tvs_score=r._compute_tvs(keywords, None, 1),
            rpm_score=rpm,
            idi_score=idi,
            sdi_score=sdi,
            shm_score=1.2,
            vph_score=1.0,
            sat_score=r._compute_sat(competing),
            audience_type=audience,
            niche_category=niche_category,
            demand_query=demand_query,
        )

    def _measure_and_gate_synthetic(
        self, synth: List[Dict[str, Any]]
    ) -> List[TopicCandidate]:
        """
        STRICT gate for synthesized topics (not news → no presumption of relevance):
          * Every synthetic candidate MUST be measured with precise_topic_demand on
            its exact narrow query. Unmeasured (None / API / quota fall-through) ⇒
            CULLED — a narrow topic with no measurable competitor demand is a bad bet.
          * Measured but below the OPPORTUNITY_MIN_SCORE floor ⇒ CULLED (same floor
            the RSS opportunity hard gate enforces, applied to ALL phases here).
          * Measured opportunity updates the candidate's vph/avg-views for TOPSIS.
        Only measured, floor-clearing candidates are returned.
        """
        if not synth:
            return []
        kept: List[TopicCandidate] = []
        for idx, spec in enumerate(synth, start=1):
            try:
                cand = self._score_synthetic_candidate(spec, idx)
            except Exception as e:
                print(f"[FactRetriever] Synthetic candidate skipped ({e}).")
                continue

            # ── Persistent Topic Deduplication Gate (Cross-Run Check) ────────
            from src.engine.topic_deduplicator import topic_deduplicator
            is_dup, sim_score, matched_title = topic_deduplicator.check_topic_similarity(cand.headline, cand.summary)
            if is_dup:
                print(f"[FactRetriever] Synthetic topic CULLED (duplicate/similar to '{matched_title}', sim={sim_score:.4f}): '{cand.headline[:80]}'")
                continue

            demand_query = getattr(cand, "demand_query", "") or cand.headline[:120]
            demand = None
            try:
                demand = youtube_topic_demand.precise_topic_demand(demand_query)
            except Exception:
                demand = None
            if not (demand and demand.get("competitor_30d_avg_views")):
                print(f"[FactRetriever] Synthetic topic CULLED (unmeasured demand): '{cand.headline[:80]}'")
                continue
            count = float(demand.get("video_count") or 0)
            avg_views = float(demand["competitor_30d_avg_views"])
            cand.competitor_30d_avg_views = avg_views
            cand.competing_video_count = count
            cand.opportunity_score = compute_opportunity(avg_views, count)
            if cand.opportunity_score < OPPORTUNITY_MIN_SCORE:
                print(f"[FactRetriever] Synthetic topic CULLED (opportunity {cand.opportunity_score:.2f} < {OPPORTUNITY_MIN_SCORE}): '{cand.headline[:80]}'")
                continue
            # vph proxy ON measured views/hr so TOPSIS VPH reflects real velocity.
            vph_raw = float(demand.get("views_per_hour") or 0)
            cand.vph_score = round(max(0.5, min(3.0, vph_raw / 250.0)), 4)
            kept.append(cand)
        return kept

    def _verify_winner_audience(self, winner: TopicCandidate) -> TopicCandidate:
        """
        Re-label the already-selected winner's audience_type / niche_category using
        an LLM, correcting keyword-classifier misroutes (e.g. a geopolitics story
        that the substring matcher tagged as ``finance_edu``).

        This runs AFTER the winner is chosen — it only re-labels the single winner,
        it never re-runs TOPSIS or picks a different candidate. On any failure
        (LLM unavailable, empty/garbage output, low confidence, unknown label) it
        leaves the keyword-derived label intact, so the pipeline never aborts or
        retries on classification. ~1 LLM call per run.
        """
        from src.engine.audience_taxonomy import (
            AUDIENCE_TAXONOMY, niche_for,
        )
        keyword_label = getattr(winner, "audience_type", "general") or "general"
        try:
            from src.engine.llm_client import LLMClient
            llm = LLMClient()
            taxonomy_desc = "\n".join(
                f"- {k}: {v['description']}"
                for k, v in sorted(AUDIENCE_TAXONOMY.items(), key=lambda kv: kv[1]["priority"])
            )
            system_prompt = (
                "You are an audience-classification expert for a YouTube documentary "
                "channel. Given a news headline and summary, assign the single best "
                "audience category from the taxonomy below. Be precise: a story about "
                "geopolitics, war, nuclear threats or international misinformation must be "
                "'geopolitics', NOT 'finance_edu' or 'general'. Return ONLY valid JSON."
            )
            prompt = (
                f"Taxonomy (audience_type -> description):\n{taxonomy_desc}\n\n"
                f"Headline: {winner.headline}\n"
                f"Summary: {winner.summary}\n\n"
                "Return JSON: {\"audience_type\": str, \"confidence\": float in [0,1], "
                "\"reason\": str}"
            )
            result = llm.generate_json(prompt, system_prompt=system_prompt, route="classify")
            if not result:
                return winner
            label = str(result.get("audience_type", "")).strip().lower()
            confidence = float(result.get("confidence", 0.0) or 0.0)
            if label not in AUDIENCE_TAXONOMY:
                return winner
            if label in ("blocked", "general") and keyword_label not in ("blocked", "general"):
                # LLM downgrading a specialised topic to blocked/general is riskier
                # than the keyword label; keep the keyword label unless very confident.
                if confidence < 0.9:
                    return winner
            if label == keyword_label:
                return winner
            if confidence < 0.6:
                return winner
            winner.audience_type = label
            winner.niche_category = niche_for(label)
            print(f"[FactRetriever] Audience re-labeled by LLM verify: "
                  f"{keyword_label} -> {label} (conf={confidence:.2f})")
        except Exception as e:
            print(f"[FactRetriever] Audience LLM-verify skipped (kept keyword label): {e}")
        return winner

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

    def _wb_country_code(self, region: str) -> str:
        """Map a region/market label to a World Bank ISO3 code for the macro
        enrichment fact. Understands the region_intelligence market codes too."""
        _MARKET_ISO = {
            "us": "USA", "uk": "GBR", "ca": "CAN", "au": "AUS", "eu": "DEU",
            "india": "IND", "in": "IND", "global": "USA", "all": "USA",
        }
        r = (region or "all").lower().strip()
        return _MARKET_ISO.get(r, "USA")

    def _growth_score(self, cand: "TopicCandidate") -> float:
        """
        Pre-YPP (GROWTH) selection proxy.

        The objective before YPP unlock is watch-time + subscriber growth, NOT ad
        revenue (which is $0 for an unmonetized channel). This scores a candidate
        by expected view-volume — the single quantity that drives BOTH watch-hours
        and subscriber conversion — amplified by novelty (save/return likelihood)
        and shareability (social spread → subscribers). Saturation is already
        penalised inside ``estimate_predicted_views``, so crowded niches naturally
        score low. Phantom ad revenue is intentionally NOT used.
        """
        try:
            idi = max(0.0, float(getattr(cand, "idi_score", 0.0) or 0.0))
            ctr = 0.08 + idi * 0.04
            predicted_views = monetization_optimizer.estimate_predicted_views(
                tvs_score=float(getattr(cand, "tvs_score", 0.0) or 0.0),
                ctr_score=ctr,
                idi_score=idi,
                sat_score=float(getattr(cand, "sat_score", 0.0) or 0.0),
            )
            shm = max(0.5, float(getattr(cand, "shm_score", 1.0) or 1.0))
            return float(predicted_views * (0.5 + idi) * shm)
        except Exception:
            # Safe fallback so selection never crashes: plain discovery proxy.
            return float(
                (getattr(cand, "tvs_score", 0.0) or 0.0)
                + (getattr(cand, "idi_score", 0.0) or 0.0)
            )

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

        # Narrow tool-topic synthesis (LLM, grounded in the day's RSS corpus).
        # Adds evergreen, search-demand topics RSS news never surfaces, then a
        # STRICT per-candidate demand gate: unmeasured synthetic = culled, and
        # measured-but-below-floor = culled (all-channel-phases). Toggle via
        # TOOL_TOPIC_SYNTHESIS=0 (env). Never crashes the run on LLM failure.
        if os.getenv("TOOL_TOPIC_SYNTHESIS", "1").strip().lower() not in ("0", "false"):
            try:
                synth_specs = self._synthesize_narrow_topics([c.headline for c in candidates])
                synth_kept = self._measure_and_gate_synthetic(synth_specs)
                if synth_kept:
                    print(f"[FactRetriever] {len(synth_specs)} synthesized tool topics → {len(synth_kept)} passed strict demand gate.")
                    candidates = list(synth_kept) + candidates
            except Exception as e:
                print(f"[FactRetriever] Tool-topic synthesis pass failed ({e}); using RSS candidates only.")

        # ── Documentary / investigative storability gate ─────────────────────
        # Direct news (announcements, press releases, price ticks, results) that
        # our automation CANNOT turn into a 6-act documentary is NEVER considered
        # for ranking. Deterministic verdict + optional LLM cross-check (which may
        # only cull near-boundary topics, never resurrect culled ones). Evergreen
        # synthesized tool topics pass through by design. If EVERYTHING is culled
        # -> SHIP BEST-WITH-WARNING: the highest documentary-scoring survivor is
        # retained with a loud log line (the RAG-sufficiency gate still protects
        # factual depth downstream, and the channel stays alive on thin days).
        from src.engine.documentary_potential import (
            gate_candidates, refine_with_llm, score_documentary_potential,
        )
        kept, culled = gate_candidates(candidates)
        for cand, audit in culled:
            print(f"[FactRetriever] CULLED (direct news, no doc potential): '{cand.headline[:90]}' ({audit['reason']})")
        if not kept:
            fallback_winner = max(candidates, key=lambda c: score_documentary_potential(c)["score"]) if candidates else None
            if fallback_winner is None:
                raise ValueError("No topic candidates survive after the documentary storability gate.")
            print(f"[FactRetriever] WARNING: ALL {len(candidates)} candidates failed the documentary storability gate. "
                  f"Shipping best-with-warning: '{fallback_winner.headline[:90]}'")
            kept = [fallback_winner]
        else:
            llm_culls = refine_with_llm(kept)
            if llm_culls:
                kept = [c for c in kept if c.candidate_id not in llm_culls]
                if not kept:
                    kept = [max(candidates, key=lambda c: score_documentary_potential(c)["score"])]
                print(f"[FactRetriever] LLM cross-check culled {len(llm_culls)} near-boundary direct-news topics.")
        candidates = kept

        # ── Dynamic region decision (ad-revenue-weighted, decided HERE) ─────
        # region=="all" (the DEFAULT) means the market is selected from the
        # day/time window + topic affinity + per-market RPM + events, then flows
        # to every downstream stage. A CLI --global/--india pins it (fixed mode).
        dynamic_region = (region or "all").strip().lower() in ("all", "auto", "dynamic")
        projected_publish_min = None
        region_intel = None
        region_profiles: Dict[str, Any] = {}
        if dynamic_region:
            try:
                from src.engine import region_intelligence as region_intel
                now = datetime.datetime.now(datetime.timezone.utc)
                projected_publish_min = now.hour * 60 + now.minute + region_intel.DEFAULT_RUNTIME_MIN
                for c in candidates:
                    try:
                        region_profiles[c.candidate_id] = region_intel.candidate_region_profile(
                            c, projected_publish_min)
                    except Exception:
                        continue
            except Exception as e:
                print(f"[FactRetriever] Region intelligence unavailable ({e}); defaulting to global.")
                dynamic_region = False

        # Populate the revenue-led TOPSIS 8th criterion: every candidate carries
        # its best-market expected ad revenue (dynamic) or the fixed-region
        # forecast, so ranking itself is regional-ad-revenue weighted.
        for c in candidates:
            if dynamic_region and c.candidate_id in region_profiles:
                c.regional_revenue_usd = round(
                    region_profiles[c.candidate_id]["region_revenue_usd"], 2)
            else:
                try:
                    revx = monetization_optimizer.calculate_revenue_yield(
                        c, estimated_runtime_mins=13.0, region=region)
                    c.regional_revenue_usd = round(float(revx["total_expected_revenue_usd"]), 2)
                except Exception:
                    c.regional_revenue_usd = 0.0

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

        # Pre-YPP saturation floor (GROWTH only). A brand-new, unmonetized
        # channel cannot win a fully-saturated niche (sat_score >= SATURATION_FLOOR
        # ≈ >=10 competing videos). Cull those before ranking so they can never be
        # selected; abort the run if the entire pool is oversaturated rather than
        # ship a topic the channel has no realistic path to traction in.
        if channel_phase == "GROWTH":
            sat_gated = [
                c for c in candidates
                if (getattr(c, "sat_score", 0.0) or 0.0) < SATURATION_FLOOR
            ]
            if not sat_gated:
                raise ValueError(
                    f"All {len(candidates)} candidate topics are fully saturated "
                    f"(sat_score >= {SATURATION_FLOOR:.2f}). Refusing to select an "
                    "oversaturated niche pre-YPP — the channel cannot compete in it."
                )
            culled = len(candidates) - len(sat_gated)
            if culled:
                print(f"[FactRetriever] Saturation floor culled {culled} fully-saturated (pre-YPP) topic(s).")
            candidates = sat_gated

        # Rank candidates using phase-aware TOPSIS Decision Engine
        ranked_candidates = self.topsis_engine.rank_candidates(candidates, channel_phase=channel_phase)

        # ── Analytics feedback tie-break ("top growth drivers") ────────────────
        # A soft, non-fatal bias that "doubles down on what works": candidates
        # whose audience/niche historically converts (subscriber-gain + retention
        # signal from logs/analytics_feedback.json) get a small boost to their
        # TOPSIS score before ordering. No signal → no-op (identical to before).
        # Gated by CSVG_ANALYTICS_FEEDBACK=1 (default on) and never raises.
        try:
            if os.getenv("CSVG_ANALYTICS_FEEDBACK", "1").strip().lower() not in ("0", "false", "no"):
                from src.engine.analytics_feedback import analytics_feedback
                _biased = []
                for _c in ranked_candidates:
                    _aud = getattr(_c, "audience_type", "") or ""
                    _bias = analytics_feedback.get_audience_bias(_aud)
                    if _bias > 1.0 and _c.topsis_score is not None:
                        _c.topsis_score = round(_c.topsis_score * _bias, 4)
                    _biased.append(_c)
                ranked_candidates = sorted(_biased, key=lambda x: x.topsis_score or 0.0, reverse=True)
        except Exception as e:
            print(f"[FactRetriever] Analytics feedback bias skipped (non-fatal): {e}")

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

        # In the GROWTH phase, prioritize synthesized evergreen topics to build a
        # stable, long-term search-driven subscriber base (per the "one evergreen
        # video a day" strategy), rather than short-lived current affairs/news.
        if channel_phase == "GROWTH":
            evergreens = [c for c in ranked_candidates if str(getattr(c, "candidate_id", "")).startswith("narrow-synth-")]
            others = [c for c in ranked_candidates if not str(getattr(c, "candidate_id", "")).startswith("narrow-synth-")]
            ranked_candidates = evergreens + others

        # Apply pre-filtering using Audience and Revenue Gates
        # High-RPM enforcement: if any genuinely-classified (non-general, non-blocked)
        # niche candidate exists, do not let a low-value 'general' lifestyle topic win,
        # even in GROWTH phase (protects the revenue orientation of the channel).
        has_specialized = any(
            getattr(c, "audience_type", "") not in ("general", "blocked") for c in ranked_candidates
        )

        # ── Winner selection (gates) ──────────────────────────────────────────────
        # The dynamic region decision was already made pre-TOPSIS (region_profiles)
        # and TOPSIS was already ranked with the phase-aware weights. Here we only
        # enforce the audience/niche/revenue/competitor gates, then pick the winner.
        #   * REVENUE / SCALE : the band candidate with the best revenue-led region
        #     score (the TOPSIS band keeps the ranking phase-aware).
        #   * GROWTH (pre-YPP) : the band candidate with the best GROWTH score
        #     (view-volume × novelty × shareability) — NEVER the phantom ad-revenue
        #     score, which is $0 for an unmonetized channel.
        winner = None
        winner_market = "us"
        winner_region_score = -1.0
        region_band = max(1, int(round(len(ranked_candidates) * 0.5)))
        ranked_slice = ranked_candidates[:region_band] if dynamic_region else ranked_candidates

        for rank_idx, cand in enumerate(ranked_slice):
            # 1. Audience Gate (block 'blocked' categories like gossip/entertainment)
            if getattr(cand, "audience_type", "") == "blocked":
                continue

            # 1b. Prefer high-RPM niches over generic 'general' content when available
            if has_specialized and getattr(cand, "audience_type", "") == "general":
                continue

            # 1c. Region score for this candidate (precomputed in region_profiles)
            cand_market = None
            region_score = 0.0
            if dynamic_region and cand.candidate_id in region_profiles:
                cand_market = region_profiles[cand.candidate_id]["market"]
                region_score = region_profiles[cand.candidate_id]["score"]

            # 2. Revenue Gate (in REVENUE / SCALE phases, expected revenue must be >= MIN).
            #    Evaluated against the market THIS candidate would be targeted at.
            gate_region = cand_market if cand_market else region
            if channel_phase in ("REVENUE", "SCALE"):
                rev = monetization_optimizer.calculate_revenue_yield(cand, estimated_runtime_mins=13.0, region=gate_region)
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

            # ── Selection score ──────────────────────────────────────────────────
            # GROWTH optimises for audience growth (view-volume × novelty ×
            # shareability), NOT ad revenue (which is $0 pre-YPP). REVENUE/SCALE
            # keep the revenue-led region score.
            if channel_phase == "GROWTH":
                select_score = self._growth_score(cand)
            else:
                select_score = region_score

            if dynamic_region:
                if select_score > winner_region_score:
                    winner, winner_market, winner_region_score = cand, cand_market or "us", select_score
            else:
                winner = cand
                break

        if not winner:
            winner = ranked_candidates[0]

        # ── Absolute TOPSIS quality floor ──────────────────────────────────────
        # TOPSIS C* is RELATIVE (0.5 = mediocre compromise, 1.0 = ideal profile). A
        # winner below the floor means the whole candidate pool is weak; refuse to
        # publish a low-quality compromise rather than ship it. Tune via env.
        if winner.topsis_score is not None and winner.topsis_score < TOPSIS_MIN_SCORE:
            raise ValueError(
                f"Selected topic '{winner.headline}' has TOPSIS score "
                f"{winner.topsis_score:.4f} below the quality floor "
                f"({TOPSIS_MIN_SCORE:.4f}). Refusing to publish a weak-compromise topic."
            )

        # Re-label the winner's audience/niche via a single LLM verify call (corrects
        # keyword misroutes; never re-selects or solves — falls back to keyword label).
        winner = self._verify_winner_audience(winner)

        # Persist the decided region so media producer / story designer / revenue
        # forecast all target the SAME market.
        eff_region = region
        l2_region = "global"
        region_reason = f"fixed CLI region '{region}'" if not dynamic_region else None
        if dynamic_region and winner.candidate_id in region_profiles:
            prof = region_profiles[winner.candidate_id]
            eff_region = prof["market"]
            l2_region = prof["l2_region"]
            region_reason = (
                f"dynamic:{prof['market']} (published ~{projected_publish_min // 60}:{projected_publish_min % 60:02d} UTC, "
                f"rev=${prof['region_revenue_usd']:.2f}, {prof['reason']})"
            )
            state.region = l2_region
            state.region_market = prof["market"]
            state.region_reason = region_reason
            print(f"[FactRetriever] DYNAMIC REGION -> market={prof['market']} l2={l2_region} ({region_reason})")
        else:
            state.region_market = l2_region if l2_region != "global" else eff_region
            state.region_reason = region_reason or ""

        # Compute revenue forecast for the winning topic (exact market RPM).
        # Pass channel_stats so the reported forecast is maturity-scaled + flagged
        # (an unmonetized/new channel no longer shows a $1,800 "realized" forecast).
        rev = monetization_optimizer.calculate_revenue_yield(
            winner, estimated_runtime_mins=13.0, region=eff_region,
            channel_stats=getattr(state, "channel_stats", None),
        )
        state.revenue_forecast = RevenueForecast(
            predicted_views=rev["predicted_views"],
            estimated_rpm_usd=rev["estimated_rpm_usd"],
            midroll_multiplier=rev["midroll_multiplier"],
            base_ad_revenue_usd=rev["base_ad_revenue_usd"],
            total_expected_revenue_usd=rev["total_expected_revenue_usd"],
            audience_type=getattr(winner, "audience_type", "general"),
            niche_category=getattr(winner, "niche_category", "Technology & Artificial Intelligence"),
            monetization_eligible=rev["monetization_eligible"],
            maturity_scaled=rev["maturity_scaled"],
            maturity_factor=rev["maturity_factor"],
            projected_views_at_scale=rev["projected_views_at_scale"],
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
                "region": state.region,
                "region_market": state.region_market,
                "region_reason": state.region_reason,
                "channel_phase": channel_phase,
                "revenue_forecast_usd": rev["total_expected_revenue_usd"],
                "regional_revenue_usd": winner.regional_revenue_usd,
                "audience_type": getattr(winner, "audience_type", "general"),
            },
            state_hash=compute_state_hash(state),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        return msg
