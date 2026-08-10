import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.schemas.state import TopicCandidate


class MonetizationYieldOptimizer:
    """
    Stage 7: Advanced Mathematical & ML Monetization Yield Optimizer:
    Calculates expected ad revenue yield R(i) = Views(i) * (RPM(i) / 1000) * MidRollMultiplier.
    Uses Pareto Multi-Objective Optimization and enforces:
    - Target Revenue View Threshold: V_req = (E_target / R_est) * 1000
    - Niche RPM Matrix lookup for real-world RPM ranges per content category
    - Competitor 30-day view volume scraping & rejection gate
    """

    # Niche RPM Matrix: Category -> (min_net_rpm_usd, max_net_rpm_usd, mid_net_rpm_usd)
    # CREATOR-NET RPM per 1,000 views (what lands in the creator's account, after
    # YouTube's revenue share/Premium/etc.). Keys are aligned to the category
    # strings produced by RSS ingestion (AUDIENCE_NICHE_MAP) so revenue forecasts
    # resolve correctly instead of falling back to defaults.
    #
    # Values recalibrated to recent (Q2 2026), US/UK/CA/AU monetized long-form
    # planning ranges (TubeAnalytics/CreatorsMetrics/OutlierKit). NOTE: RPM is NOT
    # a fixed percentage of CPM (it includes Premium/memberships), so these are
    # benchmarked net-RPM bands, not a CPM x multiplier.
    #
    # Finance/Investing highest; Real Estate and enterprise AI/Tech also top-tier;
    # Health/Science/Space mid; consumer entertainment lowest.
    NICHE_RPM_MATRIX = {
        "Personal Finance & Investing":           (8.0, 18.0, 13.0),
        # Finance *education* (concepts only) pays the highest CPM of the group
        "Personal Finance Education":             (10.0, 22.0, 15.0),
        "Technology & Artificial Intelligence":   (6.0, 14.0, 9.5),
        "Business & Entrepreneurship":            (7.0, 16.0, 11.0),
        "Health & Wellness":                      (3.0, 8.0, 5.0),
        "Science & Innovation":                   (5.0, 12.0, 8.0),
        "Space & Scientific Innovation":          (5.0, 12.0, 8.0),
        "History & Documentary":                  (4.0, 10.0, 6.5),
        "Global Trends & Infotainment":           (1.5, 5.0, 3.0),
        "Real Estate":                            (7.0, 18.0, 12.0),
        # Legacy aliases (RAG-derived category names) for safety
        "Global Economics & Finance":             (8.0, 18.0, 13.0),
        "Global Trends & Cultural Infotainment":  (1.5, 5.0, 3.0),
        "Health & Science":                       (3.0, 8.0, 5.0),
        "Geopolitics & World Affairs":            (3.0, 9.0, 5.5),
    }

    def _net_rpm_usd(self, candidate: TopicCandidate) -> float:
        """
        Returns a realistic creator-NET RPM in USD for a candidate, derived from
        the niche benchmark midpoint scaled by how strongly the topic matches the
        high-RPM taxonomy (rpm_score in [0.3, 1.0]).
        """
        rpm_min, rpm_max, rpm_mid = self.NICHE_RPM_MATRIX.get(
            getattr(candidate, "niche_category", ""), (3.0, 10.0, 6.0)
        )
        match = getattr(candidate, "rpm_score", 0.5)
        scaled = rpm_mid * (0.8 + 0.4 * match)
        return round(max(rpm_min, min(rpm_max, scaled)), 2)

    def calculate_required_views(self, target_revenue_usd: float, category: str) -> Dict[str, float]:
        """
        Stage 7: Revenue Yield Filter Formula:
        V_req = (E_target / R_est) * 1000
        Calculates minimum views required to achieve target revenue for a given niche RPM.
        """
        rpm_min, rpm_max, rpm_mid = self.NICHE_RPM_MATRIX.get(
            category, (4.0, 10.0, 7.0)
        )
        v_req_optimistic = (target_revenue_usd / rpm_max) * 1000
        v_req_realistic  = (target_revenue_usd / rpm_mid) * 1000
        v_req_pessimistic = (target_revenue_usd / rpm_min) * 1000
        return {
            "category": category,
            "target_revenue_usd": target_revenue_usd,
            "niche_rpm_mid_usd": rpm_mid,
            "v_req_optimistic": round(v_req_optimistic),
            "v_req_realistic": round(v_req_realistic),
            "v_req_pessimistic": round(v_req_pessimistic),
        }

    def filter_by_competitor_volume(
        self, category: str, competitor_30d_avg_views: float, target_revenue_usd: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Competitor Volume Rejection Gate:
        Rejects topic if competitor 30-day average views < V_req (realistic).
        This ensures the niche has sufficient organic demand to sustain target revenue.

        ``target_revenue_usd`` defaults to the channel's derived per-video revenue
        gate (monthly goal ÷ daily cadence, ``channel_phase_manager.
        REVENUE_GATE_MIN_USD``) so the gate, the gate and the $2,000/month goal
        never drift apart.
        """
        if target_revenue_usd is None:
            try:
                from src.engine.channel_phase_manager import REVENUE_GATE_MIN_USD
                target_revenue_usd = REVENUE_GATE_MIN_USD
            except Exception:
                target_revenue_usd = 33.33
        v_req = self.calculate_required_views(target_revenue_usd, category)
        v_required = v_req["v_req_realistic"]
        passes = competitor_30d_avg_views >= v_required
        return {
            "passes_revenue_gate": passes,
            "competitor_30d_avg_views": competitor_30d_avg_views,
            "v_req_realistic": v_required,
            "decision": "APPROVED" if passes else f"REJECTED — needs {v_required:,.0f} views, competitor shows {competitor_30d_avg_views:,.0f}"
        }

    def calculate_midroll_multiplier(self, script_runtime_minutes: float) -> float:
        """
        Calculates mid-roll ad revenue multiplier based on video length.
        - < 8 mins: 1.0 (No mid-rolls allowed)
        - 8-11 mins: 1.8 (2 mid-roll ads)
        - 11-14 mins: 2.6 (3 mid-roll ads)
        - >= 14 mins: 3.4 (4 mid-roll ads)
        """
        if script_runtime_minutes < 8.0:
            return 1.0
        elif script_runtime_minutes < 11.0:
            return 1.8
        elif script_runtime_minutes < 14.0:
            return 2.6
        else:
            return 3.4

    def estimate_predicted_views(
        self, tvs_score: float, ctr_score: float, idi_score: float, sat_score: float, base_views: float = 25000.0
    ) -> float:
        """
        Predicts expected view volume V(i) using trend velocity, CTR, novelty, and saturation penalty.
        """
        tvs_factor = (max(10.0, tvs_score) / 50.0) ** 1.5
        ctr_factor = (max(0.04, ctr_score) / 0.08) ** 2.0
        novelty_factor = max(0.2, idi_score)
        saturation_penalty = max(0.1, (1.0 - min(0.9, sat_score))) ** 0.8

        predicted_views = base_views * tvs_factor * ctr_factor * novelty_factor * saturation_penalty
        return float(np.round(predicted_views, 2))

    # Monthly seasonal RPM factor (planning, based on industry data: Q4 = holiday
    # ad-spend peak, Q1 = annual low). Applied to the revenue forecast.
    _SEASONAL_MULTIPLIER = {
        1: 0.80, 2: 0.82, 3: 0.88,   # Q1 low
        4: 0.93, 5: 0.96, 6: 0.95,   # Q2 recovery
        7: 0.98, 8: 1.00, 9: 1.05,   # Q3 growth
        10: 1.25, 11: 1.40, 12: 1.45,  # Q4 peak
    }

    def _seasonal_multiplier(self, now=None) -> float:
        import datetime as _dt
        now = now or _dt.datetime.now(_dt.timezone.utc)
        return self._SEASONAL_MULTIPLIER.get(now.month, 1.0)

    def _locale_multiplier(self, region: str) -> float:
        """
        Audience-locale RPM factor. Shares its per-market RPM baselines with
        :mod:`region_intelligence` so the revenue forecast agrees with the
        market the topic was selected for:
          us 1.00 | uk 0.97 | ca 0.96 | au 0.94 | eu 0.90 | india 0.25
        Legacy labels still work: "global"/"all" -> 1.00 (US-led), "india" and
        any Asia/South-market alias -> 0.25, "eu"/europe  -> 0.90.
        """
        try:
            from .region_intelligence import MARKETS as _R_MARKETS
            r = (region or "all").lower().strip()
            if r in _R_MARKETS:
                return float(_R_MARKETS[r]["rpm"])
        except Exception:
            pass  # import/edge guard: fall through to the legacy heuristic
        r = (region or "all").lower()
        if any(k in r for k in ("india", "asia", "sa", "africa", "pakistan", "philippines", "indonesia", "ph", "in")):
            return 0.25
        if any(k in r for k in ("eu", "europe", "de", "nl", "no")):
            return 0.90
        if any(k in r for k in ("uk", "ca", "au")):
            return 0.97 if "uk" in r else (0.96 if "ca" in r else 0.94)
        return 1.00  # us/uk/ca/au / all / global

    def calculate_revenue_yield(
        self, candidate: TopicCandidate, estimated_runtime_mins: float = 13.0, region: str = "all"
    ) -> Dict[str, float]:
        """
        Calculates total expected ad revenue yield in USD R(i) for a topic candidate.
        Applies niche net-RPM plus audience-locale and seasonal planning multipliers.
        """
        rpm = self._net_rpm_usd(candidate)  # realistic creator-net RPM for the niche
        ctr_est = 0.08 + (candidate.idi_score * 0.04)  # Estimated CTR between 8% and 12%
        
        predicted_views = self.estimate_predicted_views(
            tvs_score=candidate.tvs_score,
            ctr_score=ctr_est,
            idi_score=candidate.idi_score,
            sat_score=candidate.sat_score
        )

        midroll_multiplier = self.calculate_midroll_multiplier(estimated_runtime_mins)
        base_ad_revenue = (predicted_views / 1000.0) * rpm
        total_revenue_yield = (
            base_ad_revenue * midroll_multiplier
            * self._locale_multiplier(region) * self._seasonal_multiplier()
        )

        return {
            "predicted_views": round(predicted_views, 0),
            "estimated_rpm_usd": round(rpm, 2),
            "midroll_multiplier": midroll_multiplier,
            "locale_multiplier": self._locale_multiplier(region),
            "seasonal_multiplier": self._seasonal_multiplier(),
            "base_ad_revenue_usd": round(base_ad_revenue, 2),
            "total_expected_revenue_usd": round(total_revenue_yield, 2)
        }

    def rank_candidates_by_revenue_pareto(
        self, candidates: List[TopicCandidate]
    ) -> List[Tuple[TopicCandidate, float]]:
        """
        Ranks candidates using Multi-Objective Pareto Revenue Maximization.
        """
        scored = []
        for cand in candidates:
            metrics = self.calculate_revenue_yield(cand)
            rev_yield = metrics["total_expected_revenue_usd"]
            scored.append((cand, rev_yield))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


monetization_optimizer = MonetizationYieldOptimizer()
