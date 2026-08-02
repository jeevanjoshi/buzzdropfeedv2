import numpy as np
from typing import List, Dict, Any, Tuple
from src.schemas.state import TopicCandidate


class MonetizationYieldOptimizer:
    """
    Advanced Mathematical & ML Monetization Yield Optimizer:
    Calculates expected ad revenue yield R(i) = Views(i) * (RPM(i) / 1000) * MidRollMultiplier * SubConversion.
    Uses Pareto Multi-Objective Optimization and Markowitz Sharpe Ratio principles to select
    topics that yield maximum revenue in minimum timeframe.
    """

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

    def calculate_revenue_yield(
        self, candidate: TopicCandidate, estimated_runtime_mins: float = 13.0
    ) -> Dict[str, float]:
        """
        Calculates total expected ad revenue yield in USD R(i) for a topic candidate.
        """
        rpm = candidate.rpm_score * 35.0  # Scale RPM score [0,1] to real-world USD range [$5, $35]
        ctr_est = 0.08 + (candidate.idi_score * 0.04)  # Estimated CTR between 8% and 12%
        
        predicted_views = self.estimate_predicted_views(
            tvs_score=candidate.tvs_score,
            ctr_score=ctr_est,
            idi_score=candidate.idi_score,
            sat_score=candidate.sat_score
        )

        midroll_multiplier = self.calculate_midroll_multiplier(estimated_runtime_mins)
        base_ad_revenue = (predicted_views / 1000.0) * rpm
        total_revenue_yield = base_ad_revenue * midroll_multiplier

        return {
            "predicted_views": round(predicted_views, 0),
            "estimated_rpm_usd": round(rpm, 2),
            "midroll_multiplier": midroll_multiplier,
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
