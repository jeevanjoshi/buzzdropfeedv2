"""Opportunity score for topic selection.

Views-per-competitor opportunity: how much attention each competing video on a
topic claims on average. Higher is better (an underserved niche with a few
high-views videos is the sweet spot). The raw ratio is log-tamed into a
monotonic [0, 1] score so it composes cleanly with other TOPSIS criteria and
feeds the opportunity hard gate (``OPPORTUNITY_MIN_SCORE``).
"""

import math


def compute_opportunity(competitor_30d_avg_views: float, competing_video_count: float) -> float:
    """Map ``competitor_30d_avg_views / max(1, competing_count)`` to [0, 1].

    - 0 competitors or 0 measured views -> 0.0 (unknown / no observable demand).
    - Monotonic: more views-per-competitor always raises the score.
    - Log-tamed so a single runaway competitor can't saturate the score.
    """
    if not competitor_30d_avg_views or competitor_30d_avg_views <= 0:
        return 0.0
    raw = competitor_30d_avg_views / max(1.0, float(competing_video_count or 0.0))
    return round(1.0 - 1.0 / (1.0 + math.log1p(raw)), 4)
