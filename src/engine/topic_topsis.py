import numpy as np
from typing import List, Optional
from src.schemas.state import TopicCandidate


# ─────────────────────────────────────────────────────────────────────────────
# Phase-aware TOPSIS weight vectors [TVS, RPM, IDI, SDI, SHM, VPH, SAT, REGION_REV]
# ─────────────────────────────────────────────────────────────────────────────
# REGION_REV (index 7) = expected ad revenue for the candidate's best region
# (``candidate.regional_revenue_usd``). Per the converged plan, "ad revenue in
# the region has the HIGHEST weight" in the MONETISED phases:
#   REVENUE/SCALE : index 7 holds the single largest weight.
# GROWTH        : DISCOVERY-LED — TVS (trend) + IDI (novelty) lead for
#                   watch-time/subscriber growth. A pre-YPP channel earns $0 in
#                   ads, so REGION_REV (index 7) is ZEROED: the unscaled revenue
#                   forecast is fictional for an unmonetized channel and must not
#                   steer selection. Shareability (SHM, index 4) and underserved
#                   demand (VPH, index 5) are boosted to drive subscriber
#                   conversion, and the saturation cost weight (SAT, index 6) is
#                   raised so a small channel avoids crowded niches it cannot win.
TOPSIS_WEIGHTS_GROWTH  = [0.25, 0.03, 0.25, 0.07, 0.15, 0.13, 0.12, 0.00]

# REVENUE : REGION_REV leads; niche RPM (market ceiling) is the next-strongest.
TOPSIS_WEIGHTS_REVENUE = [0.14, 0.20, 0.12, 0.05, 0.08, 0.08, 0.05, 0.28]

# SCALE   : Balanced-but-revenue-led — REGION_REV highest, RPM next.
TOPSIS_WEIGHTS_SCALE   = [0.15, 0.18, 0.12, 0.05, 0.10, 0.07, 0.05, 0.28]


def rank_topics_topsis(
    candidates: List[TopicCandidate],
    weights: Optional[List[float]] = None,
    channel_phase: str = "REVENUE",
) -> List[TopicCandidate]:
    """
    Ranks candidate topics using TOPSIS (Technique for Order Preference by Similarity
    to Ideal Solution) with phase-aware weights.

    Criteria (8 columns):
    1. TVS (Trend Velocity)            — Benefit (+)
    2. RPM (Advertiser Similarity)     — Benefit (+)  <- market ceiling
    3. IDI (Semantic Novelty)          — Benefit (+)
    4. SDI (Sentiment Disruption)      — Benefit (+)
    5. SHM (Social Media Hype)         — Benefit (+)
    6. VPH (YouTube Views-per-Hour)    — Benefit (+)
    7. SAT (Market Saturation Penalty) — Cost   (-)
    8. REGION_REV (Regional ad revenue)— Benefit (+)  <- HIGHEST single weight
    """
    if not candidates:
        return []

    if len(candidates) == 1:
        candidates[0].topsis_score = 1.0
        return candidates

    # Select weight vector based on channel phase
    if weights is None:
        if channel_phase == "SCALE":
            weights = TOPSIS_WEIGHTS_SCALE
        elif channel_phase == "REVENUE":
            weights = TOPSIS_WEIGHTS_REVENUE
        else:
            weights = TOPSIS_WEIGHTS_GROWTH

    m = len(candidates)
    n = 8  # Criteria count

    # Construct Decision Matrix X (m x 8)
    X = np.zeros((m, n), dtype=float)
    for idx, c in enumerate(candidates):
        X[idx, 0] = c.tvs_score
        X[idx, 1] = c.rpm_score
        X[idx, 2] = c.idi_score
        X[idx, 3] = c.sdi_score
        X[idx, 4] = getattr(c, "shm_score", 1.0)
        X[idx, 5] = getattr(c, "vph_score", 1.0)
        X[idx, 6] = c.sat_score
        X[idx, 7] = max(0.0, float(getattr(c, "regional_revenue_usd", 0.0) or 0.0))

    # Step 1: Normalize Matrix R
    col_norms = np.sqrt(np.sum(X ** 2, axis=0))
    col_norms[col_norms < 1e-6] = 1e-6
    R = X / col_norms

    # Step 2: Calculate Weighted Normalized Matrix V
    w = np.array(weights, dtype=float)
    w = w / np.sum(w)  # Ensure weights sum to 1.0
    V = R * w

    # Step 3: Determine Ideal (A+) and Anti-Ideal (A-) Solutions
    # Columns 0..5 and 7 are Benefit (+); Column 6 (SAT) is Cost (-)
    A_plus  = np.zeros(n, dtype=float)
    A_minus = np.zeros(n, dtype=float)

    for j in range(n):
        if j == 6:                  # SAT = cost criterion: best is lowest
            continue
        A_plus[j]  = np.max(V[:, j])
        A_minus[j] = np.min(V[:, j])

    A_plus[6]  = np.min(V[:, 6])
    A_minus[6] = np.max(V[:, 6])

    # Step 4: Calculate Euclidean Distances S+ and S-
    S_plus  = np.sqrt(np.sum((V - A_plus)  ** 2, axis=1))
    S_minus = np.sqrt(np.sum((V - A_minus) ** 2, axis=1))

    # Step 5: Calculate Relative Closeness C_i*
    denom = S_plus + S_minus
    denom[denom < 1e-6] = 1e-6
    C_star = S_minus / denom

    for idx, c in enumerate(candidates):
        c.topsis_score = float(np.round(C_star[idx], 4))

    ranked_candidates = sorted(candidates, key=lambda x: x.topsis_score or 0.0, reverse=True)
    return ranked_candidates


class TopicTOPSISEngine:
    """
    Class wrapper around rank_topics_topsis.
    Accepts optional channel_phase so the Orchestrator can inject it.
    """

    def rank_candidates(
        self,
        candidates: List[TopicCandidate],
        weights: Optional[List[float]] = None,
        channel_phase: str = "REVENUE",
    ) -> List[TopicCandidate]:
        return rank_topics_topsis(candidates, weights=weights, channel_phase=channel_phase)
