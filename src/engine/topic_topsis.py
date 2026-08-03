import numpy as np
from typing import List, Optional
from src.schemas.state import TopicCandidate


# ─────────────────────────────────────────────────────────────────────────────
# Phase-aware TOPSIS weight vectors [TVS, RPM, IDI, SDI, SHM, VPH, SAT]
# ─────────────────────────────────────────────────────────────────────────────
# GROWTH  : Prioritise IDI (novelty/searchability) + TVS (trending velocity).
#           RPM weight is minimal — ads not yet enabled, focus is watch-time.
TOPSIS_WEIGHTS_GROWTH  = [0.30, 0.05, 0.30, 0.10, 0.15, 0.05, 0.05]

# REVENUE : RPM is the primary revenue ceiling — given the highest single weight.
#           TVS remains important for organic reach; IDI ensures freshness.
TOPSIS_WEIGHTS_REVENUE = [0.22, 0.30, 0.18, 0.05, 0.12, 0.08, 0.05]

# SCALE   : Balanced — sustain RPM while maintaining reach for brand partnerships.
TOPSIS_WEIGHTS_SCALE   = [0.20, 0.28, 0.20, 0.05, 0.15, 0.07, 0.05]


def rank_topics_topsis(
    candidates: List[TopicCandidate],
    weights: Optional[List[float]] = None,
    channel_phase: str = "REVENUE",
) -> List[TopicCandidate]:
    """
    Ranks candidate topics using TOPSIS (Technique for Order Preference by Similarity
    to Ideal Solution) with phase-aware weights.

    Criteria (7 columns):
    1. TVS (Trend Velocity)            — Benefit (+)
    2. RPM (Advertiser Similarity)     — Benefit (+)  <- 30% in REVENUE phase
    3. IDI (Semantic Novelty)          — Benefit (+)
    4. SDI (Sentiment Disruption)      — Benefit (+)
    5. SHM (Social Media Hype)         — Benefit (+)
    6. VPH (YouTube Views-per-Hour)    — Benefit (+)
    7. SAT (Market Saturation Penalty) — Cost   (-)
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
    n = 7  # Criteria count

    # Construct Decision Matrix X (m x 7)
    X = np.zeros((m, n), dtype=float)
    for idx, c in enumerate(candidates):
        X[idx, 0] = c.tvs_score
        X[idx, 1] = c.rpm_score
        X[idx, 2] = c.idi_score
        X[idx, 3] = c.sdi_score
        X[idx, 4] = getattr(c, "shm_score", 1.0)
        X[idx, 5] = getattr(c, "vph_score", 1.0)
        X[idx, 6] = c.sat_score

    # Step 1: Normalize Matrix R
    col_norms = np.sqrt(np.sum(X ** 2, axis=0))
    col_norms[col_norms < 1e-6] = 1e-6
    R = X / col_norms

    # Step 2: Calculate Weighted Normalized Matrix V
    w = np.array(weights, dtype=float)
    w = w / np.sum(w)  # Ensure weights sum to 1.0
    V = R * w

    # Step 3: Determine Ideal (A+) and Anti-Ideal (A-) Solutions
    # Columns 0..5 are Benefit (+); Column 6 is Cost (-)
    A_plus  = np.zeros(n, dtype=float)
    A_minus = np.zeros(n, dtype=float)

    for j in range(6):  # Benefit criteria
        A_plus[j]  = np.max(V[:, j])
        A_minus[j] = np.min(V[:, j])

    # Cost criterion (Saturation): best = lowest SAT
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
