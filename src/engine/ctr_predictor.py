import re
from typing import Dict, Any


class CTRCuriosityPredictor:
    """
    ML/Semantic CTR & Curiosity Gap Estimator based on Loewenstein's Information Gap Theory:
    Evaluates title curiosity gap, emotional urgency, and temporal relevance to predict CTR %.
    """

    URGENCY_KEYWORDS = [
        "truth", "hidden", "secret", "why", "how", "warning", "stopped", "shocking",
        "changed", "never", "revealed", "paradigm", "collapse", "disruption", "2026"
    ]

    def predict_ctr(self, headline: str, summary: str) -> Dict[str, Any]:
        """
        Calculates predicted CTR % and curiosity gap score for a given headline.
        """
        headline_lower = headline.lower()
        words = re.findall(r'\b[a-z]{3,}\b', headline_lower)

        # 1. Emotional Urgency Score
        urgency_matches = sum(1 for w in words if w in self.URGENCY_KEYWORDS)
        urgency_score = min(1.0, 0.3 + (urgency_matches * 0.25))

        # 2. Information Gap Density (Question vs Paradigm)
        if any(w in headline_lower for w in ["how i", "why", "what happens", "the truth behind", "explained"]):
            info_gap_score = 0.90
        elif "?" in headline:
            info_gap_score = 0.85
        else:
            info_gap_score = 0.65

        # 3. Base CTR Calculation (Industry Average: 5.0% - 12.0%)
        predicted_ctr = 5.0 + (urgency_score * 3.5) + (info_gap_score * 3.5)
        predicted_ctr = float(round(min(14.5, max(4.5, predicted_ctr)), 2))

        return {
            "headline": headline,
            "predicted_ctr_pct": predicted_ctr,
            "urgency_score": round(urgency_score, 2),
            "information_gap_score": round(info_gap_score, 2)
        }


ctr_predictor = CTRCuriosityPredictor()
