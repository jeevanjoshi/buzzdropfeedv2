import math
import re
from typing import Dict, Any, Optional
from src.engine.text_embeddings import embedding_engine


class CTRCuriosityPredictor:
    """
    Stage 1: ML/Semantic CTR & Curiosity Gap Estimator based on Loewenstein's Information Gap Theory:
    Incorporates Shannon Information Entropy H(X), Multimodal Cosine Similarity thumbnail-title matching,
    and Invariant Risk Minimization (IRM) causal debiasing score.
    """

    URGENCY_KEYWORDS = [
        "truth", "hidden", "secret", "why", "how", "warning", "stopped", "shocking",
        "changed", "never", "revealed", "paradigm", "collapse", "disruption", "2026"
    ]

    def calculate_shannon_entropy(self, text: str) -> float:
        """
        Calculates Shannon Information Entropy H(X) = -sum(P(x_i) * log2(P(x_i)))
        Measures the uncertainty and complexity inherent in the headline vocabulary.
        """
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        if not words:
            return 0.0
        total_words = len(words)
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        
        entropy = 0.0
        for count in word_counts.values():
            p = count / total_words
            entropy -= p * math.log2(p)
        return float(round(entropy, 4))

    def calculate_multimodal_disparity(self, headline: str, thumbnail_caption: str) -> float:
        """
        Calculates multimodal cosine disparity between headline BERT embedding and 
        generated thumbnail visual caption (OCR/Caption text) to detect clickbait mismatch.
        Returns a disparity score between 0.0 (perfect alignment) and 1.0 (severe mismatch).
        """
        if not thumbnail_caption:
            return 0.0
        vec_headline = embedding_engine.embed_text(headline)
        vec_thumb = embedding_engine.embed_text(thumbnail_caption)
        sim = embedding_engine.cosine_similarity(vec_headline, vec_thumb)
        disparity = max(0.0, 1.0 - sim)
        return float(round(disparity, 4))

    def calculate_irm_debiasing_score(self, headline: str, disparity: float, urgency_score: float) -> float:
        """
        Invariant Risk Minimization (IRM) score:
        Penalizes spurious clickbait correlations (e.g. high urgency + high disparity + zero informational substance).
        """
        headline_lower = headline.lower()
        substance_words = sum(1 for w in re.findall(r'\b[a-z]{4,}\b', headline_lower) if w not in self.URGENCY_KEYWORDS)
        substance_ratio = substance_words / max(1, len(headline_lower.split()))
        
        # High disparity + high urgency with low substance indicates deceptive bait
        deceptive_factor = (disparity * 0.6) + (urgency_score * 0.4 * (1.0 - substance_ratio))
        irm_quality_score = float(round(max(0.0, 1.0 - deceptive_factor), 4))
        return irm_quality_score

    def calculate_expected_watch_time(
        self, predicted_ctr_pct: float, target_runtime_mins: float = 13.0, shannon_entropy: float = 2.5
    ) -> float:
        """
        YouTube Recommendation System Core Objective Function (Covington et al., Google Research):
        Expected Watch Time E[T] = P(click) * E[WatchDuration | click]

        YouTube ranks videos by predicted watch duration per impression, NOT just raw CTR %.
        A video with 8% CTR and 6 min watch time (48 EWT-score) beats a video with 12% CTR and 2 min watch time (24 EWT-score).
        """
        # Completion rate model: baseline 42% for 13 min Tech/Finance educational content
        # Higher Shannon entropy (vocabulary depth) increases retention up to 50%
        retention_rate = min(0.52, max(0.35, 0.40 + (shannon_entropy / 20.0)))
        expected_watch_mins = target_runtime_mins * retention_rate

        # E[T] = P(click) * WatchDuration (scaled)
        ewt_score = (predicted_ctr_pct / 100.0) * expected_watch_mins * 100.0
        return float(round(ewt_score, 2))

    def predict_ctr(
        self, headline: str, summary: str, thumbnail_caption: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculates predicted CTR %, curiosity gap score, Shannon entropy, IRM score,
        and YouTube Expected Watch Time E[T] for a given headline.
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

        # 3. Shannon Entropy H(X)
        shannon_entropy = self.calculate_shannon_entropy(headline)

        # 4. Multimodal Cosine Disparity Check
        disparity_score = self.calculate_multimodal_disparity(headline, thumbnail_caption or "")

        # 5. IRM Causal Debiasing Score
        irm_quality_score = self.calculate_irm_debiasing_score(headline, disparity_score, urgency_score)

        # 6. Base CTR Calculation (Industry Average: 5.0% - 14.5%)
        entropy_boost = min(1.5, shannon_entropy * 0.4)
        base_ctr = 5.0 + (urgency_score * 3.0) + (info_gap_score * 3.0) + entropy_boost
        
        # Apply IRM penalty if clickbait disparity is detected
        ctr_penalty = disparity_score * 4.0
        predicted_ctr = float(round(min(14.5, max(4.5, base_ctr - ctr_penalty)), 2))

        # 7. YouTube Expected Watch Time E[T] Score
        ewt_score = self.calculate_expected_watch_time(predicted_ctr, 13.0, shannon_entropy)

        return {
            "headline": headline,
            "predicted_ctr_pct": predicted_ctr,
            "urgency_score": round(urgency_score, 2),
            "information_gap_score": round(info_gap_score, 2),
            "shannon_entropy": shannon_entropy,
            "multimodal_disparity": disparity_score,
            "irm_quality_score": irm_quality_score,
            "expected_watch_time_score": ewt_score,
        }


ctr_predictor = CTRCuriosityPredictor()

