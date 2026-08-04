import re
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity


HIGH_RPM_TAXONOMY = {
    "finance_global": ["fed", "interest", "rate", "inflation", "bonds", "stocks", "sec", "treasury", "market", "portfolio", "wall street", "banking", "economy", "recession", "wealth"],
    "tech_global": ["semiconductor", "ai", "nvidia", "chips", "cloud", "data center", "apple", "microsoft", "google", "meta", "crypto", "tsmc", "quantum", "software", "saas"],
    "finance_tech_india": ["rbi", "nifty", "sensex", "sebi", "startup", "unicorn", "tata", "reliance", "adani", "infosys", "tcs", "gdp", "upi", "fii", "dii", "ipo", "paytm", "zerodha"]
}


def compute_keyword_vector(text: str, vocabulary: List[str]) -> np.ndarray:
    """
    Computes TF-style term frequency vector for input text across vocabulary.
    """
    text_lower = text.lower()
    vector = np.zeros(len(vocabulary), dtype=float)
    for idx, word in enumerate(vocabulary):
        if word in text_lower:
            vector[idx] += 1.0
    return vector


def calculate_rpm_cosine_similarity(text: str) -> float:
    """
    Calculates maximum cosine similarity of candidate text across High-RPM Taxonomy sub-categories.
    Returns RPM score normalized in [0.0, 1.0].
    """
    max_sim = 0.0

    for category, vocab in HIGH_RPM_TAXONOMY.items():
        candidate_vec = compute_keyword_vector(text, vocab)
        taxonomy_vec = np.ones(len(vocab), dtype=float)

        norm_cand = np.linalg.norm(candidate_vec)
        norm_tax = np.linalg.norm(taxonomy_vec)

        if norm_cand > 1e-6:
            similarity = np.dot(candidate_vec, taxonomy_vec) / (norm_cand * norm_tax)
            if similarity > max_sim:
                max_sim = similarity

    if max_sim < 1e-6:
        return 0.1  # Baseline default score

    # Scale similarity to [0.3, 1.0] for realistic RPM range
    score = 0.3 + (max_sim * 0.7)
    return float(np.round(min(1.0, score), 4))


def calculate_semantic_novelty_index(
    candidate_text: str, past_published_texts: List[str]
) -> float:
    """
    Calculates Information Density & Novelty Index (IDI):
    IDI = 1.0 - max( CosineSim(candidate, past_video_i) )
    Higher IDI means the topic is fresh and non-repetitive.
    """
    if not past_published_texts:
        return 1.0  # Completely novel

    try:
        corpus = [candidate_text] + past_published_texts
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Calculate cosine similarity of candidate (index 0) against all past texts (index 1 onwards)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        max_sim = float(np.max(similarities))
    except Exception:
        max_sim = 0.0

    idi_score = 1.0 - max_sim
    return float(np.round(max(0.0, idi_score), 4))


class TextEmbeddingEngine:
    """
    TF-IDF based text embedding engine for cosine similarity computation using scikit-learn.
    Used by CTRCuriosityPredictor for multimodal title-thumbnail disparity checking.
    Provides embed_text() and cosine_similarity() as a stable API surface.
    """

    def __init__(self, vocab_size: int = 128):
        self.vectorizer = HashingVectorizer(n_features=vocab_size, norm='l2', stop_words='english')

    def embed_text(self, text: str, vocab_size: int = 128) -> np.ndarray:
        """
        Produces a fixed-dimension TF sparse embedding for the given text.
        Uses scikit-learn HashingVectorizer to prevent collisions.
        """
        if vocab_size != self.vectorizer.n_features:
            self.vectorizer = HashingVectorizer(n_features=vocab_size, norm='l2', stop_words='english')
        return self.vectorizer.transform([text]).toarray()[0]

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Computes cosine similarity between two embedding vectors.
        """
        return float(cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0][0])


embedding_engine = TextEmbeddingEngine()
