import re
import numpy as np
from typing import List, Dict


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
    Combines TF Cosine Similarity with Jaccard Key-Entity Overlap.
    """
    if not past_published_texts:
        return 1.0  # Completely novel

    # Filter stopwords and extract significant entity words (>3 chars)
    stopwords = {"with", "this", "that", "from", "they", "have", "been", "will", "more", "about",
                 "their", "which", "over", "after", "news", "today", "report", "says", "says", "2026"}
    
    def extract_entity_set(text: str) -> set:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        return set(w for w in words if w not in stopwords)

    cand_entities = extract_entity_set(candidate_text)
    if not cand_entities:
        return 1.0

    max_sim = 0.0

    # Vocabulary for TF Cosine Similarity
    combined_corpus = " ".join([candidate_text] + past_published_texts).lower()
    words = list(set([w.strip(",.!?") for w in combined_corpus.split() if len(w) > 3 and w not in stopwords]))

    if words:
        cand_vec = compute_keyword_vector(candidate_text, words)
        cand_norm = np.linalg.norm(cand_vec)

        for past_text in past_published_texts:
            # 1. Cosine similarity
            cos_sim = 0.0
            if cand_norm > 1e-6:
                past_vec = compute_keyword_vector(past_text, words)
                past_norm = np.linalg.norm(past_vec)
                if past_norm > 1e-6:
                    cos_sim = float(np.dot(cand_vec, past_vec) / (cand_norm * past_norm))

            # 2. Entity Jaccard similarity
            past_entities = extract_entity_set(past_text)
            jaccard_sim = 0.0
            if past_entities and cand_entities:
                intersection = len(cand_entities.intersection(past_entities))
                union = len(cand_entities.union(past_entities))
                jaccard_sim = intersection / union if union > 0 else 0.0
                # If 3+ significant entities match, boost similarity to 0.85
                if intersection >= 3:
                    jaccard_sim = max(jaccard_sim, 0.85)
                elif intersection >= 2:
                    jaccard_sim = max(jaccard_sim, 0.65)

            # Combined similarity metric
            sim = max(cos_sim, jaccard_sim)
            if sim > max_sim:
                max_sim = sim

    idi_score = 1.0 - max_sim
    return float(np.round(max(0.0, idi_score), 4))


class TextEmbeddingEngine:
    """
    Lightweight TF-based text embedding engine for cosine similarity computation.
    Used by CTRCuriosityPredictor for multimodal title-thumbnail disparity checking.
    Provides embed_text() and cosine_similarity() as a stable API surface.
    """

    def embed_text(self, text: str, vocab_size: int = 128) -> np.ndarray:
        """
        Produces a fixed-dimension TF sparse embedding for the given text.
        Uses a deterministic character-hash vocabulary of `vocab_size` buckets.
        """
        words = text.lower().split()
        vector = np.zeros(vocab_size, dtype=float)
        for word in words:
            bucket = hash(word) % vocab_size
            vector[abs(bucket)] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 1e-8:
            vector = vector / norm
        return vector

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Computes cosine similarity between two embedding vectors.
        Returns a value in [-1.0, 1.0], where 1.0 = identical direction.
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


embedding_engine = TextEmbeddingEngine()

