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
    """
    if not past_published_texts:
        return 1.0  # Completely novel

    combined_corpus = " ".join([candidate_text] + past_published_texts).lower()
    words = list(set([w.strip(",.!?") for w in combined_corpus.split() if len(w) > 3]))

    if not words:
        return 1.0

    cand_vec = compute_keyword_vector(candidate_text, words)
    cand_norm = np.linalg.norm(cand_vec)

    if cand_norm < 1e-6:
        return 1.0

    max_sim = 0.0
    for past_text in past_published_texts:
        past_vec = compute_keyword_vector(past_text, words)
        past_norm = np.linalg.norm(past_vec)
        if past_norm > 1e-6:
            sim = np.dot(cand_vec, past_vec) / (cand_norm * past_norm)
            if sim > max_sim:
                max_sim = sim

    idi_score = 1.0 - max_sim
    return float(np.round(max(0.0, idi_score), 4))
