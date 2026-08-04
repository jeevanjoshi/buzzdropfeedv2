"""
Topic Deduplicator — persistent history manager for published topics.

Prevents the pipeline from generating videos on identical or semantically similar topics across runs.
Stores history in `published_topics.json` and performs semantic cosine similarity checks.
"""
import os
import json
import datetime
from typing import List, Dict, Any, Tuple
from src.engine.text_embeddings import calculate_semantic_novelty_index, compute_keyword_vector, embedding_engine

_HISTORY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../published_topics.json"))

# Similarity threshold: candidate with CosineSim > 0.60 against past topic is rejected
SIMILARITY_REJECTION_THRESHOLD = 0.60


def load_published_history() -> List[Dict[str, Any]]:
    """Loads published topic history from JSON file."""
    if os.path.exists(_HISTORY_FILE):
        try:
            with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def save_published_history(history: List[Dict[str, Any]]) -> None:
    """Persists published topic history to JSON file."""
    try:
        os.makedirs(os.path.dirname(_HISTORY_FILE), exist_ok=True)
        with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except IOError:
        pass


def record_published_topic(headline: str, summary: str = "", keywords: List[str] = None) -> None:
    """
    Appends a newly published topic to history and persists to disk.
    Called by PublisherAgent upon successful YouTube upload.
    """
    history = load_published_history()
    entry = {
        "headline": headline,
        "summary": summary[:200] if summary else "",
        "keywords": keywords or [],
        "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    history.append(entry)
    # Keep rolling window of last 500 published topics
    history = history[-500:]
    save_published_history(history)


def check_topic_similarity(headline: str, summary: str = "") -> Tuple[bool, float, str]:
    """
    Checks candidate headline/summary against all past published topics.
    Returns tuple: (is_duplicate: bool, max_similarity: float, matched_past_headline: str)
    """
    history = load_published_history()
    if not history:
        return False, 0.0, ""

    past_headlines = [item["headline"] for item in history]
    
    # Compute IDI novelty score (IDI = 1.0 - max_similarity)
    cand_text = f"{headline} {summary}".strip()
    idi = calculate_semantic_novelty_index(cand_text, past_headlines)
    max_sim = float(round(1.0 - idi, 4))

    if max_sim >= SIMILARITY_REJECTION_THRESHOLD:
        # Find which past headline matched most closely
        best_match = ""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            corpus = [cand_text] + past_headlines
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            best_idx = int(np.argmax(sims))
            best_match = past_headlines[best_idx]
        except Exception:
            best_match = past_headlines[0]
        return True, max_sim, best_match

    return False, max_sim, ""


# Export singleton interface
topic_deduplicator = type("TopicDeduplicator", (), {
    "load_published_history": staticmethod(load_published_history),
    "save_published_history": staticmethod(save_published_history),
    "record_published_topic": staticmethod(record_published_topic),
    "check_topic_similarity": staticmethod(check_topic_similarity),
    "SIMILARITY_REJECTION_THRESHOLD": SIMILARITY_REJECTION_THRESHOLD,
})()
