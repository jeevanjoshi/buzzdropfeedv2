import re
import os
import gc
import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# The backend is lazily loaded only when USE_SEMANTIC_GATES=1 AND torch/
# transformers are present (master-only; the Pi and hermetic tests keep the
# TF-IDF fallback). The model stays RESIDENT for the process lifetime: the
# ~0.8 GB cost is far cheaper than a 10-20s reload per pipeline run (4-6/day),
# and release() only frees the model weights (~50 MB) anyway since torch stays
# imported in-process. Call release() manually only if you need that headroom
# back mid-run.
#
# NOTE: the flag is read LAZILY (at each check), not at import time, because
# entrypoints call load_dotenv() AFTER importing this module.
def _semantic_flag() -> bool:
    return os.getenv("USE_SEMANTIC_GATES", "").strip().lower() in ("1", "true", "yes")


class SemanticEmbeddingBackend:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, enabled: Optional[bool] = None):
        self._enabled_override = enabled
        self._model = None

    def _enabled(self) -> bool:
        return self._enabled_override if self._enabled_override is not None else _semantic_flag()

    @property
    def available(self) -> bool:
        """True if the backend is enabled and its deps are importable."""
        if not self._enabled():
            return False
        if self._model is not None:
            return True
        try:
            import torch  # noqa: F401
            import sentence_transformers  # noqa: F401
            return True
        except Exception:
            return False

    def load(self) -> None:
        """Loads the MiniLM model (first call ~10-20s on 2 vCPU). No-op if already
        loaded, disabled, or deps are missing."""
        if not self._enabled() or self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
        except Exception:
            self._model = None

    def release(self) -> None:
        """Frees the model weights + torch overhead. The model re-loads lazily on
        next use; not called automatically (resident by design)."""
        if self._model is None:
            return
        self._model = None
        gc.collect()

    def encode_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Normalised sentence vectors (n, 384) or None when unavailable."""
        if not self._enabled():
            return None
        self.load()
        if self._model is None:
            return None
        try:
            vecs = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        except Exception:
            return None


semantic_embedder = SemanticEmbeddingBackend()


def semantic_pairwise_similarity(texts: List[str]) -> Optional[np.ndarray]:
    """Pairwise cosine-similarity matrix (n x n) from MiniLM embeddings, or None
    when the semantic backend is unavailable (callers fall back to TF-IDF)."""
    if not texts or not semantic_embedder.available:
        return None
    vecs = semantic_embedder.encode_batch(texts)
    if vecs is None:
        return None
    return vecs @ vecs.T


def semantic_max_similarity(query_text: str, candidates: List[str]) -> Optional[float]:
    """Highest MiniLM cosine similarity between one text and a list of candidates
    (e.g. a narration sentence vs the clean RAG corpus). None when unavailable."""
    if not candidates or not semantic_embedder.available:
        return None
    vecs = semantic_embedder.encode_batch([query_text] + candidates)
    if vecs is None:
        return None
    sims = vecs[0] @ vecs[1:].T
    return float(sims.max()) if len(sims) else None


def semantic_topic_membership(token: str, anchor_words: List[str]) -> Optional[float]:
    """
    Max MiniML cosine similarity between one narration token and a set of topic
    anchor words (topic keywords + headline/summary tokens). Single-word
    embeddings are noisy, so matching against the TOPIC'S OWN words (rather than
    a whole-sentence anchor) gives a clean signal: 'chinese'~'china' lands ~0.68,
    while off-topic filler like 'gadget' stays ~0.3. None when unavailable.
    """
    if not anchor_words or not semantic_embedder.available:
        return None
    vecs = semantic_embedder.encode_batch([token] + anchor_words)
    if vecs is None:
        return None
    return float((vecs[0] @ vecs[1:].T).max())


# Calibrated on live pipeline data: a narration sentence whose whole-sentence
# meaning ~== a clean RAG corpus sentence is a true verbatim copy at sim >= 0.94
# (the 1.00s are literal lifts). Fact-dense but legitimately rephrased narration
# — which MUST preserve names/numbers/dates and therefore can't drop below ~0.85
# no matter how well it is rewritten — clusters at 0.80-0.93, so that band is
# deliberately NOT a copy. Shared by the Observer gate and the StoryDesigner
# local dissolve pass so writer and validator agree on the same threshold.
COPY_SEMANTIC_HARD_THRESHOLD = 0.94


HIGH_RPM_TAXONOMY = {
    "finance_global": ["fed", "interest", "rate", "inflation", "bonds", "stocks", "sec", "treasury", "market", "portfolio", "wall street", "banking", "economy", "recession", "wealth"],
    "tech_global": ["semiconductor", "ai", "nvidia", "chips", "cloud", "data center", "apple", "microsoft", "google", "meta", "crypto", "tsmc", "quantum", "software", "saas"],
    "finance_tech_india": ["rbi", "nifty", "sensex", "sebi", "startup", "unicorn", "tata", "reliance", "adani", "infosys", "tcs", "gdp", "upi", "fii", "dii", "ipo", "paytm", "zerodha"],
    "legal_law": ["lawsuit", "court", "supreme court", "judge", "verdict", "prosecution", "indictment", "settlement", "regulatory", "compliance", "fine", "penalty", "attorney", "lawyer", "sued"],
    "real_estate": ["real estate", "mortgage", "housing", "home prices", "property", "rental", "homeowners", "construction", "foreclosure", "realtor"],
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
