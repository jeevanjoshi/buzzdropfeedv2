import os
import json
import datetime
import threading
from typing import List, Dict, Any, Optional
from src.engine.logger import logger


class FeedbackMemory:
    """
    Feedback Memory Manager (Self-Correction Loop).
    Stores and retrieves successful LLM corrections from past runs to guide
    the StoryDesigner prompt using dynamic in-context learning.

    Industry standard practices applied:
    - Thread-safe access via threading.Lock()
    - Atomic file writes using temporary files and rename (os.replace)
    - Failsafe execution: never raises exceptions or crashes the pipeline
    - Semantic relevance retrieval using in-memory sentence-transformers (when available)
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.file_path = os.path.join(self.log_dir, "feedback_memory.json")
        self._lock = threading.Lock()

    def record_correction(
        self,
        topic: str,
        violation: str,
        original_text: str,
        corrected_text: str
    ) -> None:
        """
        Records a successfully corrected shot/script fragment.
        """
        try:
            with self._lock:
                history = self._load_raw()
                history.append({
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "topic": topic,
                    "violation": violation,
                    "original_text": original_text,
                    "corrected_text": corrected_text
                })
                # Cap the history size to 100 entries to prevent memory bloating
                if len(history) > 100:
                    history = history[-100:]
                self._atomic_write(history)
        except Exception as e:
            logger.warning("FEEDBACK_MEMORY", f"Failed to record correction: {e}")

    def get_relevant_feedback(self, current_topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant feedback entries for the current topic.
        If semantic embeddings are enabled, performs cosine similarity matching.
        Otherwise, falls back to the most recent entries.
        """
        try:
            with self._lock:
                history = self._load_raw()
            if not history:
                return []

            # Try to use semantic embeddings for similarity search
            try:
                from src.engine.text_embeddings import semantic_embedder
                if semantic_embedder and getattr(semantic_embedder, "model", None) is not None:
                    # Embed the current topic query
                    q_vec = semantic_embedder.encode_batch([current_topic])

                    # Extract and embed all unique past topics
                    past_topics = list(set(entry["topic"] for entry in history))
                    topic_vectors = semantic_embedder.encode_batch(past_topics)

                    # Calculate similarity score: cosine similarity (dot product)
                    scores = q_vec[0] @ topic_vectors.T

                    # Map topics to their similarity scores
                    topic_scores = {topic: float(score) for topic, score in zip(past_topics, scores)}

                    # Sort history entries by the similarity score of their topics
                    scored_entries = []
                    for entry in history:
                        score = topic_scores.get(entry["topic"], 0.0)
                        scored_entries.append((entry, score))

                    # Sort descending by score
                    scored_entries.sort(key=lambda x: x[1], reverse=True)
                    # Return top N entries
                    return [item[0] for item in scored_entries[:limit]]
            except Exception:
                pass  # Fallback to time-based sorting if embeddings fail/are disabled

            # Time-based fallback: return the most recent entries (newest first)
            return history[-limit:][::-1]
        except Exception as e:
            logger.warning("FEEDBACK_MEMORY", f"Failed to retrieve feedback: {e}")
            return []

    def _load_raw(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _atomic_write(self, data: List[Dict[str, Any]]) -> None:
        tmp_file = self.file_path + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_file, self.file_path)


feedback_memory = FeedbackMemory()
