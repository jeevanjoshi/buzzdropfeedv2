"""Corpus-driven generic-token detector (IDF-style).

Replaces the hardcoded ``_CITATION_GENERIC_TOKENS`` denylist. A token is treated
as generic (no topical signature) when it appears across a HIGH FRACTION of a
background corpus: words that circulate through many unrelated feed/news stories
(research, researchers, model, ai, found, ...) have high document frequency and
therefore carry no topic signature.

Two layers keep it self-maintaining:

1. **In-run, over the whole feed pool.** Each pipeline run observes EVERY
   candidate topic's facts (not just the winner) plus the full retrieved corpus.
   A leak word that circulates in the day's feeds is seen many times and becomes
   generic for that run automatically.

2. **Persistent, across runs.** Document counts roll up into
   ``logs/term_register.json`` so a historically-generic word stays excluded even
   on a day where it happens to appear few times. The register grows on its own —
   no hand-curated list.

Transparency / known limit: frequency catches RECURRING / ubiquitous leak
vocabulary. A word that leaks exactly ONCE, in only the winning topic's own
snippets (never elsewhere in the feed pool), will not yet have high document
frequency — IDF has no "this was a mistake" signal. That one-shot novel case is
out of scope for a frequency model (it needs explicit failure feedback).

The module is failure-tolerant: if persistence or the background corpus is
unavailable it degrades to in-memory only (or to the small built-in stopword
fallback) and never raises.
"""

import os
import re
import json
import threading
import datetime
from typing import Dict, Set, List

# Rolling document-frequency record (auto-created, gitignored).
REGISTER_FILE_ENV = "CSVG_TERM_REGISTER"
REGISTER_FILE_DEFAULT = "logs/term_register.json"

# A token whose document frequency (fraction of observed background documents
# that contain it) is at/above this is "generic". Tuned so ubiquitous news words
# (~researchers/model/found) clear it while distinctive topic words do not.
DF_RATIO_THRESHOLD = 0.18

_TOKEN_RE = re.compile(r"[a-z][a-z0-9'-]{3,}")

# Small, safe baseline: never counted as distinctive even offline. This is a
# residual stopword safety net, NOT the primary mechanism (IDF is). It carries
# over the previously hand-tuned generic set so a fresh register (or a corpus
# too small to measure) still behaves sensibly.
_BASELINE_GENERIC = {
    "ai", "new", "model", "models", "open", "news", "first", "one", "two",
    "year", "years", "day", "week", "month", "time", "way", "make", "makes",
    "like", "world", "people", "said", "says", "say", "found", "research",
    "researchers", "study", "studies", "scientist", "scientists", "according",
    "report", "reported", "reports", "launch", "launches", "launched",
    "media", "tech", "technology", "company", "companies", "firm",
    # ── previously hardcoded _CITATION_GENERIC_TOKENS (preserved) ──────────
    "launching", "release", "releases", "released", "latest", "make",
    "world", "week", "people", "time", "way", "really", "just", "were", "was",
    "has", "have", "had", "finds", "finding", "researched", "studies",
    "suggest", "suggests", "suggested", "suggesting", "according", "experts",
    "expert", "findings", "revealed", "reveal", "reveals", "appear", "appears",
    "appeared", "previously", "realized", "rather", "followed", "follow",
    "sophisticated", "surprisingly", "surprise", "surprising", "year-old",
    "daily", "month-old", "these", "those", "being", "been", "still", "while",
    "because", "between", "during", "without",
}


class TermRegister:
    """Accumulates document-frequency term stats and reports generic tokens."""

    def __init__(self, file_path: str = None):
        self.file_path = file_path or os.getenv(REGISTER_FILE_ENV, REGISTER_FILE_DEFAULT)
        # doc_freq[token] = number of background docs containing the token
        self.doc_freq: Dict[str, int] = {}
        self.total_docs: int = 0
        self._lock = threading.Lock()
        self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.doc_freq = {str(k): int(v) for k, v in (data.get("doc_freq") or {}).items()}
                self.total_docs = int(data.get("total_docs") or 0)
        except Exception:
            self.doc_freq = {}
            self.total_docs = 0

    def _save(self) -> None:
        try:
            d = os.path.dirname(self.file_path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.file_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({
                    "total_docs": self.total_docs,
                    "doc_freq": self.doc_freq,
                    "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }, f)
            os.replace(tmp, self.file_path)
        except Exception:
            pass  # persistence is best-effort; never break the pipeline

    # ── observation ──────────────────────────────────────────────────────
    def tokens(self, text: str) -> Set[str]:
        if not text:
            return set()
        return set(_TOKEN_RE.findall((text or "").lower()))

    def observe(self, documents: List[str]) -> None:
        """Accumulate document-frequency counts from a list of text documents."""
        docs = [d for d in (documents or []) if d and d.strip()]
        if not docs:
            return
        try:
            with self._lock:
                for doc in docs:
                    toks = self.tokens(doc)
                    for t in toks:
                        self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
                    self.total_docs += 1
                self._save()
        except Exception:
            pass

    # ── query ────────────────────────────────────────────────────────────
    def generic_tokens(self, extra_docs: List[str] = None) -> Set[str]:
        """
        Return the current set of generic tokens.

        * Persistent doc-frequency (across runs) first.
        * Then, if provided, the current run's feed pool (extra_docs) is folded
          in-memory so brand-new circulating words are caught immediately.
        * Residual baseline stopwords are always excluded.
        """
        generic = set(_BASELINE_GENERIC)
        try:
            with self._lock:
                df = dict(self.doc_freq)
                total = self.total_docs
            # In-run pool: combine with the running history for this evaluation.
            if extra_docs:
                seen: Dict[str, int] = dict(df)
                n = total
                for doc in extra_docs:
                    for t in self.tokens(doc):
                        seen[t] = seen.get(t, 0) + 1
                    n += 1
                    total = n
                df = seen
            denom = max(total, 1)
            for tok, cnt in df.items():
                if cnt / denom >= DF_RATIO_THRESHOLD:
                    generic.add(tok)
        except Exception:
            pass
        return generic

    def reset(self) -> None:
        """Clear the register (tests / manual)."""
        with self._lock:
            self.doc_freq = {}
            self.total_docs = 0
        self._save()


term_register = TermRegister()
