"""Shared act-chapter timing for YouTube descriptions.

Single source of truth for the CHAPTERS block + ``chapter_timestamps`` used by
BOTH ``story_designer.generate_seo_metadata`` (design time, from shot
``duration_estimate``) and ``publisher.publish_video`` (publish time, from
measured durations). Keeping one definition here removes the static-timestamp /
regex-replacement drift that previously shipped wrong chapter times.

Behavior:
* Offsets accumulate per shot with an optional crossfade overlap subtracted
  between consecutive shots, mirroring how the master timeline is actually
  assembled.
* Act 1 always starts at 0:00.
* When shots/durations are unavailable, the static 6-entry fallback is used
  (matches the legacy layout) so descriptions are never empty/malformed.
"""

from typing import List, Tuple, Optional, Any

ACT_NAMES: List[str] = [
    "Act 1: The Inciting Incident",
    "Act 2: Historical Precedents & Origins",
    "Act 3: Deep Technical Mechanics",
    "Act 4: Actionable Real-World Impact",
    "Act 5: Critical Risks & Counter-Arguments",
    "Act 6: Strategic Future Verdict",
]

# Legacy hardcoded times, used only when no shots/durations are available.
STATIC_ACT_STARTS: List[int] = [0, 135, 270, 405, 540, 675]  # 0:00, 2:15, ...


_PROSE_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "by", "at", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those", "it", "its", "he", "she", "they", "them",
    "we", "you", "your", "our", "has", "have", "had", "do", "does", "did", "will",
    "would", "can", "could", "should", "about", "into", "over", "after", "before",
    "not", "no", "what", "which", "who", "how", "why", "when", "where", "there",
    "here", "their", "his", "her", "more", "most", "some", "any", "also", "just",
    "because", "while", "during", "between", "through", "against", "around",
    "now", "back", "still", "under", "along", "within", "across", "been", "get",
    "got", "make", "made", "take", "took", "given", "give", "come", "came",
    "since", "even", "only", "then", "than", "very", "much", "many", "like",
    "these", "those", "those", "things", "thing", "way", "ways", "part", "say",
    "says", "said", "told", "tells", "might", "must", "may", "shall", "need",
}


def _meaningful_terms(text: str, top_n: int = 3) -> List[str]:
    """Top content-bearing terms from a narration snippet, in document order.
    Sentence-openers and weak verbs are ignored so the phrase is subject-led."""
    import re as _re
    import collections

    if not text:
        return []
    words = _re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text.lower())
    # Skip the first ~5 words: sentence openers ("this is", "the case") almost
    # never carry the chapter's real subject.
    candidate = words[5:]
    meaningful = [w for w in candidate if w not in _PROSE_STOPWORDS and len(w) >= 4]
    if not meaningful:
        return []
    counts = collections.Counter(meaningful)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], candidate.index(kv[0])))
    terms = [w for w, _ in ordered[:top_n]]
    return sorted(terms, key=lambda t: candidate.index(t))


def derive_contextual_act_titles(
    shots: Optional[Any] = None,
    act_names: Optional[List[str]] = None,
) -> List[str]:
    """
    Build per-act chapter labels FROM THE ACTUAL SCRIPT content instead of the
    static generic ``ACT_NAMES``. Groups the shots by ``act_index`` and condenses
    each act's ``narration_text`` into a short topical phrase, e.g.
    "Act 3: Quantum Error Correction", so every video ships unique chapters
    that match what its narration actually covers.

    Falls back to the default ``ACT_NAMES`` when no usable shot content exists so
    descriptions are never empty/malformed.
    """
    labels = list(act_names or ACT_NAMES)
    if not shots:
        return labels

    acts: dict = {}
    for shot in shots:
        act = getattr(shot, "act_index", None)
        if not act:
            continue
        narr = " ".join(
            str(x) for x in (
                getattr(shot, "narration_text", None),
                getattr(shot, "beat_summary", None),
            ) if x
        )
        acts.setdefault(act, []).append(narr)

    out: List[str] = []
    for i in range(1, len(labels) + 1):
        narrs = acts.get(i) or []
        context = " ".join(narrs)
        generic = labels[i - 1].split(":", 1)[-1].strip()
        terms = _meaningful_terms(context)
        if len(terms) >= 2:
            # Trim back to the shoulder words so the title stays <= ~7 words.
            phrase = " ".join(terms[: len(terms)])
            out.append(f"Act {i}: {phrase.title()}")
        else:
            out.append(f"Act {i}: {generic}")
    return out


def _mmss(total_seconds: float) -> str:
    total = int(round(total_seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def compute_act_chapters(
    shots: Optional[Any] = None,
    measured_durations: Optional[List[float]] = None,
    crossfade: float = 0.0,
    act_names: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Compute per-act chapter start times.

    Args:
        shots: iterable of shot objects with ``.act_index`` (int) and
            ``.duration_estimate`` (float). May be None/empty.
        measured_durations: optional list of ACTUAL shot durations (publish time).
        crossfade: seconds of overlap subtracted between consecutive shots.
        act_names: optional 6-element list of act labels; defaults to ACT_NAMES.

    Returns:
        (chapter_lines, chapter_timestamps):
          * chapter_lines: ["m:ss - Act N: ...", ...] (for the description block)
          * chapter_timestamps: ["m:ss Act N: ...", ...] (for SEOMetadata)
    """
    labels = list(act_names or ACT_NAMES)
    starts = [0.0] * len(labels)

    if shots:
        shot_list = list(shots)
        act_starts: dict = {}
        running = 0.0
        durs = list(measured_durations or [])
        for idx, shot in enumerate(shot_list):
            act = getattr(shot, "act_index", None)
            if act is None:
                continue
            if act not in act_starts:
                act_starts[act] = running
            dur = durs[idx] if idx < len(durs) else getattr(shot, "duration_estimate", None) or 0.0
            running += dur
            if crossfade > 0 and idx < len(shot_list) - 1:
                running -= crossfade
        # Acts are 1-indexed; map onto the 0-indexed labels list.
        for i in range(1, len(labels) + 1):
            starts[i - 1] = act_starts.get(i, 0.0)
    else:
        for i, secs in enumerate(STATIC_ACT_STARTS):
            if i < len(starts):
                starts[i] = float(secs)

    # Act 1 always anchors at 0:00.
    if starts:
        starts[0] = 0.0

    chapter_lines = []
    chapter_timestamps = []
    for i, label in enumerate(labels):
        ts = _mmss(starts[i])
        chapter_lines.append(f"{ts} - {label}")
        chapter_timestamps.append(f"{ts} {label}")
    return chapter_lines, chapter_timestamps
