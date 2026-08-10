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
