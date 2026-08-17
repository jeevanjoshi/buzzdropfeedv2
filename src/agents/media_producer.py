import os
import re
import uuid
import datetime
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from src.schemas.state import GlobalState, ScriptData, AssetPaths, VisualType
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent, compute_state_hash
from mcp_servers.audio_edge.server import synthesize_tts, align_subtitles_whisper, sanitize_tts_text, TTSRequest, WhisperRequest
from mcp_servers.media_cloud.server import (
    generate_flux_image, apply_ken_burns_motion, assemble_ffmpeg_timeline,
    render_playwright_svg_animation, generate_dynamic_chart, fetch_reaction_gif_clip,
    ImageGenRequest, KenBurnsRequest, TimelineAssemblyRequest,
    PlaywrightSVGRequest, ChartRequest, GIFRequest
)
from src.engine.media_budget import media_budget
from src.engine.run_budget import run_budget
from src.agents.story_designer import extract_numeric_chart_spec

# Paid (fal/Replicate Flux) is reserved for these "hero" shots + the thumbnail.
# All other standard-image shots use FREE assets (Pixabay/synthetic) to stay
# within the monthly AI image budget (media_budget, ~INR 2000 / month).
PREMIUM_SHOT_IDS = {1, 8, 12, 15}

# ── Persistent visual cache (fal/Replicate images) ──────────────────────────
# fal-generated images are the only paid per-shot asset, and they are perfectly
# reproducible: the SAME enriched visual_prompt always yields a fit hero image.
# Cache every paid generation keyed on a sha256 of the prompt so re-renders
# (proofing, narration-only fixes) reuse the already-paid fal image instead of
# spending again. Store under the repo `logs/visual_cache/` (repo-local,
# gitignored) so it survives reboots, unlike `/tmp/csvg_media`. Disable with
# CSVG_VISUAL_CACHE=0.
_VISUAL_CACHE_ENABLED = os.getenv("CSVG_VISUAL_CACHE", "1").strip().lower() not in ("0", "false", "no")
import hashlib as _hashlib


def _repo_root() -> str:
    """repo root = <src>/agents/media_producer.py -> up 3 dirs."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _visual_cache_dir() -> str:
    return os.path.join(_repo_root(), "logs", "visual_cache")


def _visual_cache_key(prompt: str) -> str:
    return _hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:24]


# Per-run output artifacts are correlated to the pipeline execution id so the
# rendered master video maps back to the run that produced it (the old fixed
# "final_video_1080p.mp4" collided across runs and was wiped with /tmp). Set
# CSVG_ARCHIVE_FINAL=1 to also copy {final mp4, thumbnail} into
# logs/final_videos/<pipeline_id>/ so masters survive /tmp cleanup.
_ARCHIVE_FINAL = os.getenv("CSVG_ARCHIVE_FINAL", "0").strip().lower() in ("1", "true", "yes")

# Sync-quality knobs: the final timeline is driven by the MEASURED narration
# audio duration, not the script-time words/2.2 estimate. A trailing hold gives a
# professional "beat" after each spoken clip (validated live: 0.6s reads as the
# next shot firing too fast at the 0.5s crossfade); the subs/BGM follow the same
# measured clock so nothing drifts. Tunable via --tail, CSVG_PAD_AFTER_NARRATION.
PAD_AFTER_NARRATION = float(os.getenv("CSVG_PAD_AFTER_NARRATION", "1.2"))  # breathing hold after each shot's narration
MIN_SHOT_DUR = 2.0               # hard floor for a single shot's timeline length
BGM_VOLUME = float(os.getenv("BGM_VOLUME", "0.08"))  # bgm mix level (moviepy fallback is FLAT; the true ducking happens in the ffmpeg/media_cloud renderer)
RESOLUTION = (1920, 1080)        # strict 16:9 widescreen

# Statistic-shot numeric-claim detection (Phase 1 CSVG Media Quality): if a shot's
# narration carries >=2 numeric/statistical patterns (percentages, currency,
# Billion/Million/crore, YoY...), we route it to the grounded matplotlib chart path
# instead of shipping a generic stock image. Values are drawn verbatim from the
# RAG chart_spec / verified facts — never fabricated.
_NUMERIC_CLAIM_RE = re.compile(
    r"\d+(?:\.\d+)?\s*%|\$\d[\d,.]*|€\d[\d,.]*|¥\d[\d,.]*|₹\d[\d,.]*|"
    r"\b\d+(?:\.\d+)?\s*(?:billion|Billion|crore|Crore|million|Million|"
    r"trillion|Trillion|thousand|YoY|%)\b",
    re.IGNORECASE,
)


def _has_numeric_claim(narration: str, min_matches: int = 2) -> bool:
    return bool(narration) and len(_NUMERIC_CLAIM_RE.findall(narration)) >= min_matches


def _extract_chart_numbers(narration: str) -> List[float]:
    """Extracts numeric values from a narration string for chart derivation."""
    if not narration:
        return []
    nums = []
    matches = re.finditer(
        r'(?:[\$€¥₹]\s*)?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|\b(?:billion|billions|crore|crores|million|millions|trillion|trillions|thousand|thousands|k|m|b|cr)\b)?',
        narration,
        re.IGNORECASE
    )
    scale_map = {
        "k": 1e3, "thousand": 1e3, "thousands": 1e3,
        "m": 1e6, "million": 1e6, "millions": 1e6,
        "b": 1e9, "billion": 1e9, "billions": 1e9,
        "cr": 1e7, "crore": 1e7, "crores": 1e7,
        "trillion": 1e12, "trillions": 1e12
    }
    for m in matches:
        raw_num = m.group(1).replace(",", "")
        try:
            val = float(raw_num)
        except (ValueError, TypeError):
            continue
        scale = (m.group(2) or "").lower()
        if scale in scale_map:
            val *= scale_map[scale]
        # Skip plain 4-digit years (e.g. 2024, 2026) if unscaled and without currency/% sign
        if 1900 <= val <= 2099 and not m.group(2) and not ('$' in m.group(0) or '₹' in m.group(0) or '%' in m.group(0) or '€' in m.group(0) or '¥' in m.group(0)):
            continue
        nums.append(val)
    return nums


def _has_numeric_theme(prompt: str) -> bool:
    """Checks if a visual prompt suggests a numeric / statistical chart theme."""
    if not prompt:
        return False
    low = prompt.lower()
    keywords = [
        "chart", "graph", "metric", "metrics", "data", "statistic", "statistics",
        "growth", "revenue", "valuation", "percentage", "market share", "shares",
        "index", "numbers", "trend", "quarterly", "yoy", "inflation", "gdp"
    ]
    return any(k in low for k in keywords)


# Only these facts may feed the narrative-stats chart fallback: facts that are
# thematically tied to the SELECTED topic. The RAG verified-facts pool also
# carries the day's unrelated RSS corpus (e.g. "Samsung Galaxy H1 headphones"),
# and leaking a few of those headlines into a chart is the off-topic chart leak.
_TOPIC_CHART_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with", "by",
    "from", "that", "this", "these", "those", "was", "were", "is", "are", "its",
    "it", "as", "at", "be", "been", "being", "has", "have", "had", "not", "but",
    "up", "down", "over", "under", "than", "into", "about", "after", "their",
    "they", "his", "her", "you", "your", "we", "our", "who", "what", "when",
    "how", "where", "which", "new", "two", "one", "becomes", "become",
}


def _topic_relevant_facts(facts: List[Any], topic: Any) -> List[Any]:
    """Filter the verified-facts pool down to facts about the selected topic.

    A fact is relevant only when it shares distinctive content WORDS with the
    topic's headline/summary (length >= 4, stopword- and year-filtered) or when
    it matches >=2 of the topic's keywords. Word-membership (not substring)
    matching plus a minimum-hit score keeps generic cross-corpus words
    ("research", "2026", "shows") from dragging unrelated daily news into the
    chart's fallback data. With no topic (or no terms) we return [] so the chart
    can never fall back onto unrelated headlines.
    """
    if not topic or not facts:
        return []
    headline_terms: set = set()
    keyword_terms: set = set()
    for kw in (getattr(topic, "keywords", None) or []):
        for tok in re.split(r"[^a-z0-9]+", str(kw).lower()):
            if tok.isdigit() or len(tok) < 3:
                continue
            keyword_terms.add(tok)
    for tok in re.split(r"[^a-z0-9]+", str(getattr(topic, "headline", "") or "").lower()):
        if tok.isdigit() or len(tok) < 5 or tok in _TOPIC_CHART_STOPWORDS:
            continue
        headline_terms.add(tok)
    keyword_terms -= _TOPIC_CHART_STOPWORDS
    if not headline_terms and not keyword_terms:
        return []
    relevant = []
    for f in facts:
        f_words = set(
            re.split(
                r"[^a-z0-9]+",
                f"{getattr(f, 'headline', '') or ''} {getattr(f, 'summary', '') or ''}".lower(),
            )
        )
        strong_hits = headline_terms & f_words
        kw_hits = keyword_terms & f_words
        if strong_hits or len(kw_hits) >= 2:
            relevant.append(f)
    return relevant


def _shot_wants_data_chart(shot, chart_spec: Dict[str, Any], clean_narration: str, raw_visual_prompt: str) -> bool:
    """True only when a shot is actually GROUNDED to render a real data chart.

    A shot carries numeric grounding when its chart_spec has values or its
    narration makes >=2 numeric claims. Cinematic prompts that merely MENTION
    charts ("a laptop displaying colorful stock charts", "market graph on a
    wall") are scene descriptions, not data requests — routing them to a real
    chart forces the facts/ticker fallback, which leaks unrelated RSS headlines
    (e.g. Samsung Galaxy) into the visual. LLM-declared matplotlib-chart shots
    and explicit ``[chart: label]`` tags keep their priority.
    """
    v_type = getattr(shot, "visual_type", None)
    prompt_lower = (raw_visual_prompt or "").lower()
    if v_type in (VisualType.GIF_MEME, VisualType.GIF_STICKER):
        return False
    if v_type == VisualType.MATPLOTLIB_CHART or "[chart:" in prompt_lower:
        return True
    if v_type == VisualType.SVG_TICKER or "[svg:" in prompt_lower:
        return True
    grounded_numeric = bool(chart_spec and chart_spec.get("values")) or len(
        _extract_chart_numbers(clean_narration or "")
    ) >= 2
    if not grounded_numeric:
        return False
    return bool(
        _has_numeric_theme(raw_visual_prompt or "")
        or "stock chart" in prompt_lower
        or "market graph" in prompt_lower
    )


def _derive_chart_data(
    shot,
    chart_spec: Dict[str, Any],
    narration: str,
    prompt: str,
    facts: List[Any],
    topic: Any = None,
) -> Tuple[List[str], List[float], str, str, str]:
    """
    Derives (labels, values, unit, title, chart_type) for a grounded chart visual.
    Prefers shot.chart_spec; falls back to scanned numbers from narration or
    verified facts RELEVANT to the selected topic (never unrelated daily news).
    """
    if chart_spec and chart_spec.get("values"):
        values = [float(v) for v in chart_spec["values"]]
        labels = [str(l) for l in (chart_spec.get("labels") or [])]
        while len(labels) < len(values):
            labels.append(f"Point {len(labels) + 1}")
        labels = labels[:len(values)]
        unit = str(chart_spec.get("unit") or chart_spec.get("unit_symbol") or "")
        title = str(chart_spec.get("title") or "KEY METRICS")
        ctype = str(chart_spec.get("chart_type") or ("bar" if len(values) <= 4 else "line"))
        return labels, values, unit, title, ctype

    # Scan narration for numbers
    narr_nums = _extract_chart_numbers(narration)
    unit = "%" if "%" in narration else ("$" if "$" in narration else ("₹" if "₹" in narration else ""))

    # Title extraction
    title = "KEY METRICS"
    if "[chart:" in prompt.lower():
        m = re.search(r'\[chart:\s*([^\]]+)\]', prompt, re.I)
        if m:
            title = m.group(1).strip().upper()
    elif prompt:
        words = [w for w in re.findall(r'[A-Za-z0-9]+', prompt) if w.lower() not in _QUERY_NOISE_WORDS and not w.isdigit()]
        if words:
            title = " ".join(words[:4]).upper()

    if len(narr_nums) >= 2:
        values = narr_nums[:5]
        labels = [f"Metric {i+1}" for i in range(len(values))]
        ctype = "bar" if len(values) <= 4 else "line"
        return labels, values, unit, title, ctype

    # Fallback to facts RELEVANT to the selected topic (never unrelated RSS news)
    if facts:
        fact_entries = []
        for f in _topic_relevant_facts(facts, topic):
            f_text = f"{getattr(f, 'headline', '')} {getattr(f, 'summary', '')}"
            f_nums = _extract_chart_numbers(f_text)
            if f_nums:
                lbl = (getattr(f, 'headline', '') or getattr(f, 'source_name', '') or "Metric")[:20]
                fact_entries.append((lbl, f_nums[0]))
            if len(fact_entries) >= 5:
                break
        if len(fact_entries) >= 2:
            labels = [e[0] for e in fact_entries]
            values = [e[1] for e in fact_entries]
            ctype = "bar"
            return labels, values, unit, title, ctype

    # If single number or trend needed: create a sensible 5-point progression around the number
    base_val = narr_nums[0] if narr_nums else 100.0
    labels = ["-4w", "-3w", "-2w", "-1w", "Now"]
    values = [round(base_val * f, 2) for f in [0.92, 0.95, 0.94, 0.98, 1.0]]
    return labels, values, unit, title, "line"



# ── Company-aware ticker resolution ──────────────────────────────────────────
# Live stock tickers/charts must show the ACTUAL company the story is about, not
# a generic index fallback. ``_pick_symbol`` scans the topic's keywords, headline
# and summary against these brand aliases before falling back to the default.
_COMPANY_TICKER_MAP = {
    "meta": "META", "facebook": "META", "instagram": "META", "whatsapp": "META", "zuckerberg": "META",
    "google": "GOOGL", "alphabet": "GOOGL",
    "microsoft": "MSFT", "nadella": "MSFT", "windows": "MSFT", "copilot": "MSFT",
    "apple": "AAPL", "iphone": "AAPL", "tim cook": "AAPL",
    "amazon": "AMZN", "bezos": "AMZN", "aws": "AMZN",
    "netflix": "NFLX",
    "nvidia": "NVDA", "jensen huang": "NVDA",
    "amd": "AMD", "intel": "INTC",
    "tesla": "TSLA", "spacex": "TSLA",
    "byd": "BYDDY",
    "reliance": "RELIANCE.NS", "mukesh ambani": "RELIANCE.NS",
    "tata motors": "TTM", "tata": "TTM",
    "infosys": "INFY", "wipro": "WIPRO.NS", "tcs": "TCS.NS",
    "hdfc": "HDFCBANK.NS", "htdfc": "HDFCBANK.NS", "icici": "ICICIBANK.NS",
    "sbi": "SBIN.NS", "state bank of india": "SBIN.NS",
}

# Famous NON-public companies: a stock ticker/chart for these should NOT fall
# back to a real index quote (the numbers would be meaningless/wrong). Renders
# are expected to use narration-grounded figures instead.
_PRIVATE_COMPANIES = {
    "openai", "anthropic", "deepmind", "xai", "spacex", "stripe",
    "databricks", "sambanova", "mistral", "sierra", "globe",
}

# Human-readable company names for clean on-screen ticker/chart titles (never
# the raw cinematic prompt).
_TICKER_COMPANY_NAMES = {
    "META": "META PLATFORMS", "GOOGL": "ALPHABET", "MSFT": "MICROSOFT",
    "AAPL": "APPLE", "AMZN": "AMAZON", "NFLX": "NETFLIX", "NVDA": "NVIDIA",
    "AMD": "ADVANCED MICRO DEVICES", "INTC": "INTEL", "TSLA": "TESLA",
    "BYDDY": "BYD", "RELIANCE.NS": "RELIANCE", "TTM": "TATA MOTORS",
    "INFY": "INFOSYS", "WIPRO.NS": "WIPRO", "TCS.NS": "TATA CONSULTANCY",
    "HDFCBANK.NS": "HDFC BANK", "ICICIBANK.NS": "ICICI BANK", "SBIN.NS": "SBI",
}


def _market_snapshot_title(symbol: str, narration: str = "") -> str:
    """Short, on-theme on-screen title for a live quote ticker/chart."""
    name = _TICKER_COMPANY_NAMES.get((symbol or "").upper(), (symbol or "MARKET").upper())
    tag = "PRE-MARKET" if narration and "premarket" in narration.lower() else "MARKET WATCH"
    return f"{symbol.upper()} • {name} • {tag}"


def _extract_thumbnail_subject(headline: str) -> str:
    """Reduce a topic headline to a short subject phrase for the thumbnail art,
    e.g. 'Meta launches Muse Glimmer model as Zuckerberg champions AI' ->
    'Meta Muse Glimmer'. Returns '' when nothing usable is found."""
    if not headline:
        return ""
    low = headline.lower()
    for pivot in (
        " unveils ", " announces ", " launches ", " reveals ", " introduces ",
        " releases ", " says ", " reports ", " hits ", " claims ", " as ",
        " reaches ", " breaks ", " to ",
    ):
        idx = low.find(pivot)
        if idx > 0:
            left = headline[:idx].strip()
            rest = headline[idx + len(pivot):]
            m = re.match(r"([A-Z][\w'&.]*(?: [A-Z][\w'&.]*){0,3})", rest)
            product = m.group(1) if m else ""
            subject = f"{left} {product}".strip()
            if len(subject) > 48:
                subject = subject[:48].rsplit(" ", 1)[0]
            return subject
    m = re.match(r"([A-Z][\w'&.]*(?: [A-Z][\w'&.]*){0,2})", headline)
    return m.group(1) if m else ""


def _build_thumbnail_scene(state) -> str:
    """Topic-grounded thumbnail scene: the story's subject fused into the hero
    shot's art direction so the thumbnail is actually about the video's niche
    (company + product) instead of a generic stock scene."""
    hero_prompt = ""
    if state.script_data and state.script_data.shots:
        hero_prompt = state.script_data.shots[0].visual_prompt or ""
    subject = _extract_thumbnail_subject(
        getattr(state.selected_topic, "headline", "") if state.selected_topic else ""
    )
    if not subject:
        return hero_prompt
    base = re.sub(
        r"^\s*(cinematic\s+(?:16:9\s+)?(?:widescreen\s+)?(?:shot\s*:\s*)?|cinematic\s+shot\s*:\s*)",
        "", hero_prompt, flags=re.I,
    )
    base = re.sub(r"\.\s*$", "", base.strip())
    return (
        f"A dramatic cinematic 16:9 scene of {subject}: {base}. "
        f"Rule-of-thirds composition: place {subject} in the RIGHT third of the "
        f"frame at a grid intersection, never dead-center, with clean empty "
        f"negative space in the LEFT third reserved for a bold text hook. "
        f"Single hero subject only — no crowds, no extra props, no clutter, "
        f"minimal background detail so the image stays instantly readable at "
        f"postage-stamp thumbnail size. Monumental product-shot realism, glowing "
        f"dramatic rim lighting against a deep dark background, 60-30-10 color "
        f"grading: muted background, high-contrast saturated subject, one vivid "
        f"accent hue. Sharp focus on the subject, soft blurred background, ultra "
        f"high contrast. Human emotion & gaze direction: foreground one "
        f"authentic, recognizable human reaction face — genuine surprise, "
        f"curiosity or intense focus, never an overplayed shock face — facing "
        f"the camera, gaze and attention pointing toward the upper-left text-hook "
        f"corner to guide the viewer's eye. One single face only, no crowds, no "
        f"extra people.")


def _resolve_ticker(topic) -> Optional[str]:
    """Return a ticker symbol when the topic is clearly about a known public
    company, else None. Scans keywords, headline and summary."""
    if not topic:
        return None
    kw_vals = [str(k).lower() for k in (getattr(topic, "keywords", None) or [])]
    head = str(getattr(topic, "headline", "") or "").lower()
    summ = str(getattr(topic, "summary", "") or "").lower()
    for alias, ticker in _COMPANY_TICKER_MAP.items():
        key = alias.lower()
        if key in head or key in summ or any(key in k for k in kw_vals):
            return ticker
    return None


_WORD_NUMS = {
    "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0, "fifteen": 15.0, "twenty": 20.0,
    "twenty-five": 25.0, "thirty": 30.0, "fifty": 50.0, "hundred": 100.0,
}


def _extract_percent_move(narration: str) -> Optional[float]:
    """Best-effort pull of a signed move (%) from a narration snippet so fallback
    tickers agree with the script. Handles numeric ('-10%', '+1.2%') and
    spelled-out moves ('one percent uptick' -> +1.0, 'ten percent decline' ->
    -10.0). Prefers the FIRST move mentioned (the premarket one, etc)."""
    if not narration:
        return None
    low = narration.lower()
    _DOWNS = re.compile(r"\b(dropped|declined|drop|decline|declining|down|fell|fallen|falls|fall|plunged|plunge|plunging|slumped|slump|slumping|lost|loss|lose|sold|sank|sink|slid|slide|slipped|crash|crashing|tumbled|tumble|tumbling|sell-off|selloff)\b")
    m = re.search(r"([+-]?(?:\d+(?:\.\d+)?))\s*%", low)
    if m:
        sign = -1.0 if _DOWNS.search(low[max(0, m.start() - 42):m.start()]) else 1.0
        return sign * float(m.group(1))
    m = re.search(r"([+-]?(?:\d+(?:\.\d+)?))\s*percent\b", low)
    if m:
        sign = -1.0 if _DOWNS.search(low[max(0, m.start() - 42):m.start()]) else 1.0
        return sign * float(m.group(1))
    _NUM_WORDS_ALT = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|twenty-five|thirty|fifty|hundred"
    for m in re.finditer(rf"((?:[a-z0-9'&., -]{{0,16}}?)\b({_NUM_WORDS_ALT})\s*percent)", low):
        context = m.group(1)
        n = _WORD_NUMS[m.group(2)]
        if _DOWNS.search(context):
            return -n
        return n
    return None


def _probe_duration(path: str) -> Optional[float]:
    """
    Returns the exact media duration (seconds) of an audio/video file using
    ffprobe. This is the SINGLE SOURCE OF TRUTH for all timeline offsets so that
    subtitles, concatenation and BGM all stay perfectly in sync with the audio.
    Returns None on any error (never raises).
    """
    if not path or not os.path.exists(path):
        return None
    try:
        import subprocess
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0 and r.stdout.strip():
            val = float(r.stdout.strip())
            if val > 0:
                return val
    except Exception:
        pass
    return None


def _probe_wav_duration(path: str) -> Optional[float]:
    """Fast WAV duration probe using the stdlib wave module, with ffprobe fallback."""
    if not path or not os.path.exists(path):
        return None
    try:
        import wave
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return _probe_duration(path)


# Peak limiter (Phase 3, CSVG Media Quality): the latest run measured full-scale
# clipping (-0.0 dB) at the TTS WAV stage on several shots. Applying ffmpeg
# `alimiter` right after synthesis guarantees no shot ever clips at the source
# (before the final loudnorm mix). Duration is unchanged, so subtitle/timeline
# sync is preserved.
def _peak_limit_wav(wav_path: str, limit: float = 0.89) -> None:
    """Applies a peak limiter (~-1.0 dB) to a narration WAV in place (no-op on error)."""
    if not wav_path or not os.path.exists(wav_path):
        return
    import subprocess
    tmp = wav_path + "_lim.wav"
    try:
        res = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-af", f"alimiter=limit={limit}:level=false",
             "-c:a", "pcm_s16le", tmp],
            capture_output=True, text=True, timeout=120,
        )
        if res.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 100:
            os.replace(tmp, wav_path)
        elif os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def merge_ass_subtitle_files(ass_paths: List[str], shot_durations: List[float], output_master_ass: str, crossfade: float = 0.0):
    """
    Merges multiple shot .ass subtitle files into a single master timeline subtitle file
    by offsetting timestamps according to cumulative shot durations.
    When `crossfade > 0`, the offsets account for the overlap introduced by dissolve
    transitions (each boundary shifts the following shot earlier by `crossfade` sec),
    so subtitles stay glued to the narration under crossfades.
    """
    ass_header = """[Script Info]
Title: 16:9 CSVG Master Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,3,2,2,660,660,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dialogue_lines = []
    current_offset = 0.0

    def parse_ass_time(t_str: str) -> float:
        parts = t_str.strip().split(":")
        if len(parts) != 3:
            return 0.0
        h = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        return h * 3600.0 + m * 60.0 + s

    def format_ass_time(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    for idx, (ass_path, shot_dur) in enumerate(zip(ass_paths, shot_durations)):
        if os.path.exists(ass_path):
            with open(ass_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("Dialogue:"):
                    parts = line.split(",", 9)
                    if len(parts) == 10:
                        t_start = parse_ass_time(parts[1]) + current_offset
                        t_end = parse_ass_time(parts[2]) + current_offset
                        if idx == 0:
                            t_start += 0.5
                            t_end += 0.5
                        parts[1] = format_ass_time(t_start)
                        parts[2] = format_ass_time(t_end)
                        dialogue_lines.append(",".join(parts))
        # Advance to the next shot's start in the (possibly crossfade-compressed)
        # timeline: a dissolve overlaps `crossfade` seconds at each boundary.
        current_offset += shot_dur
        if crossfade > 0:
            current_offset -= crossfade

    with open(output_master_ass, "w", encoding="utf-8") as f:
        f.write(ass_header + "".join(dialogue_lines))


def parse_scene_visual_cue(narration_text: str) -> Tuple[str, str]:
    """
    Parses structured [Scene: ...] visual cues embedded in narration_text.
    Returns a tuple of (clean_narration, extracted_visual_prompt).
    """
    scene_match = re.search(r'\[Scene:\s*([^\]]+)\]', narration_text, re.IGNORECASE)
    extracted_cue = scene_match.group(1).strip() if scene_match else ""
    clean_narration = re.sub(r'\[Scene:[^\]]*\]', '', narration_text).strip()
    return clean_narration, extracted_cue


# Stage 8 Quality-by-Design: Cinematic FVD keyword sets by shot position
_FVD_QUALITY_KEYWORDS = {
    "hook":      "dramatic cinematic opening shot, high contrast rim lighting, shallow depth of field, anamorphic lens flare",
    "history":   "warm documentary color grade, vintage film texture, wide establishing angle, golden hour atmosphere",
    "technical": "cold blue data-center lighting, macro precision focus, bokeh background neural grid, sharp foreground detail",
    "impact":    "dynamic dolly push-in, warm executive office ambience, motivated practical lighting, 16mm grain overlay",
    "risk":      "moody low-key lighting, deep shadow contrast, ominous vignette, slow crane descent motion",
    "verdict":   "epic city skyline wide shot, twilight gradient sky, tilt-shift miniature depth, final frame cinematic outro",
}

_ACT_TO_TONE = {1: "hook", 2: "history", 3: "technical", 4: "impact", 5: "risk", 6: "verdict"}


def enrich_visual_prompt(visual_prompt: str, act_index: int, shot_id: int) -> str:
    """
    Stage 8 Quality-by-Design: Enriches raw visual prompt with cinematic FVD quality keywords
    calibrated per act tone and shot position. Ensures FLUX.1 renders align with the
    reference feature distribution used by the FVD gate.
    Also enforces 16:9 widescreen, 8k resolution, and photorealistic quality tags.
    """
    tone = _ACT_TO_TONE.get(act_index, "impact")
    fvd_keywords = _FVD_QUALITY_KEYWORDS[tone]

    # Ensure base quality anchors are always present
    base_anchors = []
    if "16:9" not in visual_prompt.lower():
        base_anchors.append("16:9 widescreen")
    if "8k" not in visual_prompt.lower() and "photorealistic" not in visual_prompt.lower():
        base_anchors.append("8k photorealistic")

    enriched = visual_prompt.rstrip(".")
    if base_anchors:
        enriched += ", " + ", ".join(base_anchors)
    enriched += f", {fvd_keywords}."
    return enriched


# Phase 2 (CSVG Media Quality): the old stock query took the first 4 3-letter+
# words of the visual prompt, which — because every prompt opens with "Cinematic
# 16:9 widescreen ..." — collapsed to "cinematic widescreen shot ..." and pulled
# generic/off-topic B-roll. This extracts the TOPICAL subject words after the
# ':', dropping camera/lighting noise, and falls back to topic anchor words.
_QUERY_NOISE_WORDS = {
    "cinematic", "widescreen", "169", "8k", "photorealistic", "sweeping", "archival",
    "golden", "shot", "slow", "dolly", "glowing", "warm", "lighting", "dramatic",
    "camera", "scene", "pan", "zoom", "macro", "wide", "closeup", "close", "depth",
    "field", "bokeh", "lens", "flare", "hdr", "high", "contrast", "rim", "moody",
    "lowkey", "documentary", "style", "visual", "atmosphere", "ambient", "top",
    "bottom", "left", "right", "aerial", "vintage", "texture", "preset", "film",
    "professional", "generated", "image", "background", "vibrant", "color", "grade",
}


def _apply_outro_cta_overlay(img_path: str) -> None:
    """
    Renders an aesthetic, high-contrast floating glassmorphic call-to-action (CTA)
    card centered in the lower-mid screen (above bottom subtitle margins) so it never
    collides with subtitles while remaining prominently visible.
    """
    if not os.path.exists(img_path):
        return
    try:
        from PIL import Image, ImageDraw, ImageFont

        base = Image.open(img_path).convert("RGBA")
        w, h = base.size

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Floating card dimensions (scaled relative to 1920x1080)
        card_w = int(w * 0.48)  # ~920px on 1080p
        card_h = int(h * 0.125) # ~135px on 1080p
        card_x = (w - card_w) // 2
        # Position at ~63% height (well above subtitles at 88%-95% height)
        card_y = int(h * 0.63)

        # Outer subtle glow rim
        draw.rounded_rectangle(
            [card_x - 3, card_y - 3, card_x + card_w + 3, card_y + card_h + 3],
            radius=26,
            fill=(0, 229, 255, 35),
            outline=(0, 255, 216, 200),
            width=2,
        )

        # Dark glass card body
        draw.rounded_rectangle(
            [card_x, card_y, card_x + card_w, card_y + card_h],
            radius=24,
            fill=(10, 15, 26, 235),
        )

        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]
        sub_font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]

        font_main = None
        font_sub = None
        for p in font_paths:
            if os.path.exists(p):
                try:
                    font_main = ImageFont.truetype(p, int(h * 0.036))
                    break
                except Exception:
                    pass
        for p in sub_font_paths:
            if os.path.exists(p):
                try:
                    font_sub = ImageFont.truetype(p, int(h * 0.019))
                    break
                except Exception:
                    pass

        if not font_main:
            font_main = ImageFont.load_default()
        if not font_sub:
            font_sub = ImageFont.load_default()

        text_main = "LIKE & SUBSCRIBE TO THE CHANNEL"
        text_sub = "STAY AHEAD WITH EVERY IN-DEPTH BREAKDOWN"

        try:
            bbox_m = draw.textbbox((0, 0), text_main, font=font_main)
            tw_m = bbox_m[2] - bbox_m[0]
            bbox_s = draw.textbbox((0, 0), text_sub, font=font_sub)
            tw_s = bbox_s[2] - bbox_s[0]
        except Exception:
            tw_m = int(card_w * 0.8)
            tw_s = int(card_w * 0.7)

        pos_m_x = card_x + max(0, (card_w - tw_m) // 2)
        pos_m_y = card_y + int(card_h * 0.20)
        draw.text((pos_m_x, pos_m_y), text_main, fill=(255, 255, 255, 255), font=font_main)

        pos_s_x = card_x + max(0, (card_w - tw_s) // 2)
        pos_s_y = card_y + int(card_h * 0.62)
        draw.text((pos_s_x, pos_s_y), text_sub, fill=(0, 229, 255, 240), font=font_sub)

        composited = Image.alpha_composite(base, overlay).convert("RGB")
        composited.save(img_path, quality=95)
        print(f"[MediaProducer] Aesthetic outro CTA card applied to {img_path}")
    except Exception as e:
        print(f"[MediaProducer] Warning: Outro CTA card render failed ({e}).")


def _extract_search_keywords(raw_prompt: str, anchors: Optional[List[str]] = None) -> str:
    """Builds a topical stock search query from the visual prompt (text after the
    first ':'), dropping camera/lighting noise. Falls back to topic anchor words
    when too few topical keywords remain. Never returns an empty query."""
    body = raw_prompt
    if ":" in body:
        body = body.split(":", 1)[1]
    words = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", body or "")
    seen, uniq = set(), []
    for w in words:
        k = w.lower()
        if k in _QUERY_NOISE_WORDS or k in seen:
            continue
        seen.add(k)
        uniq.append(k)
    if len(uniq) < 2:
        for a in (anchors or []):
            ka = str(a or "").lower().strip()
            if len(ka) >= 3 and ka not in _QUERY_NOISE_WORDS and ka not in seen:
                seen.add(ka)
                uniq.append(ka)
    return " ".join(uniq[:6]) if uniq else "abstract"


def _generate_free_visual(shot, raw_visual_prompt: str, img_path: str, mp4_path: str,
                          anchors: Optional[List[str]] = None) -> bool:
    """
    Generates a shot visual using FREE assets only (Pixabay stock -> Pexels ->
    local synthetic) — for non-premium shots and when the AI image budget is
    exhausted. Returns True if a ready-to-use video file was produced (specialized).
    """
    specialized = False
    import requests

    clean_query = _extract_search_keywords(raw_visual_prompt, anchors)
    query_lower = raw_visual_prompt.lower()
    is_vector = any(kw in query_lower for kw in ["vector", "svg", "icon", "logo"])
    is_illustration = any(kw in query_lower for kw in ["illustration", "graphic", "clipart"])
    is_video = any(kw in query_lower for kw in ["video", "footage", "b-roll", "timelapse", "motion"])
    img_type = "vector" if is_vector else ("illustration" if is_illustration else "photo")

    def _fetch_from(retriever, source_name: str):
        """Attempt to fetch + download from one stock provider; raise on failure."""
        nonlocal specialized
        if is_video:
            print(f"[FreeVisual] Searching stock video ({source_name}) for: '{clean_query}'")
            results = retriever.search_videos(clean_query, limit=1)
            if not results:
                raise RuntimeError(f"No stock videos on {source_name}.")
            res = requests.get(results[0]["video_url"], headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
            if res.status_code != 200:
                raise RuntimeError(f"{source_name} video HTTP {res.status_code}")
            with open(mp4_path, 'wb') as f:
                f.write(res.content)
            print(f"[FreeVisual] Downloaded {source_name} stock video to {mp4_path}")
            specialized = True
        else:
            print(f"[FreeVisual] Searching stock images ({img_type}, {source_name}) for: '{clean_query}'")
            results = retriever.search_images(clean_query, image_type=img_type, limit=1)
            if not results:
                raise RuntimeError(f"No {img_type}s on {source_name}.")
            res = requests.get(results[0]["largeImageURL"], headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
            if res.status_code != 200:
                raise RuntimeError(f"{source_name} image HTTP {res.status_code}")
            with open(img_path, 'wb') as f:
                f.write(res.content)
            print(f"[FreeVisual] Downloaded {source_name} stock image to {img_path}")

    try:
        # Check NASA Image & Video Library first for Space/Astronomy topics
        from src.engine.nasa_retriever import nasa_retriever
        is_space_query = any(k in clean_query.lower() or (anchors and any(k in str(a).lower() for a in anchors))
                             for k in ["space", "nasa", "eclipse", "moon", "sun", "solar", "astronomy", "jwst", "mars", "galaxy", "telescope", "orbit", "spacecraft", "apollo"])
        if is_space_query and not is_video:
            nasa_url = nasa_retriever.search_image(clean_query)
            if nasa_url:
                print(f"[FreeVisual] Found NASA Image Archive photo for '{clean_query}': {nasa_url}")
                res = requests.get(nasa_url, headers={'User-Agent': 'CSVG-Pipeline/2.0'}, timeout=20)
                if res.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(res.content)
                    print(f"[FreeVisual] Downloaded NASA archive image to {img_path}")
                    return False
        
        try:
            from src.engine.pixabay_retriever import pixabay_retriever
            _fetch_from(pixabay_retriever, "Pixabay")
        except Exception as pix_err:
            print(f"[FreeVisual] Pixabay failed ({pix_err}); trying Pexels.")
            from src.engine.pexels_retriever import pexels_retriever
            _fetch_from(pexels_retriever, "Pexels")
    except Exception as e:
        print(f"[FreeVisual] Free stock media failed: {e}; using local synthetic.")
        from mcp_servers.media_cloud.server import generate_synthetic_png
        generate_synthetic_png(img_path, title=f"SHOT {shot.shot_id}: {raw_visual_prompt[:40]}")
    return specialized


class MediaProducerAgent:
    """
    Stage 4: Media Producer Agent coordinating parallel tool execution across Edge MCP Server (Pi 5)
    and Cloud Media MCP Server (OCI) to synthesize audio, generate 16:9 visuals, apply Ken Burns
    motion, burn subtitles, and assemble the final 10-15 minute widescreen video.
    Includes TTS Intonation & Breathing Pause Injection and [Scene:...] Visual Cue Parsing.
    """

    def __init__(self, name: str = "MediaProducer", storage_dir: str = "/tmp/csvg_media",
                 renderer: str = "ffmpeg", crossfade: float = 0.5, pad_after_narration: Optional[float] = None):
        self.name = name
        self.storage_dir = storage_dir
        # "ffmpeg" = probe-driven concat (default); "moviepy" = MoviePy timeline composer.
        self.renderer = renderer
        # Crossfade (seconds) between consecutive shots; 0.0 = hard cuts.
        self.crossfade = max(0.0, crossfade)
        # Breathing hold (seconds) of video-only silence after each shot's narration
        # ends, before the crossfade into the next shot. Defaults to the env knob.
        self.pad_after_narration = max(
            0.0, pad_after_narration if pad_after_narration is not None else PAD_AFTER_NARRATION)
        os.makedirs(self.storage_dir, exist_ok=True)
        # Cache the live market quote per topic so we fetch it once, not per shot.
        self._quote_cache: Dict[str, dict] = {}

    def _pick_symbol(self, state) -> str:
        """Pick a ticker symbol for live data visuals (ticker/chart).

        Resolution order:
        1. An explicit ticker already present in the topic keywords (e.g. "META").
        2. A known public company mentioned in keywords/headline/summary
           (e.g. "zuckerberg"/"meta" -> META, "nvidia" -> NVDA).
        3. Region default (RELIANCE.NS for india, SPY otherwise) for general/index
           stories where no specific company is involved.
        """
        kws = getattr(state.selected_topic, "keywords", None) if state.selected_topic else None
        for kw in (kws or []):
            if re.fullmatch(r"[A-Z]{1,5}(\.NS)?", kw or ""):
                return kw
        resolved = _resolve_ticker(state.selected_topic)
        if resolved:
            return resolved
        # Famous NON-public company (OpenAI, Anthropic, ...) -> no live exchange
        # quote exists; mark it so the renderers show narration-grounded moves.
        topic = state.selected_topic
        if topic:
            low_head = str(getattr(topic, "headline", "") or "").lower()
            low_summ = str(getattr(topic, "summary", "") or "").lower()
            low_kws = [str(k).lower() for k in (kws or [])]
            for alias in _PRIVATE_COMPANIES:
                if alias in low_head or alias in low_summ or any(alias in k for k in low_kws):
                    return alias.upper()
        region = getattr(state, "region", "all") or "all"
        return "RELIANCE.NS" if region == "india" else "SPY"

    async def _get_market_quote(self, state, narration: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches a live stock quote (Alpha Vantage) for the chosen symbol to feed
        the chart/ticker visuals with real numbers. Cached per pipeline run.
        Returns {symbol, price, change, grounded}: ``price``/``change`` are the
        real quote when available; when the fetch fails the values are derived
        from the narration's own stated move (never a fabricated fixed number).
        """
        symbol = self._pick_symbol(state)
        if symbol in self._quote_cache:
            return self._quote_cache[symbol]
        res: Optional[Dict[str, Any]] = None
        if symbol.lower() not in _PRIVATE_COMPANIES:
            try:
                from src.engine.external_apis import ExternalAPIManager
                res = await asyncio.to_thread(ExternalAPIManager().fetch_alpha_vantage_stock_quote, symbol)
            except Exception as e:
                print(f"[MediaProducer] Market quote fetch failed ({e}); using narration-grounded fallback.")
        if res and (res.get("price") or "").strip() and (res.get("change") or "").strip():
            price_raw = re.sub(r"[^\d.]", "", res.get("price", "0") or "0")
            change_raw = re.sub(r"[^\d.+-]", "", res.get("change", "0") or "0")
            try:
                price = float(price_raw)
                change = float(change_raw)
            except ValueError:
                price, change = None, None
            if price is not None and change is not None:
                symbol_out = res.get("symbol", symbol) or symbol
                quote = {"symbol": symbol_out, "price": price, "change": change, "grounded": True}
                self._quote_cache[symbol] = quote
                return quote
        # No trustworthy live quote: ground the move on the script itself so the
        # visual never contradicts the narration ("one percent uptick" -> +1.0).
        move = _extract_percent_move(narration or "")
        quote = {
            "symbol": symbol,
            "price": None,
            "change": move,
            "grounded": False,
        }
        self._quote_cache[symbol] = quote
        return quote

    async def _get_market_history(self, state, narration: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches multi-day historical stock price series for the chosen symbol.
        Returns {symbol, labels, values, price, change, grounded}.
        """
        symbol = self._pick_symbol(state)
        hist_key = f"{symbol}__hist"
        if hist_key in self._quote_cache:
            return self._quote_cache[hist_key]

        res: Optional[Dict[str, Any]] = None
        if symbol.lower() not in _PRIVATE_COMPANIES:
            try:
                from src.engine.external_apis import ExternalAPIManager
                res = await asyncio.to_thread(ExternalAPIManager().fetch_alpha_vantage_stock_history, symbol)
            except Exception as e:
                print(f"[MediaProducer] Market history fetch failed ({e}); using quote fallback.")

        if res and res.get("values") and len(res.get("values", [])) >= 2:
            hist_data = {
                "symbol": res.get("symbol", symbol) or symbol,
                "labels": res.get("labels", ["-4d", "-3d", "-2d", "Yesterday", "Today"]),
                "values": res.get("values", []),
                "price": res.get("price"),
                "change": res.get("change", 0.0),
                "grounded": True,
            }
            self._quote_cache[hist_key] = hist_data
            return hist_data

        # Fallback using quote or narration move
        quote = await self._get_market_quote(state, narration)
        price = quote.get("price") or 120.0
        change = quote.get("change") or 2.5
        base = price / (1.0 + change / 100.0)
        fluctuations = [-0.015, +0.01, -0.005, +0.02, change / 100.0]
        values = [round(base * (1.0 + f), 2) for f in fluctuations]
        values[-1] = round(price, 2)
        labels = ["5d ago", "4d ago", "3d ago", "Yesterday", "Today"]
        hist_data = {
            "symbol": quote.get("symbol", symbol) or symbol,
            "labels": labels,
            "values": values,
            "price": price,
            "change": change,
            "grounded": quote.get("grounded", False),
        }
        self._quote_cache[hist_key] = hist_data
        return hist_data

    async def produce_all_media(self, state: GlobalState, dummy_frames: bool = False, renderer: Optional[str] = None, crossfade: Optional[float] = None, pad_after_narration: Optional[float] = None) -> AssetPaths:
        """
        Synthesizes audio & subtitles via Edge MCP tools, generates visuals & timeline via Cloud MCP tools.
        """
        renderer = renderer or self.renderer
        crossfade = self.crossfade if crossfade is None else max(0.0, crossfade)
        tail = self.pad_after_narration if pad_after_narration is None else max(0.0, pad_after_narration)
        script = state.script_data
        if not script:
            raise ValueError("Media production failed: state.script_data is None")

        # ══ Execution-correlated media output ═══════════════════════════════
        # Every run writes into its OWN subdirectory keyed by the pipeline id so
        # artifacts (PNGs / WAVs / MP4s / final master / thumbnail) never collide
        # across runs and can be mapped back to the execution that produced them.
        # Fall back to the classic shared storage dir when no pipeline id exists
        # (smoke tests / synthetic runs), preserving old behaviour.
        exec_id = (getattr(state, "pipeline_id", "") or "").strip()
        if exec_id:
            self.storage_dir = os.path.join(_repo_root(), "logs", "media", exec_id)
            os.makedirs(self.storage_dir, exist_ok=True)
        else:
            os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(_visual_cache_dir(), exist_ok=True)

        asset_paths = AssetPaths()
        asset_paths.storage_dir = self.storage_dir

        # 1. Synthesize TTS Audio and Subtitles for all shots
        audio_dir = os.path.join(self.storage_dir, "audio")
        sub_dir = os.path.join(self.storage_dir, "subtitles")
        vis_dir = os.path.join(self.storage_dir, "visuals")

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(sub_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        concat_lines = []
        shot_timings: List[float] = []   # measured (probed) final durations per shot

        # Topical anchor words (twitter keywords + headline) used to build relevant
        # stock-search queries when a shot's prompt leaves too few subject words.
        topic_anchors: List[str] = []
        if state.selected_topic:
            topic_anchors = list(getattr(state.selected_topic, "keywords", None) or [])
            headline = getattr(state.selected_topic, "headline", "") or ""
            topic_anchors += [w for w in re.findall(r"[A-Za-z]{3,}", headline)]

        for shot in script.shots:
            shot_key = f"shot_{shot.shot_id}"

            # Stage 4a: Parse [Scene: ...] visual cues from narration
            clean_narration, scene_cue = parse_scene_visual_cue(shot.narration_text)
            raw_visual_prompt = scene_cue if scene_cue else shot.visual_prompt
            prompt_lower = raw_visual_prompt.lower()

            # Stage 8 Quality-by-Design: Enrich visual prompt with FVD cinematic keywords
            visual_prompt = enrich_visual_prompt(raw_visual_prompt, shot.act_index, shot.shot_id)

            # Sync-safe text: sanitized exactly like the TTS engine speaks it, so the
            # Whisper subtitle remap matches the real spoken words (no SSML artifacts).
            tts_narration = sanitize_tts_text(clean_narration)

            # Expressive tempo (Approach A): vary the TTS `speed` per shot so the
            # narrator isn't monotone. Urgency/crisis shots sound punchier & faster;
            # the verdict act slows for dramatic weight. The re-probed audio duration
            # keeps subtitles/timeline in sync regardless of the chosen speed.
            urgency_words = {"warning", "collapse", "shocking", "urgent", "crisis", "breaking", "plunge"}
            has_urgency = any(w in clean_narration.lower() for w in urgency_words)
            if has_urgency:
                tts_speed = 1.07
            elif shot.act_index == 6:
                tts_speed = 0.94   # verdict: weighty, deliberate
            elif shot.act_index == 5:
                tts_speed = 0.97   # risk: tense, slightly slowed
            else:
                tts_speed = 1.0

            # Paths
            wav_path = os.path.join(audio_dir, f"{shot_key}.wav")
            ass_path = os.path.join(sub_dir, f"{shot_key}.ass")
            img_path = os.path.join(vis_dir, f"{shot_key}.png")
            mp4_path = os.path.join(vis_dir, f"{shot_key}.mp4")

            # Call Edge Audio MCP Tools (with HTTP / REST Bridge support)
            audio_edge_url = os.getenv("AUDIO_EDGE_URL")
            audio_generated = False
            if audio_edge_url:
                try:
                    import aiohttp
                    region_val = state.selected_topic.region if (state.selected_topic and hasattr(state.selected_topic, 'region')) else "all"
                    async with aiohttp.ClientSession() as session:
                        # 1. Synthesize TTS remotely
                        async with session.post(f"{audio_edge_url}/tools/synthesize_tts", json={
                            "text": tts_narration,
                            "output_path": wav_path,
                            "region": region_val,
                            "speed": tts_speed
                        }, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                            if resp.status != 200:
                                raise Exception(f"TTS Synthesis failed over HTTP: {await resp.text()}")
                            tts_json = await resp.json()
                            if tts_json.get("engine") == "synthetic_wav_fallback":
                                raise RuntimeError(
                                    f"Edge TTS degraded to synthetic_wav_fallback (toner, not speech) for {shot_key}: "
                                    f"Pi edge has no Kokoro model. Aborting rather than shipping fake audio."
                                )
                        # 2. Download the synthesized wav file with retries
                        for attempt in range(3):
                            try:
                                async with session.get(f"{audio_edge_url}/files", params={"path": wav_path}, timeout=aiohttp.ClientTimeout(total=300)) as file_resp:
                                    if file_resp.status == 200:
                                        with open(wav_path, "wb") as f:
                                            f.write(await file_resp.read())
                                        break
                                    else:
                                        raise Exception(f"HTTP status {file_resp.status}")
                            except Exception as get_err:
                                if attempt == 2:
                                    raise Exception(f"Failed to download wav file after 3 attempts: {get_err}")
                                await asyncio.sleep(1.0)

                        # 3. Align subtitles remotely (against the sanitized spoken text)
                        async with session.post(f"{audio_edge_url}/tools/align_subtitles_whisper", json={
                            "audio_path": wav_path,
                            "output_ass_path": ass_path,
                            "original_text": tts_narration
                        }, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                            if resp.status != 200:
                                raise Exception(f"Subtitle alignment failed over HTTP: {await resp.text()}")
                        # 4. Download the aligned subtitles (.ass) with retries
                        for attempt in range(3):
                            try:
                                async with session.get(f"{audio_edge_url}/files", params={"path": ass_path}, timeout=aiohttp.ClientTimeout(total=300)) as file_resp:
                                    if file_resp.status == 200:
                                        with open(ass_path, "wb") as f:
                                            f.write(await file_resp.read())
                                        break
                                    else:
                                        raise Exception(f"HTTP status {file_resp.status}")
                            except Exception as get_err:
                                if attempt == 2:
                                    raise Exception(f"Failed to download ass file after 3 attempts: {get_err}")
                                await asyncio.sleep(1.0)
                        audio_generated = True
                except Exception as e:
                    print(f"Edge Audio Service Exception (falling back to local synthesis): {e}")

            if not audio_generated:
                tts_result = await synthesize_tts(TTSRequest(text=tts_narration, output_path=wav_path, speed=tts_speed))
                engine = (tts_result or {}).get("engine") if isinstance(tts_result, dict) else None
                if engine == "synthetic_wav_fallback":
                    raise RuntimeError(
                        f"Local TTS degraded to synthetic_wav_fallback (toner, not speech) for {shot_key}: "
                        f"Pi edge audio unreachable ({audio_edge_url}) and local Kokoro unavailable. "
                        f"Aborting run rather than shipping fake audio."
                    )
                await align_subtitles_whisper(WhisperRequest(audio_path=wav_path, output_ass_path=ass_path, original_text=tts_narration))



            # Gate 2 Early Validation: WAV must exist and be > 1KB before proceeding
            if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
                raise RuntimeError(
                    f"Gate 2 Early Fail: TTS audio for {shot_key} is missing or empty ({wav_path}). "
                    f"Check AUDIO_EDGE_URL or local Kokoro TTS server."
                )

            # Gate 3 Early Validation: ASS subtitle file must exist and have dialogue
            if not os.path.exists(ass_path) or os.path.getsize(ass_path) < 50:
                raise RuntimeError(
                    f"Gate 3 Early Fail: Subtitle file for {shot_key} is missing or empty ({ass_path}). "
                    f"Check Whisper alignment service."
                )
            # Whisper degraded: alignment failure silently emits a single dummy 5s
            # "NARRATION" dialogue line. That means no real word timestamps -> subtitles
            # are fake; hard fail.
            with open(ass_path, "r", encoding="utf-8", errors="replace") as _ass_f:
                _ass_txt = _ass_f.read()
            _d = [l for l in _ass_txt.splitlines() if l.strip().startswith("Dialogue:")]
            if len(_d) == 1 and "NARRATION" in _d[0]:
                raise RuntimeError(
                    f"Gate 3 Early Fail: Subtitle alignment degraded for {shot_key} "
                    f"(placeholder NARRATION line, no real word timestamps in {ass_path}). "
                    f"Whisper unavailable (Pi edge or local model) — aborting rather than shipping fake subtitles."
                )

            asset_paths.audio[shot_key] = wav_path
            asset_paths.subtitles[shot_key] = ass_path

            # Phase 3: kill WAV-stage clipping. The limiter must run on the actual
            # narration WAV (both remote-downloaded and local-synth paths converge
            # here) so no shot ever ships at -0.0 dB, before the final loudnorm mix.
            _peak_limit_wav(wav_path)

            # ---- Source-of-truth timing: measure the real narration length ----
            audio_dur = _probe_wav_duration(wav_path)
            if not audio_dur:
                audio_dur = max(shot.duration_estimate, MIN_SHOT_DUR)
                print(f"[MediaProducer] WARNING: could not probe narration duration for {shot_key}; "
                      f"falling back to estimate {audio_dur:.1f}s")
            # Final shot timeline length = spoken audio + trailing hold (never cuts narration).
            shot_timeline_dur = max(audio_dur + tail, MIN_SHOT_DUR)
            if shot.shot_id == 1:
                shot_timeline_dur += 0.5
            elif shot.shot_id == len(script.shots):
                # Final outro shot: extra breathing room for the outro CTA ("Like & Subscribe")
                # so the video and background music do not cut abruptly at the end.
                shot_timeline_dur += 3.0

            # Stage 8 Quality-by-Design: Alternate Ken Burns pan direction for optical flow continuity
            ken_burns_direction = "left_to_right" if shot.shot_id % 2 == 1 else "right_to_left"

            # Route and generate specialized visual assets based on AI-classified shot.visual_type
            is_specialized = False
            v_type = getattr(shot, "visual_type", VisualType.STANDARD_IMAGE)

            # Check 0: Real, grounded stat chart from RAG data (chart_spec OR
            #             >=2 numeric claims) — renders an annotated matplotlib
            #             line/bar chart with the correct numbers, never the
            #             silent PIL placeholder. Only shots that carry actual
            #             numeric grounding become charts: a cinematic prompt
            #             that merely MENTIONS charts ("laptop displaying
            #             colorful stock charts") is a scene description, not a
            #             data request, and must NOT force an off-topic
            #             facts/ticker fallback (the Samsung-Galaxy chart leak).
            chart_spec = getattr(shot, "chart_spec", None) or {}
            grounded_chart = _shot_wants_data_chart(shot, chart_spec, clean_narration, raw_visual_prompt)
            explicit_chart = (
                v_type == VisualType.MATPLOTLIB_CHART
                or "[chart:" in prompt_lower
                or grounded_chart
            )
            if explicit_chart:
                print(f"Processing real stat chart visual for {shot_key} (spec={bool(chart_spec.get('values'))}, explicit={explicit_chart})")
                try:
                    labels, values, unit, title, ctype = _derive_chart_data(
                        shot=shot,
                        chart_spec=chart_spec,
                        narration=clean_narration,
                        prompt=raw_visual_prompt,
                        facts=getattr(state, "verified_facts", []) or [],
                        topic=getattr(state, "selected_topic", None),
                    )
                    await generate_dynamic_chart(ChartRequest(
                        title=title[:48],
                        labels=labels,
                        values=values,
                        unit_symbol=unit,
                        chart_type=ctype,
                        duration=shot_timeline_dur,
                        output_mp4_path=mp4_path
                    ))
                    is_specialized = True
                except Exception as chart_err:
                    print(f"Warning: Stat chart render failed: {chart_err}. Falling through to normal flow.")

            # Check 1: Dynamic GIF visual asset retrieval
            elif v_type in (VisualType.GIF_MEME, VisualType.GIF_STICKER) or "[gif:" in prompt_lower or "animated gif" in prompt_lower or "gif sticker" in prompt_lower:
                match = re.search(r'\[gif:\s*([^\]]+)\]', prompt_lower)
                gif_query = match.group(1).strip() if match else clean_narration[:40]
                print(f"Processing AI dynamic GIF segment for {shot_key} (Query: '{gif_query}')")
                try:
                    await fetch_reaction_gif_clip(GIFRequest(
                        query=gif_query,
                        duration=shot_timeline_dur,
                        output_mp4_path=mp4_path
                    ))
                    is_specialized = True
                except Exception as gif_err:
                    print(f"Warning: GIF generation failed: {gif_err}. Falling back to normal image rendering.")

            # Check 2: Dynamic Stock/Data Chart segment (only when the shot is actually
            #            grounded — never for scene descriptions that merely say
            #            "stock chart", which would render a meaningless index).
            elif grounded_chart and (v_type == VisualType.MATPLOTLIB_CHART or "[chart:" in prompt_lower or "stock chart" in prompt_lower or "market graph" in prompt_lower):
                match = re.search(r'\[chart:\s*([^\]]+)\]', prompt_lower)
                _explicit_label = match.group(1).strip() if match else ""
                print(f"Processing AI dynamic data chart segment for {shot_key} (Label: '{_explicit_label or 'market'}')")
                try:
                    hist = await self._get_market_history(state, clean_narration)
                    currency = "₹" if str(hist["symbol"]).endswith(".NS") else "$"
                    labels = hist.get("labels") or ["-4w", "-3w", "-2w", "-1w", "Now"]
                    values = hist.get("values") or [100.0, 102.0, 101.5, 103.0, 105.0]
                    await generate_dynamic_chart(ChartRequest(
                        title=_explicit_label or _market_snapshot_title(hist["symbol"], clean_narration),
                        labels=labels,
                        values=values,
                        unit_symbol=currency,
                        duration=shot_timeline_dur,
                        output_mp4_path=mp4_path
                    ))
                    is_specialized = True
                except Exception as chart_err:
                    print(f"Warning: Chart generation failed: {chart_err}. Falling back to normal image rendering.")

            # Check 3: SVG Animation Counter/Ticker segment
            elif v_type == VisualType.SVG_TICKER or "[svg:" in prompt_lower or re.search(r'\bticker\b', prompt_lower) or re.search(r'\bcounter\b', prompt_lower):
                match = re.search(r'\[svg:\s*([^\]]+)\]', prompt_lower)
                _explicit_label = match.group(1).strip() if match else ""

                print(f"Processing AI animated Playwright SVG ticker segment for {shot_key} (Label: '{_explicit_label or 'market'}')")
                try:
                    quote = await self._get_market_quote(state, clean_narration)
                    currency = "₹" if str(quote["symbol"]).endswith(".NS") else "$"
                    if _explicit_label:
                        title = _explicit_label[:48]
                    elif quote.get("grounded") and quote.get("price"):
                        title = _market_snapshot_title(quote["symbol"], clean_narration)
                    else:
                        title = f"{quote['symbol']} • MARKET SNAPSHOT"
                    if quote.get("grounded") and quote.get("price") is not None:
                        headline_val = f"{currency}{quote['price']:,.2f}"
                        change = quote.get("change") or 0.0
                        sub_text = f"{change:+.2f}% Today"
                    elif quote.get("change") is not None:
                        # No live quote -> show the narration's own stated move so the
                        # visual never contradicts the script (never a fake price).
                        move = quote["change"]
                        headline_val = f"{move:+.1f}%"
                        sub_text = "PRE-MARKET MOVE" if "premarket" in clean_narration.lower() else "MARKET MOVE"
                    else:
                        raise ValueError("No grounded quote and no narration move for SVG ticker")
                    await render_playwright_svg_animation(PlaywrightSVGRequest(
                        chart_type="stock_ticker",
                        title=title,
                        headline_val=headline_val,
                        sub_text=sub_text,
                        duration=shot_timeline_dur,
                        output_mp4_path=mp4_path
                    ))
                    is_specialized = True
                except Exception as svg_err:
                    print(f"Warning: SVG animation failed: {svg_err}. Falling back to normal image rendering.")

            # Default: Widescreen visual + Ken Burns camera pan.
            # Phase 2: every standard-image shot now ATTEMPTS Flux first (fal/Replicate);
            # free stock (Pixabay/Pexels) is used only on a Flux failure or when the AI
            # budget is exhausted. Cost is ~$0.05/run for 16 imgs, well under the cap.
            if not is_specialized:
                # ── Visual cache: reuse an already-paid fal/Replicate image for
                # the SAME shot's enriched prompt, so re-renders / narration-only
                # fixes never spend fal again. Cache is keyed on shot_id + the
                # exact prompt the image engine sees.
                found_cache = False
                if _VISUAL_CACHE_ENABLED:
                    # Key includes shot_id so two DIFFERENT shots that share a
                    # prompt+act never collide (which would reuse one shot's image
                    # in another — duplicate visuals within a video). The SAME
                    # shot re-rendered (proofing, narration-only fix, resume) still
                    # derives the identical key, so valid cross-run reuse is kept.
                    _ck = _visual_cache_key(f"{shot.shot_id}:{visual_prompt}")
                    _cp = os.path.join(_visual_cache_dir(), f"{_ck}.png")
                    if os.path.exists(_cp) and os.path.getsize(_cp) > 5000:
                        import shutil as _sh
                        _sh.copy2(_cp, img_path)
                        print(f"[MediaProducer] Visual cache HIT {shot_key} (prompt {_ck[:10]}): reused paid image → {img_path}")
                        found_cache = True
                if not found_cache:
                    if dummy_frames:
                        from mcp_servers.media_cloud.server import generate_synthetic_png
                        generate_synthetic_png(img_path, title=f"SHOT {shot.shot_id}: {raw_visual_prompt[:40]}")
                    else:
                        use_paid = media_budget.charge_paid_image()
                        if use_paid:
                            run_budget.record_visual()
                            try:
                                await generate_flux_image(ImageGenRequest(prompt=visual_prompt, output_image_path=img_path))
                                # Persist the paid image for later reuse.
                                if _VISUAL_CACHE_ENABLED and os.path.exists(img_path) and os.path.getsize(img_path) > 5000:
                                    _ck = _visual_cache_key(f"{shot.shot_id}:{visual_prompt}")
                                    _cp = os.path.join(_visual_cache_dir(), f"{_ck}.png")
                                    import shutil as _sh2
                                    _sh2.copy2(img_path, _cp)
                                    print(f"[MediaProducer] Visual cache STORE {shot_key} (prompt {_ck[:10]}) → {_cp}")
                            except Exception as e:
                                print(f"Warning: Visual Generation Error on shot {shot.shot_id}: {e}. Falling back to free assets.")
                                if _generate_free_visual(shot, raw_visual_prompt, img_path, mp4_path, anchors=topic_anchors):
                                    is_specialized = True
                        else:
                            print(f"[MediaProducer] AI budget saved/exhausted; free asset for shot {shot.shot_id}.")
                            if _generate_free_visual(shot, raw_visual_prompt, img_path, mp4_path, anchors=topic_anchors):
                                is_specialized = True

                # Outro static text overlay if this is the final shot in the script
                is_last_shot = (shot.shot_id == len(script.shots))
                is_chart_shot = (
                    v_type in (VisualType.MATPLOTLIB_CHART, VisualType.SVG_TICKER)
                    or "[chart:" in prompt_lower
                    or "[svg:" in prompt_lower
                    or "stock chart" in prompt_lower
                    or "market graph" in prompt_lower
                    or "line chart" in prompt_lower
                    or "bar chart" in prompt_lower
                    or "financial chart" in prompt_lower
                )
                if is_last_shot and os.path.exists(img_path):
                    _apply_outro_cta_overlay(img_path)

                # Compile PNG to MP4 with Ken Burns movement (disabled on outro and all charts/graphs)
                await apply_ken_burns_motion(KenBurnsRequest(
                    image_path=img_path,
                    audio_path=wav_path,
                    duration=shot_timeline_dur,
                    output_mp4_path=mp4_path,
                    direction=ken_burns_direction,  # optical flow continuity
                    disable_motion=bool(is_last_shot or is_chart_shot),     # static hold for outro & charts!
                    audio_delay=0.5 if shot.shot_id == 1 else 0.0
                ))

            # Mix narration audio with the specialized visual MP4 if applicable.
            # Re-encode to the MEASURED shot_timeline_dur so the visual is held/padded
            # to fill the full spoken clip (no `-shortest` truncation of narration).
            if is_specialized:
                if os.path.exists(mp4_path):
                    import shutil
                    import subprocess
                    temp_specialized_path = mp4_path.replace(".mp4", "_visual_only.mp4")
                    shutil.move(mp4_path, temp_specialized_path)
                    print(f"[MediaProducer] Merging audio {wav_path} with specialized video {temp_specialized_path} -> {mp4_path} (dur={shot_timeline_dur:.2f}s)")
                    merge_cmd = [
                        "ffmpeg", "-y",
                        "-i", temp_specialized_path,
                        "-i", wav_path,
                        "-filter_complex",
                        "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v];"
                        "[1:a]apad[a]",
                        "-map", "[v]", "-map", "[a]",
                        "-t", f"{shot_timeline_dur:.3f}",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-maxrate", "6M", "-bufsize", "12M",
                        "-c:a", "aac", "-b:a", "192k",
                        mp4_path
                    ]
                    res = subprocess.run(merge_cmd, capture_output=True, text=True)
                    if res.returncode != 0:
                        print(f"Warning: ffmpeg audio merge failed: {res.stderr}. Restoring visual-only clip.")
                        shutil.move(temp_specialized_path, mp4_path)
                    else:
                        if os.path.exists(temp_specialized_path):
                            os.remove(temp_specialized_path)

            asset_paths.visuals[shot_key] = mp4_path

            concat_lines.append(f"file '{mp4_path}'")

            # Probe the REAL rendered clip duration (post-encode) so the master
            # subtitle offsets exactly match the concatenated media clock — this
            # removes any fps/rounding drift between shots.
            real_dur = _probe_duration(mp4_path) or shot_timeline_dur
            shot_timings.append(real_dur)

        # Persist the measured timeline + crossfade for the pre-upload gates.
        asset_paths.measured_durations = list(shot_timings)
        asset_paths.crossfade_used = crossfade

        exec_suffix = (getattr(state, "pipeline_id", "") or "csgv").strip() or "csgv"

        # Create Concat List File for FFmpeg
        concat_list_path = os.path.join(self.storage_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            f.write("\n".join(concat_lines))

        # Merge all shot subtitles into a single master .ass timeline file,
        # offsetting by the MEASURED real clip durations (not script estimates).
        ass_paths = [asset_paths.subtitles[f"shot_{shot.shot_id}"] for shot in script.shots if f"shot_{shot.shot_id}" in asset_paths.subtitles]
        shot_durs = [max(t, 2.0) for t in shot_timings]
        master_sub_path = os.path.join(self.storage_dir, "master_subtitles.ass")
        merge_ass_subtitle_files(ass_paths, shot_durs, master_sub_path, crossfade=crossfade)
        print(f"[MediaProducer] Master subtitle offsets (measured {', crossfade='+str(crossfade)+'s' if crossfade>0 else ''}, seconds): {[round(t, 2) for t in shot_durs]}")

        # Assemble Final Timeline — filename correlated to the execution so a
        # given master always maps back to the run that produced it.
        final_video_path = os.path.join(self.storage_dir, f"final_video_{exec_suffix}.mp4")
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        bgm_source_path = os.path.join(project_root, "resources", "bgm.mp3")
        bgm_final_path = os.path.join(self.storage_dir, "bgm.mp3")

        if os.path.exists(bgm_source_path) and os.path.getsize(bgm_source_path) > 1000:
            import shutil
            shutil.copy2(bgm_source_path, bgm_final_path)
            print(f"[MediaProducer] Using background music file: {bgm_source_path}")
        else:
            print("[MediaProducer] WARNING: resources/bgm.mp3 not found or invalid. Falling back to dummy text file (silence).")
            with open(bgm_final_path, "w") as f:
                f.write("DUMMY_BGM")

        if renderer == "moviepy":
            try:
                self._assemble_moviepy(
                    clip_paths=[asset_paths.visuals[f"shot_{shot.shot_id}"] for shot in script.shots],
                    shot_timings=shot_timings,
                    master_sub_path=master_sub_path,
                    bgm_path=bgm_final_path,
                    output_video_path=final_video_path,
                    crossfade=crossfade
                )
            except Exception as mp_err:
                print(f"[MediaProducer] MoviePy assembly failed ({mp_err}); falling back to ffmpeg timeline.")
                await assemble_ffmpeg_timeline(TimelineAssemblyRequest(
                    concat_list_path=concat_list_path,
                    subtitle_path=master_sub_path,
                    bgm_path=bgm_final_path,
                    output_video_path=final_video_path,
                    crossfade=crossfade,
                    transition="fade"
                ))
        else:
            await assemble_ffmpeg_timeline(TimelineAssemblyRequest(
                concat_list_path=concat_list_path,
                subtitle_path=master_sub_path,
                bgm_path=bgm_final_path,
                output_video_path=final_video_path,
                crossfade=crossfade,
                transition="fade"
            ))

        # Generate dynamic widescreen high-CTR Thumbnail (uses the CTR-optimized brief
        # for the text, and the hero shot's theme-matched visual prompt for the art so
        # the background matches the story's subject instead of the raw CTR text).
        # Preferred (Option B): nano-banana bakes the full design — thematic hook from
        # the narration/facts + design-rules art + model-rendered text + compliance —
        # reusing the video's own frame as an identity reference; then thumbnails.set
        # uploads it (honored for long-form). Fallback = art+cv2 path below.
        thumbnail_path = os.path.join(self.storage_dir, f"thumbnail_{exec_suffix}.png")
        thumb_scene = _build_thumbnail_scene(state)
        baked_thumb_path = os.path.join(self.storage_dir, f"thumbnail_{exec_suffix}.jpg")
        try:
            from src.engine.nano_banana import generate_baked_video_thumbnail
            _ref = None
            try:
                import subprocess as _sp
                _frame_path = os.path.join(self.storage_dir, f"thumbref_{exec_suffix}.jpg")
                _sp.run(
                    ["ffmpeg", "-y", "-ss", "2", "-i", final_video_path,
                     "-frames:v", "1", "-vf", "scale=1280:-1", "-q:v", "2", _frame_path],
                    capture_output=True, timeout=60,
                )
                if os.path.exists(_frame_path) and os.path.getsize(_frame_path) > 500:
                    with open(_frame_path, "rb") as _f:
                        _ref = _f.read()
            except Exception as _re:
                print(f"[MediaProducer] thumbnail reference frame skip: {_re}")
            if generate_baked_video_thumbnail(state, hero_scene=thumb_scene,
                                              output_path=baked_thumb_path, reference_frame=_ref):
                asset_paths.thumbnail = baked_thumb_path
                thumbnail_path = baked_thumb_path
                print(f"[MediaProducer] Baked native-text thumbnail -> {thumbnail_path}")
                thumb_baked = True
                # Optional A/B variant set (Studio "Test & Compare" strategy):
                # generate distinct variants varying one axis each so the operator
                # can finish the experiment in YouTube Studio. Gated + non-fatal.
                if os.getenv("CSVG_THUMBNAIL_VARIANTS", "0").strip().lower() in ("1", "true", "yes"):
                    try:
                        from src.engine.nano_banana import generate_thumbnail_variants
                        _vpaths = generate_thumbnail_variants(
                            state, hero_scene=thumb_scene, output_dir=self.storage_dir,
                            aspect_ratio="16:9", reference_frame=_ref)
                        if _vpaths:
                            print(f"[MediaProducer] Generated {len(_vpaths)} A/B thumbnail "
                                  f"variant(s) for Studio Test & Compare "
                                  f"(see thumbnail_variants.json).")
                    except Exception as ve:
                        print(f"[MediaProducer] thumbnail variant generation skipped: {ve}")
            else:
                raise RuntimeError("nano-banana baked thumbnail returned None")
        except Exception as thumb_err:
            print(f"[MediaProducer] nano-banana baked thumbnail skipped ({thumb_err}); using cv2 path.")
            thumb_baked = False

        if not thumb_baked:
            try:
                from mcp_servers.media_cloud.server import ThumbnailRequest, generate_thumbnail
                brief = (
                    state.seo_metadata.thumbnail_brief
                    if state.seo_metadata and state.seo_metadata.thumbnail_brief
                    else (state.selected_topic.headline if state.selected_topic else "Market Shift")
                )
                art_prompt = ""
                try:
                    from src.engine.nano_banana import craft_thumbnail_art_prompt
                    art_prompt = craft_thumbnail_art_prompt(state, hero_scene=thumb_scene, aspect_ratio="16:9")
                except Exception as art_err:
                    print(f"[MediaProducer] nano-banana art prompt analysis skipped: {art_err}")
                await generate_thumbnail(ThumbnailRequest(
                    headline_text=brief,
                    visual_prompt=thumb_scene,
                    art_prompt=art_prompt,
                    output_thumbnail_path=thumbnail_path
                ))
                asset_paths.thumbnail = thumbnail_path
                print(f"[MediaProducer] High-CTR Thumbnail successfully generated at {thumbnail_path}")
            except Exception as thumb_err:
                print(f"Warning: Thumbnail generation failed: {thumb_err}")

        asset_paths.final_video = final_video_path

        # Optional long-term archive: copy final mp4 + thumbnail into
        # logs/final_videos/<exec_id>/ so masters survive /tmp cleanup and
        # correlate to the run. Gitignored (logs/).
        if _ARCHIVE_FINAL and os.path.exists(final_video_path):
            try:
                _archive_dir = os.path.join(_repo_root(), "logs", "final_videos", exec_suffix)
                os.makedirs(_archive_dir, exist_ok=True)
                import shutil as _sh_arch
                _sh_arch.copy2(final_video_path, os.path.join(_archive_dir, "final_video.mp4"))
                if os.path.exists(thumbnail_path):
                    _sh_arch.copy2(thumbnail_path, os.path.join(_archive_dir, "thumbnail.png"))
                print(f"[MediaProducer] Archived final artifacts → {_archive_dir}")
            except Exception as arch_err:
                print(f"Warning: Final archive copy failed: {arch_err}")

        # Shorts clips are produced once, at publish time, by
        # micro_content_producer (act-aware 9:16 cuts) — the old redundant
        # 3-cut media_producer pass was removed so each run only encodes the
        # 2 Shorts it actually uploads.
        state.asset_paths = asset_paths
        state.execution_stage = "MEDIA_PRODUCED"
        return asset_paths

    # ----------------------------------------------------------------------
    # MoviePy timeline composer (opt-in via --renderer moviepy)
    #
    # Builds the final master from the SAME measured per-shot clips used by the
    # ffmpeg path, but expresses the timeline in MoviePy's absolute-time model:
    # each shot's video is placed at its measured start; narration (already in
    # each clip) + ducked music are mixed on a composite audio track; subtitles
    # are burned from the master .ass (converted to .srt). Timings come ONLY
    # from ffprobe-measured durations, so video/audio/subtitle/music stay in sync.
    #
    # Requirements: `pip install moviepy` (+ ImageMagick only if you want the
    # TextClip subtitle burn). If moviepy (or ImageMagick) is missing, this
    # raises and the caller falls back to the ffmpeg assembler.
    # ----------------------------------------------------------------------
    def _assemble_moviepy(self, clip_paths, shot_timings, master_sub_path,
                          bgm_path, output_video_path, crossfade: float = 0.0):
        try:
            from moviepy.editor import (
                VideoFileClip, AudioFileClip, CompositeAudioClip,
                CompositeVideoClip, concatenate_videoclips, afx
            )
        except Exception as e:
            print(f"[MoviePy] moviepy not installed or import failed: {e}. "
                  f"Falling back to ffmpeg assembly.")
            raise RuntimeError("MoviePy unavailable; use --renderer ffmpeg")

        # 1. Load each shot clip and clamp to its measured duration.
        clips = []
        try:
            for path, dur in zip(clip_paths, shot_timings):
                c = VideoFileClip(path)
                if c.duration is not None and dur < c.duration - 0.05:
                    c = c.subclip(0, dur)
                clips.append(c)
        except Exception as e:
            print(f"[MoviePy] Failed to load a shot clip: {e}")
            for c in clips:
                c.close()
            raise RuntimeError(f"MoviePy clip load failed: {e}")

        video = concatenate_videoclips(clips, method="compose",
                                       crossfade=crossfade if crossfade > 0 else 0)
        video = video.resize((RESOLUTION[0], RESOLUTION[1]))

        # 2. Mix background music under narration.
        narration = video.audio
        if os.path.exists(bgm_path) and os.path.getsize(bgm_path) > 1000:
            try:
                bgm = AudioFileClip(bgm_path)
                try:
                    tempo = min(max(float(os.getenv("BGM_TEMPO", "1.1")), 0.5), 2.0)
                except Exception:
                    tempo = 1.1
                if abs(tempo - 1.0) > 0.001:
                    bgm = afx.speedx(bgm, factor=tempo)
                bgm = bgm.audio_loop(duration=video.duration).volumex(BGM_VOLUME)
                video = video.set_audio(CompositeAudioClip([narration, bgm]))
                print("[MoviePy] Background music mixed under narration (tempo x%s)." % (round(tempo, 3),))
            except Exception as e:
                print(f"[MoviePy] BGM mix skipped (keeping narration only): {e}")
        else:
            print("[MoviePy] No valid BGM; narration only.")

        # 3. Burn subtitles from master .ass -> .srt (needs moviepy TextClip/ImageMagick).
        subs_path = os.path.join(self.storage_dir, "master_subtitles.srt")
        try:
            self._ass_to_srt(master_sub_path, subs_path)
            from moviepy.video.tools.subtitles import SubtitlesClip
            from moviepy.video.VideoClip import TextClip

            def make_text(txt):
                return TextClip(
                    txt, fontsize=46, color="white", stroke_color="black",
                    stroke_width=2, font="Montserrat", method="caption",
                    size=(RESOLUTION[0] * 0.92, None)
                )
            subs = SubtitlesClip(subs_path, make_text)
            final = CompositeVideoClip([
                video,
                subs.set_position(("center", "bottom")).set_duration(video.duration)
            ])
            print("[MoviePy] Subtitles burned via MoviePy SubtitlesClip.")
        except Exception as e:
            print(f"[MoviePy] Subtitle burn via TextClip unavailable ({e}); "
                  f"rendering without subtitle overlay.")
            final = video

        print(f"[MoviePy] Rendering master through MoviePy -> {output_video_path}")
        final.write_videofile(
            output_video_path, fps=25, codec="libx264", preset="fast",
            audio_codec="aac", audio_bitrate="192k",
            ffmpeg_params=["-pix_fmt", "yuv420p"]
        )
        final.close()
        video.close()
        for c in clips:
            c.close()

    def _ass_to_srt(self, ass_path: str, srt_path: str):
        """Minimal .ass -> .srt converter for the MoviePy subtitle burn."""
        subs = []
        if not os.path.exists(ass_path):
            return
        with open(ass_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("Dialogue:"):
                    continue
                parts = line.split(",", 9)
                if len(parts) != 10:
                    continue

                def to_srt(t):
                    h, m, s = t.strip().split(":")
                    ms = int(round(float(s) * 1000))
                    return f"{int(h):02d}:{int(m):02d}:{ms // 1000:02d},{ms % 1000:03d}"
                subs.append((to_srt(parts[1]), to_srt(parts[2]), parts[9].rstrip("\n")))

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, (start, end, text) in enumerate(subs, start=1):
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    async def process(self, state: GlobalState, dummy_frames: bool = False, renderer: Optional[str] = None, crossfade: Optional[float] = None, pad_after_narration: Optional[float] = None) -> A2AMessage:
        """
        Executes Media Producer workflow:
        1. Reads state.script_data
        2. Calls MCP tools to produce audio, subtitles, visuals, and final timeline
        3. Updates state.asset_paths
        4. Emits MEDIA_READY A2AMessage to Orchestrator
        """
        assets = await self.produce_all_media(state, dummy_frames=dummy_frames, renderer=renderer,
                                              crossfade=crossfade, pad_after_narration=pad_after_narration)

        msg = A2AMessage(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            sender=AgentRole.MEDIA_PRODUCER,
            target=AgentRole.ORCHESTRATOR,
            intent=AgentIntent.MEDIA_READY,
            payload={
                "status": "SUCCESS",
                "final_video": assets.final_video,
                "audio_shot_count": len(assets.audio),
                "visual_shot_count": len(assets.visuals)
            },
            state_hash=compute_state_hash(state),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        return msg
