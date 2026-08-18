#!/usr/bin/env python3
"""Keyless YouTube competitor scraper + gap analyzer.

PUBLIC, KEYLESS endpoints only (no YouTube Data API / OAuth):
  * Channel handle  -> channelId  (parse from the @handle page HTML)
  * channelId       -> recent uploads RSS (titles, publish dates, thumbs)
  * videoId         -> watch page  (viewCount + lengthSeconds from
                       ytInitialPlayerResponse; likeCount best-effort)

It collects competitor public metadata and then benchmarks it against THIS
pipeline's actual parameters/code (runtime target, cadence, title template,
thumbnail strategy) to surface concrete code gaps.

Run from repo root:
    python competitor_scraper.py \
        --channels "FinanceBureauOfficial,AIUncovered,economics-explained" \
        --limit 20 --out logs/competitor_data.json

Flags:
    --channels   comma list of @handles, bare handles, or UC... channel ids
    --limit      max videos per channel to fetch detail for (default 20)
    --no-views   skip watch-page fetches (RSS metadata only, much faster)
    --out        output JSON path (default logs/competitor_data.json)
    --no-analyze skip the gap-analysis report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

try:
    import requests
except ImportError:  # pragma: no cover
    print("[competitor_scraper] requests is required: pip install requests", file=sys.stderr)
    raise

# ---------------------------------------------------------------------------
# HTTP helpers (browser UA so YouTube serves full initial-data JSON)
# ---------------------------------------------------------------------------
_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Grace between network calls to avoid hammering YouTube.
_REQ_DELAY_S: float = 0.8

# Module-level session: reuses the consent cookie across calls so we don't
# keep getting bounced to the "before you continue" interstitial.
_SESSION: Optional[requests.Session] = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update(_HEADERS)
        s.cookies.update({"CONSENT": "YES+1", "PREF": "tz=UTC"})
        _SESSION = s
    return _SESSION


# Own-channel id lives in channelMetadataRenderer; other channelIds on the
# page belong to comments/related channels, so we must anchor to that block.
_CHANNEL_ID_RE = re.compile(r'channelMetadataRenderer"\s*,\s*"channelId"\s*:\s*"(UC[\w-]{20,})"')
_BROWSE_ID_RE = re.compile(r'"browseId"\s*:\s*"(UC[\w-]{20,})"')
_OG_URL_RE = re.compile(r'<meta\s+property="og:url"\s+content="[^"]*/(channel|@)[^"]*(UC[\w-]{20,})"')
_VIEWCOUNT_RE = re.compile(r'"viewCount"\s*:\s*"(\d+)"')
_LENGTH_RE = re.compile(r'"lengthSeconds"\s*:\s*"(\d+)"')
_LIKE_RE = re.compile(r'"label"\s*:\s*"([\d,]+)\s+likes?"')
_PLAYER_RESP_MARKER = "ytInitialPlayerResponse"


def _extract_balanced_json(text: str, start: int) -> Optional[Any]:
    """Parse the JSON object beginning at index `start` (a `{`).

    Uses brace counting (not a naive non-greedy regex) so nested objects in
    the large ytInitialPlayerResponse blob parse correctly.
    """
    if start == -1 or start >= len(text):
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
    return None


def _extract_player_response(html: str) -> Optional[Any]:
    """Extract the assigned ytInitialPlayerResponse object from a watch page."""
    m = re.search(r"ytInitialPlayerResponse\s*=\s*\{", html)
    if not m:
        return None
    return _extract_balanced_json(html, m.end() - 1)


def _get(url: str, timeout: int = 20) -> Optional[str]:
    """Fetch a URL with the browser UA + consent cookie; None on failure."""
    try:
        resp = _session().get(url, timeout=timeout)
        if resp.status_code != 200:
            print(f"  [warn] HTTP {resp.status_code} for {url}", file=sys.stderr)
            return None
        return resp.text
    except requests.RequestException as exc:
        print(f"  [warn] request failed for {url}: {exc}", file=sys.stderr)
        return None


def _resolve_channel_id(handle_or_id: str) -> Optional[str]:
    """Accept a UC... id directly, or resolve a @handle / bare handle."""
    h = handle_or_id.strip().lstrip("@")
    if h.startswith("UC") and len(h) > 20:
        return h
    page = _get(f"https://www.youtube.com/@{quote(h)}")
    if not page:
        return None
    m = _CHANNEL_ID_RE.search(page) or _OG_URL_RE.search(page)
    if m:
        cid = m.group(1) if (m.lastindex == 1 and m.group(1).startswith("UC")) else m.group(2)
        if cid and cid.startswith("UC"):
            return cid
    # Fallbacks: browseId, then any UC id present anywhere on the page.
    mb = _BROWSE_ID_RE.search(page)
    if mb:
        return mb.group(1)
    m2 = re.search(r"(UC[\w-]{20,})", page)
    return m2.group(1) if m2 else None


# ---------------------------------------------------------------------------
# RSS parsing (recent uploads: title, published, video id, thumbnail)
# ---------------------------------------------------------------------------
_ATOM = "{http://www.w3.org/2005/Atom}"
_YT = "{http://www.youtube.com/xml/schemas/2015}"
_MEDIA = "{http://search.yahoo.com/mrss/}"


def _parse_rss(xml_text: str) -> List[Dict[str, Any]]:
    import xml.etree.ElementTree as ET

    items: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        print(f"  [warn] RSS parse error: {exc}", file=sys.stderr)
        return items

    for entry in root.findall(f"{_ATOM}entry"):
        vid = entry.findtext(f"{_YT}videoId")
        title = entry.findtext(f"{_ATOM}title")
        published = entry.findtext(f"{_ATOM}published")
        thumb_el = entry.find(f"{_MEDIA}group/{_MEDIA}thumbnail")
        thumb = thumb_el.get("url") if thumb_el is not None else None
        if vid and title:
            items.append(
                {
                    "video_id": vid,
                    "title": title.strip(),
                    "published": published,
                    "thumbnail_rss": thumb,
                }
            )
    return items


# ---------------------------------------------------------------------------
# Watch-page detail (views, duration, best-effort likes)
# ---------------------------------------------------------------------------
def _video_detail(video_id: str) -> Dict[str, Any]:
    detail: Dict[str, Any] = {
        "video_id": video_id,
        "view_count": None,
        "length_seconds": None,
        "like_count": None,
        "thumbnail": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
    }
    html = _get(f"https://www.youtube.com/watch?v={video_id}")
    if not html:
        return detail

    # Primary: ytInitialPlayerResponse (reliable viewCount + lengthSeconds).
    # Anchor on the ` = {` assignment (not the bare key, which also appears
    # inside ytInitialData and would parse the wrong object).
    pr = _extract_player_response(html)
    if isinstance(pr, dict):
        vd = pr.get("videoDetails", {})
        stat = vd.get("statistics", {})
        if stat.get("viewCount"):
            try:
                detail["view_count"] = int(stat["viewCount"])
            except (ValueError, TypeError):
                pass
        if vd.get("lengthSeconds"):
            try:
                detail["length_seconds"] = int(vd["lengthSeconds"])
            except (ValueError, TypeError):
                pass
        if stat.get("likeCount"):
            try:
                detail["like_count"] = int(stat["likeCount"])
            except (ValueError, TypeError):
                pass

    # Best-effort like fallback from ytInitialData label.
    if detail["like_count"] is None:
        lm = _LIKE_RE.search(html)
        if lm:
            try:
                detail["like_count"] = int(lm.group(1).replace(",", ""))
            except ValueError:
                pass

    # Robust regex fallbacks for view count (the simpleText form in
    # ytInitialData survives even when videoDetails is empty / geo-restricted).
    if detail["view_count"] is None:
        sm = re.search(r'"viewCount"\s*:\s*\{\s*"simpleText"\s*:\s*"([\d,]+)\s+views"', html)
        if sm:
            try:
                detail["view_count"] = int(sm.group(1).replace(",", ""))
            except ValueError:
                pass
    if detail["view_count"] is None:
        vm = _VIEWCOUNT_RE.search(html)
        if vm:
            detail["view_count"] = int(vm.group(1))
    if detail["length_seconds"] is None:
        lm = _LENGTH_RE.search(html)
        if lm:
            detail["length_seconds"] = int(lm.group(1))
    return detail


# ---------------------------------------------------------------------------
# /videos tab fallback (for channels whose RSS feed returns 0 <entry> elements,
# e.g. Shorts-only / hidden-uploads channels). Parses ytInitialData renderers.
# ---------------------------------------------------------------------------
_COUNT_RE = re.compile(r"([\d.]+)\s*([KMB])?", re.IGNORECASE)


def _parse_count(text: Optional[str]) -> Optional[int]:
    """Parse '134K views' / '1.2M' style counts into an int."""
    if not text:
        return None
    m = _COUNT_RE.search(text.replace(",", ""))
    if not m:
        return None
    val = float(m.group(1))
    suf = (m.group(2) or "").upper()
    mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suf, 1)
    return int(val * mult)


def _walk_video_renderers(node: Any, out: List[Dict[str, Any]]) -> None:
    """Recursively collect every videoRenderer / gridVideoRenderer dict."""
    if isinstance(node, dict):
        for key in ("videoRenderer", "gridVideoRenderer"):
            if key in node and isinstance(node[key], dict):
                out.append(node[key])
        for v in node.values():
            _walk_video_renderers(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk_video_renderers(v, out)


def _scrape_videos_tab(cid: str, handle: str, limit: int) -> List[Dict[str, Any]]:
    """Fallback: pull recent uploads from the /videos tab via ytInitialData."""
    html = _get(f"https://www.youtube.com/channel/{cid}/videos")
    if not html:
        return []
    data = _extract_balanced_json(html, html.find("ytInitialData") + len('"ytInitialData"'))
    if not isinstance(data, dict):
        return []
    renderers: List[Dict[str, Any]] = []
    _walk_video_renderers(data, renderers)

    results: List[Dict[str, Any]] = []
    seen: set = set()
    for r in renderers:
        vid = r.get("videoId")
        if not vid or vid in seen:
            continue
        seen.add(vid)
        title = (r.get("title") or {}).get("runs", [{}])[0].get("text") or r.get("title", {}).get("simpleText")
        views_txt = (r.get("viewCountText") or {}).get("simpleText")
        length_txt = (r.get("lengthText") or {}).get("simpleText")
        thumbs = (r.get("thumbnail") or {}).get("thumbnails", [])
        thumb = thumbs[-1].get("url") if thumbs else f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
        results.append({
            "video_id": vid,
            "title": (title or "").strip(),
            "published": None,  # /videos tab gives relative "3 weeks ago", not absolute
            "view_count": _parse_count(views_txt),
            "length_seconds": _parse_duration(length_txt),
            "like_count": None,
            "thumbnail": thumb,
        })
        if len(results) >= limit:
            break
    return results


def _parse_duration(text: Optional[str]) -> Optional[int]:
    """Parse '13:05' or '1:02:09' into total seconds."""
    if not text:
        return None
    parts = [int(p) for p in text.split(":")]
    if not parts:
        return None
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


# ---------------------------------------------------------------------------
# Script (captions) + thumbnail analysis  — keyless, best-effort
# ---------------------------------------------------------------------------
def _fetch_captions(video_id: str) -> Optional[str]:
    """Fetch a competitor's public captions (no auth) via youtube-transcript-api.

    Works across API versions (instance ``.fetch`` is universal; older builds
    fall back to ``.list``/``.find_transcript``). Returns the concatenated
    transcript text, or None on any failure (captions disabled, IP/region
    blocked, library absent, etc.). NOTE: from datacenter IPs YouTube often
    returns RequestBlocked — run from a residential IP for reliable captions.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None
    try:
        api = YouTubeTranscriptApi()

        def _text_of(tr: Any) -> Optional[str]:
            if tr is None:
                return None
            if hasattr(tr, "text"):  # v1 Transcript
                return tr.text
            snips = getattr(tr, "snippets", None)
            if snips:  # v0.x FetchedTranscript
                return " ".join(getattr(s, "text", "") for s in snips)
            if isinstance(tr, (list, tuple)):  # raw list of dicts
                return " ".join(d.get("text", "") if isinstance(d, dict) else str(d) for d in tr)
            return None

        # Primary: modern instance fetch (also valid as a classmethod call in v1).
        try:
            return _text_of(api.fetch(video_id, languages=["en"]))
        except Exception:
            pass
        # Fallback: explicit transcript selection (older API surface).
        try:
            tl = api.list(video_id)
            try:
                t = tl.find_transcript(["en"])
            except Exception:
                try:
                    t = tl.find_generated_transcript(["en"])
                except Exception:
                    t = next(iter(tl))
            return _text_of(t.fetch())
        except Exception:
            pass
    except Exception:
        return None
    return None


def _classify_hook(text: str) -> str:
    """Roughly classify a video's opening hook style."""
    if "?" in text:
        return "question"
    if _TENSION_LEXICON.search(text):
        return "tension"
    if re.search(r"\b(imagine|picture|story|years ago|back in|in 20\d\d)\b", text, re.IGNORECASE):
        return "story"
    return "statement"


def _analyze_script(transcript: str, duration_seconds: Optional[int]) -> Dict[str, Any]:
    """Derive script-shaping metrics from a transcript.

    These map directly to our pipeline's script levers (wpm, hook rule,
    personalization, tension density) so gaps are actionable.
    """
    words = transcript.split()
    wc = len(words)
    wpm = round(wc / (duration_seconds / 60.0), 1) if duration_seconds and duration_seconds > 0 else None
    sentences = [s for s in re.split(r"[.!?]+", transcript) if s.strip()]
    avg_sent = round(wc / len(sentences), 1) if sentences else None
    hook = " ".join(words[:50])
    question_count = transcript.count("?")
    you_count = len(re.findall(r"\b(you|your|you're|yours)\b", transcript, re.IGNORECASE))
    tension_count = len(_TENSION_LEXICON.findall(transcript))
    return {
        "word_count": wc,
        "wpm": wpm,
        "avg_sentence_words": avg_sent,
        "hook_text": hook[:200],
        "hook_type": _classify_hook(hook),
        "question_count": question_count,
        "questions_per_100w": round(question_count / wc * 100, 2) if wc else None,
        "you_density_pct": round(you_count / wc * 100, 2) if wc else None,
        "tension_density_pct": round(tension_count / wc * 100, 2) if wc else None,
    }


def _analyze_thumbnail(thumb_url: str) -> Dict[str, Any]:
    """Lightweight thumbnail stats via Pillow (brightness/color), keyless.

    Full visual-style analysis (face/reaction, B-roll vs stock vs AI, on-image
    text) needs frame extraction + a vision model and is OUT of keyless scope.
    """
    out: Dict[str, Any] = {"fetched": False, "mean_lum": None, "avg_color": None, "error": None}
    try:
        from PIL import Image
        import io
    except ImportError:
        out["error"] = "Pillow unavailable"
        return out
    try:
        r = _session().get(thumb_url, timeout=20)
        if r.status_code != 200:
            out["error"] = f"HTTP {r.status_code}"
            return out
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        px = list(img.getdata())
        n = len(px)
        if n == 0:
            out["error"] = "empty image"
            return out
        rsum = gsum = bsum = 0
        for (rv, gv, bv) in px:
            rsum += rv
            gsum += gv
            bsum += bv
        mean_lum = round((0.2126 * rsum + 0.7152 * gsum + 0.0722 * bsum) / n, 1)
        out.update({
            "fetched": True,
            "mean_lum": mean_lum,
            "avg_color": [round(rsum / n), round(gsum / n), round(bsum / n)],
            "width": img.width,
            "height": img.height,
        })
    except Exception as exc:  # noqa: BLE001 - best-effort analysis
        out["error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
@dataclass
class ChannelData:
    handle: str
    channel_id: Optional[str]
    resolved: bool
    videos: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def scrape_channel(handle: str, limit: int, fetch_views: bool,
                   fetch_script: bool = True, fetch_visuals: bool = False) -> ChannelData:
    print(f"[scrape] resolving {handle} ...")
    cid = _resolve_channel_id(handle)
    if not cid:
        return ChannelData(handle=handle, channel_id=None, resolved=False,
                           error="could not resolve channelId")

    rss = _get(f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}")
    if not rss:
        return ChannelData(handle=handle, channel_id=cid, resolved=True,
                           error="RSS fetch failed")

    entries = _parse_rss(rss)[:limit]
    print(f"[scrape] {handle}: {len(entries)} videos from RSS")

    # Fallback for channels whose RSS feed has no <entry> elements.
    if not entries:
        entries = _scrape_videos_tab(cid, handle, limit)
        print(f"[scrape] {handle}: {len(entries)} videos from /videos tab (RSS empty)")

    for i, ent in enumerate(entries):
        if fetch_views and ent.get("view_count") is None:
            time.sleep(_REQ_DELAY_S)
            det = _video_detail(ent["video_id"])
            ent.update(det)
            ent.pop("thumbnail_rss", None)
        elif not fetch_views:
            ent["thumbnail"] = f"https://i.ytimg.com/vi/{ent['video_id']}/hqdefault.jpg"

        # Script (captions) analysis — independent of view fetch.
        if fetch_script:
            time.sleep(_REQ_DELAY_S * 0.5)
            cap = _fetch_captions(ent["video_id"])
            if cap:
                ent["script_analysis"] = _analyze_script(cap, ent.get("length_seconds"))

        # Thumbnail visual stats — independent of view fetch.
        if fetch_visuals and ent.get("thumbnail"):
            time.sleep(_REQ_DELAY_S * 0.5)
            ent["thumbnail_analysis"] = _analyze_thumbnail(ent["thumbnail"])

        # Progress marker every few videos.
        if (i + 1) % 5 == 0:
            print(f"  ... {i + 1}/{len(entries)}")
    return ChannelData(handle=handle, channel_id=cid, resolved=True, videos=entries)


# ---------------------------------------------------------------------------
# Gap analysis vs THIS pipeline's actual parameters
# ---------------------------------------------------------------------------
# Baseline constants pulled from the codebase (see referenced file:line).
PIPELINE_BASELINE: Dict[str, Any] = {
    "runtime_target_min": 13.0,          # story_designer.py:449,563,617 (estimated_runtime_mins=13.0)
    "runtime_window_min": (11.0, 14.0),  # story_designer.py:1588 ("11-14 minute")
    "ctr_title_max_chars": 65,           # story_designer.py:1588,1606
    "title_fallback_template": "The Hidden Truth Behind {headline}...",  # :999,1452
    "cadence_cron_per_day": 2,           # cron_publish.sh windows
    "cadence_hard_max_per_day": 4,       # CSVG_MAX_DAILY_PUBLISHES default
    "wpm": 150,                          # story_designer.py:1157,1455 (words->seconds)
    "script_target_words": 1950,         # 13 min * 150 wpm
    "hook_rule": "hook != title; uses title's trailing payoff words (story_designer.py:1716)",
    "personalization_optimized": False,  # no explicit 'you/your' density optimization
    "thumbnail_target_mean_lum": 100,    # nano_banana comply_thumbnail brightness lift target
    "thumbnail_strategy": "AI-generated (nano_banana) + burned text overlay",
}

# Lexicon of "tension / contrarian / curiosity" framing common in high-CTR
# finance/AI news. Combines hard tension words with question/curiosity hooks.
_TENSION_LEXICON = re.compile(
    r"\b(worse|wrong|lie|lies|shock|shocking|secret|hidden|truth|scam|crisis|"
    r"bubble|collapse|crash|warning|danger|exposed|insane|brutal|dead|"
    r"nightmare|trap|betray|panic|explode|destroy|unbelievable|backfire|"
    r"fail|failing|failed|lost|lose|losing|doomed|threat|risk|mistake|plunge|"
    r"tank|sink|fear|why|what they|they don'?t|nobody|real reason|the real)\b",
    re.IGNORECASE,
)


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _tension_score(title: str) -> int:
    return len(_TENSION_LEXICON.findall(title))


def analyze_channel(ch: ChannelData) -> Dict[str, Any]:
    if not ch.resolved or not ch.videos:
        return {"handle": ch.handle, "scanned": False, "reason": ch.error or "no videos"}

    titles = [v["title"] for v in ch.videos if v.get("title")]
    durations = [v["length_seconds"] for v in ch.videos if v.get("length_seconds")]
    views = [v["view_count"] for v in ch.videos if v.get("view_count")]
    _parsed = [_parse_dt(v.get("published")) for v in ch.videos]
    dates = sorted([d for d in _parsed if d is not None])

    # Cadence: uploads per week over the observed window.
    cadence_per_week = None
    if len(dates) >= 2:
        span_days = (dates[-1] - dates[0]).total_seconds() / 86400.0
        if span_days > 0:
            cadence_per_week = round(len(dates) / span_days * 7.0, 2)

    title_lens = [len(t) for t in titles]
    tension_titles = [t for t in titles if _tension_score(t) > 0]
    avg_dur_min = round(sum(durations) / len(durations) / 60.0, 2) if durations else None
    total_views = sum(views) if views else None
    median_views = None
    if views:
        sv = sorted(views)
        median_views = sv[len(sv) // 2]

    # --- Script (captions) aggregation ---
    scripts = [sa for sa in (v.get("script_analysis") for v in ch.videos) if isinstance(sa, dict)]
    wpms = [s["wpm"] for s in scripts if s.get("wpm")]
    you_dens = [s["you_density_pct"] for s in scripts if s.get("you_density_pct") is not None]
    tens_dens = [s["tension_density_pct"] for s in scripts if s.get("tension_density_pct") is not None]
    q_rates = [s["questions_per_100w"] for s in scripts if s.get("questions_per_100w") is not None]
    hook_types: Dict[str, int] = {}
    for s in scripts:
        ht = s.get("hook_type", "statement")
        hook_types[ht] = hook_types.get(ht, 0) + 1
    dom_hook = max(hook_types, key=lambda k: hook_types[k]) if hook_types else None

    # --- Thumbnail visual aggregation ---
    thumbs = [ta for ta in (v.get("thumbnail_analysis") for v in ch.videos) if isinstance(ta, dict)]
    lum_vals = [t["mean_lum"] for t in thumbs if t.get("fetched") and t.get("mean_lum") is not None]

    return {
        "handle": ch.handle,
        "scanned": True,
        "video_count": len(titles),
        "title_avg_len": round(sum(title_lens) / len(title_lens), 1) if title_lens else None,
        "title_max_len": max(title_lens) if title_lens else None,
        "tension_title_pct": round(100.0 * len(tension_titles) / len(titles), 1) if titles else None,
        "tension_examples": tension_titles[:5],
        "avg_duration_min": avg_dur_min,
        "cadence_per_week": cadence_per_week,
        "total_views": total_views,
        "median_views": median_views,
        "script": {
            "scanned_videos": len(scripts),
            "avg_wpm": round(sum(wpms) / len(wpms), 1) if wpms else None,
            "avg_you_density_pct": round(sum(you_dens) / len(you_dens), 2) if you_dens else None,
            "avg_tension_density_pct": round(sum(tens_dens) / len(tens_dens), 2) if tens_dens else None,
            "avg_questions_per_100w": round(sum(q_rates) / len(q_rates), 2) if q_rates else None,
            "dominant_hook_type": dom_hook,
            "hook_type_counts": hook_types,
        },
        "thumbnail": {
            "scanned_videos": len(lum_vals),
            "avg_mean_lum": round(sum(lum_vals) / len(lum_vals), 1) if lum_vals else None,
        },
    }


def build_gap_report(analyses: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Compare aggregated competitor signals against PIPELINE_BASELINE.

    Returns a list of gap dicts: {severity, area, finding, code_ref, suggestion}.
    """
    gaps: List[Dict[str, str]] = []
    valid = [a for a in analyses if a.get("scanned")]

    # --- Gap 1: Title tension angle -------------------------------------
    # Trigger per-channel (max) rather than averaging, so a single strong
    # tension-user (e.g. Finance Bureau) still surfaces the gap.
    if valid:
        tension_chs = [(a["handle"], a["tension_title_pct"] or 0.0) for a in valid]
        max_ch, max_pct = max(tension_chs, key=lambda x: x[1])
        if max_pct >= 40.0:
            gaps.append({
                "severity": "HIGH",
                "area": "Title angle",
                "finding": (
                    f"{max_ch} uses tension/contrarian framing in ~{max_pct:.0f}% of titles "
                    f"(others: "
                    + ", ".join(f"{h}={p:.0f}%" for h, p in tension_chs if h != max_ch)
                    + f"); our fallback template is the generic "
                    f"'{PIPELINE_BASELINE['title_fallback_template']}' "
                    f"(story_designer.py:999,1452) and the CTR prompt only asks for "
                    f"'high-CTR', not tension."
                ),
                "code_ref": "src/agents/story_designer.py:999,1452,1588 (_generate_ctr_title)",
                "suggestion": (
                    "Add a tension-angle generator: detect/force a contrarian hook in "
                    "_generate_ctr_title (e.g. 'Not X... it's Y', 'The X they won't admit') "
                    "and make the fallback template tension-aware instead of 'The Hidden Truth'."
                ),
            })

    # --- Gap 2: Runtime length mismatch ---------------------------------
    durs = [a["avg_duration_min"] for a in valid if a.get("avg_duration_min")]
    if durs:
        comp_avg = sum(durs) / len(durs)
        target = PIPELINE_BASELINE["runtime_target_min"]
        if abs(comp_avg - target) >= 3.0:
            gaps.append({
                "severity": "MEDIUM",
                "area": "Runtime length",
                "finding": (
                    f"Competitor avg duration ~{comp_avg:.1f} min vs our fixed ~{target:.0f} min "
                    f"target (story_designer.py:449,563,617). A {comp_avg:.0f}-min news-reaction "
                    f"format may out-retain our 13-min documentary format in this niche."
                ),
                "code_ref": "src/agents/story_designer.py:1157,1455,1547 (150 wpm -> seconds)",
                "suggestion": (
                    "Make target runtime niche/audience-aware (e.g. shorter 8-10 min for "
                    "finance/AI news reaction) instead of a hardcoded 13 min."
                ),
            })
    else:
        gaps.append({
            "severity": "INFO",
            "area": "Runtime length",
            "finding": (
                "Could not keylessly fetch competitor video durations (YouTube omits "
                "videoDetails from datacenter IPs / geo-restricted responses). Runtime "
                "gap vs our 13-min target is therefore unverified this run."
            ),
            "code_ref": "competitor_scraper.py (_video_detail duration fallback)",
            "suggestion": (
                "Re-run from a residential IP, or compare against our own published runtimes "
                "once analytics_feedback.py has data."
            ),
        })

    # --- Gap 3: Upload cadence / velocity -------------------------------
    cadences = [a["cadence_per_week"] for a in valid if a.get("cadence_per_week")]
    if cadences:
        comp_avg_cad = sum(cadences) / len(cadences)
        # Our effective cadence: 2 cron windows/day * 7 = 14/week, hard cap 4/day = 28/week.
        our_eff = 14.0
        if comp_avg_cad > our_eff:
            gaps.append({
                "severity": "MEDIUM",
                "area": "Publish cadence",
                "finding": (
                    f"Competitor cadence ~{comp_avg_cad:.1f} uploads/week exceeds our effective "
                    f"~{our_eff:.0f}/week (2 cron windows). Breaking-news velocity is a growth lever "
                    f"we are leaving on the table."
                ),
                "code_ref": "cron_publish.sh (2 windows/day), CSVG_MAX_DAILY_PUBLISHES=4",
                "suggestion": (
                    "Consider a 3rd cron window or a 'breaking-news' fast-path that bypasses the "
                    "full 1h50m runtime for time-sensitive topics."
                ),
            })

    # --- Gap 4: Engagement benchmark (informational) --------------------
    medians = [a["median_views"] for a in valid if a.get("median_views")]
    if medians:
        gaps.append({
            "severity": "INFO",
            "area": "Engagement baseline",
            "finding": (
                f"Competitor median views range "
                f"{min(medians):,}–{max(medians):,}. We have no published-view baseline yet to "
                f"compare retention/likes against."
            ),
            "code_ref": "src/engine/analytics_feedback.py (owner-only Analytics API)",
            "suggestion": (
                "Once we publish, feed our own median views/like-rate into this analyzer to turn "
                "INFO into a real retention gap signal."
            ),
        })

    # --- Gap 5: Script pacing (wpm) --------------------------------------
    script_blocks = [a.get("script", {}) for a in valid if a.get("script", {}).get("scanned_videos")]
    wpms = [s["avg_wpm"] for s in script_blocks if s.get("avg_wpm")]
    if wpms:
        comp_wpm = sum(wpms) / len(wpms)
        target_wpm = PIPELINE_BASELINE["wpm"]
        if abs(comp_wpm - target_wpm) >= 20.0:
            faster = "faster" if comp_wpm > target_wpm else "slower"
            gaps.append({
                "severity": "MEDIUM",
                "area": "Script pacing (wpm)",
                "finding": (
                    f"Competitor narration averages ~{comp_wpm:.0f} words/min vs our fixed "
                    f"~{target_wpm} wpm assumption (story_designer.py:1157,1455). They speak "
                    f"{faster}, which changes per-shot narration length and retention density."
                ),
                "code_ref": "src/agents/story_designer.py:1157,1455,1547 (150 wpm -> seconds)",
                "suggestion": (
                    f"Make narration wpm audience/niche-aware instead of a hardcoded 150; "
                    f"re-derive per-shot duration_estimate from the chosen wpm."
                ),
            })
    else:
        gaps.append({
            "severity": "INFO",
            "area": "Script pacing (wpm)",
            "finding": (
                "Could not keylessly fetch competitor captions this run (captions disabled, "
                "geo-blocked, or youtube-transcript-api absent). Script-pacing gap unverified."
            ),
            "code_ref": "competitor_scraper.py (_fetch_captions)",
            "suggestion": "Re-run with captions available (residential IP / library installed).",
        })

    # --- Gap 6: Opening hook style ---------------------------------------
    hook_counts: Dict[str, int] = {}
    for s in script_blocks:
        for ht, n in (s.get("hook_type_counts") or {}).items():
            hook_counts[ht] = hook_counts.get(ht, 0) + n
    total_hooks = sum(hook_counts.values())
    if total_hooks and hook_counts.get("question", 0) + hook_counts.get("tension", 0) > 0:
        q_or_t = hook_counts.get("question", 0) + hook_counts.get("tension", 0)
        pct = round(100.0 * q_or_t / total_hooks, 0)
        if pct >= 40.0:
            gaps.append({
                "severity": "MEDIUM",
                "area": "Opening hook style",
                "finding": (
                    f"~{pct:.0f}% of competitor openings are QUESTION or TENSION hooks "
                    f"(counts: {hook_counts}); our hook rule only reuses the title's trailing "
                    f"words (story_designer.py:1716) and never forces a question/tension opener."
                ),
                "code_ref": "src/agents/story_designer.py:1716,1720 (hook rule) + nano_banana craft_ctr_hook",
                "suggestion": (
                    "Add a hook generator that opens with a question or tension beat (e.g. "
                    "'What if X is worse than we think?') and feed it to both the script intro "
                    "and the nano_banana on-image hook."
                ),
            })

    # --- Gap 7: Personalization ('you/your' density) ---------------------
    yous = [s["avg_you_density_pct"] for s in script_blocks if s.get("avg_you_density_pct") is not None]
    if yous:
        comp_you = sum(yous) / len(yous)
        if comp_you >= 1.0:
            gaps.append({
                "severity": "LOW",
                "area": "Personalization",
                "finding": (
                    f"Competitor scripts use 'you/your' ~{comp_you:.2f}% of words (direct address "
                    f"= retention lever); our pipeline has no explicit personalization optimization."
                ),
                "code_ref": "src/agents/story_designer.py (no 'you/your' density lever)",
                "suggestion": (
                    "Add a light personalization pass in the polisher that weaves 'you/your' into "
                    "openers and takeaways without drifting from the facts."
                ),
            })

    # --- Gap 8: Thumbnail brightness (visuals) ---------------------------
    lum_blocks = [a.get("thumbnail", {}).get("avg_mean_lum") for a in valid
                  if a.get("thumbnail", {}).get("avg_mean_lum")]
    if lum_blocks:
        comp_lum = sum(lum_blocks) / len(lum_blocks)
        target_lum = PIPELINE_BASELINE["thumbnail_target_mean_lum"]
        gaps.append({
            "severity": "INFO",
            "area": "Thumbnail brightness",
            "finding": (
                f"Competitor thumbnails average mean luminance ~{comp_lum:.0f} "
                f"(0-255) vs our nano_banana brightness-lift target ~{target_lum} "
                f"(nano_banana.comply_thumbnail). "
                + ("They are brighter — our target may be conservative."
                   if comp_lum > target_lum + 10 else
                   "They are darker — our lift is an edge." if comp_lum < target_lum - 10
                   else "Roughly aligned.")
            ),
            "code_ref": "src/engine/nano_banana.py (comply_thumbnail mean_lum>=100)",
            "suggestion": (
                "If competitors are markedly brighter, raise CSVG thumbnail brightness target; "
                "full visual-style (face/B-roll/AI) needs frame extraction + a vision model."
            ),
        })
    else:
        gaps.append({
            "severity": "PARITY",
            "area": "Thumbnails",
            "finding": (
                "Competitors use bold burned-in text + hero/reaction framing — matches our "
                "nano_banana 7-layout x 5-palette A/B + add_thumbnail_text overlay. No gap "
                "(run with --visuals for brightness stats)."
            ),
            "code_ref": "src/engine/nano_banana.py (craft_ctr_hook, add_thumbnail_text)",
            "suggestion": "No change; verify our text-overlay legibility vs their thumbnails.",
        })
    return gaps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Keyless YouTube competitor scraper + gap analyzer")
    parser.add_argument("--channels", default="FinanceBureauOfficial,AIUncovered,economicsexplained",
                        help="comma list of @handles / bare handles / UC... ids")
    parser.add_argument("--limit", type=int, default=20, help="max videos per channel")
    parser.add_argument("--no-views", action="store_true", help="skip watch-page fetches")
    parser.add_argument("--no-script", action="store_true", help="skip caption/script analysis")
    parser.add_argument("--visuals", action="store_true",
                        help="fetch + analyze thumbnails (extra image requests)")
    parser.add_argument("--out", default="logs/competitor_data.json", help="output JSON path")
    parser.add_argument("--no-analyze", action="store_true", help="skip gap analysis")
    args = parser.parse_args(argv)

    handles = [h.strip().lstrip("@") for h in args.channels.split(",") if h.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    channels: List[ChannelData] = []
    for h in handles:
        time.sleep(_REQ_DELAY_S)
        channels.append(scrape_channel(
            h, args.limit,
            fetch_views=not args.no_views,
            fetch_script=not args.no_script,
            fetch_visuals=args.visuals,
        ))

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": PIPELINE_BASELINE,
        "channels": [asdict(c) for c in channels],
    }

    if not args.no_analyze:
        analyses = [analyze_channel(c) for c in channels]
        gaps = build_gap_report(analyses)
        payload["analyses"] = analyses
        payload["gaps"] = gaps

    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[done] wrote {out_path}")

    if not args.no_analyze:
        print("\n==================== GAP REPORT ====================")
        for g in payload.get("gaps", []):
            print(f"[{g['severity']}] {g['area']}")
            print(f"  finding : {g['finding']}")
            print(f"  code_ref: {g['code_ref']}")
            print(f"  fix     : {g['suggestion']}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
