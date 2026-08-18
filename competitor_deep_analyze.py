#!/usr/bin/env python3
"""Deep competitor video/audio analyzer (keyless, node-aware).

Builds on ``competitor_scraper.py`` but goes further: for each competitor
video it DOWNLOADS the media (yt-dlp, residential-IP friendly — run this on
the Raspberry Pi), then:

  * AUDIO  - transcribes with faster-whisper, computes speech coverage
             (=> BGM/SFX presence proxy), narration wpm, segment pacing.
  * VIDEO  - detects shots via ffmpeg scene-cut, samples frames, and
             classifies visual style (face/reaction, B-roll/stock, data
             charts, AI-generated, on-screen text) with Gemini vision
             (GEMINI_API_KEY) when available; falls back to PIL brightness.

Everything is optional/guarded: missing yt-dlp / faster-whisper / ffmpeg /
Gemini degrade gracefully and the report notes what was skipped. No YouTube
Data API, no edits to core pipeline code.

Run on the Pi (residential IP) for reliable downloads + captions:
    python competitor_deep_analyze.py --channels "FinanceBureauOfficial" --limit 3
    python competitor_deep_analyze.py --video-ids "5p1Xi9KEUiE,abcd1234" --frames 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse the safe, keyless helpers from the lightweight scraper (no network at
# import time, no core-code changes).
from competitor_scraper import (  # noqa: E402
    _resolve_channel_id,
    _parse_rss,
    _scrape_videos_tab,
    _fetch_captions,
    _analyze_script,
    _analyze_thumbnail,
    _TENSION_LEXICON,
    PIPELINE_BASELINE,
    _session,
)

_REQ_DELAY_S = 0.5


# ---------------------------------------------------------------------------
# Media download (yt-dlp) — run where YouTube allows it (residential IP / Pi)
# ---------------------------------------------------------------------------
def _have(bin_name: str) -> bool:
    return shutil.which(bin_name) is not None


def download_media(video_id: str, workdir: str) -> Dict[str, Optional[str]]:
    """Download a low-res copy (video+audio) and an audio-only mp3.

    Returns paths (or None) for ``video`` and ``audio``. Best-effort; callers
    must handle missing files.
    """
    out: Dict[str, Optional[str]] = {"video": None, "audio": None}
    if not _have("yt-dlp"):
        return out
    base = os.path.join(workdir, f"{video_id}")
    # Audio-only mp3 (for whisper transcription).
    try:
        subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "mp3", "-o", f"{base}.%(ext)s",
             f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=300,
        )
        for ext in ("mp3", "m4a", "webm", "ogg"):
            p = f"{base}.{ext}"
            if os.path.exists(p):
                out["audio"] = p
                break
    except Exception:
        pass
    # Low-res video (for frames / shot detection).
    try:
        subprocess.run(
            ["yt-dlp", "-f", "bestvideo[height<=360]+bestaudio/best[height<=360]",
             "-o", f"{base}.mp4", f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=300,
        )
        if os.path.exists(f"{base}.mp4"):
            out["video"] = f"{base}.mp4"
    except Exception:
        pass
    return out


def _probe_duration(path: Optional[str]) -> Optional[float]:
    if not _have("ffprobe") or not path or not os.path.exists(path):
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=60,
        )
        return float(r.stdout.strip()) if r.stdout.strip() else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Audio: transcription + structural analysis
# ---------------------------------------------------------------------------
def transcribe_audio(audio_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Transcribe audio with faster-whisper. Returns {text, segments, words}."""
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(audio_path, beam_size=5)
        segs = []
        words = 0
        for s in segments:
            segs.append({"start": round(s.start, 2), "end": round(s.end, 2),
                         "text": s.text.strip()})
            words += len(s.text.split())
        text = " ".join(s["text"] for s in segs)
        return {"text": text, "segments": segs, "word_count": words}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def analyze_audio(audio_path: Optional[str], transcript: Optional[Dict[str, Any]],
                  duration: Optional[float]) -> Dict[str, Any]:
    """Derive audio-shaping signals (BGM proxy, voice pacing) from whisper."""
    out: Dict[str, Any] = {"transcribed": bool(transcript and transcript.get("text"))}
    if not audio_path:
        return out
    if not transcript or not transcript.get("text"):
        out["error"] = "no transcript"
        return out
    segs = transcript.get("segments", [])
    dur = duration or (segs[-1]["end"] if segs else None)
    speech_cov = None
    if dur and dur > 0 and segs:
        speech = sum((s["end"] - s["start"]) for s in segs)
        speech_cov = round(100.0 * speech / dur, 1)
    # Speech coverage < ~80% implies meaningful non-speech audio (BGM/SFX).
    bgm = None
    if speech_cov is not None:
        bgm = speech_cov < 80.0
    wpm = None
    if dur and dur > 0:
        wpm = round(transcript["word_count"] / (dur / 60.0), 1)
    out.update({
        "duration_sec": round(dur, 1) if dur else None,
        "word_count": transcript["word_count"],
        "wpm": wpm,
        "speech_coverage_pct": speech_cov,
        "bgm_or_sfx_likely": bgm,
        "segment_count": len(segs),
        "avg_segment_sec": round(sum((s["end"] - s["start"]) for s in segs) / len(segs), 2) if segs else None,
        "voice_note": "single narrator (AI vs human not distinguishable from audio alone)",
    })
    return out


# ---------------------------------------------------------------------------
# Video: shot detection + frame sampling
# ---------------------------------------------------------------------------
def detect_shots(video_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not _have("ffmpeg") or not video_path or not os.path.exists(video_path):
        return None
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", video_path, "-filter:v",
             "select='gt(scene,0.4)',showinfo", "-f", "null", "-"],
            capture_output=True, text=True, timeout=300,
        )
        # Count shot-change events from showinfo timestamps.
        pts = re.findall(r"pts_time:([\d.]+)", r.stderr)
        changes = [float(x) for x in pts]
        dur = _probe_duration(video_path)
        shot_count = len(changes) + 1 if changes else None
        avg = round(dur / shot_count, 1) if (shot_count and dur) else None
        return {
            "shot_count": shot_count,
            "avg_shot_sec": avg,
            "scene_threshold": 0.4,
        }
    except Exception:
        return None


def sample_frames(video_path: Optional[str], outdir: str, n: int, duration: Optional[float]) -> List[str]:
    if not _have("ffmpeg") or not video_path or not os.path.exists(video_path):
        return []
    os.makedirs(outdir, exist_ok=True)
    fps = f"{n}/{duration:.0f}" if duration and duration > 0 else f"{n}/600"
    pat = os.path.join(outdir, "frame_%03d.jpg")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}",
             "-frames:v", str(n), pat],
            capture_output=True, text=True, timeout=300,
        )
        return sorted(
            str(p) for p in Path(outdir).glob("frame_*.jpg")
        )[:n]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Visual classification: Gemini vision (preferred) or PIL brightness (fallback)
# ---------------------------------------------------------------------------
def _gemini_vision_classify(frame_paths: List[str], api_key: str) -> List[Dict[str, Any]]:
    """Classify each frame via Gemini 2.5 Flash vision. Returns per-frame dicts."""
    import base64
    import requests

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    results: List[Dict[str, Any]] = []
    prompt = (
        "You are analyzing a YouTube video frame for a competitor-gap study. "
        "Respond ONLY with compact JSON: "
        '{"style": one of [face_reaction, b_roll_stock, data_chart, ai_generated, '
        'screen_recording, text_card, other], "has_text_overlay": true/false, '
        '"brightness": 1-10, "description": "<=12 words>"}'
    )
    for fp in frame_paths:
        try:
            with open(fp, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                    ]
                }],
                "generationConfig": {"responseMimeType": "application/json"},
            }
            resp = requests.post(url, json=payload, timeout=60)
            data = resp.json()
            txt = data["candidates"][0]["content"]["parts"][0]["text"]
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if m:
                results.append(json.loads(m.group(0)))
            else:
                results.append({"raw": txt})
        except Exception as exc:  # noqa: BLE001
            results.append({"error": str(exc)})
    return results


def analyze_frames(frames: List[str], use_vision: bool) -> Dict[str, Any]:
    """Classify sampled frames; aggregate style mix + brightness."""
    out: Dict[str, Any] = {"frame_count": len(frames), "method": None}
    if not frames:
        return out
    # Always compute brightness via PIL (light, keyless).
    try:
        from PIL import Image
        import io
        lums: List[float] = []
        for fp in frames:
            try:
                img = Image.open(fp).convert("RGB")
                px = list(img.getdata())
                n = len(px)
                if n:
                    rs = gs = bs = 0
                    for (r, g, b) in px:
                        rs += r; gs += g; bs += b
                    lums.append((0.2126 * rs + 0.7152 * gs + 0.0722 * bs) / n)
            except Exception:
                pass
        if lums:
            out["avg_frame_lum"] = round(sum(lums) / len(lums), 1)
            out["method"] = "pillow_brightness"
    except ImportError:
        pass

    if use_vision:
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            cls = _gemini_vision_classify(frames, key)
            out["method"] = "gemini_vision"
            out["frames"] = cls
            styles: Dict[str, int] = {}
            text_frames = 0
            for c in cls:
                s = c.get("style")
                if s:
                    styles[s] = styles.get(s, 0) + 1
                if c.get("has_text_overlay"):
                    text_frames += 1
            total = len(cls) or 1
            out["style_mix"] = styles
            out["text_overlay_pct"] = round(100.0 * text_frames / total, 1)
        else:
            out["vision_note"] = "GEMINI_API_KEY not set; used brightness fallback"
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def deep_analyze_video(video_id: str, use_vision: bool = True,
                       frames: int = 8, channel: str = "",
                       title: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {"video_id": video_id, "channel": channel, "title": title}
    tmp = tempfile.mkdtemp(prefix=f"csv_deep_{video_id}_")
    try:
        media = download_media(video_id, tmp)
        dur = _probe_duration(media.get("video") or media.get("audio"))
        dur_int = int(dur) if dur else None
        # Transcript: prefer whisper (works on residential IP); fall back to
        # keyless captions (also fine from residential IP).
        transcript = None
        if media.get("audio"):
            transcript = transcribe_audio(media["audio"])
        if not (transcript and transcript.get("text")):
            cap = _fetch_captions(video_id)
            if cap:
                transcript = {"text": cap, "segments": [], "word_count": len(cap.split())}
        if transcript and transcript.get("text"):
            result["script_analysis"] = _analyze_script(transcript["text"], dur_int)
            result["transcript_excerpt"] = transcript["text"][:300]
        if media.get("audio"):
            result["audio_analysis"] = analyze_audio(media["audio"], transcript, dur)
        if media.get("video"):
            result["shots"] = detect_shots(media["video"])
            fr = sample_frames(media["video"], os.path.join(tmp, "frames"), frames, dur)
            if fr:
                result["visual_analysis"] = analyze_frames(fr, use_vision)
        thumb = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        result["thumbnail_analysis"] = _analyze_thumbnail(thumb)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Flatten nested analyses into dashboard-friendly top-level fields while
    # preserving the nested structures (consumed by build_deep_gap_report).
    sc = result.get("script_analysis") or {}
    result["duration_sec"] = sc.get("duration_sec")
    au = result.get("audio_analysis") or {}
    result["wpm"] = au.get("wpm")
    result["bgm_level"] = au.get("bgm_or_sfx_likely")
    sh = result.get("shots") or {}
    result["shot_count"] = sh.get("shot_count")
    va = result.get("visual_analysis") or {}
    result["visual_style_mix"] = va.get("style_mix") or {}
    return result


# ---------------------------------------------------------------------------
# Channel iteration + deep gap report
# ---------------------------------------------------------------------------
def collect_video_ids(handle: str, limit: int) -> List[Dict[str, str]]:
    cid = _resolve_channel_id(handle)
    if not cid:
        return []
    rss = _session().get(f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}")
    items: List[Dict[str, str]] = []
    if rss and rss.status_code == 200:
        items = [{"video_id": v["video_id"], "title": v.get("title", "")}
                 for v in _parse_rss(rss.text)[:limit]]
    if not items:
        items = [{"video_id": v["video_id"], "title": v.get("title", "")}
                 for v in _scrape_videos_tab(cid, handle, limit)[:limit]]
    return items


def build_deep_gap_report(videos: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    gaps: List[Dict[str, str]] = []
    shots = [v.get("shots", {}) for v in videos if v.get("shots")]
    shot_counts = [s["shot_count"] for s in shots if s.get("shot_count")]
    avg_shots = round(sum(shot_counts) / len(shot_counts), 1) if shot_counts else None

    auds = [v.get("audio_analysis", {}) for v in videos if v.get("audio_analysis")]
    bgm = [a["bgm_or_sfx_likely"] for a in auds if a.get("bgm_or_sfx_likely") is not None]
    wpms = [a["wpm"] for a in auds if a.get("wpm")]
    bgm_pct = round(100.0 * sum(1 for b in bgm if b) / len(bgm), 0) if bgm else None

    vis = [v.get("visual_analysis", {}) for v in videos if v.get("visual_analysis")]
    style_mixes: Dict[str, int] = {}
    for vx in vis:
        for s, n in (vx.get("style_mix") or {}).items():
            style_mixes[s] = style_mixes.get(s, 0) + n
    text_pcts = [vx["text_overlay_pct"] for vx in vis if vx.get("text_overlay_pct") is not None]

    # Gap A: shot count vs our ~15-18 target shots.
    if avg_shots is not None:
        tgt = 15
        if abs(avg_shots - tgt) >= 5:
            gaps.append({
                "severity": "MEDIUM",
                "area": "Shot density / pacing",
                "finding": (
                    f"Competitor avg ~{avg_shots:.0f} shots/video; our pipeline targets "
                    f"~{tgt} shots (story_designer target_shots). "
                    f"{'They cut faster (more shots)' if avg_shots > tgt else 'They use longer holds'}."
                ),
                "code_ref": "src/agents/story_designer.py (target_shots, --tail padding)",
                "suggestion": "Tune target_shots / --tail hold per niche to match competitor retention pacing.",
            })

    # Gap B: BGM/SFX usage.
    if bgm_pct is not None:
        gaps.append({
            "severity": "INFO",
            "area": "Background music / SFX",
            "finding": (
                f"~{bgm_pct:.0f}% of analyzed videos show notable non-speech audio "
                f"(BGM/SFX likely, via speech-coverage proxy). Our pipeline uses BGM with a "
                f"sidechain duck (media_cloud). Confirm our music bed matches competitor intensity."
            ),
            "code_ref": "mcp_servers/media_cloud/server.py (BGM sidechain)",
            "suggestion": "If competitors are music-heavy, raise BGM_VOLUME / lower duck threshold.",
        })

    # Gap C: narration wpm.
    if wpms:
        comp = sum(wpms) / len(wpms)
        if abs(comp - PIPELINE_BASELINE["wpm"]) >= 20:
            gaps.append({
                "severity": "MEDIUM",
                "area": "Narration pace (wpm)",
                "finding": (
                    f"Competitor narration ~{comp:.0f} wpm vs our hardcoded {PIPELINE_BASELINE['wpm']} "
                    f"wpm assumption."
                ),
                "code_ref": "src/agents/story_designer.py:1157,1455 (150 wpm -> seconds)",
                "suggestion": "Make narration wpm niche-aware instead of hardcoded 150.",
            })

    # Gap D: visual style mix.
    if style_mixes:
        gaps.append({
            "severity": "MEDIUM",
            "area": "Visual style mix",
            "finding": (
                f"Competitor frame style mix: {style_mixes}. "
                f"Avg on-screen text overlay: "
                f"{round(sum(text_pcts)/len(text_pcts),0) if text_pcts else 'n/a'}%. "
                f"Our pipeline leans matplotlib charts + AI visuals (nano_banana)."
            ),
            "code_ref": "src/agents/media_producer.py (charts), src/engine/nano_banana.py",
            "suggestion": (
                "If competitors favor face_reaction / b_roll_stock over data_chart, consider "
                "blending more reaction/B-roll cutaways into our shot mix."
            ),
        })
    return gaps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Deep competitor video/audio analyzer")
    p.add_argument("--channels", default="FinanceBureauOfficial",
                   help="comma list of @handles / bare handles / UC... ids")
    p.add_argument("--video-ids", default="", help="comma list of specific video IDs")
    p.add_argument("--limit", type=int, default=3, help="videos per channel")
    p.add_argument("--frames", type=int, default=8, help="frames sampled per video")
    p.add_argument("--no-vision", action="store_true", help="skip Gemini vision (brightness only)")
    p.add_argument("--out", default="logs/competitor_deep.json", help="output JSON")
    args = p.parse_args(argv)

    explicit: List[str] = [v.strip() for v in args.video_ids.split(",") if v.strip()]
    items: List[Dict[str, str]] = [{"video_id": v, "title": ""} for v in explicit]
    if not items:
        for h in [x.strip().lstrip("@") for x in args.channels.split(",") if x.strip()]:
            time.sleep(_REQ_DELAY_S)
            items.extend(collect_video_ids(h, args.limit))

    print(f"[deep] analyzing {len(items)} videos (residential IP / Pi recommended)")
    videos = []
    for it in items:
        vid = it["video_id"]
        time.sleep(_REQ_DELAY_S)
        print(f"[deep] {vid} ...")
        videos.append(deep_analyze_video(
            vid, use_vision=not args.no_vision, frames=args.frames,
            channel=args.channels, title=it.get("title", "")))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": PIPELINE_BASELINE,
        "videos": videos,
        "deep_gaps": build_deep_gap_report(videos),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[deep] wrote {args.out}")
    print("\n=================== DEEP GAP REPORT ===================")
    for g in payload["deep_gaps"]:
        print(f"[{g['severity']}] {g['area']}: {g['finding']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
