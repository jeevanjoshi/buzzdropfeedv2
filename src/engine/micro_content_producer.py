import os
import re
import subprocess
import logging
from typing import List, Optional, Dict, Any
from src.schemas.state import GlobalState, VisualType

logger = logging.getLogger("CSVG_PIPELINE")

# Visual types that are deadly on Shorts (static data walls read as "boring" on
# a vertical, sound-on, mobile-first feed and crater CTR). Never include them in
# a Short — the trailer is built exclusively from cinematic / narrative beats.
_CHART_TYPES = frozenset({VisualType.MATPLOTLIB_CHART, VisualType.SVG_TICKER})

# Trailer tuning (all env-overridable).
_SEG_LEN = float(os.getenv("CSVG_SHORTS_SEG_LEN", "8.0"))          # seconds per montage beat
_MAX_SEGS = int(os.getenv("CSVG_SHORTS_SEGS", "5"))                # beats per Short
_TRAILER = os.getenv("CSVG_SHORTS_TRAILER", "1").strip().lower() not in ("0", "false", "no")
_XFADE = float(os.getenv("CSVG_SHORTS_XFADE", "0.35"))             # beat-to-beat dissolve

# Per-act dramatic weight — a Shorts trailer should climb toward the Act 5
# reveal / Act 6 verdict, not flatten the arc.
_ACT_TENSION = {1: 0.9, 2: 0.55, 3: 0.85, 4: 0.9, 5: 1.0, 6: 0.8}

# Narration signals that flag a "tension / hook" beat worth cutting into a trailer.
_HOOK_WORDS = (
    "secret", "shocking", "hidden", "never", "reveals", "what happens", "truth",
    "exposed", "insane", "unbelievable", "crisis", "collapse", "surge", "warning",
    "you won't believe", "this is what", "here's why", "the catch", "backfire",
    "explosion", "meltdown", "betrayal", "winner", "loser",
)


def _probe_duration(path: str) -> float:
    """Return the real decodable duration (seconds) of a media file, or 0.0."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30)
        v = float(r.stdout.strip())
        return v if v > 0 else 0.0
    except Exception:
        return 0.0


def _probe_speech_end(video_path: str, start_s: float, max_dur: float,
                      noise_db: float = -35.0, min_sil: float = 0.25) -> Optional[float]:
    """Return the time (relative to `start_s`) of the last spoken audio inside the
    window [start_s, start_s+max_dur], or None if it cannot be determined.

    Used to end Shorts on a natural breath instead of mid-word. ffmpeg's
    silencedetect timestamps are relative to the seeked (0-based) window because
    `-ss` precedes `-i`.
    """
    try:
        cmd = [
            "ffmpeg", "-ss", f"{start_s:.2f}", "-i", video_path,
            "-t", f"{max_dur:.2f}",
            "-af", f"silencedetect=n={noise_db}:d={min_sil}",
            "-f", "null", "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              text=True, timeout=90)
        txt = proc.stderr or ""
        silence_starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", txt)]
        silence_ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", txt)]
        last_ss = silence_starts[-1] if silence_starts else None
        last_se = silence_ends[-1] if silence_ends else None
        # Speech ends where the final silence begins (trailing silence, or silence
        # that runs past the window end). Otherwise speech continues to max_dur.
        if last_ss is not None and (last_se is None or last_ss >= last_se):
            speech_end = min(max_dur, last_ss + 0.15)
            if speech_end >= 2.0:
                return speech_end
        return max_dur
    except Exception:
        return None


_COVER_ENABLED = os.getenv("CSVG_SHORTS_COVERS", "1").strip().lower() not in ("0", "false", "no")


def _shorts_hook(state: GlobalState) -> str:
    """Shorts cover text = a short, tension/curiosity phrase that makes a scroller
    stop. Prefer the SEO *title* (which is already written as a curiosity gap,
    e.g. "Why the FCC Wants Your Internet to Stay Slow") over the bare keyword
    `thumbnail_brief`, then the topic headline as fallback."""
    hook = ""
    seo = state.seo_metadata
    if seo and seo.title:
        hook = seo.title
    elif state.selected_topic and state.selected_topic.headline:
        hook = state.selected_topic.headline
    elif seo and seo.thumbnail_brief:
        hook = seo.thumbnail_brief
    try:
        from mcp_servers.media_cloud.server import _shorten_thumbnail_text
        return _shorten_thumbnail_text(hook or "WATCH THIS")
    except Exception:
        words = (hook or "WATCH THIS").split()[:4]
        return " ".join(words).upper() if words else "WATCH THIS"


def _fit_cover_font(text: str, font_path: str, max_width: int = 940, hi: int = 260, lo: int = 120):
    from PIL import ImageFont
    for size in range(hi, lo - 1, -2):
        font = ImageFont.truetype(font_path, size)
        l, t, r, b = font.getbbox(text)
        if (r - l) <= max_width:
            return font
    return ImageFont.truetype(font_path, lo)


def _compose_shorts_cover(frame_png: str, out_png: str, hook: str) -> None:
    """
    Burns the bold hook onto a single 9:16 frame → the dedicated cover Shorts
    creators pick during upload:
    - Single focal point (one line of bold text over a dimmed scene).
    - Text centred horizontally in the TOP two-thirds (bottom 20% + right 15%
      are UI/channel-avatar/engagement safe zones and stay clear).
    - High-contrast heavy sans-serif outline + drop shadow + 10% accent bar.
    """
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    img = Image.open(frame_png).convert("RGB")
    W, H = img.size
    if W != 1080 or H != 1920:
        img = img.resize((1080, 1920), Image.LANCZOS)
    # Lightly dim the scene so the single focal element (hook) pops — keep it
    # bright enough to stay catchy on mobile (research: high-contrast, not dark).
    img = img.point(lambda p: int(p * 0.82))
    base = img.convert("RGBA")

    # Bottom 20% (titles / engagement row) + right 15% (Like/Comment/Share)
    # safe-zone dims — pure PIL so no numpy dependency here.
    grad = Image.new("L", (W, H), 0)
    gd = ImageDraw.Draw(grad)
    for yy in range(H, int(H * 0.80), -1):
        gd.line([(0, yy), (W, yy)], fill=int(round(210 * (H - yy) / (H * 0.20))))
    for xx in range(W, int(W * 0.85), -1):
        gd.line([(xx, 0), (xx, H)], fill=int(round(110 * (W - xx) / (W * 0.15))))
    dark = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dark.putalpha(grad)
    base = Image.alpha_composite(base, dark)

    try:
        from mcp_servers.media_cloud.server import _bold_font_path
        font_path = _bold_font_path()
    except Exception:
        font_path = None
    if not font_path:
        base.convert("RGB").save(out_png, "PNG")
        return
    font = _fit_cover_font(hook, font_path, max_width=int(W * 0.87), hi=int(H * 0.16), lo=int(H * 0.065))
    l, t, r, b = font.getbbox(hook)
    txt_w, txt_h = r - l, b - t
    x0 = (W - txt_w) // 2
    y0 = int(H * 0.30) - txt_h  # centered in the top two-thirds
    d = ImageDraw.Draw(base)

    # Soft blurred drop shadow for separation on any scene.
    sh = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.text((x0, y0 + 8), hook, font=font, fill=(0, 0, 0, 200), stroke_width=4, stroke_fill=(0, 0, 0, 200))
    sh = sh.filter(ImageFilter.GaussianBlur(9))
    base = Image.alpha_composite(base, sh)
    d = ImageDraw.Draw(base)
    d.text((x0, y0), hook, font=font, fill=(255, 255, 255, 255),
           stroke_width=12, stroke_fill=(20, 20, 24, 255))
    # 10% accent: a bold amber bar (60-30-10) under the hook.
    bar_w = max(int(txt_w * 0.45), 180)
    bar_y = y0 + txt_h + 18
    d.rounded_rectangle([x0 + 4, bar_y, x0 + 4 + bar_w, bar_y + 16], radius=8, fill=(255, 211, 25, 255))

    base.convert("RGB").save(out_png, "PNG")


def _extract_short_frame(final_video: str, start_s: float, out_png: str) -> bool:
    """Pull one sharp 9:16 centre-cropped 1080x1920 frame from the master."""
    try:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{max(0.0, start_s) + 1.3:.2f}", "-i", final_video,
            "-frames:v", "1",
            "-vf", "crop=ih*9/16:ih,scale=1080:1920:flags=lanczos",
            "-q:v", "2", out_png,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True, timeout=30)
        return os.path.exists(out_png) and os.path.getsize(out_png) > 1000
    except Exception:
        return False


def _cover_to_video(cover_png: str, out_mp4: str) -> bool:
    """1-second silent 1080x1920 cover clip (the burned-in frame creators pick
    as the Shorts cover)."""
    try:
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", cover_png,
            "-f", "lavfi", "-t", "1", "-i", "anullsrc=r=44100:cl=mono",
            "-t", "1", "-r", "25", "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "96k", "-shortest", out_mp4,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True, timeout=120)
        return os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 1000
    except Exception:
        return False


def _prepend_cover(clip: str, cover_mp4: str, out_mp4: str) -> bool:
    """Splice the 1s cover to the front of the Short (same 1080x1920 / 25fps)."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", cover_mp4, "-i", clip,
            "-filter_complex",
            "[0:v]fps=25,settb=AVTB,scale=1080:1920,setsar=1[v0];"
            "[1:v]fps=25,settb=AVTB,setsar=1[v1];"
            "[0:a]aresample=44100,asetpts=PTS-STARTPTS[a0];"
            "[1:a]aresample=44100,asetpts=PTS-STARTPTS[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]",
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out_mp4,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True, timeout=180)
        return os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 1000
    except Exception:
        return False


def _shot_timeline(state: GlobalState) -> List[Dict[str, Any]]:
    """Map the master's real (ffprobe-measured) timeline into per-shot windows
    with the visual type, so we can exclude chart/ticker frames from Shorts."""
    shots = state.script_data.shots if state.script_data else []
    durs = state.asset_paths.measured_durations if state.asset_paths else []
    timeline: List[Dict[str, Any]] = []
    t = 0.0
    for idx, shot in enumerate(shots):
        d = durs[idx] if idx < len(durs) else (shot.duration_estimate or 5.0)
        d = max(float(d), 1.0)
        timeline.append({
            "idx": idx,
            "start": t,
            "end": t + d,
            "dur": d,
            "act": shot.act_index,
            "chart": shot.visual_type in _CHART_TYPES,
            "narration": (shot.narration_text or ""),
        })
        t += d
    return timeline


def _seg_score(seg: Dict[str, Any]) -> float:
    """Dramatic-tension score for a candidate Short beat."""
    s = _ACT_TENSION.get(seg["act"], 0.5)
    n = seg["narration"].lower()
    if any(w in n for w in _HOOK_WORDS):
        s += 0.25
    # Slightly favour longer, self-contained beats (more room for a hook line).
    s += min(seg["dur"], 10.0) * 0.01
    return s


def _select_trailer_segments(timeline: List[Dict[str, Any]], max_shorts: int,
                             clip_idx: int) -> List[Dict[str, Any]]:
    """Pick the gripping, non-chart beats for Short `clip_idx`.

    Beats are ranked by tension, then rotated per-Short so each published Short
    is a different mini-trailer (Short 1 = top-ranked beats, Short 2 = next
    band). Never returns chart/ticker shots.
    """
    cands = [s for s in timeline if (not s["chart"]) and s["dur"] >= 3.0]
    if not cands:
        # Degenerate: every beat is a chart — fall back to non-empty beats so we
        # still ship *something* (better than a blank Short), but this is rare.
        cands = [s for s in timeline if s["dur"] >= 3.0] or list(timeline)
    ranked = sorted(cands, key=_seg_score, reverse=True)
    offset = (clip_idx - 1) * max(1, _MAX_SEGS // max(1, max_shorts))
    selected = ranked[offset:offset + _MAX_SEGS] or ranked[:_MAX_SEGS]
    # A trailer plays in narrative order, not ranked order.
    return sorted(selected, key=lambda s: s["start"])


class MicroContentProducer:
    """
    Automated short-form micro-content producer.
    Extracts high-impact 30-60s acts/shots from the rendered 16:9 master video,
    crops to 9:16 vertical ratio (Shorts / Reels / TikTok format), and — per the
    Shorts spec — burns in a dedicated high-impact 1-second hook cover frame at
    the START so a strong cover can be selected during mobile upload instead of a
    random mid-video frame. Covers follow the safe zones (focal content centred
    in the top two-thirds; bottom 20% / right 15% kept clear).
    """

    def __init__(self, output_dir: str = "logs/shorts"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_shorts(self, state: GlobalState, max_shorts: int = 2) -> List[str]:
        """
        Build up to `max_shorts` vertical 9:16 Shorts that read as tension-gripping
        *trailers* of the long-form master — not a flat crop of one act.

        Each Short is a montage of the most dramatic, non-chart narrative beats
        (Act 1 hook → Act 5 reveal → Act 6 verdict) cut together with quick
        xfade dissolves and a burned-in hook cover at the very front. Chart /
        ticker frames are never selected, because static data walls crater Shorts
        CTR on a sound-on, mobile-first feed.
        """
        generated_clips: List[str] = []
        pipeline_id = state.pipeline_id or "run"
        final_video = state.asset_paths.final_video if state.asset_paths else None

        if not final_video or not os.path.exists(final_video):
            logger.info("[MICRO_CONTENT] Master video not present on disk; generating mock short clip metadata.")
            for i in range(1, max_shorts + 1):
                mock_path = os.path.join(self.output_dir, f"short_{pipeline_id}_clip{i}.mp4")
                generated_clips.append(mock_path)
            return generated_clips

        timeline = _shot_timeline(state)
        hook = _shorts_hook(state)
        cover_paths: List[str] = []
        segs: List[Dict[str, Any]] = []

        for clip_idx in range(1, max_shorts + 1):
            if _TRAILER and timeline:
                segs = _select_trailer_segments(timeline, max_shorts, clip_idx)
                # Safety net: a chart/ticker frame must NEVER reach a Short.
                _chart_leak = [s["act"] for s in segs if s["chart"]]
                if _chart_leak:
                    logger.error(
                        f"[MICRO_CONTENT] Chart/ticker leak into Short #{clip_idx} "
                        f"acts {_chart_leak} — excluding from montage.")
                    segs = [s for s in segs if not s["chart"]]
                base = self._build_montage(final_video, segs, pipeline_id, clip_idx)
            else:
                base = None

            if base and os.path.exists(base) and os.path.getsize(base) > 1000:
                out_clip = self._apply_cover(final_video, base, segs[0]["start"],
                                            clip_idx, hook, cover_paths, pipeline_id, state)
            else:
                # Fallback: a single contiguous, speech-aligned crop from the
                # strongest non-chart beat (never a chart) so a publish never dies.
                fb_start = timeline[0]["start"] if timeline else 0.0
                logger.warning(f"[MICRO_CONTENT] Trailer montage for clip #{clip_idx} "
                               f"failed; falling back to single contiguous crop.")
                out_clip = self._render_single_clip(final_video, fb_start, 30.0,
                                                   clip_idx, hook, cover_paths,
                                                   pipeline_id, state)
            if out_clip:
                generated_clips.append(out_clip)

        # Record the 9:16 cover PNGs (parallel to `shorts`) so the publisher can
        # upload them as Shorts thumbnails via youtube.thumbnails.set.
        try:
            if hasattr(state.asset_paths, "shorts_covers"):
                state.asset_paths.shorts_covers = list(cover_paths)
        except Exception:
            pass

        return generated_clips

    def _trim_segment(self, final_video: str, start_s: float, dur: float,
                      pipeline_id: str, clip_idx: int, seg_idx: int) -> Optional[str]:
        """Crop one non-chart beat from the master to a 9:16, 25fps, speech-aligned
        clip with soft in/out fades. Returns the path or None."""
        out = os.path.join(self.output_dir, f"short_{pipeline_id}_m{clip_idx}_s{seg_idx}.mp4")
        # End on the last spoken word (never mid-syllable) when detectable.
        speech_end = _probe_speech_end(final_video, start_s, dur)
        dur = speech_end if (speech_end and 2.0 <= speech_end <= dur) else dur
        dur = max(dur, 3.0)
        fade_in_d = 0.3
        fade_out_d = min(0.5, dur * 0.4)
        fade_out_st = max(fade_in_d, dur - fade_out_d)
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start_s:.2f}", "-i", final_video,
            "-t", f"{dur:.2f}",
            "-vf", f"crop=ih*9/16:ih,scale=1080:1920:flags=lanczos,"
                   f"fade=t=in:st=0:d={fade_in_d:.2f},fade=t=out:st={fade_out_st:.2f}:d={fade_out_d:.2f}",
            "-af", f"afade=t=in:st=0:d={fade_in_d:.2f},afade=t=out:st={fade_out_st:.2f}:d={fade_out_d:.2f}",
            "-r", "25", "-pix_fmt", "yuv420p", "-ar", "44100",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=True, timeout=120)
            if os.path.exists(out) and os.path.getsize(out) > 1000:
                return out
        except Exception as e:
            logger.warning(f"[MICRO_CONTENT] Segment trim failed ({e}); skipping beat.")
        return None

    def _build_montage(self, final_video: str, segs: List[Dict[str, Any]],
                       pipeline_id: str, clip_idx: int) -> Optional[str]:
        """Assemble a tension-gripping trailer from the chosen beats: trim each
        beat, then xfade them into one 9:16 Short. Returns the path or None."""
        if not segs:
            return None
        seg_paths: List[str] = []
        for si, seg in enumerate(segs, 1):
            seg_len = min(seg["dur"], _SEG_LEN)
            p = self._trim_segment(final_video, seg["start"], seg_len, pipeline_id, clip_idx, si)
            if p:
                seg_paths.append(p)
        if not seg_paths:
            return None
        if len(seg_paths) == 1:
            out = os.path.join(self.output_dir, f"short_{pipeline_id}_montage{clip_idx}.mp4")
            try:
                subprocess.run(["ffmpeg", "-y", "-i", seg_paths[0], "-c", "copy", out],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=True, timeout=120)
                if os.path.exists(out) and os.path.getsize(out) > 1000:
                    return out
            except Exception:
                pass
            return seg_paths[0]
        return self._concat_xfade(seg_paths,
                                  os.path.join(self.output_dir, f"short_{pipeline_id}_montage{clip_idx}.mp4"))

    def _concat_xfade(self, clips: List[str], out_path: str, xf: float = 0.35) -> Optional[str]:
        """Concatenate equally-sized 9:16 / 25fps / 44.1k clips with crossfades."""
        try:
            durs = [_probe_duration(c) for c in clips]
            if any(d <= 0 for d in durs):
                raise ValueError("unable to probe one or more clip durations")
            vlabel = "[0:v]"
            alabel = "[0:a]"
            parts: List[str] = []
            for i in range(1, len(clips)):
                cum = sum(durs[:i]) - i * xf
                parts.append(f"{vlabel}[{i}:v]xfade=transition=fade:duration={xf:.2f}:offset={cum:.3f}[v{i}]")
                parts.append(f"{alabel}[{i}:a]acrossfade=d={xf:.2f}[a{i}]")
                vlabel = f"[v{i}]"
                alabel = f"[a{i}]"
            cmd = ["ffmpeg", "-y"]
            for c in clips:
                cmd += ["-i", c]
            cmd += ["-filter_complex", ";".join(parts),
                    "-map", vlabel, "-map", alabel,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=True, timeout=240)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                return out_path
        except Exception as e:
            logger.warning(f"[MICRO_CONTENT] xfade concat failed ({e}); using first beat only.")
        return clips[0] if clips else None

    def _apply_cover(self, final_video: str, base_clip: str, start_s: float,
                     clip_idx: int, hook: str, cover_paths: List[str],
                     pipeline_id: str, state: GlobalState) -> str:
        """Burn the 1s hook cover onto the front of `base_clip` (custom-frame
        rule for Shorts covers). Falls back to the plain clip on any failure."""
        if not (_COVER_ENABLED and base_clip):
            return base_clip
        try:
            cover_png = os.path.join(self.output_dir, f"short_{pipeline_id}_cover{clip_idx}.png")
            cover_mp4 = os.path.join(self.output_dir, f"short_{pipeline_id}_cover{clip_idx}.mp4")
            with_cover = os.path.join(self.output_dir, f"short_{pipeline_id}_clip{clip_idx}_cover.mp4")
            _framed = _extract_short_frame(final_video, start_s, cover_png)
            _used_baked = False
            # Preferred: full thematic native-text cover art (model-rendered text +
            # compliance), prepended as the first frame.
            try:
                from src.engine.nano_banana import generate_baked_shorts_cover
                ref_bytes = None
                if _framed and os.path.getsize(cover_png) > 500:
                    with open(cover_png, "rb") as _rf:
                        ref_bytes = _rf.read()
                baked_out = os.path.join(self.output_dir, f"short_{pipeline_id}_baked{clip_idx}.jpg")
                if generate_baked_shorts_cover(state, output_path=baked_out,
                                               reference_frame=ref_bytes):
                    cover_png = baked_out
                    _framed = True
                    _used_baked = True
            except Exception as e:
                logger.warning(f"[MICRO_CONTENT] nano-banana baked cover skipped ({e}); using frame.")
            # Fallback: nano-banana art (PIL hook burned below) over the frame.
            if not _used_baked:
                try:
                    from src.engine.nano_banana import generate_shorts_cover_art
                    if generate_shorts_cover_art(state, output_path=cover_png):
                        _framed = True
                except Exception as e:
                    logger.warning(f"[MICRO_CONTENT] nano-banana Shorts cover skipped ({e}); using frame.")
            if _framed:
                if not _used_baked:
                    _compose_shorts_cover(cover_png, cover_png, hook)
                cover_paths.append(cover_png)
            if _framed and _cover_to_video(cover_png, cover_mp4) and _prepend_cover(base_clip, cover_mp4, with_cover):
                logger.info(f"[MICRO_CONTENT] Prefixed 1s hook cover -> {with_cover}")
                return with_cover
        except Exception as e:
            logger.warning(f"[MICRO_CONTENT] Shorts cover burn skipped ({e}); using plain clip.")
        return base_clip

    def _render_single_clip(self, final_video: str, start_s: float, clip_dur: float,
                            clip_idx: int, hook: str, cover_paths: List[str],
                            pipeline_id: str, state: GlobalState) -> Optional[str]:
        """Legacy single contiguous crop (used only as a trailer-montage fallback)."""
        out = os.path.join(self.output_dir, f"short_{pipeline_id}_clip{clip_idx}.mp4")
        speech_end = _probe_speech_end(final_video, start_s, clip_dur)
        trim = speech_end if (speech_end and 2.0 <= speech_end <= clip_dur) else clip_dur
        fade_in_d = 0.4
        fade_out_d = min(1.0, trim * 0.3)
        fade_out_st = max(fade_in_d, trim - fade_out_d)
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start_s:.2f}", "-i", final_video,
            "-t", f"{trim:.2f}",
            "-vf", f"crop=ih*9/16:ih,scale=1080:1920:flags=lanczos,"
                   f"fade=t=in:st=0:d={fade_in_d:.2f},fade=t=out:st={fade_out_st:.2f}:d={fade_out_d:.2f}",
            "-af", f"afade=t=in:st=0:d={fade_in_d:.2f},afade=t=out:st={fade_out_st:.2f}:d={fade_out_d:.2f}",
            "-r", "25", "-pix_fmt", "yuv420p", "-ar", "44100",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=True, timeout=120)
            if os.path.exists(out) and os.path.getsize(out) > 1000:
                return self._apply_cover(final_video, out, start_s, clip_idx, hook,
                                         cover_paths, pipeline_id, state)
        except Exception as e:
            logger.warning(f"[MICRO_CONTENT] Single-crop fallback failed for clip #{clip_idx}: {e}")
        return None


micro_content_producer = MicroContentProducer()