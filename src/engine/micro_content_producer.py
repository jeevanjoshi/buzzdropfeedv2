import os
import re
import subprocess
import logging
from typing import List, Optional
from src.schemas.state import GlobalState

logger = logging.getLogger("CSVG_PIPELINE")


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
    """Reuse the SEO thumbnail hook (<=3 words, never the video title) as the
    Shorts cover text."""
    brief = ""
    if state.seo_metadata and state.seo_metadata.thumbnail_brief:
        brief = state.seo_metadata.thumbnail_brief
    elif state.selected_topic and state.selected_topic.headline:
        brief = state.selected_topic.headline
    try:
        from mcp_servers.media_cloud.server import _shorten_thumbnail_text
        return _shorten_thumbnail_text(brief or "WATCH THIS")
    except Exception:
        words = (brief or "WATCH THIS").split()[:3]
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
        Extracts up to `max_shorts` vertical 9:16 video clips from the master video,
        each prefixed by a 1-second burned-in hook cover frame when the master exists.
        """
        generated_clips = []
        pipeline_id = state.pipeline_id or "run"
        final_video = state.asset_paths.final_video if state.asset_paths else None

        if not final_video or not os.path.exists(final_video):
            logger.info("[MICRO_CONTENT] Master video not present on disk; generating mock short clip metadata.")
            for i in range(1, max_shorts + 1):
                mock_path = os.path.join(self.output_dir, f"short_{pipeline_id}_clip{i}.mp4")
                generated_clips.append(mock_path)
            return generated_clips

        # Identify candidate start times from shots (e.g. Act 1 hook & Act 3 revelation)
        shots = state.script_data.shots if state.script_data else []
        durs = state.asset_paths.measured_durations if state.asset_paths else []

        start_offsets = []
        running_time = 0.0
        for idx, shot in enumerate(shots):
            dur = durs[idx] if idx < len(durs) else shot.duration_estimate
            # Pick start of Act 1 (shot 0) and Act 3 (shot index where act_index == 3)
            if shot.act_index in (1, 3) and len(start_offsets) < max_shorts:
                start_offsets.append((idx, running_time, min(dur, 45.0)))
            running_time += dur

        if not start_offsets:
            start_offsets = [(0, 0.0, 30.0)]

        hook = _shorts_hook(state)
        cover_paths = []
        for clip_idx, (shot_idx, start_s, clip_dur) in enumerate(start_offsets, 1):
            out_clip = os.path.join(self.output_dir, f"short_{pipeline_id}_clip{clip_idx}.mp4")
            # Speech-align the end so the Short never cuts mid-word: end at the
            # last spoken sample inside the window (falling back to the fixed shot
            # length if silencedetect is unavailable).
            speech_end = _probe_speech_end(final_video, start_s, clip_dur)
            trim_dur = speech_end if (speech_end and 2.0 <= speech_end <= clip_dur) else clip_dur
            # FFmpeg command to crop 16:9 to 9:16 center crop and trim the clip.
            # A short fade-IN (video + audio) smooths the hard cut from the 1s hook
            # cover, and a ~1.0s fade-OUT ends the Short on a natural breath instead
            # of cutting abruptly mid-word.
            fade_in_d = 0.4
            fade_out_d = min(1.0, trim_dur * 0.3)
            fade_out_st = max(fade_in_d, trim_dur - fade_out_d)
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_s:.2f}",
                "-i", final_video,
                "-t", f"{trim_dur:.2f}",
                "-vf", f"crop=ih*9/16:ih,scale=1080:1920,fade=t=in:st=0:d={fade_in_d:.2f},fade=t=out:st={fade_out_st:.2f}:d={fade_out_d:.2f}",
                "-af", f"afade=t=in:st=0:d={fade_in_d:.2f},afade=t=out:st={fade_out_st:.2f}:d={fade_out_d:.2f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                out_clip
            ]
            ok_trim = False
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=120)
                ok_trim = os.path.exists(out_clip) and os.path.getsize(out_clip) > 1000
                # Guard: if the master was truncated/short at this start time the
                # extracted clip will be far shorter than requested — never ship it.
                if ok_trim:
                    _got = _probe_duration(out_clip)
                    if _got < trim_dur * 0.9:
                        logger.warning(
                            f"[MICRO_CONTENT] Short clip #{clip_idx} truncated "
                            f"(got {_got:.1f}s of {trim_dur:.1f}s) — master likely "
                            f"missing data at start {start_s:.1f}s; skipping."
                        )
                        ok_trim = False
                    else:
                        logger.info(f"[MICRO_CONTENT] Rendered 9:16 Short clip #{clip_idx}: {out_clip}")
                else:
                    logger.warning(f"[MICRO_CONTENT] Trimmed clip #{clip_idx} empty: {out_clip}")
            except Exception as e:
                logger.warning(f"[MICRO_CONTENT] FFmpeg crop failed for clip #{clip_idx}: {e}")

            base_clip = out_clip if ok_trim else None

            # Burn in the 1s hook cover frame at the start (custom-frame rule).
            if _COVER_ENABLED and base_clip:
                cover_png = os.path.join(self.output_dir, f"short_{pipeline_id}_cover{clip_idx}.png")
                try:
                    cover_mp4 = os.path.join(self.output_dir, f"short_{pipeline_id}_cover{clip_idx}.mp4")
                    with_cover = os.path.join(self.output_dir, f"short_{pipeline_id}_clip{clip_idx}_cover.mp4")
                    _framed = _extract_short_frame(final_video, start_s, cover_png)
                    _used_baked = False
                    # Option B (preferred): full thematic native-text cover —
                    # thematic hook + design-rules art + model-rendered text +
                    # compliance, then prepended as the first frame.
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
                        out_clip = with_cover
                        logger.info(f"[MICRO_CONTENT] Prefixed 1s hook cover -> {with_cover}")
                except Exception as e:
                    logger.warning(f"[MICRO_CONTENT] Shorts cover burn skipped ({e}); using plain clip.")
            generated_clips.append(out_clip)

        # Record the 9:16 cover PNGs (parallel to `shorts`) so the publisher can
        # upload them as Shorts thumbnails via youtube.thumbnails.set.
        try:
            if hasattr(state.asset_paths, "shorts_covers"):
                state.asset_paths.shorts_covers = list(cover_paths)
        except Exception:
            pass

        return generated_clips


micro_content_producer = MicroContentProducer()