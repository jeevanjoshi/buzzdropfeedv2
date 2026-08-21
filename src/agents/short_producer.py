"""ShortProducerAgent — dedicated professional YouTube Shorts producer.

This is the SEPARATE short-video producer the pipeline previously lacked. Shorts are
no longer center-crops of the 16:9 master (see src/engine/micro_content_producer.py);
instead they are FIRST-CLASS 9:16 vertical deliverables with their own short-native
script (hook -> payoff arc), vertical-composed visuals, punchy TTS, mobile captions,
and a short-specific music bed.

Design + scaffold status: the authoring path (LLM short-native script) and the
vertical media renderer (TTS + 9:16 Ken-Burns + mobile captions + hook cover) are
implemented and reused from the existing media toolchain. Portrait visuals are now
real: each beat requests a 9:16 FLUX still (budget-gated, see media_budget) with a
subject-safe `reframe_16_9_to_9_16` crop fallback, eliminating the old center-crop
from the 16:9 master. The short-native music bed (`_short_music_bed`) is mood-aware
(`_classify_short_mood`) and downloads/selects an optimal track per Short, falling
back to the master bgm.mp3. Short-specific captions (`_style_mobile_ass`) restyle the
merged track into a 9:16 mobile-safe style: large, bold, high-contrast, bottom-center.
Remaining "pro-level" polish — true ML subject-tracking — is marked TODO and lands
incrementally (see docs/SHORTS_PRODUCER_DESIGN.md).

Enabled by default (CSVG_SHORTS_PRODUCER=0 disables it, falling back to the
legacy micro_content_producer crop path).
"""

import os
import subprocess
import shutil
import json

from typing import Dict, Any, Optional, List

from src.schemas.state import (
    GlobalState, ShortScript, ShortBeat, AssetPaths,
)
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent
from mcp_servers.audio_edge.server import (
    synthesize_tts, align_subtitles_whisper, sanitize_tts_text,
    TTSRequest, WhisperRequest,
)
from mcp_servers.media_cloud.server import (
    assemble_ffmpeg_timeline, TimelineAssemblyRequest, ImageGenRequest,
    generate_flux_image,
)
from src.agents.media_producer import merge_ass_subtitle_files
from src.engine.llm_client import LLMClient
from src.engine.media_budget import media_budget


# ── Vertical format constants ──────────────────────────────────────────────────
SHORT_RESOLUTION = (1080, 1920)          # strict 9:16
SHORT_FPS = 25
SHORT_CROSSFADE = 0.25                    # tighter than the master's 0.5
SHORT_PAD_AFTER = 0.4                     # short breathing hold after each beat
SHORT_HOOK_COVER_SECONDS = 1.0            # hook cover prepended as first frame
SHORT_MAX_BEATS = 8                       # cap beats per Short (~40-60s)
SHORT_TTS_SPEED = 1.06                    # punchier cadence than the long-form
SHORT_ASPECT = "9:16"                      # portrait aspect requested from FLUX
# Short-specific caption style: large, bold, high-contrast, bottom-center for 9:16.
SHORT_CAPTION_STYLE = (
    "Style: Default,Montserrat,20,&H00FFFFFF,&H000000FF,&H00000000,"
    "&H64000000,-1,0,0,0,100,100,0,0,1,3,2,2,80,80,220,1"
)

# A/B "test & compare" variant set (mirrors CSVG_THUMBNAIL_VARIANTS design language):
# a primary Short plus 3 experiment variants emphasising different retention levers.
SHORT_VARIANT_MODES = ["hook_led", "hero_object", "text_led"]


def short_variants_enabled() -> bool:
    """Opt-in A/B variant set. Off by default (one primary Short per run)."""
    return os.getenv("CSVG_SHORTS_VARIANTS", "0").strip().lower() in ("1", "true", "yes")


def short_variant_specs(pipeline_id: str) -> List[Dict[str, Any]]:
    """Deterministic primary + 3 A/B variant specs derived from pipeline_id.

    Each variant stresses a different retention lever while reusing the same
    short-native script (no extra LLM authoring calls):
      - hook_led:   longer punchy hook cover (more seconds on the open).
      - hero_object: leads with the hero beat's subject line on the cover.
      - text_led:   caption-forward (tighter bottom margin, larger on-screen text area).
    """
    specs: List[Dict[str, Any]] = [{
        "mode": "primary", "role": "primary", "label": "Primary",
        "cover_seconds": None, "lead_beat": 0, "caption_margin_v": 220,
    }]
    for mode in SHORT_VARIANT_MODES:
        if mode == "hook_led":
            cover_seconds, lead_beat, margin_v = 1.8, 0, 220
        elif mode == "hero_object":
            cover_seconds, lead_beat, margin_v = 1.0, 1, 220
        else:  # text_led
            cover_seconds, lead_beat, margin_v = 1.2, 0, 170
        specs.append({
            "mode": mode, "role": "variant", "label": mode.replace("_", " ").title(),
            "cover_seconds": cover_seconds, "lead_beat": lead_beat,
            "caption_margin_v": margin_v,
        })
    return specs


def _write_short_variants_manifest(work: str, pipeline_id: str,
                                   specs: List[Dict[str, Any]],
                                   paths: List[str]) -> str:
    """Writes logs/media/<pid>/shorts/shorts_variants.json so the operator can finish
    the experiments as YouTube 'test & compare' Shorts in Studio (like thumbnails)."""
    out = os.path.join(work, "shorts_variants.json")
    os.makedirs(work, exist_ok=True)
    data = {
        "pipeline_id": pipeline_id,
        "primary": paths[0] if paths else None,
        "variants": [
            {"mode": s["mode"], "role": s["role"], "label": s["label"], "path": p}
            for s, p in zip(specs, paths)
        ],
    }
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass
    return out


# ── Pure helpers (unit-tested; see tests/test_hermetic_e2e.py) ─────────────────
def build_short_script_prompt(state: GlobalState) -> str:
    """Builds the LLM prompt that authors a short-native script.

    Grounded in the same verified facts / topic the long-form used, but reframed
    for a mobile, self-contained, retention-optimised vertical video."""
    topic = state.selected_topic
    headline = getattr(topic, "headline", "") if topic else ""
    niche = getattr(topic, "niche_category", "") if topic else ""
    facts = state.verified_facts or []
    fact_lines = "\n".join(f"- {f.headline}: {f.summary}" for f in facts[:6])
    master_title = ""
    if state.script_data:
        master_title = state.script_data.title
    seo_hook = ""
    if state.seo_metadata and state.seo_metadata.thumbnail_brief:
        seo_hook = state.seo_metadata.thumbnail_brief

    return (
        "You are a YouTube Shorts retention specialist. Write ONE vertical (9:16) "
        "Short that stands alone — a viewer who never sees the long video must "
        "understand and be hooked.\n\n"
        f"Topic: {headline}\n"
        f"Niche: {niche}\n"
        f"Long-form title (context only): {master_title}\n"
        f"Hook keyword (optional): {seo_hook}\n\n"
        f"Verified facts:\n{fact_lines}\n\n"
        "Return JSON: {\n"
        '  "hook_line": str (1-3s pattern-interrupt curiosity gap, <=12 words),\n'
        '  "title": str (<=60 chars, CTR-optimised),\n'
        '  "beats": [ { "is_hook": bool, "narration_text": str (<=18 words, punchy), '
        '"visual_prompt": str (9:16 vertical cinematic brief, no text), '
        '"duration_estimate": float } ]\n'
        "}\n"
        "Rules: 4-8 beats, opening beat is_hook=true, NO spoken citations, NO "
        "fabricated numbers, every claim must come from the verified facts. Self-contained."
    )


def parse_short_script(raw: Dict[str, Any], state: GlobalState) -> ShortScript:
    """Parses an LLM short-script payload into a validated ShortScript.

    Pure / deterministic so it is fully unit-testable without an LLM."""
    raw = raw or {}
    beats_raw = raw.get("beats") or []
    beats: List[ShortBeat] = []
    for i, b in enumerate(beats_raw[:SHORT_MAX_BEATS], start=1):
        if not isinstance(b, dict):
            continue
        narr = sanitize_tts_text(str(b.get("narration_text", ""))).strip()
        if not narr:
            continue
        beats.append(ShortBeat(
            beat_id=i,
            is_hook=bool(b.get("is_hook", i == 1)),
            narration_text=narr,
            visual_prompt=str(b.get("visual_prompt", "cinematic vertical scene"))[:400],
            duration_estimate=float(b.get("duration_estimate", 5.0)) or 5.0,
        ))
    if not beats:
        # Defensive: never emit an empty Short.
        beats.append(ShortBeat(
            beat_id=1, is_hook=True,
            narration_text=(raw.get("hook_line") or "This changes everything.")[:200],
            visual_prompt="cinematic vertical reveal", duration_estimate=5.0,
        ))
    total = sum(b.duration_estimate for b in beats)
    return ShortScript(
        title=str(raw.get("title", state.script_data.title if state.script_data else "Short"))[:90],
        hook_line=str(raw.get("hook_line", beats[0].narration_text))[:200],
        beats=beats,
        estimated_runtime_seconds=total,
    )


def generate_synthetic_vertical_png(output_path: str, title: str, subtitle: str = "") -> str:
    """Generates a broadcast-grade 9:16 synthetic PNG (free fallback visual).

    The FLUX portrait path (CSVG portrait aspect) is a TODO; until then every
    beat uses this deterministic placeholder so the vertical renderer never blocks
    on a missing paid image."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        from PIL import Image, ImageDraw
        W, H = SHORT_RESOLUTION
        img = Image.new("RGB", (W, H), color=(12, 16, 28))
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, W - 40, H - 40], outline=(0, 255, 204), width=6)
        draw.line([(40, H // 2), (W - 40, H // 2)], fill=(0, 150, 200), width=3)
        draw.text((80, H // 2 - 40), (title or "9:16 SHORT")[:40].upper(), fill=(255, 255, 255))
        if subtitle:
            draw.text((80, H // 2 + 10), subtitle[:60], fill=(200, 230, 255))
        img.save(output_path, "PNG")
    except Exception as e:
        print(f"[ShortProducer] PIL vertical image gen error: {e}")
    return output_path


def _beat_motion(beat_id: int, is_opener: bool = False) -> Dict[str, Any]:
    """Deterministic per-beat Ken-Burns motion so the Short doesn't pan identically
    on every beat. The opener beat gets a 'punch' (aggressive zoom) as a
    pattern-interrupt that jolts attention right after the hook cover."""
    pan_choices = ["center", "up", "down", "left", "right"]
    pan = pan_choices[beat_id % len(pan_choices)]
    zoom_dir = "out" if (beat_id % 2 == 0) else "in"
    motion: Dict[str, Any] = {"pan": pan, "zoom_dir": zoom_dir}
    if is_opener:
        motion["punch"] = True
    return motion


def _build_ken_burns_vf(motion: Optional[Dict[str, Any]], nb: int) -> str:
    """Vertical-aware Ken-Burns zoompan filter. `motion` drives zoom direction,
    pan drift, and an optional aggressive 'punch' zoom for pattern-interrupts."""
    motion = motion or {}
    z_in = motion.get("zoom_dir", "in") != "out"
    punch = bool(motion.get("punch"))
    z_target = 1.6 if punch else 1.35
    z_min, z_max = (1.0, z_target) if z_in else (z_target, 1.0)
    z_step = (z_max - z_min) / max(nb, 1)
    z_expr = (f"min({z_min:.4f}+{z_step:.6f}*on,{z_max:.4f})" if z_in
              else f"max({z_min:.4f}+{z_step:.6f}*on,{z_max:.4f})")

    pan = motion.get("pan", "center")
    pan_map = {"center": 0.0, "up": -0.35, "down": 0.35, "left": -0.35, "right": 0.35}
    p = pan_map.get(pan, 0.0)
    if pan in ("up", "down"):
        cy = f"ih/2-(ih/zoom/2)+({p})*(ih-ih/zoom/2)"
        cx = "iw/2-(iw/zoom/2)"
    else:
        cx = f"iw/2-(iw/zoom/2)+({p})*(iw-iw/zoom/2)"
        cy = "ih/2-(ih/zoom/2)"
    return (
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
        f"zoompan=z='{z_expr}':x='{cx}':y='{cy}':d=1:s=1080x1920:fps={SHORT_FPS},"
        f"fps={SHORT_FPS},format=yuv420p"
    )


def _render_vertical_clip(image_path: str, audio_path: Optional[str],
                           duration: float, out_path: str,
                           motion: Optional[Dict[str, Any]] = None) -> str:
    """Renders a 9:16 Ken-Burns clip from a still image + narration audio. Per-beat
    `motion` (see `_beat_motion`) varies the pan/zoom so the Short has visual
    diversity instead of a uniform drift.

    NOTE: src/servers/media_cloud apply_ken_burns_motion hardcodes 1920x1080, so we
    use a vertical-aware ffmpeg zoompan here instead of that shared helper."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dur = max(duration, 2.0)
    if not shutil.which("ffmpeg"):
        print("[ShortProducer] WARNING: ffmpeg missing; writing placeholder MP4.")
        with open(out_path, "w") as f:
            f.write(f"DUMMY_SHORT_MP4_{image_path}")
        return out_path
    cmd = ["ffmpeg", "-y", "-nostdin", "-loop", "1", "-i", image_path]
    has_audio = bool(audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 100)
    if has_audio:
        cmd += ["-i", audio_path]
    nb = max(int(dur * SHORT_FPS) - 1, 1)
    vf = _build_ken_burns_vf(motion, nb)
    cmd += ["-vf", vf, "-t", f"{dur:.3f}", "-r", str(SHORT_FPS),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd.append(out_path)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ShortProducer] vertical render failed: {res.stderr[-300:]}")
    return out_path


def _image_to_video(image_path: str, duration: float, out_path: str) -> str:
    """Converts a still cover image into a silent 9:16 video clip (hook frame)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if not shutil.which("ffmpeg"):
        with open(out_path, "w") as f:
            f.write(f"DUMMY_COVER_{image_path}")
        return out_path
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loop", "1", "-i", image_path,
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,"
               "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-t", f"{duration:.3f}", "-r", str(SHORT_FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        out_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ShortProducer] cover->video failed: {res.stderr[-200:]}")
    return out_path


def _make_hook_cover(state: GlobalState, short_script: ShortScript, out_path: str,
                     lead_beat_index: int = 0) -> str:
    """Burns the hook line onto a 9:16 cover frame (YouTube ignores thumbnails.set
    for Shorts, so the hook must be a frame). `lead_beat_index` lets variant covers
    surface a different subject line (e.g. hero-object variant)."""
    hook = short_script.hook_line or (short_script.beats[0].narration_text if short_script.beats else "")
    title = short_script.title
    subject = ""
    beats = short_script.beats or []
    if 0 <= lead_beat_index < len(beats):
        subject = (beats[lead_beat_index].narration_text or "")[:40]
    sub = hook[:55]
    if subject and subject.lower() not in hook.lower():
        sub = f"{sub}  |  {subject}"
    return generate_synthetic_vertical_png(out_path, title[:30], sub[:80])


def _is_portrait(image_path: str) -> Optional[bool]:
    """Returns True if the image is already 9:16 (portrait), False if landscape,
    or None if the dimensions cannot be read (e.g. PIL missing)."""
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            w, h = im.size
        if w <= 0 or h <= 0:
            return None
        # Portrait when height/width is within tolerance of 16/9.
        ratio = h / w
        if abs(ratio - (16.0 / 9.0)) < 0.15:
            return True
        if abs(ratio - (9.0 / 16.0)) < 0.15:
            return False
        # Ambiguous; treat >= square as needing reframe to portrait.
        return False if ratio < 1.0 else True
    except Exception:
        return None


def reframe_16_9_to_9_16(src_path: str, dst_path: str, anchor: str = "center") -> str:
    """Subject-aware reframe of a 16:9 (landscape) still into a 9:16 (portrait)
    crop. `anchor` biases the crop window toward where the subject sits:
    'center' (default), 'left', 'right'. True ML subject-tracking (face/object
    detection) is a future refinement; this guarantees correct vertical framing
    and keeps the subject off the cropped edges.

    Uses ffmpeg crop; falls back to a copy if ffmpeg is unavailable.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    if not shutil.which("ffmpeg"):
        try:
            shutil.copy2(src_path, dst_path)
        except Exception:
            pass
        return dst_path
    anchor_map = {"left": "iw/2*(1-9/16)", "right": "iw-(iw/2*(9/16))", "center": "(iw-ih*9/16)/2"}
    xpos = anchor_map.get(anchor, anchor_map["center"])
    # Crop a 9:16 window (width = height*9/16) from the 16:9 source, centered
    # vertically, x anchored by `anchor`.
    vf = (
        f"crop=ih*9/16:ih:{xpos}:0,"
        f"scale={SHORT_RESOLUTION[0]}:{SHORT_RESOLUTION[1]}:flags=lanczos,format=yuv420p"
    )
    cmd = ["ffmpeg", "-y", "-nostdin", "-i", src_path, "-vf", vf,
           "-frames:v", "1", dst_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not (os.path.exists(dst_path) and os.path.getsize(dst_path) > 0):
        print(f"[ShortProducer] reframe failed ({res.stderr[-200:]}); copying source.")
        try:
            shutil.copy2(src_path, dst_path)
        except Exception:
            pass
    return dst_path


# ── Agent ──────────────────────────────────────────────────────────────────────
class ShortProducerAgent:
    """Stage 4c: dedicated professional Shorts producer.

    Produces a short-native 9:16 video per run: authors a short script via the LLM,
    then renders vertical TTS + Ken-Burns visuals + mobile captions + a hook cover,
    assembled independently of the master timeline. With CSVG_SHORTS_VARIANTS=1 it
    additionally emits a primary + 3 A/B experiment variants (hook-led / hero-object
    / text-led) plus a shorts_variants.json manifest (primary is what gets published;
    the rest are finished as Studio test-and-compare experiments).
    """

    def __init__(self, name: str = "ShortProducer", llm_client: Optional[LLMClient] = None,
                 storage_dir: str = "/tmp/csvg_shorts"):
        self.name = name
        self.llm_client = llm_client or LLMClient()
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    # ── Stage 1: author the short-native script ──────────────────────────────
    def author_short_script(self, state: GlobalState) -> Optional[ShortScript]:
        llm = self.llm_client
        if llm and llm.is_available():
            prompt = build_short_script_prompt(state)
            system = ("You write viral, fact-grounded YouTube Shorts scripts. "
                      "Respond ONLY with the requested JSON.")
            try:
                raw = llm.generate_json(prompt, system_prompt=system, route="generate")
                if raw:
                    return parse_short_script(raw, state)
            except Exception as e:
                print(f"[ShortProducer] LLM short-script authoring failed ({e}); using fallback.")
        return self._fallback_short_script(state)

    def _fallback_short_script(self, state: GlobalState) -> ShortScript:
        """Deterministic short script from the master topic when no LLM is available."""
        topic = state.selected_topic
        headline = getattr(topic, "headline", "This changes everything") if topic else "This changes everything"
        facts = state.verified_facts or []
        beats: List[ShortBeat] = []
        hook = headline.split(".")[0][:80] or "You won't believe what just happened."
        beats.append(ShortBeat(beat_id=1, is_hook=True, narration_text=hook[:120],
                               visual_prompt="cinematic vertical reveal of the story subject",
                               duration_estimate=4.0))
        for i, f in enumerate(facts[:5], start=2):
            narr = sanitize_tts_text(f.headline)[:160] or f.summary[:160]
            beats.append(ShortBeat(
                beat_id=i, is_hook=False, narration_text=narr,
                visual_prompt=f"vertical data-driven scene illustrating {narr[:40]}",
                duration_estimate=5.0,
            ))
        if len(beats) < 2:
            beats.append(ShortBeat(beat_id=len(beats) + 1, is_hook=False,
                                   narration_text="The details change everything.",
                                   visual_prompt="cinematic vertical closing shot",
                                   duration_estimate=5.0))
        total = sum(b.duration_estimate for b in beats)
        return ShortScript(
            title=(state.script_data.title if state.script_data else headline)[:90],
            hook_line=hook[:200], beats=beats, estimated_runtime_seconds=total,
        )

    # ── Stage 2: render the short-native vertical media ──────────────────────
    async def produce_short_media(self, state: GlobalState, short_script: ShortScript,
                                 spec: Optional[Dict[str, Any]] = None,
                                 suffix: str = "") -> AssetPaths:
        assets = AssetPaths()
        if not short_script or not short_script.beats:
            return assets

        spec = spec or {}
        cover_seconds = float(spec.get("cover_seconds") or SHORT_HOOK_COVER_SECONDS)
        lead_beat = int(spec.get("lead_beat", 0) or 0)
        caption_margin_v = int(spec.get("caption_margin_v", 220) or 220)

        exec_id = (getattr(state, "pipeline_id", "") or "csgv").strip() or "csgv"
        work = os.path.join(_repo_root(), "logs", "media", exec_id, "shorts")
        os.makedirs(work, exist_ok=True)
        assets.storage_dir = work
        audio_dir = os.path.join(work, "audio")
        sub_dir = os.path.join(work, "subtitles")
        vis_dir = os.path.join(work, "visuals")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(sub_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        concat_lines: List[str] = []
        ass_paths: List[str] = []
        durations: List[float] = []

        for beat in short_script.beats:
            beat_key = f"beat_{beat.beat_id}"
            wav = os.path.join(audio_dir, f"{beat_key}.wav")
            ass = os.path.join(sub_dir, f"{beat_key}.ass")
            img = os.path.join(vis_dir, f"{beat_key}.png")
            mp4 = os.path.join(vis_dir, f"{beat_key}.mp4")
            try:
                # Pro-level vertical visual: request a real 9:16 FLUX still (budget
                # gated, like the master's hero shots) so the subject is composed for
                # portrait instead of center-cropped from a 16:9 frame. Fall back to
                # the free synthetic vertical still when FLUX is unavailable / capped.
                portrait_ok = False
                if media_budget.charge_paid_image():
                    try:
                        await generate_flux_image(ImageGenRequest(
                            prompt=beat.visual_prompt, output_image_path=img,
                            aspect=SHORT_ASPECT))
                        portrait_ok = os.path.exists(img) and os.path.getsize(img) > 1000
                    except Exception as fe:
                        print(f"[ShortProducer] FLUX portrait failed for {beat_key} ({fe}); using fallback.")
                if not portrait_ok:
                    generate_synthetic_vertical_png(img, beat.visual_prompt[:30])
                # Defensive reframe: if the generated still is landscape (e.g. a FLUX
                # param hiccup), crop it to 9:16 around the subject-safe center.
                if _is_portrait(img) is False:
                    reframe_16_9_to_9_16(img, img)

                tts_res = await synthesize_tts(TTSRequest(
                    text=beat.narration_text, output_path=wav, speed=SHORT_TTS_SPEED))
                if isinstance(tts_res, dict) and tts_res.get("engine") == "synthetic_wav_fallback":
                    raise RuntimeError(
                        f"Short TTS degraded to synthetic_wav_fallback for {beat_key}; "
                        f"aborting rather than shipping fake audio.")
                await align_subtitles_whisper(WhisperRequest(
                    audio_path=wav, output_ass_path=ass, original_text=beat.narration_text))

                audio_dur = _probe_wav_duration(wav) or beat.duration_estimate
                clip_dur = max(audio_dur + SHORT_PAD_AFTER, 2.0)
                is_opener = (beat is short_script.beats[0])
                _render_vertical_clip(img, wav, clip_dur, mp4,
                                      motion=_beat_motion(beat.beat_id, is_opener=is_opener))

                concat_lines.append(f"file '{mp4}'")
                if os.path.exists(ass):
                    ass_paths.append(ass)
                durations.append(clip_dur)
            except Exception as e:
                print(f"[ShortProducer] beat {beat_key} failed ({e}); skipping beat.")
                continue

        if not concat_lines:
            print("[ShortProducer] No beats rendered; short production aborted.")
            return assets

        # Hook cover as the opening frame (YouTube Shorts need an in-video hook).
        cover_png = os.path.join(work, f"hook_cover{suffix}.png")
        cover_out = cover_png
        try:
            _make_hook_cover(state, short_script, cover_png, lead_beat_index=lead_beat)
            cover_mp4 = os.path.join(work, f"hook_cover{suffix}.mp4")
            _image_to_video(cover_png, cover_seconds, cover_mp4)
            if os.path.exists(cover_mp4) and os.path.getsize(cover_mp4) > 0:
                concat_lines.insert(0, f"file '{cover_mp4}'")
                durations.insert(0, cover_seconds)
                ass_paths.insert(0, "")  # silent cover, no subtitle track
        except Exception as e:
            print(f"[ShortProducer] hook cover failed ({e}); continuing without cover.")

        # Merge beat subtitles into a single master .ass (skip empty cover entry).
        rendered_ass = [a for a in ass_paths if a]
        rendered_dur = [d for a, d in zip(durations, ass_paths) if a]
        master_ass = os.path.join(work, f"short_subtitles{suffix}.ass")
        try:
            merge_ass_subtitle_files(rendered_ass, rendered_dur, master_ass,
                                     crossfade=SHORT_CROSSFADE)
        except Exception as e:
            print(f"[ShortProducer] subtitle merge failed ({e}); continuing.")
            master_ass = ""

        # Short-specific caption styling: restyle the merged 16:9 master .ass into
        # a 9:16 mobile-safe track (large, bold, high-contrast, bottom-center).
        if master_ass and os.path.exists(master_ass) and any(
                b.caption_style == "mobile" for b in short_script.beats):
            styled_ass = os.path.join(work, f"short_subtitles_mobile{suffix}.ass")
            master_ass = _style_mobile_ass(master_ass, styled_ass, margin_v=caption_margin_v)

        # Assemble the 9:16 Short.
        final_path = os.path.join(work, f"short_{exec_id}{suffix}.mp4")
        mood = _classify_short_mood(short_script, state)
        bgm = _short_music_bed(work, mood)
        try:
            await assemble_ffmpeg_timeline(TimelineAssemblyRequest(
                concat_list_path=_write_concat(work, concat_lines, suffix=suffix),
                subtitle_path=master_ass,
                bgm_path=bgm,
                output_video_path=final_path,
                crossfade=SHORT_CROSSFADE,
                transition="fade",
            ))
        except Exception as e:
            print(f"[ShortProducer] assembly failed ({e}); short not produced.")
            return assets

        if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
            assets.final_video = final_path
            assets.shorts = [final_path]
            if cover_out and os.path.exists(cover_out):
                assets.shorts_covers = [cover_out]
        return assets

    # ── Orchestration entrypoint ─────────────────────────────────────────────
    async def process(self, state: GlobalState, dummy_frames: bool = False) -> A2AMessage:
        short_script = self.author_short_script(state)
        state.short_script = short_script
        assets = AssetPaths()

        exec_id = (getattr(state, "pipeline_id", "") or "csgv").strip() or "csgv"
        work = os.path.join(_repo_root(), "logs", "media", exec_id, "shorts")

        # A/B "test & compare" variant set: produce a primary Short plus 3 experiment
        # variants (hook-led / hero-object / text-led) reusing the same script, and
        # write a manifest so the operator can finish them as Studio experiments.
        if short_variants_enabled() and short_script and short_script.beats:
            specs = short_variant_specs(exec_id)
            paths: List[str] = []
            covers: List[str] = []
            for i, spec in enumerate(specs):
                sfx = "" if i == 0 else f"_{spec['mode']}"
                a = await self.produce_short_media(state, short_script, spec=spec, suffix=sfx)
                if a.shorts:
                    paths.extend(a.shorts)
                    assets.storage_dir = a.storage_dir or work
                    if a.shorts_covers:
                        covers.extend(a.shorts_covers)
            if paths:
                _write_short_variants_manifest(work, exec_id, specs, paths)
                assets.shorts = paths
                assets.shorts_covers = covers[:len(paths)]
                assets.final_video = paths[0]
            else:
                # All variants failed; fall back to a single primary render.
                assets = await self.produce_short_media(state, short_script)
        else:
            assets = await self.produce_short_media(state, short_script)

        # Write the produced clips back onto the shared state so the publisher
        # (and the orchestrator's SHORTS_PRODUCED log check) sees them. Without this,
        # the legacy crop path would silently win even after a successful render.
        if assets.shorts:
            state.asset_paths.shorts = assets.shorts
            state.asset_paths.shorts_covers = assets.shorts_covers
            state.asset_paths.final_video = assets.final_video
            state.asset_paths.storage_dir = assets.storage_dir

        return A2AMessage(
            message_id="m4b",
            sender=AgentRole.MEDIA_PRODUCER,
            target=AgentRole.ORCHESTRATOR,
            intent=AgentIntent.MEDIA_READY,
            payload={"short_video": (assets.shorts or [None])[0]},
            timestamp="0",
        )


# ── Internal utilities ─────────────────────────────────────────────────────────
def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _probe_wav_duration(wav_path: str) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
            capture_output=True, text=True)
        return float(out.stdout.strip())
    except Exception:
        return None


def _write_concat(work: str, lines: List[str], suffix: str = "") -> str:
    p = os.path.join(work, f"short_concat_list{suffix}.txt")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    return p


# Mood -> short-native music track. "Optimal accordingly": the bed is chosen to
# match the Short's emotional arc, not reused verbatim from the long-form master.
# Operators drop royalty-free tracks at resources/shorts_music/<mood>.mp3 (or set
# CSVG_SHORTS_MUSIC_BASE_URL to a hosted library; the file is downloaded + cached).
SHORT_MOOD_TRACKS = {
    "upbeat_discovery": "upbeat_discovery",
    "inspiring": "inspiring",
    "tense": "tense",
    "curiosity": "curiosity",
    "calm": "calm",
    "epic": "epic",
}


def _classify_short_mood(short_script: ShortScript, state: GlobalState) -> str:
    """Deterministically maps the Short's content to a music mood (no LLM, hermetic).

    Prefers the short's own hook/title keywords, then the master topic's
    audience/niche, falling back to an energetic 'upbeat_discovery' default that
    suits Shorts discovery."""
    text = " ".join([
        (short_script.hook_line or ""),
        (short_script.title or ""),
    ] + [b.narration_text for b in (short_script.beats or [])]).lower()

    # Tense / conflict signals -> tense bed.
    if any(k in text for k in ("war", "crisis", "collapse", "crash", "threat", "scam", "ban", "lawsuit", "risk")):
        return "tense"
    # Inspiring / hopeful signals -> inspiring bed.
    if any(k in text for k in ("breakthrough", "wins", "hope", "future", "saves", "success", "record", "first")):
        return "inspiring"
    # Calm / explanatory signals -> calm bed.
    if any(k in text for k in ("how to", "explained", "guide", "tutorial", "why", "what is")):
        return "calm"
    # Curiosity / mystery signals -> curiosity bed.
    if any(k in text for k in ("secret", "shock", "you won't believe", "mystery", "revealed", "truth")):
        return "curiosity"

    # Fall back to the master topic's audience taxonomy when available.
    topic = state.selected_topic
    audience = getattr(topic, "audience_type", "") if topic else ""
    if audience in ("tech", "science", "business"):
        return "upbeat_discovery"
    if audience in ("finance_edu", "investor", "real_estate"):
        return "epic"
    return "upbeat_discovery"


def _download_short_music(url: str, dest: str) -> bool:
    """Downloads a royalty-free music file from an operator-configured URL.
    No URLs are hard-coded; the caller supplies `url` (e.g. from
    CSVG_SHORTS_MUSIC_BASE_URL). Returns True on success."""
    try:
        import requests
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        with requests.get(url, timeout=30, stream=True) as r:
            if r.status_code != 200:
                print(f"[ShortProducer] music download HTTP {r.status_code} for {url}")
                return False
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        return os.path.exists(dest) and os.path.getsize(dest) > 1000
    except Exception as e:
        print(f"[ShortProducer] music download failed ({e}); using fallback.")
        return False


def _short_music_bed(work: str, mood: str = "upbeat_discovery") -> str:
    """Resolves the Short's short-native BGM bed, chosen by mood.

    Resolution order:
      1. Curated local library resources/shorts_music/<mood>.mp3
      2. Download from CSVG_SHORTS_MUSIC_BASE_URL/<mood>.mp3 (cached locally)
      3. Fallback to the master resources/bgm.mp3 (never ship silence)
    """
    out = os.path.join(work, "short_bgm.mp3")
    repo_music_dir = os.path.join(_repo_root(), "resources", "shorts_music")
    track_name = SHORT_MOOD_TRACKS.get(mood, "upbeat_discovery")
    local_track = os.path.join(repo_music_dir, f"{track_name}.mp3")

    # 1. Curated local library.
    if os.path.exists(local_track) and os.path.getsize(local_track) > 1000:
        try:
            shutil.copy2(local_track, out)
            return out
        except Exception:
            pass

    # 2. Operator-hosted library (download + cache for next run).
    base = os.getenv("CSVG_SHORTS_MUSIC_BASE_URL", "").strip().rstrip("/")
    if base:
        cache = os.path.join(repo_music_dir, f"{track_name}.mp3")
        os.makedirs(repo_music_dir, exist_ok=True)
        if _download_short_music(f"{base}/{track_name}.mp3", cache) and os.path.getsize(cache) > 1000:
            try:
                shutil.copy2(cache, out)
                return out
            except Exception:
                pass

    # 3. Fallback to the long-form master BGM.
    src = os.path.join(_repo_root(), "resources", "bgm.mp3")
    if os.path.exists(src) and os.path.getsize(src) > 1000:
        try:
            shutil.copy2(src, out)
            return out
        except Exception:
            pass
    with open(out, "w") as f:
        f.write("DUMMY_BGM")
    return out


def _style_mobile_ass(src_ass: str, dst_ass: str, margin_v: int = 220) -> str:
    """Restyles a merged 16:9 master .ass into a 9:16 short-native caption track:
    large bold high-contrast centered subtitles safe for mobile portrait viewing.
    Rewrites PlayRes to 1080x1920, swaps the Default style, and normalizes each
    Dialogue line's margins. `margin_v` controls the bottom margin (tighter = more
    on-screen text area for text-led variants). Returns the styled path (or src)."""
    try:
        with open(src_ass, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return src_ass
    out: List[str] = []
    for line in lines:
        if line.startswith("PlayResX:"):
            out.append("PlayResX: 1080\n")
            continue
        if line.startswith("PlayResY:"):
            out.append("PlayResY: 1920\n")
            continue
        if line.startswith("Title:"):
            out.append("Title: 9:16 CSVG Short Subtitles\n")
            continue
        if line.startswith("Style: Default"):
            out.append(SHORT_CAPTION_STYLE + "\n")
            continue
        if line.startswith("Dialogue:"):
            parts = line.rstrip("\n").split(",", 9)
            if len(parts) == 10:
                parts[3] = "Default"
                parts[5] = "80"
                parts[6] = "80"
                parts[7] = str(int(margin_v))
                line = ",".join(parts) + "\n"
        out.append(line)
    try:
        with open(dst_ass, "w", encoding="utf-8") as f:
            f.write("".join(out))
        return dst_ass
    except Exception:
        return src_ass
