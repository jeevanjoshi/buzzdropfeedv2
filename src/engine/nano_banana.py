"""Google nano-banana (gemini-2.5-flash-image) AI thumbnail generation.

Two entry points:

1. Pipeline mode — analyses the pipeline's own video content (title, hook
   brief, hero shot scene, act titles, key facts) with the LLM to craft a
   high-converting, click-worthy image prompt, then renders 16:9 long-form
   thumbnail art and 9:16 Shorts cover art via Google's nano-banana image
   model. The rendered PNGs feed the existing pipelines:
     * long-form  -> media_cloud `generate_thumbnail` cv2 text overlay
     * Shorts     -> micro_content_producer cover composition
   and are uploaded with `youtube.thumbnails.set` (video AND shorts).

2. Public-link mode (web-tool parity) — given a PUBLIC YouTube URL the tool
   itself fetches the video's metadata (title + description via the YouTube
   Data API `videos.list`, API key only) and the video's own thumbnail frame,
   analyses them, generates a new 16:9/9:16 thumbnail, and can set it on the
   video with `thumbnails.set` (channel-owner videos only). This mirrors "Nana
   Banana"-style web tools: they download the link's context and feed it to
   Gemini — the image model cannot fetch a URL by itself.

Pure functions, no FastAPI, no network at import time. Env gate:
`CSVG_NANO_BANANA_THUMBNAILS=1` (default) + `GEMINI_API_KEY`/`GOOGLE_API_KEY`.
Any failure degrades gracefully to None (callers keep the Flux/cv2, frame
extraction, or skip path) — thumbnail problems must never abort the pipeline.
"""
import os
import re as _re
import time
from typing import Optional, List, Dict, Any

NANO_BANANA_MODEL = "gemini-2.5-flash-image"


# ── availability gate ────────────────────────────────────────────────────────
def is_nano_banana_available() -> bool:
    """True when the env gate is on, a backend (AI-Studio key OR billed
    GOOGLE_CLOUD_PROJECT via ADC) exists and the google-genai SDK is installed.
    Never performs network I/O."""
    if os.getenv("CSVG_NANO_BANANA_THUMBNAILS", "1").strip().lower() in ("0", "false", "no"):
        return False
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    has_backend = bool((key and len(key) >= 8) or os.getenv("GOOGLE_CLOUD_PROJECT"))
    if not has_backend:
        return False
    try:
        from google import genai  # noqa: F401
        return True
    except ImportError:
        return False


def _genai_client():
    """Build a Gemini client. Prefers Vertex AI (project-scoped ADC — the
    billed path already used by llm_client) when GOOGLE_CLOUD_PROJECT is set,
    else the AI-Studio developer API key. Mirrors llm_client's routing."""
    from google import genai as _genai_mod
    from google.genai import types as _types

    if os.getenv("GOOGLE_CLOUD_PROJECT"):
        return _genai_mod.Client(http_options=_types.HttpOptions(api_version="v1"))
    return _genai_mod.Client(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        vertexai=False, enterprise=False,
    )


def _extract_inline_png(response) -> Optional[bytes]:
    """Pull the first image/PNG inline part from a generate_content response."""
    for candidate in getattr(response, "candidates", []) or []:
        for part in getattr(candidate.content, "parts", []) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "mime_type", "").lower().startswith("image/"):
                return bytes(inline.data)
    return None


# ── image generation ─────────────────────────────────────────────────────────
def generate_image(prompt: str, aspect_ratio: str = "16:9",
                   reference: Optional[bytes] = None) -> Optional[bytes]:
    """Generate one PNG via nano-banana. When `reference` (the source video's
    own frame) is provided the image is edited/regenerated from it so subject
    identity is preserved (web-tool "analyse the link" behaviour).

    aspect_ratio: "16:9" (long-form thumbnail) | "9:16" (Shorts cover).
    Primary path is the recommended `generate_content` image-modality API
    (generate_images/edit_images are deprecated/404 on current SDKs); that
    path also carries the optional reference image. Never raises — returns
    None so callers fall back to the Flux/cv2 or frame-extraction path."""
    if not is_nano_banana_available():
        return None
    if aspect_ratio not in ("16:9", "9:16"):
        aspect_ratio = "16:9"
    try:
        client = _genai_client()
        from google import genai as _genai_mod
        types = _genai_mod.types

        contents: Any = prompt
        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
        )
        if reference:
            config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
            )
            contents = [types.Part.from_text(text=prompt)]
            refs = reference if isinstance(reference, (list, tuple)) else [reference]
            for r in refs[:6]:
                if r:
                    contents.append(types.Part.from_bytes(data=r, mime_type="image/jpeg"))
        # Primary path — the recommended `generate_content` image-modality API.
        # Vertex rate-limits image predict calls per-minute, so transient 429s
        # get retried with backoff instead of immediately degrading.
        attempts = 3
        delay = 6.0
        last_err = None
        for attempt in range(attempts):
            try:
                resp = client.models.generate_content(
                    model=NANO_BANANA_MODEL, contents=contents, config=config
                )
                data = _extract_inline_png(resp)
                if data:
                    return bytes(data)
            except Exception as e1:
                last_err = e1
                if "RESOURCE_EXHAUSTED" in str(e1) or "429" in str(e1):
                    print(f"[NanoBanana] image quota backoff {delay:.0f}s (attempt {attempt + 1}/{attempts})...")
                    time.sleep(delay)
                    delay = min(delay * 2, 90.0)
                else:
                    print(f"[NanoBanana] generate_content image mode failed ({e1}); "
                          "trying legacy generate_images.")
                    break
        print(f"[NanoBanana] generate_content gave up after {attempts} attempts.")

        if reference is None:
            try:
                resp = client.models.generate_images(
                    model=NANO_BANANA_MODEL,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio=aspect_ratio,
                        output_mime_type="image/png",
                    ),
                )
                image = (resp.generated_images or [None])[0]
                if image is not None and getattr(image, "image", None) is not None:
                    gdata = getattr(image.image, "image_bytes", None)
                    if gdata:
                        return bytes(gdata)
            except Exception as e2:
                print(f"[NanoBanana] generate_images failed ({e2}).")
    except Exception as e:
        print(f"[NanoBanana] image generation failed: {e}")
    return None


def generate_image_edit(prompt: str, reference_bytes: Optional[bytes],
                        aspect_ratio: str = "16:9") -> Optional[bytes]:
    """Edit a reference frame into a new thumbnail. Thin wrapper over
    `generate_image` with the reference attached; returns raw PNG bytes or
    None."""
    if not reference_bytes:
        return None
    return generate_image(prompt, aspect_ratio=aspect_ratio, reference=reference_bytes)


def _write_png(data: Optional[bytes], output_path: str) -> bool:
    """Persist PNG bytes to disk. Returns True only when the file exists and
    is non-empty."""
    if not data:
        return False
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(data)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        print(f"[NanoBanana] failed to write PNG {output_path}: {e}")
        return False


# ── ready-to-upload text overlay (web-tool parity) ──────────────────────────
def _shorten_ctr_text(text: str) -> str:
    """Reduce to a single high-impact line: uppercase, at most 5 words —
    '3-5 words max' per high-CTR thumbnail guidelines (longer is unreadable)."""
    try:
        clean = _re.sub(r"\s+", " ", str(text or "")).strip().upper()
        words = clean.split()
        if len(words) > 5:
            words = words[:5]
        return " ".join(words) if words else "WATCH THIS"
    except Exception:
        return "WATCH THIS"


def _text_render_block(hook: str, aspect_ratio: str = "16:9") -> str:
    """Strict, guideline-compliant instruction that asks nano-banana to render
    the hook text NATIVELY inside the thumbnail. Enforces YouTube's on-image
    text rules explicitly so the model doesn't free-run with extra/small text:
    exact words only (0-3), ONE line, LARGE bold type (>=14% of frame height),
    max 60% frame width, high-contrast, safe-zone placement."""
    if aspect_ratio == "9:16":
        placement = (
            "centered horizontally in the TOP of the frame, its baseline around 12-18% "
            "of image height, so the bottom two-thirds stay clear for the subject and the "
            "engagement UI safe-zones (bottom 20% / right 15%) remain completely empty"
        )
    else:
        placement = (
            "in the UPPER-LEFT corner inside safe margins, away from the bottom-right "
            "(timestamp) and bottom (title) UI zones, and never overlapping the subject's face"
        )
    return (
        f"\nCRITICAL TEXT REQUIREMENT (obey EXACTLY): render ONLY these words — \"{hook}\" — "
        f"and no other words, no extra letters, no second line.\n"
        f"Placement: {placement}.\n"
        f"Typography rules (YouTube thumbnail guidelines): 3-5 words max; ONE single line; "
        f"thick, ultra-bold heavy sans-serif in the style of Bebas Neue or Montserrat Extra "
        f"Bold (cap-height at least 60px equivalent on a 1280px-wide canvas / 14% of image "
        f"height); the whole word string occupies between 40% and 60% of frame width so it "
        f"is instantly readable on mobile. The text MUST be separated from the background "
        f"with a thick black outline AND a drop shadow AND/OR a high-contrast color block "
        f"behind it (e.g. yellow text on dark navy, or white on red/black). HIGH-CONTRAST, "
        f"bright saturated lettering that pops in both light and dark YouTube mode. Do not "
        f"split words across lines, do not shrink the text, do not add punctuation or "
        f"hashtags. Keep the rest of the image clean and simple."
    )


def _bold_font_path() -> Optional[str]:
    try:
        from mcp_servers.media_cloud.server import _bold_font_path as _mcp_bold
        return _mcp_bold()
    except Exception:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None


def _fit_bold_font(text: str, font_path: Optional[str], max_width: int,
                   hi: int, lo: int):
    """Largest unused-bold font whose single line still fits max_width (px).
    Returns (font, size)."""
    from PIL import ImageFont
    path = font_path
    for size in range(hi, lo - 1, -2):
        f = ImageFont.truetype(path, size) if path else ImageFont.load_default()
        l, t, r, b = f.getbbox(text)
        if (r - l) <= max_width:
            return f, size
    f = ImageFont.truetype(path, lo) if path else ImageFont.load_default()
    return f, lo


def add_thumbnail_text(data: Optional[bytes], headline: str, subtitle: str = "",
                       aspect_ratio: str = "16:9") -> Optional[bytes]:
    """Burn a ready-to-upload high-CTR text overlay onto generated art.
    16:9 -> hook in the upper-LEFT safe zone (timestamp-safe); 9:16 -> hook
    centered in the top two-thirds. Returns PNG bytes or None on any failure
    (callers fall back to the plain art)."""
    if not data:
        return None
    try:
        from io import BytesIO
        from PIL import Image, ImageDraw, ImageFilter, ImageFont
        target = (1280, 720) if aspect_ratio == "16:9" else (1080, 1920)
        img = Image.open(BytesIO(data)).convert("RGB")
        if img.size != target:
            img = img.resize(target, Image.LANCZOS)
        base = img.convert("RGBA")
        W, H = target

        hook = _shorten_ctr_text(headline)
        font_path = _bold_font_path()
        if aspect_ratio == "16:9":
            font, fsize = _fit_bold_font(hook, font_path,
                                         max_width=int(W * 0.80), hi=148, lo=60)
            x0 = int(W * 0.065)
            y0 = int(H * 0.21)
        else:
            font, fsize = _fit_bold_font(hook, font_path,
                                         max_width=int(W * 0.88), hi=236, lo=110)
            l, t, r, b = font.getbbox(hook)
            x0 = (W - (r - l)) // 2
            y0 = int(H * 0.16)
        l, t, r, b = font.getbbox(hook)
        tw, th = r - l, b - t

        # Soft blurred drop shadow separates the glyphs on any art tone.
        sh = Image.new("RGBA", base.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        sd.text((x0, y0 + 10), hook, font=font, fill=(0, 0, 0, 220),
                stroke_width=8, stroke_fill=(0, 0, 0, 220))
        sh = sh.filter(ImageFilter.GaussianBlur(8))
        base = Image.alpha_composite(base, sh)
        d = ImageDraw.Draw(base)

        # Ultra-bold reach: thick dark outline + same-color second pass.
        d.text((x0, y0), hook, font=font, fill=(255, 255, 255, 255),
               stroke_width=12, stroke_fill=(10, 12, 16, 255))
        d.text((x0, y0), hook, font=font, fill=(255, 255, 255, 255),
               stroke_width=4, stroke_fill=(255, 255, 255, 255))

        # 10% accent bar (60-30-10 color rule).
        bar_w = max(int(tw * 0.5), 110)
        d.rounded_rectangle([x0, y0 + th + 18, x0 + bar_w, y0 + th + 18 + 16],
                            radius=8, fill=(255, 211, 25, 255))

        if aspect_ratio == "16:9" and subtitle:
            sub = _re.sub(r"\s+", " ", str(subtitle).strip().upper())[:40]
            if sub:
                sfont, _sf = _fit_bold_font(sub, font_path, max_width=int(W * 0.6), hi=38, lo=22)
                d.text((x0 + 4, y0 + fsize + 16), sub, font=sfont,
                       fill=(255, 255, 255, 230), stroke_width=3, stroke_fill=(12, 14, 18, 255))

        buf = BytesIO()
        base.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"[NanoBanana] text overlay failed; returning raw art: {e}")
        return data


# ── YouTube custom-thumbnail compliance (programmatic guideline check) ──────
def comply_thumbnail(data: Optional[bytes], aspect_ratio: str = "16:9"):
    """Enforce YouTube's published custom-thumbnail rules and return
    (final_bytes, report):

    * exact content resolution 1280x720 (16:9) / 1080x1920 (9:16)
    * brightness lift when the art is too dark for the feed (YouTube "pop")
    * modest contrast + color boost
    * JPEG export with a hard <=2MB budget (rules: JPG/PNG/GIF/BMP/WEBP ≤2MB)

    report holds the measured checks so callers can log a PASS/FAIL checklist.
    On any failure returns (data, {"error": ...}) — never aborts."""
    report = {"aspect_ratio": aspect_ratio}
    if not data:
        report["error"] = "no image data"
        return data, report
    try:
        from io import BytesIO
        from PIL import Image, ImageEnhance
        target = (1280, 720) if aspect_ratio == "16:9" else (1080, 1920)
        img = Image.open(BytesIO(data)).convert("RGB")
        if img.size != target:
            img = img.resize(target, Image.LANCZOS)
        report["dims_ok"] = True

        try:
            hist = img.convert("L").histogram()
            total = sum(hist)
            mean_lum = sum(i * g for i, g in enumerate(hist)) / total if total else 0
        except Exception:
            mean_lum = 128.0
        report["mean_lum"] = round(mean_lum, 1)

        # Feed "pop": lift dark art, then boost contrast + color evenly.
        if mean_lum < 0.33 * 255:
            img = ImageEnhance.Brightness(img).enhance(min(1.45, (0.40 * 255) / max(mean_lum, 1.0)))
        img = ImageEnhance.Contrast(img).enhance(1.10)
        img = ImageEnhance.Color(img).enhance(1.12)
        img = ImageEnhance.Brightness(img).enhance(1.02)

        # JPEG with a byte budget — never ship anything a YouTube re-encode
        # could push over the 2MB custom-thumbnail cap.
        best = None
        last = None
        for q in (92, 88, 84, 78, 72, 64, 55):
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=True)
            b = buf.getvalue()
            last = (b, q)
            if len(b) <= 2048 * 1024:
                best = last
                break
        if best is None:
            best = last
        data, quality = best
        report["format"] = "jpeg"
        report["quality"] = quality
        report["size_kb"] = round(len(data) / 1024)
    except Exception as e:
        report["error"] = str(e)
    return data, report


def _thumbmime(path: str) -> str:
    """image/png for .png, else image/jpeg (thumbnails are exported as JPEG)."""
    low = path.lower()
    if low.endswith(".png"):
        return "image/png"
    return "image/jpeg"


# ── art-prompt crafting (LLM analysis with deterministic fallback) ──────────
def _headline(state) -> str:
    if state.seo_metadata and state.seo_metadata.title:
        return state.seo_metadata.title
    if state.script_data and state.script_data.title:
        return state.script_data.title
    return getattr(state.selected_topic, "headline", "") if state.selected_topic else ""


def _hook(state) -> str:
    if state.seo_metadata and state.seo_metadata.thumbnail_brief:
        return state.seo_metadata.thumbnail_brief
    return _headline(state)


def _facts(state) -> List[str]:
    out = []
    for f in (state.verified_facts or []):
        s = (getattr(f, "summary", "") or "").strip()
        if s:
            out.append(s)
        if len(out) >= 4:
            break
    return out


def _meta_from_state(state) -> Dict[str, Any]:
    acts = ""
    if state.seo_metadata and getattr(state.seo_metadata, "act_titles", None):
        acts = "; ".join([a for a in state.seo_metadata.act_titles if a][:6])
    narration = ""
    if state.script_data and state.script_data.shots:
        narration = " ".join([
            (getattr(s, "narration_text", "") or "").strip() for s in state.script_data.shots
            if (getattr(s, "narration_text", "") or "").strip()
        ])[:3000]
    facts = "\n".join(_facts(state))
    description = facts
    if narration:
        description = (description + "\n\nTRANSCRIPT:\n" + narration) if description else narration
    return {
        "video_id": getattr(state, "pipeline_id", ""),
        "title": _headline(state),
        "hook": _hook(state),
        "hero_scene": "",
        "description": description,
        "context": f"Act structure: {acts}",
        "thumb_bytes": None,
        "transcript": narration,
        "frames": [],
    }


def _shorten(text: str, limit: int = 90) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) > limit:
        return text[:limit].rsplit(" ", 1)[0]
    return text


def _fallback_prompt_from_meta(meta: Dict[str, Any], aspect_ratio: str = "16:9") -> str:
    """Deterministic high-CTR art prompt when the LLM analysis pass is
    unavailable. Mirrors the pipeline composition rules (rule-of-thirds,
    no-text guard, human-emotion face hook) parameterized by aspect ratio."""
    hook = _shorten((meta or {}).get("hook") or (meta or {}).get("title") or "the story's subject")
    subject = hook if hook else "the story's subject"
    if len(subject) > 64:
        subject = subject[:64].rsplit(" ", 1)[0]
    scene = (meta.get("hero_scene") or "").strip() or subject
    desc = _shorten(meta.get("description") or "", 240)

    if aspect_ratio == "9:16":
        framing = (
            "Vertical 9:16 portrait YouTube Shorts cover, 1080x1920. CORE DESIGN RULES: "
            "SIMPLICITY — limit the frame to 2-3 visual elements max (subject face + one "
            "clear prop/backdrop + the text zone); RULE OF THIRDS — place the face or main "
            "subject on an upper grid intersection and leave 30-40% of the frame as clean "
            "negative space (reserved for the bold text hook in the top-center); HIGH "
            "CONTRAST — bright, saturated colors that pop in both light and dark YouTube "
            "mode (e.g. a vivid accent like yellow or red against a deep navy/black "
            "background); genuine human emotion — one expressive face showing a clear "
            "reaction (curiosity, shock, or a problem-to-solve), never a blank stare. "
            "Minimal background detail, no clutter — instantly readable at phone-feed size."
        )
    else:
        framing = (
            "Widescreen 16:9 landscape YouTube thumbnail, 1280x720. CORE DESIGN RULES: "
            "SIMPLICITY — limit the frame to 2-3 visual elements max (subject face + one "
            "clear prop/backdrop + the text zone); RULE OF THIRDS — place the face or main "
            "subject on a grid intersection in the RIGHT third and leave 30-40% of the "
            "frame as clean negative space (the LEFT third, reserved for the bold text "
            "overlay), never dead-centered; HIGH CONTRAST — bright, saturated colors that "
            "pop in both light and dark YouTube mode (e.g. a vivid accent like yellow or "
            "red against a deep navy/black background); genuine human emotion — one "
            "expressive face showing a clear reaction (curiosity, shock, or a "
            "problem-to-solve), never a blank stare. Minimal background detail, no clutter "
            "— instantly readable at postage-stamp size."
        )
    ctx = f" Theme reference: {desc}." if desc else ""
    return (
        f"{framing} Subject: {subject}. Art direction: {scene}.{ctx} "
        "Monumental, high-production cinematic realism, dramatic rim "
        "lighting, ultra high contrast, vivid vibrant colors. "
        "Keep the on-screen subject visually consistent with the video's own "
        "frame. "
        "Absolutely no text, no words, no letters, no numbers, no typography, "
        "no watermark, no logo, no UI overlay."
    )


def _llm_art_prompt(meta: Dict[str, Any], aspect_ratio: str,
                    llm) -> Optional[str]:
    """One LLM pass analysing the video context into a click-worthy art
    prompt. Returns the prompt string, or None when no usable prompt came
    back."""
    hook = _shorten(meta.get("hook") or meta.get("title") or "", 64)
    title = _shorten(meta.get("title") or "", 160)
    desc = _shorten(meta.get("description") or "", 1000)
    scene = _shorten(meta.get("hero_scene") or "", 200)
    ctx = _shorten(meta.get("context") or "", 300)
    target = "9:16 vertical Shorts cover (1080x1920)" if aspect_ratio == "9:16" else "16:9 widescreen thumbnail (1280x720)"

    system_prompt = (
        "You are a senior YouTube thumbnail art director. Analyse the video's "
        "story and write a single detailed, high-converting, click-worthy image "
        "generation prompt for the thumbnail art (any visible text overlay is "
        "added separately — the art must contain NO text). Return ONLY valid "
        "JSON of the form {\"art_prompt\": \"...\"}."
    )
    user_prompt = (
        f"Video title: {title or '(unknown)'}\n"
        f"Thumbnail text hook (displayed separately, do NOT render text): {hook}\n"
        f"Hero scene art direction: {scene or '(none given)'}\n"
        f"{ctx}\n"
        f"Key facts / description:\n{desc or '(none)'}\n"
        f"\nWrite ONE photorealistic, high-CTR image prompt for a {target}.\n"
        "Follow these CORE DESIGN RULES in the art: SIMPLICITY — max 2-3 visual elements; "
        "RULE OF THIRDS — subject/face on a grid intersection with 30-40% clean negative "
        "space reserved for a text overlay; HIGH CONTRAST — bright saturated colors that "
        "pop in both light and dark YouTube mode (e.g. a vivid yellow or red accent against "
        "deep navy/black); GENUINE EMOTION — one expressive human face with a clear "
        "reaction (curiosity, shock, or a problem-to-solve), never a blank stare, never a "
        "fake grin; minimal clutter, sharp focus, instantly readable at small size. Keep "
        "subject identity consistent with the video's content. Absolutely no "
        "text/words/letters/numbers/logos/watermarks/UI in the art."
    )
    parsed = llm.generate_json(user_prompt, system_prompt=system_prompt, route="generate")
    art = (parsed or {}).get("art_prompt")
    if art and isinstance(art, str) and len(art.strip()) >= 20:
        return art.strip()
    return None


def craft_ctr_hook(meta: Dict[str, Any], llm=None) -> str:
    """Derive a 2-3 word THEMATIC, curiosity-driven on-image hook from the
    video's actual content (title + description/transcript) — not the title
    verbatim. Falls back to the first 3 words of the title when the LLM is
    unavailable. Never raises."""
    fallback = _shorten_ctr_text((meta or {}).get("hook") or (meta or {}).get("title") or "")
    if not is_nano_banana_available():
        return fallback
    try:
        if llm is None:
            from src.engine.llm_client import LLMClient
            llm = LLMClient()
        title = _shorten((meta or {}).get("title") or "", 160)
        desc = _shorten((meta or {}).get("description") or "", 1500)
        system_prompt = (
            "You craft high-CTR on-image YouTube thumbnail hooks. Return ONLY valid JSON of "
            "the form {\"hook\": \"...\"}. The hook must be 3-5 words, UPPERCASE, a thematic, "
            "curiosity-driven, engaging phrase that captures the video's CORE THEME and "
            "makes people want to click (for example \"SILENT PAY CUT\", \"WHO REALLY WINS?\", "
            "\"YOUR MONEY SHRINKS\"). Make it emotionally engaging (shock, curiosity, or a "
            "clear problem-to-solve) but never the video title verbatim and never a lie that "
            "contradicts the content. Keep it short — 3 to 5 words max. No punctuation except "
            "an optional trailing question mark."
        )
        user_prompt = (
            f"Video title: {title or '(unknown)'}\n"
            f"Content / transcript:\n{desc or '(none)'}\n\n"
            "Write the 2-3 word thematic thumbnail hook:"
        )
        parsed = llm.generate_json(user_prompt, system_prompt=system_prompt, route="generate")
        hook = (parsed or {}).get("hook")
        if hook and isinstance(hook, str):
            cleaned = _shorten_ctr_text(hook)
            if cleaned and cleaned != "WATCH THIS":
                print(f"[NanoBanana] thematic hook: {cleaned}")
                return cleaned
    except Exception as e:
        print(f"[NanoBanana] hook craft failed; using fallback: {e}")
    return fallback


def craft_thumbnail_prompt_from_metadata(meta: Dict[str, Any], aspect_ratio: str = "16:9",
                                         llm=None) -> str:
    """Analyse video metadata (title/hook/description/hero scene) with the LLM
    and return a click-worthy art prompt. Falls back to the deterministic
    prompt when the LLM is unavailable or unusable. Never raises."""
    fallback = _fallback_prompt_from_meta(meta, aspect_ratio)
    if not is_nano_banana_available():
        return fallback
    try:
        if llm is None:
            from src.engine.llm_client import LLMClient
            llm = LLMClient()
        art = _llm_art_prompt(meta, aspect_ratio, llm)
        return art if art else fallback
    except Exception as e:
        print(f"[NanoBanana] LLM art-prompt analysis failed; using fallback: {e}")
        return fallback


def craft_thumbnail_art_prompt(state, hero_scene: str = "", aspect_ratio: str = "16:9",
                               llm=None) -> str:
    """Pipeline mode: analyse the video (title/hook/acts/facts + hero scene)
    with the LLM and return a click-worthy art prompt. Falls back to the
    deterministic prompt when the LLM is unavailable. Never raises."""
    meta = _meta_from_state(state)
    if hero_scene:
        meta["hero_scene"] = hero_scene
    return craft_thumbnail_prompt_from_metadata(meta, aspect_ratio=aspect_ratio, llm=llm)


# ── public YouTube link analysis (web-tool parity) ──────────────────────────
def extract_video_id(url_or_id: str) -> str:
    """Parse a YouTube link to its 11-char video id. Accepts watch?v=,
    youtu.be/, /shorts/, /embed/, /live/, or a bare 11-char id. Returns ''
    when nothing resolvable is found (never guesses from arbitrary URLs)."""
    text = (url_or_id or "").strip()
    if not text:
        return ""
    if _re.fullmatch(r"[A-Za-z0-9_-]{11}", text):
        return text
    for pat in (
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"watch\?.*?[?&]v=([A-Za-z0-9_-]{11})",
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"/(?:shorts|embed|live|v|u)/[^?#\s]*([A-Za-z0-9_-]{11})",
    ):
        m = _re.search(pat, text)
        if m:
            return m.group(1)
    return ""


def _like_real_key(value: Optional[str]) -> bool:
    """True when an env key looks like a real credential (not the example.env
    'xxx' placeholders)."""
    if not value:
        return False
    s = value.strip()
    if len(s) < 10 or s.lower().startswith("your_"):
        return False
    return len(set(s)) > 3  # "xxxxxxxxxx..." placeholder heuristic


def fetch_video_transcript(video_id: str) -> str:
    """Best-effort auto-captions transcript for a public video — the single
    richest content signal about what the video actually says.

    Prefers the OAuth caption download (the SAME credentials the upload path
    uses — owner-only, works from the OCI datacenter IP), then falls back to
    youtube-transcript-api. Returns '' when unavailable. Never raises."""
    try:
        from mcp_servers.youtube_cloud.server import fetch_caption_transcript_oauth
        text = fetch_caption_transcript_oauth(video_id)
        if text:
            return text
    except Exception as e:
        print(f"[NanoBanana] OAuth transcript unavailable: {e}")
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        try:
            tl = api.fetch(video_id, languages=["en"])
            texts = [getattr(s, "text", "") for s in tl]
        except TypeError:
            tl = api.get_transcript(video_id)
            texts = [s.get("text", "") for s in tl]
        except Exception:
            tl = api.fetch(video_id)
            texts = [getattr(s, "text", "") for s in tl]
        return " ".join([t for t in texts if t])[:6000]
    except Exception as e:
        print(f"[NanoBanana] transcript fetch unavailable: {e}")
        return ""


def extract_video_frames(video_id: str, max_frames: int = 4) -> List[bytes]:
    """Download a low-res copy of the video and pull `max_frames` representative
    JPEG frames spread across its duration. These are the video's REAL visuals
    (vs its single cover frame) and are fed to the model so the thumbnail
    reflects the actual content. Returns a list of JPEG bytes; [] on any
    failure (callers fall back to the single cover frame)."""
    import subprocess
    frames: List[bytes] = []
    tmpdir = None
    try:
        import yt_dlp
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="csvg_vid_")
        outtmpl = os.path.join(tmpdir, "v.%(ext)s")
        ydl_opts = {
            "format": "best[height<=480]/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 20,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        src = ydl.prepare_filename(info) if "ydl" in dir() else ""
        if not src or not os.path.exists(src):
            import glob as _glob
            cands = _glob.glob(os.path.join(tmpdir, "v.*"))
            src = cands[0] if cands else ""
        if not src or not os.path.exists(src):
            return frames

        dur = 0.0
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", src],
            capture_output=True, text=True, timeout=60)
        try:
            dur = float(r.stdout.strip())
        except ValueError:
            dur = 0.0
        if dur <= 0:
            return frames

        n = min(max_frames, 4)
        for i in range(n):
            t = max(0.0, dur * (i + 0.5) / n)
            out = os.path.join(tmpdir, f"f{i}.jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", src, "-frames:v", "1",
                 "-vf", "scale=640:-1", "-q:v", "3", out],
                capture_output=True, timeout=60)
            if os.path.exists(out) and os.path.getsize(out) > 1000:
                with open(out, "rb") as f:
                    frames.append(f.read())
        print(f"[NanoBanana] extracted {len(frames)} video frame(s) for {video_id}")
    except Exception as e:
        print(f"[NanoBanana] frame extraction unavailable: {e}")
    finally:
        if tmpdir:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
    return frames


def fetch_link_video_metadata(video_id: str) -> Dict[str, Any]:
    """Best-effort metadata for a PUBLIC video: title + description via the
    YouTube Data API `videos.list` (API-key only, no OAuth), plus the video's
    own current thumbnail frame (maxres/hqdefault) as a visual reference that
    gets fed to nano-banana's EDIT mode. Never raises; returns a dict always."""
    meta = {"video_id": video_id, "title": "", "description": "", "hook": "",
            "hero_scene": "", "context": "", "thumb_bytes": None,
            "transcript": "", "frames": []}
    if not video_id:
        return meta
    try:
        import requests
        key = os.getenv("YOUTUBE_API_KEY") or os.getenv("YOUTUBE_API_KEY_FALLBACK")
        if _like_real_key(key):
            r = requests.get("https://www.googleapis.com/youtube/v3/videos",
                             params={"part": "snippet", "id": video_id, "key": key}, timeout=15)
            if r.status_code == 200:
                items = (r.json() or {}).get("items", [])
                if items:
                    snip = items[0].get("snippet", {})
                    meta["title"] = snip.get("title", "")
                    meta["description"] = snip.get("description", "")
        # The video's own frame → edit-ref for subject identity.
        for res in ("maxresdefault", "hqdefault"):
            r = requests.get(f"https://i.ytimg.com/vi/{video_id}/{res}.jpg", timeout=10)
            if r.status_code == 200 and r.content and len(r.content) > 500 and r.content[:2] == b"\xff\xd8":
                meta["thumb_bytes"] = r.content
                break
    except Exception as e:
        print(f"[NanoBanana] link metadata fetch failed: {e}")
    # Real content: transcript + representative frames (best-effort).
    meta["transcript"] = fetch_video_transcript(video_id)
    meta["frames"] = extract_video_frames(video_id)
    if meta.get("transcript"):
        meta["description"] = (meta.get("description") or "") + "\n\nTRANSCRIPT:\n" + meta["transcript"]
    return meta


def _finalize_thumbnail(data: Optional[bytes], aspect_ratio: str, output_path: str,
                        tag: str = "") -> Optional[str]:
    """Run the compliance pass and write the final file (.jpg when the pass
    emits JPEG). Returns the saved path or None."""
    if data is None:
        return None
    data, report = comply_thumbnail(data, aspect_ratio=aspect_ratio)
    if report.get("size_kb"):
        print(f"[NanoBanana] {tag}{aspect_ratio} compliance: "
              f"dims={report.get('dims_ok')}, mean_lum={report.get('mean_lum')}, "
              f"jpeg_q={report.get('quality')}, size={report.get('size_kb')}KB")
    if report.get("format") == "jpeg" and output_path.lower().endswith(".png"):
        output_path = os.path.splitext(output_path)[0] + ".jpg"
    if _write_png(data, output_path):
        print(f"[NanoBanana] Generated {tag}{aspect_ratio} link thumbnail -> {output_path}")
        return output_path
    return None


def generate_bare_thumbnail(video_url: str, aspect_ratio: str = "16:9",
                            output_path: str = "") -> Optional[str]:
    """Bare-prompt experiment: sends the literal dynamic link in the prompt —
    'Generate a thumbnail for this video <url>' — with the video's own frame as
    a visual reference (the model cannot open the URL itself; this just tests
    how it responds when the link is embedded in the text). No LLM analysis, no
    CTR text block, no guideline art direction. Compliance pass still applies."""
    prompt = f"Generate a thumbnail for this video {video_url}"
    meta = fetch_link_video_metadata(extract_video_id(video_url))
    ref = meta.get("frames") or ([meta.get("thumb_bytes")] if meta.get("thumb_bytes") else None)
    data = generate_image(prompt, aspect_ratio=aspect_ratio, reference=ref)
    if data is None:
        data = generate_image(prompt, aspect_ratio=aspect_ratio)
    return _finalize_thumbnail(data, aspect_ratio, output_path, tag="bare-")


def generate_link_thumbnail(meta: Dict[str, Any], aspect_ratio: str = "16:9",
                            output_path: str = "", add_text: bool = True,
                            native_text: bool = True) -> Optional[str]:
    """Generate a READY-TO-UPLOAD, YouTube-compliant 16:9/9:16 thumbnail for a
    public video link.

    Text is rendered NATIVELY by nano-banana inside the generated art (the
    web-tool behaviour) when `native_text=True` — the exact high-CTR 3-word
    hook is baked into the image-generation prompt with guideline placement
    (16:9 upper-left / 9:16 top-third). Set `native_text=False` to fall back to
    the PIL text overlay instead; `add_text=False` ships art only.

    Always ends with a compliance pass (exact dims, brightness lift, JPEG
    <=2MB). Output is written with a .jpg extension when compliance emits
    JPEG. Returns the saved path or None."""
    meta["hook"] = craft_ctr_hook(meta)
    hook = _shorten_ctr_text(meta["hook"])
    prompt = craft_thumbnail_prompt_from_metadata(meta, aspect_ratio=aspect_ratio)
    if add_text and native_text:
        prompt = prompt + _text_render_block(hook, aspect_ratio)
    ref = meta.get("frames") or ([meta.get("thumb_bytes")] if meta.get("thumb_bytes") else None)
    data = generate_image(prompt, aspect_ratio=aspect_ratio, reference=ref)
    if data is None:
        data = generate_image(prompt, aspect_ratio=aspect_ratio)
    if data is not None and add_text and not native_text:
        subtitle = meta.get("title", "") if aspect_ratio == "16:9" else ""
        data = add_thumbnail_text(data, headline=hook, subtitle=subtitle, aspect_ratio=aspect_ratio)
    text_tag = "text-" if add_text else ""
    mode_tag = "native-" if (add_text and native_text) else ""
    return _finalize_thumbnail(data, aspect_ratio, output_path, tag=mode_tag + text_tag)


# ── pipeline helpers ─────────────────────────────────────────────────────────
def generate_video_thumbnail_art(state, hero_scene: str = "", output_path: str = "") -> Optional[str]:
    """Generate 16:9 long-form thumbnail ART (no text — the cv2 overlay is
    applied by media_cloud's generate_thumbnail). Returns the saved path or
    None."""
    prompt = craft_thumbnail_art_prompt(state, hero_scene=hero_scene, aspect_ratio="16:9")
    data = generate_image(prompt, aspect_ratio="16:9")
    if _write_png(data, output_path):
        print(f"[NanoBanana] Generated 16:9 thumbnail art -> {output_path}")
        return output_path
    return None


def generate_shorts_cover_art(state, output_path: str = "") -> Optional[str]:
    """Generate 9:16 Shorts cover ART (the hook text is burned on later by
    micro_content_producer's cover composer). Returns the saved path or None."""
    prompt = craft_thumbnail_art_prompt(state, aspect_ratio="9:16")
    data = generate_image(prompt, aspect_ratio="9:16")
    if _write_png(data, output_path):
        print(f"[NanoBanana] Generated 9:16 Shorts cover art -> {output_path}")
        return output_path
    return None


def generate_baked_video_thumbnail(state, hero_scene: str = "", output_path: str = "",
                                   reference_frame: Optional[bytes] = None,
                                   llm=None) -> Optional[str]:
    """Baked 16:9 long-form thumbnail in the same high-CTR native-text design
    as the Shorts cover: thematic hook from the video content + design-rules
    art + model-rendered text + compliance. Ready for youtube.thumbnails.set
    (long-form: unlike Shorts, thumbnails.set IS honored by YouTube).

    Falls back to None so callers keep the art+cv2 path.
    reference_frame: optional mid-video frame used as an identity reference.
    llm: optional injectable LLM client for hermetically testable crafting."""
    meta = _meta_from_state(state)
    if hero_scene:
        meta["hero_scene"] = hero_scene
    meta["hook"] = craft_ctr_hook(meta, llm=llm)
    hook = _shorten_ctr_text(meta["hook"])
    prompt = craft_thumbnail_prompt_from_metadata(meta, aspect_ratio="16:9", llm=llm)
    prompt = prompt + _text_render_block(hook, "16:9")
    data = generate_image(prompt, aspect_ratio="16:9", reference=reference_frame)
    return _finalize_thumbnail(data, "16:9", output_path, tag="baked-video-")


def generate_baked_shorts_cover(state, output_path: str = "",
                                reference_frame: Optional[bytes] = None,
                                llm=None) -> Optional[str]:
    """Option-B cover for a NEW Short: full high-CTR design baked in one image —
    thematic hook derived from the video content + design-rules art + text
    rendered NATIVELY by the model + compliance pass. The returned JPEG is
    prepended as the Short's first frame (no external text burn needed).

    Falls back to None so callers keep the art+compose path.
    reference_frame: an existing Short frame (same video) used as an identity
    reference while the new cover is generated.
    llm: optional injectable LLM client for hermetically testable crafting."""
    meta = _meta_from_state(state)
    if state.script_data and state.script_data.shots:
        meta["hero_scene"] = state.script_data.shots[0].visual_prompt or ""
    meta["hook"] = craft_ctr_hook(meta, llm=llm)
    hook = _shorten_ctr_text(meta["hook"])
    prompt = craft_thumbnail_prompt_from_metadata(meta, aspect_ratio="9:16", llm=llm)
    prompt = prompt + _text_render_block(hook, "9:16")
    data = generate_image(prompt, aspect_ratio="9:16", reference=reference_frame)
    return _finalize_thumbnail(data, "9:16", output_path, tag="baked-cover-")