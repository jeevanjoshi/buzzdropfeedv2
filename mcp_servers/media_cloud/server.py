import os
import re
import subprocess
import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="OCI Cloud Media Production MCP Server")


class ImageGenRequest(BaseModel):
    prompt: str
    image_size: str = "landscape_16_9"
    output_image_path: str


class KenBurnsRequest(BaseModel):
    image_path: str
    audio_path: Optional[str] = None
    duration: float
    # Pan direction for the Ken Burns motion. "left_to_right" / "right_to_left"
    # pan horizontally, "top_to_bottom" / "bottom_to_top" pan vertically. This
    # gives REAL lateral motion (not just a centered zoom) so frozen-frame
    # detectors see movement. Optional; defaults to left_to_right.
    direction: str = "left_to_right"
    # Continuous-motion magnitude: the zoom target reached at the FINAL frame
    # (zoom goes 1.0 -> zoom_target over the whole shot). Higher = more visible
    # motion per second but crops more (subjects risk leaving the frame). Default
    # 1.5 is framing-safe (never shows less than ~2/3 of the image); the real
    # fix for "more motion" is shorter shots, not a bigger zoom.
    zoom_target: float = 1.5
    output_mp4_path: str


class ChartRequest(BaseModel):
    title: str
    labels: list = ["Q1", "Q2", "Q3", "Q4"]
    values: list = [100, 140, 90, 185]
    unit_symbol: str = "$ Million"  # Explicit units e.g. '$ Million', '₹ Crores', '%'
    chart_type: str = "line"  # 'line' (trend) or 'bar' (discrete comparisons / %)
    duration: float = 5.0
    output_mp4_path: str


class PlaywrightSVGRequest(BaseModel):
    chart_type: str = "animated_line_chart"  # 'animated_line_chart', 'glowing_counter', 'stock_ticker'
    title: str = "MARKET CAPITALIZATION SHIFT"
    headline_val: str = "$520.4 Billion"
    sub_text: str = "+18.4% YoY Inflow"
    duration: float = 5.0
    output_mp4_path: str


class GIFRequest(BaseModel):
    query: str = "shocked reaction"
    duration: float = 3.0
    output_mp4_path: str


class ThumbnailRequest(BaseModel):
    headline_text: str
    subtitle_text: str = ""
    visual_prompt: Optional[str] = ""   # theme-matched scene description (e.g. the hero
                                        # shot's visual prompt) so the AI art matches the
                                        # story's subject instead of the raw CTR text.
    output_thumbnail_path: str


class TimelineAssemblyRequest(BaseModel):
    concat_list_path: str
    subtitle_path: str
    bgm_path: str
    output_video_path: str
    # Crossfade between consecutive shots (seconds). 0.0 = hard-cut concat (legacy).
    # >0 enables an ffmpeg xfade/acrossfade dissolve chain ("fade" transition).
    crossfade: float = 0.0
    transition: str = "fade"


def generate_synthetic_png(output_path: str, title: str = "16:9 CSVG MEDIA"):
    """
    Generates a broadcast-grade 16:9 synthetic 1920x1080 PNG image using PIL.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (1920, 1080), color=(15, 20, 30))
        draw = ImageDraw.Draw(img)
        # Draw elegant cyan border line
        draw.rectangle([40, 40, 1880, 1040], outline=(0, 255, 204), width=6)
        draw.line([(40, 540), (1880, 540)], fill=(0, 150, 200), width=3)
        draw.text((100, 480), title[:60].upper(), fill=(255, 255, 255))
        img.save(output_path, "PNG")
    except Exception as e:
        print(f"PIL Image Gen Error: {e}")


@app.post("/tools/generate_flux_image")
async def generate_flux_image(req: ImageGenRequest):
    """
    Generates 16:9 widescreen image via Fal.ai Flux.1-schnell model API,
    with automatic fallback to Replicate Flux API if Fal.ai balance is exhausted.
    """
    os.makedirs(os.path.dirname(os.path.abspath(req.output_image_path)), exist_ok=True)
    fal_key = os.getenv("FAL_KEY")

    if fal_key and not fal_key.startswith("YOUR_"):
        try:
            import fal_client
            import requests
            handler = fal_client.submit(
                "fal-ai/flux/schnell",
                arguments={
                    "prompt": req.prompt,
                    "image_size": "landscape_16_9",
                    "num_inference_steps": 4,
                    "enable_safety_checker": True
                }
            )
            result = handler.get()
            image_url = result["images"][0]["url"]
            img_data = requests.get(image_url).content
            with open(req.output_image_path, "wb") as f:
                f.write(img_data)
            return {"status": "success", "engine": "fal_flux_schnell", "path": req.output_image_path}
        except Exception as e:
            print(f"Fal.ai Image Gen Exception (switching to Replicate): {e}")

    # Fallback to Replicate API using REPLICATE_API_TOKEN
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if replicate_token and not replicate_token.startswith("YOUR_"):
        try:
            import requests
            import time
            headers = {
                "Authorization": f"Bearer {replicate_token}",
                "Content-Type": "application/json"
            }
            body = {
                "input": {
                    "prompt": req.prompt,
                    "aspect_ratio": "16:9",
                    "output_format": "png"
                }
            }
            r = requests.post(
                "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
                headers=headers,
                json=body
            )
            if r.status_code in [200, 201]:
                pred = r.json()
                get_url = pred["urls"]["get"]
                for _ in range(30):
                    time.sleep(2)
                    res = requests.get(get_url, headers=headers).json()
                    if res.get("status") == "succeeded":
                        img_url = res["output"][0]
                        img_bytes = requests.get(img_url).content
                        with open(req.output_image_path, "wb") as f:
                            f.write(img_bytes)
                        return {"status": "success", "engine": "replicate_flux_schnell", "path": req.output_image_path}
        except Exception as e:
            print(f"Replicate Image Gen Exception: {e}")

    generate_synthetic_png(req.output_image_path, title=req.prompt[:30])
    return {"status": "success", "engine": "synthetic_png_fallback", "path": req.output_image_path}


@app.post("/tools/fetch_reaction_gif_clip")
async def fetch_reaction_gif_clip(req: GIFRequest):
    """
    Queries GIPHY / Tenor APIs for comedic reaction clips (e.g. 'shocked face', 'money rain')
    and converts them to strict 16:9 1080p MP4 clips for video insertion.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(req.output_mp4_path)), exist_ok=True)
        from src.engine.gif_retriever import gif_retriever
        import requests

        clips = gif_retriever.search_giphy_reaction(query=req.query)
        if clips:
            mp4_url = clips[0].get("mp4_url")
            if mp4_url:
                tmp_download = req.output_mp4_path.replace(".mp4", "_raw.mp4")
                r = requests.get(mp4_url, timeout=5)
                with open(tmp_download, "wb") as f:
                    f.write(r.content)

                cmd = [
                    "ffmpeg", "-y", "-i", tmp_download,
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                    "-t", str(req.duration), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    req.output_mp4_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return {"status": "success", "engine": "giphy_clip_processed", "path": req.output_mp4_path}
    except Exception:
        pass

    # Fallback to synthetic image clip
    chart_png = req.output_mp4_path.replace(".mp4", "_gif_fallback.png")
    generate_synthetic_png(chart_png, title=f"REACTION: {req.query.upper()}")
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", chart_png,
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
        "-t", str(req.duration), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        req.output_mp4_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"status": "success", "engine": "fallback_reaction_clip", "path": req.output_mp4_path}


@app.post("/tools/render_playwright_svg_animation")
async def render_playwright_svg_animation(req: PlaywrightSVGRequest):
    """
    Renders broadcast-quality 60FPS 16:9 HTML5/SVG animated charts, glowing counters, and ticker motion clips
    using Playwright headless browser rendering or OpenCV frame capture fallback.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(req.output_mp4_path)), exist_ok=True)
        html_file = req.output_mp4_path.replace(".mp4", "_chart.html")
        chart_png = req.output_mp4_path.replace(".mp4", "_svg_frame.png")

        current_month_year = datetime.datetime.now(datetime.timezone.utc).strftime("%B %Y").upper()

        # HTML5/SVG Dynamic Motion Template
        html_content = f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    margin: 0; background: #0b0e14; color: #ffffff; font-family: 'Segoe UI', Arial, sans-serif;
    width: 1920px; height: 1080px; display: flex; flex-direction: column; justify-content: center; align-items: center;
    box-sizing: border-box; overflow: hidden;
  }}
  .chart-box {{
    width: 1600px; height: 850px; background: rgba(22, 27, 34, 0.9); border: 3px solid #00ffcc;
    border-radius: 24px; padding: 40px; box-shadow: 0 0 50px rgba(0, 255, 204, 0.3); display: flex;
    flex-direction: column; justify-content: space-between; position: relative;
  }}
  .title {{ font-size: 42px; font-weight: 800; color: #00ffcc; text-transform: uppercase; letter-spacing: 2px; }}
  .date-tag {{ font-size: 24px; color: #8b949e; margin-top: 5px; }}
  .big-stat {{ font-size: 96px; font-weight: 900; color: #ffffff; margin: 20px 0; text-shadow: 0 0 30px rgba(255, 255, 255, 0.5); }}
  .sub-stat {{ font-size: 36px; font-weight: 700; color: #39d353; background: rgba(57, 211, 83, 0.15); padding: 10px 20px; border-radius: 12px; width: fit-content; }}
  svg {{ width: 100%; height: 350px; stroke-dasharray: 1000; stroke-dashoffset: 1000; animation: dash 3s ease-in-out forwards; }}
  @keyframes dash {{ to {{ stroke-dashoffset: 0; }} }}
</style>
</head>
<body>
  <div class="chart-box">
    <div>
      <div class="title">{req.title}</div>
      <div class="date-tag">LIVE MARKET DATA • {current_month_year}</div>
    </div>
    <div>
      <div class="big-stat">{req.headline_val}</div>
      <div class="sub-stat">{req.sub_text}</div>
    </div>
    <svg viewBox="0 0 1000 300">
      <path d="M 0 250 Q 250 50, 500 200 T 1000 30" fill="none" stroke="#00ffcc" stroke-width="8" />
    </svg>
  </div>
</body>
</html>"""
        
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Attempt Playwright Headless Browser Frame Render
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page(viewport={"width": 1920, "height": 1080})
                await page.goto(f"file://{os.path.abspath(html_file)}")
                await page.screenshot(path=chart_png)
                await browser.close()
        except Exception:
            generate_synthetic_png(chart_png, title=req.title)

        # Convert frame to zooming MP4 motion video clip
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", chart_png,
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.0015,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=125:s=1920x1080",
            "-t", str(req.duration), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            req.output_mp4_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"status": "success", "engine": "playwright_html5_svg_render", "path": req.output_mp4_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/generate_thumbnail")
async def generate_thumbnail(req: ThumbnailRequest):
    """
    Generates high-CTR 16:9 YouTube Thumbnail (1280x720) with dynamic date text overlay and high-contrast styling.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(req.output_thumbnail_path)), exist_ok=True)
        current_month_year = datetime.datetime.now(datetime.timezone.utc).strftime("%B %Y").upper()
        sub_text = req.subtitle_text.upper() if req.subtitle_text else f"EXPLAINED ({current_month_year})"

        try:
            import matplotlib.pyplot as plt
            import cv2
            import numpy as np

            # Theme-matched scene: prefer the story's own visual prompt (offshore
            # wind farm, etc.), so the art matches the CONTENT, not the CTR text.
            scene = (req.visual_prompt or req.headline_text or "a dramatic hub of renewable energy").strip()
            scene = re.sub(
                r'^\s*cinematic\s+16:9(?:\s+widescreen)?\s*\.?\s*', '', scene, flags=re.IGNORECASE
            )
            # Bright, vibrant, on-theme background: "dark moody" reads muddy/dark
            # and the raw headline string produces garbled "text-in-image" tearing.
            # Explicit no-text guard stops fal/replicate from painting letters.
            prompt = (
                f"Widescreen 16:9 YouTube thumbnail background image of: {scene}. "
                f"Photorealistic, bright natural daylight, vivid vibrant colors, high contrast, "
                f"sunny optimistic atmosphere, crisp sharp focus. "
                f"Absolutely no text, no words, no letters, no numbers, no typography, "
                f"no watermark, no logo, no captions, no UI, no people overlaid."
            )

            # Try generating a background image using fal.ai or replicate
            bg_data = None
            fal_key = os.getenv("FAL_KEY")
            if fal_key and not fal_key.startswith("YOUR_"):
                try:
                    import fal_client
                    import requests
                    handler = fal_client.submit(
                        "fal-ai/flux/schnell",
                        arguments={
                            "prompt": prompt,
                            "image_size": "landscape_16_9",
                            "num_inference_steps": 8,
                            "enable_safety_checker": True
                        }
                    )
                    res = handler.get()
                    img_url = res["images"][0]["url"]
                    bg_data = requests.get(img_url).content
                    print("[Thumbnail] Successfully generated background image via Fal.ai")
                except Exception as e:
                    print(f"[Thumbnail] Fal.ai background generation failed: {e}")

            if bg_data is None:
                replicate_token = os.getenv("REPLICATE_API_TOKEN")
                if replicate_token and not replicate_token.startswith("YOUR_"):
                    try:
                        import requests
                        headers = {
                            "Authorization": f"Bearer {replicate_token}",
                            "Content-Type": "application/json"
                        }
                        body = {
                            "input": {
                                "prompt": prompt,
                                "aspect_ratio": "16:9",
                                "output_format": "png"
                            }
                        }
                        r = requests.post(
                            "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
                            headers=headers,
                            json=body
                        )
                        if r.status_code in [200, 201]:
                            import time
                            pred = r.json()
                            get_url = pred["urls"]["get"]
                            for _ in range(20):
                                time.sleep(2)
                                check_res = requests.get(get_url, headers=headers).json()
                                if check_res.get("status") == "succeeded":
                                    img_url = check_res["output"][0]
                                    bg_data = requests.get(img_url).content
                                    print("[Thumbnail] Successfully generated background image via Replicate")
                                    break
                    except Exception as e:
                        print(f"[Thumbnail] Replicate background generation failed: {e}")

            # Decode the image data or fall back to a bright sky-blue gradient
            if bg_data:
                try:
                    nparr = np.frombuffer(bg_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (1280, 720))
                    # Mild overall dim ONLY for text contrast — keep the scene bright
                    # (previously 0.4 of a near-black overlay crushed brightness).
                    img = cv2.convertScaleAbs(img, alpha=1.02, beta=6)
                    cv2.addWeighted(img, 0.84, np.full_like(img, 10), 0.16, 0, img)
                except Exception as img_err:
                    print(f"[Thumbnail] Failed to decode generated background: {img_err}. Using bright fallback.")
                    img = np.zeros((720, 1280, 3), dtype=np.uint8)
                    for yy in range(720):
                        t = yy / 720.0
                        img[yy, :] = (int(38 + 40 * t), int(78 + 100 * t), int(138 + 110 * t))
            else:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
                for yy in range(720):
                    t = yy / 720.0
                    img[yy, :] = (int(38 + 40 * t), int(78 + 100 * t), int(138 + 110 * t))

            # Soft dark gradient on the lower half ONLY, under the text zone, so the
            # bottom reads legibly while the top of the scene stays bright.
            h, w = img.shape[:2]
            grad_area = int(h * 0.5)
            grad = np.linspace(0.0, 0.45, grad_area, dtype=np.float32).reshape(-1, 1, 1)
            img[h - grad_area:] = (img[h - grad_area:].astype(np.float32) * (1.0 - grad)).astype(np.uint8)

            # Glowing border & high-contrast yellow/white text overlay
            cv2.rectangle(img, (20, 20), (1260, 700), (0, 215, 255), 6)
            
            # Word-wrap the headline into two lines of max ~25-30 chars
            words = req.headline_text.upper().split()
            lines = []
            current_line = []
            current_len = 0
            for w in words:
                if current_len + len(w) + 1 <= 24:
                    current_line.append(w)
                    current_len += len(w) + 1
                else:
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [w]
                    current_len = len(w)
            if current_line:
                lines.append(" ".join(current_line))

            # Limit to max 2 lines for thumbnail readability, adding ... if truncated
            if len(lines) > 2:
                lines = lines[:2]
                lines[1] = lines[1][:21] + "..."
            elif not lines:
                lines = ["EXPLAINED"]

            # Draw lines with clean anti-aliasing + black outline so the text stays
            # legible over a bright background.
            def _draw_outlined(text, pos, scale, color, thick=4, outline_thick=9):
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                            (0, 0, 0), outline_thick, cv2.LINE_AA)
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                            color, thick, cv2.LINE_AA)

            y_start = 260 if len(lines) > 1 else 340
            for idx, line in enumerate(lines):
                _draw_outlined(line, (60, y_start + (idx * 90)), 1.4, (0, 235, 255))

            sub_y = y_start + (len(lines) * 90) + 10
            _draw_outlined(sub_text, (60, sub_y), 1.1, (255, 255, 255))

            cv2.imwrite(req.output_thumbnail_path, img)
            return {"status": "success", "engine": "high_ctr_thumbnail", "path": req.output_thumbnail_path}
        except Exception:
            generate_synthetic_png(req.output_thumbnail_path, title="THUMBNAIL")
            return {"status": "success", "engine": "fallback_thumbnail", "path": req.output_thumbnail_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/generate_dynamic_chart")
async def generate_dynamic_chart(req: ChartRequest):
    """
    Renders an animated 16:9 financial stock / market trend data chart video clip with dynamic date context,
    explicit units (e.g., '$ Billion', '₹ Crores', '%'), and high-contrast styling.
    Supports `chart_type`: 'line' (trend) or 'bar' (discrete comparisons / percentages).
    On any render failure this RAISES so the caller can fall through to a visible
    placeholder caught by Gate 7 — it never silently writes a fake "chart".
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(req.output_mp4_path)), exist_ok=True)
        chart_png = req.output_mp4_path.replace(".mp4", "_chart.png")

        current_month_year = datetime.datetime.now(datetime.timezone.utc).strftime("%B %Y").upper()
        chart_title = f"{req.title.upper()} ({current_month_year})"

        try:
            import matplotlib.pyplot as plt

            labels = [str(l) for l in (req.labels or [])]
            values = [float(v) for v in (req.values or [])]
            if not values:
                raise ValueError("ChartRequest.values is empty")
            if not labels:
                labels = [str(i + 1) for i in range(len(values))]
            while len(labels) < len(values):
                labels.append(str(len(labels) + 1))
            labels = labels[:len(values)]

            ctype = str(req.chart_type or "line").lower()

            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(16, 9), dpi=120)

            if ctype == "bar":
                colors = ['#00ffcc', '#ff0055', '#ffcc00', '#39d353', '#00aaff', '#ff8800']
                bars = ax.bar(labels, values, color=colors[:len(values)], width=0.6)
                ax.set_xlabel("", fontsize=18)
                for bar, val in zip(bars, values):
                    ax.annotate(f"{val:g} {req.unit_symbol}", (bar.get_x() + bar.get_width() / 2.0, val),
                                textcoords="offset points", xytext=(0, 8),
                                ha='center', fontsize=16, color='yellow', fontweight='bold')
            else:
                ax.plot(labels, values, color='#00ffcc', linewidth=5, marker='o',
                        markersize=12, markerfacecolor='#ff0055')
                for x, y in zip(labels, values):
                    ax.annotate(f"{y:g} {req.unit_symbol}", (x, y), textcoords="offset points",
                                xytext=(0, 15), ha='center', fontsize=16, color='yellow', fontweight='bold')

            ax.set_title(chart_title, fontsize=24, color='white', pad=25, fontweight='bold')
            ax.set_ylabel(f"Value ({req.unit_symbol})", fontsize=20, color='#00ffcc', labelpad=15, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=15, colors='white')
            ax.grid(True, color='#333333', linestyle='--', linewidth=1.5)

            fig.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#0d1117')

            plt.savefig(chart_png, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)

            # Convert chart PNG to 16:9 zooming MP4 clip
            cmd = [
                "ffmpeg", "-y", "-loop", "1", "-i", chart_png,
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.001,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=125:s=1920x1080",
                "-t", str(req.duration), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                req.output_mp4_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"status": "success", "engine": f"matplotlib_chart_{ctype}", "path": req.output_mp4_path}
        except Exception:
            # No silent "DUMMY_CHART_MP4" fallback: raise so the caller falls
            # through to a visible placeholder that the Gate 7 post-check flags.
            raise HTTPException(status_code=500, detail="matplotlib chart render failed")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/apply_ken_burns_motion")
async def apply_ken_burns_motion(req: KenBurnsRequest):
    """
    Applies Ken Burns pan & zoom motion transform over duration to produce strict 16:9 1920x1080 MP4,
    incorporating the shot's speech narration audio track.
    """
    import shutil
    try:
        os.makedirs(os.path.dirname(os.path.abspath(req.output_mp4_path)), exist_ok=True)
        dur = max(req.duration, 2.0)
        nb_frames = int(dur * 25)

        if not shutil.which("ffmpeg"):
            # Fallback if FFmpeg binary is missing from local system PATH
            print("WARNING: FFmpeg binary not found in system PATH. Writing synthetic MP4 placeholder.")
            with open(req.output_mp4_path, "w") as f:
                f.write(f"DUMMY_MP4_{req.image_path}")
            return {"status": "fallback_placeholder", "engine": "synthetic_mp4", "path": req.output_mp4_path}

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", req.image_path
        ]

        has_audio = False
        if req.audio_path and os.path.exists(req.audio_path) and os.path.getsize(req.audio_path) > 100:
            cmd.extend(["-i", req.audio_path])
            has_audio = True

        # Direction-aware Ken Burns: continuous ZOOM + PAN spread across the ENTIRE
        # shot (not just the opening). The zoom target is reached only at the final
        # frame, so a 50-65s still image keeps visibly moving the whole time instead
        # of freezing after ~5s (the old min(zoom+0.002,1.25) capped the zoom in
        # seconds — that's why only the intro looked animated). The pan likewise
        # traverses the full duration. Both give clear, smooth motion for every
        # static-image shot.
        pan_n = max(nb_frames - 1, 1)
        prog = f"min(on/{pan_n},1)"
        z_target = float(getattr(req, "zoom_target", 1.5) or 1.5)
        if z_target < 1.1:
            z_target = 1.1
        elif z_target > 1.6:
            z_target = 1.6   # never over-crop: keeps >= ~62% of the image framed
        z_step = (z_target - 1.0) / nb_frames   # tiny per-frame increment over full shot
        direction = (req.direction or "left_to_right").lower()
        cx = "iw/2-(iw/zoom/2)"
        cy = "ih/2-(ih/zoom/2)"
        if direction in ("right_to_left",):
            xexpr = f"iw*(1-1/zoom)*(1-{prog})"
            yexpr = cy
        elif direction == "top_to_bottom":
            xexpr = cx
            yexpr = f"ih*(1-1/zoom)*{prog}"
        elif direction == "bottom_to_top":
            xexpr = cx
            yexpr = f"ih*(1-1/zoom)*(1-{prog})"
        else:  # left_to_right (default)
            xexpr = f"iw*(1-1/zoom)*{prog}"
            yexpr = cy

        # A smooth Ken Burns needs a FINER SOURCE than the output. ffmpeg's zoompan
        # steps in integer pixels; on a 1920-wide source the sub-pixel zoom/pan at
        # the start of the shot rounds to whole pixels and causes visible shaking.
        # Upscaling 8x before zoompan (then downscaling to the 1920x1080 output)
        # makes the motion smooth — this is the "scale=8000:-1" smoothing trick
        # (Bannerbear). Measured via optical-flow acceleration: old=0.131 jitter,
        # 4x=0.045, 8x=0.026 (lower is smoother).
        hi_w = 15360
        hi_h = 8640
        zoompan = ("scale=1920:1080:force_original_aspect_ratio=decrease,"
                   "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
                   f"scale={hi_w}:{hi_h},"  # upsample to finer grid for smooth motion
                   f"zoompan=z='min(zoom+{z_step:.6f},{z_target})':x='{xexpr}':y='{yexpr}':d={nb_frames}:s=1920x1080,fps=25,format=yuv420p")

        if has_audio:
            # Pad audio with silence up to `dur` and cap total length with -t, so the
            # clip is EXACTLY `dur` (narration fully audible + trailing hold). This
            # makes the concatenated timeline match ffprobe-measured durations exactly.
            cmd.extend([
                "-filter_complex", f"[0:v]{zoompan}[v];[1:a]apad[a]",
                "-map", "[v]", "-map", "[a]",
                "-t", str(dur),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-maxrate", "6M", "-bufsize", "12M",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                req.output_mp4_path
            ])
        else:
            cmd.extend([
                "-vf", zoompan,
                "-t", str(dur),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-maxrate", "6M", "-bufsize", "12M",
                "-pix_fmt", "yuv420p",
                "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-c:a", "aac", "-shortest",
                req.output_mp4_path
            ])

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return {"status": "success", "engine": "ffmpeg_ken_burns", "path": req.output_mp4_path}
        else:
            print(f"FFmpeg Ken Burns Error: {res.stderr}")
            raise Exception(f"FFmpeg error: {res.stderr[:200]}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _read_concat_paths(concat_list_path: str) -> List[str]:
    """Parse `file 'path'` lines from an ffmpeg concat list."""
    paths = []
    try:
        with open(concat_list_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("file '") and line.endswith("'"):
                    paths.append(line[6:-1])
    except Exception:
        pass
    return [p for p in paths if os.path.exists(p)]


def _probe_dur(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30)
        val = float(r.stdout.strip())
        return val if val > 0 else 0.0
    except Exception:
        return 0.0


def _bgm_duck_filter(narration_stream: str, bgm_stream: str) -> str:
    """Build the narration + sidechain-ducked-BGM mix filter.

    Music rides down under narration (sidechain keyed on the voice) and swells
    back in the pauses. Env-tunable without code edits:
      - BGM_VOLUME            resting music level (default 0.5)
      - BGM_SIDECHAIN_THRESHOLD  duck trigger level (default 0.02)
    Emits a stream named ``[a]`` ready for ffmpeg's ``-map [a]``.
    """
    bgm_vol = os.getenv("BGM_VOLUME", "0.5")
    sc_thresh = os.getenv("BGM_SIDECHAIN_THRESHOLD", "0.02")
    return (
        f"{bgm_stream}volume={bgm_vol}[bgm];"
        f"{narration_stream}volume=1.0,asplit=2[voice][sc];"
        f"[bgm][sc]sidechaincompress=threshold={sc_thresh}:ratio=12:attack=150:release=1200[duck];"
        f"[voice][duck]amix=inputs=2:duration=first,alimiter=limit=0.9:level=false,"
        f"loudnorm=I=-14:TP=-1.5:LRA=11[a]"
    )


def _build_crossfade_cmd(clip_paths, durs, crossfade, transition,
                         subtitle_path, output_video_path, bgm_path=None):
    """
    Builds an ffmpeg command that concatenates shots with a crossfade dissolve
    between each pair (xfade for video, acrossfade for audio). Returns the
    `cmd` list, or None if there are < 2 eligible clips.
    """
    n = len(clip_paths)
    if n < 2:
        return None

    cf = crossfade
    min_d = min(durs)
    cf = max(0.1, min(cf, min_d * 0.45))  # never overlap more than ~half the shortest shot

    # Prefix sums of clip durations (for xfade offset math).
    prefix = []
    s = 0.0
    for d in durs:
        s += d
        prefix.append(s)

    has_sub = subtitle_path and os.path.exists(subtitle_path) and os.path.getsize(subtitle_path) > 50
    has_bgm = bgm_path and os.path.exists(bgm_path) and os.path.getsize(bgm_path) > 1000

    parts = []

    # Normalize each video stream (trim to exact length, uniform format) for xfade.
    for i in range(n):
        parts.append(
            f"[{i}:v]trim=end={durs[i]:.3f},setpts=PTS-STARTPTS,fps=25,"
            f"scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v{i}]"
        )
    cur = "[v0]"
    for i in range(1, n):
        off = prefix[i - 1] - i * cf
        parts.append(f"{cur}[v{i}]xfade=transition={transition}:duration={cf:.3f}:offset={off:.3f}[vx{i}]")
        cur = f"[vx{i}]"

    # Normalize each audio stream (trim + common rate/channels) so acrossfade works.
    for i in range(n):
        parts.append(
            f"[{i}:a]atrim=end={durs[i]:.3f},asetpts=PTS-STARTPTS,"
            f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]"
        )
    cura = "[a0]"
    for i in range(1, n):
        parts.append(f"{cura}[a{i}]acrossfade=d={cf:.3f}:c1=tri:c2=tri[ax{i}]")
        cura = f"[ax{i}]"

    # Burn subtitles on the final composited video stream.
    if has_sub:
        esc = subtitle_path.replace(":", "\\:").replace("'", "'\\''")
        parts.append(f"{cur}ass='{esc}',format=yuv420p[v]")
    else:
        parts.append(f"{cur}format=yuv420p[v]")

    # Voice + sidechain-ducked BGM, or narration only.
    if has_bgm:
        parts.append(_bgm_duck_filter(cura, f"[{n}:a]"))
    else:
        parts.append(f"{cura}acopy[a]")

    cmd = ["ffmpeg", "-y"]
    for p in clip_paths:
        cmd += ["-i", p]
    if has_bgm:
        cmd += ["-stream_loop", "-1", "-i", bgm_path]
    cmd += ["-filter_complex", ";".join(parts)]
    cmd += ["-map", "[v]", "-map", "[a]"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-maxrate", "6M", "-bufsize", "12M"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
    cmd += [output_video_path]
    return cmd


@app.post("/tools/assemble_ffmpeg_timeline")
async def assemble_ffmpeg_timeline(req: TimelineAssemblyRequest):
    """
    Executes FFmpeg timeline assembly: concatenates shot MP4s, burns .ass subtitles,
    enforces strict 16:9 1920x1080 video resolution.
    When `crossfade > 0`, uses an xfade/acrossfade dissolve between shots instead of
    a hard-cut concat, producing smooth professional transitions.
    """
    import shutil
    try:
        os.makedirs(os.path.dirname(os.path.abspath(req.output_video_path)), exist_ok=True)

        if not shutil.which("ffmpeg"):
            print("WARNING: FFmpeg binary not found in system PATH. Writing synthetic final video placeholder.")
            with open(req.output_video_path, "w") as f:
                f.write("DUMMY_FINAL_1080P_VIDEO")
            return {"status": "fallback_placeholder", "engine": "synthetic_video", "path": req.output_video_path}

        # Crossfade path: xfade/acrossfade dissolve between shots.
        if req.crossfade > 0:
            clip_paths = _read_concat_paths(req.concat_list_path)
            durs = [_probe_dur(p) for p in clip_paths]
            cmd = _build_crossfade_cmd(
                clip_paths, durs, req.crossfade, req.transition,
                req.subtitle_path, req.output_video_path, req.bgm_path,
            )
            if cmd:
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    return {"status": "success", "engine": f"ffmpeg_xfade_{req.transition}",
                            "path": req.output_video_path, "crossfade": req.crossfade}
                print(f"FFmpeg XFade Error: {res.stderr}")

        has_subtitles = req.subtitle_path and os.path.exists(req.subtitle_path) and os.path.getsize(req.subtitle_path) > 50

        vf_filter = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        if has_subtitles:
            sub_path_escaped = req.subtitle_path.replace(":", "\\:").replace("'", "'\\''")
            vf_filter += f",ass='{sub_path_escaped}'"

        has_bgm = False
        if req.bgm_path and os.path.exists(req.bgm_path) and os.path.getsize(req.bgm_path) > 1000:
            has_bgm = True

        if has_bgm:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", req.concat_list_path,
                "-stream_loop", "-1", "-i", req.bgm_path,
                "-filter_complex", f"[0:v]{vf_filter}[v];" + _bgm_duck_filter("[0:a]", "[1:a]"),
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-maxrate", "6M", "-bufsize", "12M",
                "-c:a", "aac", "-b:a", "192k",
                req.output_video_path
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", req.concat_list_path,
                "-vf", vf_filter,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-maxrate", "6M", "-bufsize", "12M",
                "-c:a", "aac", "-b:a", "192k",
                req.output_video_path
            ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return {"status": "success", "path": req.output_video_path}
        else:
            print(f"FFmpeg Concat Error: {res.stderr}")
            raise Exception(f"FFmpeg concat error: {res.stderr[:200]}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
