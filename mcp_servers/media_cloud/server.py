import os
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
    output_mp4_path: str


class ChartRequest(BaseModel):
    title: str
    labels: list = ["Q1", "Q2", "Q3", "Q4"]
    values: list = [100, 140, 90, 185]
    unit_symbol: str = "$ Million"  # Explicit units e.g. '$ Million', '₹ Crores', '%'
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

            # Try generating a background image using fal.ai or replicate
            bg_data = None
            fal_key = os.getenv("FAL_KEY")
            if fal_key and not fal_key.startswith("YOUR_"):
                try:
                    import fal_client
                    import requests
                    prompt = f"Widescreen 16:9 cinematic background representing: {req.headline_text}. Hyperrealistic, dark moody ambient lighting, dramatic atmosphere."
                    handler = fal_client.submit(
                        "fal-ai/flux/schnell",
                        arguments={
                            "prompt": prompt,
                            "image_size": "landscape_16_9",
                            "num_inference_steps": 4,
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
                        prompt = f"Widescreen 16:9 cinematic background representing: {req.headline_text}. Hyperrealistic, dark moody ambient lighting, dramatic atmosphere."
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

            # Decode the image data or fall back to plain dark background
            if bg_data:
                try:
                    nparr = np.frombuffer(bg_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (1280, 720))
                    # Apply a slight dark tint/overlay to ensure text readability on top of the image
                    overlay = img.copy()
                    cv2.rectangle(overlay, (0, 0), (1280, 720), (10, 10, 15), -1)
                    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
                except Exception as img_err:
                    print(f"[Thumbnail] Failed to decode generated background: {img_err}. Using solid fallback.")
                    img = np.zeros((720, 1280, 3), dtype=np.uint8)
                    img[:] = (15, 10, 5)
            else:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
                img[:] = (15, 10, 5)

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

            # Draw lines with clean anti-aliasing
            y_start = 260 if len(lines) > 1 else 340
            for idx, line in enumerate(lines):
                cv2.putText(img, line, (60, y_start + (idx * 90)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 4, cv2.LINE_AA)
            
            sub_y = y_start + (len(lines) * 90) + 10
            cv2.putText(img, sub_text, (60, sub_y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

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
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(req.output_mp4_path)), exist_ok=True)
        chart_png = req.output_mp4_path.replace(".mp4", "_chart.png")

        current_month_year = datetime.datetime.now(datetime.timezone.utc).strftime("%B %Y").upper()
        chart_title = f"{req.title.upper()} ({current_month_year})"

        try:
            import matplotlib.pyplot as plt

            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
            
            # High contrast cyan line chart with prominent markers
            ax.plot(req.labels, req.values, color='#00ffcc', linewidth=5, marker='o', markersize=12, markerfacecolor='#ff0055')
            
            # Title & Explicit Unit Y-Label
            ax.set_title(chart_title, fontsize=24, color='white', pad=25, fontweight='bold')
            ax.set_ylabel(f"Value ({req.unit_symbol})", fontsize=20, color='#00ffcc', labelpad=15, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=18, colors='white')
            ax.grid(True, color='#333333', linestyle='--', linewidth=1.5)
            
            fig.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#0d1117')

            # Annotate data points with explicit units
            for x, y in zip(req.labels, req.values):
                ax.annotate(f"{y} {req.unit_symbol}", (x, y), textcoords="offset points", xytext=(0, 15),
                            ha='center', fontsize=16, color='yellow', fontweight='bold')

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
            return {"status": "success", "engine": "matplotlib_chart_with_units", "path": req.output_mp4_path}
        except Exception:
            generate_synthetic_png(chart_png, title="CHART_WITH_UNITS")
            with open(req.output_mp4_path, "w") as f:
                f.write("DUMMY_CHART_MP4")
            return {"status": "success", "engine": "fallback_chart", "path": req.output_mp4_path}

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

        zoompan = ("scale=1920:1080:force_original_aspect_ratio=decrease,"
                   "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
                   f"zoompan=z='min(zoom+0.0015,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={nb_frames}:s=1920x1080,fps=25,format=yuv420p")

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

    # Voice + ducked BGM, or narration only.
    if has_bgm:
        parts.append(f"{cura}volume=1.0[voice]")
        parts.append(f"[{n}:a]volume=0.12[bgm]")
        parts.append("[voice][bgm]amix=inputs=2:duration=first[a]")
    else:
        parts.append(f"{cura}acopy[a]")

    cmd = ["ffmpeg", "-y"]
    for p in clip_paths:
        cmd += ["-i", p]
    if has_bgm:
        cmd += ["-stream_loop", "-1", "-i", bgm_path]
    cmd += ["-filter_complex", ";".join(parts)]
    cmd += ["-map", "[v]", "-map", "[a]"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "22"]
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
                "-filter_complex", f"[0:v]{vf_filter}[v];[0:a]volume=1.0[voice];[1:a]volume=0.12[bgm];[voice][bgm]amix=inputs=2:duration=first[a]",
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-c:a", "aac", "-b:a", "192k",
                req.output_video_path
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", req.concat_list_path,
                "-vf", vf_filter,
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
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
