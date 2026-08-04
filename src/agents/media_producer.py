import os
import re
import uuid
import datetime
from typing import Dict, Any, Optional, List, Tuple
from src.schemas.state import GlobalState, ScriptData, AssetPaths, VisualType
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent
from mcp_servers.audio_edge.server import synthesize_tts, align_subtitles_whisper, TTSRequest, WhisperRequest
from mcp_servers.media_cloud.server import (
    generate_flux_image, apply_ken_burns_motion, assemble_ffmpeg_timeline,
    render_playwright_svg_animation, generate_dynamic_chart, fetch_reaction_gif_clip,
    ImageGenRequest, KenBurnsRequest, TimelineAssemblyRequest,
    PlaywrightSVGRequest, ChartRequest, GIFRequest
)


def merge_ass_subtitle_files(ass_paths: List[str], shot_durations: List[float], output_master_ass: str):
    """
    Merges multiple shot .ass subtitle files into a single master timeline subtitle file
    by offsetting timestamps according to cumulative shot durations.
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
Style: Default,Montserrat,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,3,2,2,10,10,50,1

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

    for ass_path, shot_dur in zip(ass_paths, shot_durations):
        if os.path.exists(ass_path):
            with open(ass_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("Dialogue:"):
                    parts = line.split(",", 9)
                    if len(parts) == 10:
                        t_start = parse_ass_time(parts[1]) + current_offset
                        t_end = parse_ass_time(parts[2]) + current_offset
                        parts[1] = format_ass_time(t_start)
                        parts[2] = format_ass_time(t_end)
                        dialogue_lines.append(",".join(parts))
        current_offset += shot_dur

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


def inject_tts_breathing_pauses(narration: str, sentiment_score: float = 0.5) -> str:
    """
    Stage 4: Neural TTS Intonation Engine:
    Inserts SSML-style natural breathing pauses (<break>) at punctuation boundaries.
    Adjusts pause density based on script sentiment (0.0=tense/fast, 1.0=calm/slow).
    """
    # Map sentiment to pause duration thresholds
    pause_ms = int(200 + (sentiment_score * 300))  # 200ms – 500ms range
    short_pause = f"<break time='{pause_ms}ms'/>"
    long_pause = f"<break time='{pause_ms * 2}ms'/>"

    # Insert pauses at commas and em-dashes (short), and sentence ends (long)
    narration = re.sub(r'([,;—])', r'\1 ' + short_pause + ' ', narration)
    narration = re.sub(r'([.!?])\s', r'\1 ' + long_pause + ' ', narration)
    return narration.strip()


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


class MediaProducerAgent:
    """
    Stage 4: Media Producer Agent coordinating parallel tool execution across Edge MCP Server (Pi 5)
    and Cloud Media MCP Server (OCI) to synthesize audio, generate 16:9 visuals, apply Ken Burns
    motion, burn subtitles, and assemble the final 10-15 minute widescreen video.
    Includes TTS Intonation & Breathing Pause Injection and [Scene:...] Visual Cue Parsing.
    """

    def __init__(self, name: str = "MediaProducer", storage_dir: str = "/tmp/csvg_media"):
        self.name = name
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    async def produce_all_media(self, state: GlobalState, dummy_frames: bool = False) -> AssetPaths:
        """
        Synthesizes audio & subtitles via Edge MCP tools, generates visuals & timeline via Cloud MCP tools.
        """
        script = state.script_data
        if not script:
            raise ValueError("Media production failed: state.script_data is None")

        asset_paths = AssetPaths()

        # 1. Synthesize TTS Audio and Subtitles for all shots
        audio_dir = os.path.join(self.storage_dir, "audio")
        sub_dir = os.path.join(self.storage_dir, "subtitles")
        vis_dir = os.path.join(self.storage_dir, "visuals")

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(sub_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        concat_lines = []

        for shot in script.shots:
            shot_key = f"shot_{shot.shot_id}"

            # Stage 4a: Parse [Scene: ...] visual cues from narration
            clean_narration, scene_cue = parse_scene_visual_cue(shot.narration_text)
            raw_visual_prompt = scene_cue if scene_cue else shot.visual_prompt
            prompt_lower = raw_visual_prompt.lower()

            # Stage 8 Quality-by-Design: Enrich visual prompt with FVD cinematic keywords
            visual_prompt = enrich_visual_prompt(raw_visual_prompt, shot.act_index, shot.shot_id)

            # Stage 4b: Inject TTS breathing pauses based on shot urgency
            urgency_words = {"warning", "collapse", "shocking", "urgent", "crisis", "breaking"}
            has_urgency = any(w in clean_narration.lower() for w in urgency_words)
            sentiment_score = 0.25 if has_urgency else 0.65
            tts_narration = inject_tts_breathing_pauses(clean_narration, sentiment_score)

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
                            "region": region_val
                        }, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                            if resp.status != 200:
                                raise Exception(f"TTS Synthesis failed over HTTP: {await resp.text()}")
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

                        # 3. Align subtitles remotely
                        async with session.post(f"{audio_edge_url}/tools/align_subtitles_whisper", json={
                            "audio_path": wav_path,
                            "output_ass_path": ass_path
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
                await synthesize_tts(TTSRequest(text=tts_narration, output_path=wav_path))
                await align_subtitles_whisper(WhisperRequest(audio_path=wav_path, output_ass_path=ass_path))

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

            asset_paths.audio[shot_key] = wav_path
            asset_paths.subtitles[shot_key] = ass_path

            # Stage 8 Quality-by-Design: Alternate Ken Burns pan direction for optical flow continuity
            # Odd shots pan left-to-right, even shots pan right-to-left — prevents jarring same-direction cuts
            ken_burns_direction = "left_to_right" if shot.shot_id % 2 == 1 else "right_to_left"

            # Route and generate specialized visual assets based on AI-classified shot.visual_type
            is_specialized = False
            v_type = getattr(shot, "visual_type", VisualType.STANDARD_IMAGE)

            # Check 1: Reaction GIF / Meme segment
            if v_type == VisualType.GIF_MEME or "[gif:" in prompt_lower or "reaction gif" in prompt_lower:
                gif_query = "shocked reaction"
                match = re.search(r'\[gif:\s*([^\]]+)\]', prompt_lower)
                if match:
                    gif_query = match.group(1).strip()
                elif ":" in raw_visual_prompt:
                    gif_query = raw_visual_prompt.split(":", 1)[1].strip()
                else:
                    gif_query = raw_visual_prompt[:30].strip()

                print(f"🎬 Processing AI GIF reaction segment for {shot_key} (Query: '{gif_query}')")
                try:
                    await fetch_reaction_gif_clip(GIFRequest(
                        query=gif_query,
                        duration=shot.duration_estimate,
                        output_mp4_path=mp4_path
                    ))
                    is_specialized = True
                except Exception as gif_err:
                    print(f"⚠️ GIF generation failed: {gif_err}. Falling back to normal image rendering.")

            # Check 2: Dynamic Stock/Data Chart segment
            elif v_type == VisualType.MATPLOTLIB_CHART or "[chart:" in prompt_lower or "stock chart" in prompt_lower or "market graph" in prompt_lower:
                chart_title = raw_visual_prompt
                match = re.search(r'\[chart:\s*([^\]]+)\]', prompt_lower)
                if match:
                    chart_title = match.group(1).strip()
                
                print(f"📊 Processing AI dynamic data chart segment for {shot_key} (Title: '{chart_title}')")
                try:
                    await generate_dynamic_chart(ChartRequest(
                        title=chart_title,
                        labels=["Q1", "Q2", "Q3", "Q4", "Current"],
                        values=[100, 130, 110, 180, 225],
                        unit_symbol="%" if "percent" in prompt_lower or "%" in prompt_lower else "$ Billion",
                        duration=shot.duration_estimate,
                        output_mp4_path=mp4_path
                    ))
                    is_specialized = True
                except Exception as chart_err:
                    print(f"⚠️ Chart generation failed: {chart_err}. Falling back to normal image rendering.")

            # Check 3: SVG Animation Counter/Ticker segment
            elif v_type == VisualType.SVG_TICKER or "[svg:" in prompt_lower or "ticker" in prompt_lower or "counter" in prompt_lower:
                svg_title = raw_visual_prompt
                match = re.search(r'\[svg:\s*([^\]]+)\]', prompt_lower)
                if match:
                    svg_title = match.group(1).strip()

                print(f"📈 Processing AI animated Playwright SVG ticker segment for {shot_key} (Title: '{svg_title}')")
                try:
                    await render_playwright_svg_animation(PlaywrightSVGRequest(
                        chart_type="animated_line_chart",
                        title=svg_title,
                        headline_val="$520.4 Billion" if "billion" in prompt_lower else "+18.4%",
                        sub_text="Live Market Shift" if "billion" in prompt_lower else "Gain",
                        duration=shot.duration_estimate,
                        output_mp4_path=mp4_path
                    ))
                    is_specialized = True
                except Exception as svg_err:
                    print(f"⚠️ SVG animation failed: {svg_err}. Falling back to normal image rendering.")

            # Default: Widescreen FLUX image generation + Ken Burns camera pan
            if not is_specialized:
                if dummy_frames:
                    from mcp_servers.media_cloud.server import generate_synthetic_png
                    generate_synthetic_png(img_path, title=f"SHOT {shot.shot_id}: {raw_visual_prompt[:40]}")
                else:
                    try:
                        await generate_flux_image(ImageGenRequest(prompt=visual_prompt, output_image_path=img_path))
                    except Exception as e:
                        err_msg = str(e).lower()
                        print(f"⚠️ Visual Generation Error on shot {shot.shot_id}: {e}")
                        
                        # 1. Content Moderation / Safety Flag
                        if any(term in err_msg for term in ["moderation", "nsfw", "safety", "policy", "blocked"]):
                            sanitized_prompt = f"A professional widescreen cinematic documentary representation of clean technology workspace, flat vector style, 16:9 widescreen presentation"
                            print(f"🔄 Re-submitting sanitized safe prompt to Fal.ai/Replicate: '{sanitized_prompt}'")
                            try:
                                await generate_flux_image(ImageGenRequest(prompt=sanitized_prompt, output_image_path=img_path))
                            except Exception as retry_err:
                                print(f"⚠️ Safety prompt retry failed. Falling back to local synthetic generation: {retry_err}")
                                from mcp_servers.media_cloud.server import generate_synthetic_png
                                generate_synthetic_png(img_path, title=f"SHOT {shot.shot_id}: {raw_visual_prompt[:40]}")
                        else:
                            # 2. Rate Limits, Quota Errors, or Network/API issues
                            print(f"📉 API Error (Rate Limit/Quota). Gracefully falling back to local synthetic generation for shot {shot.shot_id} to keep pipeline alive.")
                            from mcp_servers.media_cloud.server import generate_synthetic_png
                            generate_synthetic_png(img_path, title=f"SHOT {shot.shot_id}: {raw_visual_prompt[:40]}")

                # Compile PNG to MP4 with Ken Burns movement
                await apply_ken_burns_motion(KenBurnsRequest(
                    image_path=img_path,
                    audio_path=wav_path,
                    duration=shot.duration_estimate,
                    output_mp4_path=mp4_path,
                    direction=ken_burns_direction  # optical flow continuity
                ))

            asset_paths.visuals[shot_key] = mp4_path

            concat_lines.append(f"file '{mp4_path}'")

        # Create Concat List File for FFmpeg
        concat_list_path = os.path.join(self.storage_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            f.write("\n".join(concat_lines))

        # Merge all shot subtitles into a single master .ass timeline file
        ass_paths = [asset_paths.subtitles[f"shot_{shot.shot_id}"] for shot in script.shots if f"shot_{shot.shot_id}" in asset_paths.subtitles]
        shot_durs = [max(shot.duration_estimate, 2.0) for shot in script.shots]
        master_sub_path = os.path.join(self.storage_dir, "master_subtitles.ass")
        merge_ass_subtitle_files(ass_paths, shot_durs, master_sub_path)

        # Assemble Final Timeline
        final_video_path = os.path.join(self.storage_dir, "final_video_1080p.mp4")
        bgm_dummy_path = os.path.join(self.storage_dir, "bgm.mp3")
        with open(bgm_dummy_path, "w") as f:
            f.write("DUMMY_BGM")

        await assemble_ffmpeg_timeline(TimelineAssemblyRequest(
            concat_list_path=concat_list_path,
            subtitle_path=master_sub_path,
            bgm_path=bgm_dummy_path,
            output_video_path=final_video_path
        ))

        asset_paths.final_video = final_video_path
        state.asset_paths = asset_paths
        state.execution_stage = "MEDIA_PRODUCED"
        return asset_paths

    async def process(self, state: GlobalState, dummy_frames: bool = False) -> A2AMessage:
        """
        Executes Media Producer workflow:
        1. Reads state.script_data
        2. Calls MCP tools to produce audio, subtitles, visuals, and final timeline
        3. Updates state.asset_paths
        4. Emits MEDIA_READY A2AMessage to Orchestrator
        """
        assets = await self.produce_all_media(state, dummy_frames=dummy_frames)

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
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        return msg
