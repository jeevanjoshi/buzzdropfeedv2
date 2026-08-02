import os
import uuid
import datetime
from typing import Dict, Any, Optional, List
from src.schemas.state import GlobalState, ScriptData, AssetPaths
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent
from mcp_servers.audio_edge.server import synthesize_tts, align_subtitles_whisper, TTSRequest, WhisperRequest
from mcp_servers.media_cloud.server import (
    generate_flux_image, apply_ken_burns_motion, assemble_ffmpeg_timeline,
    ImageGenRequest, KenBurnsRequest, TimelineAssemblyRequest
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


class MediaProducerAgent:
    """
    Media Producer Agent coordinating parallel tool execution across Edge MCP Server (Pi 5)
    and Cloud Media MCP Server (OCI) to synthesize audio, generate 16:9 visuals, apply Ken Burns
    motion, burn subtitles, and assemble the final 10-15 minute widescreen video.
    """

    def __init__(self, name: str = "MediaProducer", storage_dir: str = "/tmp/csvg_media"):
        self.name = name
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    async def produce_all_media(self, state: GlobalState) -> AssetPaths:
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
            
            # Paths
            wav_path = os.path.join(audio_dir, f"{shot_key}.wav")
            ass_path = os.path.join(sub_dir, f"{shot_key}.ass")
            img_path = os.path.join(vis_dir, f"{shot_key}.png")
            mp4_path = os.path.join(vis_dir, f"{shot_key}.mp4")

            # Call Edge Audio MCP Tools (with HTTP / REST Bridge support)
            audio_edge_url = os.getenv("AUDIO_EDGE_URL")
            if audio_edge_url:
                import aiohttp
                region_val = state.selected_topic.region if (state.selected_topic and hasattr(state.selected_topic, 'region')) else "all"
                async with aiohttp.ClientSession() as session:
                    # 1. Synthesize TTS remotely
                    async with session.post(f"{audio_edge_url}/tools/synthesize_tts", json={
                        "text": shot.narration_text,
                        "output_path": wav_path,
                        "region": region_val
                    }) as resp:
                        if resp.status != 200:
                            raise Exception(f"TTS Synthesis failed over HTTP: {await resp.text()}")
                    # 2. Download the synthesized wav file
                    async with session.get(f"{audio_edge_url}/files", params={"path": wav_path}) as file_resp:
                        if file_resp.status == 200:
                            with open(wav_path, "wb") as f:
                                f.write(await file_resp.read())
                        else:
                            raise Exception(f"Failed to download wav file: {await file_resp.text()}")

                    # 3. Align subtitles remotely
                    async with session.post(f"{audio_edge_url}/tools/align_subtitles_whisper", json={
                        "audio_path": wav_path,
                        "output_ass_path": ass_path
                    }) as resp:
                        if resp.status != 200:
                            raise Exception(f"Subtitle alignment failed over HTTP: {await resp.text()}")
                    # 4. Download the aligned subtitles (.ass)
                    async with session.get(f"{audio_edge_url}/files", params={"path": ass_path}) as file_resp:
                        if file_resp.status == 200:
                            with open(ass_path, "wb") as f:
                                f.write(await file_resp.read())
                        else:
                            raise Exception(f"Failed to download ass file: {await file_resp.text()}")
            else:
                await synthesize_tts(TTSRequest(text=shot.narration_text, output_path=wav_path))
                await align_subtitles_whisper(WhisperRequest(audio_path=wav_path, output_ass_path=ass_path))

            asset_paths.audio[shot_key] = wav_path
            asset_paths.subtitles[shot_key] = ass_path

            # Call Cloud Media MCP Tools
            await generate_flux_image(ImageGenRequest(prompt=shot.visual_prompt, output_image_path=img_path))
            await apply_ken_burns_motion(KenBurnsRequest(image_path=img_path, audio_path=wav_path, duration=shot.duration_estimate, output_mp4_path=mp4_path))
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

    async def process(self, state: GlobalState) -> A2AMessage:
        """
        Executes Media Producer workflow:
        1. Reads state.script_data
        2. Calls MCP tools to produce audio, subtitles, visuals, and final timeline
        3. Updates state.asset_paths
        4. Emits MEDIA_READY A2AMessage to Orchestrator
        """
        assets = await self.produce_all_media(state)

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
