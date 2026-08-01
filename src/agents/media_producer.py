import os
import uuid
import datetime
from typing import Dict, Any, Optional
from src.schemas.state import GlobalState, ScriptData, AssetPaths
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent
from mcp_servers.audio_edge.server import synthesize_tts, align_subtitles_whisper, TTSRequest, WhisperRequest
from mcp_servers.media_cloud.server import (
    generate_flux_image, apply_ken_burns_motion, assemble_ffmpeg_timeline,
    ImageGenRequest, KenBurnsRequest, TimelineAssemblyRequest
)


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

            # Call Edge Audio MCP Tools
            await synthesize_tts(TTSRequest(text=shot.narration_text, output_path=wav_path))
            await align_subtitles_whisper(WhisperRequest(audio_path=wav_path, output_ass_path=ass_path))
            asset_paths.audio[shot_key] = wav_path
            asset_paths.subtitles[shot_key] = ass_path

            # Call Cloud Media MCP Tools
            await generate_flux_image(ImageGenRequest(prompt=shot.visual_prompt, output_image_path=img_path))
            await apply_ken_burns_motion(KenBurnsRequest(image_path=img_path, duration=shot.duration_estimate, output_mp4_path=mp4_path))
            asset_paths.visuals[shot_key] = mp4_path

            concat_lines.append(f"file '{mp4_path}'")

        # Create Concat List File for FFmpeg
        concat_list_path = os.path.join(self.storage_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            f.write("\n".join(concat_lines))

        # Assemble Final Timeline
        final_video_path = os.path.join(self.storage_dir, "final_video_1080p.mp4")
        master_sub_path = asset_paths.subtitles.get("shot_1", "")
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
