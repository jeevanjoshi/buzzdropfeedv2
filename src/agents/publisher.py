import uuid
import datetime
from typing import Dict, Any, Optional
from src.schemas.state import GlobalState, UploadMetadata
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent
from mcp_servers.youtube_cloud.server import check_quota_available, upload_youtube_resumable, QuotaCheckRequest, UploadRequest


class PublisherAgent:
    """
    Publisher Agent managing YouTube MCP upload tools, checking daily API quotas,
    injecting synthetic content metadata tags, and updating GlobalState.
    """

    def __init__(self, name: str = "Publisher"):
        self.name = name

    async def publish_video(self, state: GlobalState, daily_uploads: int = 0) -> UploadMetadata:
        """
        Interfaces with YouTube Publishing MCP Server to check quota and upload video.
        """
        if not state.asset_paths.final_video:
            raise ValueError("Publishing failed: state.asset_paths.final_video is None")

        # 1. Quota Check
        quota_res = await check_quota_available(QuotaCheckRequest(current_daily_uploads=daily_uploads))
        if not quota_res["is_safe"]:
            raise RuntimeError(f"YouTube daily API quota limit reached ({quota_res['used_units']} / {quota_res['quota_limit']} units used)")

        # 2. Upload Video
        title = state.script_data.title if state.script_data else state.selected_topic.headline
        desc = (
            f"Deep-dive financial storytelling breakdown on: {title}.\n\n"
            f"Sources & Facts: {state.selected_topic.source_url if state.selected_topic else 'Verified Financial News Feeds'}\n\n"
            f"Disclaimer: AI-synthesized visualization for educational & infotainment storytelling."
        )

        upload_res = await upload_youtube_resumable(UploadRequest(
            video_path=state.asset_paths.final_video,
            title=title,
            description=desc,
            tags=state.selected_topic.keywords if state.selected_topic else ["finance", "tech"]
        ))

        meta = UploadMetadata(
            video_id=upload_res.get("video_id", "demo_id"),
            status="PUBLISHED",
            retry_count=0,
            synthetic_content_flag=True
        )
        state.upload_metadata = meta
        state.execution_stage = "PUBLISHED_SUCCESS"
        return meta

    async def process(self, state: GlobalState, daily_uploads: int = 0) -> A2AMessage:
        """
        Executes Publisher Agent workflow:
        1. Reads state.asset_paths.final_video
        2. Calls YouTube MCP server tools
        3. Updates state.upload_metadata
        4. Emits PUBLISHED_SUCCESS A2AMessage
        """
        meta = await self.publish_video(state, daily_uploads=daily_uploads)

        msg = A2AMessage(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            sender=AgentRole.PUBLISHER,
            target=AgentRole.ORCHESTRATOR,
            intent=AgentIntent.PUBLISHED_SUCCESS,
            payload={
                "status": "SUCCESS",
                "video_id": meta.video_id,
                "youtube_url": f"https://www.youtube.com/watch?v={meta.video_id}",
                "synthetic_content": True
            },
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        return msg
