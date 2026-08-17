import os
from typing import Dict, Any, List
from src.schemas.state import GlobalState
from src.engine.llm_client import LLMClient
from mcp_servers.youtube_cloud.server import (
    list_comments,
    reply_comment,
    ListCommentsRequest,
    ReplyCommentRequest
)


class YouTubeEngagementEngine:
    """
    Manages automated viewer engagement on YouTube videos:
    - Queries recent viewer comments
    - Filters out existing replies / channel's own comments
    - Uses LLMClient (fact-grounded) to craft high-value, natural replies
    - Inserts replies via comments.insert to boost dwell time and engagement velocity
    """

    def __init__(self):
        self.llm = LLMClient()

    async def reply_to_viewers(self, state: GlobalState, video_id: str, max_replies: int = 3) -> int:
        """
        Fetches top comments and replies using verified facts from GlobalState.
        Returns the number of successful replies posted.
        """
        if not video_id or not state:
            return 0

        # Feature toggle (default enabled = 1)
        if os.getenv("YOUTUBE_REPLY_BOT", "1").strip().lower() in ("0", "false", "no"):
            print("[YouTubeEngagement] Reply bot disabled via YOUTUBE_REPLY_BOT=0")
            return 0

        # Don't run live API calls on demo/mock video IDs
        if video_id.startswith("demo_") or video_id == "demo_id" or video_id.startswith("hermetic-"):
            print(f"[YouTubeEngagement] Demo/hermetic video id ({video_id}); reply bot skipped.")
            return 0

        replies_posted = 0
        try:
            res = await list_comments(ListCommentsRequest(video_id=video_id, max_results=15))
            if res.get("status") != "success":
                return 0

            comments = res.get("comments", [])
            if not comments:
                print(f"[YouTubeEngagement] No viewer comments found for video {video_id}.")
                return 0

            facts_context = "\n".join([f"- {f}" for f in (state.verified_facts or [])[:5]])
            topic_headline = state.selected_topic.headline if state.selected_topic else ""

            for comment in comments:
                if replies_posted >= max_replies:
                    break

                cid = comment.get("comment_id")
                text = comment.get("text", "").strip()
                author = comment.get("author", "")

                if not cid or not text:
                    continue

                # Skip low-effort/spam or very short comments
                if len(text) < 5:
                    continue

                prompt = (
                    f"You are the creator of a high-quality educational documentary on: '{topic_headline}'.\n"
                    f"A viewer named '{author}' left this comment:\n"
                    f'"{text}"\n\n'
                    f"Verified facts from our research:\n{facts_context}\n\n"
                    f"Write a friendly, insightful, and concise reply (1-2 sentences max). "
                    f"Incorporate verified facts if relevant, or thank them for watching. "
                    f"Do NOT include hashtags or emojis overload."
                )

                try:
                    reply_text = (self.llm.generate(prompt, route="publisher", temperature=0.7) or "").strip()
                    # Strip any surrounding quotes
                    if reply_text.startswith('"') and reply_text.endswith('"'):
                        reply_text = reply_text[1:-1].strip()

                    if reply_text:
                        reply_res = await reply_comment(ReplyCommentRequest(
                            parent_comment_id=cid,
                            reply_text=reply_text
                        ))
                        if reply_res.get("status") == "success":
                            replies_posted += 1
                            print(f"[YouTubeEngagement] Replied to {author}: '{reply_text[:60]}...'")
                except Exception as err:
                    print(f"[YouTubeEngagement] Failed generating/posting reply for comment {cid}: {err}")

        except Exception as e:
            print(f"[YouTubeEngagement] Error in reply_to_viewers: {e}")

        return replies_posted


youtube_engagement = YouTubeEngagementEngine()
