import os
import logging
import aiohttp
from typing import List, Dict, Any, Optional
from src.schemas.state import GlobalState
from src.schemas.seed_distribution import CommunitySeedPackage, RedditDraft

logger = logging.getLogger("CSVG_PIPELINE")

class SeedDistributorEngine:
    """
    Intelligent Seed Traffic Distribution Engine.
    Discovers high-relevance niche subreddits and forums, generates 70/30 Value-First draft content,
    and dispatches Discord/Slack webhooks with 1-click Markdown payloads for creator approval.
    """

    def __init__(self):
        self.discord_webhook = os.getenv("DISCORD_SEED_WEBHOOK_URL")
        self.slack_webhook = os.getenv("SLACK_SEED_WEBHOOK_URL")

    def select_target_subreddits(self, state: GlobalState) -> List[str]:
        """
        Uses keywords and semantic relevance to select target subreddits.
        """
        kws = state.selected_topic.keywords if state.selected_topic and state.selected_topic.keywords else ["finance", "tech"]
        defaults = ["technology", "finance", "economics", "artificial", "wallstreetbets", "stocks"]

        matched = []
        for kw in kws:
            kw_lower = kw.lower()
            for sub in defaults:
                if kw_lower in sub or sub in kw_lower:
                    if sub not in matched:
                        matched.append(sub)

        if not matched:
            matched = ["technology", "finance"]

        return matched[:3]

    def create_seed_package(self, state: GlobalState, youtube_url: str) -> CommunitySeedPackage:
        """
        Generates 70/30 Value-First seed post drafts for Reddit, Hacker News, LinkedIn, X, and Blog embeds.
        """
        pipeline_id = state.pipeline_id or "run"
        title = state.seo_metadata.title if state.seo_metadata else (
            state.script_data.title if state.script_data else "Deep-Dive Financial Storytelling"
        )
        summary = state.selected_topic.summary if state.selected_topic else "Key market trends and technological shifts."
        subreddits = self.select_target_subreddits(state)

        reddit_drafts = []
        for sub in subreddits:
            body = (
                f"**Key Takeaways & Core Analysis:**\n\n"
                f"1. {summary[:200]}\n"
                f"2. Early signals show significant institutional interest and rapid adoption metrics across global markets.\n"
                f"3. Risk factors include regulatory uncertainty and technical scaling hurdles in the near term.\n\n"
                f"--- \n"
                f"*I put together a full visual animated video breakdown with the raw source data here if you'd like to see the charts:* [{title}]({youtube_url})"
            )
            reddit_drafts.append(RedditDraft(
                target_subreddit=sub,
                title=f"Analysis: {title}",
                body_markdown=body
            ))

        hn_post = (
            f"Show HN: {title}\n\n"
            f"We synthesized deep data sources on {summary[:150]}.\n"
            f"Watch the full visual breakdown: {youtube_url}"
        )

        blog_markdown = (
            f"# {title}\n\n"
            f"*{summary}*\n\n"
            f"[![Watch Visual Video]({state.asset_paths.thumbnail if state.asset_paths else ''})]({youtube_url})\n\n"
            f"## Deep Dive\n{summary}"
        )

        linkedin_post = (
            f"📈 **{title}**\n\n"
            f"{summary}\n\n"
            f"Read our key insights and watch the full visual narrative breakdown here: {youtube_url}\n\n"
            f"#finance #business #tech #economics"
        )

        x_thread = [
            f"🧵 Analysis: {title}\n\n{summary[:180]} (1/3)",
            f"Key metrics indicate growing institutional integration but highlights regulatory shifts on the horizon. (2/3)",
            f"Watch the full visual breakdown and data source files here: {youtube_url} (3/3)"
        ]

        return CommunitySeedPackage(
            pipeline_id=pipeline_id,
            video_title=title,
            youtube_url=youtube_url,
            target_subreddits=subreddits,
            reddit_drafts=reddit_drafts,
            hn_post_draft=hn_post,
            blog_article_markdown=blog_markdown,
            linkedin_post_draft=linkedin_post,
            x_thread_draft=x_thread
        )

    async def dispatch_webhook_notification(self, package: CommunitySeedPackage) -> bool:
        """
        Sends formatted Discord/Slack webhook notification containing 1-click copyable drafts.
        """
        if not self.discord_webhook and not self.slack_webhook:
            logger.info(f"[SEED_DISTRIBUTOR] Seed package created for pipeline {package.pipeline_id}. (No Webhook URL set, skipping dispatch).")
            return False

        fields = []
        for idx, draft in enumerate(package.reddit_drafts[:2], 1):
            fields.append({
                "name": f"📌 Reddit Draft #{idx} (r/{draft.target_subreddit})",
                "value": f"**Title:** {draft.title}\n```markdown\n{draft.body_markdown[:300]}...\n```",
                "inline": False
            })

        if package.linkedin_post_draft:
            fields.append({
                "name": "👔 LinkedIn Post Draft",
                "value": f"```text\n{package.linkedin_post_draft[:300]}...\n```",
                "inline": False
            })

        if package.x_thread_draft:
            x_str = "\n\n".join([f"Tweet {i}: {t}" for i, t in enumerate(package.x_thread_draft, 1)])
            fields.append({
                "name": "🐦 X/Twitter Thread Draft",
                "value": f"```text\n{x_str[:300]}...\n```",
                "inline": False
            })

        try:
            async with aiohttp.ClientSession() as session:
                if self.discord_webhook:
                    payload = {
                        "username": "BuzzDrop Seed Assistant",
                        "embeds": [{
                            "title": f"🚀 Seed Traffic Ready: {package.video_title}",
                            "url": package.youtube_url,
                            "color": 5814783,
                            "description": f"Video uploaded successfully! Copy the drafts below to seed early traffic.",
                            "fields": fields,
                            "footer": {"text": f"Pipeline ID: {package.pipeline_id}"}
                        }]
                    }
                    async with session.post(self.discord_webhook, json=payload) as resp:
                        logger.info(f"[SEED_DISTRIBUTOR] Dispatched Discord seed notification (status {resp.status}).")

                if self.slack_webhook:
                    slack_text = (
                        f"🚀 *Seed Traffic Ready: {package.video_title}*\n"
                        f"URL: {package.youtube_url}\n\n"
                    )
                    for idx, draft in enumerate(package.reddit_drafts[:2], 1):
                        slack_text += (
                            f"*r/{draft.target_subreddit} Draft #{idx}*\n"
                            f"```\n{draft.body_markdown[:200]}...\n```\n"
                        )
                    if package.linkedin_post_draft:
                        slack_text += f"*LinkedIn Draft:*\n```\n{package.linkedin_post_draft[:200]}...\n```\n"
                    if package.x_thread_draft:
                        x_flat = "\n".join(package.x_thread_draft)
                        slack_text += f"*X Thread Draft:*\n```\n{x_flat[:200]}...\n```\n"

                    slack_payload = {"text": slack_text}
                    async with session.post(self.slack_webhook, json=slack_payload) as resp:
                        logger.info(f"[SEED_DISTRIBUTOR] Dispatched Slack seed notification (status {resp.status}).")

            return True
        except Exception as e:
            logger.warning(f"[SEED_DISTRIBUTOR] Failed to dispatch webhook: {e}")
            return False

seed_distributor = SeedDistributorEngine()


