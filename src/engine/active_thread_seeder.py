import os
import logging
from typing import List, Dict, Any, Optional
from src.schemas.state import GlobalState
from src.engine.llm_client import LLMClient

logger = logging.getLogger("CSVG_PIPELINE")


class BasePlatformSeeder:
    """
    Abstract interface for platform-specific seeding engines
    to prepare for multi-platform support.
    """
    def search_active_threads(self, query: str, limit: int) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def post_reply(self, thread_id: str, text: str) -> bool:
        raise NotImplementedError


class RedditSeeder(BasePlatformSeeder):
    """
    Reddit-specific seeder implementation using the official PRAW client.
    """
    def __init__(self):
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.username = os.getenv("REDDIT_USERNAME")
        self.password = os.getenv("REDDIT_PASSWORD")
        self.user_agent = os.getenv("REDDIT_USER_AGENT", "CSVG-Bot/1.0")
        
        self.reddit = None
        if all([self.client_id, self.client_secret, self.username, self.password]):
            try:
                import praw
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    username=self.username,
                    password=self.password,
                    user_agent=self.user_agent
                )
            except Exception as e:
                logger.warning(f"[RedditSeeder] Failed to initialize PRAW: {e}")

    def is_available(self) -> bool:
        return self.reddit is not None

    def search_active_threads(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.is_available():
            return []
        
        results = []
        try:
            # Search globally across Reddit, sorting by relevance or comments, limited to the last 24 hours
            submissions = self.reddit.subreddit("all").search(
                query, sort="relevance", time_filter="day", limit=limit
            )
            for sub in submissions:
                # Skip locked or archived threads
                if sub.locked or sub.archived:
                    continue
                results.append({
                    "id": sub.id,
                    "title": sub.title,
                    "selftext": sub.selftext,
                    "subreddit": sub.subreddit.display_name,
                    "url": sub.url,
                    "score": sub.score,
                    "num_comments": sub.num_comments,
                    "submission_obj": sub
                })
        except Exception as e:
            logger.warning(f"[RedditSeeder] Global search failed: {e}")
        return results

    def get_comments_context(self, submission, max_comments: int = 3) -> str:
        context_lines = []
        try:
            # Load comments
            submission.comments.replace_more(limit=0)
            for comment in submission.comments[:max_comments]:
                # Safeguard against deleted users or blank comments
                author_name = comment.author.name if comment.author else "[deleted]"
                body_snippet = comment.body[:300] if comment.body else ""
                context_lines.append(f"- Comment by u/{author_name}: {body_snippet}")
        except Exception as e:
            logger.warning(f"[RedditSeeder] Comment context loading failed: {e}")
        return "\n".join(context_lines)

    def post_reply(self, thread_id: str, text: str) -> bool:
        if not self.is_available():
            return False
        try:
            submission = self.reddit.submission(id=thread_id)
            comment = submission.reply(text)
            logger.info(
                f"[RedditSeeder] Successfully posted reply on r/{submission.subreddit.display_name} "
                f"thread '{submission.title}'! Comment ID: {comment.id}"
            )
            return True
        except Exception as e:
            logger.error(f"[RedditSeeder] Failed to post reply to thread {thread_id}: {e}")
            return False


class ActiveThreadSeederEngine:
    """
    Intelligent seeding assistant that searches for active online discussions,
    uses the LLM to write custom context-aware replies, and submits them.
    """
    def __init__(self):
        self.seeders = {
            "reddit": RedditSeeder()
        }
        self.llm_client = None
        self.warmup_mode = os.getenv("ACTIVE_SEEDER_WARMUP", "0").strip().lower() in ("1", "true", "yes")

    async def seed_active_discussions(self, state: GlobalState, youtube_url: str):
        logger.info("[ActiveThreadSeeder] Starting active thread comment reply bot...")
        
        # Initialize LLM client lazily
        if self.llm_client is None:
            self.llm_client = LLMClient()

        # 1. Determine the search queries
        title = (
            getattr(getattr(state, "seo_metadata", None), "title", "")
            or getattr(getattr(state, "script_data", None), "title", "")
            or (state.selected_topic.headline if state.selected_topic else "Finance and Tech Trends")
        )
        keywords = state.selected_topic.keywords if state.selected_topic and state.selected_topic.keywords else []
        keywords = keywords if isinstance(keywords, list) else list(keywords)
        
        if not keywords:
            keywords = ["finance", "economy", "technology"]

        # Search query options: topic headline, or top keywords combined
        search_queries = []
        if state.selected_topic and getattr(state.selected_topic, "headline", ""):
            search_queries.append(state.selected_topic.headline)
        if len(keywords) >= 2:
            search_queries.append(f"{keywords[0]} {keywords[1]}")
        search_queries.append(keywords[0])

        # 2. Gather candidates from active seeders
        reddit_seeder = self.seeders.get("reddit")
        if not reddit_seeder or not reddit_seeder.is_available():
            logger.info("[ActiveThreadSeeder] Reddit credentials not configured. Skipping active thread seeding.")
            return

        candidates = []
        for query in search_queries:
            logger.info(f"[ActiveThreadSeeder] Searching Reddit for '{query}'...")
            found = reddit_seeder.search_active_threads(query, limit=5)
            if found:
                candidates.extend(found)
                # Break early if we have found threads
                break
        
        if not candidates:
            logger.warning("[ActiveThreadSeeder] No active relevant Reddit threads found from the last 24 hours.")
            return

        # Deduplicate candidates by ID
        unique_candidates = {}
        for c in candidates:
            unique_candidates[c["id"]] = c
        deduped_candidates = list(unique_candidates.values())

        # Sort by comments/score to find the most active thread (max visibility)
        deduped_candidates.sort(key=lambda x: -x["num_comments"])
        target_thread = deduped_candidates[0]
        
        logger.info(
            f"[ActiveThreadSeeder] Selected target thread: r/{target_thread['subreddit']} - "
            f"'{target_thread['title']}' (ID: {target_thread['id']}, Comments: {target_thread['num_comments']})"
        )

        # 3. Load full comment context
        comments_context = reddit_seeder.get_comments_context(target_thread["submission_obj"], max_comments=3)

        # 4. Invoke LLM to generate context-aware response
        verified_facts_str = ""
        if state.verified_facts:
            verified_facts_str = "\n".join(
                [f"- {getattr(f, 'summary', '') or getattr(f, 'headline', '')}" for f in state.verified_facts[:3]]
            )

        system_prompt = (
            "You are an expert commenter participating in an online discussion forum thread. "
            "Your task is to write a highly helpful, engaging, and organic comment responding to the thread. "
            "Follow these critical guidelines:\n"
            "- Do not sound like a bot, promotional marketer, or corporate advertiser.\n"
            "- Do not start with phrases like 'Here is a breakdown' or 'In this video...'.\n"
            "- Add real, factual value to the discussion. Speak like an experienced human user.\n"
            "- Tailor your comment directly to the specific posts and top comments context provided.\n"
            "You must return a JSON object in this exact format:\n"
            "{\n"
            "  \"comment\": \"<your response text here>\",\n"
            "  \"reasoning\": \"<brief explanation of why this fits the thread>\"\n"
            "}"
        )

        prompt = (
            f"Thread Title: {target_thread['title']}\n"
            f"Thread Subreddit: r/{target_thread['subreddit']}\n"
            f"Thread Content: {target_thread['selftext'][:1000]}\n\n"
            f"Context of Existing Comments in Thread:\n{comments_context}\n\n"
            f"Our Video Topic: {title}\n"
            f"Verified Facts we know:\n{verified_facts_str}\n\n"
        )

        if self.warmup_mode:
            prompt += (
                "CRITICAL: We are in account WARMUP mode. "
                "Do NOT include any links, URLs, mentions of a YouTube video, or promotional text. "
                "Write a 100% informational, engaging comment that answers the thread."
            )
        else:
            prompt += (
                "CRITICAL: Naturally mention or cite our video breakdown as a source/reference. "
                "For example: 'I put together a full visual animated breakdown on this here: [link]'. "
                "Use the exact URL: {youtube_url} but replace the anchor text appropriately. "
                "Do not make the link drop feel forced or spammy. Keep it subtle and at the end."
            ).format(youtube_url=youtube_url)

        if not self.llm_client.is_available():
            logger.warning("[ActiveThreadSeeder] LLM Client is not available. Skipping reply generation.")
            return

        logger.info("[ActiveThreadSeeder] Requesting context-aware reply from LLM...")
        llm_res = self.llm_client.generate_json(prompt, system_prompt=system_prompt, route="generate")
        if not llm_res or "comment" not in llm_res:
            logger.warning(
                f"[ActiveThreadSeeder] LLM reply generation failed or returned invalid JSON structure: {llm_res}"
            )
            return

        comment_text = llm_res["comment"].strip()
        reasoning = llm_res.get("reasoning", "")
        logger.info(f"[ActiveThreadSeeder] LLM generated reply (Reasoning: '{reasoning}'):\n{comment_text}")

        # 5. Post reply
        success = reddit_seeder.post_reply(target_thread["id"], comment_text)
        if success:
            logger.info(f"[ActiveThreadSeeder] Active thread comment successfully posted!")
        else:
            logger.warning(f"[ActiveThreadSeeder] Failed to post reply to Reddit.")


active_thread_seeder = ActiveThreadSeederEngine()
