import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from src.schemas.state import GlobalState
from src.engine.llm_client import LLMClient
from src.engine.reddit_json_client import RedditJsonClient
from src.engine.reddit_browser_poster import RedditBrowserPoster

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
        has_praw = os.getenv("REDDIT_CLIENT_ID") and not os.getenv("REDDIT_CLIENT_ID").startswith("xxxx")
        self.browser_poster = RedditBrowserPoster()
        # Browser automation is the primary backend (renders where .json is blocked).
        if has_praw:
            self.seeders = {"reddit": RedditSeeder()}
        elif self.browser_poster.has_accounts():
            self.seeders = {"reddit": self.browser_poster}
        else:
            self.seeders = {"reddit": RedditJsonClient()}
        self.llm_client = None
        self.warmup_mode = os.getenv("ACTIVE_SEEDER_WARMUP", "1").strip().lower() in ("1", "true", "yes")

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
        if not reddit_seeder:
            logger.info("[ActiveThreadSeeder] No Reddit seeder available. Skipping active thread seeding.")
            return

        candidates = []
        for query in search_queries:
            logger.info(f"[ActiveThreadSeeder] Searching Reddit for '{query}'...")
            found = await asyncio.to_thread(reddit_seeder.search_active_threads, query, limit=25)
            if found:
                candidates.extend(found)
        
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
        
        posted_subreddits = set()
        retained_count = 0
        target_retained = 15
        # Minimum topical overlap before we touch a thread. Link drops into
        # off-topic threads are the #1 AutoMod / self-promo trigger, so we hold
        # links to strongly-relevant threads and keep warmup (no-link) comments
        # to at-least-tangentially-relevant ones.
        _kw_tokens = {str(k).lower() for k in keywords if len(str(k)) >= 4}
        min_link_rel = int(os.getenv("ACTIVE_SEEDER_MIN_LINK_RELEVANCE", "3"))
        min_warmup_rel = int(os.getenv("ACTIVE_SEEDER_MIN_WARMUP_RELEVANCE", "1"))

        for target_thread in deduped_candidates:
            if retained_count >= target_retained:
                break

            subreddit = target_thread.get("subreddit")
            if not subreddit:
                continue

            if subreddit.lower() in posted_subreddits:
                continue

            # Topical fit: count distinct topic keywords present in the thread
            # title + body. Skips blatant self-promo into unrelated discussions.
            _blob = f"{target_thread.get('title', '')} {target_thread.get('selftext', '')}".lower()
            _rel = sum(1 for k in _kw_tokens if k in _blob)
            _floor = min_warmup_rel if self.warmup_mode else min_link_rel
            if _rel < _floor:
                logger.info(
                    f"[ActiveThreadSeeder] Skipping r/{subreddit} (relevance={_rel} < {_floor}): "
                    f"'{target_thread['title'][:50]}'"
                )
                continue

            logger.info(
                f"[ActiveThreadSeeder] Selected target thread ({retained_count + 1}/{target_retained}): "
                f"r/{subreddit} - '{target_thread['title']}' (ID: {target_thread['id']}, Comments: {target_thread['num_comments']}, Relevance: {_rel})"
            )

            # 3. Load full comment context
            comments_context = self._get_comments_context(reddit_seeder, target_thread)

            # 4. Invoke LLM to generate context-aware response
            verified_facts_str = ""
            if state.verified_facts:
                verified_facts_str = "\n".join(
                    [f"- {getattr(f, 'summary', '') or getattr(f, 'headline', '')}" for f in state.verified_facts[:3]]
                )

            system_prompt = (
                "You are an expert commenter participating in an online discussion forum thread. "
                "Your task is to write a highly helpful, engaging, and organic comment responding to the thread. "
                "Follow these critical guidelines to avoid automated filters and look completely human:\n"
                "- Do not sound like a bot, promotional marketer, or corporate advertiser.\n"
                "- Speak like a regular casual user: use natural phrasing, contractions (e.g., i'm, don't, it's), and informal structure. Avoid perfect 'academic' essay styles.\n"
                "- Do not use stereotypical AI buzzwords or transitions (e.g., 'delve', 'crucial', 'testament', 'furthermore', 'moreover', 'in summary', 'ultimately', 'it's important to remember').\n"
                "- Keep formatting casual: avoid neat bullet lists, numbered points, or bold headers unless completely natural for the context. Keep paragraphs short (1-2 paragraphs max, 2-4 sentences total is optimal).\n"
                "- Do not start with phrases like 'Here is a breakdown', 'In this video...', or 'Based on the thread...'.\n"
                "- Add real, factual value to the discussion. Speak like an experienced human user.\n"
                "- Tailor your comment directly to the specific posts and top comments context provided.\n"
                "You must return a JSON object in this exact format:\n"
                "{\n"
                "  \"comment\": \"<your response text here>\",\n"
                "  \"reasoning\": \"<brief explanation of why this fits the thread>\"\n"
                "}"
            )

            # Use a plain, un-obfuscated YouTube short link. Obfuscated variants
            # (``youtube[dot]com``, zero-width spaces, backticks) are exactly what
            # AutoMod / Reddit's spam filter are trained to catch — they make a
            # comment look MORE like spam, not less. A clean link from an
            # established account in an on-topic thread is far less likely to be
            # removed.
            video_id = youtube_url.split("v=")[-1].split("&")[0] if "v=" in youtube_url else youtube_url.split("/")[-1]
            clean_url = f"https://youtu.be/{video_id}"

            prompt = (
                f"Thread Title: {target_thread['title']}\n"
                f"Thread Subreddit: r/{subreddit}\n"
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
                    "For example: 'I saw a good visual breakdown of this on youtube (url) that explains...' or 'put together a quick visualization of this here: url'. "
                    "Use the exact URL/reference: {clean_url} but replace or weave the anchor text/reference absolutely naturally. "
                    "You can place the URL reference anywhere in the comment (beginning, middle, or end). "
                    "Do not make the link drop feel forced or spammy. Keep it subtle."
                ).format(clean_url=clean_url)

            if not self.llm_client.is_available():
                logger.warning("[ActiveThreadSeeder] LLM Client is not available. Skipping reply generation.")
                return

            logger.info("[ActiveThreadSeeder] Requesting context-aware reply from LLM...")
            llm_res = self.llm_client.generate_json(prompt, system_prompt=system_prompt, route="generate")
            if not llm_res or "comment" not in llm_res:
                logger.warning(
                    f"[ActiveThreadSeeder] LLM reply generation failed or returned invalid JSON structure: {llm_res}"
                )
                continue

            comment_text = llm_res["comment"].strip()
            reasoning = llm_res.get("reasoning", "")
            logger.info(f"[ActiveThreadSeeder] LLM generated reply (Reasoning: '{reasoning}'):\n{comment_text}")

            # 5. Post reply
            success = await asyncio.to_thread(self._post_reply, target_thread, comment_text)
            if success:
                logger.info(f"[ActiveThreadSeeder] Active thread comment successfully posted and verified on r/{subreddit}!")
                posted_subreddits.add(subreddit.lower())
                retained_count += 1
            else:
                logger.warning(f"[ActiveThreadSeeder] Reply failed or unverified (possible shadowban/Automod filtering) on r/{subreddit}. Trying other relevant groups...")
        
        logger.info(f"[ActiveThreadSeeder] Seeding finished. Total retained comments posted: {retained_count}/{target_retained}")

    def _get_comments_context(self, seeder, target_thread: Dict[str, Any]) -> str:
        """Loads top comment context from whichever seeder backend is active."""
        if hasattr(seeder, "get_comments_context"):
            try:
                ctx = seeder.get_comments_context(
                    target_thread.get("subreddit"), target_thread.get("id"), max_comments=3
                )
                if ctx:
                    return ctx
            except Exception:
                pass
            submission_obj = target_thread.get("submission_obj")
            if submission_obj is not None:
                try:
                    return seeder.get_comments_context(submission_obj, max_comments=3)
                except Exception:
                    pass
        return ""

    def _post_reply(self, target_thread: Dict[str, Any], comment_text: str) -> bool:
        """Posts via Playwright browser automation (falls back to PRAW if creds exist)."""
        seeder = self.seeders.get("reddit")
        if isinstance(seeder, (RedditBrowserPoster, RedditJsonClient)):
            if not self.browser_poster.has_accounts():
                logger.warning("[ActiveThreadSeeder] No Reddit accounts configured for browser posting. Skipping post.")
                return False
            return self.browser_poster.post_reply(
                thread_id=target_thread.get("id"),
                subreddit=target_thread.get("subreddit"),
                permalink=target_thread.get("permalink"),
                text=comment_text,
            )
        # Legacy PRAW path (requires REDDIT_CLIENT_ID/CLIENT_SECRET).
        return seeder.post_reply(target_thread["id"], comment_text)


active_thread_seeder = ActiveThreadSeederEngine()
