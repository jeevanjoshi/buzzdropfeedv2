import os
import logging
import aiohttp
from typing import List, Dict, Any, Optional
from src.schemas.state import GlobalState
from src.schemas.seed_distribution import CommunitySeedPackage, RedditDraft

logger = logging.getLogger("CSVG_PIPELINE")

# Relevance floor for a subreddit vs the produced video (the seeded MiniLM gate —
# measured on live titles the correct match scores >= 0.50 while off-topic sits
# ~0.2-0.4; ``SEED_RELEVANCE_THRESHOLD`` default 0.50, raise toward the nominal
# 0.75 via env for a stricter gate). The seed gate runs on the RESIDENT MiniLM
# sentence transformer (``SemanticEmbeddingBackend``, master-only — torch + the
# cached all-MiniLM-L6-v2 model), default ON; ``SEED_RELEVANCE_SEMANTIC=0`` forces
# the TF-IDF fallback, which scores on a smaller scale and uses its own
# calibrated floor (0.30). The backend actually used is logged each run. Failure
# -> keyword pre-matching decides (never errors).
SEED_RELEVANCE_THRESHOLD = float(os.getenv("SEED_RELEVANCE_THRESHOLD", "0.50"))
SEED_RELEVANCE_TFIDF_THRESHOLD = float(os.getenv("SEED_RELEVANCE_TFIDF_THRESHOLD", "0.30"))
_SEED_SEMANTIC = os.getenv("SEED_RELEVANCE_SEMANTIC", "1").strip().lower() in ("1", "true", "yes")

# Resident MiniLM backend for the seed relevance gate (default ON when torch +
# the cached all-MiniLM-L6-v2 model are present; SEED_RELEVANCE_SEMANTIC=0 → TF-IDF).
# Loaded lazily on first encode (resident for the process lifetime afterwards).
if _SEED_SEMANTIC:
    # Point Hugging Face at the repo-local model cache so the seed gate loads
    # the resident MiniLM OFFLINE (the same .hf_cache the semantic gates use).
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _HF_CACHE = os.path.join(_REPO_ROOT, ".hf_cache")
    if os.path.isdir(_HF_CACHE):
        for _k in ("HF_HOME", "TRANSFORMERS_CACHE"):
            if not os.getenv(_k):
                os.environ.setdefault(_k, _HF_CACHE)
try:
    from src.engine.text_embeddings import SemanticEmbeddingBackend
    _seed_embedder = SemanticEmbeddingBackend(enabled=_SEED_SEMANTIC)
except Exception:
    _seed_embedder = None

# Subreddit → topical descriptor anchors, so embedding a subreddit against the
# video's title/summary/keywords is meaningful (a bare sub name is too short to
# tensor-encode reliably).
_SUBREDDIT_DESCRIPTIONS = {
    "technology": "technology software hardware innovation gadgets computing",
    "finance": "finance investing markets money economy stocks portfolio",
    "economics": "economics macro economy markets inflation growth policy",
    "artificial": "artificial intelligence ai machine learning models gpu",
    "ArtificialInteligence": "artificial intelligence ai machine learning research",
    "wallstreetbets": "stocks trading options markets wall street speculation",
    "stocks": "stocks investing equity markets earnings trading nasdaq",
    "science": "science research physics biology discovery experiments",
    "space": "space nasa rocket astronomy cosmos exploration mars",
    "history": "history historical documentary archaeology ancient civilization",
    "investing": "investing stocks funds index compound growth markets",
    "Futurology": "future technology innovation artificial intelligence trends",
    "IndiaInvestments": "india investing markets sensex nifty economy stocks",
    "AusFinance": "australia finance home loans investing asx markets",
    "PersonalFinanceCanada": "canada personal finance investing rrsps tfra markets",
}


class SeedDistributorEngine:
    """
    Intelligent Seed Traffic Distribution Engine.
    Discovers high-relevance niche subreddits and forums, generates 70/30 Value-First draft content,
    and dispatches Discord/Slack webhooks with 1-click Markdown payloads for creator approval.
    """

    def __init__(self):
        self.discord_webhook = os.getenv("DISCORD_SEED_WEBHOOK_URL")
        self.slack_webhook = os.getenv("SLACK_SEED_WEBHOOK_URL")

    @staticmethod
    def _video_text(state: GlobalState) -> str:
        """Concatenated video subject text used as the semantic query."""
        title = (getattr(getattr(state, "seo_metadata", None), "title", "")
                 or getattr(getattr(state, "script_data", None), "title", ""))
        summary = (getattr(state.selected_topic, "summary", "") if state.selected_topic else "")
        kws = (state.selected_topic.keywords if state.selected_topic and state.selected_topic.keywords else [])
        return " ".join([title, summary] + (kws if isinstance(kws, list) else [str(kws)])).lower()

    @staticmethod
    def _semantic_relevance(text: str, anchor_text: str) -> Optional[tuple]:
        """
        True whole-string cosine relevance of ``text`` vs ``anchor_text``:
        the RESIDENT MiniLM encoder (``_seed_embedder``) first — the documented
        seed gate — otherwise the TF-IDF char-n-gram fallback (sklearn, always
        available). Returns (similarity, backend) or None on failure; callers
        fall back to the keyword pre-match without ever erroring.
        """
        try:
            vecs = None
            if _seed_embedder is not None:
                vecs = _seed_embedder.encode_batch([text, anchor_text])
            if vecs is not None:
                sim = float(vecs[0] @ vecs[1])
                return sim, "minilm"
        except Exception as e:
            logger.warning(f"[SEED_DISTRIBUTOR] MiniLM relevance failed ({e}); falling back to TF-IDF.")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            _matrix = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5),
                max_features=2000, sublinear_tf=True,
            ).fit_transform([text, anchor_text])
            sim = float(cosine_similarity(_matrix[0:1], _matrix[1:2])[0][0])
            return sim, "tfidf"
        except Exception as e:
            logger.warning(f"[SEED_DISTRIBUTOR] semantic relevance failed ({e}); keyword-only selection.")
            return None

    def select_target_subreddits(self, state: GlobalState) -> List[str]:
        """
        Selects target subreddits via keyword pre-matching COMPOUNDED by a
        semantic-relevance gate: a subreddit is only included when its whole
        descriptor vs the video's actual content (title + summary + keywords)
        clears the backend's threshold — MiniLM 0.75 (the documented gate) or the
        calibrated TF-IDF floor when only sklearn is present. Empty result
        degrades to the best keyword match with a warning (never an empty
        package).
        """
        kws = state.selected_topic.keywords if state.selected_topic and state.selected_topic.keywords else ["finance", "tech"]
        kws = kws if isinstance(kws, list) else list(kws)
        kw_set = {str(k).lower() for k in kws}

        # 1. Keyword pre-match: candidate pool = subs whose NAME or TOPIC
        #    descriptor shares a keyword token (broad; the semantic gate decides).
        pool: List[str] = []
        for sub, desc in _SUBREDDIT_DESCRIPTIONS.items():
            dwords = set(desc.split())
            if kw_set & dwords or any(k in sub.lower() or sub.lower() in k for k in kw_set):
                pool.append(sub)
        if not pool:
            pool = ["technology", "finance"]

        text = self._video_text(state)

        # 2. Semantic gate over the whole descriptor string.
        scored: List[tuple] = []          # (sim, backend, sub)
        for sub in pool:
            anchor = _SUBREDDIT_DESCRIPTIONS[sub] + " " + " ".join(kws)
            res = self._semantic_relevance(text, anchor)
            if res is not None:
                scored.append((res[0], res[1], sub))
        scored.sort(key=lambda x: -x[0])

        backend = scored[0][1] if scored else "tfidf"
        threshold = SEED_RELEVANCE_THRESHOLD if backend == "minilm" else SEED_RELEVANCE_TFIDF_THRESHOLD
        passed = [sub for sim, _, sub in scored if sim >= threshold]

        if passed:
            logger.info(
                f"[SEED_DISTRIBUTOR] Relevance gate ({backend}, >= {threshold:.2f}) passed for "
                f"{len(passed)} subreddit(s): {[f'r/{s}' for s in passed]}")
            return passed[:3]

        # 3. Fallback: nothing cleared the backend-appropriate threshold — keep
        #    the best match with a loud warning (a rough-fit sub beats no seeds).
        best = scored[0][2] if scored else pool[0]
        best_sim = scored[0][0] if scored else 0.0
        logger.warning(
            f"[SEED_DISTRIBUTOR] No subreddit reached relevance >= {threshold:.2f} "
            f"({backend}: best sim={best_sim:.3f} on r/{best}); falling back to keyword match for seeding.")
        return [best]

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

        # 70/30 Value-First takeaways GROUNDED in the run's verified facts
        # (never fabricated): top-2 fact summaries become the numbered bullets;
        # a fallback risk line only fills when the corpus is short.
        fact_lines: List[str] = []
        for vf in (state.verified_facts or [])[:2]:
            _s = (getattr(vf, "summary", None) or getattr(vf, "headline", None) or "").strip()
            if _s:
                fact_lines.append(_s[:200])
        if len(fact_lines) < 2:
            fact_lines.append("Risk factors include regulatory uncertainty and technical scaling hurdles in the near term.")
        risk_line = fact_lines[1] if len(fact_lines) >= 2 else fact_lines[0]
        takeaway_lines = [
            f"1. {fact_lines[0] if fact_lines else summary[:200]}",
            f"2. {risk_line}",
            "3. The shift is structural — data sources show a sustained move, not a one-off tick.",
        ]

        reddit_drafts = []
        for sub in subreddits:
            body = (
                "**Key Takeaways & Core Analysis:**\n\n"
                + "\n".join(takeaway_lines)
                + f"\n\n--- \n"
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
            "Key metrics indicate growing institutional integration but highlights regulatory shifts on the horizon. (2/3)",
            f"Watch the full visual breakdown and data source files here: {youtube_url} (3/3)"
        ]

        tiktok_caption = (
            f"🔍 {title}\n\n"
            f"{summary[:150]}... \n\n"
            f"Watch the full breakdown on YouTube! Link in bio. 🎥✨\n"
            f"#finance #money #investing #education #datapack #insights"
        )

        instagram_caption = (
            f"📊 {title}\n\n"
            f"{summary[:200]}...\n\n"
            f"👉 Tap the link in our bio to watch the complete visual analysis on YouTube!\n\n"
            f"#business #technology #investing #data #stockmarket #globalfinance"
        )

        pinterest_draft = (
            f"Board Idea: Finance & Technology Infographics\n"
            f"Pin Title: {title[:95]}\n"
            f"Pin Description: {summary[:450]}\n"
            f"Destination Link: {youtube_url}"
        )

        telegram_post = (
            f"📢 **{title}**\n\n"
            f"{summary}\n\n"
            f"💡 *Key Takeaways:*\n"
            f"{takeaway_lines[0]}\n"
            f"{takeaway_lines[1]}\n\n"
            f"🔗 Watch the full documentary breakdown here: {youtube_url}"
        )

        return CommunitySeedPackage(
            pipeline_id=pipeline_id,
            video_title=title,
            youtube_url=youtube_url,
            target_subreddits=subreddits,
            reddit_drafts=reddit_drafts,
            hn_post_draft=hn_post,
            blog_article_markdown=blog_markdown,
            linkedin_post_draft=linkedin_post,
            x_thread_draft=x_thread,
            tiktok_caption=tiktok_caption,
            instagram_caption=instagram_caption,
            pinterest_draft=pinterest_draft,
            telegram_post=telegram_post
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

        if package.tiktok_caption:
            fields.append({
                "name": "🎵 TikTok Caption",
                "value": f"```text\n{package.tiktok_caption[:300]}...\n```",
                "inline": False
            })

        if package.instagram_caption:
            fields.append({
                "name": "📸 Instagram Caption",
                "value": f"```text\n{package.instagram_caption[:300]}...\n```",
                "inline": False
            })

        if package.pinterest_draft:
            fields.append({
                "name": "📌 Pinterest Pin Draft",
                "value": f"```text\n{package.pinterest_draft[:300]}...\n```",
                "inline": False
            })

        if package.telegram_post:
            fields.append({
                "name": "✈️ Telegram Post",
                "value": f"```text\n{package.telegram_post[:300]}...\n```",
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
                            "description": "Video uploaded successfully! Copy the drafts below to seed early traffic.",
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
                    if package.tiktok_caption:
                        slack_text += f"*TikTok Caption:*\n```\n{package.tiktok_caption[:200]}...\n```\n"
                    if package.instagram_caption:
                        slack_text += f"*Instagram Caption:*\n```\n{package.instagram_caption[:200]}...\n```\n"
                    if package.pinterest_draft:
                        slack_text += f"*Pinterest Pin:*\n```\n{package.pinterest_draft[:200]}...\n```\n"
                    if package.telegram_post:
                        slack_text += f"*Telegram Post:*\n```\n{package.telegram_post[:200]}...\n```\n"

                    slack_payload = {"text": slack_text}
                    async with session.post(self.slack_webhook, json=slack_payload) as resp:
                        logger.info(f"[SEED_DISTRIBUTOR] Dispatched Slack seed notification (status {resp.status}).")

            return True
        except Exception as e:
            logger.warning(f"[SEED_DISTRIBUTOR] Failed to dispatch webhook: {e}")
            return False

seed_distributor = SeedDistributorEngine()
