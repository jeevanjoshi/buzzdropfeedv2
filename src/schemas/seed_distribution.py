from pydantic import BaseModel, Field
from typing import List, Optional

class RedditDraft(BaseModel):
    target_subreddit: str = Field(description="Subreddit name e.g. 'technology' or 'finance'")
    title: str = Field(description="High-CTR Reddit thread title")
    body_markdown: str = Field(description="Informational post body with natural video link citation (70/30 rule)")
    suggested_flair: Optional[str] = None

class CommunitySeedPackage(BaseModel):
    pipeline_id: str
    video_title: str
    youtube_url: str
    target_subreddits: List[str] = Field(default_factory=list)
    reddit_drafts: List[RedditDraft] = Field(default_factory=list)
    hn_post_draft: Optional[str] = None
    quora_answers: List[dict] = Field(default_factory=list)
    blog_article_markdown: Optional[str] = None
    linkedin_post_draft: Optional[str] = None
    x_thread_draft: List[str] = Field(default_factory=list)
    tiktok_caption: Optional[str] = None
    instagram_caption: Optional[str] = None
    pinterest_draft: Optional[str] = None
    telegram_post: Optional[str] = None


class DiscordEmbedField(BaseModel):
    name: str
    value: str
    inline: bool = False

class DiscordWebhookPayload(BaseModel):
    username: str = "BuzzDrop Seed Assistant"
    avatar_url: str = "https://i.imgur.com/8N4Z4Z4.png"
    embeds: List[dict] = Field(default_factory=list)
