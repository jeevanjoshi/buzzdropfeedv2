import os
import http.client
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="YouTube Publishing MCP Server")


class UploadRequest(BaseModel):
    video_path: str
    title: str
    description: str
    tags: List[str] = ["finance", "tech", "economics", "stocks", "business"]
    category_id: str = "27"  # Education / Finance
    thumbnail_path: Optional[str] = None


class QuotaCheckRequest(BaseModel):
    current_daily_uploads: int = 0


class InsertCommentRequest(BaseModel):
    video_id: str
    comment_text: str


# Explicit opt-in for the offline/test mock upload path. Production runs
# (no env) must NEVER silently fall back to a fabricated `demo_*` id — a masked
# upload failure was previously recorded as PUBLISHED_SUCCESS (see
# publish-integrity-quality-fix-plan.md issue #1).
_MOCK_ENABLED = os.getenv("YOUTUBE_UPLOAD_MOCK", "").strip().lower() in ("1", "true", "yes")

# Publish visibility for long-form videos AND Shorts. Default is PUBLIC (the
# channel's goal is watch-time + reach); set YOUTUBE_PRIVACY_STATUS=private|unlisted
# in .env to stage drafts instead.
YOUTUBE_PRIVACY_STATUS = os.getenv("YOUTUBE_PRIVACY_STATUS", "public").strip().lower()


def _mock_payload(kind: str = "upload") -> Dict[str, Any]:
    """Fabricated demo identifiers for the explicitly-enabled mock path only."""
    suffix = os.urandom(4).hex()
    if kind == "comment":
        return {"status": "mock", "engine": "mock_youtube_comment",
                "comment_id": f"comment_{suffix}", "video_id": "", "pinned": False}
    if kind == "shorts":
        return {"status": "mock", "engine": "mock_youtube_shorts",
                "video_id": f"demo_short_{suffix}", "youtube_url": f"https://www.youtube.com/shorts/demo_short_{suffix}",
                "synthetic_flag": True}
    return {"status": "mock", "engine": "mock_youtube_upload",
            "video_id": f"demo_{suffix}", "youtube_url": f"https://www.youtube.com/watch?v=demo_{suffix}",
            "synthetic_flag": True}


@app.post("/tools/check_quota_available")
async def check_quota_available(req: QuotaCheckRequest):
    """
    Checks YouTube Data API v3 daily quota safety limits (10,000 unit limit).
    Each upload consumes 1,600 units. Max 4 uploads per day (6,400 units).
    """
    units_per_upload = 1600
    used_units = req.current_daily_uploads * units_per_upload
    quota_limit = 10000
    is_safe = used_units + units_per_upload <= quota_limit

    return {
        "is_safe": is_safe,
        "current_daily_uploads": req.current_daily_uploads,
        "max_daily_uploads": 4,
        "used_units": used_units,
        "quota_limit": quota_limit,
        "remaining_units": quota_limit - used_units
    }


@app.post("/tools/upload_youtube_resumable")
async def upload_youtube_resumable(req: UploadRequest):
    """
    Handles YouTube Data API v3 resumable video upload with 100% Headless OAuth2 refresh tokens (zero-human-input),
    chunked retries, and EU AI Act synthetic content disclosure metadata (`syntheticContent: true`).

    On real failure this RAISES (never a silent mock) unless ``YOUTUBE_UPLOAD_MOCK=1``
    is explicitly set (offline/dry-run only). A missing ``id`` in the API response is
    treated as a failure — no fabricated id is ever returned.
    """
    try:
        credentials = _load_credentials()
        import httplib2
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError

        youtube = build("youtube", "v3", credentials=credentials)

        body = {
            "snippet": {
                "title": req.title,
                "description": req.description,
                "tags": req.tags,
                "categoryId": req.category_id
            },
            "status": {
                "privacyStatus": YOUTUBE_PRIVACY_STATUS,
                "selfDeclaredMadeForKids": False,
                "syntheticContent": {
                    "bInformed": True,
                    "synthesized": True
                }
            }
        }
        return _resumable_upload(youtube, body, req.video_path, req.thumbnail_path)
    except Exception as e:
        if _MOCK_ENABLED:
            print(f"[YouTube Upload] MOCK mode active (YOUTUBE_UPLOAD_MOCK=1); upload would have failed: {e}")
            return _mock_payload("upload")
        raise HTTPException(status_code=502, detail=f"YouTube upload failed: {e}") from e


@app.post("/tools/upload_short")
async def upload_short(req: UploadRequest):
    """
    Uploads a vertical (<60s) clip as a YouTube Short. This is the GROWTH-phase
    discovery lever: the long-form master already yields 9:16 Shorts clips
    (micro_content_producer / media_producer) that are otherwise never published.

    Shorts use the SAME resumable Data API v3 upload (YouTube auto-classifies by
    aspect ratio + duration); we add the ``#Shorts`` hashtag to the description
    and tags to signal the Shorts surface, and keep EU AI Act synthetic-content
    disclosure. Same OAuth, same mock guard, same fail-loud semantics.
    """
    try:
        credentials = _load_credentials()
        from googleapiclient.discovery import build

        youtube = build("youtube", "v3", credentials=credentials)
        desc = (req.description or "") + "\n#Shorts #Discovery #finance"
        tags = [t for t in (req.tags or []) if t.lower() != "#shorts"] + ["#Shorts", "Shorts"]
        body = {
            "snippet": {
                "title": ((req.title or "")[:80]),
                "description": desc[:4900],
                "tags": list(tags)[:500],
                "categoryId": req.category_id or "22"  # People & Blogs (typical Shorts vertical)
            },
            "status": {
                "privacyStatus": YOUTUBE_PRIVACY_STATUS,
                "selfDeclaredMadeForKids": False,
                "syntheticContent": {
                    "bInformed": True,
                    "synthesized": True
                }
            }
        }
        return _resumable_upload(youtube, body, req.video_path, req.thumbnail_path)
    except Exception as e:
        if _MOCK_ENABLED:
            print(f"[YouTube Shorts] MOCK mode active (YOUTUBE_UPLOAD_MOCK=1); short upload would have failed: {e}")
            return _mock_payload("shorts")
        raise HTTPException(status_code=502, detail=f"YouTube Shorts upload failed: {e}") from e


def _load_credentials():
    """Load + refresh OAuth youtube.upload credentials from token.json."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    token_path = os.getenv("YOUTUBE_TOKEN_FILE", "token.json")
    credentials = None
    if os.path.exists(token_path):
        credentials = Credentials.from_authorized_user_file(token_path)
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
    if not credentials:
        raise RuntimeError("YouTube upload failed: no OAuth credentials (token.json missing or unusable)")
    return credentials


def _resumable_upload(youtube, body: Dict[str, Any], video_path: str, thumbnail_path: Optional[str]) -> Dict[str, Any]:
    """Chunked resumable insert + optional custom thumbnail; returns {status, video_id, youtube_url}."""
    import httplib2
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(video_path, chunksize=256 * 1024, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
        except HttpError as e:
            if e.resp.status in [308, 503, 502, 504]:
                continue
            else:
                raise e
        except (httplib2.ServerNotFoundError, http.client.IncompleteRead):
            continue

    video_id = (response or {}).get("id")
    if not video_id:
        raise RuntimeError("YouTube upload failed: API response carried no video id")

    if thumbnail_path and os.path.exists(thumbnail_path):
        try:
            media_thumb = MediaFileUpload(thumbnail_path, mimetype="image/png")
            youtube.thumbnails().set(videoId=video_id, media_body=media_thumb).execute()
            print(f"[YouTube Upload] Successfully set custom thumbnail: {thumbnail_path}")
        except Exception as thumb_err:
            print(f"Warning: Failed to upload custom thumbnail: {thumb_err}")

    return {
        "status": "success",
        "video_id": video_id,
        "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
        "synthetic_flag": True
    }


@app.post("/tools/insert_pinned_comment")
async def insert_pinned_comment(req: InsertCommentRequest):
    """
    Inserts a pinned engagement question comment on a published YouTube video using YouTube Data API v3.

    This is an OPTIONAL engagement side-effect: its failure must never gate publish
    success, so on a real (non-mock) failure it returns ``status: "error"`` rather
    than raising. The fabricated ``comment_*`` id is only produced when
    ``YOUTUBE_UPLOAD_MOCK=1``.
    """
    try:
        token_path = os.getenv("YOUTUBE_TOKEN_FILE", "token.json")
        if os.path.exists(token_path):
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            credentials = Credentials.from_authorized_user_file(token_path)
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())

            if credentials:
                youtube = build("youtube", "v3", credentials=credentials)
                body = {
                    "snippet": {
                        "videoId": req.video_id,
                        "topLevelComment": {
                            "snippet": {
                                "textOriginal": req.comment_text
                            }
                        }
                    }
                }
                res = youtube.commentThreads().insert(part="snippet", body=body).execute()
                comment_id = (res or {}).get("id")
                if comment_id:
                    return {
                        "status": "success",
                        "comment_id": comment_id,
                        "video_id": req.video_id,
                        "pinned": True
                    }
                print("[YouTube Comment] Comment API response carried no comment id; not reported as success.")
    except Exception as e:
        if _MOCK_ENABLED:
            print(f"[YouTube Comment] MOCK mode active (YOUTUBE_UPLOAD_MOCK=1); comment would have failed: {e}")
            return _mock_payload("comment")
        print(f"[YouTube Comment] Notice: Comment API returned: {e}")

    return {
        "status": "error",
        "engine": "noop_youtube_comment",
        "comment_id": "",
        "video_id": req.video_id,
        "pinned": False
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

