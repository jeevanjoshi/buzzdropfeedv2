#!/usr/bin/env python3
"""
Generate a click-worthy thumbnail for a PUBLIC YouTube video link via Google
nano-banana (gemini-2.5-flash-image), then optionally publish it with
youtube.thumbnails.set (applies only when the video belongs to the
authenticated channel).

The "analyse this link" step is done here (fetching the video's title,
description and its own frame as a visual reference), exactly like web
"analysis/generate" tools do — then Gemini generates from that context.

Usage:
    python run_nano_banana_link.py https://youtu.be/wnswgKlpLmY --aspect both
    python run_nano_banana_link.py wnswgKlpLmY --aspect 16:9 --apply
    python run_nano_banana_link.py ... --no-apply   # generate only, don't upload
"""
import os
import sys
import argparse
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.engine.nano_banana import (
    extract_video_id,
    fetch_link_video_metadata,
    generate_link_thumbnail,
    generate_bare_thumbnail,
)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Generate (and optionally set) a nano-banana thumbnail for a public YouTube video."
    )
    parser.add_argument("url", help="YouTube video URL or 11-char Video ID")
    parser.add_argument("--aspect", default="16:9", choices=["16:9", "9:16", "both"],
                        help="Thumbnail aspect ratio (16:9 long-form / 9:16 Shorts cover / both)")
    parser.add_argument("--out", default="logs/link_thumbnails", help="Output directory for generated PNGs")
    parser.add_argument("--text", default="native", choices=["native", "overlay", "none"],
                        help="native = nano-banana renders the high-CTR text INSIDE the image "
                             "(web-tool style); overlay = our PIL copy overlay; none = art only")
    parser.add_argument("--bare", action="store_true",
                        help="Bare-prompt experiment: send only 'Generate a thumbnail for this "
                             "video <link>' with the video's own frame (no analysis/text). "
                             "Overrides --text.")
    parser.add_argument("--no-apply", action="store_true",
                        help="Generate only — do NOT upload via youtube.thumbnails.set")
    args = parser.parse_args()

    video_id = extract_video_id(args.url)
    if not video_id:
        print(f"[LinkThumb] ERROR: could not parse a video id from: {args.url}", file=sys.stderr)
        return 1

    aspects = ["16:9", "9:16"] if args.aspect == "both" else [args.aspect]
    print(f"=== Nano-banana link thumbnail ===")
    print(f"Video URL/ID: {args.url} -> video_id={video_id}")

    meta = fetch_link_video_metadata(video_id)
    print(f"Title: {meta.get('title') or '(n/a)'} | frame reference: {'yes' if meta.get('thumb_bytes') else 'no'}")

    os.makedirs(args.out, exist_ok=True)
    paths = {}
    for a in aspects:
        out_path = os.path.join(args.out, f"{video_id}_{a}.png")
        if args.bare:
            saved = generate_bare_thumbnail(args.url, aspect_ratio=a, output_path=out_path)
        else:
            saved = generate_link_thumbnail(meta, aspect_ratio=a, output_path=out_path,
                                            add_text=args.text != "none",
                                            native_text=args.text == "native")
        if saved:
            paths[a] = saved
            print(f"[LinkThumb] Generated {a} thumbnail -> {saved}")

    if not paths:
        print("[LinkThumb] ERROR: no thumbnail generated (check GEMINI_API_KEY / CSVG_NANO_BANANA_THUMBNAILS).", file=sys.stderr)
        return 2

    if args.no_apply:
        print("[LinkThumb] Skipping upload (--no-apply).")
        return 0

    # Apply to the video: thumbnails.set (requires channel ownership + eligibility).
    from mcp_servers.youtube_cloud.server import _load_credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from src.engine.nano_banana import _thumbmime

    credentials = _load_credentials()
    youtube = build("youtube", "v3", credentials=credentials)
    applied = []
    for a in ("16:9", "9:16"):
        if a not in paths:
            continue
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(paths[a], mimetype=_thumbmime(paths[a])),
            ).execute()
            applied.append(a)
            print(f"[LinkThumb] Applied custom thumbnail ({a}) -> https://youtu.be/{video_id}")
        except Exception as e:
            print(f"[LinkThumb] thumbnails.set failed for {a}: {e}")
    print("[LinkThumb] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())