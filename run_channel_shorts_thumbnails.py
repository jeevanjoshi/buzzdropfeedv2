#!/usr/bin/env python3
"""
Generate YouTube-guideline-compliant thumbnails for ALL Shorts on the channel
and upload them with youtube.thumbnails.set.

Uses the SAME OAuth upload auth (token.json, must include youtube.force-ssl to
read captions). For each Short it:
  1. lists the channel's videos via the uploads playlist,
  2. keeps Shorts (duration <= 60s),
  3. fetches the video's title + description + OAuth caption transcript + cover
     frame (best-effort; yt-dlp frame download is IP-blocked on OCI),
  4. derives a thematic 2-5 word hook + click-worthy art via nano-banana
     (native model-rendered text, design rules enforced),
  5. runs the compliance pass (exact 1080x1920, brightness, JPEG <=2MB),
  6. uploads the thumbnail with youtube.thumbnails.set.

SAFETY: default is --dry-run (lists + generates but does NOT upload). Pass
--upload to actually push thumbnails. Use --limit to cap the batch and
--skip-generated to reuse files already produced.

Usage:
    python run_channel_shorts_thumbnails.py --dry-run --limit 3
    python run_channel_shorts_thumbnails.py --upload --limit 10
    python run_channel_shorts_thumbnails.py --upload --all
"""
import os
import sys
import re
import time
import json
import argparse
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(".env")


def _client():
    from mcp_servers.youtube_cloud.server import _load_credentials
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=_load_credentials())


def list_channel_video_ids(youtube, cap=1000):
    """All video ids from the channel's uploads playlist."""
    ch = youtube.channels().list(part="contentDetails", mine=True).execute()
    if not ch.get("items"):
        return []
    uploads = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    ids = []
    page = None
    while True:
        res = youtube.playlistItems().list(
            part="contentDetails", playlistId=uploads, maxResults=50, pageToken=page
        ).execute()
        for it in res.get("items", []):
            ids.append(it["contentDetails"]["videoId"])
        page = res.get("nextPageToken")
        if not page or len(ids) >= cap:
            break
    return ids


def _iso_to_seconds(dur_iso: str) -> int:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(\d+)S", dur_iso or "PT0S")
    if not m:
        return 0
    return int(m.group(3) or 0) + 60 * int(m.group(2) or 0) + 3600 * int(m.group(1) or 0)


def fetch_durations(youtube, ids):
    out = {}
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        res = youtube.videos().list(part="contentDetails", id=",".join(batch)).execute()
        for it in res.get("items", []):
            out[it["id"]] = _iso_to_seconds(it["contentDetails"]["duration"])
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate + upload thumbnails for all channel Shorts.")
    parser.add_argument("--upload", action="store_true",
                        help="Actually upload via thumbnails.set (default is dry-run, no upload)")
    parser.add_argument("--all", action="store_true", help="All videos (not just <=60s Shorts)")
    parser.add_argument("--limit", type=int, default=0, help="Cap number of videos processed (0 = unlimited)")
    parser.add_argument("--out", default="logs/channel_thumbnails", help="Output dir for generated JPEGs")
    parser.add_argument("--skip-generated", action="store_true",
                        help="Skip Shorts whose thumbnail JPEG already exists on disk")
    args = parser.parse_args()

    from src.engine import nano_banana as nb

    os.makedirs(args.out, exist_ok=True)
    youtube = _client()

    print("Listing channel videos (uploads playlist)...")
    ids = list_channel_video_ids(youtube)
    print(f"Total channel videos: {len(ids)}")
    if not ids:
        print("No videos found (check token scopes include youtube.readonly).")
        return 1

    durs = fetch_durations(youtube, ids)
    if args.all:
        short_ids = ids
        print("Including ALL videos (--all).")
    else:
        short_ids = [v for v in ids if durs.get(v, 61) <= 60]
        print(f"Shorts (<=60s): {len(short_ids)} of {len(ids)} videos.")

    if args.limit:
        short_ids = short_ids[:args.limit]
    print(f"Processing {len(short_ids)} video(s)...")

    done, failed, uploaded = [], [], []
    for idx, vid in enumerate(short_ids, 1):
        dur = durs.get(vid, 0)
        out_path = os.path.join(args.out, f"{vid}_9:16.jpg")
        if args.skip_generated and os.path.exists(out_path):
            print(f"[{idx}/{len(short_ids)}] skip (exists): {vid}")
            done.append(vid)
            continue
        try:
            print(f"[{idx}/{len(short_ids)}] generating {vid} ({dur}s)...")
            meta = nb.fetch_link_video_metadata(vid)
            saved = nb.generate_link_thumbnail(meta, aspect_ratio="9:16", output_path=out_path)
            if not saved:
                print(f"  !! generation failed for {vid}")
                failed.append(vid)
                continue
            done.append(vid)
            if args.upload:
                from googleapiclient.http import MediaFileUpload
                youtube.thumbnails().set(
                    videoId=vid,
                    media_body=MediaFileUpload(saved, mimetype=nb._thumbmime(saved)),
                ).execute()
                uploaded.append(vid)
                print(f"  UPLOADED -> https://youtu.be/{vid}")
            else:
                print(f"  generated (dry-run, no upload): {saved}")
        except Exception as e:
            print(f"  !! error {vid}: {e}")
            failed.append(vid)

    print("\n=== SUMMARY ===")
    print(f"generated: {len(done)}  failed: {len(failed)}  uploaded: {len(uploaded)}")
    print(f"output dir: {os.path.abspath(args.out)}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())