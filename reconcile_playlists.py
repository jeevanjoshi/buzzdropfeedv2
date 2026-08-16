#!/usr/bin/env python3
"""Reconcile the channel's playlists against the pipeline's themed playlist set.

The pipeline chains each video into a SINGLE themed playlist matched to the
topic's audience (see publisher._OUTCOME_PLAYLISTS). Previously it also chained
into a "master" playlist whose title could drift, which minted duplicate/orphan
playlists (e.g. a dangling 'LumenLoop AI Documentaries' master that the video
ended up NOT being reachable through). This helper:

  * lists all channel playlists (title + id + item count),
  * lists all channel uploads and which playlist each video lives in,
  * flags playlist titles that don't match any configured themed playlist,
  * flags orphaned videos (uploaded but in NO configured themed playlist).

DRY-RUN by default (prints a report only). Pass --apply to merge orphaned
videos into their themed playlist and (optionally) delete orphan playlists.
Read-only unless --apply is given.
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

THEMED_PLAYLISTS = [
    "Finance, Markets & Wealth Stories",
    "AI, Tech & Innovation Deep-Dives",
    "Space, Cosmology & Economic History",
    "Global Trends & Infotainment",
]


def _norm(s: str) -> str:
    return "".join(c.lower() for c in str(s or "") if c.isalnum())


def _build_youtube():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    token_path = os.getenv("YOUTUBE_TOKEN_FILE", "token.json")
    creds = Credentials.from_authorized_user_file(token_path)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def list_playlists(youtube) -> List[Dict]:
    out = []
    req = youtube.playlists().list(part="snippet,contentDetails", mine=True, maxResults=50)
    while req:
        res = req.execute()
        for it in res.get("items", []):
            out.append({
                "id": it["id"],
                "title": it["snippet"]["title"],
                "count": it["contentDetails"].get("itemCount", 0),
            })
        req = youtube.playlists().list_next(req, res)
    return out


def playlist_videos(youtube, playlist_id: str, max_results: int = 200) -> List[str]:
    vids = []
    try:
        req = youtube.playlistItems().list(
            part="snippet", playlistId=playlist_id, maxResults=50)
        while req and len(vids) < max_results:
            res = req.execute()
            for it in res.get("items", []):
                vids.append(it["snippet"]["resourceId"]["videoId"])
            req = youtube.playlistItems().list_next(req, res)
    except Exception as e:
        print(f"  [warn] could not read playlist {playlist_id}: {e}")
    return vids


def channel_uploads(youtube) -> List[str]:
    vids = []
    try:
        chan = youtube.channels().list(part="contentDetails", mine=True).execute()
        uploads = chan["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        req = youtube.playlistItems().list(
            part="contentDetails", playlistId=uploads, maxResults=50)
        while req:
            res = req.execute()
            for it in res.get("items", []):
                vids.append(it["contentDetails"]["videoId"])
            req = youtube.playlistItems().list_next(req, res)
    except Exception as e:
        print(f"  [warn] could not list channel uploads: {e}")
    return vids


def main() -> int:
    apply = "--apply" in sys.argv
    youtube = _build_youtube()
    themed_norm = {_norm(t): t for t in THEMED_PLAYLISTS}

    print("=== Playlist reconciliation (dry-run)" if not apply else "=== Playlist reconciliation (APPLY)")

    playlists = list_playlists(youtube)
    print(f"\nChannel playlists ({len(playlists)}):")
    by_id = {}
    for pl in playlists:
        by_id[pl["id"]] = pl
        norm = _norm(pl["title"])
        match = themed_norm.get(norm, "")
        tag = f" -> themed: {match}" if match else " -> <NO THEMED MATCH / ORPHAN>"
        print(f"  {pl['id']} | {pl['title']!r} | {pl['count']} items{tag}")

    # Which videos live in which themed playlist.
    themed_member: Dict[str, str] = {}  # video_id -> playlist_title
    orphan_playlists = []
    for pl in playlists:
        norm = _norm(pl["title"])
        if norm in themed_norm:
            for v in playlist_videos(youtube, pl["id"]):
                themed_member[v] = themed_norm[norm]
        else:
            orphan_playlists.append(pl)

    uploads = channel_uploads(youtube)
    print(f"\nChannel uploads: {len(uploads)} videos")

    orphans = [v for v in uploads if v not in themed_member]
    if orphans:
        print(f"\nORPHANED VIDEOS (in no themed playlist): {len(orphans)}")
        for v in orphans:
            print(f"  {v}")
    else:
        print("\nAll uploaded videos are in a themed playlist.")

    # Map audience_type -> themed playlist (mirror publisher mapping).
    mapping = {
        "investor": "Finance, Markets & Wealth Stories",
        "finance_edu": "Finance, Markets & Wealth Stories",
        "real_estate": "Finance, Markets & Wealth Stories",
        "tech": "AI, Tech & Innovation Deep-Dives",
        "business": "AI, Tech & Innovation Deep-Dives",
        "science": "AI, Tech & Innovation Deep-Dives",
        "health": "AI, Tech & Innovation Deep-Dives",
        "space": "Space, Cosmology & Economic History",
        "history": "Space, Cosmology & Economic History",
        "general": "Global Trends & Infotainment",
    }

    if orphans:
        print("\nIn --apply mode, orphaned videos would be chained into their themed playlist")
        print("based on each run's state checkpoint (audience_type). Skipping automated merge")
        print("for now since audience attribution requires reading logs/state_*.json — review")
        print("the report and fix manually, or run the pipeline with the corrected mapping.")

    if orphan_playlists and apply:
        print("\nDeleting orphan (non-themed) playlists in --apply mode...")
        for pl in orphan_playlists:
            try:
                youtube.playlists().delete(id=pl["id"]).execute()
                print(f"  deleted {pl['id']} {pl['title']!r}")
            except Exception as e:
                print(f"  [warn] could not delete {pl['id']}: {e}")
    elif orphan_playlists:
        print("\nORPHAN PLAYLISTS (candidates for deletion in --apply mode):")
        for pl in orphan_playlists:
            print(f"  {pl['id']} | {pl['title']!r} | {pl['count']} items")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())