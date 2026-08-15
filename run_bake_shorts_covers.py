#!/usr/bin/env python3
"""
Bake the generated cover INTO the Short's first frame — the only reliable way
to control a Shorts cover (thumbnails.set is ignored by YouTube for Shorts).

For each Short (mapped to its local clip via state checkpoints):
  1. generate a 9:16 high-CTR cover (nano-banana native text + compliance),
  2. prepend a 1-second cover clip at the START of the video so the cover IS
     the first frame YouTube picks as the cover,
  3. re-encode to a single clean 9:16 MP4 (cover + original clip).

Default is dry-run (only builds the baked MP4s). Pass --upload to publish
them and --delete-original to remove the old Short after a successful upload.

Usage:
    python run_bake_shorts_covers.py            # dry-run, all mapped shorts
    python run_bake_shorts_covers.py --limit 2  # dry-run, first 2
    python run_bake_shorts_covers.py --upload --dry-run 0 --delete-original
"""
import os
import sys
import glob
import json
import asyncio
import subprocess
import argparse
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(".env")


def build_short_clip_map():
    """video_id -> local short clip path, resolved via the state checkpoints
    (upload_metadata.shorts_video_id csv is parallel to asset_paths.shorts)."""
    mapping = {}
    for fp in sorted(glob.glob("logs/state_*.json")):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        um = d.get("upload_metadata", {}) or {}
        ap = d.get("asset_paths", {}) or {}
        sv = str(um.get("shorts_video_id", "") or "")
        clips = ap.get("shorts", []) or []
        ids = [x for x in sv.split(",") if x]
        for i, vid in enumerate(ids):
            if i < len(clips) and os.path.exists(clips[i]):
                mapping[vid] = clips[i]
    return mapping


def probe(path):
    """{width, height, fps} via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "default=noprint_wrappers=1", path],
        capture_output=True, text=True, timeout=60)
    out = {"width": 0, "height": 0, "fps": 25.0}
    for line in r.stdout.splitlines():
        k, _, v = line.partition("=")
        if k in ("width", "height"):
            out[k] = int(v)
        elif k == "r_frame_rate" and "/" in v:
            num, den = v.split("/")
            out["fps"] = round(float(num) / float(den)) if float(den) else 25.0
    return out


def _cover_video(cover_jpg: str, W: int, H: int, fps: int, out_mp4: str) -> bool:
    """1-second silent cover clip at the clip's exact resolution (~first frame)."""
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", cover_jpg,
        "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo",
        "-vf", (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={fps},format=yuv420p"),
        "-t", "1", "-r", str(fps), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-shortest", out_mp4,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return r.returncode == 0 and os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 1000


def _prepend(cover_mp4: str, clip: str, W: int, H: int, fps: int, out_mp4: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", cover_mp4, "-i", clip,
        "-filter_complex",
        f"[0:v]fps={fps},settb=AVTB,scale={W}:{H},setsar=1,format=yuv420p[v0];"
        f"[1:v]fps={fps},settb=AVTB,scale={W}:{H},setsar=1,format=yuv420p[v1];"
        "[0:a]aresample=44100,asetpts=PTS-STARTPTS[a0];"
        "[1:a]aresample=44100,asetpts=PTS-STARTPTS[a1];"
        "[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]",
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", out_mp4,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print("  !! ffmpeg:", (r.stderr or "")[-200:])
    return r.returncode == 0 and os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 1000


def bake_cover_into_clip(cover_jpg: str, clip_mp4: str, out_dir: str, video_id: str) -> str:
    """Return the baked MP4 path, or '' on failure."""
    meta = probe(clip_mp4)
    W, H, fps = meta["width"], meta["height"], int(min(max(meta["fps"], 24), 30))
    if not W or not H:
        return ""
    cover_mp4 = os.path.join(out_dir, f"_cover_{video_id}.mp4")
    baked = os.path.join(out_dir, f"baked_{video_id}.mp4")
    if not _cover_video(cover_jpg, W, H, fps, cover_mp4):
        return ""
    if not _prepend(cover_mp4, clip_mp4, W, H, fps, baked):
        return ""
    return baked


async def _upload_batch(baked_paths):
    from mcp_servers.youtube_cloud.server import upload_short, UploadRequest
    results = []
    for vid, baked in baked_paths:
        res = await upload_short(UploadRequest(
            video_path=baked,
            title=f"CSVG Short (baked cover) #{vid}",
            description="",
            tags=["#Shorts"],
            category_id="22",
        ))
        results.append((vid, res))
    return results


def main():
    parser = argparse.ArgumentParser(description="Bake cover into the first frame of channel Shorts.")
    parser.add_argument("--upload", action="store_true", help="Upload the baked MP4 as a new Short")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Build baked MP4s locally without uploading (default)")
    parser.add_argument("--delete-original", action="store_true",
                        help="Delete the original Short after a successful upload")
    parser.add_argument("--limit", type=int, default=0, help="Max shorts to process (0 = all mapped)")
    parser.add_argument("--video", help="Only process this specific video id")
    parser.add_argument("--out", default="logs/baked_shorts", help="Output dir")
    parser.add_argument("--skip-cover", action="store_true", help="Reuse an existing cover if present")
    args = parser.parse_args()
    args.dry_run = not args.upload

    from src.engine import nano_banana as nb
    os.makedirs(args.out, exist_ok=True)

    mapping = build_short_clip_map()
    if not mapping:
        print("No mapped shorts found in logs/state_*.json (need shorts_video_id + asset_paths.shorts).")
        return 1
    print(f"Mapped {len(mapping)} shorts -> local clips.")
    if args.video:
        mapping = {k: v for k, v in mapping.items() if k.strip() == args.video.strip()}
    items = list(mapping.items())
    if args.limit:
        items = items[:args.limit]
    print(f"Processing {len(items)} short(s) -> {args.out}")

    baked_paths = {}
    for idx, (vid, clip) in enumerate(items, 1):
        try:
            cover = os.path.join(args.out, f"cover_{vid}.jpg")
            if not (args.skip_cover and os.path.exists(cover)):
                print(f"[{idx}/{len(items)}] generating cover for {vid}...")
                meta = nb.fetch_link_video_metadata(vid)
                out_cand = os.path.join(args.out, f"cover_{vid}.png")
                nb.generate_link_thumbnail(meta, aspect_ratio="9:16", output_path=out_cand)
                cover = out_cand if os.path.exists(out_cand) else cover
            print(f"[{idx}/{len(items)}] baking cover into {os.path.basename(clip)}...")
            baked = bake_cover_into_clip(cover, clip, args.out, vid)
            if not baked:
                print(f"  !! bake failed for {vid}")
                continue
            baked_paths[vid] = baked
            print(f"  baked -> {baked}")
        except Exception as e:
            print(f"  !! error {vid}: {e}")

    uploaded = []
    if args.upload and baked_paths:
        results = asyncio.run(_upload_batch([(v, p) for v, p in baked_paths.items()]))
        for vid, res in results:
            new_id = str(res.get("video_id", ""))
            if len(new_id) == 11:
                uploaded.append((vid, new_id))
                print(f"  UPLOADED {vid} -> https://youtu.be/{new_id}")
            else:
                print(f"  !! upload failed for {vid}: {res}")

    deleted = []
    if args.upload and args.delete_original and uploaded:
        from mcp_servers.youtube_cloud.server import _load_credentials
        from googleapiclient.discovery import build
        yt = build("youtube", "v3", credentials=_load_credentials())
        for old_vid, _new_id in uploaded:
            try:
                yt.videos().delete(id=old_vid).execute()
                deleted.append(old_vid)
                print(f"  DELETED original {old_vid}")
            except Exception as e:
                print(f"  !! could not delete {old_vid}: {e}")

    print("\n=== SUMMARY ===")
    print(f"baked: {len(baked_paths)}  uploaded: {len(uploaded)}  deleted-originals: {len(deleted)}")
    print(f"output: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    main()