"""
SURGICAL re-render of the Meta video (`csvg-exec-20260810-113142`).

Fixes ONLY the corrupted Shot 18 narration (scraped byline/hashtag/tagline junk
appended after the real outro question) and then reassembles the master:

1. Clean shot 18 narration  -> everything before "# Meta launches new ...".
2. Re-synthesize shot 18 TTS on the Pi (AUDIO_EDGE_URL) and download the wav.
3. Peak-limit the wav (same as the production limiter).
4. Re-render shot_18.mp4 with the CACHED visual (visual cache) + new audio via
   Ken Burns (same params as media_producer: direction by shot parity, tail hold,
   measured duration).
5. Rebuild the master subtitle .ass (merge all shot .ass at measured offsets).
6. Reassemble the final master (ffmpeg timeline with crossfade + BGM).
7. Regenerate the thumbnail (applies the new adaptive text-contrast fix).

Because Shot 18 is the LAST shot, re-rendering it does not shift the timing of
shots 1-17; only shot 18's duration and the master total change.

Run from repo root with the venv activated. Requires reachable AUDIO_EDGE_URL.
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys

# So `mcp_servers`, `src`, `resources` resolve when run as `scripts/...` (repo root).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mcp_servers.audio_edge.server import sanitize_tts_text, TTSRequest, WhisperRequest
from mcp_servers.audio_edge.server import synthesize_tts, align_subtitles_whisper
from mcp_servers.media_cloud.server import (
    apply_ken_burns_motion, assemble_ffmpeg_timeline, generate_thumbnail,
    KenBurnsRequest, TimelineAssemblyRequest, ThumbnailRequest,
)
from src.agents.media_producer import (
    _peak_limit_wav, _probe_wav_duration, _probe_duration,
    merge_ass_subtitle_files, _repo_root, _visual_cache_key, _visual_cache_dir,
    MIN_SHOT_DUR,
)

PIPELINE = "csvg-exec-20260810-113142"
STATE_FILE = f"logs/state_{PIPELINE}.json"
SCRIPT_FILE = f"logs/script_{PIPELINE}.json"
MEDIA_DIR = f"logs/media/{PIPELINE}"
TAIL = float(os.getenv("CSVG_PAD_AFTER_NARRATION", "1.2"))
CROSSFADE = 0.5
TRANSITION = "fade"
JUNK_MARKER = "# Meta launches new artificial intelligence model as Zuckerberg champions open-weight push."
BGM = os.path.join(_repo_root(), "resources", "bgm.mp3")


def load_state():
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def clean_shot18(narration: str) -> str:
    idx = narration.find(JUNK_MARKER)
    if idx != -1:
        return narration[:idx].strip()
    # Fallback: cut at the first standalone '#' tagline after 'comments below'.
    m = re.search(r"(#\s+[A-Z])", narration)
    if m:
        return narration[: m.start()].strip()
    return narration.strip()


async def make_tts_and_subs(text, wav_path, ass_path, tts_speed=0.94):
    """Synthesize TTS on the Pi and download wav + aligned subtitles."""
    audio_edge_url = os.getenv("AUDIO_EDGE_URL")
    import aiohttp
    async with aiohttp.ClientSession() as session:
        # synthesize on the Pi (its _resolve_local_path maps /home/ubuntu -> /home/jeevanjoshi)
        async with session.post(f"{audio_edge_url}/tools/synthesize_tts", json={
            "text": text, "output_path": wav_path, "region": "all", "speed": tts_speed,
        }, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"TTS failed: {await resp.text()}")
        # download wav
        for attempt in range(3):
            async with session.get(f"{audio_edge_url}/files", params={"path": wav_path},
                                   timeout=aiohttp.ClientTimeout(total=300)) as r:
                if r.status == 200:
                    with open(wav_path, "wb") as f:
                        f.write(await r.read())
                    break
            await asyncio.sleep(1.0)
        else:
            raise RuntimeError("TTS wav download failed after 3 attempts")
        # align subtitles on the Pi
        async with session.post(f"{audio_edge_url}/tools/align_subtitles_whisper", json={
            "audio_path": wav_path, "output_ass_path": ass_path, "original_text": text,
        }, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Subtitle align failed: {await resp.text()}")
        for attempt in range(3):
            async with session.get(f"{audio_edge_url}/files", params={"path": ass_path},
                                   timeout=aiohttp.ClientTimeout(total=300)) as r:
                if r.status == 200:
                    with open(ass_path, "wb") as f:
                        f.write(await r.read())
                    break
            await asyncio.sleep(1.0)
        else:
            raise RuntimeError("ASS download failed after 3 attempts")


async def rebuild_shot18(state):
    script = state["script_data"]
    shot18 = next(s for s in script["shots"] if s["shot_id"] == 18)
    raw = shot18["narration_text"]
    clean = clean_shot18(raw)
    print(f"[surgical] Shot 18 narration {len(raw.split())} -> {len(clean.split())} words")

    shots = script["shots"]
    key18 = "shot_18"
    wav_path = os.path.join(MEDIA_DIR, "audio", f"{key18}.wav")
    ass_path = os.path.join(MEDIA_DIR, "subtitles", f"{key18}.ass")
    img_path = os.path.join(MEDIA_DIR, "visuals", f"{key18}.png")
    mp4_path = os.path.join(MEDIA_DIR, "visuals", f"{key18}.mp4")

    tts_narration = sanitize_tts_text(clean)
    # Shot 18 is act 6 (verdict) -> slowed, deliberate.
    tts_speed = 0.94 if shot18.get("act_index") == 6 else 1.0

    if not os.path.exists(img_path):
        raise RuntimeError(f"Shot 18 visual missing: {img_path}")

    await make_tts_and_subs(tts_narration, wav_path, ass_path, tts_speed)
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
        raise RuntimeError(f"TTS wav missing/small: {wav_path}")
    if not os.path.exists(ass_path) or os.path.getsize(ass_path) < 50:
        raise RuntimeError(f"ASS missing/small: {ass_path}")
    _peak_limit_wav(wav_path)

    audio_dur = _probe_wav_duration(wav_path)
    if not audio_dur:
        audio_dur = max(shot18.get("duration_estimate", 44.0), MIN_SHOT_DUR)
    shot_timeline_dur = max(audio_dur + TAIL, MIN_SHOT_DUR)
    print(f"[surgical] shot18 audio_dur={audio_dur:.2f}s + tail {TAIL} -> timeline {shot_timeline_dur:.2f}s")

    ken_direction = "left_to_right" if shot18.get("shot_id") % 2 == 1 else "right_to_left"
    await apply_ken_burns_motion(KenBurnsRequest(
        image_path=img_path, audio_path=wav_path,
        duration=shot_timeline_dur, output_mp4_path=mp4_path,
        direction=ken_direction,
    ))
    real_dur = _probe_duration(mp4_path) or shot_timeline_dur
    print(f"[surgical] shot18.mp4 rebuilt, real_dur={real_dur:.2f}s")

    # ---- Reassemble master ----
    # Measure all shots' real durations as they stand on disk.
    durs = []
    for s in shots:
        p = os.path.join(MEDIA_DIR, "visuals", f"shot_{s['shot_id']}.mp4")
        durs.append(max(_probe_duration(p) or MIN_SHOT_DUR, MIN_SHOT_DUR))

    # master subtitles
    ass_paths = [os.path.join(MEDIA_DIR, "subtitles", f"shot_{s['shot_id']}.ass") for s in shots]
    master_sub = os.path.join(MEDIA_DIR, "master_subtitles.ass")
    merge_ass_subtitle_files(ass_paths, durs, master_sub, crossfade=CROSSFADE)

    # concat list
    concat_list = os.path.join(MEDIA_DIR, "concat_list.txt")
    concat_lines = []
    for s in shots:
        _p = os.path.join(MEDIA_DIR, "visuals", "shot_%s.mp4" % s["shot_id"])
        concat_lines.append(f"file '{_p}'")
    with open(concat_list, "w") as f:
        f.write("\n".join(concat_lines))

    bgm_dst = os.path.join(MEDIA_DIR, "bgm.mp3")
    if os.path.exists(BGM) and os.path.getsize(BGM) > 1000:
        shutil.copy2(BGM, bgm_dst)
    else:
        with open(bgm_dst, "w") as f:
            f.write("DUMMY_BGM")

    final_video = os.path.join(MEDIA_DIR, f"final_video_{PIPELINE}.mp4")
    await assemble_ffmpeg_timeline(TimelineAssemblyRequest(
        concat_list_path=concat_list, subtitle_path=master_sub,
        bgm_path=bgm_dst, output_video_path=final_video,
        crossfade=CROSSFADE, transition=TRANSITION,
    ))
    print(f"[surgical] final master rebuilt -> {final_video}")

    # ---- regenerate thumbnail (new contrast fix) ----
    thumb = os.path.join(MEDIA_DIR, f"thumbnail_{PIPELINE}.png")
    brief = state.get("seo_metadata") or {}
    headline = brief.get("thumbnail_brief") or (state.get("selected_topic") or {}).get("headline", "Meta")
    hero = None
    if shots:
        hero = shots[0].get("visual_prompt")
    await generate_thumbnail(ThumbnailRequest(
        headline_text=headline, visual_prompt=hero, output_thumbnail_path=thumb,
    ))
    print(f"[surgical] thumbnail regenerated -> {thumb}")

    # ---- update state ----
    state["script_data"]["shots"][17]["narration_text"] = clean
    asset = state["asset_paths"]
    asset["final_video"] = final_video
    asset["thumbnail"] = thumb
    asset["measured_durations"] = [round(x, 3) for x in durs]
    asset["crossfade_used"] = CROSSFADE
    if state.get("upload_metadata"):
        state["upload_metadata"] = {"video_id": "", "status": "PENDING", "retry_count": 0, "synthetic_content_flag": True}
    save_state(state)
    print("[surgical] state updated. DONE.")


async def main():
    state = load_state()
    if not os.path.exists(MEDIA_DIR):
        print(f"ERROR: media dir missing: {MEDIA_DIR}")
        return 1
    await rebuild_shot18(state)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
