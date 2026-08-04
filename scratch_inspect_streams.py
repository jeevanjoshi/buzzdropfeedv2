import os
import sys
import json
import subprocess

def run_ffprobe(file_path):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", file_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr}")
    return json.loads(res.stdout)

def main():
    video_path = "/tmp/csvg_media/final_video_1080p.mp4"
    bgm_path = "/tmp/csvg_media/bgm.mp3"
    subs_path = "/tmp/csvg_media/master_subtitles.ass"

    print("=== Media File Diagnostics ===")
    
    # 1. Check Subtitles File
    if os.path.exists(subs_path):
        size = os.path.getsize(subs_path)
        print(f"[Subtitles] master_subtitles.ass exists ({size} bytes).")
        # Read first 5 dialogue lines
        with open(subs_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        dialogues = [l.strip() for l in lines if l.startswith("Dialogue:")]
        print(f"[Subtitles] Total Dialogue lines: {len(dialogues)}")
        print("[Subtitles] First 3 lines sample:")
        for d in dialogues[:3]:
            print(f"  {d}")
    else:
        print("[Subtitles] ERROR: master_subtitles.ass missing!")

    # 2. Check BGM
    if os.path.exists(bgm_path):
        size = os.path.getsize(bgm_path)
        print(f"[Music] bgm.mp3 exists ({size} bytes).")
    else:
        print("[Music] ERROR: bgm.mp3 missing!")

    # 3. Inspect final video streams with ffprobe
    if os.path.exists(video_path):
        size = os.path.getsize(video_path)
        print(f"[Video] final_video_1080p.mp4 exists ({size / (1024*1024):.2f} MB).")
        try:
            info = run_ffprobe(video_path)
            format_info = info.get("format", {})
            streams = info.get("streams", [])
            
            print(f"[Video] Format: {format_info.get('format_long_name')}")
            print(f"[Video] Total Duration: {float(format_info.get('duration', 0.0)):.2f}s")
            
            for i, stream in enumerate(streams):
                codec_type = stream.get("codec_type")
                codec_name = stream.get("codec_name")
                print(f"  Stream #{i}: {codec_type} ({codec_name})")
                if codec_type == "video":
                    print(f"    Resolution: {stream.get('width')}x{stream.get('height')}")
                    print(f"    Frame rate: {stream.get('r_frame_rate')}")
                elif codec_type == "audio":
                    print(f"    Channels: {stream.get('channels')}")
                    print(f"    Sample rate: {stream.get('sample_rate')} Hz")
        except Exception as e:
            print(f"[Video] ERROR analyzing video: {e}")
    else:
        print("[Video] ERROR: final_video_1080p.mp4 missing!")

    # 4. Check TTS individual files count
    audio_dir = "/tmp/csvg_media/audio"
    if os.path.exists(audio_dir):
        files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
        print(f"[TTS] Total shot audio files (.wav): {len(files)}")
    else:
        print("[TTS] ERROR: audio directory missing!")

if __name__ == "__main__":
    main()
