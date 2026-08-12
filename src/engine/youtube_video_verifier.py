import os
import re
import sys
import subprocess
from typing import List, Dict, Any, Tuple, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from faster_whisper import WhisperModel

from src.engine.llm_client import LLMClient
from src.engine.external_apis import ExternalAPIManager

_REPO_MEDIA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/logs/media"

class YouTubeVideoVerifier:
    """
    Automated YouTube Video Sync, Coherence, and RAG-based Relevance Audit Tool.
    1. Downloads audio and retrieves transcripts/subtitles from YouTube.
    2. Identifies speech vs subtitle timestamp sync issues (drift).
    3. Evaluates script narrative coherence using LLM.
    4. Evaluates content relevance using Exa semantic search RAG fact verification.
    5. Suggests actionable fixes.
    """

    def __init__(self, model_name: str = "google/gemini-2.5-flash"):
        self.llm_client = LLMClient(model=model_name)
        self.api_manager = ExternalAPIManager()

    def _latest_local_final_video(self) -> str:
        """Resolve the most recently rendered execution-correlated final master.
        Since media_producer now writes into logs/media/<pipeline_id>/ with a
        per-run filename, find the newest final_video_*.mp4 there; fall back to
        the legacy shared /tmp path for older flows."""
        legacy = "/tmp/csvg_media/final_video_1080p.mp4"
        import glob
        candidates = glob.glob(os.path.join(_REPO_MEDIA_DIR, "**", "final_video_*.mp4"), recursive=True)
        if candidates:
            return max(candidates, key=os.path.getmtime)
        return legacy

    def _latest_local_master_subs(self) -> str:
        """Resolve the newest execution-correlated master_subtitles.ass."""
        legacy = "/tmp/csvg_media/master_subtitles.ass"
        import glob
        candidates = glob.glob(os.path.join(_REPO_MEDIA_DIR, "**", "master_subtitles.ass"), recursive=True)
        if candidates:
            return max(candidates, key=os.path.getmtime)
        return legacy
        
    def extract_video_id(self, url_or_id: str) -> str:
        """Extracts the 11-character YouTube video ID."""
        if len(url_or_id) == 11:
            return url_or_id
        
        patterns = [
            r'(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/)([a-zA-Z0-9_-]{11})',
            r'^([a-zA-Z0-9_-]{11})$'
        ]
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        return url_or_id

    def download_audio(self, video_id: str) -> str:
        """Downloads audio track of a YouTube video using yt-dlp."""
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        output_dir = "/tmp/yt_verifier"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_id}.mp3")
        
        if os.path.exists(output_path):
            print(f"[Verifier] Audio already downloaded at: {output_path}")
            return output_path
            
        print(f"[Verifier] Downloading audio for {video_id} via yt-dlp...")
        yt_dlp_bin = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
        if not os.path.exists(yt_dlp_bin):
            yt_dlp_bin = "yt-dlp"

        cmd = [
            yt_dlp_bin,
            "-x",
            "--audio-format", "mp3",
            "-o", os.path.join(output_dir, f"{video_id}.%(ext)s"),
            video_url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download audio track: {result.stderr}")
            
        print(f"[Verifier] Audio downloaded successfully to: {output_path}")
        return output_path

    def get_youtube_subtitles(self, video_id: str) -> List[Dict[str, Any]]:
        """Retrieves subtitles/captions using youtube-transcript-api."""
        print(f"[Verifier] Fetching subtitles/captions for {video_id}...")
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            # Try to fetch English, fallback to auto-generated English, or first available
            try:
                transcript = transcript_list.find_transcript(['en'])
            except Exception:
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                except Exception:
                    # Just get the first available transcript
                    transcript = next(iter(transcript_list))
                    
            data = transcript.fetch()
            print(f"[Verifier] Fetched {len(data)} subtitle entries.")
            return data
        except Exception as e:
            print(f"[Verifier] Warning: Failed to fetch captions via API ({e}). Attempting fallback with yt-dlp...")
            # Fallback to downloading auto-generated subtitles using yt-dlp
            output_dir = "/tmp/yt_verifier"
            os.makedirs(output_dir, exist_ok=True)
            vtt_pattern = os.path.join(output_dir, f"{video_id}.en.vtt")
            
            yt_dlp_bin = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
            if not os.path.exists(yt_dlp_bin):
                yt_dlp_bin = "yt-dlp"

            cmd = [
                yt_dlp_bin,
                "--write-auto-subs",
                "--sub-lang", "en",
                "--skip-download",
                "-o", os.path.join(output_dir, video_id),
                f"https://www.youtube.com/watch?v={video_id}"
            ]
            subprocess.run(cmd, capture_output=True)
            
            if os.path.exists(vtt_pattern):
                return self.parse_vtt(vtt_pattern)
            elif os.path.exists(vtt_pattern.replace(".en.vtt", ".en.vtt.vtt")):
                return self.parse_vtt(vtt_pattern.replace(".en.vtt", ".en.vtt.vtt"))
                
            raise RuntimeError(f"Could not retrieve subtitles from YouTube: {e}")

    def parse_vtt(self, filepath: str) -> List[Dict[str, Any]]:
        """Rudimentary WebVTT file parser for fallback subtitles."""
        entries = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        time_pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})'
        current_time = None
        current_text = []
        
        def vtt_to_sec(vtt_t):
            parts = vtt_t.split(":")
            h = int(parts[0])
            m = int(parts[1])
            s = float(parts[2])
            return h * 3600 + m * 60 + s
            
        for line in lines:
            line = line.strip()
            match = re.search(time_pattern, line)
            if match:
                if current_time and current_text:
                    entries.append({
                        "start": current_time[0],
                        "duration": current_time[1] - current_time[0],
                        "text": " ".join(current_text)
                    })
                start = vtt_to_sec(match.group(1))
                end = vtt_to_sec(match.group(2))
                current_time = (start, end)
                current_text = []
            elif line and not line.startswith("WEBVTT") and not line.startswith("Kind:") and not line.startswith("Language:"):
                current_text.append(line)
                
        if current_time and current_text:
            entries.append({
                "start": current_time[0],
                "duration": current_time[1] - current_time[0],
                "text": " ".join(current_text)
            })
        return entries

    def transcribe_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribes downloaded audio using faster-whisper to get actual speech timestamps."""
        print(f"[Verifier] Running faster-whisper on {audio_path}...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        speech_entries = []
        for segment in segments:
            speech_entries.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        print(f"[Verifier] Transcribed {len(speech_entries)} speech segments.")
        return speech_entries

    def check_speech_subtitle_sync(
        self, speech_segments: List[Dict[str, Any]], subtitles: List[Dict[str, Any]], max_drift: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Detects drift and mismatches by aligning subtitle text with transcription segments.
        """
        sync_issues = []
        
        for idx, sub in enumerate(subtitles):
            sub_start = sub["start"]
            sub_end = sub_start + sub.get("duration", sub.get("end", sub_start + 2.0) - sub_start)
            sub_text = sub["text"].strip().lower()
            
            # Find overlapping or closest speech segments
            best_match = None
            max_overlap = -1.0
            
            for speech in speech_segments:
                overlap = min(sub_end, speech["end"]) - max(sub_start, speech["start"])
                if overlap > 0 and overlap > max_overlap:
                    max_overlap = overlap
                    best_match = speech
            
            # If no direct overlap, look for the closest segment within 2 seconds
            if not best_match:
                closest = None
                min_dist = 2.0
                for speech in speech_segments:
                    dist = min(abs(sub_start - speech["start"]), abs(sub_end - speech["end"]))
                    if dist < min_dist:
                        min_dist = dist
                        closest = speech
                best_match = closest
                
            if best_match:
                drift = abs(sub_start - best_match["start"])
                
                # Check for sync drift
                if drift > max_drift:
                    sync_issues.append({
                        "type": "drift",
                        "subtitle_index": idx,
                        "text": sub["text"],
                        "subtitle_start": round(sub_start, 2),
                        "speech_start": round(best_match["start"], 2),
                        "drift": round(drift, 2),
                        "severity": "HIGH" if drift > 1.5 else "MEDIUM",
                        "description": f"Subtitle starts at {sub_start:.2f}s but speech occurs at {best_match['start']:.2f}s (Drift: {drift:.2f}s)"
                    })
                
                # Check for text/word mismatch
                sub_words = set(re.findall(r'\b\w{3,}\b', sub_text))
                speech_words = set(re.findall(r'\b\w{3,}\b', best_match["text"].lower()))
                
                if sub_words and speech_words:
                    mismatched_words = sub_words - speech_words
                    mismatch_rate = len(mismatched_words) / len(sub_words)
                    
                    if mismatch_rate > 0.4:  # More than 40% discrepancy in key words
                        sync_issues.append({
                            "type": "mismatch",
                            "subtitle_index": idx,
                            "text": sub["text"],
                            "speech_text": best_match["text"],
                            "mismatch_rate": round(mismatch_rate, 2),
                            "severity": "HIGH" if mismatch_rate > 0.7 else "MEDIUM",
                            "description": f"Subtitle text '{sub['text']}' differs significantly from transcribed speech '{best_match['text']}'"
                        })
            else:
                sync_issues.append({
                    "type": "missing_speech",
                    "subtitle_index": idx,
                    "text": sub["text"],
                    "subtitle_start": round(sub_start, 2),
                    "severity": "HIGH",
                    "description": f"Subtitle displayed but no spoken speech detected in that time window ({sub_start:.2f}s - {sub_end:.2f}s)"
                })
                
        return sync_issues

    def analyze_coherence_and_relevance(
        self, full_transcript: str, video_title: str, description: str, seo_tags: List[str], query_topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Uses Exa for RAG context extraction and LLM for coherence + relevance analysis of the video text, title, description, and tags.
        """
        topic = query_topic or video_title
        print(f"[Verifier] Fetching RAG grounding data via Exa for topic: {topic}...")
        
        # 1. Fetch facts from Exa
        facts = self.api_manager.fetch_exa_semantic_facts(topic)
        rag_context = "\n".join([f"- {f.headline}: {f.summary} ({f.url})" for f in facts])
        
        # 2. Perform LLM Evaluation
        system_prompt = (
            "You are a Senior Video Editor, SEO Marketer, and Content Quality Auditor. "
            "Your job is to audit a video transcript/script for script coherence, logical flow, "
            "and relevance to the main topic, and also evaluate if the video title, description, "
            "and SEO tags are accurately aligned with the transcript and grounding RAG context facts."
        )
        
        tags_str = ", ".join(seo_tags)
        prompt = f"""
Audit the following YouTube Video content and metadata:
---
VIDEO TITLE: {video_title}
VIDEO DESCRIPTION: {description}
SEO TAGS: {tags_str}
TARGET TOPIC: {topic}
---
RAG GROUNDING FACTS (USE AS SOURCE OF TRUTH):
{rag_context}
---
TRANSCRIPT CONTENT:
{full_transcript[:6000]} # Truncate if too long
---
Please perform:
1. Script Coherence Check: Does the script have a smooth narrative arc? Are there logical leaps, awkward sentence transitions, or disjointed topics?
2. Relevance Audit: Identify any sentences, paragraphs, or segments that are irrelevant, off-topic, containing repetitive template slop, or diverging into fluff. Refer back to the target topic and RAG Grounding Facts.
3. Metadata & SEO Alignment: Are the Title, Description, and SEO tags relevant to the actual spoken transcript and topic? Suggest optimizations.
4. Fix Suggestions: Detail exact corrections, deletions, or structural modifications to make the script, title, description, and tags flow seamlessly and stay completely on-topic.

Return the response as a valid JSON object matching this schema:
{{
  "coherence_score": 0.0 to 10.0,
  "coherence_findings": ["finding 1", "finding 2"],
  "relevance_score": 0.0 to 10.0,
  "irrelevant_segments": [
    {{
      "text": "the exact text of the irrelevant segment",
      "reason": "why it is irrelevant or off-topic compared to target/RAG facts",
      "suggested_fix": "suggested deletion/rewrite"
    }}
  ],
  "metadata_seo_issues": [
    {{
      "field": "Title / Description / Tags",
      "issue": "the discrepancy found",
      "suggested_fix": "suggested correction"
    }}
  ],
  "structural_fixes": ["general fix 1", "general fix 2"]
}}
"""
        result = self.llm_client.generate_json(prompt, system_prompt)
        if not result:
            # Fallback
            result = {
                "coherence_score": 5.0,
                "coherence_findings": ["Could not parse LLM analysis response."],
                "relevance_score": 5.0,
                "irrelevant_segments": [],
                "metadata_seo_issues": [],
                "structural_fixes": ["Verify API connection and retry audit."]
            }
        return result

    def _build_youtube_client(self):
        """Builds an authenticated YouTube API client from stored token.json."""
        token_path = os.getenv("YOUTUBE_TOKEN_FILE", "token.json")
        if not os.path.exists(token_path):
            return None
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            creds = Credentials.from_authorized_user_file(token_path)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            if not creds or not creds.valid:
                return None
            return build("youtube", "v3", credentials=creds)
        except Exception as e:
            print(f"[Verifier] Failed to build YouTube client: {e}")
            return None

    def parse_ass(self, filepath: str) -> List[Dict[str, Any]]:
        """Parses ASS subtitle file and returns standard format list."""
        entries = []
        def to_secs(t_str):
            h, m, s = t_str.strip().split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)

        if not os.path.exists(filepath):
            return []

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('Dialogue:'):
                    parts = line.split(',', 9)
                    if len(parts) == 10:
                        start = to_secs(parts[1])
                        end = to_secs(parts[2])
                        text = re.sub(r'{[^}]+}', '', parts[9]).strip() # Strip style tags
                        entries.append({
                            "start": start,
                            "duration": end - start,
                            "text": text
                        })
        return entries

    def verify_video(self, url_or_id: str, target_topic: Optional[str] = None) -> Dict[str, Any]:
        """Runs the full verification pipeline on a YouTube video."""
        video_id = self.extract_video_id(url_or_id)
        
        # Get metadata using token-auth (Google API) or yt-dlp
        video_title = f"YouTube Video {video_id}"
        description = ""
        seo_tags = []
        is_private = False
        
        youtube = self._build_youtube_client()
        if youtube:
            try:
                print(f"[Verifier] Stored token.json found. Fetching video {video_id} metadata via YouTube API...")
                video_resp = youtube.videos().list(
                    part="snippet,status",
                    id=video_id
                ).execute()
                items = video_resp.get("items", [])
                if items:
                    v = items[0]
                    video_title = v["snippet"].get("title", video_title)
                    description = v["snippet"].get("description", "")
                    seo_tags = v["snippet"].get("tags", [])
                    is_private = v["status"].get("privacyStatus") == "private"
                    print(f"[Verifier] Successfully retrieved metadata via API. Private={is_private}")
            except Exception as e:
                print(f"[Verifier] Failed to retrieve metadata via API: {e}")
        
        if not is_private:
            # Try to get metadata using yt-dlp if API wasn't used or metadata empty
            try:
                print("[Verifier] Fetching video metadata (title, description, tags)...")
                meta_cmd = ["./venv/bin/yt-dlp", "--dump-json", f"https://www.youtube.com/watch?v={video_id}"]
                res = subprocess.run(meta_cmd, capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    import json
                    meta = json.loads(res.stdout.strip())
                    video_title = meta.get("title", video_title)
                    description = meta.get("description", "")
                    seo_tags = meta.get("tags", [])
            except Exception as e:
                print(f"[Verifier] Failed to retrieve rich metadata: {e}")
            
        # 1. Fetch subtitle/captions track from YouTube
        subtitles = []
        used_local_subtitles = False
        try:
            subtitles = self.get_youtube_subtitles(video_id)
        except Exception as e:
            print(f"[Verifier] Failed to retrieve captions from YouTube: {e}")
            # Fallback to local subtitle file if private video
            local_ass = self._latest_local_master_subs()
            if os.path.exists(local_ass):
                print(f"[Verifier] Falling back to local master subtitle file: {local_ass}")
                subtitles = self.parse_ass(local_ass)
                used_local_subtitles = True
            else:
                return {"status": "error", "message": f"Failed to retrieve subtitles: {e}"}
            
        if not subtitles:
            return {"status": "error", "message": "No subtitles/captions could be found or parsed."}

        # Compile full transcript text for coherence/RAG analysis
        full_transcript = " ".join([s["text"] for s in subtitles])
        
        # 2. Download and transcribe audio to find actual spoken word timings
        sync_issues = []
        try:
            # Resolve the latest execution-correlated final master (falls back to
            # the legacy shared /tmp path for backward compatibility).
            local_video = self._latest_local_final_video()
            if is_private and local_video and os.path.exists(local_video):
                print(f"[Verifier] Video is private. Bypassing yt-dlp download, using local file: {local_video}")
                audio_path = local_video
            else:
                try:
                    audio_path = self.download_audio(video_id)
                except Exception as dl_err:
                    if local_video and os.path.exists(local_video):
                        print(f"[Verifier] yt-dlp download failed ({dl_err}). Falling back to local file: {local_video}")
                        audio_path = local_video
                    else:
                        raise dl_err

            speech_segments = self.transcribe_audio(audio_path)
            # Compare and check sync
            sync_issues = self.check_speech_subtitle_sync(speech_segments, subtitles)
        except Exception as e:
            print(f"[Verifier] Error running audio sync check: {e}")
            sync_issues = [{"type": "error", "description": f"Could not perform audio sync check: {e}", "severity": "HIGH"}]
            
        # 3. Analyze coherence and RAG relevance
        semantic_report = self.analyze_coherence_and_relevance(
            full_transcript=full_transcript,
            video_title=video_title,
            description=description,
            seo_tags=seo_tags,
            query_topic=target_topic
        )
        
        # 4. Synthesize overall fixes/report
        report = {
            "video_id": video_id,
            "video_title": video_title,
            "description": description,
            "seo_tags": seo_tags,
            "target_topic": target_topic or video_title,
            "subtitle_count": len(subtitles),
            "used_local_subtitles": used_local_subtitles,
            "sync_analysis": {
                "total_issues": len(sync_issues),
                "issues": sync_issues
            },
            "content_analysis": semantic_report
        }
        return report
