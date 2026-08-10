import os
import re
import wave
import struct
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


app = FastAPI(title="Pi 5 Edge Audio & STT MCP Server")


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_bella(2)+af_heart(1)"
    region: str = "all"
    output_path: str
    speed: float = 1.0

from typing import Optional
import difflib


class WhisperRequest(BaseModel):
    audio_path: str
    output_ass_path: str
    original_text: Optional[str] = None


def sanitize_tts_text(text: str) -> str:
    """
    Strips raw SSML tags, break commands, slashes, and formatting artifacts
    so the TTS model never speaks garbage like 'break time 300 ms forward slash'.
    """
    # Expand currencies first, e.g. $100 -> 100 dollars, $520.4 Billion -> 520.4 Billion dollars
    text = re.sub(r'\$(\d+(?:,\d+)*(?:\.\d+)?)\s*(billion|million|trillion|usd)?\b', 
                  lambda m: f"{m.group(1).replace(',', '')} {m.group(2) if m.group(2) else ''} dollars".strip().replace("  ", " "), 
                  text, flags=re.IGNORECASE)
    text = re.sub(r'\$(\d+)', r'\1 dollars', text)

    # Expand large numbers to text (e.g. 5000000 -> 5 million)
    def num_repl(match):
        num_str = match.group(1).replace(",", "")
        val = int(num_str)
        if val >= 1_000_000_000 and val % 1_000_000 == 0:
            billions = val / 1_000_000_000
            return f"{int(billions) if billions.is_integer() else billions:.1f} billion"
        elif val >= 1_000_000 and val % 10_000 == 0:
            millions = val / 1_000_000
            return f"{int(millions) if millions.is_integer() else millions:.1f} million"
        elif val >= 1_000 and val % 100 == 0:
            thousands = val / 1_000
            return f"{int(thousands) if thousands.is_integer() else thousands:.1f} thousand"
        return match.group(1)
    
    text = re.sub(r'\b(\d{1,3}(?:,\d{3})+|\d{4,15})\b', num_repl, text)

    # Remove HTML / SSML tags e.g. <break time="300ms"/>
    cleaned = re.sub(r'<[^>]+>', ' ', text)
    # Remove raw slash words or spoken artifacts
    cleaned = re.sub(r'\b(forward slash|slash|break time|ms)\b', '', cleaned, flags=re.IGNORECASE)
    # Remove markdown asterisks, brackets, hashes
    cleaned = re.sub(r'[\*\_\[\]\#\/]', ' ', cleaned)
    # Expand common technical acronyms for natural fluid speech
    acronym_map = {
        r'\bAOV\b': 'average order value',
        r'\bAIO\b': 'AI optimization',
        r'\bSEO\b': 'search engine optimization',
        r'\bAPI\b': 'A P I',
        r'\bGPU\b': 'G P U',
        r'\bCPU\b': 'C P U',
        r'\bTTS\b': 'text to speech',
        r'\bLLM\b': 'large language model'
    }
    for pat, repl in acronym_map.items():
        cleaned = re.sub(pat, repl, cleaned, flags=re.IGNORECASE)
    # Collapse multiple whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned



def generate_synthetic_wav(output_path: str, duration_sec: float = 3.0, sample_rate: int = 24000):
    """
    Generates a clean synthetic WAV audio file for offline / fallback TTS synthesis.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for i in range(num_samples):
            sample = int(32767.0 * 0.01 * (i % 100 / 100.0))
            wav_file.writeframes(struct.pack('<h', sample))


def _resolve_local_path(path: str) -> str:
    """
    Translates absolute paths sent from remote nodes (e.g. OCI /home/ubuntu/...)
    to the local node's repo root or tmp dir to prevent permission/missing dir errors.
    """
    if not path:
        return path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    # Map repo paths (e.g. /home/ubuntu/buzzdropfeedv2/logs/...) -> /home/jeevanjoshi/buzzdropfeedv2/logs/...
    if "/buzzdropfeedv2/" in path:
        rel = path.split("/buzzdropfeedv2/", 1)[1]
        return os.path.join(base_dir, rel)
    # Map legacy /tmp/csvg_media paths if any
    return path


@app.post("/tools/synthesize_tts")
async def synthesize_tts(req: TTSRequest):
    """
    Synthesizes neural TTS audio using Kokoro-ONNX model with clean text sanitization
    and region-appropriate voice selection (Neutral / Indian English).
    """
    try:
        req.output_path = _resolve_local_path(req.output_path)
        output_dir = os.path.dirname(os.path.abspath(req.output_path))
        os.makedirs(output_dir, exist_ok=True)

        # 1. Clean and sanitize text to prevent TTS reading raw SSML / slash artifacts
        clean_text = sanitize_tts_text(req.text)

        # 2. Select voice based on region / sanitize blend string
        selected_voice = req.voice
        if "(" in selected_voice or "+" in selected_voice or selected_voice == "af_bella(2)+af_heart(1)":
            selected_voice = "af_sarah" if req.region == "india" else "af_bella"

        try:
            from kokoro_onnx import Kokoro
            import soundfile as sf
            
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            # Kokoro v1.0 checkpoint (better quality). Falls back to v0.19 if missing.
            kokoro_path = os.path.join(base_dir, "kokoro-v1.0.onnx")
            voices_path = os.path.join(base_dir, "voices-v1.0.bin")
            if not (os.path.exists(kokoro_path) and os.path.exists(voices_path)):
                kokoro_path = os.path.join(base_dir, "kokoro-v0.19.onnx")
                voices_path = os.path.join(base_dir, "voices.bin")
            if os.path.exists(kokoro_path) and os.path.exists(voices_path):
                kokoro = Kokoro(kokoro_path, voices_path)
                # Clamp the tempo so it stays intelligible; speed>1 = faster/punchier.
                speed = max(0.85, min(1.15, float(getattr(req, "speed", 1.0) or 1.0)))
                samples, sample_rate = kokoro.create(clean_text, voice=selected_voice, speed=speed, lang="en-us")
                sf.write(req.output_path, samples, sample_rate)
                return {"status": "success", "engine": "kokoro_onnx", "path": req.output_path, "voice": selected_voice}
        except Exception as e:
            print(f"Kokoro TTS Exception: {e}")

        words_count = len(clean_text.split())
        est_duration = max(3.0, round(words_count / 2.2, 1))
        generate_synthetic_wav(req.output_path, duration_sec=est_duration)
        return {"status": "success", "engine": "synthetic_wav_fallback", "path": req.output_path, "duration": est_duration}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/align_subtitles_whisper")
async def align_subtitles_whisper(req: WhisperRequest):
    """
    Generates word-level timestamp alignment and compiles a 16:9 YouTube Shorts / Widescreen .ass subtitle file.
    """
    try:
        req.audio_path = _resolve_local_path(req.audio_path)
        req.output_ass_path = _resolve_local_path(req.output_ass_path)
        output_dir = os.path.dirname(os.path.abspath(req.output_ass_path))
        os.makedirs(output_dir, exist_ok=True)

        ass_header = """[Script Info]
Title: 16:9 CSVG Auto Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,3,2,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        dialogues = []
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(req.audio_path, word_timestamps=True)
            
            # Extract all words with timestamps from Whisper transcription
            all_whisper_words = []
            for segment in segments:
                for w in segment.words:
                    if w.word.strip():
                        all_whisper_words.append({
                            "word": w.word.strip(),
                            "start": w.start,
                            "end": w.end
                        })
            
            # Align with original text using dynamic NLP Sequence Matching
            if req.original_text and all_whisper_words:
                orig_words_raw = req.original_text.split()
                orig_words_clean = [re.sub(r'[^\w\s]', '', w).upper() for w in orig_words_raw]
                w_words_clean = [re.sub(r'[^\w\s]', '', w["word"]).upper() for w in all_whisper_words]
                
                sm = difflib.SequenceMatcher(None, w_words_clean, orig_words_clean)
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag == 'equal':
                        for k in range(i1, i2):
                            orig_idx = j1 + (k - i1)
                            if orig_idx < len(orig_words_raw):
                                all_whisper_words[k]["word"] = orig_words_raw[orig_idx]
                    elif tag in ('replace', 'delete'):
                        orig_chunk = orig_words_raw[j1:j2]
                        if orig_chunk:
                            for idx, k in enumerate(range(i1, i2)):
                                if idx < len(orig_chunk):
                                    all_whisper_words[k]["word"] = orig_chunk[idx]
                                else:
                                    all_whisper_words[k]["word"] = ""
                            if len(orig_chunk) > (i2 - i1) and i2 - 1 >= 0:
                                leftover = " ".join(orig_chunk[(i2 - i1):])
                                all_whisper_words[i2 - 1]["word"] += " " + leftover
                        else:
                            for k in range(i1, i2):
                                all_whisper_words[k]["word"] = ""
                
                # Filter out deleted/empty words
                all_whisper_words = [w for w in all_whisper_words if w["word"].strip()]
            
            # Chunk the aligned words for display
            chunk_size = 5
            for i in range(0, len(all_whisper_words), chunk_size):
                chunk = all_whisper_words[i:i + chunk_size]
                start_sec = chunk[0]["start"]
                end_sec = chunk[-1]["end"]
                if end_sec - start_sec < 1.2:
                    end_sec = start_sec + 1.2
                start_str = f"{int(start_sec//3600)}:{int((start_sec%3600)//60):02d}:{start_sec%60:05.2f}"
                end_str = f"{int(end_sec//3600)}:{int((end_sec%3600)//60):02d}:{end_sec%60:05.2f}"
                phrase_clean = " ".join(w["word"].strip() for w in chunk).upper()
                if phrase_clean:
                    dialogues.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{phrase_clean}")
        except Exception as e:
            print(f"Faster Whisper alignment exception: {e}")

        if not dialogues:
            dialogues.append("Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,NARRATION")

        with open(req.output_ass_path, "w", encoding="utf-8") as f:
            f.write(ass_header + "\n".join(dialogues) + "\n")

        return {"status": "success", "path": req.output_ass_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
async def get_file(path: str):
    local_path = _resolve_local_path(path)
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(local_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
