#!/usr/bin/env python3
"""
CSVG Production Pre-Flight Health Check.

Run BEFORE the pipeline launches. Validates every external dependency a real
(publishing) run needs and exits non-zero if any REQUIRED check fails, so
run_production.sh never starts a run that is guaranteed to die mid-way or
silently skip publishing.

Checks:
  [env]     Required .env keys are set (secrets presence only, values masked).
  [bin]     ffmpeg + ffprobe present (render + quality gates need them).
  [llm]     LLM provider is usable (is_available on the configured provider).
  [pi]      Audio edge node (AUDIO_EDGE_URL) is reachable for TTS + Whisper.
  [yt]      YouTube OAuth token + client secret exist, and today's upload count
            is within the ~10,000-unit daily quota (1,600 units/upload, max 4).

Exit semantics:
  0  = all REQUIRED checks passed (proceed).
  1  = at least one REQUIRED check failed (abort the run).

OPT-IN live probes (skipped by default so this stays cheap / non-destructive):
  --probe-llm  make a real 1-token LLM call to confirm the model answers
               (costs a tiny amount of quota on the LLM provider).
  --probe-yt   refresh the OAuth token and run a real `channels.list` call to
               confirm upload+read scopes (costs ~1 unit of YouTube quota).

Usage:
  source venv/bin/activate && python healthcheck.py
  source venv/bin/activate && python healthcheck.py --probe-llm --probe-yt
"""
import os
import sys
import json
import shutil
import datetime
from urllib.request import urlopen, Request

import dotenv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_OK, _WARN, _FAIL = "PASS", "WARN", "FAIL"
results = []


def record(component: str, name: str, status: str, detail: str = ""):
    results.append((component, name, status, detail))
    icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(status, "?")
    print(f"{icon} [{component:5}] {name:24} {status:4} {detail}")


def env_key(name: str) -> str:
    return (os.getenv(name) or "").strip()


# ---------------------------------------------------------------------------
# [env] Secrets present (no values leaked)
# ---------------------------------------------------------------------------
def check_required_env():
    required = [
        "PREFERRED_LLM_PROVIDER", "LLM_MODEL",
        "AUDIO_EDGE_URL", "YOUTUBE_TOKEN_FILE", "YOUTUBE_CLIENT_SECRET",
        "YOUTUBE_API_KEY", "NOTIFY_EMAIL_TO", "NOTIFY_EMAIL_FROM",
    ]
    # At least one usable LLM credential is mandatory (no boilerplate fallback).
    llm_keys = [k for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
                if env_key(k)]
    missing = [k for k in required if not env_key(k)]
    if missing:
        record("env", "required keys", _FAIL, f"missing: {', '.join(missing)}")
        return False
    record("env", "required keys", _OK,
           f"provider={env_key('PREFERRED_LLM_PROVIDER')} model={env_key('LLM_MODEL')}")
    if llm_keys:
        record("env", "LLM credential", _OK, f"present ({len(llm_keys)} of 3 providers)")
    else:
        record("env", "LLM credential", _FAIL, "no OPENROUTER/OPENAI/GEMINI key set")
        return False
    return True


# ---------------------------------------------------------------------------
# [bin] ffmpeg / ffprobe
# ---------------------------------------------------------------------------
def check_binaries():
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        record("bin", "ffmpeg/ffprobe", _FAIL, f"missing: {', '.join(missing)}")
        return False
    record("bin", "ffmpeg/ffprobe", _OK, f"{shutil.which('ffmpeg')} / {shutil.which('ffprobe')}")
    return True


# ---------------------------------------------------------------------------
# [llm] Provider availability
# ---------------------------------------------------------------------------
def check_llm(probe: bool = False):
    try:
        from src.engine.llm_client import LLMClient
        client = LLMClient()
    except Exception as e:
        record("llm", "LLMClient import", _FAIL, str(e))
        return False

    provider = env_key("PREFERRED_LLM_PROVIDER").lower() or "cloud"
    available = False
    try:
        if provider in ("local", "llama"):
            available = client.is_local_llm_available()
            detail = "local llama.cpp"
        else:
            available = client.is_cloud_llm_available()
            detail = f"cloud api key present ({provider})"
    except Exception as e:
        record("llm", "provider check", _FAIL, str(e))
        return False

    if not available:
        record("llm", "provider availability", _FAIL,
               f"{detail} but unusable — no LLM fallback, aborting.")
        return False
    record("llm", "provider availability", _OK, detail)

    if probe:
        try:
            import asyncio
            out = asyncio.run(client.generate_json(
                system_prompt="Reply with exactly this JSON, nothing else:",
                prompt='{"ok": true}',
            ))
            ok = bool(out and out.get("ok"))
            record("llm", "live probe", _OK if ok else _FAIL,
                   "model answered" if ok else "model did not return expected JSON")
            return ok
        except Exception as e:
            record("llm", "live probe", _FAIL, str(e))
            return False
    return True


# ---------------------------------------------------------------------------
# [pi] Audio edge node reachability (TTS + Whisper)
# ---------------------------------------------------------------------------
def check_audio_edge():
    url = env_key("AUDIO_EDGE_URL").rstrip("/")
    if not url:
        record("pi", "audio edge URL", _FAIL, "AUDIO_EDGE_URL not set")
        return False
    try:
        from urllib.error import HTTPError
        try:
            req = Request(f"{url}/", headers={"User-Agent": "csvg-healthcheck"})
            with urlopen(req, timeout=8) as resp:
                status = resp.status
        except HTTPError as he:
            # Any HTTP response (incl. 404 for an unknown root route) proves the
            # node is up; only a transport error means it's unreachable.
            status = he.code
        record("pi", "audio edge reachable", _OK, f"{url} responded HTTP {status}")
        return True
    except Exception as e:
        record("pi", "audio edge reachable", _FAIL, f"{url} unreachable: {e}")
        return False


# ---------------------------------------------------------------------------
# [yt] OAuth auth files + quota estimate
# ---------------------------------------------------------------------------
def _count_today_uploads() -> int:
    """Count today's uploads from published_topics.json (UTC date match)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "published_topics.json")
    if not os.path.exists(path):
        return 0
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    try:
        with open(path, "r", encoding="utf-8") as f:
            topics = json.load(f)
    except (json.JSONDecodeError, IOError):
        return 0
    n = 0
    for t in topics if isinstance(topics, list) else []:
        dt = (t or {}).get("published_at") or (t or {}).get("publishedAt") or ""
        if dt.startswith(today):
            n += 1
    return n


def check_youtube(probe: bool = False):
    token_file = env_key("YOUTUBE_TOKEN_FILE") or "token.json"
    client_secret = env_key("YOUTUBE_CLIENT_SECRET") or "client_secret.json"
    ok_files = True
    if not os.path.exists(token_file):
        record("yt", "OAuth token", _FAIL, f"{token_file} missing")
        ok_files = False
    if not os.path.exists(client_secret):
        record("yt", "client secret", _FAIL, f"{client_secret} missing")
        ok_files = False
    if ok_files:
        record("yt", "OAuth files", _OK,
               f"{token_file}, {client_secret} present")

    # Quota estimate: 1600 units/upload, 10,000/day cap, max 4 uploads.
    today = _count_today_uploads()
    units = today * 1600
    remaining = 10000 - units - 1600  # minus the upcoming upload
    if today >= 4:
        record("yt", "quota", _FAIL,
               f"{today} uploads today (max 4) — daily cap reached")
        return False
    if remaining < 0:
        record("yt", "quota", _FAIL,
               f"{today} uploads today → next upload exceeds 10,000-unit cap")
        return False
    record("yt", "quota", _OK,
           f"{today} upload(s) today; {remaining} units remain after next upload")

    if probe:
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request as GAuthRequest
            from googleapiclient.discovery import build
            creds = Credentials.from_authorized_user_file(
                token_file, ["https://www.googleapis.com/auth/youtube.upload"])
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(GAuthRequest())
            yt = build("youtube", "v3", credentials=creds)
            yt.channels().list(part="snippet", mine=True).execute()
            record("yt", "live auth probe", _OK, "token refresh + channels.list OK")
        except Exception as e:
            record("yt", "live auth probe", _FAIL, str(e))
            return False
    return ok_files


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main(argv):
    probe_llm = "--probe-llm" in argv
    probe_yt = "--probe-yt" in argv

    # Load .env exactly like the pipeline (must precede app imports).
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        print("✗ [env]  .env file missing — copy example.env -> .env")
        return 1
    dotenv.load_dotenv(env_path, override=False)

    print("=" * 62)
    print("CSVG Production Pre-Flight Health Check")
    print("=" * 62)

    checks = [
        ("env", check_required_env),
        ("bin", check_binaries),
        ("llm", lambda: check_llm(probe=probe_llm)),
        ("pi", check_audio_edge),
        ("yt", lambda: check_youtube(probe=probe_yt)),
    ]

    ok = True
    for comp, fn in checks:
        try:
            if not fn():
                ok = False
        except Exception as e:
            record(comp, "check error", _FAIL, str(e))
            ok = False

    print("=" * 62)
    failed = [r for r in results if r[2] == "FAIL"]
    warned = [r for r in results if r[2] == "WARN"]
    if failed:
        print(f"HEALTH CHECK: FAILED — {len(failed)} required check(s) failing. "
              f"Run aborted.")
        for _, name, _, detail in failed:
            print(f"  ✗ {name}: {detail}")
        print("Fix the failures above, then re-run ./run_production.sh")
        return 1
    print(f"HEALTH CHECK: PASSED — all required checks green"
          + (f" (+{len(warned)} warning(s))" if warned else "")
          + ". Proceeding with production run.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))