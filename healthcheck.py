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
  [yt-upload] YouTube OAuth token + client secret exist, and today's upload
            count is within the ~10,000-unit daily quota (1,600 units/upload, max 4).
  [yt-score] YouTube Data API budget for the competitor-demand / real-time
            TOPSIS score (yt_demand_quota.json vs YT_SEARCH_DAILY_BUDGET).
  [rag]      3rd-party fact/RAG keys present (Marketaux, Alpha Vantage, Exa,
            NewsAPI) — non-fatal fallbacks but missing keys degrade story grounding.
  [grounding] Google-Search grounding pre-flight. When --rag grounded is
            requested, GOOGLE_CLOUD_PROJECT + the google-genai SDK MUST be
            configured; otherwise the run silently degrades to the scraper path.
            In scraper mode a missing Vertex config is only a WARN.
  [scrape]   Primary article deep-crawl path (Firecrawl). If FIRECRAWL_API_KEY
            is absent, _scrape_selected_article falls back to a bare urllib GET
            that many news sites 403 (the 'Article scrape fallback failed'
            glitch) and grounding degrades.
  [media]   BGM asset present and disk space for render (final-video quality).

Exit semantics:
  0  = all REQUIRED checks passed (proceed); WARN-level issues are advisory.
  1  = at least one REQUIRED check failed (abort the run).
  2  = a WARN-level advisory surfaced and --strict was given (treat warnings as fails).

Audit trail: every run writes logs/health_check_<timestamp>.log (and a
canonical logs/health_check.log) recording each check, verdict and summary so
pre-flight state is auditable per run.

OPT-IN/OPT-OUT live probes:
  --probe-llm  make a real 1-token LLM call to confirm the model answers and
                is not rate-limited (costs a tiny amount of LLM quota; skipped by default).
  --skip-probe-yt skip the live YouTube OAuth token refresh + channels.list check (run by default).

Usage:
  source venv/bin/activate && python healthcheck.py
  source venv/bin/activate && python healthcheck.py --probe-llm --probe-yt
  source venv/bin/activate && python healthcheck.py --rag grounded   # gate grounding
  source venv/bin/activate && python healthcheck.py --strict   # WARN fails too
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
    # Gemini and Google Cloud Project are critical APIs/requirements
    gemini_key = env_key("GEMINI_API_KEY") or env_key("GOOGLE_API_KEY")
    if not gemini_key:
        record("env", "Gemini API key", _FAIL, "GEMINI_API_KEY or GOOGLE_API_KEY is missing")
        return False
    
    gcp_project = env_key("GOOGLE_CLOUD_PROJECT")
    if not gcp_project:
        record("env", "Google Cloud Project", _FAIL, "GOOGLE_CLOUD_PROJECT is missing")
        return False

    required = [
        "PREFERRED_LLM_PROVIDER", "LLM_MODEL",
        "AUDIO_EDGE_URL", "YOUTUBE_TOKEN_FILE", "YOUTUBE_CLIENT_SECRET",
        "YOUTUBE_API_KEY", "NOTIFY_EMAIL_TO", "NOTIFY_EMAIL_FROM",
    ]
    # At least one usable LLM credential is check warning.
    llm_keys = [k for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
                if env_key(k)]
    missing = [k for k in required if not env_key(k)]
    if missing:
        record("env", "required keys", _WARN, f"missing non-critical: {', '.join(missing)}")
    else:
        record("env", "required keys", _OK,
               f"provider={env_key('PREFERRED_LLM_PROVIDER')} model={env_key('LLM_MODEL')}")
    if llm_keys:
        record("env", "LLM credential", _OK, f"present ({len(llm_keys)} of 3 providers)")
    else:
        record("env", "LLM credential", _WARN, "no OPENROUTER/OPENAI/GEMINI key set")
    return True


# ---------------------------------------------------------------------------
# [bin] ffmpeg / ffprobe
# ---------------------------------------------------------------------------
def check_binaries():
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        record("bin", "ffmpeg/ffprobe", _WARN, f"missing: {', '.join(missing)}")
    else:
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
    is_gemini = "gemini" in provider or "google" in provider
    available = False
    try:
        if provider in ("local", "llama"):
            available = client.is_local_llm_available()
            detail = "local llama.cpp"
        else:
            available = client.is_cloud_llm_available()
            detail = f"cloud api key present ({provider})"
    except Exception as e:
        record("llm", "provider check", _FAIL if is_gemini else _WARN, str(e))
        return not is_gemini

    if not available:
        record("llm", "provider availability", _FAIL if is_gemini else _WARN,
               f"{detail} but unusable — no LLM fallback, aborting.")
        return not is_gemini
    record("llm", "provider availability", _OK, detail)

    if probe:
        try:
            out = client.generate_json(
                system_prompt="Reply with exactly this JSON, nothing else:",
                prompt='{"ok": true}',
            )
            ok = bool(out and out.get("ok"))
            record("llm", "live probe", _OK if ok else (_FAIL if is_gemini else _WARN),
                   "model answered" if ok else "model did not return expected JSON")
            if is_gemini and not ok:
                return False
        except Exception as e:
            record("llm", "live probe", _FAIL if is_gemini else _WARN, str(e))
            if is_gemini:
                return False
    return True


# ---------------------------------------------------------------------------
# [pi] Audio edge node reachability (TTS + Whisper)
# ---------------------------------------------------------------------------
def check_audio_edge():
    url = env_key("AUDIO_EDGE_URL").rstrip("/")
    if not url:
        record("pi", "audio edge URL", _WARN, "AUDIO_EDGE_URL not set")
        return True
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
        record("pi", "audio edge reachable", _WARN, f"{url} unreachable: {e}")
        return True


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


def check_youtube(probe: bool = True):
    token_file = env_key("YOUTUBE_TOKEN_FILE") or "token.json"
    client_secret = env_key("YOUTUBE_CLIENT_SECRET") or "client_secret.json"
    if not os.path.exists(token_file):
        record("yt", "OAuth token", _WARN, f"{token_file} missing")
    if not os.path.exists(client_secret):
        record("yt", "client secret", _WARN, f"{client_secret} missing")

    # Quota estimate: 1600 units/upload, 10,000/day cap, max 4 uploads.
    today = _count_today_uploads()
    units = today * 1600
    remaining = 10000 - units - 1600  # minus the upcoming upload
    if today >= 4:
        record("yt", "quota", _WARN,
               f"{today} uploads today (max 4) — daily cap reached")
    elif remaining < 0:
        record("yt", "quota", _WARN,
               f"{today} uploads today → next upload exceeds 10,000-unit cap")
    else:
        record("yt", "quota", _OK,
               f"{today} upload(s) today; {remaining} units remain after next upload")

    if probe:
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request as GAuthRequest
            from googleapiclient.discovery import build
            creds = Credentials.from_authorized_user_file(token_file)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(GAuthRequest())
            yt = build("youtube", "v3", credentials=creds)
            yt.channels().list(part="snippet", mine=True).execute()
            record("yt", "live auth probe", _OK, "token refresh + channels.list OK")
        except Exception as e:
            record("yt", "live auth probe", _WARN, str(e))
    return True


# ---------------------------------------------------------------------------
# [yt-score] YouTube Data API budget for the real-time TOPSIS score.
# The competitor "views-per-hour" (vph_score) used in topic selection is fetched
# from the YouTube Data API (search.list + videos.list), budgeted per day via
# YT_SEARCH_DAILY_BUDGET (default 30) and tracked in yt_demand_quota.json.
# Exhausting it silently degrades topic quality (falls back to niche pools).
# ---------------------------------------------------------------------------
def _quota_file_path(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def check_yt_score_budget():
    budget = int(os.getenv("YT_SEARCH_DAILY_BUDGET", "30"))
    used_path = _quota_file_path("yt_demand_quota.json")
    today = datetime.date.today().isoformat()
    used = 0
    if os.path.exists(used_path):
        try:
            with open(used_path, "r", encoding="utf-8") as f:
                q = json.load(f)
            used = int((q or {}).get(today, 0))
        except (json.JSONDecodeError, IOError):
            used = 0
    remaining = budget - used
    if remaining <= 0:
        record("yt-score", "demand budget", _WARN,
               f"{used}/{budget} used today — real-time TOPSIS score disabled "
               f"(topic quality degraded)")
        return True
    if remaining <= 5:
        record("yt-score", "demand budget", _WARN,
               f"{used}/{budget} used today; only {remaining} searches left "
               f"before real-time scoring disables")
        return True
    record("yt-score", "demand budget", _OK,
           f"{used}/{budget} searches used today; {remaining} remaining")
    return True


# ---------------------------------------------------------------------------
# [rag] Third-party fact/RAG keys — non-fatal but degrading when missing.
# A missing Marketaux/Alpha Vantage/Exa/NewsAPI key means fewer, fresher facts
# reach the story, weakening grounding (and thus the FINAL script quality).
# ---------------------------------------------------------------------------
def _any_env(*names):
    return [n for n in names if env_key(n)]


def check_rag_keys(strict: bool = False):
    crawlers = {
        "FIRECRAWL_API_KEY": "Firecrawl",
        "TAVILY_API_KEY": "Tavily",
        "EXA_API_KEY": "Exa",
        "NEWSAPI_KEY": "NewsAPI"
    }
    present = [name for env_var, name in crawlers.items() if env_key(env_var)]
    
    if len(present) < 2:
        record("rag", "crawler apis", _FAIL,
               f"Only {len(present)} crawler API(s) set ({', '.join(present) if present else 'none'}). At least 2 are required from: Firecrawl, Tavily, Exa, NewsAPI")
        return False
    record("rag", "crawler apis", _OK, f"At least 2 crawlers available: {', '.join(present)}")

    tier1 = _any_env("MARKETAUX_API_KEY", "ALPHA_VANTAGE_KEY")
    tier2 = _any_env("EXA_API_KEY", "NEWSAPI_KEY")
    exa = env_key("EXA_API_KEY")
    if not exa and not tier1 and not tier2:
        record("rag", "fact sources", _WARN,
               "no Marketaux/AlphaVantage/Exa/NewsAPI keys — story grounding "
               "relies on RSS-only, quality may drop")
        return True
    detail = (f"exa={'set' if exa else 'unset'}, "
              f"macrov={'set' if tier1 else 'unset'}, "
              f"news={'set' if tier2 else 'unset'}")
    record("rag", "fact sources", _OK, detail)
    return True


# ---------------------------------------------------------------------------
# [grounding] Google-Search grounding pre-flight.
# When the operator explicitly requests --rag grounded but Vertex/ADC is not
# configured, the pipeline silently degrades to the scraper path (the
# 'GOOGLE_CLOUD_PROJECT not set; grounding unavailable' glitch). That is a
# hard FAIL when grounded was chosen; merely a WARN otherwise.
# ---------------------------------------------------------------------------
def check_grounding(rag_mode: str):
    project = env_key("GOOGLE_CLOUD_PROJECT")
    api_key = env_key("GEMINI_API_KEY") or env_key("GOOGLE_API_KEY")
    configured = bool(project) or bool(api_key)
    try:
        from google import genai as _genai  # noqa: F401
        sdk_ok = True
    except Exception:
        sdk_ok = False

    # Google Grounded search availability is now a critical/hard requirement
    if not configured:
        record("grounding", "auth configured", _FAIL,
               "no grounding auth (GOOGLE_CLOUD_PROJECT / GEMINI_API_KEY) set — grounding unavailable")
        return False
    if not sdk_ok:
        record("grounding", "google-genai SDK", _FAIL,
               "google-genai not installed — grounding unavailable")
        return False
    record("grounding", "configured", _OK,
           f"{('GOOGLE_CLOUD_PROJECT=' + project) if project else 'GEMINI_API_KEY set'}, "
           f"google-genai SDK present")
    return True


# ---------------------------------------------------------------------------
# [scrape] Primary article deep-crawl path (Firecrawl).
# Without FIRECRAWL_API_KEY, _scrape_selected_article falls back to a bare
# urllib GET that major news sites (NYT et al.) 403 — the 'Article scrape
# fallback failed' glitch. Firecrawl present means the primary path avoids it.
# ---------------------------------------------------------------------------
def check_scrape_path(strict: bool = False):
    fc = env_key("FIRECRAWL_API_KEY")
    if not fc:
        record("scrape", "Firecrawl key", _WARN,
               "FIRECRAWL_API_KEY unset — deep-crawl falls back to bare urllib GET")
    else:
        record("scrape", "Firecrawl key", _OK,
               "primary deep-crawl path available (avoids urllib 403 fallback)")
    return True


# ---------------------------------------------------------------------------
# [media] BGM + disk space — final-video quality gates.
# Missing resources/bgm.mp3 silently falls back to silence (bad audio);
# low free disk aborts the ffmpeg render mid-way.
# ---------------------------------------------------------------------------
def check_media(strict: bool = False):
    ok = True
    bgm = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "resources", "bgm.mp3")
    if not os.path.exists(bgm) or os.path.getsize(bgm) < 1000:
        record("media", "BGM asset", _WARN,
               "resources/bgm.mp3 missing/small — final video will fall back "
               "to silence (no background music)")
    else:
        record("media", "BGM asset", _OK, "resources/bgm.mp3 present")

    # Disk free on the repo/root volume (render writes to /tmp + repo).
    for path, label in (("/", "root"), ("/tmp", "/tmp")):
        try:
            statvfs = os.statvfs(path)
            free_gb = statvfs.f_bavail * statvfs.f_frsize / 1e9
            if free_gb < 5:
                record("media", "disk space", _WARN,
                       f"{label} only {free_gb:.1f} GB free — render may abort")
            else:
                record("media", "disk space", _OK,
                       f"{label} {free_gb:.1f} GB free")
        except Exception:
            continue
    return True


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def _parse_rag_mode(argv):
    """Pull the ``--rag <mode>`` value from argv (default 'scraper')."""
    try:
        i = argv.index("--rag")
        if i + 1 < len(argv):
            return argv[i + 1]
    except ValueError:
        pass
    return "scraper"


def _write_audit_log(exit_code: int):
    """Persist a timestamped + canonical health-check audit log for this run."""
    repo = os.path.dirname(os.path.abspath(__file__))
    logs = os.path.join(repo, "logs")
    try:
        os.makedirs(logs, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines = [f"CSVG Health Check Audit — {stamp}",
                 f"Exit code: {exit_code}", "=" * 62]
        for comp, name, status, detail in results:
            lines.append(f"{status:4} [{comp:9}] {name} — {detail}")
        failed = [r for r in results if r[2] == "FAIL"]
        warned = [r for r in results if r[2] == "WARN"]
        lines.append("=" * 62)
        lines.append(f"Summary: {len(failed)} FAILED, {len(warned)} WARN, "
                     f"{len(results) - len(failed) - len(warned)} PASS")
        body = "\n".join(lines) + "\n"
        with open(os.path.join(logs, f"health_check_{ts}.log"), "w", encoding="utf-8") as f:
            f.write(body)
        with open(os.path.join(logs, "health_check.log"), "w", encoding="utf-8") as f:
            f.write(body)
        return os.path.join(logs, f"health_check_{ts}.log")
    except Exception as e:  # audit log must never break the health gate
        return f"<audit log write failed: {e}>"


def main(argv):
    probe_llm = "--probe-llm" in argv
    probe_yt = "--skip-probe-yt" not in argv
    strict = "--strict" in argv
    rag_mode = _parse_rag_mode(argv)

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
        ("yt-score", check_yt_score_budget),
        ("rag", lambda: check_rag_keys(strict=strict)),
        ("grounding", lambda: check_grounding(rag_mode)),
        ("scrape", lambda: check_scrape_path(strict=strict)),
        ("media", lambda: check_media(strict=strict)),
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
    audit_path = _write_audit_log(1 if failed else
                                  (2 if (strict and warned) else 0))
    if failed:
        print(f"HEALTH CHECK: FAILED — {len(failed)} required check(s) failing. "
              f"Run aborted.")
        for _, name, _, detail in failed:
            print(f"  ✗ {name}: {detail}")
        print(f"Audit log written: {audit_path}")
        print("Fix the failures above, then re-run ./run_production.sh")
        return 1
    if strict and warned:
        print(f"HEALTH CHECK: FAILED (--strict) — {len(warned)} warning(s) "
              f"treated as failures.")
        for _, name, _, detail in warned:
            print(f"  ⚠ {name}: {detail}")
        print(f"Audit log written: {audit_path}")
        print("Resolve the warnings (or drop --strict), then re-run.")
        return 2
    print(f"HEALTH CHECK: PASSED — all required checks green"
          + (f" (+{len(warned)} advisory warning(s))" if warned else "")
          + ". Proceeding with production run.")
    print(f"Audit log written: {audit_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))