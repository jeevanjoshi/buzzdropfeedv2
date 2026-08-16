"""Realtime provider API-usage tracking (fal, Google/Gemini, OpenRouter).

Two complementary halves:

1. **Live capture ledger** — records *actual* per-request usage as the pipeline
   runs: real token counts read back from OpenRouter / Vertex / Gemini response
   metadata and image counts from nano-banana (Google) and fal/Replicate
   generation. Token counts are real (never estimated); the USD figure is
   estimated from a small pricing table (``estimate_model_usd``) and clearly
   marked as an estimate. The ledger is persisted atomically + thread-safely to
   ``logs/provider_usage.json`` so the dashboard shows live usage next to the
   per-run budget estimates.

2. **Provider pulls** — ``pull_all()`` queries each provider's OWN usage/billing
   API for authoritative numbers:
     * fal          -> GET https://api.fal.ai/v1/models/usage (real endpoint
                       cost) + /v1/account/billing (credit balance).
     * OpenRouter   -> GET https://openrouter.ai/api/v1/credits (total credits
                       purchased + used). Requires a *management* key
                       (OPENROUTER_MANAGEMENT_KEY); a normal ``sk-or-...`` key
                       yields a 403 which is surfaced verbatim with a hint.
     * Google       -> Vertex AI: best-effort GET
                       https://cloudbilling.googleapis.com/v1/projects/<p>/billingInfo
                       (confirms billing is enabled + which account; raw dollar
                       amounts would need a BigQuery billing export, so Google's
                       real usage numbers come from the live-capture token
                       ledger).
   Every fetch is failure-tolerant (never raises), each provider is independent,
   and results are cached into ``provider_pull`` inside the ledger file.

Everything is a no-op when ``CSVG_API_USAGE`` is disabled (default on), and no
network / provider SDK import happens at module import time.
"""

import os
import json
import time
import threading
import datetime
from typing import Dict, List, Optional, Tuple, Any

FAL_BASE = "https://api.fal.ai/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
GOOGLE_BILLING_BASE = "https://cloudbilling.googleapis.com/v1"

LEDGER_PATH = os.getenv("API_USAGE_FILE", "logs/provider_usage.json")
GATE_ENV = os.getenv("CSVG_API_USAGE", "1").strip().lower()
FLUSH_INTERVAL_SECS = float(os.getenv("API_USAGE_FLUSH_SECS", "2"))
MAX_RUN_RECORDS = int(os.getenv("API_USAGE_MAX_RUNS", "200"))
MAX_OPENROUTER_IDS_PER_SESSION = int(os.getenv("API_USAGE_MAX_IDS_PER_SESSION", "200"))

# Approx USD per 1M tokens (prompt, completion) keyed on model-name substrings;
# fallback constants apply for anything unknown. Est cost is an approximation —
# authoritative numbers come from the provider pulls above.
MODEL_PRICING_1M: List[Tuple[str, float, float]] = [
    ("gemini-2.5-flash", 0.30, 2.50),
    ("gemini-2.5-pro", 1.25, 10.00),
    ("gemini-3", 1.25, 10.00),
    ("deepseek-v4", 0.25, 1.25),
    ("deepseek-v3", 0.30, 0.90),
    ("gpt-5", 1.50, 6.00),
    ("gpt-4o", 2.50, 10.00),
    ("gpt-4o-mini", 0.15, 0.60),
    ("claude-3-5", 3.00, 15.00),
    ("claude-3-7", 3.00, 15.00),
    ("llama-3.3", 0.20, 0.40),
]
FALLBACK_PROMPT_1M = float(os.getenv("API_USAGE_FALLBACK_PROMPT_1M", "1.00"))
FALLBACK_OUTPUT_1M = float(os.getenv("API_USAGE_FALLBACK_OUTPUT_1M", "3.00"))


def estimate_model_usd(model: Optional[str], in_tokens: int, out_tokens: int) -> float:
    """Best-effort USD for a token count based on a tiny pricing table."""
    name = (model or "").lower()
    pin, pout = FALLBACK_PROMPT_1M, FALLBACK_OUTPUT_1M
    for prefix, p_in, p_out in MODEL_PRICING_1M:
        if prefix in name:
            pin, pout = p_in, p_out
            break
    return round(in_tokens / 1_000_000 * pin + out_tokens / 1_000_000 * pout, 8)


def parse_openrouter_usage(usage: Any) -> Tuple[int, int]:
    """(input_tokens, output_tokens) from an OpenRouter/OpenAI-style ``usage``
    object (dict or attribute-accessible object). Never raises."""
    if not usage:
        return 0, 0
    try:
        if hasattr(usage, "keys"):
            d = dict(usage)
        else:
            d = {k: getattr(usage, k) for k in ("prompt_tokens", "completion_tokens")}
        return int(d.get("prompt_tokens") or 0), int(d.get("completion_tokens") or 0)
    except (TypeError, ValueError, AttributeError):
        return 0, 0


def parse_gemini_usage(usage_metadata: Any) -> Tuple[int, int]:
    """(input_tokens, output_tokens) from a Google GenAI ``usage_metadata``
    (dict or ``google.genai.types`` usage object). Handles AI Studio / Vertex
    shapes. Never raises."""
    if not usage_metadata:
        return 0, 0
    try:
        if hasattr(usage_metadata, "keys"):
            d = dict(usage_metadata)
        else:
            d = {k: getattr(usage_metadata, k) for k in
                 ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")
                 if hasattr(usage_metadata, k)}
        in_tok = int(d.get("promptTokenCount") or 0)
        out_tok = int(d.get("candidatesTokenCount") or 0)
        if not in_tok and not out_tok:
            in_tok = int(d.get("totalTokenCount") or 0)
        return in_tok, out_tok
    except (TypeError, ValueError, AttributeError):
        return 0, 0


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _empty_counters() -> dict:
    return {
        "calls": 0,
        "in_tokens": 0,
        "out_tokens": 0,
        "images": 0,
        "est_usd": 0.0,
        "last_used": None,
    }


def _acc(counters: dict, in_tokens: int = 0, out_tokens: int = 0,
         images: int = 0, est_usd: float = 0.0) -> None:
    counters["calls"] = int(counters.get("calls", 0)) + 1
    counters["in_tokens"] = int(counters.get("in_tokens", 0)) + int(in_tokens)
    counters["out_tokens"] = int(counters.get("out_tokens", 0)) + int(out_tokens)
    counters["images"] = int(counters.get("images", 0)) + int(images)
    counters["est_usd"] = round(float(counters.get("est_usd", 0.0)) + est_usd, 6)
    counters["last_used"] = _utcnow()


def _atomic_dump(path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except (OSError, IOError):
        pass


class ProviderUsageLedger:
    """Thread-safe realtime ledger of provider usage, persisted atomically."""

    def __init__(self, path: Optional[str] = None, enabled: Optional[bool] = None):
        self.path = path or LEDGER_PATH
        self._enabled = GATE_ENV not in ("0", "false", "no")
        if enabled is not None:
            self._enabled = enabled
        self._lock = threading.RLock()
        self._totals: dict = {}
        self._daily: dict = {}
        self._provider_pull: dict = {}
        self._runs: dict = {}
        self._active_run: Optional[str] = None
        self._active_sessions: dict = {}
        self._last_save = 0.0
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (IOError, OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        live = data.get("live") or {}
        if isinstance(live, dict):
            self._totals = {p: dict(c) for p, c in live.items() if isinstance(c, dict)}
        daily = data.get("daily") or {}
        if isinstance(daily, dict):
            self._daily = daily
        pp = data.get("provider_pull") or {}
        if isinstance(pp, dict):
            self._provider_pull = pp
        runs = data.get("runs") or {}
        if isinstance(runs, dict):
            self._runs = runs

    # ── recording ────────────────────────────────────────────────────────
    def is_enabled(self) -> bool:
        return self._enabled

    def record(self, provider: str, in_tokens: int = 0, out_tokens: int = 0,
               images: int = 0, est_usd: float = 0.0) -> None:
        """Accumulate one call into ``provider`` counters (all-time + daily),
        persisting the ledger. No-op when disabled or provider empty."""
        provider = (provider or "").strip().lower()
        if not self._enabled or not provider:
            return
        in_tokens = max(0, int(in_tokens))
        out_tokens = max(0, int(out_tokens))
        images = max(0, int(images))
        est_usd = max(0.0, float(est_usd))
        day = _utcnow()[:10]
        with self._lock:
            c = self._totals.setdefault(provider, _empty_counters())
            _acc(c, in_tokens=in_tokens, out_tokens=out_tokens, images=images, est_usd=est_usd)
            dc = self._daily.setdefault(day, {}).setdefault(provider, _empty_counters())
            _acc(dc, in_tokens=in_tokens, out_tokens=out_tokens, images=images, est_usd=est_usd)
            active = self._active_run
            if active:
                sess = self._active_sessions.get(active)
                if sess:
                    sc = sess["counts"].setdefault(provider, _empty_counters())
                    _acc(sc, in_tokens=in_tokens, out_tokens=out_tokens, images=images, est_usd=est_usd)
            now = time.monotonic()
        if now - self._last_save >= FLUSH_INTERVAL_SECS:
            self.flush()

    def record_llm(self, provider: str, model: Optional[str] = None,
                   in_tokens: int = 0, out_tokens: int = 0) -> None:
        est = estimate_model_usd(model, in_tokens, out_tokens)
        self.record(provider, in_tokens=in_tokens, out_tokens=out_tokens,
                    est_usd=est)

    def record_openroute(self, model: Optional[str] = None, in_tokens: int = 0,
                       out_tokens: int = 0, gen_id: Optional[str] = None) -> None:
        """Record one OpenRouter LLM call with REAL token counts, plus the
        OpenRouter generation id (''gen-...``) so a run can be cross-referenced
        against the openrouter.ai activity tab. Cost is estimated."""
        self.record_llm("openrouter", model=model, in_tokens=in_tokens, out_tokens=out_tokens)
        if gen_id:
            with self._lock:
                sess = self._active_sessions.get(self._active_run) if self._active_run else None
                if sess and gen_id not in sess["ids"]:
                    sess["ids"].append(gen_id)
                    if len(sess["ids"]) > MAX_OPENROUTER_IDS_PER_SESSION:
                        del sess["ids"][:-MAX_OPENROUTER_IDS_PER_SESSION]

    def record_visual(self, provider: str, images: int = 1, est_usd: float = 0.0) -> None:
        self.record(provider, images=images, est_usd=est_usd)

    def set_provider_pull(self, provider: str, data: Optional[dict]) -> None:
        provider = (provider or "").strip().lower()
        if data is None or not provider:
            return
        with self._lock:
            self._provider_pull[provider] = dict(data)
        self.flush()

    # ── per-run attribution ──────────────────────────────────────────────
    def begin_run(self, pipeline_id: Optional[str]) -> None:
        """Mark a pipeline session as the attribution target for all subsequent
        record() calls. Each begin_run creates a new session inside the run's
        record (so a resumed run accumulates attempts)."""
        if not pipeline_id:
            return
        if not self._enabled:
            self._active_run = pipeline_id
            return
        with self._lock:
            if self._active_run == pipeline_id:
                return
            self._active_run = pipeline_id
            run = self._runs.setdefault(pipeline_id, {
                "pipeline_id": pipeline_id,
                "started_at": None,
                "finished_at": None,
                "attempts": 0,
                "result": "in_progress",
                "retried": False,
                "sessions": [],
                "costs": {},
                "totals": {"est_usd": 0.0, "calls": 0, "in_tokens": 0, "out_tokens": 0, "images": 0},
            })
            if not run.get("started_at"):
                run["started_at"] = _utcnow()
            run["result"] = "in_progress"
            sess = {"started_at": _utcnow(), "finished_at": None,
                    "status": "in_progress", "stage": "INITIALIZATION",
                    "outcome": "in_progress", "counts": {}, "openrouter_ids": []}
            run["sessions"].append(sess)
            run["attempts"] = int(run.get("attempts") or 0) + 1
            self._active_sessions[pipeline_id] = {"counts": sess["counts"], "ids": sess["openrouter_ids"]}

    def end_run(self, pipeline_id: Optional[str], status: str = "success",
                stage: Optional[str] = None) -> None:
        """Seal the current session of a run with its final status/stage and
        derive an outcome bucket (published / till_upload / failed / aborted /
        retried_success). Also aggregates per-provider costs across sessions."""
        if not pipeline_id or not self._enabled:
            return
        with self._lock:
            if self._active_run == pipeline_id:
                self._active_run = None
                sess_counts = self._active_sessions.pop(pipeline_id, None)
            else:
                sess_counts = None
            run = self._runs.get(pipeline_id)
            if not run or not run.get("sessions"):
                return
            last = run["sessions"][-1]
            if sess_counts:
                last["counts"] = sess_counts["counts"]
                last["openrouter_ids"] = sess_counts["ids"]
            last["finished_at"] = _utcnow()
            last["status"] = str(status)
            last["stage"] = str(stage or last.get("stage") or "in_progress")
            last["outcome"] = run_session_outcome(last["status"], last["stage"])
            run["finished_at"] = last["finished_at"]
            run["result"] = run_merge_result(run)
            run["retried"] = int(run.get("attempts") or 0) > 1
            run["costs"], run["totals"] = self._aggregate_run_costs(run)
            self._prune_runs()
        self.flush()

    @staticmethod
    def _aggregate_run_costs(run: dict) -> tuple:
        """Sum per-provider counters + totals across a run's sessions."""
        agg: dict = {}
        for sess in run.get("sessions") or []:
            for prov, c in (sess.get("counts") or {}).items():
                a = agg.setdefault(prov, _empty_counters())
                for k in ("calls", "in_tokens", "out_tokens", "images"):
                    a[k] = int(a.get(k) or 0) + int(c.get(k) or 0)
                a["est_usd"] = round(float(a.get("est_usd") or 0.0) + float(c.get("est_usd") or 0.0), 6)
        totals = {"est_usd": 0.0, "calls": 0, "in_tokens": 0, "out_tokens": 0, "images": 0}
        for prov, a in agg.items():
            totals["est_usd"] = round(totals["est_usd"] + float(a.get("est_usd") or 0.0), 6)
            for k in ("calls", "in_tokens", "out_tokens", "images"):
                totals[k] = int(totals.get(k) or 0) + int(a.get(k) or 0)
        return agg, totals

    def _prune_runs(self) -> None:
        if len(self._runs) <= MAX_RUN_RECORDS:
            return
        for pid in sorted(self._runs, key=lambda p: self._runs[p].get("started_at") or "")[:-MAX_RUN_RECORDS]:
            self._runs.pop(pid, None)

    def run_costs(self, pipeline_id: str) -> Optional[dict]:
        """Aggregated per-provider cost counters for a run (across sessions)."""
        with self._lock:
            run = self._runs.get(pipeline_id)
            if not run:
                return None
            return run.get("costs") or self._aggregate_run_costs(run)

    # ── persistence / reading ────────────────────────────────────────────
    def payload(self) -> dict:
        with self._lock:
            return {
                "schema": "provider_usage/v1",
                "updated_at": _utcnow(),
                "live": {p: dict(c) for p, c in sorted(self._totals.items())},
                "daily": {d: {p: dict(c) for p, c in sorted(provs.items())}
                          for d, provs in sorted(self._daily.items())},
                "provider_pull": {p: dict(v) for p, v in sorted(self._provider_pull.items())},
                "runs": {pid: self._run_payload(pid, run)
                         for pid, run in sorted(self._runs.items())},
            }

    def _run_payload(self, pid: str, run: dict) -> dict:
        run = dict(run)
        run["pipeline_id"] = pid
        run["sessions"] = [
            {k: v for k, v in dict(s).items() if k != "counts"}
            for s in run.get("sessions") or []
        ]
        return run

    def flush(self) -> Optional[dict]:
        if not self._enabled:
            return None
        payload = self.payload()
        _atomic_dump(self.path, payload)
        self._last_save = time.monotonic()
        return payload

    def reset(self) -> None:
        with self._lock:
            self._totals = {}
            self._daily = {}
            self._runs = {}
            self._active_run = None
            self._active_sessions = {}
        self.flush()

    def live_total_usd(self) -> float:
        with self._lock:
            return round(sum(c.get("est_usd", 0.0) for c in self._totals.values()), 6)


def run_session_outcome(status: str, stage: Optional[str]) -> str:
    """Bucket a single pipeline session into an outcome label.

    * failed / retried_success / published / till_upload / aborted.
    ``failed`` -> a raised error; ``published``/``till_upload`` -> terminal
    success stages; anything else that ended without error is ``aborted``
    (e.g. KeyboardInterrupt / Stop-pipeline while a stage was mid-run)."""
    st = (status or "").lower()
    if st == "failed":
        return "failed"
    stage_s = (stage or "").upper()
    if stage_s in ("PUBLISHED_SUCCESS", "PUBLISHED", "DONE"):
        return "published"
    if (stage_s in ("QUALITY_VERIFIED", "PIPELINE_COMPLETE_TILL_UPLOAD")
            or stage_s.startswith("PIPELINE_COMPLETE")):
        return "till_upload"
    if st == "aborted":
        return "aborted"
    return "aborted"


def run_merge_result(run: dict) -> str:
    """Derive the run-level result from its sessions:
    retried_success if a failed/aborted session was followed by success;
    failed if the latest session failed; otherwise the latest session's
    outcome (published / till_upload / aborted / in_progress)."""
    sessions = run.get("sessions") or []
    if not sessions:
        return "in_progress"
    last = run_session_outcome(sessions[-1].get("status", ""), sessions[-1].get("stage"))
    prior_failed = any(run_session_outcome(s.get("status"), s.get("stage")) in ("failed", "aborted")
                       for s in sessions[:-1])
    if last in ("published", "till_upload") and prior_failed:
        return "retried_success"
    return last


# Process-wide ledger used by llm_client / nano_banana / media_cloud + the CLI.
api_usage = ProviderUsageLedger()


# ── provider pulls ───────────────────────────────────────────────────────────

def _http_get(url: str, headers: dict, timeout: float) -> Tuple[int, Any]:
    """Small requests wrapper that never raises."""
    import requests
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        return resp.status_code, body
    except Exception as e:  # noqa: BLE001
        return 0, {"error": str(e)}


def _http_post(url: str, headers: dict, payload: dict, timeout: float) -> Tuple[int, Any]:
    """Small POST wrapper that never raises."""
    import requests
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        return resp.status_code, body
    except Exception as e:  # noqa: BLE001
        return 0, {"error": str(e)}


def fetch_fal_usage(days: int = 7, timeout: float = 15.0) -> dict:
    """Realtime fal usage + credit balance via the fal Platform API.

    Returns a dict with ``ok``, plus ``balance_usd`` / ``used_usd`` /
    ``currency`` / ``period_start`` / ``period_end`` and a ``top`` breakdown
    when the platform API is reachable, else ``ok: False`` with ``error``."""
    key = os.getenv("FAL_KEY", "")
    if not key or key.startswith("YOUR_") or len(key) < 8:
        return {"ok": False, "error": "FAL_KEY not set in .env", "hint": "set FAL_KEY"}
    end = datetime.date.fromisoformat(_utcnow()[:10])
    start = end - datetime.timedelta(days=days)
    headers = {"Authorization": f"Key {key}"}
    url = (f"{FAL_BASE}/models/usage?start={start.isoformat()}&end={end.isoformat()}"
           f"&expand=summary&expand=time_series")
    code, body = _http_get(url, headers, timeout)
    if code != 200 or not isinstance(body, dict):
        return {"ok": False, "error": _err(body) or f"HTTP {code}",
                "hint": "fal Platform API unreachable or key lacks usage access"}
    summary = body.get("summary") or []
    used = round(sum(float(r.get("cost_total") or 0.0) for r in summary), 6)
    top = sorted(summary, key=lambda r: float(r.get("cost_total") or 0.0), reverse=True)[:8]
    result = {
        "ok": True, "source": "fal platform api", "fetched_at": _utcnow(),
        "period_start": start.isoformat(), "period_end": end.isoformat(),
        "period_days": days, "used_usd": used, "currency": "USD",
        "top": [{"endpoint": r.get("endpoint_id"), "quantity": r.get("quantity"),
                 "unit": r.get("unit"), "cost_usd": r.get("cost_usd")} for r in top],
    }
    # Lifetime/cumulative usage ("used so far") — wide window, fal clamps it to
    # the account's billing history.
    life_code, life_body = _http_get(
        f"{FAL_BASE}/models/usage?start=2000-01-01&end={end.isoformat()}&expand=summary",
        headers, timeout)
    if life_code == 200 and isinstance(life_body, dict):
        life_summary = life_body.get("summary") or []
        result["used_lifetime_usd"] = round(sum(float(r.get("cost_total") or 0.0)
                                                for r in life_summary), 6)
    bill_code, bill = _http_get(f"{FAL_BASE}/account/billing?expand=credits", headers, timeout)
    if bill_code == 200 and isinstance(bill, dict):
        creds = bill.get("credits") or {}
        result["balance_usd"] = creds.get("current_balance")
        result["currency"] = creds.get("currency") or result.get("currency")
    return result


def fetch_openrouter_usage(timeout: float = 15.0, analytics_days: int = 7) -> dict:
    """OpenRouter credits + usage via /credits (Bearer <API key>), the run-key's
    OWN usage via /auth/key, plus a per-model breakdown filtered to that key via
    /analytics/query (management key).

    The regular ``sk-or-...`` key legitimately returns /auth/key (its own usage,
    weekly/monthly/daily + limit) and /credits; per-key history needs a
    management key. The analytics results are filtered to the pipeline's own
    key (OPENROUTER_FILTER_API_KEY, default ``buzzdropfeedv2`` = the .ea4 key)."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    mgmt_key = os.getenv("OPENROUTER_MANAGEMENT_KEY", "")
    key = mgmt_key or api_key
    if not key or len(key) < 5:
        return {"ok": False, "error": "OPENROUTER_API_KEY not set in .env",
                "hint": "set OPENROUTER_API_KEY (or a pk_ management key)"}
    status, body = _http_get(f"{OPENROUTER_BASE}/credits",
                             {"Authorization": f"Bearer {key}"}, timeout)
    if status != 200 or not isinstance(body, dict):
        hint = None
        if status == 403:
            hint = ("/credits needs a management key — create one at openrouter.ai/keys"
                    " and set OPENROUTER_MANAGEMENT_KEY.")
        return {"ok": False, "code": status, "error": _err(body) or f"HTTP {status}", "hint": hint}
    d = body.get("data") or {}
    result = {"ok": True, "source": "openrouter /credits + /auth/key", "fetched_at": _utcnow(),
              "total_credits": d.get("total_credits"),
              "total_usage": d.get("total_usage"),
              "usage_usd": d.get("total_usage"),
              "key_type": "management" if mgmt_key else "regular"}
    if api_key:
        result["run_key"] = fetch_openrouter_key_usage(api_key, timeout=timeout)
    if mgmt_key:
        filter_key = os.getenv("OPENROUTER_FILTER_API_KEY", "buzzdropfeedv2").strip()
        result["analytics"] = fetch_openrouter_analytics(mgmt_key, days=analytics_days,
                                                         api_key_id=filter_key or None,
                                                         timeout=timeout)
    else:
        result["analytics_unavailable"] = True
        result["hint"] = ("regular sk-or- key: /credits + /auth/key only (own usage). "
                          "Set a management key (pk_...) + OPENROUTER_FILTER_API_KEY "
                          "to unlock per-key analytics.")
    return result


def fetch_openrouter_key_usage(key: str, timeout: float = 10.0) -> dict:
    """Usage/limits for THIS api key via /auth/key (works with any key)."""
    status, body = _http_get(f"{OPENROUTER_BASE}/auth/key",
                             {"Authorization": f"Bearer {key}"}, timeout)
    if status == 200 and isinstance(body, dict):
        d = body.get("data") or {}
        return {"ok": True, "label": d.get("label"),
                "usage": d.get("usage"), "usage_daily": d.get("usage_daily"),
                "usage_weekly": d.get("usage_weekly"), "usage_monthly": d.get("usage_monthly"),
                "limit": d.get("limit"), "limit_remaining": d.get("limit_remaining"),
                "limit_reset": d.get("limit_reset")}
    return {"ok": False, "code": status, "error": _err(body) or f"HTTP {status}"}


def fetch_openrouter_analytics(key: str, days: int = 7, api_key_id: Optional[str] = None,
                               timeout: float = 15.0) -> dict:
    """Per-model usage breakdown via OpenRouter /analytics/query (management key
    only). When ``api_key_id`` is given, only rows attributed to that key are
    aggregated (the pipeline's own run key). Metrics: real request_count +
    total_usage ($ cost). Never raises."""
    import datetime as _dt
    start = _dt.date.today() - _dt.timedelta(days=max(1, days))
    payload = {
        "metrics": ["request_count", "total_usage"],
        "dimensions": ["api_key_id", "model"],
        "granularity": "day",
        "time_range": {"start": f"{start.isoformat()}T00:00:00Z",
                       "end": _dt.date.today().isoformat() + "T23:59:59Z"},
        "limit": 200,
    }
    status, resp = _http_post(f"{OPENROUTER_BASE}/analytics/query",
                              {"Authorization": f"Bearer {key}"}, payload, timeout)
    if status != 200 or not isinstance(resp, dict):
        return {"ok": False, "code": status, "error": _err(resp) or f"HTTP {status}",
                "hint": "/analytics/query requires a management (pk_) key"}
    inner = resp.get("data")
    if isinstance(inner, dict):
        rows = inner.get("data") or inner.get("rows") or []
    elif isinstance(inner, list):
        rows = inner
    else:
        rows = resp.get("rows") or []
    by_model: Dict[str, dict] = {}
    total_req = 0
    total_cost = 0.0
    matched_key = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        rid = r.get("api_key_id") or ""
        if api_key_id and rid != api_key_id:
            continue
        matched_key = matched_key or rid
        model = str(r.get("model") or "unknown")
        cnt = int(r.get("request_count") or 0)
        cost = float(r.get("total_usage") or 0.0)
        entry = by_model.setdefault(model, {"requests": 0, "cost_usd": 0.0})
        entry["requests"] += cnt
        entry["cost_usd"] = round(entry["cost_usd"] + cost, 6)
        total_req += cnt
        total_cost = round(total_cost + cost, 6)
    top = sorted(by_model.items(), key=lambda kv: kv[1]["requests"], reverse=True)[:12]
    return {"ok": True, "period_days": days,
            "api_key_id": matched_key or api_key_id or None,
            "total_requests": total_req, "total_cost_usd": total_cost,
            "per_model": [{"model": m, "requests": v["requests"], "cost_usd": v["cost_usd"]}
                          for m, v in top]}


def fetch_google_vertex_billing(project: Optional[str] = None, timeout: float = 15.0) -> dict:
    """Best-effort Vertex billing status via the Google Cloud Billing API.

    Confirms billing is enabled and which account is linked, using Application
    Default Credentials (the same auth llm_client uses for Vertex). It does NOT
    return dollar totals — those require a BigQuery billing export — so Google
    usage numbers come from the live-capture token ledger instead."""
    project = project or os.getenv("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        return {"ok": False, "error": "GOOGLE_CLOUD_PROJECT not set in .env"}
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError:
        return {"ok": False, "error": "google-auth not installed",
                "hint": "pip install google-auth (already a requirement)"}
    try:
        creds, _proj = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        status, body = _http_get(f"{GOOGLE_BILLING_BASE}/projects/{project}/billingInfo",
                                 {"Authorization": f"Bearer {creds.token}"}, timeout)
        if status == 200 and isinstance(body, dict):
            return {"ok": True, "source": "cloudbilling /projects/{p}/billingInfo",
                    "fetched_at": _utcnow(), "project": project,
                    "billing_enabled": bool(body.get("billingEnabled")),
                    "billing_account": body.get("billingAccountName"),
                    "hint": ("raw $ totals require a BigQuery billing export; "
                             "real usage here comes from the live token ledger")}
        return {"ok": False, "code": status,
                "error": _err(body) or f"HTTP {status}", "project": project}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e), "project": project}


def pull_all(days: int = 7, include: Optional[List[str]] = None) -> dict:
    """Fetch authoritative usage from each enabled provider and cache it into
    the ledger's ``provider_pull`` section. ``include`` restricts to a subset
    (e.g. ["fal", "openrouter"]). Never raises; returns the merged payload."""
    want = [p.strip().lower() for p in (include or ["fal", "openrouter", "google"]) if p]
    if "fal" in want:
        api_usage.set_provider_pull("fal", fetch_fal_usage(days=days))
    if "openrouter" in want:
        api_usage.set_provider_pull("openrouter", fetch_openrouter_usage())
    if "google" in want:
        api_usage.set_provider_pull("google", fetch_google_vertex_billing())
    return api_usage.payload()


def _err(body: Any) -> Optional[str]:
    if isinstance(body, dict):
        e = body.get("error")
        if isinstance(e, dict):
            return e.get("message") or str(e)
        if isinstance(e, str):
            return e
        return body.get("message")
    return None