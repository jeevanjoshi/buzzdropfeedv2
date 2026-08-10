"""Per-run cost & quota budget tracking.

Records roughly how much budget a single live pipeline run consumes across the
paid services (LLM, RAG search, AI visuals, Google grounding) in estimated USD,
plus the YouTube Data API quota units it spends (search.list / videos.list /
upload / channels.list). Persists per run to ``logs/run_budget_<id>.json`` and
accumulates a rolling aggregate at ``logs/run_budget.json`` so monthly / cross-run
totals can be inspected without reading every per-run file.

During a live run a background thread ALSO flushes the in-progress record every
``RUN_BUDGET_FLUSH_SECS`` seconds (default 4) so the dashboard can show live
budget + current stage rather than only the final write.

Estimates are deliberately conservative and env-tunable (see ``*_USD`` / caps
below) so they stay useful as a health/budget check even though real provider
billing varies. Every record call is a no-op when no run is active, and every
save is failure-tolerant (never raises / never breaks the pipeline).
"""

import os
import json
import time
import threading
import datetime
from typing import Optional

# Estimated per-call/unit costs (USD). Env-tunable for calibration.
LLM_PER_CALL_USD = float(os.getenv("BUDGET_LLM_PER_CALL_USD", "0.01"))
RAG_PER_CALL_USD = float(os.getenv("BUDGET_RAG_PER_CALL_USD", "0.002"))
VISUAL_PER_IMAGE_USD = float(os.getenv("BUDGET_VISUAL_PER_IMAGE_USD", "0.003"))
GROUNDING_PER_CALL_USD = float(os.getenv("BUDGET_GROUNDING_PER_CALL_USD", "0.005"))

# YouTube Data API v3 quota cost in units per operation (free tier ~10k/day).
YT_UNITS = {
    "search": 100,          # search.list
    "videos_batch": 1,      # videos.list (batch)
    "channels": 1,          # channels.list
    "videos_update": 50,    # videos.update (metadata)
    "upload": 1600,         # videos.insert (resumable upload)
}

BUDGET_LOGS_DIR = os.getenv("BUDGET_LOGS_DIR", "logs")
AGGREGATE_FILE = os.path.join(BUDGET_LOGS_DIR, "run_budget.json")
FLUSH_INTERVAL_SECS = float(os.getenv("RUN_BUDGET_FLUSH_SECS", "4"))


class RunBudget:
    """Accumulates per-run budget usage; start() resets, save() finalises."""

    def __init__(self):
        self._active = False
        self._pipeline_id = None
        self._started_at = None
        self._stage = "in_progress"
        self._usd: dict = {}
        self._yt: dict = {}
        self._calls: dict = {}
        self._notes: list = []
        self._lock = threading.Lock()
        self._flush_thread = None
        self.logs_dir = os.getenv("BUDGET_LOGS_DIR", "logs")

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self, pipeline_id: Optional[str] = None) -> None:
        with self._lock:
            self._active = True
            self._pipeline_id = pipeline_id
            self._stage = "in_progress"
            self._started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._usd = {}
            self._yt = {}
            self._calls = {}
            self._notes = []
        self._start_flush_thread()

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self._stage = str(stage)

    def save(self, pipeline_id: Optional[str] = None, status: str = "in_progress",
             stage: str = "", extra: Optional[dict] = None) -> Optional[dict]:
        """Finalise and persist the run budget + update the rolling aggregate."""
        rec = self._build_rec(status=status, stage=stage or self._stage, extra=extra)
        out = self._write(rec, pipeline_id, status)
        with self._lock:
            self._active = False  # stops the live-flush thread
        return out

    def flush(self) -> Optional[dict]:
        """Persist the IN-PROGRESS record so the dashboard shows live numbers."""
        with self._lock:
            if not self._active:
                return None
            started_at = self._started_at
            pipeline_id = self._pipeline_id
        rec = self._build_rec(status="in_progress", stage=self._stage, extra=None)
        return self._write(rec, pipeline_id, "in_progress")

    def _build_rec(self, status: str, stage: str, extra: Optional[dict]) -> dict:
        with self._lock:
            usd, yt, calls = dict(self._usd), dict(self._yt), dict(self._calls)
            rid, started_at = self._pipeline_id, self._started_at
        categories = set(usd) | set(yt) | set(calls)
        rec = {
            "pipeline_id": rid,
            "started_at": started_at,
            "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": status,
            "stage": stage,
            "categories": {
                cat: {
                    "est_usd": round(usd.get(cat, 0.0), 6),
                    "yt_units": int(yt.get(cat, 0)),
                    "calls": int(calls.get(cat, 0)),
                }
                for cat in sorted(categories)
            },
            "totals": {
                "est_usd": round(sum(usd.values()), 6),
                "yt_units": int(sum(yt.values())),
            },
            "notes": list(self._notes),
        }
        if extra:
            rec.update(extra)
        return rec

    def _write(self, rec: dict, pipeline_id: Optional[str], _status: str) -> Optional[dict]:
        rid = (pipeline_id or rec.get("pipeline_id")) or "unknown"
        try:
            os.makedirs(self.logs_dir, exist_ok=True)
            aggregate_file = os.path.join(self.logs_dir, "run_budget.json")
            with self._lock:
                per_run = os.path.join(self.logs_dir, f"run_budget_{rid}.json")
                self._atomic_dump(per_run, rec)
                agg: dict = {}
                if os.path.exists(aggregate_file):
                    try:
                        with open(aggregate_file, "r", encoding="utf-8") as f:
                            agg = json.load(f)
                    except (json.JSONDecodeError, IOError, OSError):
                        agg = {}
                rec["pipeline_id"] = rid
                agg[rid] = rec
                self._atomic_dump(aggregate_file, agg)
                return rec
        except (IOError, OSError):
            return None

    @staticmethod
    def _atomic_dump(path: str, data: dict) -> None:
        """Write JSON atomically (temp file + rename) so a concurrent reader /
        rsync never sees a half-written file."""
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def _start_flush_thread(self) -> None:
        thr = self._flush_thread
        if thr is not None and thr.is_alive():
            return
        t = threading.Thread(target=self._flush_loop, name="run-budget-flush", daemon=True)
        self._flush_thread = t
        t.start()

    def _flush_loop(self) -> None:
        while True:
            time.sleep(FLUSH_INTERVAL_SECS)
            with self._lock:
                if not self._active:
                    return
            self.flush()

    # ── recording ──────────────────────────────────────────────────────────
    def record(self, category: str, usd: float = 0.0, yt_units: int = 0,
               calls: int = 1, note: Optional[str] = None) -> None:
        with self._lock:
            if not self._active:
                return
            self._usd[category] = round(self._usd.get(category, 0.0) + usd, 6)
            self._yt[category] = int(self._yt.get(category, 0)) + int(yt_units)
            self._calls[category] = int(self._calls.get(category, 0)) + int(calls)
            if note:
                self._notes.append(note)

    def record_llm(self) -> None:
        self.record("llm", usd=LLM_PER_CALL_USD)

    def record_search(self, provider: str = "") -> None:
        self.record("search", usd=RAG_PER_CALL_USD, note=f"search:{provider}" if provider else None)

    def record_visual(self) -> None:
        self.record("visuals", usd=VISUAL_PER_IMAGE_USD)

    def record_grounding(self) -> None:
        self.record("grounding", usd=GROUNDING_PER_CALL_USD)

    def record_yt(self, kind: str) -> None:
        units = int(YT_UNITS.get(kind, 0))
        self.record("youtube", yt_units=units, note=f"yt.{kind}:{units}")

    # ── summary for logging ────────────────────────────────────────────────
    def totals(self) -> dict:
        with self._lock:
            return {
                "est_usd": round(sum(self._usd.values()), 6),
                "yt_units": int(sum(self._yt.values())),
                "categories": sorted(set(self._usd) | set(self._yt) | set(self._calls)),
            }


run_budget = RunBudget()
