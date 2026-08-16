#!/usr/bin/env python3
"""
Realtime API-usage report for fal, Google (Gemini/Vertex) and OpenRouter.

Queries each provider's own usage/billing API when possible and merges that
with the live per-request usage captured by the pipeline (real token counts
recorded from LLM / Gemini response metadata). Results are cached into
``logs/provider_usage.json`` so the Rust dashboard can render the same numbers.

Usage:
    python get_api_usage.py                      # pull + print table
    python get_api_usage.py --days 30            # 30-day fal report window
    python get_api_usage.py --providers fal,google
    python get_api_usage.py --no-pull            # local ledger only (offline)
    python get_api_usage.py --json               # machine-readable payload
"""
import os
import sys
import json
import argparse
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.engine import api_usage as au


def _fmt_usd(v):
    if v is None:
        return "n/a"
    try:
        return f"${float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_counters(c: dict) -> str:
    c = c or {}
    parts = []
    if c.get("calls"):
        parts.append(f"{c['calls']} calls")
    if c.get("in_tokens"):
        parts.append(f"{c['in_tokens']:,} in-tok")
    if c.get("out_tokens"):
        parts.append(f"{c['out_tokens']:,} out-tok")
    if c.get("images"):
        parts.append(f"{c['images']} img")
    est = c.get("est_usd")
    if est:
        parts.append(f"est {_fmt_usd(est)}")
    return ", ".join(parts) or "no usage recorded"


def _print_provider_pull(name: str, p: dict) -> None:
    if not p:
        print(f"  provider pull : <never pulled>  (run --pull)")
        return
    if not p.get("ok"):
        hint = p.get("hint")
        print(f"  provider pull : ERROR — {p.get('error')}"
              + (f"\n                 hint: {hint}" if hint else ""))
        return
    if name == "fal":
        print(f"  provider pull : {_fmt_usd(p.get('used_usd'))} used "
              f"[{p.get('period_start')} → {p.get('period_end')}], "
              f"balance {_fmt_usd(p.get('balance_usd'))} {p.get('currency','')} ({p.get('source')})")
    elif name == "openrouter":
        rk = p.get("run_key") or {}
        ana = p.get("analytics") or {}
        print(f"  key (buzzdropfeedv2): {_fmt_usd(rk.get('usage_monthly'))} this month, "
              f"{_fmt_usd(rk.get('usage_weekly'))} this week, {_fmt_usd(rk.get('usage_daily'))} today"
              + (f" | limit {_fmt_usd(rk.get('limit'))}, {_fmt_usd(rk.get('limit_remaining'))} remaining ({rk.get('limit_reset')})"
                 if rk.get("limit") else ""))
        if ana and ana.get("ok"):
            top = ana.get("per_model") or []
            print(f"  usage 7d      : key '{ana.get('api_key_id')}' → "
                  f"{ana.get('total_requests', 0)} req, {_fmt_usd(ana.get('total_cost_usd'))} in last {ana.get('period_days')}d"
                  + ("; " + ", ".join(f"{m['model']}={m['requests']}" for m in top[:5]) if top else ""))
        print(f"  account       : balance {_fmt_usd(p.get('total_credits'))} (all keys)")
    elif name == "google":
        ba = p.get("billing_account")
        print(f"  provider pull : project {p.get('project')} · vertex billing {'enabled' if p.get('billing_enabled') else 'DISABLED'}"
              + (f" — {ba}" if ba else ""))
        print(f"                 {p.get('hint') or ''}")


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Realtime fal / Google / OpenRouter API-usage report.")
    ap.add_argument("--days", type=int, default=7, help="fal usage window in days (default 7)")
    ap.add_argument("--providers", default="fal,openrouter,google",
                    help="comma-separated providers to pull (default all)")
    ap.add_argument("--no-pull", action="store_true", help="skip live provider API fetches")
    ap.add_argument("--json", action="store_true", help="print the full payload as JSON")
    ap.add_argument("--runs", type=int, default=10,
                    help="how many per-run cost records to show (default 10)")
    ap.add_argument("--livetotals", action="store_true", help="print only live-capture totals (KEY=value)")
    args = ap.parse_args()

    if not args.no_pull:
        picks = [p.strip() for p in args.providers.split(",") if p.strip()]
        au.pull_all(days=max(1, args.days), include=picks)

    payload = au.api_usage.payload()

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    if args.livetotals:
        for prov, c in sorted(payload["live"].items()):
            print(f"{prov}.calls={c.get('calls', 0)} "
                  f"{prov}.in_tokens={c.get('in_tokens', 0)} "
                  f"{prov}.out_tokens={c.get('out_tokens', 0)} "
                  f"{prov}.images={c.get('images', 0)} "
                  f"{prov}.est_usd={c.get('est_usd', 0.0)}")
        return 0

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   REALTIME API USAGE — fal · Google · OpenRouter                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"ledger: {au.api_usage.path}  (updated {payload.get('updated_at')})")
    for name, label in (("fal", "fal (visuals)"), ("google", "google (gemini/vertex)"),
                        ("openrouter", "openrouter (llm)")):
        counters = payload["live"].get(name, {})
        pull = payload["provider_pull"].get(name)
        print(f"\n▸ {label.upper()}")
        print(f"  live capture : {_fmt_counters(counters)}")
        _print_provider_pull(name, pull)

    _print_runs(payload.get("runs") or {}, args.runs)
    print()
    return 0


def _print_runs(runs: dict, limit: int) -> None:
    if not runs:
        print("\n▸ PER-RUN API COST: no completed runs tracked yet."
              " Each pipeline run's provider cost is attributed to its id as it runs.")
        return
    rows = sorted(runs.items(), key=lambda kv: kv[1].get("finished_at") or kv[1].get("started_at") or "",
                  reverse=True)[:limit]
    print("\n▸ RUNS (provider API cost + outcome)")
    print(f"  {'run_id':<24} {'outcome':<15} {'attempts':<9} {'est.$':>10} "
          f"{'calls':>6} {'in-tok':>9} {'out-tok':>9} {'img':>5}  detail")
    for pid, r in rows:
        t = r.get("totals") or {}
        outcome = r.get("result") or "?"
        suffix = ""
        if outcome == "retried_success":
            suffix = " (fail+success)"
        elif int(r.get("attempts") or 0) > 1 and r.get("retried"):
            suffix = " (retried)"
        sess = r.get("sessions") or []
        id_count = sum(len(s.get("openrouter_ids") or []) for s in sess)
        costs = " ".join(f"{p}~{_fmt_usd(c.get('est_usd'))}" for p, c in (r.get("costs") or {}).items())
        detail = f"or_ids={id_count}" + (f" | {costs}" if costs else "")
        print(f"  {pid:<24} {(outcome + ' ').ljust(15)} {str(int(r.get('attempts') or 0)):<7} "
              f"{_fmt_usd(t.get('est_usd')):>10} {str(t.get('calls', 0)):>6} "
              f"{str(t.get('in_tokens', 0)):>9} {str(t.get('out_tokens', 0)):>9} "
              f"{str(t.get('images', 0)):>5}  {detail}")


if __name__ == "__main__":
    sys.exit(main())