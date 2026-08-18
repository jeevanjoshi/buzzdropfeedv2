#!/usr/bin/env python3
"""Competitor gap engine — consolidate + persist fix-list.

Reads the gap reports already produced by the lightweight benchmark
(``competitor_scraper.py`` -> ``logs/competitor_data.json`` key ``gaps``) and
the deep analyzer (``competitor_deep_analyze.py`` -> ``logs/competitor_deep.json``
key ``deep_gaps``), merges/dedupes them, and emits:

  * ``docs/COMPETITOR_GAPS.md``  — human-readable, checkbox fix-list for later work
  * ``logs/competitor_gaps.json`` — structured, for the dashboard SPA

Run:
    python -m src.engine.competitor_gap_engine \
        --light logs/competitor_data.json \
        --deep  logs/competitor_deep.json \
        --out   docs/COMPETITOR_GAPS.md \
        --json-out logs/competitor_gaps.json

Either input is optional; missing files are skipped gracefully.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3, "PARITY": 4}


@dataclass
class Gap:
    id: str
    severity: str
    area: str
    finding: str
    code_ref: str
    suggestion: str
    source: str  # "light" | "deep"
    status: str = "open"


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _load_gaps(path: Optional[str], source: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except Exception:
        return []
    key = "deep_gaps" if source == "deep" else "gaps"
    gaps = data.get(key) or []
    if not isinstance(gaps, list):
        return []
    for g in gaps:
        g.setdefault("source", source)
    return gaps


def consolidate(light_path: Optional[str], deep_path: Optional[str]) -> List[Gap]:
    """Merge light + deep gaps, dedupe by area, prefer higher severity / deep."""
    raw = _load_gaps(light_path, "light") + _load_gaps(deep_path, "deep")
    by_area: Dict[str, Gap] = {}
    for g in raw:
        sev = (g.get("severity") or "INFO").upper()
        area = (g.get("area") or "unknown").strip()
        finding = g.get("finding") or ""
        code_ref = g.get("code_ref") or ""
        suggestion = g.get("suggestion") or ""
        src = g.get("source") or "light"
        cand = Gap(
            id="",
            severity=sev,
            area=area,
            finding=finding,
            code_ref=code_ref,
            suggestion=suggestion,
            source=src,
        )
        existing = by_area.get(area)
        if existing is None:
            by_area[area] = cand
        else:
            # Keep the higher-severity entry; tie-break to deep source.
            if (SEVERITY_ORDER.get(sev, 9) < SEVERITY_ORDER.get(existing.severity, 9)
                    or (sev == existing.severity and src == "deep")):
                by_area[area] = cand

    gaps = list(by_area.values())
    # Stable id from area + severity.
    for i, g in enumerate(sorted(gaps, key=lambda x: (SEVERITY_ORDER.get(x.severity, 9), x.area))):
        g.id = f"GAP-{i+1:02d}"
    return gaps


def _severity_counts(gaps: List[Gap]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for g in gaps:
        out[g.severity] = out.get(g.severity, 0) + 1
    return out


def write_markdown(gaps: List[Gap], path: str, light_path: str, deep_path: str) -> None:
    counts = _severity_counts(gaps)
    lines: List[str] = []
    lines.append("# Competitor Gap Analysis — Fix List\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")
    lines.append("Auto-consolidated from competitor benchmarks. Use the checkboxes as a "
                 "later-fix backlog; each item points at the exact code location.\n")
    lines.append(f"- Light benchmark: `{light_path or 'n/a'}`")
    lines.append(f"- Deep analysis:   `{deep_path or 'n/a'}`\n")
    lines.append("## Summary\n")
    for sev in ["HIGH", "MEDIUM", "LOW", "INFO", "PARITY"]:
        if counts.get(sev):
            lines.append(f"- **{sev}**: {counts[sev]}")
    lines.append("")
    lines.append("## Action Items (Later Fixes)\n")
    for sev in ["HIGH", "MEDIUM", "LOW", "INFO", "PARITY"]:
        group = [g for g in gaps if g.severity == sev]
        if not group:
            continue
        lines.append(f"### {sev}\n")
        for g in group:
            lines.append(f"- [ ] **{g.area}** — {g.finding}")
            if g.code_ref:
                lines.append(f"  - Code: `{g.code_ref}`")
            if g.suggestion:
                lines.append(f"  - Fix: {g.suggestion}")
            lines.append(f"  - id: `{g.id}` · source: {g.source}")
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def write_json(gaps: List[Gap], path: str) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "severity_counts": _severity_counts(gaps),
        "gaps": [asdict(g) for g in gaps],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Consolidate competitor gaps -> fix-list")
    p.add_argument("--light", default="logs/competitor_data.json", help="lightweight benchmark JSON")
    p.add_argument("--deep", default="logs/competitor_deep.json", help="deep analyzer JSON")
    p.add_argument("--out", default="docs/COMPETITOR_GAPS.md", help="markdown fix-list path")
    p.add_argument("--json-out", default="logs/competitor_gaps.json", help="structured JSON path")
    args = p.parse_args(argv)

    gaps = consolidate(args.light or None, args.deep or None)
    if not gaps:
        print("[gap_engine] no gaps found in inputs; wrote empty reports")
    write_markdown(gaps, args.out, args.light, args.deep)
    write_json(gaps, args.json_out)
    print(f"[gap_engine] {len(gaps)} gaps -> {args.out} + {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
