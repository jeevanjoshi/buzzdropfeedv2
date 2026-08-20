#!/usr/bin/env python
"""
rebuild_shorts.py — Re-cut the Shorts for already-published videos using the NEW
tension-gripping trailer-montage logic (charts/tickers excluded).

WHY: the old Shorts path cropped Act-1/Act-3 *starts* and could include
matplotlib/svg-ticker frames (a Shorts-CTR killer). The montage path now selects
only non-chart narrative beats and xfade-joins them. This regenerates the Short
video files locally from the on-disk master.

NOTE: YouTube cannot replace a Short's video in place. After generating the new
clips you must decide manually whether to upload them as NEW Shorts (and optionally
remove the old ones). This script only produces the local files.

USAGE:
  python rebuild_shorts.py --last 3
  python rebuild_shorts.py --pipeline-id csvg-exec-20260820-112251
  python rebuild_shorts.py --all

Per-video cost: the masters are large and may not have fast-seek keyframes, so
each crop can do a linear decode. On a fast host expect ~1-3 min per Short.
Set CSVG_SHORTS_COVERS=0 to skip the nano-banana cover (fastest) if you only
want the montage.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.schemas.state import GlobalState
from src.engine import micro_content_producer as mc


def _state_files() -> list:
    files = sorted(glob.glob("logs/state_csvg-exec-*.json"), reverse=True)
    # Skip obvious backups (.bak / .bakN).
    return [f for f in files if ".bak" not in f]


def rebuild(pid: str) -> list:
    path = f"logs/state_csvg-exec-{pid}.json"
    if not os.path.exists(path):
        print(f"[rebuild_shorts] {path} not found; skipping.")
        return []
    d = json.load(open(path))
    st = GlobalState.model_validate(d)
    if not (st.asset_paths and st.asset_paths.final_video
            and os.path.exists(st.asset_paths.final_video)):
        print(f"[rebuild_shorts] {pid}: master video missing on disk; skipping.")
        return []
    outdir = f"logs/shorts_recut/{pid}"
    os.makedirs(outdir, exist_ok=True)
    prod = mc.MicroContentProducer(output_dir=outdir)
    clips = prod.generate_shorts(st, max_shorts=2)
    kept = [c for c in clips if c and os.path.exists(c)]
    print(f"[rebuild_shorts] {pid}: wrote {len(kept)} clip(s) -> {outdir}")
    for c in kept:
        print(f"    {c}  ({os.path.getsize(c)//1024} KB)")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=0, help="rebuild the N most recent runs")
    ap.add_argument("--all", action="store_true", help="rebuild every run with a master on disk")
    ap.add_argument("--pipeline-id", action="append", default=[], help="explicit pipeline id(s)")
    args = ap.parse_args()

    pids: list = list(args.pipeline_id)
    files = _state_files()
    if args.all:
        for f in files:
            pid = os.path.basename(f).replace("state_csvg-exec-", "").replace(".json", "")
            pids.append(pid)
    elif args.last:
        for f in files[:args.last]:
            pid = os.path.basename(f).replace("state_csvg-exec-", "").replace(".json", "")
            pids.append(pid)

    if not pids:
        print("No targets. Use --last N, --all, or --pipeline-id <id>.")
        return 1

    total = 0
    for pid in pids:
        total += len(rebuild(pid))
    print(f"[rebuild_shorts] done. {total} Short clip(s) regenerated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
