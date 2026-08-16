#!/usr/bin/env python3
"""
One-off / cron CLI to refresh the analytics feedback store.

Pulls per-video growth metrics (views, watch time, average view percentage,
subscribers gained) from the YouTube Analytics API v2 for every video the
pipeline has published, correlates them back to their topic niche, and writes
logs/analytics_feedback.json (consumed by FactRetriever's "top growth drivers"
topic-selection bias and surfaced via the dashboard /api/analytics endpoint).

Usage:
    python run_analytics_feedback.py [--force] [--max-videos N]

    --force          bypass the rate-limit (default refresh is a no-op if the
                     store was captured within the last 6h).
    --max-videos N   cap how many videos to pull (default 30) to bound quota.
"""
import os
import sys
import argparse
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Refresh the YouTube analytics feedback store.")
    parser.add_argument("--force", action="store_true", help="Bypass the refresh rate-limit.")
    parser.add_argument("--max-videos", type=int, default=30, help="Max videos to pull per refresh.")
    args = parser.parse_args()

    from src.engine.analytics_feedback import analytics_feedback, get_niche_signal

    print(f"=== Analytics Feedback Refresh (force={args.force}, max_videos={args.max_videos}) ===")
    store = analytics_feedback.refresh(max_videos=args.max_videos, force=args.force)
    captured = store.get("captured_at", "")
    if not captured:
        print("[AnalyticsFeedback] No refresh performed (rate-limited or no data available).")
        return 0

    signal = get_niche_signal()
    print(f"Store captured at: {captured}")
    print(f"Videos tracked: {len(store.get('videos', []))}")
    print("Niche signal (audience_type -> signal):")
    for aud, b in signal.items():
        print(f"  {aud:16s} videos={b['videos']:>3}  views={b['views']:>8}  "
              f"subs_gained={b['subscribers_gained']:>5}  retention={b['retention']:>5.1f}%  signal={b['signal']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())