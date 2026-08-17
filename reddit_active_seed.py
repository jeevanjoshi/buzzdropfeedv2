"""Run the active-thread Reddit seeder ON THE PI (residential IP).

Reddit heavily spam-filters / AutoMod-deletes comments posted from datacenter
IPs, so the OCI master must never post directly. The publisher serialises
GlobalState to JSON, ships it here over SSH stdin, and we reconstruct it and run
the seeder locally on the Pi where the residential IP keeps us out of the spam
bucket.

Usage (invoked by publisher.py via SSH):
    python reddit_active_seed.py --state logs/pi_seed_state_<id>.json \
        --url https://www.youtube.com/watch?v=XXXX
"""
import os
os.environ.setdefault("CSVG_LOG_FILENAME", "seeding_execution.log")
import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sys.path.insert(0, ".")

from src.schemas.state import GlobalState
from src.engine.active_thread_seeder import active_thread_seeder
import asyncio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True,
                    help="path to GlobalState JSON written by the publisher")
    ap.add_argument("--url", required=True, help="published YouTube url")
    a = ap.parse_args()
    try:
        with open(a.state) as f:
            state = GlobalState.model_validate(json.load(f))
    except Exception as e:
        logging.error(f"Could not load GlobalState from {a.state}: {e}")
        sys.exit(1)
    asyncio.run(active_thread_seeder.seed_active_discussions(state, a.url))


if __name__ == "__main__":
    main()
