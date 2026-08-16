#!/usr/bin/env python3
"""Host-aware media/log cleanup with a space-pressure guard.

Keeps disk bounded on BOTH the OCI master (source of truth) and the Raspberry
Pi 5 (dashboard mirror) so a publish never fails from a full disk.

Tiering (media is regenerable; checkpoints/ledgers are the source of truth):
  Tier A  never deleted — state checkpoints (kept newest N for resume),
           run_budget*/provider_usage/quota, published_topics, session/accounts,
           fixed assets (kokoro*.onnx, voices.bin, resources/).
  Tier B  per-run intermediates — logs/media/<run>/ (per-shot audio, visuals,
           .ass subs, clips, and a REDUNDANT final-video copy) — kept newest N.
  Tier C  bounded outputs — logs/final_videos, shorts, baked_shorts,
           link_thumbnails, channel_thumbnails, visual_cache — kept newest N.

Safety:
  * skips anything modified within CSVG_CLEANUP_PROTECTED_MINUTES (the active
    run / freshly-synced mirror), pruning only completed old runs;
  * under space pressure (free < CSVG_CLEANUP_MIN_FREE_GB) force-prunes the
    keep counts down to 1;
  * every prune is best-effort (never raises).

Host-aware defaults — OCI keeps originals, the Pi keeps only what the dashboard
mirrors. Override with --host or the CSVG_CLEANUP_* env knobs.

Usage:
    python cleanup.py [--dry-run] [--host auto|oci|pi] [--force]
"""
import os
import sys
import glob
import time
import logging
import getpass
import shutil
import argparse
import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = os.path.abspath(os.getenv("CSVG_ROOT", os.path.dirname(os.path.abspath(__file__))))
DRY = False  # set True by --dry-run: report sizes, delete nothing

# Host-aware defaults (env CSVG_* override). Pi keeps only the dashboard mirror.
DEFAULTS = {
    "oci": {
        "KEEP_MEDIA_RUNS": 2,
        "KEEP_FINAL_VIDEOS": 5,
        "KEEP_SHORTS": 3,
        "KEEP_VISUAL_CACHE": 2,
        "KEEP_STATE_FILES": 10,
        "MIN_FREE_GB": 15.0,
    },
    "pi": {
        "KEEP_MEDIA_RUNS": 1,
        "KEEP_FINAL_VIDEOS": 1,
        "KEEP_SHORTS": 2,
        "KEEP_VISUAL_CACHE": 2,
        "KEEP_STATE_FILES": 10,
        "MIN_FREE_GB": 3.0,
    },
}


def detect_host() -> str:
    if os.getenv("CSVG_HOST"):
        return os.getenv("CSVG_HOST").strip().lower()
    user = getpass.getuser()
    if user == "jeevanjoshi" or os.path.exists("/home/jeevanjoshi"):
        return "pi"
    return "oci"


def _cfg(host: str, key: str):
    d = DEFAULTS.get(host, DEFAULTS["oci"])
    env_name = "CSVG_CLEANUP_" + key
    default = str(d[key])
    if key in ("MIN_FREE_GB",):
        return float(os.getenv(env_name, default))
    return int(os.getenv(env_name, default))


def _free_gb() -> float:
    try:
        return shutil.disk_usage(ROOT).free / (1024 ** 3)
    except Exception:
        return float("inf")


def _dir_size(path: str) -> int:
    total = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def _protected_seconds() -> float:
    return float(os.getenv("CSVG_CLEANUP_PROTECTED_MINUTES", "30")) * 60.0


def _prune(parent: str, keep: int, is_dir: bool, protected: float,
           pattern: str = "") -> int:
    """Delete all but the `keep` newest entries under `parent`, skipping anything
    modified within `protected` seconds. ``pattern`` overrides the glob (e.g.
    ``state_*.json``) so only the intended files are matched. Returns bytes freed
    (or bytes that WOULD be freed in dry-run)."""
    if not os.path.isdir(parent):
        return 0
    pat = pattern or (os.path.join(parent, "*") if is_dir else os.path.join(parent, "*.json"))
    items = glob.glob(pat)
    if is_dir:
        items = [i for i in items if os.path.isdir(i)]
    items = [i for i in items if time.time() - os.path.getmtime(i) > protected]
    if len(items) <= keep:
        return 0
    items.sort(key=os.path.getmtime, reverse=True)
    freed = 0
    for old in items[keep:]:
        try:
            sz = _dir_size(old) if os.path.isdir(old) else os.path.getsize(old)
            if DRY:
                logging.info("[dry-run] would prune %s (%.1f MB)", old, sz / 1048576)
            else:
                if is_dir:
                    shutil.rmtree(old, ignore_errors=True)
                else:
                    os.remove(old)
                logging.info("pruned %s (%.1f MB)", old, sz / 1048576)
            freed += sz
        except OSError as e:
            logging.warning("failed to prune %s: %s", old, e)
    return freed


def _prune_flat_by_run(parent: str, keep: int, protected: float) -> int:
    """Prune flat output dirs whose filenames embed a run id (e.g.
    ``short_<run>_clip1.mp4``). Groups files per run, keeps the `keep` newest
    run-groups, deletes the rest. Returns bytes freed."""
    if not os.path.isdir(parent):
        return 0
    import re
    rx = re.compile(r"csvg-exec-\d+-\d+")
    groups: dict = {}
    for p in glob.glob(os.path.join(parent, "*")):
        if not os.path.isfile(p):
            continue
        m = rx.search(os.path.basename(p))
        key = m.group(0) if m else os.path.basename(p)
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            continue
        groups.setdefault(key, []).append((p, mtime))
    if not groups:
        return 0
    by_newest = sorted(groups.values(), key=lambda g: max(t for _, t in g), reverse=True)
    active = [g for g in by_newest if max(t for _, t in g) <= time.time() - protected]
    if len(active) <= keep:
        return 0
    freed = 0
    for g in active[keep:]:
        for p, _t in g:
            try:
                sz = os.path.getsize(p)
                if not DRY:
                    os.remove(p)
                freed += sz
            except OSError as e:
                logging.warning("failed to prune %s: %s", p, e)
        logging.info(("%s" % ("[dry-run] would prune" if DRY else "pruned"))
                     + " %d file(s) from %s", len(g), os.path.basename(parent))
    return freed


def _prune_flat_files(parent: str, keep: int, protected: float) -> int:
    """Prune a flat cache dir (no run ids, e.g. visual_cache) keeping the `keep`
    newest files. Returns bytes freed."""
    if not os.path.isdir(parent):
        return 0
    items = [p for p in glob.glob(os.path.join(parent, "*"))
             if os.path.isfile(p) and time.time() - os.path.getmtime(p) > protected]
    if len(items) <= keep:
        return 0
    items.sort(key=os.path.getmtime, reverse=True)
    freed = 0
    for old in items[keep:]:
        try:
            sz = os.path.getsize(old)
            if not DRY:
                os.remove(old)
            logging.info(("%s" % ("[dry-run] would prune" if DRY else "pruned"))
                         + " %s (%.2f MB)", old, sz / 1048576)
            freed += sz
        except OSError as e:
            logging.warning("failed to prune %s: %s", old, e)
    return freed


def _prune_runs(parent: str, keep: int, protected: float) -> int:
    return _prune(parent, keep, is_dir=True, protected=protected)


def main() -> int:
    global DRY
    ap = argparse.ArgumentParser(description="CSVG host-aware media/log cleanup.")
    ap.add_argument("--dry-run", action="store_true", help="report what would be pruned, delete nothing")
    ap.add_argument("--host", choices=["auto", "oci", "pi"], default="auto",
                    help="host policy (default auto-detect)")
    ap.add_argument("--force", action="store_true",
                    help="ignore the space-pressure guard and always use configured keeps")
    args = ap.parse_args()
    DRY = args.dry_run

    host = detect_host() if args.host == "auto" else args.host
    keep_media = _cfg(host, "KEEP_MEDIA_RUNS")
    keep_final = _cfg(host, "KEEP_FINAL_VIDEOS")
    keep_shorts = _cfg(host, "KEEP_SHORTS")
    keep_vis = _cfg(host, "KEEP_VISUAL_CACHE")
    keep_state = _cfg(host, "KEEP_STATE_FILES")
    min_free = _cfg(host, "MIN_FREE_GB")

    # Space guard: if the disk is getting full, tighten to keep=1 unless --force.
    free = _free_gb()
    aggressive = (not args.force) and free < min_free
    if aggressive:
        keep_media = min(keep_media, 1)
        keep_final = min(keep_final, 1)
        keep_shorts = min(keep_shorts, 1)
        keep_vis = min(keep_vis, 1)
        logging.warning("disk low (%.1f GB free < %.0f GB) — aggressive keep=1", free, min_free)

    protected = _protected_seconds()
    if args.dry_run:
        logging.info("[dry-run] host=%s keep_media=%d keep_final=%d keep_shorts=%d keep_vis=%d",
                     host, keep_media, keep_final, keep_shorts, keep_vis)

    start_free = _free_gb()
    freed = 0

    # (parent, keep, kind) — kind: run_dirs (subdir per run), flat_run (files
    # embedding a run id), flat_files (plain cache).
    targets = [
        (os.path.join(ROOT, "logs", "media"), keep_media, "run_dirs"),
        (os.path.join(ROOT, "logs", "final_videos"), keep_final, "run_dirs"),
        (os.path.join(ROOT, "logs", "shorts"), keep_shorts, "flat_run"),
        (os.path.join(ROOT, "logs", "baked_shorts"), keep_shorts, "flat_run"),
        (os.path.join(ROOT, "logs", "visual_cache"), keep_vis, "flat_files"),
        (os.path.join(ROOT, "logs", "link_thumbnails"), 2, "flat_run"),
        (os.path.join(ROOT, "logs", "channel_thumbnails"), 2, "flat_run"),
    ]

    for parent, keep, kind in targets:
        if kind == "run_dirs":
            freed += _prune(parent, keep, is_dir=True, protected=protected)
        elif kind == "flat_run":
            freed += _prune_flat_by_run(parent, keep, protected)
        else:
            freed += _prune_flat_files(parent, keep, protected)

    # state files: keep newest for resume (file-based prune; only state_*.json).
    freed += _prune(os.path.join(ROOT, "logs"), keep_state, is_dir=False,
                    protected=protected, pattern=os.path.join(ROOT, "logs", "state_*.json"))

    end_free = _free_gb()
    logging.info("cleanup done (host=%s): freed %.1f MB, free %.1f GB (was %.1f GB)",
                 host, freed / 1048576, end_free, start_free)
    return 0


if __name__ == "__main__":
    sys.exit(main())