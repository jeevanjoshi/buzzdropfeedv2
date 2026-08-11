"""Pi log/media maintenance: keeps disk bounded automatically.

Removes:
  * old logs/media run dirs (keeps the newest K, default 2 - the active run and
    a spare for resume)
  * stale final video copies (the OCI host holds the originals; the Pi only
    mirrors for the dashboard)
  * old pipeline state/script/trajectory files (resume only needs the latest)

Safe: the runtime keeps writing to these paths; cleanup only removes completed
old runs. Sources of truth (state/checkpoints, session, accounts) are never
touched here.
"""
import glob
import logging
import os
import shutil
import subprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = os.getenv("CSVG_ROOT", os.path.dirname(os.path.abspath(__file__)))
KEEP_MEDIA_RUNS = int(os.getenv("CSVG_KEEP_MEDIA_RUNS", "2"))
KEEP_STATE_FILES = int(os.getenv("CSVG_KEEP_STATE_FILES", "10"))


def _free_gb():
    st = shutil.disk_usage(ROOT)
    return st.free / (1024 ** 3)


def _prune_dir(parent: str, keep: int, sort_key):
    """Keep `keep` newest entries by sort_key(file); delete older ones."""
    items = glob.glob(os.path.join(parent, "*"))
    items = [i for i in items if os.path.isdir(i)]
    if len(items) <= keep:
        return 0
    items.sort(key=sort_key, reverse=True)
    freed = 0
    for old in items[keep:]:
        try:
            sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(old) for f in fs)
            shutil.rmtree(old, ignore_errors=True)
            freed += sz
            logging.info(f"pruned {old} ({sz / 1048576:.1f} MB)")
        except Exception as e:
            logging.warning(f"failed to prune {old}: {e}")
    return freed


def main():
    start_free = _free_gb()
    freed = 0
    media = os.path.join(ROOT, "logs", "media")
    if os.path.isdir(media):
        freed += _prune_dir(media, KEEP_MEDIA_RUNS, os.path.getmtime)
    fv = os.path.join(ROOT, "logs", "final_videos")
    if os.path.isdir(fv):
        freed += _prune_dir(fv, 1, os.path.getmtime)
    # old scrape/temporary dirs we never need on the Pi
    for name in ("visual_cache",):
        p = os.path.join(ROOT, "logs", name)
        if os.path.isdir(p):
            freed += _prune_dir(p, 2, os.path.getmtime)
    # state files: keep only the newest for resume
    states = sorted(glob.glob(os.path.join(ROOT, "logs", "state_*.json")), key=os.path.getmtime, reverse=True)
    for old in states[KEEP_STATE_FILES:]:
        try:
            os.remove(old)
        except Exception:
            pass
    end_free = _free_gb()
    logging.info(
        f"cleanup done: freed {freed / 1048576:.1f} MB, free now {end_free:.1f} GB "
        f"(was {start_free:.1f} GB)."
    )


if __name__ == "__main__":
    main()