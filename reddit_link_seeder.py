"""Generic, data-driven Reddit link seeder.

Reads ALL published videos from the pipeline (logs/state_*.json), minus any
excluded by seed_campaigns.json, dynamically matches each active discussion
thread to the MOST relevant published video, and posts a genuine on-topic
comment with that video's YouTube link into niche-but-active subreddits.

Designed to be reused by other seeders later: the client (RedditBrowserPoster)
is passed in and everything (config, video discovery, comment generation) is
generic.

Usage:
    python reddit_link_seeder.py                # all published videos
    python reddit_link_seeder.py --video uguUdaOJfw8   # one video
    python reddit_link_seeder.py --max 2        # cap posts this run
"""
import os
os.environ["CSVG_LOG_FILENAME"] = "seeding_execution.log"
import argparse
import glob
import json
import logging
import re
import sys
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sys.path.insert(0, ".")

from src.engine.reddit_browser_poster import RedditBrowserPoster

UA = "Mozilla/5.0 (X11; Linux x86_64) Chrome/120 Safari/537.36"
DEFAULT_CONFIG = os.getenv("REDDIT_SEED_CONFIG", "seed_campaigns.json")
STATE_GLOB = os.getenv("REDDIT_STATE_GLOB", "logs/state_*.json")


# --------------------------------------------------------------------------
# Generic data: published videos from the pipeline
# --------------------------------------------------------------------------
def load_published_videos(exclude_ids=()):
    seen = {}
    for f in sorted(glob.glob(STATE_GLOB)):
        try:
            st = json.load(open(f))
        except Exception:
            continue
        um = st.get("upload_metadata") or {}
        vid = (um.get("video_id") or "").strip()
        if not vid or len(vid) > 64:
            continue
        if vid.lower().startswith("demo") or vid.lower() == "demo_id":
            continue
        if vid in exclude_ids or vid in seen:
            continue
        seo = st.get("seo_metadata") or {}
        title = seo.get("title") or (st.get("script_data") or {}).get("title") or st.get("title") or ""
        kws = (seo.get("keywords") or (st.get("selected_topic") or {}).get("keywords") or [])
        kws = [k for k in kws if isinstance(k, str) and len(k) > 3]
        STOP = {"that","this","with","from","your","what","when","have","will","really","about","there","their","would","could","should","being","been","than","then","them","they","were","which","while","after","before","over","under","into","only","also","just","these","those","where","because","again","more","most","some","very","such","each","other","about","does","doing","made","make","makeup","make"}
        kws = [k for k in kws if k.lower() not in STOP]
        title_tokens = [t for t in re.findall(r"[A-Za-z]{4,}", title or "") if t.lower() not in STOP]
        title_ngrams = [title_tokens[i] + " " + title_tokens[i + 1] for i in range(len(title_tokens) - 1)]
        # Title-derived terms are the most descriptive and get priority.
        kws = list(dict.fromkeys(title_ngrams + title_tokens + kws))
        facts = []
        for vf in (st.get("verified_facts") or [])[:3]:
            if isinstance(vf, dict):
                facts.append(vf.get("summary") or vf.get("headline") or "")
        if title and vid:
            seen[vid] = {
                "id": vid,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "title": title,
                "keywords": [k for k in kws if isinstance(k, str)],
                "facts": [f for f in facts if f],
            }
    return list(seen.values())


def load_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Could not load config {path}: {e}")
        return {}


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def subscriber_count(sub):
    try:
        url = f"https://old.reddit.com/r/{sub}/"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        html = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "replace")
        m = re.search(r"([\d.,]+\s*[km]?)\s+subscribers", html, re.I)
        if not m:
            return None
        s = m.group(1).lower().replace(",", "").strip()
        mult = 1000 if s.endswith("k") else (1000000 if s.endswith("m") else 1)
        if s[-1:] in ("k", "m"):
            s = s[:-1]
        return int(float(s) * mult)
    except Exception:
        return None


def relevance(video, thread, sub=""):
    text = ((thread.get("title") or "") + " " + sub).lower()
    return sum(1 for k in video["keywords"] if k and k.lower() in text)


def generate_comment(video, thread):
    """Dynamic on-topic comment. Uses the LLM when available for a context-aware reply."""
    sub = thread.get("subreddit", "")
    thread_title = thread.get("title", "")
    try:
        from src.engine.llm_client import LLMClient
        llm = LLMClient()
        if llm.is_available():
            system = (
                "You write genuine, helpful Reddit comments that add real value and never "
                "sound like ads. You must return a JSON object with one key \"comment\"."
            )
            facts = "\n".join(f"- {x}" for x in video["facts"][:3]) or "(no extra facts)"
            prompt = (
                f"Thread subreddit: r/{sub}\nThread title: {thread_title}\n\n"
                f"Our related video topic: {video['title']}\n"
                f"Verified facts we know:\n{facts}\n\n"
                "Write ONE organic, on-topic comment that adds value to this discussion. "
                "Naturally embed the reference to our channel \"Lumen Loop\" or the documentary "
                f"url ({video['url']}) directly inside the body of the paragraph (not just at the very end). "
                "If linking directly feels too promotional, mention searching for \"Lumen Loop\" channel on YouTube. "
                "Keep it under 4 sentences, no hashtags, no all-caps."
            )
            res = llm.generate_json(prompt, system_prompt=system, route="generate")
            if res and isinstance(res, dict) and res.get("comment"):
                return res["comment"].strip()
    except Exception as e:
        logging.warning(f"LLM comment failed, using template: {e}")
    return (f"Interesting discussion. The details on {video['title']} are complex — "
            f"you can find a full documentary breakdown of this on the 'Lumen Loop' YouTube channel or at {video['url']}")


# --------------------------------------------------------------------------
# Generic campaign
# --------------------------------------------------------------------------
class LinkCampaign:
    def __init__(self, config_path=DEFAULT_CONFIG, client=None):
        self.config = load_config(config_path)
        self.settings = self.config.get("settings", {})
        self.exclude = set(self.config.get("exclude_video_ids", []))
        self.client = client if client is not None else RedditBrowserPoster()

    def discover_candidates(self, videos):
        """Collect recent active threads across the videos' topics."""
        pooled, seen = [], set()
        for v in videos:
            queries = [v["title"]] + [v["keywords"][i:i + 2] and " ".join(v["keywords"][i:i + 2]) for i in range(0, len(v["keywords"]), 2)]
            for q in queries[:3]:
                try:
                    for t in self.client.search_active_threads(q, limit=3):
                        if t["id"] not in seen:
                            seen.add(t["id"])
                            pooled.append(t)
                except Exception:
                    continue
        return pooled

    def run(self, video_filter=None, max_posts=1):
        videos = load_published_videos(exclude_ids=self.exclude)
        if video_filter:
            videos = [v for v in videos if v["id"] == video_filter]
        if not videos:
            logging.info("No published videos to seed.")
            return
        logging.info(f"Published videos to consider: {len(videos)}")
        for v in videos:
            logging.info(f"  {v['id']}: {v['title'][:60]}")
        candidates = self.discover_candidates(videos)
        if not candidates:
            logging.info("No active candidate threads found this run.")
            return
        niche_max = int(self.settings.get("niche_max_subscribers", 150000))
        min_match = int(self.settings.get("min_keyword_match", 1))
        posted = 0
        for thread in candidates:
            if posted >= max_posts:
                break
            if self.client.state.was_posted(thread["id"]):
                continue
            # pick best video for this thread
            best_video, best_rel = None, 0
            for v in videos:
                r = relevance(v, thread, thread.get("subreddit", ""))
                if r > best_rel:
                    best_video, best_rel = v, r
            if best_video is None or best_rel < min_match:
                continue
            n = subscriber_count(thread["subreddit"])
            if n is not None and n > niche_max:
                logging.info(f"skip r/{thread['subreddit']} (too big: {n})")
                continue
            comment = generate_comment(best_video, thread)
            logging.info(f"POST to r/{thread['subreddit']} [{best_video['id']}]: {thread['title'][:45]}")
            ok = self.client.post_reply(thread["id"], thread["subreddit"], thread["permalink"], comment, force=True)
            if ok:
                posted += 1
        logging.info(f"Done: {posted} link comment(s) posted this run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="", help="publish a single video id")
    ap.add_argument("--max", type=int, default=0, dest="max_posts")
    a = ap.parse_args()
    max_posts = a.max_posts or int(__import__("os").getenv("REDDIT_LINK_MAX_PER_RUN", "1"))
    LinkCampaign().run(video_filter=a.video or None, max_posts=max_posts)


if __name__ == "__main__":
    main()