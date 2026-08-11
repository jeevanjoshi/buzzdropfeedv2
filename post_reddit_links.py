"""One-off ops script: seed Reddit comments linking the Ostrich + Meta videos.

Enhancements over the first version:
  * Relevance scoring: only pick threads whose title/subreddit actually overlap
    the video's topic (avoids off-topic picks like "I found a thing" for an
    archaeology video).
  * Permissive-subreddit targeting: prefer a curated list of niche subs known to
    tolerate new/low-karma commenters, and learn from prior runs which subs
    actually accepted comments (poster.record_subreddit feedback).
  * Skip threads already posted to (tracked in rotation state).
  * Proper video-linked comments via the proven old.reddit poster.
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sys.path.insert(0, ".")
from src.engine.reddit_browser_poster import RedditBrowserPoster

VIDEOS = [
    {
        "name": "ostrich",
        "url": "https://www.youtube.com/watch?v=E1T5IiXSl3E",
        "topic": ["ostrich", "eggshell", "egg", "ancient", "archaeolog", "anthropolog", "prehistoric", "human", "history", "fossil"],
        "queries": ["ostrich eggshell archaeology", "ancient ostrich egg humans",
                    "ostrich egg history", "prehistoric egg archaeological"],
        "perm_subs": ["Archaeology", "Anthropology", "science", "interestingasfuck",
                      "Paleontology", "history", "AskScience", "Ostrich"],
        "comment": ("Fascinating research - the 60,000-year-old ostrich eggshell "
                    "artwork is one of the clearest windows we have into how early "
                    "humans thought and kept social records. Here's a full visual "
                    "breakdown of what the findings reveal: {url}"),
    },
    {
        "name": "meta",
        "url": "https://www.youtube.com/watch?v=uguUdaOJfw8",
        "topic": ["meta", "ai", "model", "llm", "muse", "glimmer", "open", "weight", "zuckerberg", "machine", "learn"],
        "queries": ["Meta Muse Glimmer AI", "Muse Glimmer model benchmarks",
                    "Meta open weight model AI", "Zuckerberg AI Muse Glimmer"],
        "perm_subs": ["LocalLLaMA", "singularity", "artificial", "MachineLearning",
                      "technology", "StableDiffusion", "ArtificialInteligence", "SaaS"],
        "comment": ("Agreed with the core point - the real story with Muse Glimmer is "
                    "whether open-weight local models actually change who gets to run "
                    "serious AI. Here's a visual deep-dive on what it is and whether "
                    "it's really AI for everyone: {url}"),
    },
]


def _rel_score(topic, text):
    text = (text or "").lower()
    score = 0
    for tok in topic:
        if tok in text:
            score += 1
    return score


def _sub_permissiveness(poster, sub):
    stats = poster.state.subreddit_stats().get(sub.lower(), {})
    ok = stats.get("ok", 0)
    filt = stats.get("filtered", 0)
    if ok + filt == 0:
        return 0.0
    return ok / (ok + filt)


def _pick_thread(poster, vid, threads):
    """Score threads by topic relevance + learned sub permissiveness + curated pref."""
    best, best_score = None, -1
    curl = set(s.lower() for s in vid["perm_subs"])
    for t in threads:
        if poster.state.was_posted(t["id"]):
            continue
        rel = _rel_score(vid["topic"], t["title"] + " " + t["subreddit"])
        perm = _sub_permissiveness(poster, t["subreddit"])
        curated = 1.0 if t["subreddit"].lower() in curl else 0.0
        score = (2.0 * rel) + (1.5 * perm) + (1.0 * curated)
        if rel == 0 and curated == 0:
            score -= 3  # strongly penalise irrelevant, non-curated picks
        if score > best_score:
            best, best_score = t, score
        logging.info(f"[pick] r/{t['subreddit']} rel={rel} perm={perm:.2f} curated={curated} score={score:.1f}")
    return best


def main():
    poster = RedditBrowserPoster()
    for vid in VIDEOS:
        print(f"\n===== {vid['name']} =====")
        all_threads = []
        for q in vid["queries"]:
            try:
                all_threads.extend(poster.search_active_threads(q, limit=4))
            except Exception as e:
                logging.warning(f"[{vid['name']}] search '{q}' failed: {e}")
        # dedupe
        seen, uniq = set(), []
        for t in all_threads:
            if t["id"] not in seen:
                seen.add(t["id"])
                uniq.append(t)
        target = _pick_thread(poster, vid, uniq)
        if not target:
            print(f"[{vid['name']}] no suitable thread; skipping")
            continue
        print(f"[{vid['name']}] target r/{target['subreddit']}: {target['title'][:60]}")
        comment = vid["comment"].format(url=vid["url"])
        ok = poster.post_reply(target["id"], target["subreddit"], target["permalink"], comment)
        print(f"[{vid['name']}] POST RESULT: {ok}")
        if not ok:
            print(f"[{vid['name']}] manual check visible:",
                  poster._verify_visibility(target["subreddit"], target["id"], comment))
        print(f"[{vid['name']}] subreddit learn state:",
              poster.state.subreddit_stats().get(target["subreddit"].lower(), {}))


if __name__ == "__main__":
    main()