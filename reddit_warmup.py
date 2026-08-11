"""Low-volume, NO-LINK warmup seeder for the Pi (builds account trust so link
comments eventually stick). Runs on the residential Pi where Reddit renders.

Called either by the publisher (remotely via SSH) with the just-published video's
title/facts, or by cron for steady baseline activity. Deliberately posts only
informative, non-promotional comments on permissive subreddits, respecting the
per-account daily cap, skipping already-posted threads, and learning per-sub
permissiveness for future link posting.
"""
import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sys.path.insert(0, ".")
from src.engine.reddit_browser_poster import RedditBrowserPoster

# Curated niche subs likely to tolerate low-karma commenters, grouped by domain.
# If the video topic matches a domain, that list is preferred.
PERMISSIVE_SUBS = {
    "tech": ["LocalLLaMA", "ollama", "StableDiffusion", "technology", "singularity", "artificial", "MachineLearning"],
    "science": ["Archaeology", "Anthropology", "science", "interestingasfuck", "Paleontology", "AskScience", "history"],
    "finance": ["IndiaInvestments", "IndianStreetBets", "personalinvesting", "IndianStockTalk", "AsiaInvesting"],
    "general": ["AskReddit", "explainlikeimfive", "todayilearned", "CasualConversation"],
}

# Generic, genuinely useful no-link comment banks per domain (rotated).
NO_LINK_BANK = {
    "tech": [
        "One angle often missed is how open-weight models shift who can actually deploy this - smaller teams get real capability without big API bills.",
        "The interesting part is the local/offline angle - for many use cases being able to run it in-house matters more than raw benchmarks.",
        "Good thread. The long-term question is whether the ecosystem stays open or fragments into wrappers around a few big providers.",
    ],
    "science": [
        "Studies like this are a reminder of how much careful dating and context matter - small finds can rewrite what we thought about early humans.",
        "The storytelling on this is great, but the real value is in the methodology - replicability and sample size are what make it convincing.",
        "Fascinating. Context and preservation are everything with material this old, and the conclusions tend to be more cautious than the headlines.",
    ],
    "finance": [
        "One piece of advice I keep coming back to: keep an emergency fund liquid before increasing equity exposure, and stay consistent through volatility.",
        "Consistency and a time horizon beat trying to time the market - regular investing through the dips is what compounds best.",
        "For new investors, a diversified index SIP plus a disciplined emergency fund is boring but extremely effective over the long run.",
    ],
    "general": [
        "This is a great point - the more I read about it the more it comes down to context and trade-offs rather than a single right answer.",
        "Interesting perspective. I think the nuance matters here and the details in the thread cover it well.",
    ],
}


def detect_domain(title: str, keywords) -> str:
    title = (title or "").lower()
    if any(k in title for k in ["ai", "model", "llm", "machine learning", "open weight", "chatbot", "neural"]):
        return "tech"
    if any(k in title for k in ["archaeolog", "anthropolog", "human", "egg", "fossil", "ancient", "ostrich", "history", "science"]):
        return "science"
    if any(k in title for k in ["market", "stock", "invest", "finance", "econom", "sensex", "rbi", "budget", "mutual"]):
        return "finance"
    return "general"


def search_for(poster, query: str, limit: int = 4):
    return poster.search_active_threads(query, limit=limit)


def pick_thread(poster, threads, perm_subs, title_kws):
    best, best_score = None, -1
    curl = set(s.lower() for s in perm_subs)
    for t in threads:
        if poster.state.was_posted(t["id"]):
            continue
        rel = sum(1 for k in (title_kws or []) if k.lower() in (t["title"] or "").lower())
        perm = poster.state.subreddit_stats().get(t["subreddit"].lower(), {}).get("ok", 0)
        curated = 2.0 if t["subreddit"].lower() in curl else 0.0
        score = rel + curated + perm
        if score > best_score:
            best, best_score = t, score
    return best


def run(title: str, facts, count: int):
    poster = RedditBrowserPoster()
    if not poster.has_accounts():
        logging.warning("No accounts configured; skipping warmup.")
        return
    domain = detect_domain(title, [])
    perm_subs = PERMISSIVE_SUBS.get(domain, PERMISSIVE_SUBS["general"])
    logging.info(f"Warmup domain={domain} permissive_subs={perm_subs}")
    fact = (facts[0] if facts else "")
    posted = 0
    for sub in perm_subs:
        if posted >= count:
            break
        query = title or sub
        try:
            threads = search_for(poster, query, limit=3) or search_for(poster, sub, limit=3)
        except Exception as e:
            logging.warning(f"search fail {sub}: {e}")
            continue
        if not threads:
            continue
        tgt = pick_thread(poster, threads, perm_subs, [*(title or "").replace("-", " ").split(), sub])
        if not tgt:
            continue
        bank = NO_LINK_BANK.get(domain, NO_LINK_BANK["general"])
        comment = bank[posted % len(bank)]
        if fact:
            comment = f"{comment} {fact}".strip()
        ok = poster.post_reply(tgt["id"], tgt["subreddit"], tgt["permalink"], comment)
        logging.info(f"Warmup post to r/{tgt['subreddit']}: {ok}")
        if ok:
            posted += 1
    logging.info(f"Warmup done: {posted} comment(s) posted.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="")
    ap.add_argument("--facts", default="[]")
    ap.add_argument("--count", type=int, default=3)
    a = ap.parse_args()
    try:
        facts = json.loads(a.facts) if a.facts else []
    except Exception:
        facts = []
    run(a.title, facts, a.count)


if __name__ == "__main__":
    main()