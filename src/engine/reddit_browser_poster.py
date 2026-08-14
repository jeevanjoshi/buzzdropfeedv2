import os
import time
import json
import random
import logging
import threading
import subprocess
from datetime import date
from typing import List, Dict, Any, Optional

from playwright.sync_api import sync_playwright, Page

logger = logging.getLogger("CSVG_PIPELINE")

# Serialises chromium launches so concurrent triggers (cron + publisher) cannot
# stack several browser trees and OOM the Pi's 4GB.
_BROWSER_LAUNCH_LOCK = threading.Lock()

CONFIG_PATH = os.getenv("REDDIT_ACCOUNTS_FILE", "reddit_accounts.json")
STATE_PATH = os.getenv("REDDIT_ROTATION_STATE", "logs/reddit_rotation_state.json")
SESSION_DIR = os.getenv("REDDIT_SESSION_DIR", "logs/reddit_sessions")
PROFILE_DIR = os.getenv("REDDIT_PROFILE_DIR", "logs/reddit_profiles")
CHROMIUM_PATH = os.getenv("REDDIT_CHROMIUM_PATH", "/usr/bin/chromium")
HEADLESS = os.getenv("REDDIT_HEADLESS", "1").strip().lower() in ("1", "true", "yes")
DEFAULT_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
DEFAULT_TARGET_SUBS = os.getenv(
    "REDDIT_TARGET_SUBREDDITS",
    "IndiaInvestments,IndianStockMarket,IndianStreetBets,IndianStockTalk,AsiaInvesting,personalfinance,trading",
)
MIN_FREE_MEM_MB = int(os.getenv("REDDIT_MIN_FREE_MEM_MB", "500"))
MAX_CHROME_PROCS = int(os.getenv("REDDIT_MAX_CHROME_PROCS", "12"))
RESOURCE_WAIT_SEC = int(os.getenv("REDDIT_RESOURCE_WAIT_SEC", "60"))


class RedditRotationState:
    """Persists per-account daily counts and retired (shadowbanned) status."""

    def __init__(self, path: str = STATE_PATH):
        self.path = path
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.path) as f:
                return json.load(f)
        except Exception:
            return {"day": str(date.today()), "accounts": {}}

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _roll_day(self):
        today = str(date.today())
        if self._data.get("day") != today:
            self._data = {"day": today, "accounts": {}}

    def is_retired(self, username: str) -> bool:
        self._roll_day()
        return self._data["accounts"].get(username, {}).get("retired", False)

    def retire(self, username: str):
        self._roll_day()
        self._data["accounts"].setdefault(username, {})["retired"] = True
        self._save()

    def used_today(self, username: str) -> int:
        self._roll_day()
        return self._data["accounts"].get(username, {}).get("used", 0)

    def record_use(self, username: str):
        self._roll_day()
        acc = self._data["accounts"].setdefault(username, {})
        acc["used"] = acc.get("used", 0) + 1
        self._save()

    def record_unverified(self, username: str) -> bool:
        """Counts a comment that posted but wasn't confirmed visible. Returns
        True when the retirement threshold is reached."""
        self._roll_day()
        acc = self._data["accounts"].setdefault(username, {})
        acc["unverified"] = acc.get("unverified", 0) + 1
        threshold = int(os.getenv("REDDIT_RETIRE_AFTER_UNVERIFIED", "3"))
        self._save()
        return acc["unverified"] >= threshold

    def reset_unverified(self, username: str):
        self._roll_day()
        acc = self._data["accounts"].setdefault(username, {})
        acc["unverified"] = 0
        self._save()

    def record_subreddit(self, subreddit: str, verified: bool):
        """Learns per-subreddit permissiveness: how often comments land (visible)
        vs get filtered. Call after each post."""
        self._roll_day()
        subs = self._data.setdefault("subreddits", {})
        entry = subs.setdefault(subreddit.lower(), {"ok": 0, "filtered": 0})
        if verified:
            entry["ok"] = entry.get("ok", 0) + 1
        else:
            entry["filtered"] = entry.get("filtered", 0) + 1
        self._save()

    def subreddit_stats(self) -> Dict[str, Dict[str, int]]:
        self._roll_day()
        return self._data.get("subreddits", {})

    def record_posted_thread(self, thread_id: str):
        self._roll_day()
        posted = self._data.setdefault("posted_threads", [])
        if thread_id not in posted:
            posted.append(thread_id)
            self._save()

    def was_posted(self, thread_id: str) -> bool:
        self._roll_day()
        return thread_id in self._data.get("posted_threads", [])


class RedditBrowserPoster:
    """
    Fully-automated Reddit comment poster driven via Playwright Chromium on the
    real Chrome-derived browser binary (e.g. the Pi's /usr/bin/chromium).

    Proven flows (live-verified):
      * login via the email/username field on www.reddit.com/login
      * session persisted to logs/reddit_sessions/<user>.json and reused
      * thread discovery by loading the www search page and scraping permalinks
      * commenting via old.reddit.com's plain <textarea name=text> form

    Includes account rotation, per-account daily caps, and a best-effort
    visibility check with automatic shadowban retirement.
    """

    def __init__(self, config_path: str = CONFIG_PATH):
        self.accounts = self._load_accounts(config_path)
        self.state = RedditRotationState()
        self.settings = self._load_settings(config_path)

    def _load_accounts(self, path: str) -> List[Dict[str, Any]]:
        try:
            with open(path) as f:
                return json.load(f).get("accounts", [])
        except Exception as e:
            logger.warning(f"[RedditBrowserPoster] Could not load accounts from {path}: {e}")
            return []

    def _load_settings(self, path: str) -> Dict[str, Any]:
        try:
            with open(path) as f:
                return json.load(f).get("settings", {})
        except Exception:
            return {}

    def has_accounts(self) -> bool:
        return len(self.accounts) > 0

    def _user_id(self, account: Dict[str, Any]) -> str:
        return (account.get("username") or account.get("email") or "default").replace("/", "_")

    def _session_path(self, account: Dict[str, Any]) -> str:
        return os.path.join(SESSION_DIR, self._user_id(account) + ".json")

    def _executable(self) -> Optional[str]:
        if os.path.exists(CHROMIUM_PATH):
            return CHROMIUM_PATH
        return None

    def _proxy(self) -> Optional[Dict[str, Any]]:
        raw = os.getenv("REDDIT_PROXY", self.settings.get("proxy", ""))
        if not raw:
            return None
        return {"server": raw}

    def _pick_account(self) -> Optional[Dict[str, Any]]:
        candidates = []
        for acc in self.accounts:
            u = self._user_id(acc)
            if self.state.is_retired(u):
                continue
            cap = acc.get("daily_cap", self.settings.get("default_daily_cap", 5))
            if self.state.used_today(u) >= cap:
                continue
            candidates.append(acc)
        if not candidates:
            logger.warning("[RedditBrowserPoster] No available account (retired or at daily cap).")
            return None
        candidates.sort(key=lambda a: self.state.used_today(self._user_id(a)))
        return candidates[0]

    @staticmethod
    def _free_mem_mb() -> Optional[int]:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // 1024
        except Exception:
            pass
        return None

    @staticmethod
    def _active_chrome_procs() -> int:
        try:
            out = subprocess.check_output(["pgrep", "-c", "chromium"],
                                          stderr=subprocess.DEVNULL).decode().strip()
            return int(out or 0)
        except Exception:
            return 0

    def _acquire_resources(self) -> bool:
        """Waits (up to RESOURCE_WAIT_SEC) until free RAM and the chromium
        process cap allow launching another browser. Skips when the Pi is too
        loaded so bursts can't OOM it."""
        deadline = time.time() + RESOURCE_WAIT_SEC
        while time.time() < deadline:
            mem = self._free_mem_mb()
            procs = self._active_chrome_procs()
            if (mem is None or mem >= MIN_FREE_MEM_MB) and procs < MAX_CHROME_PROCS:
                return True
            logger.warning(
                f"[RedditBrowserPoster] Resource guard: free={mem}MB procs={procs} "
                f"(need >= {MIN_FREE_MEM_MB}MB, < {MAX_CHROME_PROCS}); backing off."
            )
            time.sleep(5)
        logger.error("[RedditBrowserPoster] Resource guard timed out; skipping this browser op.")
        return False

    def _random_pace(self, s: int):
        time.sleep(s + random.uniform(0.5, 2.0))

    def _open_context(self, account: Dict[str, Any], playwright_inst):
        """Opens a context reusing a saved session if present, else fresh."""
        path = self._session_path(account)
        storage_state = path if os.path.exists(path) else None
        user_agent = account.get("user_agent", DEFAULT_UA)
        if not self._acquire_resources():
            return None
        with _BROWSER_LAUNCH_LOCK:
            ctx = playwright_inst.chromium.launch(
                headless=HEADLESS,
                executable_path=self._executable(),
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            ).new_context(viewport={"width": 1280, "height": 900}, user_agent=user_agent,
                          proxy=self._proxy(), storage_state=storage_state)
        return ctx

    def _login(self, page: Page, account: Dict[str, Any]) -> bool:
        """Logs in via the account email/username and saves the session."""
        login_id = account.get("email") or account.get("username") or ""
        password = account.get("password", "")
        page.goto("https://www.reddit.com/login", wait_until="domcontentloaded")
        self._random_pace(4)
        try:
            page.fill("input[name=username]", login_id, timeout=15000)
            page.fill("input[name=password]", password, timeout=15000)
        except Exception as e:
            logger.warning(f"[RedditBrowserPoster] Login fields not found: {e}")
            return False
        self._random_pace(2)
        try:
            btn = page.locator("button:has-text('Log In')").first
            page.locator("button:has-text('Log In')").first.click(timeout=10000)
        except Exception:
            page.press("input[name=password]", "Enter")
        # Wait for redirection off /login
        try:
            page.wait_for_url(lambda u: "login" not in u, timeout=20000)
        except Exception:
            pass
        self._random_pace(3)
        return "login" not in (page.url or "")

    def search_active_threads(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Scrapes the www search page for recent threads. The search endpoint
        intermittently serves a CAPTCHA shell, so retry a few times and, if it
        still fails, fall back to browsing topic subreddit /new/ pages."""
        import urllib.parse
        with sync_playwright() as pw:
            acc = self.accounts[0] if self.accounts else {}
            if not self._acquire_resources():
                return []
            with _BROWSER_LAUNCH_LOCK:
                ctx = pw.chromium.launch(
                    headless=HEADLESS, executable_path=self._executable(),
                    args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
                ).new_context(viewport={"width": 1280, "height": 900},
                              user_agent=acc.get("user_agent", DEFAULT_UA), proxy=self._proxy())
            results = []
            for attempt in range(3):
                page = ctx.new_page()
                page.goto(
                    f"https://www.reddit.com/search/?q={urllib.parse.quote(query)}&sort=relevance&t=day",
                    wait_until="domcontentloaded", timeout=30000,
                )
                page.wait_for_timeout(6000)
                results = self._extract_threads(page, limit)
                page.close()
                if results:
                    break
                self._random_pace(4)
            if not results:
                # Fallback: browse likely-topic subreddits directly.
                for sub in [s.strip() for s in DEFAULT_TARGET_SUBS.split(",") if s.strip()]:
                    try:
                        page = ctx.new_page()
                        page.goto(f"https://www.reddit.com/r/{sub}/new/",
                                  wait_until="domcontentloaded", timeout=30000)
                        page.wait_for_timeout(5500)
                        hits = self._extract_threads(page, limit)
                        page.close()
                        if hits:
                            results = hits
                            break
                    except Exception:
                        continue
            ctx.close()
            return results

    @staticmethod
    def _extract_threads(page, limit: int) -> List[Dict[str, Any]]:
        perma_hits = page.eval_on_selector_all(
            "a",
            "els => els.map(e => ({href: e.getAttribute('href'), title: (e.getAttribute('title') || e.innerText || '').slice(0,200) }))"
            ".filter(x => x.href && x.href.indexOf('/comments/') >= 0)",
        )
        results = []
        seen = set()
        for h in perma_hits:
            if h["href"] in seen:
                continue
            seen.add(h["href"])
            parts = h["href"].strip("/").split("/")
            if len(parts) < 4:
                continue
            results.append({
                "id": parts[3],
                "subreddit": parts[1],
                "permalink": h["href"],
                "title": h["title"] or "",
                "selftext": "",
                "url": "https://www.reddit.com" + h["href"],
                "num_comments": 0,
            })
            if len(results) >= limit:
                break
        return results

    def get_comments_context(self, subreddit: str, thread_id: str, max_comments: int = 3) -> str:
        """Scrapes top comments from old.reddit for LLM context."""
        try:
            import urllib.request
            url = f"https://old.reddit.com/r/{subreddit}/comments/{thread_id}/"
            req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            import re
            # old.reddit comments: <div class="md"><p>...</p></div> after author names
            authors = re.findall(r'class="author">\s*(\w+)\s*<', html)
            bodies = re.findall(r'<div class="md">\s*<p>(.*?)</p>', html, re.S)
            lines = []
            for a, b in zip(authors[:max_comments], bodies[:max_comments]):
                clean = re.sub(r"<.*?>", "", b)[:300]
                lines.append(f"- Comment by u/{a}: {clean}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"[RedditBrowserPoster] Comment context scrape failed: {e}")
            return ""

    def post_reply(self, thread_id: str, subreddit: str, permalink: str, text: str, force: bool = False) -> bool:
        account = self._pick_account()
        if not account:
            return False
        if not force and self.state.was_posted(thread_id):
            logger.info(f"[RedditBrowserPoster] Thread {thread_id} already posted; skipping.")
            return False
        uid = self._user_id(account)
        logger.info(f"[RedditBrowserPoster] Posting with u/{uid} on r/{subreddit} thread {thread_id}")
        try:
            with sync_playwright() as pw:
                ctx = self._open_context(account, pw)
                if ctx is None:
                    logger.warning("[RedditBrowserPoster] Resource guard blocked post; skipping.")
                    return False
                if not os.path.exists(self._session_path(account)):
                    page = ctx.new_page()
                    if not self._login(page, account):
                        logger.error("[RedditBrowserPoster] Login failed.")
                        ctx.close()
                        return False
                    ctx.storage_state(path=self._session_path(account))
                    logger.info("[RedditBrowserPoster] Session saved.")
                page = ctx.new_page()
                ok = self._post_comment_old(page, permalink, text)
                ctx.close()
            if not ok:
                return False
            self.state.record_use(uid)
            self.state.record_posted_thread(thread_id)
            verified = self._verify_visibility(subreddit, thread_id, text)
            self.state.record_subreddit(subreddit, verified)
            if verified:
                self.state.reset_unverified(uid)
            elif self.settings.get("visibility_check_enabled", True) and self.settings.get("retire_on_shadowban", True):
                if self.state.record_unverified(uid):
                    logger.warning(
                        f"[RedditBrowserPoster] {self.settings.get('retire_after_unverified', os.getenv('REDDIT_RETIRE_AFTER_UNVERIFIED', '3'))} "
                        f"consecutive comments unverified for u/{uid}; retiring account."
                    )
                    self.state.retire(uid)
                    return False
                logger.warning(
                    f"[RedditBrowserPoster] Comment unverified for u/{uid} "
                    f"(might be AutoMod removal; retiring after sustained failures)."
                )
            logger.info(f"[RedditBrowserPoster] Comment posted by u/{uid} (visible={verified}).")
            return verified
        except Exception as e:
            logger.error(f"[RedditBrowserPoster] Posting failed for u/{uid}: {e}")
            return False

    def _post_comment_old(self, page: Page, permalink: str, text: str) -> bool:
        """Posts via old.reddit.com's plain textarea form (previously verified live)."""
        page.goto("https://old.reddit.com" + permalink, wait_until="domcontentloaded")
        self._random_pace(5)
        ta = page.locator("textarea[name=text]").first
        if ta.count() == 0:
            logger.warning("[RedditBrowserPoster] No comment textarea on old.reddit (sub may block low-karma accounts).")
            return False
        ta.focus()
        # Humanized typing: type the text character by character with random delays to mimic human keypress speed
        ta.type(text, delay=random.randint(15, 45))
        self._random_pace(2)

        def _confirm() -> bool:
            import re
            try:
                if ta.count() > 0 and len((ta.input_value() or "").strip()) == 0:
                    logger.info("[RedditBrowserPoster] Comment textarea cleared after submit (success).")
                    return True
            except Exception:
                pass
            body = page.inner_text("body") or ""
            norm = " ".join(re.sub(r"[^a-z0-9]", " ", body.lower()).split())
            probe = " ".join(re.sub(r"[^a-z0-9]", " ", text.lower()).split())[:40]
            return bool(probe) and probe in norm

        submitted = False
        for sel in ["button.save", "button[type=submit]", "input[type=submit]"]:
            try:
                btn = page.locator(sel).first
                btn.wait_for(state="visible", timeout=10000)
                btn.click(timeout=8000)
                submitted = True
                break
            except Exception:
                continue
        if not submitted:
            # last resort: JS form submit on the usertext form
            try:
                page.evaluate("document.querySelector('textarea[name=text]').closest('form').requestSubmit()")
                submitted = True
            except Exception:
                page.keyboard.press("Control+Enter")
                submitted = True
        for _ in range(4):
            self._random_pace(4)
            body_low = ""
            try:
                body_low = (page.inner_text("body") or "").lower()
            except Exception:
                pass
            if "recaptcha" in body_low or "prove your humanity" in body_low:
                logger.warning("[RedditBrowserPoster] Reddit demanded a captcha for this comment.")
            if _confirm():
                return True
        return False

    def _verify_visibility(self, subreddit: str, thread_id: str, text: str) -> bool:
        """Best-effort logged-out check via old.reddit HTML. Punctuation and
        case are stripped on both sides so inline markup can't break matching.
        Returns True on any uncertainty (network/probe failure)."""
        try:
            import urllib.request
            import re
            url = f"https://old.reddit.com/r/{subreddit}/comments/{thread_id}/"
            req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            def norm(s):
                return " ".join(re.sub(r"[^a-z0-9]", " ", re.sub(r"<.*?>", " ", s).lower()).split())
            target = norm(text)
            hay = norm(html)
            probes = [target[:40], target[15:40], target[10:35]]
            return any(p and p in hay for p in probes)
        except Exception as e:
            logger.warning(f"[RedditBrowserPoster] Visibility check failed (assume OK): {e}")
            return True