import uuid
import os
import json
import datetime
from typing import Dict, Any, Optional
from src.schemas.state import GlobalState, UploadMetadata
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent, compute_state_hash
from mcp_servers.youtube_cloud.server import (
    check_quota_available,
    upload_youtube_resumable,
    upload_short,
    insert_pinned_comment,
    upsert_playlist_add_video,
    QuotaCheckRequest,
    UploadRequest,
    InsertCommentRequest,
    UpsertPlaylistRequest,
)
from src.engine.run_budget import run_budget
from src.engine.youtube_engagement import youtube_engagement


# Fake video identifiers that must NEVER be treated as a successful publish.
# A `demo_*`/sentinel id means the upload silently fell back to the mock path
# (token failure, 403, quota, network) — recording it as PUBLISHED_SUCCESS masks
# a real outage and poisons topic dedup (see publish-integrity-quality-fix-plan.md
# issue #1). Reject them everywhere: publish, resume guard, and side effects.
def _is_real_video_id(video_id: Optional[str]) -> bool:
    if not video_id:
        return False
    vid = str(video_id).strip()
    if not vid or len(vid) > 64:
        return False
    return vid.lower() not in ("demo_id", "uploaded_demo_id") and not vid.lower().startswith("demo_")


# ─────────────────────────────────────────────────────────────────────────────
# Themed outcome-based playlist mapping (vidIQ growth tactic: build bingeable
# playlists grouped by the topic's audience/niche so the subscriber path
# (discovery → related playlist → channel) is frictionless within a theme).
# Themed-playlist routing now reads from the consolidated audience taxonomy
# (src/engine/audience_taxonomy.py), so the (title, description) per audience_type
# lives in one place and matches the playlists that already exist on the channel.
# ``_outcome_playlist_for`` resolves via ``playlist_for``.
# ─────────────────────────────────────────────────────────────────────────────

# Optional explicit override: when set, every video is chained into this single
# playlist title instead of the theme-matched one.
_MASTER_PLAYLIST_TITLE = os.getenv("YOUTUBE_PLAYLIST_TITLE", "")

# Channel handle for the subscribe deep-link baked into seed comments.
_SUBSCRIBE_URL = os.getenv(
    "YOUTUBE_SUBSCRIBE_URL",
    "https://www.youtube.com/@lumenloop-ai?sub_confirmation=1",
).strip()


def _outcome_playlist_for(audience_type: str) -> Optional[tuple]:
    """Return (title, description) for the themed playlist matching an
    audience_type, or None when the topic has no themed playlist."""
    if os.getenv("YOUTUBE_PLAYLIST_TITLE", "").strip():
        return (_MASTER_PLAYLIST_TITLE.strip(),
                 "Documentary storytelling series on global trends, finance and innovation.")
    if not audience_type:
        return None
    from src.engine.audience_taxonomy import playlist_for
    return playlist_for(audience_type)


class PublisherAgent:
    """
    Publisher Agent managing YouTube MCP upload tools, checking daily API quotas,
    injecting synthetic content metadata tags, and updating GlobalState.
    """

    def __init__(self, name: str = "Publisher"):
        self.name = name

    def _trigger_pi_warmup(self, title: str, state: GlobalState) -> bool:
        """SSH to the Pi edge node and run the no-link Reddit warmup for the
        just-published video, fire-and-forget so publish never blocks. Reads Pi
        connection config from env (mirrors sync_to_pi.sh defaults). Enabled by
        default; set REDDIT_PI_WARMUP_ON_PUBLISH=0 to disable. Non-fatal."""
        import subprocess
        if os.getenv("REDDIT_PI_WARMUP_ON_PUBLISH", "1").strip().lower() in ("0", "false", "no"):
            return False
        pi_host = os.getenv("PI5_IP", "100.108.116.100")
        pi_user = os.getenv("PI5_USER", "jeevanjoshi")
        pi_dir = os.getenv("PI5_TARGET_DIR", "/home/jeevanjoshi/buzzdropfeedv2")
        count = int(os.getenv("REDDIT_WARMUP_COUNT", "3"))
        title_json = json.dumps(title).replace('"', '\\"')
        py = (
            f"cd {pi_dir} && source venv/bin/activate && "
            f"python reddit_warmup.py --count {count}"
        )
        if title:
            py += f' --title "{title_json}"'
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
               f"{pi_user}@{pi_host}", py]
        try:
            devnull = open(os.devnull, "w")
            import threading
            def _run():
                try:
                    subprocess.run(cmd, stdout=devnull, stderr=devnull, timeout=600)
                except Exception as e:
                    print(f"[PiWarmup] background trigger failed: {e}")
                finally:
                    devnull.close()
            threading.Thread(target=_run, daemon=True).start()
            print("[PiWarmup] launched on the Pi in background (fire-and-forget).")
            return True
        except Exception as e:
            print(f"[PiWarmup] trigger setup failed: {e}")
            return False

    def _trigger_pi_active_seed(self, state: GlobalState, youtube_url: str) -> bool:
        """SSH to the Pi edge node and run the active-thread Reddit seeder there
        (residential IP). The OCI master must NOT post to Reddit directly —
        datacenter IPs get spam-filtered / AutoMod-deleted. We serialise
        GlobalState to JSON, ship it over the SSH stdin into a file on the Pi,
        then run reddit_active_seed.py which reconstructs the state and seeds.
        Fire-and-forget so publish never blocks. Non-fatal."""
        import subprocess
        import threading
        if os.getenv("REDDIT_PI_ACTIVE_SEED", "1").strip().lower() in ("0", "false", "no"):
            return False
        pi_host = os.getenv("PI5_IP", "100.108.116.100")
        pi_user = os.getenv("PI5_USER", "jeevanjoshi")
        pi_dir = os.getenv("PI5_TARGET_DIR", "/home/jeevanjoshi/buzzdropfeedv2")
        try:
            state_json = state.model_dump_json()
        except Exception as e:
            print(f"[PiActiveSeed] could not serialise state: {e}")
            return False
        pid = (state.pipeline_id or "run").replace("/", "_")
        remote = f"{pi_dir}/logs/pi_seed_state_{pid}.json"
        py = (
            f"cd {pi_dir} && source venv/bin/activate && "
            f"cat > {remote} && python reddit_active_seed.py --state {remote} --url {youtube_url}"
        )
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
               f"{pi_user}@{pi_host}", py]
        try:
            devnull = open(os.devnull, "w")
            def _run():
                try:
                    subprocess.run(cmd, input=state_json.encode("utf-8"),
                                   stdout=devnull, stderr=devnull, timeout=900)
                except Exception as e:
                    print(f"[PiActiveSeed] background trigger failed: {e}")
                finally:
                    devnull.close()
            threading.Thread(target=_run, daemon=True).start()
            print("[PiActiveSeed] launched on the Pi in background (fire-and-forget).")
            return True
        except Exception as e:
            print(f"[PiActiveSeed] trigger setup failed: {e}")
            return False

    def _build_seed_comment(self, title: str, state: GlobalState, playlist_url: Optional[str] = None) -> str:
        """
        Builds a high-CTR, topic-grounded seed comment:
        - Incorporates the core question or tension from the video's Act 6 verdict
        - Pins clear conversion CTAs (subscribe / bell / playlist)
        - Uses aesthetic clean formatting with line breaks
        """
        # channel_phase is a read-only property that defaults to GROWTH; guard
        # against a missing channel_stats so CTAs never silently degrade.
        try:
            phase = state.channel_phase or "GROWTH"
        except Exception:
            phase = "GROWTH"
        shots = state.script_data.shots if state.script_data else []
        act6_shots = [s for s in shots if getattr(s, "act_index", 0) == 6 or getattr(s, "shot_id", 0) >= 16]
        
        grounded_question = ""
        if act6_shots:
            last_text = getattr(act6_shots[-1], "narration_text", "") or getattr(act6_shots[-1], "narration", "")
            # Extract a question sentence if present in Act 6 narration
            questions = [sent.strip() for sent in last_text.split(".") if "?" in sent]
            if questions:
                grounded_question = questions[0]
                if not grounded_question.endswith("?"):
                    grounded_question += "?"

        if not grounded_question:
            headline = state.selected_topic.headline if state.selected_topic else title
            grounded_question = f"What is your take on the shift happening with {headline}?"

        # Soft CTA guard (recommendation #3): the final Act-6 shot must explicitly
        # ask viewers to like/comment/subscribe. Warn (non-fatal) if the model
        # omitted it — the pinned comment still carries the subscribe ask, but the
        # video itself should close with one.
        _cta_words = ("subscribe", "like", "comment", "bell", "join")
        _final_act6 = act6_shots[-1] if act6_shots else None
        if _final_act6 is not None:
            _final_txt = getattr(_final_act6, "narration_text", "") or ""
            if not any(w in _final_txt.lower() for w in _cta_words):
                print("[Publisher] WARNING: final Act-6 shot lacks an explicit "
                      "like/comment/subscribe CTA — the video should close with one.")

        lines = [f"📌 {grounded_question}"]

        if phase == "GROWTH":
            lines.append(f"🔔 Subscribe to the channel & tap the bell so you never miss an in-depth breakdown: {_SUBSCRIBE_URL}")
            lines.append("💬 Tell us below: what topic or company should we investigate next?")
        else:
            lines.append(f"🔔 Subscribe for more in-depth documentaries: {_SUBSCRIBE_URL}")
            lines.append("💬 Share your thoughts and questions below — we read and reply to every comment!")

        if playlist_url:
            lines.append(f"▶️ Watch the full documentary series: {playlist_url}")

        return "\n\n".join(lines)

    async def publish_video(self, state: GlobalState, daily_uploads: int = 0) -> UploadMetadata:
        """
        Interfaces with YouTube Publishing MCP Server to check quota and upload video.
        """
        if not state.asset_paths.final_video:
            raise ValueError("Publishing failed: state.asset_paths.final_video is None")

        # 1. Quota Check
        quota_res = await check_quota_available(QuotaCheckRequest(current_daily_uploads=daily_uploads))
        if not quota_res["is_safe"]:
            raise RuntimeError(f"YouTube daily API quota limit reached ({quota_res['used_units']} / {quota_res['quota_limit']} units used)")
        run_budget.record_yt("channels")

        # 2. Upload Video — prefer the rich SEO metadata (high-CTR title, full
        #    description, tags) generated by StoryDesigner when available.
        seo = state.seo_metadata
        if seo:
            title = seo.title or (state.script_data.title if state.script_data else state.selected_topic.headline)
            desc = seo.description or (state.script_data.title if state.script_data else state.selected_topic.headline)
            tags = seo.tags or []
        else:
            title = state.script_data.title if state.script_data else state.selected_topic.headline
            default_chapters_str = (
                "0:00 - Act 1: The Inciting Incident\n"
                "2:15 - Act 2: Historical Precedents & Origins\n"
                "4:30 - Act 3: Deep Technical Mechanics\n"
                "6:45 - Act 4: Actionable Real-World Impact\n"
                "9:00 - Act 5: Critical Risks & Counter-Arguments\n"
                "11:15 - Act 6: Strategic Future Verdict"
            )
            desc = (
                f"Deep-dive financial storytelling breakdown on: {title}.\n\n"
                f"CHAPTERS:\n"
                f"{default_chapters_str}\n\n"
                f"Sources & Facts: {state.selected_topic.source_url if state.selected_topic else 'Verified Financial News Feeds'}\n\n"
                f"Disclaimer: AI-synthesized visualization for educational & infotainment storytelling."
            )
            tags = state.selected_topic.keywords if state.selected_topic else ["finance", "tech"]

        # Calculate actual start times for each act from the shared helper.
        from src.engine.chapters import compute_act_chapters, derive_contextual_act_titles
        shots = state.script_data.shots if state.script_data else None
        durs = state.asset_paths.measured_durations if state.asset_paths else None
        crossfade = getattr(state.asset_paths, "crossfade_used", 0.0) if state.asset_paths else 0.0
        act_names = (seo.act_titles if seo and getattr(seo, "act_titles", None)
                     else derive_contextual_act_titles(shots))
        _ch_lines, chapter_timestamps = compute_act_chapters(
            shots=shots, measured_durations=durs, crossfade=crossfade,
            act_names=act_names,
        )
        chapters_str = "\n".join(_ch_lines)

        old_chapters_pattern = (
            "0:00 - Act 1: The Inciting Incident\n"
            "2:15 - Act 2: Historical Precedents & Origins\n"
            "4:30 - Act 3: Deep Technical Mechanics\n"
            "6:45 - Act 4: Actionable Real-World Impact\n"
            "9:00 - Act 5: Critical Risks & Counter-Arguments\n"
            "11:15 - Act 6: Strategic Future Verdict"
        )
        if "[CHAPTERS_PLACEHOLDER]" in desc:
            desc = desc.replace("[CHAPTERS_PLACEHOLDER]", chapters_str)
        elif old_chapters_pattern in desc:
            desc = desc.replace(old_chapters_pattern, chapters_str)
        else:
            import re as _re
            _desc, _n = _re.subn(
                r"CHAPTERS:\n.*?(?=\n\n|$)",
                lambda m: ("CHAPTERS:\n" + chapters_str),
                desc, count=1, flags=_re.DOTALL,
            )
            if _n:
                desc = _desc
            elif "CHAPTERS:" in desc:
                parts = desc.split("CHAPTERS:")
                desc = parts[0] + "CHAPTERS:\n" + chapters_str + "\n\n" + parts[1].split("\n\n", 1)[-1]
            else:
                desc += f"\n\nCHAPTERS:\n{chapters_str}"

        if seo:
            seo.description = desc
            seo.chapter_timestamps = chapter_timestamps

        # channel_phase is a read-only property that defaults to GROWTH; guard
        # against a missing channel_stats so CTAs never silently degrade.
        try:
            phase = state.channel_phase or "GROWTH"
        except Exception:
            phase = "GROWTH"
        _GROWTH_DESC = (
            "\n\n🔔 Subscribe to the channel & turn on all notifications so you never "
            "miss an in-depth breakdown. New documentary every day at 17:00 UTC."
        )
        if phase == "GROWTH" and _GROWTH_DESC.strip() not in desc:
            desc = desc.rstrip() + _GROWTH_DESC
            if seo:
                seo.description = desc

        thumb_path = state.asset_paths.thumbnail if state.asset_paths else None
        upload_res = await upload_youtube_resumable(UploadRequest(
            video_path=state.asset_paths.final_video,
            title=title,
            description=desc,
            tags=tags or ["finance", "tech"],
            category_id="27",
            thumbnail_path=thumb_path,
        ))
        run_budget.record_yt("upload")

        video_id = upload_res.get("video_id", "")
        if not _is_real_video_id(video_id):
            raise RuntimeError(
                f"YouTube upload did not produce a real video id (got: '{video_id}'). "
                f"Not publishing, not posting comments/seeds, not recording dedup. "
                f"Check the YouTube OAuth token/scopes (get_youtube_token.py)."
            )

        # 3a. Auto-Playlist Chaining (watch-time amplifier). The video is chained
        # into a SINGLE themed playlist matched to the topic's audience (the
        # bingeable subscriber path). The find-or-create REUSES existing channel
        # playlists by normalized title, so no duplicate/orphan master playlists
        # are created. The seed comment below links this playlist.
        playlist_id, playlist_url = None, None
        outcome_playlists = []
        try:
            if os.getenv("YOUTUBE_AUTO_PLAYLIST", "1").strip().lower() not in ("0", "false", "no"):
                aud = getattr(state.selected_topic, "audience_type", "") if state.selected_topic else ""
                outcome = _outcome_playlist_for(aud)
                if outcome:
                    out_title, out_desc = outcome
                    pl_res = await upsert_playlist_add_video(UpsertPlaylistRequest(
                        video_id=video_id,
                        playlist_title=out_title,
                        description=out_desc,
                    ))
                    if pl_res.get("status") in ("success", "mock") and pl_res.get("playlist_id"):
                        playlist_id = pl_res.get("playlist_id")
                        playlist_url = pl_res.get("playlist_url")
                        outcome_playlists.append({
                            "title": out_title,
                            "playlist_id": pl_res.get("playlist_id"),
                            "playlist_url": pl_res.get("playlist_url"),
                        })
                        print(f"[Publisher] Video chained into themed playlist '{out_title}' (id={playlist_id})")
                else:
                    print(f"[Publisher] No themed playlist for audience '{aud}'; skipping chaining.")
        except Exception as e:
            print(f"[Publisher] Auto-playlist chaining skipped (non-fatal): {e}")

        # 3b. Post Instant Grounded Seed Comment
        pinned_comment_id, pinned_comment_text = None, None
        try:
            seed_comment = self._build_seed_comment(title, state, playlist_url=playlist_url)
            pinned_comment_text = seed_comment
            res = await insert_pinned_comment(InsertCommentRequest(
                video_id=video_id,
                comment_text=seed_comment
            ))
            if res.get("status") in ("success", "mock") and res.get("comment_id"):
                pinned_comment_id = res.get("comment_id")
                print(f"[Publisher] Seed comment posted (id={pinned_comment_id}): '{seed_comment.splitlines()[0]}'")
            else:
                print(f"[Publisher] Seed comment skipped (status={res.get('status')}): {res.get('comment_id')}")
        except Exception as e:
            print(f"[Publisher] Seed comment skipped (non-fatal): {e}")

        # 3c. YouTube Comment-Reply Bot (comment velocity -> dwell time)
        try:
            if _is_real_video_id(video_id):
                await youtube_engagement.reply_to_viewers(state, video_id=video_id)
        except Exception as e:
            print(f"[Publisher] Viewer comment reply bot skipped (non-fatal): {e}")

        meta = UploadMetadata(
            video_id=video_id,
            playlist_id=playlist_id,
            playlist_url=playlist_url,
            pinned_comment_id=pinned_comment_id,
            pinned_comment_text=pinned_comment_text,
            status="PUBLISHED",
            retry_count=0,
            synthetic_content_flag=True,
            extra_metadata={"outcome_playlists": outcome_playlists} if outcome_playlists else None,
        )
        state.upload_metadata = meta
        state.execution_stage = "PUBLISHED_SUCCESS"

        # 4. Generate Short-Form 9:16 Micro-Content Clips (Shorts / Reels / TikTok)
        try:
            from src.engine.micro_content_producer import micro_content_producer
            short_paths = micro_content_producer.generate_shorts(state, max_shorts=2)
            if hasattr(state.asset_paths, "shorts"):
                state.asset_paths.shorts = short_paths
        except Exception as e:
            print(f"[MicroContentProducer] Notice: {e}")

# 4b. GROWTH phase: PUBLISH the Shorts as YouTube Shorts (#1 discovery lever
        # for a pre-YPP channel — the master is monetized long-form, the clips feed
        # subscriptions + watch hours). Non-fatal; quota-shared with long-form.
        # Each Short points back at the full documentary (description link +
        # pinned comment) so Shorts discovery funnels into the monetizable
        # long-form; YouTube's related-video picker itself is app-only, not API.
        try:
            if (state.channel_phase == "GROWTH" and state.asset_paths.shorts):
                _short_title = (seo.title if seo and seo.title else title)[:90]
                _main_url = f"https://youtu.be/{video_id}"
                _short_ids = []
                _covers = state.asset_paths.shorts_covers if state.asset_paths else []
                for _ci, _sp in enumerate(state.asset_paths.shorts):
                    # 9:16 nano-banana cover (parallel to the clip) → custom
                    # Shorts thumbnail via youtube.thumbnails.set. YouTube
                    # resizes to the required dims keeping aspect ratio.
                    _short_thumb = _covers[_ci] if _ci < len(_covers) else None
                    _short_desc = f"{desc}\n\n▶ Full video: {_main_url}"[:4900]
                    _res = await upload_short(UploadRequest(
                        video_path=_sp,
                        title=f"{_short_title} | Short",
                        description=_short_desc,
                        tags=(tags or [])[:30] + ["#Shorts"],
                        category_id="22",
                        thumbnail_path=_short_thumb,
                    ))
                    if _res and _is_real_video_id(_res.get("video_id", "")):
                        _short_ids.append(_res["video_id"])
                        run_budget.record_yt("upload")
                        # Pinned comment on the Short linking the long-form master.
                        try:
                            await insert_pinned_comment(InsertCommentRequest(
                                video_id=_res["video_id"],
                                comment_text=f"Watch the full video ▶ {_main_url}",
                            ))
                        except Exception as _short_pin_err:
                            print(f"[Publisher] Short pin comment skipped (non-fatal): {_short_pin_err}")
                if _short_ids:
                    meta.shorts_video_id = ",".join(_short_ids)
                    print(f"[Publisher] Published {len(_short_ids)} YouTube Short(s) for GROWTH discovery.")
        except Exception as e:
            print(f"[Publisher] Shorts upload skipped (non-fatal): {e}")


        # 5. Create & Dispatch Seed Traffic Package (Reddit / HN / Webhook)
        try:
            from src.engine.seed_distributor import seed_distributor
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            seed_pkg = seed_distributor.create_seed_package(state, youtube_url)
            await seed_distributor.dispatch_webhook_notification(seed_pkg)
        except Exception as e:
            print(f"[SeedDistributor] Notice: {e}")

        # 5b. Active Thread Comment/Reply Seeding (Discussion Injection)
        # MUST run on the Pi's residential IP — datacenter IPs get spam-filtered /
        # AutoMod-deleted by Reddit. The publisher ships GlobalState to the Pi and
        # fires the seeder there; it only runs locally if Pi seeding is explicitly
        # disabled via REDDIT_PI_ACTIVE_SEED=0.
        try:
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            if os.getenv("REDDIT_PI_ACTIVE_SEED", "1").strip().lower() not in ("0", "false", "no"):
                self._trigger_pi_active_seed(state, youtube_url)
            else:
                from src.engine.active_thread_seeder import active_thread_seeder
                await active_thread_seeder.seed_active_discussions(state, youtube_url)
        except Exception as e:
            print(f"[ActiveThreadSeeder] Notice: {e}")

        # 5c. No-link warmup on the Pi (residential IP) to build Reddit account
        # trust. The main pipeline runs on OCI where Reddit is IP-blocked, so we
        # instruct the Pi edge node to run reddit_warmup.py for the published
        # video. Non-fatal: failures never affect the publish result.
        try:
            if _is_real_video_id(video_id):
                self._trigger_pi_warmup(title, state)
        except Exception as e:
            print(f"[PiWarmup] Notice: {e}")

        # 5d. Analytics feedback refresh (background, non-fatal, rate-limited):
        # pull per-video growth metrics into logs/analytics_feedback.json so the
        # next run's topic selection can bias toward the top growth drivers
        # ("double down on what works"). Never blocks or fails the publish.
        try:
            if _is_real_video_id(video_id) and os.getenv("CSVG_ANALYTICS_FEEDBACK", "1").strip().lower() not in ("0", "false", "no"):
                import threading as _th
                def _refresh():
                    try:
                        from src.engine.analytics_feedback import analytics_feedback
                        analytics_feedback.refresh(force=False)
                    except Exception as _e:
                        print(f"[AnalyticsFeedback] background refresh failed: {_e}")
                _th.Thread(target=_refresh, daemon=True).start()
                print("[AnalyticsFeedback] background refresh launched (rate-limited).")
        except Exception as e:
            print(f"[AnalyticsFeedback] Notice: {e}")

        # Record to persistent deduplication history so future runs skip this topic
        from src.engine.topic_deduplicator import topic_deduplicator
        topic_headline = state.selected_topic.headline if state.selected_topic else title
        topic_summary = state.selected_topic.summary if state.selected_topic else ""
        topic_kws = state.selected_topic.keywords if state.selected_topic else []
        topic_deduplicator.record_published_topic(topic_headline, topic_summary, topic_kws)

        return meta


    async def process(self, state: GlobalState, daily_uploads: int = 0) -> A2AMessage:
        """
        Executes Publisher Agent workflow:
        1. Reads state.asset_paths.final_video
        2. Calls YouTube MCP server tools
        3. Updates state.upload_metadata
        4. Emits PUBLISHED_SUCCESS A2AMessage
        """
        meta = await self.publish_video(state, daily_uploads=daily_uploads)

        msg = A2AMessage(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            sender=AgentRole.PUBLISHER,
            target=AgentRole.ORCHESTRATOR,
            intent=AgentIntent.PUBLISHED_SUCCESS,
            payload={
                "status": "SUCCESS",
                "video_id": meta.video_id,
                "youtube_url": f"https://www.youtube.com/watch?v={meta.video_id}",
                "synthetic_content": True
            },
            state_hash=compute_state_hash(state),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        return msg
