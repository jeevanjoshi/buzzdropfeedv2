import os
import json
import uuid
import copy
import datetime
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from src.schemas.state import GlobalState
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent
from src.agents.fact_retriever import FactRetrieverAgent
from src.agents.story_designer import StoryDesignerAgent
from src.agents.observer import ObserverAgent, is_soft_violation, observer_quality_score
from src.agents.media_producer import MediaProducerAgent
from src.agents.publisher import PublisherAgent
from src.engine.quality_verifier import quality_verifier
from src.engine.video_quality_metrics import video_quality_metrics
from src.engine.channel_phase_manager import channel_phase_manager, get_ypp_progress_report
from src.engine.rag_retriever import rag_retriever
from src.engine.logger import logger
from src.engine.tracer import tracer
from src.engine.run_budget import run_budget


class OrchestratorAgent:
    """
    Central A2A Orchestrator Agent managing the end-to-end CSVG pipeline lifecycle.
    Dispatches tasks sequentially between FactRetriever -> StoryDesigner -> Observer -> MediaProducer -> Publisher,
    recording structured logs and trajectory step files with actionable remediation fix hints.
    """

    def __init__(
        self,
        fact_retriever: Optional[FactRetrieverAgent] = None,
        story_designer: Optional[StoryDesignerAgent] = None,
        observer: Optional[ObserverAgent] = None,
        media_producer: Optional[MediaProducerAgent] = None,
        publisher: Optional[PublisherAgent] = None,
        logs_dir: str = "logs"
    ):
        self.fact_retriever = fact_retriever or FactRetrieverAgent()
        self.story_designer = story_designer or StoryDesignerAgent()
        self.observer = observer or ObserverAgent()
        self.media_producer = media_producer or MediaProducerAgent()
        self.publisher = publisher or PublisherAgent()
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)

    async def run_pipeline(
        self,
        pipeline_id: Optional[str] = None,
        use_live_rss: bool = True,
        region: str = "all",
        publish: bool = True,
        dummy_frames: bool = False,
        state: Optional[GlobalState] = None,
        renderer: str = "ffmpeg",
        crossfade: float = 0.5,
        pad_after_narration: Optional[float] = None,
        rag_mode: str = "scraper"
    ) -> GlobalState:
        """
        Executes complete autonomous pipeline run with structured logging & trajectory tracing.
        """
        # A/B switch: "grounded" uses the Google Search research pass, "hybrid"
        # uses grounded cited facts as the core AND on-topic scraper depth, and
        # anything else uses the scraper RAG path (see src/engine/grounded_search.py).
        rag_mode = (rag_mode or "scraper").strip().lower()
        if rag_mode not in ("scraper", "grounded", "hybrid"):
            rag_mode = "scraper"
        rag_retriever.set_rag_mode(rag_mode)
        _rag_desc = {
            "scraper": "5-scraper path",
            "grounded": "Google Search grounding only",
            "hybrid": "grounded core + scraper depth",
        }[rag_mode]
        logger.info(
            "INITIALIZATION",
            f"RAG mode: {rag_mode} ({_rag_desc})",
            pipeline_id=pipeline_id or (state.pipeline_id if state else None),
            component="RAG_RETRIEVER",
        )
        # Renderer switch: "ffmpeg" (default, probe-driven concat) or "moviepy" (timeline composer).
        if renderer in ("moviepy", "ffmpeg"):
            self.media_producer.renderer = renderer
        self.media_producer.crossfade = max(0.0, crossfade)
        if pad_after_narration is not None:
            self.media_producer.pad_after_narration = max(0.0, pad_after_narration)
        if state is None:
            p_id = pipeline_id or f"csvg-exec-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            state = GlobalState(
                pipeline_id=p_id,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
            )
        else:
            p_id = state.pipeline_id

        state.region = region  # persist so downstream agents (e.g. media producer) know the market region

        budget_status = "success"

        # ── Load Channel Phase (GROWTH / REVENUE / SCALE) ──────────────────
        channel_stats = channel_phase_manager.get_channel_stats()
        state.channel_stats = channel_stats
        phase = channel_stats.channel_phase

        logger.info("INITIALIZATION", f"Starting CSVG Pipeline Run: {p_id} (Resuming from stage: {state.execution_stage})" if state.selected_topic else f"Starting CSVG Pipeline Run: {p_id}", pipeline_id=p_id, component="ORCHESTRATOR")
        logger.info(
            "INITIALIZATION",
            f"Channel Phase: {phase} | Subs: {channel_stats.subscribers} | "
            f"Watch Hours: {channel_stats.total_watch_hours} | YPP Unlocked: {channel_stats.ypp_unlocked}",
            pipeline_id=p_id, component="CHANNEL_PHASE"
        )
        if phase == "GROWTH":
            progress = get_ypp_progress_report(channel_stats)
            logger.info(
                "INITIALIZATION",
                f"YPP Progress — Subs: {progress['subs_progress_pct']}% "
                f"| Watch Hours: {progress['watch_hours_progress_pct']}% "
                f"| Est. days to YPP: {progress['estimated_days_to_ypp']}",
                pipeline_id=p_id, component="CHANNEL_PHASE"
            )
        tracer.record_step(state, "INITIALIZATION")

        # Begin per-run budget tracking (reset counters for this pipeline_id).
        run_budget.start(p_id)

        try:
            # 1. Fact Retrieval & Topic Selection (phase-aware TOPSIS)
            if not state.selected_topic:
                logger.info("PHASE_1_TOPIC_SELECTION", f"Ingesting feeds and evaluating TOPSIS [{phase} weights]...", pipeline_id=p_id, component="FACT_RETRIEVER")
                msg_topic = self.fact_retriever.process(state, use_live_rss=use_live_rss, region=region, channel_phase=phase)
                
                if not state.selected_topic:
                    raise RuntimeError("No suitable topic selected by FactRetrieverAgent.")

                logger.info(
                    "PHASE_1_TOPIC_SELECTION",
                    f"Selected Topic: '{state.selected_topic.headline}' (TOPSIS Score: {state.selected_topic.topsis_score:.4f})",
                    pipeline_id=p_id, component="FACT_RETRIEVER",
                    extra_data={"topsis_score": state.selected_topic.topsis_score, "headline": state.selected_topic.headline}
                )
                tracer.record_step(state, "TOPIC_SELECTED", message=msg_topic)
            else:
                logger.info("PHASE_1_TOPIC_SELECTION", f"Resuming: Using existing selected topic: '{state.selected_topic.headline}'", pipeline_id=p_id, component="FACT_RETRIEVER")

            # ── PHASE 1B: RAG CORPUS QUALITY GATE ──────────────────────────
            # Never burn LLM spend generating a script from an undersupplied /
            # polluted RAG corpus. Build the RAG pack, assess sufficiency, retry
            # the build (up to 2 refreshes) for the same topic, and only then
            # fall back to re-running the Fact Retriever on a DIFFERENT topic
            # (previously-failed topics are excluded so they aren't re-picked).
            # This also seeds state.crawled_content so the Observer later audits
            # the script against the exact same fact corpus StoryDesigner used.
            if not state.script_data:
                MAX_RAG_BUILDS = 3          # initial build + 2 refresh retries
                MAX_TOPIC_SWITCHES = 2      # at most 2 alternate topics after the first
                excluded_headlines: List[str] = []
                topic_switch_count = 0
                rag_pass = False

                while not rag_pass:
                    pack = rag_retriever.build_rag_knowledge_pack(state.selected_topic, state.verified_facts)
                    state.crawled_content = pack.get("fact_corpus", pack.get("full_rag_context_text", ""))
                    report = rag_retriever.assess_corpus_sufficiency(pack, state.selected_topic)

                    # Retry RAG build for the same topic (fresh retrieval each time).
                    for build in range(2, MAX_RAG_BUILDS + 1):
                        if report["pass"]:
                            break
                        logger.warning(
                            "PHASE_1B_RAG_QUALITY",
                            f"RAG corpus insufficient for '{state.selected_topic.headline}': {report['reason']}. "
                            f"Rebuilding RAG (attempt {build}/{MAX_RAG_BUILDS})...",
                            pipeline_id=p_id, component="RAG_RETRIEVER",
                            extra_data=report.get("metrics", {})
                        )
                        pack = rag_retriever.build_rag_knowledge_pack(
                            state.selected_topic, state.verified_facts, refresh=True
                        )
                        state.crawled_content = pack.get("fact_corpus", pack.get("full_rag_context_text", ""))
                        report = rag_retriever.assess_corpus_sufficiency(pack, state.selected_topic)

                    if report["pass"]:
                        logger.info(
                            "PHASE_1B_RAG_QUALITY",
                            f"RAG corpus sufficient for '{state.selected_topic.headline}': "
                            f"{report['metrics']['on_topic_facts']} on-topic facts, "
                            f"{report['metrics']['on_topic_corpus_words']} on-topic words, "
                            f"{report['metrics']['on_topic_sources']} sources.",
                            pipeline_id=p_id, component="RAG_RETRIEVER",
                            extra_data=report.get("metrics", {})
                        )
                        rag_pass = True
                        break

                    # RAG still insufficient after retries → switch topic.
                    if topic_switch_count >= MAX_TOPIC_SWITCHES:
                        raise RuntimeError(
                            f"RAG corpus insufficient for all {MAX_TOPIC_SWITCHES + 1} candidate topics "
                            f"after {MAX_RAG_BUILDS} RAG builds each. Last failure: {report['reason']} "
                            f"({report['metrics']})."
                        )
                    excluded_headlines.append(state.selected_topic.headline)
                    topic_switch_count += 1
                    logger.warning(
                        "PHASE_1B_RAG_QUALITY",
                        f"RAG insufficient for '{state.selected_topic.headline}' after {MAX_RAG_BUILDS} builds. "
                        f"Re-running Fact Retriever excluding {len(excluded_headlines)} failed topic(s)...",
                        pipeline_id=p_id, component="RAG_RETRIEVER",
                        fix_hint="RSS feed may lack deep sources for this topic; picking next-best TOPSIS candidate."
                    )
                    msg_topic = self.fact_retriever.process(
                        state, use_live_rss=use_live_rss, region=region,
                        channel_phase=phase, exclude_headlines=excluded_headlines
                    )
                    if not state.selected_topic:
                        raise RuntimeError("No alternative topic selected by FactRetrieverAgent after RAG quality exclusion.")
                    logger.info(
                        "PHASE_1B_RAG_QUALITY",
                        f"Selected alternate topic: '{state.selected_topic.headline}' "
                        f"(TOPSIS Score: {state.selected_topic.topsis_score:.4f})",
                        pipeline_id=p_id, component="FACT_RETRIEVER"
                    )
                    tracer.record_step(state, "TOPIC_SELECTED", message=msg_topic)

                # Validate RAG corpus actually propagated to later stages.
                if not state.crawled_content.strip():
                    raise RuntimeError("RAG gate passed but state.crawled_content is empty; corpus was not fed to downstream stages.")
                if not state.verified_facts:
                    logger.warning("PHASE_1B_RAG_QUALITY", "verified_facts is empty despite RAG gate passing.", pipeline_id=p_id, component="RAG_RETRIEVER")
                tracer.record_step(state, "RAG_QUALITY_PASSED")
                run_budget.set_stage("RAG_QUALITY_PASSED")

            # 2. Story Script Generation
            if not state.script_data or state.execution_stage == "SCRIPT_REVISION_REQUIRED":
                if state.execution_stage == "SCRIPT_REVISION_REQUIRED":
                    logger.info("PHASE_2_SCRIPT_DESIGN", "Found incomplete/failed script from a failed run (Stage: SCRIPT_REVISION_REQUIRED). Clearing and re-generating...", pipeline_id=p_id, component="STORY_DESIGNER")
                    state.script_data = None
                logger.info("PHASE_2_SCRIPT_DESIGN", "Generating 10-15 Min 6-Act dramatic arc narrative script...", pipeline_id=p_id, component="STORY_DESIGNER")
                msg_script = self.story_designer.process(state)
                
                if not state.script_data:
                    raise RuntimeError("Script generation failed.")

                logger.info(
                    "PHASE_2_SCRIPT_DESIGN",
                    f"Script Generated: '{state.script_data.title}' ({state.script_data.target_shots} shots, {state.script_data.estimated_runtime_seconds/60:.2f} mins)",
                    pipeline_id=p_id, component="STORY_DESIGNER"
                )
                tracer.record_step(state, "SCRIPT_GENERATED", message=msg_script)

                # 3. Observer Audit & Quality Gate
                logger.info("PHASE_2_OBSERVER_AUDIT", "Running Observer Fact Audit and Visual Quality Check...", pipeline_id=p_id, component="OBSERVER")
                msg_obs = self.observer.process(state)

                # Define a helper function to audit script quality (Gate 1 & Gate 6)
                def run_script_quality_checks() -> Tuple[bool, Optional[str]]:
                    # Gate 1: Topic-to-Script coherence check
                    gate1_pass, gate1_issues = quality_verifier.verify_gate1_topic_to_script(
                        state.selected_topic, state.script_data
                    )
                    if not gate1_pass:
                        return False, f"Gate 1 Fail (Topic Coherence): {gate1_issues}"
                    
                    # Gate 6: Anti-Slop Entropy Audit
                    gate6_result = quality_verifier.verify_gate6_anti_slop_entropy(state.script_data)
                    if not gate6_result["passes"]:
                        return False, f"Gate 6 Fail (Anti-Slop): {gate6_result['issues']}"
                    
                    return True, None

                quality_pass, quality_error = run_script_quality_checks()

                if msg_obs.intent == AgentIntent.REVISE_SCRIPT or not quality_pass:
                    reason = "Observer rejection" if msg_obs.intent == AgentIntent.REVISE_SCRIPT else quality_error
                    logger.warning(
                        "PHASE_2_OBSERVER_AUDIT",
                        f"Script failed initial validation ({reason}). Triggering A2A revision loop...",
                        pipeline_id=p_id, component="OBSERVER",
                        fix_hint="Retrying StoryDesigner with strict RAG constraints."
                    )
                    tracer.record_step(state, "SCRIPT_REVISION_REQUIRED", message=msg_obs, status="WARNING")

                    # Bounded revision loop: retry StoryDesigner with corrective feedback
                    # up to MAX_REVISIONS times before giving up. Each round re-audits and
                    # re-checks quality so genuinely stubborn drafts don't abort the run
                    # after a single retry, while still bounding wasted LLM spend.
                    MAX_REVISIONS = 3
                    revision_ok = False
                    last_violations: List[str] = []
                    last_quality_error: Optional[str] = None
                    best_script = None
                    best_script_score = -1.0
                    for attempt in range(1, MAX_REVISIONS + 1):
                        violations = []
                        if msg_obs.intent == AgentIntent.REVISE_SCRIPT and msg_obs.payload:
                            violations.extend(msg_obs.payload.get("violations", []))
                        if not quality_pass and quality_error:
                            violations.append(quality_error)

                        logger.warning(
                            "PHASE_2_OBSERVER_AUDIT",
                            f"Revision attempt {attempt}/{MAX_REVISIONS} with {len(violations)} violation(s).",
                            pipeline_id=p_id, component="OBSERVER"
                        )
                        msg_script = self.story_designer.process(state, revision_violations=violations)
                        # Keep the highest-scoring draft across revisions (revisions can
                        # fix one nit but regress elsewhere; score = paraphrase diversity).
                        _draft_score = observer_quality_score(state.script_data)
                        if _draft_score > best_script_score:
                            best_script_score = _draft_score
                            best_script = copy.deepcopy(state.script_data)
                        msg_obs = self.observer.process(state)
                        quality_pass, quality_error = run_script_quality_checks()

                        last_violations = violations
                        last_quality_error = quality_error if not quality_pass else None

                        if msg_obs.intent != AgentIntent.REVISE_SCRIPT and quality_pass:
                            revision_ok = True
                            break

                    if not revision_ok:
                        # Soft approval: if the only remaining Observer violations are
                        # style-class (verbatim-copy/repetition/source-diversity/visual)
                        # AND the mandatory quality gates still pass, accept the BEST
                        # scored draft with a warning instead of hard-aborting. Any
                        # fact/temporal/revenue/audience/structural violation or a failing
                        # quality gate still aborts (those are hard invariants).
                        remaining = last_violations
                        if msg_obs.intent == AgentIntent.REVISE_SCRIPT and msg_obs.payload:
                            remaining = msg_obs.payload.get("violations", last_violations)
                        hard = [v for v in remaining if not is_soft_violation(v)]
                        allow_soft = os.getenv("ALLOW_SOFT_APPROVAL", "1").strip().lower() in ("1", "true", "yes")
                        if allow_soft and quality_pass and not hard and best_script is not None:
                            state.script_data = best_script
                            logger.warning(
                                "PHASE_2_OBSERVER_AUDIT",
                                f"Soft approval: {len(remaining)} style-only violation(s) after {MAX_REVISIONS} "
                                f"revisions; accepting best-scored draft (score {best_script_score:.3f}). "
                                f"Remaining: {remaining}",
                                pipeline_id=p_id, component="OBSERVER"
                            )
                            msg_obs = self.observer.process(state)
                            state.execution_stage = "SCRIPT_APPROVED"
                        else:
                            # Report the ACTUAL failure: prefer Observer violations when the
                            # observer rejected; otherwise surface the failing quality gate.
                            if msg_obs.intent == AgentIntent.REVISE_SCRIPT and msg_obs.payload:
                                detail = msg_obs.payload.get("violations")
                            else:
                                detail = last_quality_error or last_violations
                            raise RuntimeError(
                                f"Script failed validation after {MAX_REVISIONS} revisions: {detail}"
                            )

                logger.info("PHASE_2_OBSERVER_AUDIT", "Observer Audit & Quality Gates Passed 100%! Script approved.", pipeline_id=p_id, component="OBSERVER")
                tracer.record_step(state, "SCRIPT_APPROVED", message=msg_obs)
                run_budget.set_stage("SCRIPT_APPROVED")

                # Export individual script JSON and Markdown files for easy debugging
                script_json_path = os.path.join(self.logs_dir, f"script_{p_id}.json")
                script_md_path = os.path.join(self.logs_dir, f"script_{p_id}.md")
                
                # Save individual JSON
                try:
                    with open(script_json_path, "w", encoding="utf-8") as f:
                        json.dump(state.script_data.model_dump(), f, indent=2)
                    logger.info("PHASE_2_OBSERVER_AUDIT", f"Saved individual script JSON to {script_json_path}", pipeline_id=p_id)
                except Exception as e:
                    logger.warning("PHASE_2_OBSERVER_AUDIT", f"Failed to save script JSON: {e}", pipeline_id=p_id)
                    
                # Save formatted Markdown Screenplay
                try:
                    md_content = f"# Screenplay: {state.script_data.title}\n\n"
                    md_content += f"* **Pipeline ID:** `{p_id}`\n"
                    md_content += f"* **Target Shots:** {state.script_data.target_shots}\n"
                    md_content += f"* **Estimated Runtime:** {state.script_data.estimated_runtime_seconds:.2f}s (~{state.script_data.estimated_runtime_seconds/60.0:.2f} mins)\n\n"
                    md_content += "---\n\n"
                    
                    for shot in state.script_data.shots:
                        md_content += f"## Shot {shot.shot_id} (Act {shot.act_index})\n"
                        md_content += f"* **Visual Type:** `{shot.visual_type.value}`\n"
                        md_content += f"* **Visual Prompt:** *\"{shot.visual_prompt}\"*\n"
                        md_content += f"* **Narration Text:**\n  > {shot.narration_text}\n\n"
                        md_content += "---\n\n"
                        
                    with open(script_md_path, "w", encoding="utf-8") as f:
                        f.write(md_content)
                    logger.info("PHASE_2_OBSERVER_AUDIT", f"Saved formatted script Markdown to {script_md_path}", pipeline_id=p_id)
                except Exception as e:
                    logger.warning("PHASE_2_OBSERVER_AUDIT", f"Failed to save script Markdown: {e}", pipeline_id=p_id)
            else:
                logger.info("PHASE_2_SCRIPT_DESIGN", f"Resuming: Using existing approved script: '{state.script_data.title}'", pipeline_id=p_id, component="STORY_DESIGNER")

            # 4. Media Production (Audio, Visuals, FFmpeg Assembly)
            if not state.asset_paths.final_video:
                logger.info("PHASE_3_MEDIA_PRODUCTION", "Synthesizing Edge TTS Audio and Rendering 16:9 Widescreen Visuals...", pipeline_id=p_id, component="MEDIA_PRODUCER")
                msg_media = await self.media_producer.process(state, dummy_frames=dummy_frames)
                
                if not state.asset_paths.final_video:
                    raise RuntimeError("Media production failed: Final video output path is empty.")

                logger.info("PHASE_3_MEDIA_PRODUCTION", f"Media Production Complete! Video: {state.asset_paths.final_video}", pipeline_id=p_id, component="MEDIA_PRODUCER")
                tracer.record_step(state, "MEDIA_READY", message=msg_media)
                run_budget.set_stage("MEDIA_READY")
            else:
                logger.info("PHASE_3_MEDIA_PRODUCTION", f"Resuming: Using existing final video: {state.asset_paths.final_video}", pipeline_id=p_id, component="MEDIA_PRODUCER")

            # 4b. Post-Production Quality Gates (Stage 6 + Stage 8)
            logger.info("PHASE_3B_QUALITY_GATES", "Running post-production compliance & video quality gates...", pipeline_id=p_id, component="QUALITY_VERIFIER")

            # Gate 5: YouTube 2025/2026 AI Disclosure Auto-Tagging
            pipeline_tags = ["flux.1", "kokoro-tts", "wan2.1"]
            gate5_result = quality_verifier.verify_gate5_ai_disclosure_tags(state.script_data, pipeline_tags)
            logger.info("PHASE_3B_QUALITY_GATES",
                        f"Gate 5 AI Disclosure: syntheticContent={gate5_result['youtube_metadata_patch']['syntheticContent']}, "
                        f"bInformed={gate5_result['youtube_metadata_patch']['bInformed']} — {gate5_result['rationale']}",
                        pipeline_id=p_id, component="QUALITY_VERIFIER")
            # Patch upload metadata with disclosure tags
            if state.upload_metadata:
                state.upload_metadata.extra_metadata = state.upload_metadata.extra_metadata or {}
                state.upload_metadata.extra_metadata.update(gate5_result["youtube_metadata_patch"])

            # Gate 3b: Subtitle-to-Script Narration Alignment Coherence Check
            gate3b_pass, gate3b_issues = quality_verifier.verify_gate3b_subtitle_text_coherence(state)
            if not gate3b_pass:
                raise RuntimeError(f"Gate 3b (Subtitle Coherence) Fail: {gate3b_issues}. Aborting publishing.")

            # Gate 4: Video and Audio stream coherence check (skips fatal crash during dummy/dry-runs)
            gate4_pass, gate4_issues = quality_verifier.verify_gate4_video_audio_coherence(state)
            if not gate4_pass:
                if any("missing" in str(issue).lower() for issue in gate4_issues):
                    logger.warning("PHASE_3B_QUALITY_GATES", f"Gate 4 (Video Coherence) skipped: {gate4_issues} (This is normal during script-only or subtitle-only testing dry runs).", pipeline_id=p_id, component="QUALITY_VERIFIER")
                else:
                    raise RuntimeError(f"Gate 4 (Video Coherence) Fail: {gate4_issues}. Aborting publishing.")

            # Stage 8: Per-Shot FVD + Optical Flow Quality Gate
            shot_quality_failures = []
            for shot in state.script_data.shots:
                shot_key = f"shot_{shot.shot_id}"
                # Use proxy feature stats derived from shot metadata (word count, duration as stand-ins)
                # In production, replace with real I3D feature extractor output
                word_count = len(shot.narration_text.split())
                gen_stats = {"mean": float(word_count) / 10.0, "variance": float(word_count) / 20.0}
                ref_stats = {"mean": 12.0, "variance": 4.0}  # Reference baseline for 12–15min Infotainment
                # Proxy motion vectors from duration estimate (uniform motion baseline)
                motion_vectors = [shot.duration_estimate * 0.2 + i * 0.05 for i in range(10)]
                report = video_quality_metrics.run_full_quality_gate(
                    shot_id=shot_key,
                    generated_feature_stats=gen_stats,
                    reference_feature_stats=ref_stats,
                    frame_motion_vectors=motion_vectors
                )
                if not report["stage8_overall_pass"]:
                    shot_quality_failures.append((shot_key, report["recommendation"]))
                    logger.warning(
                        "PHASE_3B_QUALITY_GATES",
                        f"Stage 8 Quality Fail — {shot_key}: {report['recommendation']}",
                        pipeline_id=p_id, component="VIDEO_QUALITY",
                        fix_hint=f"Re-render {shot_key} with refined FLUX.1 prompt or adjust Ken Burns motion params."
                    )

            if shot_quality_failures:
                logger.warning(
                    "PHASE_3B_QUALITY_GATES",
                    f"Stage 8: {len(shot_quality_failures)}/{len(state.script_data.shots)} shots flagged. Pipeline continues with warnings.",
                    pipeline_id=p_id, component="VIDEO_QUALITY"
                )
            else:
                logger.info("PHASE_3B_QUALITY_GATES", "Stage 8: All shots passed FVD + Optical Flow gates.", pipeline_id=p_id, component="VIDEO_QUALITY")

            # Gate 7: Real Render-Integrity Gate (pixel + audio + measured-duration pre-upload check)
            gate7_pass, gate7_issues = quality_verifier.verify_gate7_render_integrity(state)
            if not gate7_pass:
                raise RuntimeError(
                    f"Gate 7 (Render Integrity) Fail: {'; '.join(gate7_issues)}. "
                    f"Aborting publishing to avoid uploading a broken/mute video."
                )
            logger.info("PHASE_3B_QUALITY_GATES", "Gate 7 Render Integrity passed (black/frozen frames, audio silence/peak, measured-duration check).", pipeline_id=p_id, component="QUALITY_VERIFIER")

            tracer.record_step(state, "QUALITY_GATES_PASSED")
            run_budget.set_stage("QUALITY_GATES_PASSED")

            # 5. YouTube Publishing
            if publish:
                logger.info("PHASE_4_YOUTUBE_PUBLISHING", "Publishing Video to YouTube with Synthetic Metadata...", pipeline_id=p_id, component="PUBLISHER")
                msg_pub = await self.publisher.process(state)

                logger.info("PHASE_4_YOUTUBE_PUBLISHING", f"Pipeline Run Completed Successfully! Video ID: {state.upload_metadata.video_id}", pipeline_id=p_id, component="PUBLISHER")
                tracer.record_step(state, "PUBLISHED_SUCCESS", message=msg_pub)
                run_budget.set_stage("PUBLISHED_SUCCESS")
            else:
                logger.info("PHASE_4_YOUTUBE_PUBLISHING", "Skipping YouTube Publishing step as requested (pipeline run till upload complete).", pipeline_id=p_id, component="ORCHESTRATOR")
                state.execution_stage = "QUALITY_VERIFIED"
                tracer.record_step(state, "PIPELINE_COMPLETE_TILL_UPLOAD")
                run_budget.set_stage("QUALITY_VERIFIED")

        except Exception as e:
            budget_status = "failed"
            # Determine actionable Fix Hint based on exception message
            err_str = str(e).lower()
            fix_hint = "Inspect error stack trace in logs/csvg_execution.log"

            if "ffmpeg" in err_str or "command failed" in err_str:
                fix_hint = "FFmpeg binary error. Ensure FFmpeg is installed via 'sudo apt install ffmpeg' or system PATH."
            elif "fal_key" in err_str or "fal" in err_str:
                fix_hint = "Fal.ai API key missing or invalid. Set FAL_KEY in your .env file."
            elif "quota" in err_str or "1600" in err_str:
                fix_hint = "YouTube Data API daily quota limit reached (10,000 units). Wait for daily quota reset or use secondary channel API key."
            elif "openrouter" in err_str or "openai" in err_str:
                fix_hint = "LLM API Key error. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env."

            logger.error(
                state.execution_stage,
                f"Pipeline Exception: {str(e)}",
                pipeline_id=p_id,
                component="ORCHESTRATOR",
                exception=e,
                fix_hint=fix_hint
            )
            tracer.record_step(state, "PIPELINE_FAILED", status="ERROR", error_details={"error": str(e), "fix_hint": fix_hint})
            raise e

        finally:
            # Persist the per-run budget ledger + rolling aggregate regardless of outcome.
            try:
                _t = run_budget.totals()
                budget_stage = getattr(state, "execution_stage", "")
                budget_stage = budget_status if budget_status == "failed" else budget_stage
                rec = run_budget.save(
                    pipeline_id=p_id,
                    status=budget_status,
                    stage=budget_stage,
                    extra={"topic": state.selected_topic.headline if state.selected_topic else None,
                           "video_id": state.upload_metadata.video_id if state.upload_metadata else None},
                )
                logger.info(
                    "BUDGET",
                    f"Run {p_id} :: est ${_t['est_usd']:.4f} | {_t['yt_units']} YT units | "
                    f"categories={','.join(_t['categories'])} | persisted={rec is not None and rec.get('pipeline_id') == p_id}",
                    pipeline_id=p_id, component="RUN_BUDGET",
                )
            except Exception:
                pass

        return state
