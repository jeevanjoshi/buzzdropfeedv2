import uuid
import datetime
import asyncio
from typing import Dict, Any, Optional
from src.schemas.state import GlobalState
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent
from src.agents.fact_retriever import FactRetrieverAgent
from src.agents.story_designer import StoryDesignerAgent
from src.agents.observer import ObserverAgent
from src.agents.media_producer import MediaProducerAgent
from src.agents.publisher import PublisherAgent
from src.engine.logger import logger
from src.engine.tracer import tracer


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
        publisher: Optional[PublisherAgent] = None
    ):
        self.fact_retriever = fact_retriever or FactRetrieverAgent()
        self.story_designer = story_designer or StoryDesignerAgent()
        self.observer = observer or ObserverAgent()
        self.media_producer = media_producer or MediaProducerAgent()
        self.publisher = publisher or PublisherAgent()

    async def run_pipeline(
        self, pipeline_id: Optional[str] = None, use_live_rss: bool = True, region: str = "all"
    ) -> GlobalState:
        """
        Executes complete autonomous pipeline run with structured logging & trajectory tracing.
        """
        p_id = pipeline_id or f"csvg-exec-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        state = GlobalState(
            pipeline_id=p_id,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )

        logger.info("INITIALIZATION", f"Starting CSVG Pipeline Run: {p_id}", pipeline_id=p_id, component="ORCHESTRATOR")
        tracer.record_step(state, "INITIALIZATION")

        try:
            # 1. Fact Retrieval & Topic Selection
            logger.info("PHASE_1_TOPIC_SELECTION", "Ingesting news feeds and evaluating 7-Criteria TOPSIS Decision Matrix...", pipeline_id=p_id, component="FACT_RETRIEVER")
            msg_topic = self.fact_retriever.process(state, use_live_rss=use_live_rss, region=region)
            
            if not state.selected_topic:
                raise RuntimeError("No suitable topic selected by FactRetrieverAgent.")

            logger.info(
                "PHASE_1_TOPIC_SELECTION",
                f"Selected Topic: '{state.selected_topic.headline}' (TOPSIS Score: {state.selected_topic.topsis_score:.4f})",
                pipeline_id=p_id, component="FACT_RETRIEVER",
                extra_data={"topsis_score": state.selected_topic.topsis_score, "headline": state.selected_topic.headline}
            )
            tracer.record_step(state, "TOPIC_SELECTED", message=msg_topic)

            # 2. Story Script Generation
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

            if msg_obs.intent == AgentIntent.REVISE_SCRIPT:
                logger.warning(
                    "PHASE_2_OBSERVER_AUDIT",
                    "Observer rejected initial script draft. Triggering A2A revision loop...",
                    pipeline_id=p_id, component="OBSERVER",
                    extra_data=msg_obs.payload,
                    fix_hint="Observer identified script pacing or unverified fact violations. Retrying StoryDesigner with strict RAG constraints."
                )
                tracer.record_step(state, "SCRIPT_REVISION_REQUIRED", message=msg_obs, status="WARNING")
                
                # Retry Story Designer
                msg_script = self.story_designer.process(state)
                msg_obs = self.observer.process(state)

            if msg_obs.intent == AgentIntent.REVISE_SCRIPT:
                raise RuntimeError(f"Script failed Observer audit: {msg_obs.payload.get('violations')}")

            logger.info("PHASE_2_OBSERVER_AUDIT", "Observer Audit Passed 100%! Script approved.", pipeline_id=p_id, component="OBSERVER")
            tracer.record_step(state, "SCRIPT_APPROVED", message=msg_obs)

            # 4. Media Production (Audio, Visuals, FFmpeg Assembly)
            logger.info("PHASE_3_MEDIA_PRODUCTION", "Synthesizing Edge TTS Audio and Rendering 16:9 Widescreen Visuals...", pipeline_id=p_id, component="MEDIA_PRODUCER")
            msg_media = await self.media_producer.process(state)
            
            if not state.asset_paths.final_video:
                raise RuntimeError("Media production failed: Final video output path is empty.")

            logger.info("PHASE_3_MEDIA_PRODUCTION", f"Media Production Complete! Video: {state.asset_paths.final_video}", pipeline_id=p_id, component="MEDIA_PRODUCER")
            tracer.record_step(state, "MEDIA_READY", message=msg_media)

            # 5. YouTube Publishing
            logger.info("PHASE_4_YOUTUBE_PUBLISHING", "Publishing Video to YouTube with Synthetic Metadata...", pipeline_id=p_id, component="PUBLISHER")
            msg_pub = await self.publisher.process(state)

            logger.info("PHASE_4_YOUTUBE_PUBLISHING", f"🎉 Pipeline Run Completed Successfully! Video ID: {state.upload_metadata.video_id}", pipeline_id=p_id, component="PUBLISHER")
            tracer.record_step(state, "PUBLISHED_SUCCESS", message=msg_pub)

        except Exception as e:
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

        return state
