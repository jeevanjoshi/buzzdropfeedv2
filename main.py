import sys
import asyncio

from dotenv import load_dotenv

# Load configuration variables from .env file BEFORE importing agents so modules
# that read env flags at import/init time (e.g. USE_SEMANTIC_GATES) see them.
load_dotenv()

from src.agents.orchestrator import OrchestratorAgent


def print_help() -> None:
    """Print usage for the CSVG pipeline and exit. main.py has no argparse, so
    help is handled as an explicit early-exit (never runs the pipeline)."""
    print("""CSVG Autonomous YouTube Pipeline — usage
========================================

Run the end-to-end pipeline (RSS -> topic -> script -> visuals -> render -> publish).
Unknown flags are silently ignored; there is no positional-arg help beyond this page.

BASIC
  python3 main.py                                 # default region + publish
  python3 main.py --global                         # target global region (default-ish)
  python3 main.py --india                          # target India region
  python3 main.py --offline                        # canned topic candidates (NOT a full dry-run:
                                                   #   RAG, LLM, visuals, TTS, publish still run)

DISTRIBUTION / FEED
  --till-upload | --no-upload    stop before YouTube publish
  --dummy-frames | --dummy-frame synthetic visuals, skip fal/Replicate

RENDERING
  --renderer ffmpeg|moviepy      renderer to use (default ffmpeg)
  --crossfade <seconds>          crossfade duration (float, default 0.5)
  --tail <seconds>               video-only hold after each shot's narration
                                 (float, default 1.2; 0 = cut on narration end)

RAG / RESEARCH  (A/B: Google Search grounding vs 5-scraper path)
  --rag grounded                 Google Search grounding only (cited facts)
  --rag hybrid                   grounded cited core + on-topic scraper depth
  --rag scraper                  5-scraper RAG path (default)

RESUME
  --resume <pipeline_id>         resume from logs/state_<pipeline_id>.json

EXAMPLES
  python3 main.py --global --till-upload                 # real run, no publish
  python3 main.py --global --rag grounded --till-upload  # grounded research arm, no publish
  python3 main.py --global --rag hybrid --till-upload    # grounded core + scraper depth
  python3 main.py --global --rag scraper                 # scraper RAG arm (default), publish
  python3 main.py --resume csvg-exec-20260805-185905     # resume a specific run
  python3 main.py --offline --dummy-frames --till-upload # offline-ish smoke test

NOTES
  - Not a real --help parser: shown only when --help / -h appears in argv.
  - Semantic gates (USE_SEMANTIC_GATES) and soft-approval (ALLOW_SOFT_APPROVAL)
    are env-only (.env), not CLI flags.
  - See README.md and debugging_060820260057.md for run_production.sh flags.
""")


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        return

    use_live_rss = True
    region = "all"
    publish = True
    dummy_frames = False
    renderer = "ffmpeg"
    crossfade = 0.5
    pad_after_narration = None
    rag_mode = "scraper"

    state = None
    if len(sys.argv) > 1:
        if "--offline" in sys.argv:
            use_live_rss = False
        if "--india" in sys.argv:
            region = "india"
        elif "--global" in sys.argv:
            region = "global"
        if "--rag" in sys.argv:
            for i, arg in enumerate(sys.argv):
                if arg == "--rag" and i + 1 < len(sys.argv):
                    val = sys.argv[i + 1].strip().lower()
                    if val in ("grounded", "g"):
                        rag_mode = "grounded"
                    elif val in ("hybrid", "h"):
                        rag_mode = "hybrid"
                    else:
                        rag_mode = "scraper"
        if "--till-upload" in sys.argv or "--no-upload" in sys.argv:
            publish = False
        if "--dummy-frames" in sys.argv or "--dummy-frame" in sys.argv:
            dummy_frames = True
        if "--outline-first" in sys.argv:
            os.environ["CSVG_OUTLINE_FIRST"] = "1"
            print("[main] --outline-first enabled (outline-validated beat narration)")
        if "--renderer" in sys.argv:
            for i, arg in enumerate(sys.argv):
                if arg == "--renderer" and i + 1 < len(sys.argv):
                    renderer = sys.argv[i + 1]
        if "--crossfade" in sys.argv:
            for i, arg in enumerate(sys.argv):
                if arg == "--crossfade" and i + 1 < len(sys.argv):
                    try:
                        crossfade = float(sys.argv[i + 1])
                    except ValueError:
                        crossfade = 0.5
        if "--tail" in sys.argv:
            for i, arg in enumerate(sys.argv):
                if arg == "--tail" and i + 1 < len(sys.argv):
                    try:
                        pad_after_narration = float(sys.argv[i + 1])
                    except ValueError:
                        pad_after_narration = None
        
        # Parse --resume <pipeline_id>
        for i, arg in enumerate(sys.argv):
            if arg == "--resume" and i + 1 < len(sys.argv):
                pipeline_id = sys.argv[i + 1]
                state_file = f"logs/state_{pipeline_id}.json"
                import os
                if os.path.exists(state_file):
                    print(f"Loading state checkpoint from {state_file}...")
                    from src.schemas.state import GlobalState
                    with open(state_file, "r", encoding="utf-8") as f:
                        state = GlobalState.model_validate_json(f.read())
                else:
                    print(f"Error: Checkpoint file {state_file} not found.")
                    sys.exit(1)

    orchestrator = OrchestratorAgent()
    state = asyncio.run(orchestrator.run_pipeline(
        use_live_rss=use_live_rss,
        region=region,
        publish=publish,
        dummy_frames=dummy_frames,
        state=state,
        renderer=renderer,
        crossfade=crossfade,
        pad_after_narration=pad_after_narration,
        rag_mode=rag_mode
    ))

    print(f"Summary of Pipeline Execution:")
    print(f"- Selected Topic: {state.selected_topic.headline if state.selected_topic else 'None'}")
    print(f"- TOPSIS Score: {state.selected_topic.topsis_score if state.selected_topic else 0.0}")
    print(f"- Script Title: {state.script_data.title if state.script_data else 'None'}")
    print(f"- Video Runtime: {state.script_data.estimated_runtime_seconds/60.0:.2f} mins" if state.script_data else "-")
    print(f"- Master Video Output: {state.asset_paths.final_video}")
    print(f"- YouTube Video ID: {state.upload_metadata.video_id if state.upload_metadata else 'Skipped (Till Upload Mode)'}")
    print(f"- Final Stage: {state.execution_stage}")


if __name__ == "__main__":
    main()
