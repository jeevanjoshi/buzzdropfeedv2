import sys
import asyncio
from src.agents.orchestrator import OrchestratorAgent


from dotenv import load_dotenv

# Load configuration variables from .env file
load_dotenv()


def main():
    use_live_rss = True
    region = "all"
    publish = True
    dummy_frames = False
    renderer = "ffmpeg"
    crossfade = 0.5

    state = None
    if len(sys.argv) > 1:
        if "--offline" in sys.argv:
            use_live_rss = False
        if "--india" in sys.argv:
            region = "india"
        elif "--global" in sys.argv:
            region = "global"
        if "--till-upload" in sys.argv or "--no-upload" in sys.argv:
            publish = False
        if "--dummy-frames" in sys.argv or "--dummy-frame" in sys.argv:
            dummy_frames = True
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
        crossfade=crossfade
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
