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

    orchestrator = OrchestratorAgent()
    state = asyncio.run(orchestrator.run_pipeline(
        use_live_rss=use_live_rss,
        region=region,
        publish=publish,
        dummy_frames=dummy_frames
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
