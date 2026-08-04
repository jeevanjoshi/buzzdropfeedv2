import os
import sys
from dotenv import load_dotenv

# Ensure local packages are loadable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.schemas.state import GlobalState
from src.engine.quality_verifier import quality_verifier

def main():
    load_dotenv()
    
    state_file = "logs/state_csvg-exec-20260804-105409.json"
    if not os.path.exists(state_file):
        print(f"Error: State file {state_file} not found.")
        sys.exit(1)
        
    print(f"Loading state checkpoint from {state_file}...")
    with open(state_file, "r", encoding="utf-8") as f:
        state = GlobalState.model_validate_json(f.read())
        
    print("\n--- Running Post-Production Quality Gates (Local Media Validation) ---\n")
    
    # Gate 3b: Subtitle-to-Script Coherence
    print("Checking Gate 3b (Subtitle Coherence)...")
    gate3b_pass, gate3b_issues = quality_verifier.verify_gate3b_subtitle_text_coherence(state)
    print(f"Result: {'PASS' if gate3b_pass else 'FAIL'}")
    if gate3b_issues:
        print(f"Issues: {gate3b_issues}")
    print()

    # Gate 4: Video Coherence
    print("Checking Gate 4 (Video and Audio Coherence / Duration Drift)...")
    gate4_pass, gate4_issues = quality_verifier.verify_gate4_video_audio_coherence(state)
    print(f"Result: {'PASS' if gate4_pass else 'FAIL'}")
    if gate4_issues:
        print(f"Issues: {gate4_issues}")
    print()

    # Gate 5: AI Disclosure Tags
    print("Checking Gate 5 (AI Disclosure)...")
    pipeline_tags = ["flux.1", "kokoro-tts", "wan2.1"]
    gate5_result = quality_verifier.verify_gate5_ai_disclosure_tags(state.script_data, pipeline_tags)
    print(f"Result: {gate5_result}")
    print()

if __name__ == "__main__":
    main()
