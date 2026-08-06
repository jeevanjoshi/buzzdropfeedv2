"""
Suite runner for the CSVG pipeline tests.

There is exactly ONE test module: tests/test_smoke.py — a fully hermetic
end-to-end smoke test (positive / negative / edge cases) that exercises every
orchestrator stage with mocked agents and singletons. It performs:
  * NO live/network calls
  * NO Raspberry Pi interaction
  * NO media generation (no ffmpeg / TTS / visuals)

Use the no-attach runner (imports each case function directly, no unittest):
    python run_tests.py
"""
import sys
from tests.test_smoke import (
    test_positive_full_pipeline_publishes,
    test_edge_publish_off_till_upload,
    test_edge_resume_from_state,
    test_negative_no_topic_selected,
    test_negative_rag_insufficient,
    test_negative_script_generation_fails,
    test_negative_observer_hard_reject_persists,
    test_negative_media_production_fails,
    test_negative_quality_gate_fails,
)

CASES = [
    ("POSITIVE full pipeline publishes", test_positive_full_pipeline_publishes),
    ("EDGE publish off -> QUALITY_VERIFIED", test_edge_publish_off_till_upload),
    ("EDGE resume from state", test_edge_resume_from_state),
    ("NEGATIVE no topic selected", test_negative_no_topic_selected),
    ("NEGATIVE RAG insufficient", test_negative_rag_insufficient),
    ("NEGATIVE script generation fails", test_negative_script_generation_fails),
    ("NEGATIVE observer hard reject", test_negative_observer_hard_reject_persists),
    ("NEGATIVE media production fails", test_negative_media_production_fails),
    ("NEGATIVE quality gate fails", test_negative_quality_gate_fails),
]


def run_all() -> int:
    failures = 0
    for name, fn in CASES:
        try:
            fn()
        except Exception as e:
            failures += 1
            print(f"FAIL [{name}]: {type(e).__name__}: {e}")
    print(f"\nSMOKE SUMMARY: {len(CASES) - failures}/{len(CASES)} cases passed. "
          f"(No media, no live calls, no Pi interaction.)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
