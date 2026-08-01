import os
import json
from src.engine.logger import PipelineLogger
from src.engine.tracer import PipelineTracer
from src.schemas.state import GlobalState


def test_logger_and_tracer_diagnostics():
    log_dir = "/tmp/csvg_test_logs"
    logger = PipelineLogger(log_dir=log_dir)
    tracer = PipelineTracer(log_dir=log_dir)

    state = GlobalState(pipeline_id="diag-test-001", timestamp="2026-08-02T00:00:00Z")

    # Record Info Log
    logger.info("TEST_STAGE", "Test log message", pipeline_id="diag-test-001", component="TEST")

    # Record Error Log with Fix Hint
    try:
        raise ValueError("FFmpeg execution error sample")
    except Exception as e:
        logger.error(
            "MEDIA_STAGE", "Failed to render FFmpeg clip",
            pipeline_id="diag-test-001", component="MEDIA",
            exception=e, fix_hint="Install FFmpeg via sudo apt install ffmpeg"
        )

    # Record Tracer Step
    tracer.record_step(state, "TEST_STEP", status="SUCCESS")

    # Verify Log File Content
    log_file = os.path.join(log_dir, "csvg_execution.log")
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        lines = f.readlines()
        assert len(lines) >= 2
        last_json = json.loads(lines[-1])
        assert last_json["level"] == "ERROR"
        assert "Install FFmpeg" in last_json["fix_hint"]

    # Verify Trajectory JSON File Content
    traj_file = os.path.join(log_dir, "trajectory_diag-test-001.json")
    assert os.path.exists(traj_file)
    with open(traj_file, "r") as f:
        traj_data = json.load(f)
        assert traj_data["pipeline_id"] == "diag-test-001"
        assert len(traj_data["history"]) >= 1
