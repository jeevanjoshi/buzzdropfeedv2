import os
import json
import datetime
from typing import Dict, Any, List, Optional
from src.schemas.state import GlobalState
from src.schemas.a2a import A2AMessage


class PipelineTracer:
    """
    Trajectory & State Tracer recording step-by-step A2A events and state transitions.
    Saves a trajectory JSON artifact (logs/trajectory_<pipeline_id>.json) for auditability & failure diagnosis.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def record_step(
        self,
        state: GlobalState,
        step_name: str,
        message: Optional[A2AMessage] = None,
        status: str = "SUCCESS",
        error_details: Optional[Dict[str, Any]] = None
    ):
        p_id = state.pipeline_id or "pipeline"
        file_path = os.path.join(self.log_dir, f"trajectory_{p_id}.json")

        trajectory_data = {
            "pipeline_id": p_id,
            "current_stage": state.execution_stage,
            "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "history": []
        }

        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    trajectory_data = json.load(f)
            except Exception:
                pass

        step_record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "step_name": step_name,
            "stage": state.execution_stage,
            "status": status,
            "selected_topic": state.selected_topic.headline if state.selected_topic else None,
            "topsis_score": state.selected_topic.topsis_score if state.selected_topic else None,
            "script_title": state.script_data.title if state.script_data else None,
            "a2a_message": message.model_dump() if message else None,
            "error_details": error_details or None
        }

        trajectory_data["history"].append(step_record)
        trajectory_data["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(trajectory_data, f, indent=2)


tracer = PipelineTracer()
