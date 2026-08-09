from enum import Enum
from typing import Dict, Any, Optional
import hashlib
import json
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    ORCHESTRATOR = "ORCHESTRATOR"
    FACT_RETRIEVER = "FACT_RETRIEVER"
    STORY_DESIGNER = "STORY_DESIGNER"
    OBSERVER = "OBSERVER"
    MEDIA_PRODUCER = "MEDIA_PRODUCER"
    PUBLISHER = "PUBLISHER"


class AgentIntent(str, Enum):
    TOPIC_SELECTED = "TOPIC_SELECTED"
    GENERATE_SCRIPT = "GENERATE_SCRIPT"
    REVISE_SCRIPT = "REVISE_SCRIPT"
    APPROVE_SCRIPT = "APPROVE_SCRIPT"
    MEDIA_READY = "MEDIA_READY"
    PUBLISHED_SUCCESS = "PUBLISHED_SUCCESS"
    FAILURE_REPORT = "FAILURE_REPORT"


class A2AMessage(BaseModel):
    message_id: str
    sender: AgentRole
    target: AgentRole
    intent: AgentIntent
    payload: Dict[str, Any] = Field(default_factory=dict)
    state_hash: Optional[str] = None
    timestamp: str


def compute_state_hash(state) -> str:
    """Stable fingerprint of the script-relevant state. The REVISE_SCRIPT
    handler uses it to prove it is fixing exactly the state the Observer
    audited, not a stale checkpoint."""
    try:
        sd = state.script_data
        if sd is None:
            body = {"script_data": None}
        else:
            body = {
                "script_data": {
                    "title": sd.title,
                    "target_shots": sd.target_shots,
                    "shots": [
                        {
                            "shot_id": s.shot_id,
                            "act_index": s.act_index,
                            "narration_text": s.narration_text,
                            "visual_prompt": s.visual_prompt,
                        }
                        for s in sd.shots
                    ],
                }
            }
        raw = json.dumps(body, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""
