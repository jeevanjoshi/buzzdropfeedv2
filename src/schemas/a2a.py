from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    ORCHESTRATOR = "ORCHESTRATOR"
    FACT_RETRIEVER = "FACT_RETRIEVER"
    STORY_DESIGNER = "STORY_DESIGNER"
    OBSERVER = "OBSERVER"
    MEDIA_PRODUCER = "MEDIA_PRODUCER"
    PUBLISHER = "PUBLISHER"


class AgentIntent(str, Enum):
    FETCH_TOPIC = "FETCH_TOPIC"
    TOPIC_SELECTED = "TOPIC_SELECTED"
    GENERATE_SCRIPT = "GENERATE_SCRIPT"
    REVISE_SCRIPT = "REVISE_SCRIPT"
    APPROVE_SCRIPT = "APPROVE_SCRIPT"
    PRODUCE_MEDIA = "PRODUCE_MEDIA"
    MEDIA_READY = "MEDIA_READY"
    PUBLISH_VIDEO = "PUBLISH_VIDEO"
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
