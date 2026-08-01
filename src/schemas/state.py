from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class VerifiedFact(BaseModel):
    source_id: str
    headline: str
    summary: str
    url: str
    timestamp: Optional[str] = None


class TopicCandidate(BaseModel):
    candidate_id: str
    headline: str
    summary: str
    source_url: str
    keywords: List[str]
    tvs_score: float = Field(..., description="Exponential Moving Average Trend Velocity Score")
    rpm_score: float = Field(..., description="Advertiser CPM Cosine Similarity Score")
    idi_score: float = Field(..., description="Semantic Novelty / Information Density Index")
    sdi_score: float = Field(..., description="Sentiment Disruption & Controversy Index")
    shm_score: float = Field(default=1.0, description="Social Media Hype Multiplier (X/Twitter/Reddit)")
    vph_score: float = Field(default=1.0, description="YouTube Competitor Views-per-Hour Velocity")
    sat_score: float = Field(..., description="Market Saturation Penalty")
    topsis_score: Optional[float] = Field(None, description="TOPSIS Relative Closeness Score (C_i*)")


class ShotData(BaseModel):
    shot_id: int
    act_index: int = Field(..., ge=1, le=6, description="Act 1 to 6 in 6-Act dramatic model")
    narration_text: str = Field(..., description="Narration text (max 20 words per sub-chunk)")
    visual_prompt: str = Field(..., description="16:9 cinematic visual prompt")
    duration_estimate: float = Field(default=5.0, description="Estimated duration in seconds")


class ScriptData(BaseModel):
    title: str
    target_shots: int
    shots: List[ShotData]
    estimated_runtime_seconds: float = 0.0


class AssetPaths(BaseModel):
    audio: Dict[str, str] = Field(default_factory=dict, description="Map of shot_id -> wav path")
    subtitles: Dict[str, str] = Field(default_factory=dict, description="Map of shot_id -> ass path")
    visuals: Dict[str, str] = Field(default_factory=dict, description="Map of shot_id -> mp4 path")
    final_video: Optional[str] = None


class UploadMetadata(BaseModel):
    video_id: Optional[str] = None
    status: str = "PENDING"
    retry_count: int = 0
    synthetic_content_flag: bool = True


class GlobalState(BaseModel):
    pipeline_id: str
    timestamp: str
    execution_stage: str = "INITIALIZATION"
    selected_topic: Optional[TopicCandidate] = None
    verified_facts: List[VerifiedFact] = Field(default_factory=list)
    script_data: Optional[ScriptData] = None
    asset_paths: AssetPaths = Field(default_factory=AssetPaths)
    upload_metadata: UploadMetadata = Field(default_factory=UploadMetadata)
