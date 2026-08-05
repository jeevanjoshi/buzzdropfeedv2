import asyncio
import os
import requests
from src.engine.llm_client import LLMClient
from src.schemas.state import TopicCandidate, VerifiedFact

client = LLMClient()
topic = TopicCandidate(
    candidate_id="c1",
    headline="Test Nvidia Microchip Valuation",
    summary="Nvidia unveils next-gen microchips that will change AI",
    source_url="http://example.com",
    keywords=["Nvidia", "AI", "Microchip"],
    tvs_score=0.8,
    rpm_score=0.9,
    idi_score=0.7,
    sdi_score=0.6,
    sat_score=0.5,
    topsis_score=0.8,
    niche_category="Technology"
)
facts = [
    VerifiedFact(source_id="f1", headline="Nvidia releases new chip architecture", summary="The new architecture provides 2x speedup.", url="http://example.com", source_name="Nvidia Official")
]

from src.agents.story_designer import StoryDesignerAgent
designer = StoryDesignerAgent(llm_client=client)

print("Calling generate_6act_script...")
try:
    script = designer.generate_6act_script(topic, facts)
    print("Script successfully generated!")
except Exception as e:
    print(f"Exception raised: {e}")
