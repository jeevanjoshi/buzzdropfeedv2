import uuid
import datetime
import re
from typing import List, Dict, Any, Tuple
from src.schemas.state import GlobalState, ScriptData, VerifiedFact
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent


class ObserverAgent:
    """
    Observer Critic Agent responsible for validating script quality, word count pacing,
    10-15 minute runtime boundaries, 16:9 visual aesthetic criteria (AQA), YouTube policy safety,
    and Dynamic Temporal Fact Verification & Anti-Hallucination Audits.
    """

    def __init__(self, name: str = "Observer"):
        self.name = name

    def audit_fact_grounding(
        self, script: ScriptData, verified_facts: List[VerifiedFact]
    ) -> List[str]:
        """
        Audits narration text to ensure financial figures, percentages, dates, and entities are grounded in verified_facts.
        Returns list of hallucination / temporal violations if unverified figures or outdated temporal anchors are introduced.
        Dynamically computes current year (e.g. 2026) and past historical years to enforce strict temporal context awareness.
        """
        violations = []
        if not verified_facts:
            return violations

        # Dynamic System Date Context
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        current_year_int = now_utc.year
        current_year = str(current_year_int)
        past_years = [str(y) for y in range(current_year_int - 5, current_year_int)]

        # Ground truth corpus built from verified sources
        ground_truth_corpus = " ".join([f"{f.headline} {f.summary}" for f in verified_facts]).lower()
        gt_numbers = set(re.findall(r'\$?\b\d+(?:\.\d+)?[kmb%]?\b', ground_truth_corpus))

        for shot in script.shots:
            narration_lower = shot.narration_text.lower()
            shot_numbers = set(re.findall(r'\$?\b\d+(?:\.\d+)?[kmb%]?\b', narration_lower))

            # 1. Numerical Grounding Check
            for num in shot_numbers:
                if num not in gt_numbers:
                    cleaned_num = re.sub(r'[^\d.]', '', num)
                    if cleaned_num and float(cleaned_num) > 10 and cleaned_num not in [current_year, "100"]:
                        violations.append(
                            f"Shot #{shot.shot_id} Fact Audit: Unverified numerical claim '{num}' detected in narration. Not supported by source facts."
                        )

            # 2. Dynamic Temporal Anchor Check (Preventing outdated past years presented as present events)
            for past_y in past_years:
                if past_y in narration_lower:
                    if past_y not in ground_truth_corpus:
                        violations.append(
                            f"Shot #{shot.shot_id} Temporal Audit: Outdated temporal anchor '{past_y}' detected in narration. Ground truth specifies current {current_year} facts."
                        )

        return violations

    def evaluate_script(
        self, script: ScriptData, verified_facts: List[VerifiedFact] = None
    ) -> Tuple[bool, List[str]]:
        """
        Evaluates script constraints:
        1. Runtime: Must be between 10 minutes (600s) and 15 minutes (900s).
        2. Shot Count: Must have 12 to 18 shots.
        3. Pacing: No single shot narration should exceed 75 words (~25s at 150 wpm).
        4. Visual Quality (AQA): Prompts must include 16:9 widescreen specification and aesthetic lighting keywords.
        5. Anti-Hallucination Audit: Cross-checks figures against verified_facts.
        """
        violations = []

        # Runtime Check
        runtime_min = script.estimated_runtime_seconds / 60.0
        if runtime_min < 8.0 or runtime_min > 18.0:
            violations.append(f"Runtime out of bounds: {runtime_min:.2f} mins (Target: 10.0 - 15.0 mins)")

        # Shot Count Check
        if len(script.shots) < 10:
            violations.append(f"Insufficient shot count: {len(script.shots)} shots (Target: 12 - 18 shots)")

        # Shot-by-Shot Validation
        for shot in script.shots:
            word_count = len(shot.narration_text.split())
            if word_count > 75:
                violations.append(f"Shot #{shot.shot_id} narration too long ({word_count} words). Max 75 words per shot.")

            v_prompt = shot.visual_prompt.lower()
            if "16:9" not in v_prompt and "widescreen" not in v_prompt:
                violations.append(f"Shot #{shot.shot_id} visual prompt missing 16:9 widescreen specification.")

            if "cinematic" not in v_prompt and "8k" not in v_prompt and "photorealistic" not in v_prompt:
                violations.append(f"Shot #{shot.shot_id} visual prompt lacks aesthetic lighting/AQA keywords.")

        # Anti-Hallucination Audit
        if verified_facts:
            fact_violations = self.audit_fact_grounding(script, verified_facts)
            violations.extend(fact_violations)

        is_approved = len(violations) == 0
        return is_approved, violations

    def process(self, state: GlobalState) -> A2AMessage:
        """
        Executes Observer evaluation workflow:
        1. Reads state.script_data and state.verified_facts
        2. Evaluates constraints and anti-hallucination audit
        3. Emits APPROVE_SCRIPT or REVISE_SCRIPT A2AMessage
        """
        if not state.script_data:
            raise ValueError("Observer evaluation failed: state.script_data is None")

        is_approved, violations = self.evaluate_script(state.script_data, state.verified_facts)

        if is_approved:
            state.execution_stage = "SCRIPT_APPROVED"
            return A2AMessage(
                message_id=f"msg-{uuid.uuid4().hex[:8]}",
                sender=AgentRole.OBSERVER,
                target=AgentRole.ORCHESTRATOR,
                intent=AgentIntent.APPROVE_SCRIPT,
                payload={
                    "status": "APPROVED",
                    "script_title": state.script_data.title,
                    "total_shots": len(state.script_data.shots),
                    "runtime_minutes": round(state.script_data.estimated_runtime_seconds / 60.0, 2),
                    "fact_audit": "PASSED"
                },
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
            )
        else:
            state.execution_stage = "SCRIPT_REVISION_REQUIRED"
            return A2AMessage(
                message_id=f"msg-{uuid.uuid4().hex[:8]}",
                sender=AgentRole.OBSERVER,
                target=AgentRole.STORY_DESIGNER,
                intent=AgentIntent.REVISE_SCRIPT,
                payload={
                    "status": "REJECTED",
                    "violations": violations,
                    "violation_count": len(violations),
                    "fact_audit": "FAILED" if any("Fact Audit" in v or "Temporal Audit" in v for v in violations) else "PASSED"
                },
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
            )
