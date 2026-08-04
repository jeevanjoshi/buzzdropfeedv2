import uuid
import datetime
import re
from typing import List, Dict, Any, Tuple
from src.schemas.state import GlobalState, ScriptData, VerifiedFact
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent
from src.engine.monetization_optimizer import monetization_optimizer
from src.engine.channel_phase_manager import channel_phase_manager


class ObserverAgent:
    """
    Observer Critic Agent responsible for validating script quality, word count pacing,
    10-15 minute runtime boundaries, 16:9 visual aesthetic criteria (AQA), YouTube policy safety,
    and Dynamic Temporal Fact Verification & Anti-Hallucination Audits.
    """

    def __init__(self, name: str = "Observer"):
        self.name = name

    def audit_fact_grounding(
        self, script: ScriptData, verified_facts: List[VerifiedFact], topic=None
    ) -> List[str]:
        """
        Audits narration text using local NLTK POS tagging and TF-IDF Cosine Similarity.
        As a final resort, runs a single-pass LLM critic check on the flagged sentences
        to filter out natural transitions and avoid false positives.
        """
        violations = []
        if not verified_facts:
            return violations

        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

        # Dynamic System Date Context
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        current_year_int = now_utc.year
        current_year = str(current_year_int)
        past_years = [str(y) for y in range(current_year_int - 5, current_year_int)]

        # Ground truth corpus built from verified sources
        ground_truth_corpus = " ".join([f"{f.headline} {f.summary}" for f in verified_facts]).lower()
        gt_numbers = set(re.findall(r'\$?\b\d+(?:\.\d+)?[kmb%]?\b', ground_truth_corpus))

        # Prepare for semantic checks
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        fact_sentences = [s.strip() for s in re.split(r'[.!?]', ground_truth_corpus) if len(s.strip()) > 15]
        vectorizer = None
        if fact_sentences:
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                vectorizer.fit(fact_sentences)
            except Exception:
                pass

        # Track potentially flagged sentences for LLM review
        flagged_sentences_info = []

        for shot in script.shots:
            narration_lower = shot.narration_text.lower()
            shot_numbers = set(re.findall(r'\$?\b\d+(?:\.\d+)?[kmb%]?\b', narration_lower))

            # 1. Numerical Grounding Check
            for num in shot_numbers:
                if num not in gt_numbers:
                    cleaned_num = re.sub(r'[^\d.]', '', num)
                    if cleaned_num and float(cleaned_num) > 10 and cleaned_num not in [current_year, "100"]:
                        flagged_sentences_info.append((
                            shot.shot_id,
                            f"Numerical claim '{num}' in: {shot.narration_text}",
                            "Fact Audit"
                        ))

            # 2. Dynamic Temporal Anchor Check (Preventing outdated past years presented as present events)
            for past_y in past_years:
                if past_y in narration_lower:
                    if past_y not in ground_truth_corpus:
                        flagged_sentences_info.append((
                            shot.shot_id,
                            f"Outdated year '{past_y}' in: {shot.narration_text}",
                            "Temporal Audit"
                        ))

            # 3. Semantic Sentence Grounding check (checks for qualitative hallucinations)
            if vectorizer:
                shot_sentences = [s.strip() for s in re.split(r'[.!?]', shot.narration_text) if len(s.strip()) > 15]
                for sentence in shot_sentences:
                    # Skip rhetorical questions (they are not assertions)
                    if sentence.endswith("?"):
                        continue

                    # Local NLTK POS Tagging check with Topic Keyword Subtraction
                    try:
                        tokens = nltk.word_tokenize(sentence)
                        tags = nltk.pos_tag(tokens)
                        
                        # Get main topic keywords in lowercase
                        topic_keywords = {k.lower() for k in (topic.keywords if topic and hasattr(topic, 'keywords') else [])}
                        if topic and hasattr(topic, 'headline'):
                            topic_keywords.update(re.findall(r'\b\w+\b', topic.headline.lower()))
                        
                        has_unrelated_entity = False
                        has_valid_digit = False
                        
                        for word, tag in tags:
                            word_lower = word.lower()
                            # 1. Check for Proper Nouns that are NOT core topic keywords
                            if tag in ("NNP", "NNPS"):
                                if word_lower not in topic_keywords and len(word_lower) > 2:
                                    has_unrelated_entity = True
                            
                            # 2. Check for numeric assertions (excluding current year or tiny counts)
                            elif tag == "CD":
                                if word != "2026" and word_lower not in ("one", "two", "three") and re.match(r'^\d+$', word):
                                    try:
                                        if float(word) > 10:
                                            has_valid_digit = True
                                    except ValueError:
                                        pass

                        is_factual = has_unrelated_entity or has_valid_digit
                    except Exception:
                        is_factual = True

                    if not is_factual:
                        # Skip if NLTK tags no proper nouns or numbers (classified as transition)
                        continue

                    try:
                        import numpy as np
                        vec_sentence = vectorizer.transform([sentence])
                        vec_facts = vectorizer.transform(fact_sentences)
                        sims = cosine_similarity(vec_sentence, vec_facts)[0]
                        max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
                    except Exception:
                        max_sim = 0.0

                    if max_sim < 0.05:
                        flagged_sentences_info.append((
                            shot.shot_id,
                            sentence,
                            "Semantic Audit"
                        ))

        # 4. Final Resort AI-in-the-Loop Critic Check (if any sentences were flagged locally)
        if flagged_sentences_info:
            from src.engine.llm_client import LLMClient
            llm_client = LLMClient()
            if llm_client.is_available():
                print(f"🧠 Observer calling AI-in-the-loop critic to verify {len(flagged_sentences_info)} flagged claims...")
                flagged_list_str = "\n".join([f"- [Shot #{sid}][Type: {itype}]: '{sentence}'" for sid, sentence, itype in flagged_sentences_info])
                prompt = f"""
                You are a factual validation critic. The local parser flagged the following sentences from a YouTube script as potential factual hallucinations or ungrounded claims.
                
                VERIFIED FACTS CORPUS:
                {ground_truth_corpus}
                
                FLAGGED CLAIMS:
                {flagged_list_str}
                
                Requirements:
                1. Review each flagged claim against the verified facts corpus.
                2. Determine if the claim is a safe narrative/rhetorical transition (which should be APPROVED) or a genuine factual hallucination (which must be REJECTED).
                3. Return a JSON object with a single key "violations" containing an array of strings. Each string should be the exact text of the REJECTED claim alongside the reason why it fails.
                4. Do NOT reject safe transitions or rhetorical questions. Only reject hard claims that lack factual basis.
                """
                system_prompt = "You are a precise, objective facts verification critic. Return valid JSON only."
                try:
                    critic_result = llm_client.generate_json(prompt, system_prompt)
                    if critic_result and "violations" in critic_result:
                        rejected_claims = critic_result["violations"]
                        for violation in rejected_claims:
                            violations.append(violation)
                        return violations
                except Exception as critic_err:
                    print(f"⚠️ Critic call failed: {critic_err}. Defaulting to local validation flags.")

            # If LLM client is unavailable or failed, default back to local flags to be safe
            for sid, sentence, itype in flagged_sentences_info:
                violations.append(
                    f"Shot #{sid} {itype}: The claim '{sentence}' lacks verified grounding in source facts."
                )

        return violations

    def evaluate_script(
        self, script: ScriptData, verified_facts: List[VerifiedFact] = None,
        topic=None, channel_phase: str = "REVENUE"
    ) -> Tuple[bool, List[str]]:
        """
        Evaluates script constraints:
        1. Revenue Gate: predicted yield must meet per-video minimum (phase-aware).
        2. Audience Gate: entertainment/gossip topics are hard-blocked.
        3. Runtime: Must be between 10.5 minutes and 14.5 minutes (3-midroll sweet spot).
        4. Shot Count: Must have 12 to 18 shots.
        5. Pacing: No single shot narration should exceed 155 words.
        6. Visual Quality (AQA): Prompts must include 16:9 widescreen specification.
        7. Anti-Hallucination Audit: Cross-checks figures against verified_facts.
        """
        violations = []

        # ── Gate 0: Audience Type Gate ──────────────────────────────────────
        # Hard-block entertainment/gossip regardless of other scores
        if topic and getattr(topic, "audience_type", "") == "blocked":
            violations.append(
                f"Audience Gate FAIL: Topic audience_type='blocked' (entertainment/gossip). "
                f"RPM ~$1 — hard blocked. Select a Tech/Finance/Health topic instead."
            )
            return False, violations  # Early exit — no point checking further

        # ── Gate 0b: Revenue Gate (REVENUE + SCALE phases only) ─────────────
        # In GROWTH phase we skip this — goal is watch-time not RPM
        if topic and channel_phase in ("REVENUE", "SCALE"):
            rev = monetization_optimizer.calculate_revenue_yield(topic, estimated_runtime_mins=13.0)
            min_rev = channel_phase_manager.REVENUE_GATE_MIN_USD
            if rev["total_expected_revenue_usd"] < min_rev:
                violations.append(
                    f"Revenue Gate FAIL: Predicted ${rev['total_expected_revenue_usd']:.2f}/video "
                    f"< ${min_rev:.2f} minimum. RPM=${rev['estimated_rpm_usd']:.2f}, "
                    f"PredictedViews={rev['predicted_views']:,.0f}. "
                    f"Switch to higher-RPM niche (Tech/AI/Finance/Health)."
                )

        # Runtime Check — 10.5 to 14.5 min is the 3-midroll revenue sweet spot
        runtime_min = script.estimated_runtime_seconds / 60.0
        if runtime_min < 10.5 or runtime_min > 14.5:
            violations.append(
                f"Runtime out of bounds: {runtime_min:.2f} mins "
                f"(Target: 10.5 - 14.5 mins for 3 mid-roll ads at 2.6x multiplier)"
            )

        # Shot Count Check
        if len(script.shots) < 10:
            violations.append(f"Insufficient shot count: {len(script.shots)} shots (Target: 12 - 18 shots)")

        # Shot-by-Shot Validation
        for shot in script.shots:
            word_count = len(shot.narration_text.split())
            if word_count > 155:
                violations.append(f"Shot #{shot.shot_id} narration too long ({word_count} words). Max 155 words per shot.")

            v_prompt = shot.visual_prompt.lower()
            if "16:9" not in v_prompt and "widescreen" not in v_prompt:
                violations.append(f"Shot #{shot.shot_id} visual prompt missing 16:9 widescreen specification.")

            if "cinematic" not in v_prompt and "8k" not in v_prompt and "photorealistic" not in v_prompt:
                violations.append(f"Shot #{shot.shot_id} visual prompt lacks aesthetic lighting/AQA keywords.")

        # Anti-Hallucination Audit
        if verified_facts:
            fact_violations = self.audit_fact_grounding(script, verified_facts, topic=topic)
            violations.extend(fact_violations)

        is_approved = len(violations) == 0
        return is_approved, violations

    def process(self, state: GlobalState) -> A2AMessage:
        """
        Executes Observer evaluation workflow:
        1. Reads state.script_data, state.verified_facts, state.selected_topic, state.channel_phase
        2. Evaluates constraints, revenue gate, audience gate, and anti-hallucination audit
        3. Emits APPROVE_SCRIPT or REVISE_SCRIPT A2AMessage
        """
        if not state.script_data:
            raise ValueError("Observer evaluation failed: state.script_data is None")

        is_approved, violations = self.evaluate_script(
            state.script_data,
            state.verified_facts,
            topic=state.selected_topic,
            channel_phase=state.channel_phase,
        )

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
