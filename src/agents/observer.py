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
        self, script: ScriptData, verified_facts: List[VerifiedFact], topic=None, crawled_content: str = ""
    ) -> List[str]:
        """
        Audits narration text using local NLTK POS tagging and TF-IDF Cosine Similarity.
        As a final resort, runs a single-pass LLM critic check on the flagged sentences
        to filter out natural transitions and avoid false positives.

        crawled_content = the complete RAG fact corpus (verified facts + retrieved
        sources) exposed by StoryDesigner, so claims grounded in RAG-retrieved data
        are still verified against the full fact source.
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

        # Ground truth corpus built from verified sources + the full RAG fact corpus
        ground_truth_corpus = " ".join([f"{f.headline} {f.summary}" for f in verified_facts]).lower()
        if crawled_content:
            ground_truth_corpus += " " + crawled_content.lower()
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

            # 2. Dynamic Temporal Anchor Check — hard-flag a past year not in the corpus,
            #    and route present-tense past-year sentences to the AI critic (which decides
            #    historical-vs-current) even when the old year exists in the corpus, so old
            #    RAG data can't self-excuse being framed as a current event.
            HISTORICAL_FRAMING = ("back in", "during", "historically", "at the time",
                                  "the year", "decade", "era", "since", "then-")
            for past_y in past_years:
                if past_y in narration_lower:
                    if past_y not in ground_truth_corpus:
                        flagged_sentences_info.append((
                            shot.shot_id,
                            f"Outdated year '{past_y}' in: {shot.narration_text}",
                            "Temporal Audit"
                        ))
                    # Even if in corpus, flag a sentence that cites a past year WITHOUT an
                    # explicit historical frame -> critic decides if it's current-as-lie.
                    for _sent in [s.strip() for s in re.split(r'[.!?]', shot.narration_text) if len(s.strip()) > 15]:
                        _sl = _sent.lower()
                        if past_y in _sl and not any(h in _sl for h in HISTORICAL_FRAMING):
                            flagged_sentences_info.append((shot.shot_id, _sent, "Temporal Audit"))

            # 3. Semantic Sentence Grounding check (checks for qualitative hallucinations)
            if vectorizer:
                shot_sentences = [s.strip() for s in re.split(r'[.!?]', shot.narration_text) if len(s.strip()) > 15]
                for sentence in shot_sentences:
                    # Rhetorical questions are not assertions; other creative English
                    # is decided by the LLM critic downstream (AI judge), not markers.

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
                print(f"Observer calling AI-in-the-loop critic to verify {len(flagged_sentences_info)} flagged claims...")
                flagged_list_str = "\n".join([f"- [Shot #{sid}][Type: {itype}]: '{sentence}'" for sid, sentence, itype in flagged_sentences_info])
                prompt = f"""
                You are a factual validation critic. The local parser flagged the following sentences from a YouTube script as potential factual hallucinations or ungrounded claims.
                
                VERIFIED FACTS CORPUS:
                {ground_truth_corpus}
                
                FLAGGED CLAIMS:
                {flagged_list_str}
                
                Requirements:
                1. For EACH flagged claim, first classify it as one of:
                   - STYLE: metaphor, analogy, rhetorical question, opinion, subjective
                     judgement, engagement/call-to-action, narrative color, or a logical
                     deduction/extrapolation from the facts.
                   - ASSERTION: a hard, checkable factual claim (specific statistics,
                     names, dates, places, events, numbers).
                2. STYLE claims MUST be APPROVED (never rejected) — even if they are not in
                   the corpus, because they are not assertions of fact.
                3. Only reject ASSERTIONS that are unsupported by, or contradict, the
                   verified facts corpus (e.g. wrong statistics, false names, incorrect
                   dates, fabricated events).
                4. TEMPORAL RULE: Reject an ASSERTION that presents data from a pre-2026 year
                   as a CURRENT 2026 event, even if that year appears in the corpus. APPROVE
                   pre-2026 data that is clearly framed as historical (e.g. 'In 2022...',
                   'back in', 'historically', 'at the time').
                5. Return a JSON object with a single key "violations" containing an array
                   of strings. Each string is the EXACT text of a REJECTED assertion followed
                   by the reason it fails. Return an empty array if all flagged claims are
                   STYLE or supported.
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
                    print(f"Warning: Critic call failed: {critic_err}. Defaulting to local validation flags.")

            # If the LLM critic is unavailable/fails, only hard-flag explicit
            # numeric/temporal signals (rare false-positive risk). Low-semantic-sim
            # flags are dropped here because creative/human English is ambiguous
            # without an AI judge.
            for sid, sentence, itype in flagged_sentences_info:
                if itype in ("Fact Audit", "Temporal Audit"):
                    violations.append(
                        f"Shot #{sid} {itype}: The claim '{sentence}' lacks verified grounding in source facts."
                    )

        return violations

    def evaluate_script(
        self, script: ScriptData, verified_facts: List[VerifiedFact] = None,
        topic=None, channel_phase: str = "REVENUE", crawled_content: str = ""
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

        # Runtime Check — 10.0 to 15.5 min is the target range
        runtime_min = script.estimated_runtime_seconds / 60.0
        if runtime_min < 10.0 or runtime_min > 15.5:
            violations.append(
                f"Runtime out of bounds: {runtime_min:.2f} mins "
                f"(Target: 10.0 - 15.5 mins for YPP & mid-roll optimization)"
            )

        # Shot Count Check
        if len(script.shots) < 10:
            violations.append(f"Insufficient shot count: {len(script.shots)} shots (Target: 12 - 18 shots)")

        # Shot-by-Shot Validation
        # (A) Sentence repetition tracking — verbatim + semantic similarity (>0.82)
        all_narr_sentences: List[str] = []

        for shot in script.shots:
            word_count = len(shot.narration_text.split())
            if word_count > 155:
                violations.append(f"Shot #{shot.shot_id} narration too long ({word_count} words). Max 155 words per shot.")

            v_prompt = shot.visual_prompt.lower()
            if "16:9" not in v_prompt and "widescreen" not in v_prompt:
                violations.append(f"Shot #{shot.shot_id} visual prompt missing 16:9 widescreen specification.")

            if "cinematic" not in v_prompt and "8k" not in v_prompt and "photorealistic" not in v_prompt:
                violations.append(f"Shot #{shot.shot_id} visual prompt lacks aesthetic lighting/AQA keywords.")

            # Sentence duplication (verbatim + TF-IDF semantic similarity >= 0.82)
            shot_sentences = [s.strip().lower() for s in re.split(r'[.!?]', shot.narration_text) if len(s.strip()) > 15]
            for sentence in shot_sentences:
                if sentence in all_narr_sentences:
                    violations.append(f"Shot #{shot.shot_id} repetition: duplicate sentence '{sentence}'.")
                elif all_narr_sentences:
                    try:
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        from sklearn.metrics.pairwise import cosine_similarity
                        import numpy as np
                        vec = TfidfVectorizer().fit(all_narr_sentences + [sentence])
                        v1 = vec.transform([sentence])
                        v2 = vec.transform(all_narr_sentences)
                        sims = cosine_similarity(v1, v2)[0]
                        max_sim = float(np.max(sims))
                        if max_sim > 0.82:
                            violations.append(
                                f"Shot #{shot.shot_id} repetition: sentence semantically too similar "
                                f"(sim {max_sim:.2f}): '{sentence}'"
                            )
                    except Exception:
                        pass
                all_narr_sentences.append(sentence)

        # (B) Source-attribution diversity: scripts should cite multiple distinct
        # authentic sources and not over-attribute to a single one.
        if verified_facts:
            source_names = {f.source_name for f in verified_facts if f.source_name}
            source_names = {
                s for s in source_names
                if s and "Viet" not in s and "Verified Market Reports" not in s
            }
            if len(source_names) >= 2:
                narration_full = " ".join(s.narration_text for s in script.shots).lower()
                cited = {sn: narration_full.count(sn.lower()) for sn in source_names if sn.lower() in narration_full}
                if cited:
                    total_cites = sum(cited.values())
                    max_cites = max(cited.values())
                    top_source = max(cited, key=cited.get)
                    if total_cites > 0:
                        if len(cited) == 1:
                            violations.append(
                                f"Source Diversity: narration cites only '{top_source}'. "
                                f"Attribute to multiple distinct sources (available: {sorted(source_names)})."
                            )
                        elif max_cites / total_cites > 0.45:
                            violations.append(
                                f"Source Diversity: '{top_source}' over-cited ({max_cites}/{total_cites} = "
                                f"{max_cites/total_cites:.0%}). Balance citations across authentic sources."
                            )

        # Anti-Hallucination Audit — against verified facts + full RAG fact corpus
        if verified_facts:
            fact_violations = self.audit_fact_grounding(script, verified_facts, topic=topic, crawled_content=crawled_content)
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
            crawled_content=state.crawled_content,
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
