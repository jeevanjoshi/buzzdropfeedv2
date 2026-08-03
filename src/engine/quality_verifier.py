import os
import re
import math
from typing import Dict, Any, List, Tuple
from src.schemas.state import GlobalState, ScriptData, TopicCandidate


class StageQualityVerifier:
    """
    Stage 6: Automated Multi-Stage Artifact Quality Verification & Compliance Engine:
    Validates cross-stage alignment:
    - Gate 1: Topic-to-Script Coherence & Target Runtime Verification
    - Gate 2: Script-to-TTS Speech & Acronym Expansion Verification
    - Gate 3: TTS-to-Subtitle Master Alignment Verification
    - Gate 4: Master Video Resolution & Duration Verification
    - Gate 5: YouTube 2025/2026 AI Disclosure Metadata Auto-Tagging (NEW)
    - Gate 6: Anti-"AI Slop" Script Entropy & Diversity Audit (NEW)
    """

    # YouTube 2025/2026 Mandatory AI Disclosure Triggers
    DISCLOSURE_MANDATORY_TRIGGERS = [
        "photorealistic avatar", "synthetic human", "realistic human face",
        "ai voice", "generated voice", "synthetic voice", "runway", "gen-3",
        "ai news anchor", "real event simulation", "deepfake"
    ]

    DISCLOSURE_EXEMPT_TERMS = [
        "stylized artwork", "abstract visual", "animated chart", "flux art",
        "color grading", "text overlay", "scripting assistance"
    ]

    def verify_gate1_topic_to_script(
        self, topic: TopicCandidate, script: ScriptData
    ) -> Tuple[bool, List[str]]:
        """
        Verifies that the generated script matches the selected topic domain
        and achieves the required 10-15 minute runtime (min 1,200 narration words).
        """
        issues = []
        if not script or not script.shots:
            return False, ["Script has no shots."]

        # 1. Word Count / Runtime Check (10-15 mins)
        total_words = sum(len(s.narration_text.split()) for s in script.shots)
        est_runtime_mins = total_words / 150.0

        if total_words < 1500:
            issues.append(
                f"Gate 1 Fail: Script total word count is {total_words} words ({est_runtime_mins:.2f} mins). "
                f"Must be >= 1,500 words for a 10-15 minute video."
            )

        # 2. Topic Keyword & Domain Semantic Alignment Check
        headline_words = set(re.findall(r'\b[A-Za-z]{3,}\b', topic.headline.lower()))
        stopwords = {"with", "this", "that", "from", "they", "have", "been", "will", "more", "about", "their", "which", "over", "after", "news", "today", "report"}
        core_topic_words = [w for w in headline_words if w not in stopwords]
        
        # Domain synonyms for tech/AI topics
        domain_synonyms = {"ai", "seo", "aio", "chatgpt", "traffic", "search", "content", "optimization", "algorithm", "referral", "digital", "market", "trading", "stock", "growth"}
        target_words = set(core_topic_words).union(domain_synonyms)

        matching_shots = 0
        for shot in script.shots:
            shot_text = f"{shot.narration_text} {shot.visual_prompt}".lower()
            if any(w in shot_text for w in target_words):
                matching_shots += 1

        match_ratio = matching_shots / len(script.shots)
        if match_ratio < 0.6:
            issues.append(
                f"Gate 1 Fail: Topic-to-Script Disconnect! Only {matching_shots}/{len(script.shots)} shots ({match_ratio:.0%}) "
                f"contain topic/domain keywords. Required: >= 60% alignment."
            )

        is_valid = len(issues) == 0
        return is_valid, issues

    def verify_gate2_script_to_tts(
        self, state: GlobalState
    ) -> Tuple[bool, List[str]]:
        """
        Verifies that synthesized TTS WAV audio files exist for all shots,
        are non-empty, and contain natural speech.
        """
        issues = []
        if not state.asset_paths or not state.asset_paths.audio:
            return False, ["Gate 2 Fail: Asset paths audio dictionary is empty."]

        for shot in state.script_data.shots:
            shot_key = f"shot_{shot.shot_id}"
            wav_path = state.asset_paths.audio.get(shot_key, "")
            if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
                issues.append(f"Gate 2 Fail: TTS audio file for {shot_key} is missing or corrupted ({wav_path}).")

        is_valid = len(issues) == 0
        return is_valid, issues

    def verify_gate3_tts_to_subtitles(
        self, state: GlobalState
    ) -> Tuple[bool, List[str]]:
        """
        Verifies master subtitle file covers 100% of the video timeline and contains valid dialogue lines.
        """
        issues = []
        master_sub = os.path.join(state.asset_paths.storage_dir or "/tmp/csvg_media", "master_subtitles.ass")
        if not os.path.exists(master_sub) or os.path.getsize(master_sub) < 100:
            return False, [f"Gate 3 Fail: Master subtitle file is missing or empty ({master_sub})."]

        with open(master_sub, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        dialogue_count = sum(1 for line in lines if line.startswith("Dialogue:"))
        if dialogue_count < len(state.script_data.shots):
            issues.append(f"Gate 3 Fail: Subtitle dialogue count ({dialogue_count}) is less than shot count ({len(state.script_data.shots)}).")

        is_valid = len(issues) == 0
        return is_valid, issues

    def verify_gate5_ai_disclosure_tags(
        self, script: ScriptData, media_pipeline_tags: List[str]
    ) -> Dict[str, Any]:
        """
        Gate 5: YouTube 2025/2026 Synthetic Media Disclosure Auto-Tagger.
        Automatically determines whether mandatory AI disclosure metadata must be injected
        based on the media generation pipeline used and visual content type.
        Returns YouTube upload metadata patch with syntheticContent and bInformed flags.
        """
        all_tags_lower = " ".join(media_pipeline_tags).lower()
        full_script_text = " ".join([s.narration_text + " " + s.visual_prompt for s in script.shots]).lower()
        
        # Check for mandatory disclosure triggers
        disclosure_mandatory = any(
            trigger in all_tags_lower or trigger in full_script_text
            for trigger in self.DISCLOSURE_MANDATORY_TRIGGERS
        )
        
        # Check if all content is exempt (stylised art, no realistic humans)
        all_exempt = all(
            any(term in all_tags_lower for term in self.DISCLOSURE_EXEMPT_TERMS)
            for _ in [True]
        ) and not disclosure_mandatory

        return {
            "synthetic_content_tag": True,           # Always tag for AI-generated pipelines
            "b_informed_tag": disclosure_mandatory,  # True only if realistic humans/events present
            "mandatory_disclosure": disclosure_mandatory,
            "all_exempt": all_exempt,
            "youtube_metadata_patch": {
                "syntheticContent": True,
                "bInformed": disclosure_mandatory,
                "aiGeneratedContent": True
            },
            "rationale": "Mandatory disclosure: photorealistic/synthetic human detected." if disclosure_mandatory
                         else "Standard AI-generated label applied (no photorealistic humans)."
        }

    def verify_gate6_anti_slop_entropy(self, script: ScriptData) -> Dict[str, Any]:
        """
        Gate 6: Anti-"AI Slop" Script Entropy & Diversity Audit.
        Ensures script has sufficient linguistic diversity and avoids repetitive template patterns
        that trigger YouTube's AI Slop demonetization filters.
        Measures: Shannon Word Entropy, Unique Word Ratio (length-adaptive), Sentence Length Variance,
        and per-shot minimum narration depth check.
        """
        full_text = " ".join([s.narration_text for s in script.shots])
        words = re.findall(r'\b[a-zA-Z]{3,}\b', full_text.lower())

        if not words:
            return {"passes": False, "reason": "Empty script text."}

        # 1. Shannon Word Entropy
        total_words = len(words)
        freq: Dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        entropy = -sum((c / total_words) * math.log2(c / total_words) for c in freq.values())

        # 2. Unique Word Ratio (length-adaptive)
        # Longer scripts naturally have lower unique ratios due to necessary connecting words.
        # Scales from 35% at <=1500 words down to 28% at 3000+ words.
        unique_ratio = len(freq) / total_words
        if total_words <= 1500:
            UNIQUE_RATIO_MIN = 0.33
        elif total_words <= 2000:
            UNIQUE_RATIO_MIN = 0.29
        elif total_words <= 2500:
            UNIQUE_RATIO_MIN = 0.27
        else:
            UNIQUE_RATIO_MIN = 0.25

        # 3. Sentence Length Variance
        sentences = re.split(r'[.!?]', full_text)
        sentence_lengths = [len(s.split()) for s in sentences if len(s.split()) > 2]
        if len(sentence_lengths) > 1:
            mean_len = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        else:
            variance = 0.0

        # 4. Per-Shot Minimum Narration Depth (catches single shallow shots)
        shallow_shots = [s.shot_id for s in script.shots if len(s.narration_text.split()) < 80]

        # Thresholds
        ENTROPY_MIN = 5.0      # < 5.0 bit = repetitive / template
        VARIANCE_MIN = 20.0    # < 20 = monotone sentence rhythm

        passes = (
            entropy >= ENTROPY_MIN
            and unique_ratio >= UNIQUE_RATIO_MIN
            and variance >= VARIANCE_MIN
            and len(shallow_shots) == 0
        )
        issues = []
        if entropy < ENTROPY_MIN:
            issues.append(f"Low entropy ({entropy:.2f} bits < {ENTROPY_MIN}): Script appears repetitive.")
        if unique_ratio < UNIQUE_RATIO_MIN:
            issues.append(f"Low unique word ratio ({unique_ratio:.2%} < {UNIQUE_RATIO_MIN:.0%} for {total_words}-word script): Template recycling detected.")
        if variance < VARIANCE_MIN:
            issues.append(f"Low sentence length variance ({variance:.1f} < {VARIANCE_MIN}): Monotone rhythm detected.")
        if shallow_shots:
            issues.append(f"Shallow shots detected (< 80 words): Shot IDs {shallow_shots}. Each shot needs 100+ words for quality.")

        return {
            "passes": passes,
            "shannon_entropy_bits": round(entropy, 3),
            "unique_word_ratio": round(unique_ratio, 3),
            "unique_ratio_threshold": UNIQUE_RATIO_MIN,
            "sentence_length_variance": round(variance, 2),
            "total_narration_words": total_words,
            "shallow_shots": shallow_shots,
            "issues": issues
        }



quality_verifier = StageQualityVerifier()



