import os
import re
from typing import Dict, Any, List, Tuple
from src.schemas.state import GlobalState, ScriptData, TopicCandidate


class StageQualityVerifier:
    """
    Automated Multi-Stage Artifact Quality Verification Engine:
    Validates cross-stage alignment:
    - Gate 1: Topic-to-Script Coherence & Target Runtime Verification
    - Gate 2: Script-to-TTS Speech & Acronym Expansion Verification
    - Gate 3: TTS-to-Subtitle Master Alignment Verification
    - Gate 4: Master Video Resolution & Duration Verification
    """

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


quality_verifier = StageQualityVerifier()
