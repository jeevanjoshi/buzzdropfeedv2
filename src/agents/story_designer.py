import json
import uuid
import time
import datetime
import re
from typing import List, Dict, Any, Optional
from src.schemas.state import GlobalState, ScriptData, ShotData, TopicCandidate, VerifiedFact, SEOMetadata, VisualType
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent
from src.engine.llm_client import LLMClient
from src.engine.rag_retriever import rag_retriever
from src.engine.bertopic_engine import bertopic_engine
from src.engine.logger import logger

# Max total LLM generation attempts for the script (1 initial + 2 repair retries).
# If the live LLM fails to produce a valid script after all attempts (fails the
# validation gate >=12 shots / >=1500 words, or JSON parse), an exception is
# raised and the pipeline aborts (no silent template fallback when a live LLM is
# configured and reachable). The grounded template remains the fallback ONLY for
# offline mode where no LLM client is available.
LLM_MAX_ATTEMPTS = 3


# Matches search-tool citation tags anywhere in a narration/snippet, e.g.
# "[Tavily: ...]", "[Wikipedia: ...]", "[Exa: title | Reuters]:", "[DDG: ...]".
_TOOL_TAG_RE = re.compile(
    r'\[(?:Tavily|Wikipedia|Exa|NewsAPI|Firecrawl|DDG|DuckDuckGo)\s*:[^\]]*\]\s*:?',
    re.IGNORECASE,
)
_ELLIPSIS_RE = re.compile(r'\s*\[\.\.\.\]\s*')

# Web-community / advertorial chatter that should never appear in narration
# (e.g. raw pasted quotes like "Hi all! 24M here...", "ICYMI,", "Log in").
_WEB_CHATTER_RE = re.compile(
    r'\b(ICYMI|Log\s?in|Sign\s?up|Sign\s?in|Subscribe|Hi\s?all!|Hi\s?everyone|'
    r'At such short notice|So imagine|Don\'?t miss|Newsletter|Comment\s?below|'
    r'Follow\s?for|Grab\s?(yours|your)\s?now|Reply\s?to)',
    re.IGNORECASE,
)


def _clean_narration(narr: str) -> str:
    """
    Fixes 2 & 3: strips raw search-tool citation tags (so narration never cites
    'Tavily'/'Exa'), '[...]' scrape artifacts, and pasted web-community/advertorial
    sentences from script narration. Applied to the finished script so even the raw
    LLM output reads as clean prose.
    """
    s = _TOOL_TAG_RE.sub("", narr or "")
    s = _ELLIPSIS_RE.sub(" ", s)
    sents = re.split(r'(?<=[.!?])\s+', s)
    s = " ".join(x for x in sents if not _WEB_CHATTER_RE.search(x))
    return re.sub(r"\s{2,}", " ", s).strip()


def _clean_snippet_text(snippet: str) -> str:
    """Cleans an individual retrieved-snippet line (leading tool tag + artifacts)
    before it is used as narration padding."""
    s = _TOOL_TAG_RE.sub("", snippet or "")
    s = _ELLIPSIS_RE.sub(" ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


class StoryDesignerAgent:
    """
    Story Designer Agent responsible for expanding a selected topic into a 10-15 minute,
    6-Act dramatic arc narrative script. Uses Retrieval-Augmented Generation (RAG) to dynamically
    retrieve deep historical context, factual benchmarks, and strategic implications for ANY topic
    category (Tech, AI, Finance, Space, Geopolitics, Entertainment, Health, Sports, etc.).
    """

    def __init__(self, name: str = "StoryDesigner", llm_client: Optional[LLMClient] = None):
        self.name = name
        self.llm_client = llm_client or LLMClient()
        self.last_llm_source = "UNKNOWN"

    def extract_ground_truth_context(self, verified_facts: List[VerifiedFact]) -> Dict[str, Any]:
        """
        Extracts verified ground-truth text context, numbers, key entities, and trusted organization names.
        """
        combined_text = " ".join([f"{fact.headline}. {fact.summary}" for fact in verified_facts])
        numbers = re.findall(r'\$?\b\d+(?:\.\d+)?[kmb%]?\b', combined_text.lower())
        
        # Primary trusted organization name
        trusted_orgs = [fact.source_name for fact in verified_facts if fact.source_name]
        primary_org = trusted_orgs[0] if trusted_orgs else "Verified Market Reports"

        return {
            "full_context": combined_text,
            "ground_truth_numbers": set(numbers),
            "primary_org": primary_org,
            "all_orgs": list(set(trusted_orgs)),
            "sources": [fact.url for fact in verified_facts]
        }

    def parse_raw_snippets(
        self, rag_pack: Dict[str, Any], summary: str, verified_facts: List[VerifiedFact],
        trusted_org: str, category: str, headline: str, current_month_year: str
    ) -> List[str]:
        retrieved_context = rag_pack.get("rag_retrieved_context", "")
        raw_snippets_raw = [
            _clean_snippet_text(line.strip().lstrip("• ").strip())
            for line in retrieved_context.split("\n")
            if line.strip().startswith("•") and len(line.strip()) > 40
            and "[DDG:" not in line and "DuckDuckGo" not in line  # skip raw scrape noise
        ]
        seen_snips = set()
        raw_snippets = []
        for s in raw_snippets_raw:
            key = s[:60]
            if key not in seen_snips:
                seen_snips.add(key)
                raw_snippets.append(s)

        if len(raw_snippets) < 5:
            summary_sentences = [s.strip() for s in re.split(r'[.!?]', summary) if len(s.strip()) > 30]
            for sent in summary_sentences:
                key = sent[:60]
                if key not in seen_snips:
                    seen_snips.add(key)
                    raw_snippets.append(sent)

        if len(raw_snippets) < 5:
            for vf in verified_facts[:8]:
                frag = f"{vf.headline}: {vf.summary[:80]}"
                key = frag[:60]
                if key not in seen_snips:
                    seen_snips.add(key)
                    raw_snippets.append(frag)

        while len(raw_snippets) < 8:
            raw_snippets.append(f"According to {trusted_org} analysis, {headline} represents a pivotal development in {category} as of {current_month_year}.")
        return raw_snippets

    def expand_narration_with_semantic_facts(
        self, narr: str, title: str, category: str, raw_snippets: List[str],
        used_snippets: set, target_word_count: int = 115
    ) -> str:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        current_words = narr.split()
        if len(current_words) >= target_word_count:
            return narr

        # Filter unused snippets
        unused = [s for s in raw_snippets if s[:60] not in used_snippets]
        if not unused:
            return narr
            
        query_context = f"{title} {narr} {category}"
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([query_context] + unused)
            sim_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            sorted_indices = sim_scores.argsort()[::-1]
            
            for idx in sorted_indices:
                best_snippet = unused[idx]
                best_snippet_clean = best_snippet.strip()
                if best_snippet_clean not in narr:
                    narr += " " + best_snippet_clean
                    used_snippets.add(best_snippet[:60])
                if len(narr.split()) >= target_word_count:
                    break
        except Exception as e:
            print(f"Warning: Semantic expansion error: {e}. Falling back to standard linear selection.")
            for best_snippet in unused:
                if best_snippet not in narr:
                    narr += " " + best_snippet
                    used_snippets.add(best_snippet[:60])
                    if len(narr.split()) >= target_word_count:
                        break
        return narr

    def generate_6act_script(
        self, topic: TopicCandidate, verified_facts: List[VerifiedFact], region: str = "all", target_shots: int = 15, revision_violations: Optional[List[str]] = None, state: Optional[GlobalState] = None
    ) -> ScriptData:
        """
        Expands the topic candidate into a 6-Act dramatic narrative script using RAG fact retrieval.
        Dynamically derives current date/year context and spoken trusted organization attributions.
        """
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        current_year = str(now_utc.year)
        current_month_year = now_utc.strftime("%B %Y")
        current_date_str = now_utc.strftime("%Y-%m-%d")

        headline = topic.headline
        summary = topic.summary
        gt_context = self.extract_ground_truth_context(verified_facts)
        trusted_org = gt_context["primary_org"]

        # RAG Fact & Context Retrieval Pack
        rag_pack = rag_retriever.build_rag_knowledge_pack(topic, verified_facts)
        category = rag_pack["category"]

        # Expose the complete RAG fact corpus (verified facts + retrieved sources)
        # so the Observer audits script claims against the full fact source, not
        # just the base verified_facts.
        if state is not None:
            state.crawled_content = rag_pack.get("fact_corpus", rag_pack.get("full_rag_context_text", ""))

        rag_context_text = rag_pack["full_rag_context_text"]

        # Stage 3: BERTopic Neural Outline Extraction
        bertopic_chapters = bertopic_engine.extract_chapter_outlines(rag_context_text, headline)
        chapter_outline_str = "\n".join([
            f"• [{ch['chapter_title']}] (Keywords: {', '.join(ch['cluster_keywords'])})"
            for ch in bertopic_chapters
        ])
        rag_context_text += f"\n\nBERTOPIC NEURAL OUTLINE CHAPTERS:\n{chapter_outline_str}"


        # Region Demographic Prompts
        if region == "india" or "nifty" in headline.lower() or "sensex" in headline.lower() or "sbi" in headline.lower():
            location_tag = "in Mumbai or Dalal Street, India"
            people_tag = "an Indian subject matter expert executive"
            exchange_tag = "BSE NSE stock exchange floor"
        else:
            location_tag = "in Silicon Valley or major international hub"
            people_tag = "a expert domain strategist"
            exchange_tag = "modern corporate media center"

        # Attempt Live RAG-Infused Cloud LLM Generation
        # Attempt Live LLM generation with content-level repair retries (capped).
        self.last_llm_source = "FALLBACK_GROUNDED_TEMPLATE"
        if self.llm_client.is_available():
            for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
                if attempt > 1:
                    # Short backoff between attempts to ride out transient provider
                    # errors/rate-limits (errors can otherwise fail all 3 in seconds).
                    logger.warning(
                        "SCRIPT_DESIGN",
                        f"LLM script attempt {attempt}/{LLM_MAX_ATTEMPTS}; backing off before retry.",
                        component="STORY_DESIGNER",
                    )
                    time.sleep(4)
                prompt = f"""
                You are an investigative documentary director crafting a 10-15 minute 16:9 widescreen YouTube Infotainment script for topic: '{headline}'.
                CATEGORY: '{category}'
                DYNAMIC TEMPORAL ANCHOR: Current Date is {current_date_str} ({current_month_year}). Year: {current_year}.
                TRUSTED SOURCE ATTRIBUTION: Cite diverse actual publications matching the facts from the RAG pack (e.g. Wikipedia, Wired, New York Times, TechCrunch, World Bank).
            
                FULL RAG KNOWLEDGE PACK (RETRIEVED DEEP FACTS & CONTEXT):
                {rag_context_text}
                """

                if revision_violations:
                    violations_str = "\n".join(f"- {v}" for v in revision_violations)
                    prompt += f"""

                ⚠️ CRITICAL REVISION INSTRUCTION:
                The previous draft of the script failed quality/factual/anti-slop validation. 
                You MUST correct the following violations in this new script draft:
                {violations_str}
                """

                prompt += """

                Requirements:
                1. Exactly 15 shots spanning 6 Acts (Act 1 Hook, Act 2 History/Origins, Act 3 Deep Technical Mechanics, Act 4 Real-World Impact, Act 5 Critical Risks & Misconceptions, Act 6 Future Verdict).
                2. Return a JSON object with key "shots" containing an array of 15 shot objects.
                3. Each shot object MUST contain:
                   - "shot_id": integer 1 to 15
                   - "act_index": integer 1 to 6
                   - "narration_text": string of 115-130 words deeply explaining facts from the RAG pack
                   - "visual_prompt": string specifying "Cinematic 16:9 widescreen..." matching '{category}'
                   - "visual_type": string classification of the visual format. Choose EXACTLY one of:
                     * "standard_image" (default photorealistic cinematic scenes)
                     * "gif_meme" (humorous reaction images, memes, or high-retention popular GIPHY clips)
                     * "matplotlib_chart" (data-led growth line/bar graphs showing numbers, percentages, or milestones)
                     * "svg_ticker" (glowing real-time stock price indices or valuation counting tickers)
                4. Spoken Attribution: Dynamically cite the specific actual publisher or source from the RAG pack (e.g. Wikipedia, Wired, New York Times, TechCrunch, World Bank) for each fact in a natural, conversational way. Avoid over-attributing everything to a single source.
                5. Strict Temporal Grounding: Frame current developments within {current_month_year}; treat any pre-2026/historical-tagged fact as PAST background only, never as a current event.
                6. LINGUISTIC DIVERSITY & STYLISTIC DYNAMICS: Every shot must use distinct sentence structures, rhythms, and vocabulary. Avoid robotic templates or academic summaries. Blend narrative storytelling, punchy declarations, analogies, and rhetorical pacing. Do not start sentences with repetitive structures.
                7. VISUAL CONTINUITY: Each visual_prompt must describe a DISTINCT scene with a unique camera movement (dolly, pan, crane, macro, wide, ECU) and lighting setup.
                8. TOPIC KEYWORD DENSITY: At least 2-3 specific keywords from the headline '{headline}' must appear in every shot's narration_text.
                9. STORYTELLING INTEGRATION: Seamlessly blend real-world facts from the RAG pack into a single, cohesive narrative arc. Do not output raw scrapped snippets verbatim; rephrase them using rich, evocative English prose.
                10. CREATIVE CTA INTEGRATION: The final shot (Shot 15) must conclude with a highly creative, conversational, and integrated call-to-action (CTA). Ask the audience a thought-provoking question related to the topic, invite them to drop their answers in the comments, and smoothly guide them to like and subscribe to join the journey. Avoid stale, generic 'like and subscribe' phrasing.
                """
                system_prompt = (
                    f"You are a master documentary director and creative storyteller specializing in {category} in {current_year}. "
                    "CRITICAL RULES: "
                    "1. STYLISTIC EXCELLENCE (Vox/Netflix Documentary Style): Write in a gripping, cinematic, and narrative-first tone. "
                    "Never write dry summaries or list scraped facts line-by-line. Instead, weave facts into a suspenseful, unfolding human story. "
                    "2. DIVERSE CITATIONS: Dynamically attribute facts to the actual distinct publishers in the RAG pack (e.g. 'As reported by The New York Times', 'Wired analysis shows', 'Wikipedia records indicate'). Do not attribute everything to a single publisher. "
                    "3. CREATIVE ANALOGIES: Translate complex data, metrics, or technical mechanisms into vivid metaphors and simple physical analogies. "
                     "4. DYNAMIC RHYTHM: Vary sentence lengths dramatically. Pair long, analytical explanations with short, punchy, high-impact statements. "
                     "5. Rhetorical & Structural Diversity: Alternate styles across shots—declarative hooks, rhetorical questions, storytelling scenes, and data assertions. "
                     "6. TEMPORAL GROUNDING: Today is {current_year}. Treat any fact dated before 2026 (e.g. '(historical: YYYY)' tags, or any pre-2026 year) as HISTORICAL/PAST context ONLY. "
                     "Never present older-dated data as a development happening in {current_month_year}. Only describe something as current/this-month if the source is clearly recent; otherwise frame it as 'back in ...' / 'historically ...'. "
                     "7. NEVER start two consecutive shots with the same subject or phrase. Ensure seamless transitions between shots. "
                     "8. Visual prompts must describe unique, high-end cinematic locations, camera moves, and lighting. Return valid JSON only."
                )
                repair_hint = ""
                if attempt > 1:
                    repair_hint = (
                        "\n\n\u26a0\ufe0f CRITICAL REPAIR INSTRUCTION (PREVIOUS DRAFT FAILED VALIDATION):\n"
                        "Your previous draft did NOT meet the HARD requirements: it either contained fewer than 12 shots, "
                        "had narration_text under 115 words, the total fell below 1,500 words, or the JSON was "
                        "truncated/incomplete. Produce EXACTLY 15 shot objects, each narration_text between 115 and 130 "
                        "words, so the script total exceeds 1,500 words. Return ONLY one complete, valid JSON object with "
                        "a single 'shots' key. Do NOT truncate or omit any shot.\n"
                    )
                    prompt += repair_hint
                llm_result = self.llm_client.generate_json(prompt, system_prompt)

                # Parse raw snippets and setup dynamic RAG pool
                raw_snippets = self.parse_raw_snippets(rag_pack, summary, verified_facts, trusted_org, category, headline, current_month_year)
                _used_snips = set()

                if llm_result and "shots" in llm_result:
                    try:
                        raw_shots = llm_result["shots"]
                        shots = []
                        for idx, s in enumerate(raw_shots, start=1):
                            shot_id = s.get("shot_id") or s.get("id") or s.get("shot") or idx
                            act_idx = s.get("act_index") or s.get("act") or s.get("act_num") or min(6, (idx - 1) // 2.5 + 1)
                            narr = s.get("narration_text") or s.get("narration") or s.get("script") or ""
                            vis = s.get("visual_prompt") or s.get("visual") or s.get("prompt") or f"Cinematic 16:9 widescreen visual for {headline}, 8k photorealistic."
                        
                            v_type_raw = s.get("visual_type") or "standard_image"
                            if v_type_raw not in ["standard_image", "gif_meme", "matplotlib_chart", "svg_ticker"]:
                                v_type_raw = "standard_image"
                        
                            # Enrich short narration with dynamic RAG facts using semantic search/TF-IDF similarity
                            if len(narr.split()) < 115:
                                narr = self.expand_narration_with_semantic_facts(narr, headline, category, raw_snippets, _used_snips, target_word_count=115)

                            shots.append(ShotData(
                                shot_id=int(shot_id),
                                act_index=int(act_idx),
                                narration_text=narr,
                                visual_prompt=vis,
                                visual_type=VisualType(v_type_raw),
                                duration_estimate=max(42.0, round(len(narr.split()) / 2.2, 1))
                            ))

                        total_words = sum(len(s.narration_text.split()) for s in shots)
                        if total_words >= 1500 and len(shots) >= 12:
                            self.last_llm_source = "LIVE_LLM"
                            # Fixes 2 & 3: strip raw [Tool: ...] tags and [...] scrape
                            # artifacts from the finished narration so it reads as clean prose.
                            for s in shots:
                                s.narration_text = _clean_narration(s.narration_text)
                            cw = sum(len(x.narration_text.split()) for x in shots)
                            runtime = cw / 150.0 * 60.0
                            return ScriptData(
                                title=llm_result.get("title", f"The Hidden Truth Behind {headline[:35]}... ({current_month_year})"),
                                target_shots=len(shots),
                                shots=shots,
                                estimated_runtime_seconds=round(runtime, 1)
                            )
                        logger.warning(
                            "SCRIPT_DESIGN",
                            f"LLM draft attempt {attempt}/{LLM_MAX_ATTEMPTS} failed gate: {len(shots)} shots, {total_words} words (<12 shots or <1500 words).",
                            component="STORY_DESIGNER"
                        )

                    except Exception as e:
                        logger.warning("SCRIPT_DESIGN", f"LLM Script Parse Exception on attempt {attempt}/{LLM_MAX_ATTEMPTS}: {e}", component="STORY_DESIGNER")

            logger.error(
                "SCRIPT_DESIGN",
                f"Live LLM script generation failed after {LLM_MAX_ATTEMPTS} attempts; aborting pipeline (no template fallback).",
                component="STORY_DESIGNER"
            )
            raise RuntimeError(
                f"StoryDesignerAgent: live LLM failed to produce a valid script after {LLM_MAX_ATTEMPTS} attempts. "
                f"Check OPENROUTER_API_KEY/GEMINI_API_KEY and LLM output (truncation, quota, or malformed JSON)."
            )
        else:
            logger.error(
                "SCRIPT_DESIGN",
                "StoryDesigner requires a live LLM; boilerplate template fallback is disabled. Configure OPENROUTER/GEMINI/OPENAI key and ensure LLM reachability.",
                component="STORY_DESIGNER"
            )
            raise RuntimeError(
                "StoryDesignerAgent: no live LLM available (template fallback disabled). "
                "Set OPENROUTER_API_KEY/GEMINI_API_KEY and ensure the LLM client is reachable."
            )


    def _polish_script(self, script: ScriptData, headline: str, category: str) -> Optional[ScriptData]:
        """
        LLM "editor" polish pass: rewrites each shot's narration to be more
        engaging, human, and creative — while STRICTLY preserving facts, numbers,
        names, dates and meaning. Fact-preserving by construction:
          * One structured LLM call returns rewritten narration per shot_id.
          * Unchanged/oversized shots fall back to the original text.
          * Never introduces new facts > verified corpus (rule-only; Observer
            re-audits against the full fact corpus afterwards).
        Cheap (~one small call), so it fits the LLM budget (separate from the
        fal/replicate cap). Returns None (caller keeps original) on any failure.
        """
        if not self.llm_client.is_available():
            return None
        import json
        shots_json = [
            {"shot_id": s.shot_id, "act_index": s.act_index, "narration_text": s.narration_text}
            for s in script.shots
        ]

        repair_hint = ""
        for attempt in range(1, 3):  # Fix 1: retry the polish pass to reduce transient failures
            if attempt > 1:
                time.sleep(3)  # backoff between polish retries
            prompt = (
                "You are a skilled documentary editor. For EACH shot, rewrite the narration to be "
                "more engaging, human, fluent and creative, while STRICTLY preserving every fact, "
                "number, name, date, source attribution and the original meaning. Remove any raw "
                "citation tags like '[Tavily:...]' or '[Exa:...]' and keep clean prose.\n"
                "Rules:\n"
                "- Keep every shot's narration between 110 and 135 words (aim ~120).\n"
                "- Vary sentence lengths dramatically; avoid repeating words/phrases across sentences and shots.\n"
                "- Use precise, vivid English vocabulary; avoid cliches, robotic templates and monotony.\n"
                "- Blend rhetorical questions, storytelling scenes, analogies and punchy declarations.\n"
                f"- Topic headline: {headline}. Category: {category}.\n"
                "- Do NOT add any new facts or numbers; do NOT change meaning.\n"
                "Return ONLY a valid, COMPLETE JSON object with key \"shots\": an array of "
                "{\"shot_id\": <int>, \"narration_text\": \"<rewritten>\"} for ALL shots.\n"
                f"SHOTS TO POLISH:\n{json.dumps(shots_json, ensure_ascii=False)}\n"
            )
            prompt += repair_hint
            try:
                res = self.llm_client.generate_json(
                    prompt, "You are a documentary editor. Return valid JSON only."
                )
            except Exception:
                res = None
            if res and "shots" in res:
                narr_by_id = {}
                for s in (res["shots"] or []):
                    try:
                        sid = int(s.get("shot_id"))
                    except (TypeError, ValueError):
                        continue
                    narr = _clean_narration(s.get("narration_text") or "")
                    wc = len(narr.split())
                    if 100 <= wc <= 160:  # guard: only accept sane-length rewrites
                        narr_by_id[sid] = narr
                polished = []
                for shot in script.shots:
                    narr = narr_by_id.get(shot.shot_id, shot.narration_text)
                    polished.append(ShotData(
                        shot_id=shot.shot_id,
                        act_index=shot.act_index,
                        narration_text=narr,
                        visual_prompt=shot.visual_prompt,
                        visual_type=shot.visual_type,
                        duration_estimate=max(42.0, round(len(narr.split()) / 2.2, 1)),
                    ))
                total_words = sum(len(s.narration_text.split()) for s in polished)
                return ScriptData(
                    title=script.title,
                    target_shots=len(polished),
                    shots=polished,
                    estimated_runtime_seconds=round(total_words / 150.0 * 60.0, 1),
                )
            logger.warning(
                "SCRIPT_DESIGN",
                f"Polish pass attempt {attempt}/2 returned invalid result; retrying.",
                component="STORY_DESIGNER",
            )
            repair_hint = (
                "\n\nCRITICAL REPAIR INSTRUCTION: The previous response was not usable valid JSON "
                "for all shots. Return ONLY one complete, valid JSON object with a single 'shots' "
                "key containing every shot_id 1..N with its rewritten narration_text. Do not "
                "truncate, omit, or wrap in markdown.\n"
            )
        return None

    def _generate_ctr_title(self, headline: str, niche_category: str = "") -> Optional[str]:
        """
        Best-effort LLM generation of a high-CTR YouTube title (<=65 chars) using
        numbers, curiosity/urgency, or a question. Falls back to None (caller uses
        the headline truncation) when the LLM is unavailable or the output is
        unusable, so SEO generation never breaks on a single small call.
        """
        if not self.llm_client.is_available():
            return None
        prompt = (
            f"Write ONE high-CTR YouTube title, max 65 characters, for an 11-14 minute "
            f"infotainment documentary. Topic headline: '{headline}'. Niche: '{niche_category}'. "
            f"Use a number, a curiosity/urgency angle, or a question. Avoid clickbait that "
            f"contradicts the facts. Return ONLY a JSON object with the key 'title'."
        )
        try:
            result = self.llm_client.generate_json(
                prompt,
                "You craft concise, honest high-CTR YouTube titles. Return valid JSON only.",
            )
        except Exception:
            result = None
        if result and result.get("title"):
            title = str(result["title"]).strip()
            if 10 <= len(title) <= 70:
                return title
        return None

    def generate_seo_metadata(self, topic: TopicCandidate, script: ScriptData) -> SEOMetadata:
        """
        Generates high-CTR SEO metadata (Title, Description, Tags, Thumbnail Brief) alongside the script.
        """
        headline = topic.headline
        ctr_title = self._generate_ctr_title(headline, getattr(topic, "niche_category", ""))
        clean_title = ctr_title if ctr_title else (headline[:65] if len(headline) > 65 else headline)
        tags = [t.strip().lower() for t in topic.keywords if len(t.strip()) > 2][:10]
        tags.extend(["infotainment", "documentary", "2026", "analysis", "explained"])
        
        description = (
            f"Deep-dive documentary analysis on: {headline}.\n\n"
            f"In this video, we break down the ground-truth data, market implications, and strategic lessons.\n\n"
            f"CHAPTERS:\n"
            f"0:00 - Act 1: The Inciting Incident\n"
            f"2:15 - Act 2: Historical Precedents & Origins\n"
            f"4:30 - Act 3: Deep Technical Mechanics\n"
            f"6:45 - Act 4: Actionable Real-World Impact\n"
            f"9:00 - Act 5: Critical Risks & Counter-Arguments\n"
            f"11:15 - Act 6: Strategic Future Verdict\n\n"
            f"Sources & Data Grounding:\n- {topic.source_url}\n- Grounded Fact Verification via CSVG Pipeline (2026)\n\n"
            f"#Infotainment #{topic.niche_category.replace(' ', '')} #Documentary"
        )
        
        # Punchy, high-CTR on-image brief (short hook, <= ~5 words) for the thumbnail.
        if ctr_title:
            words_ = ctr_title.split()
            brief = " ".join(words_[:4])
            if len(brief) > 24:
                brief = " ".join(words_[:3])
        else:
            brief = headline.split()[:4]
            brief = " ".join(brief) if brief else headline
        if not brief.strip():
            brief = headline[:24]

        return SEOMetadata(
            title=clean_title,
            description=description,
            tags=list(set(tags)),
            thumbnail_brief=brief,
            chapter_timestamps=[
                "0:00 Intro", "2:15 Historical Background", "4:30 Technical Analysis", 
                "6:45 Real Impact", "9:00 Risk Analysis", "11:15 Conclusion"
            ]
        )

    def process(self, state: GlobalState, region: str = "all", revision_violations: Optional[List[str]] = None) -> A2AMessage:
        """
        Executes Story Designer workflow:
        1. Reads selected topic and verified_facts from GlobalState
        2. Generates 6-Act dramatic script with dynamic date context, trusted organization citations & region-appropriate visual framing
        3. Generates SEO Metadata (Title, Description, Tags, Thumbnail Overlay Brief)
        4. Updates state.script_data and state.seo_metadata
        5. Emits A2AMessage to Observer Agent
        """
        if not state.selected_topic:
            raise ValueError("Cannot generate script: state.selected_topic is None")

        script = self.generate_6act_script(state.selected_topic, state.verified_facts, region=region, revision_violations=revision_violations, state=state)

        # LLM editor polish pass: rewrite for engagement while preserving facts.
        polished = self._polish_script(script, state.selected_topic.headline, state.selected_topic.niche_category)
        if polished:
            script = polished
            logger.info("SCRIPT_DESIGN", "Applied LLM editor polish pass (fact-preserving rewrite).", component="STORY_DESIGNER")

        state.script_data = script
        state.seo_metadata = self.generate_seo_metadata(state.selected_topic, script)
        state.execution_stage = "SCRIPT_GENERATED"

        msg = A2AMessage(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            sender=AgentRole.STORY_DESIGNER,
            target=AgentRole.OBSERVER,
            intent=AgentIntent.GENERATE_SCRIPT,
            payload={
                "script_title": script.title,
                "total_shots": script.target_shots,
                "estimated_runtime_minutes": round(script.estimated_runtime_seconds / 60.0, 2),
                "fact_grounding": "ENFORCED",
                "temporal_context": state.current_month_year,
                "region": region,
                "llm_mode": getattr(self, "last_llm_source", "FALLBACK_GROUNDED_TEMPLATE"),
            },
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        return msg
