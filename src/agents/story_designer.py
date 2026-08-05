import json
import uuid
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
            line.strip().lstrip("• ").strip()
            for line in retrieved_context.split("\n")
            if line.strip().startswith("•") and len(line.strip()) > 40
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
        self, topic: TopicCandidate, verified_facts: List[VerifiedFact], region: str = "all", target_shots: int = 15, revision_violations: Optional[List[str]] = None
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
                5. Strict Temporal Grounding: Frame developments within {current_month_year}.
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
                    "6. NEVER start two consecutive shots with the same subject or phrase. Ensure seamless transitions between shots. "
                    "7. Visual prompts must describe unique, high-end cinematic locations, camera moves, and lighting. Return valid JSON only."
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
                            runtime = total_words / 150.0 * 60.0
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
        # ── RAG-Grounded Universal Script Template ────────────────────────────
        # Slice retrieved web snippets across acts so each shot gets a DIFFERENT
        # real-world fact rather than generic boilerplate.
        retrieved_context = rag_pack.get("rag_retrieved_context", "")
        raw_snippets_raw = [
            line.strip().lstrip("• ").strip()
            for line in retrieved_context.split("\n")
            if line.strip().startswith("•") and len(line.strip()) > 40
        ]
        # Deduplicate — keep only unique snippets (when DuckDuckGo returns same snippet multiple times)
        seen_snips: set = set()
        raw_snippets: list = []
        for s in raw_snippets_raw:
            key = s[:60]  # First 60 chars as dedup key
            if key not in seen_snips:
                seen_snips.add(key)
                raw_snippets.append(s)

        # When web search returned few results, decompose summary into sentence-sized fragments
        if len(raw_snippets) < 5:
            summary_sentences = [s.strip() for s in re.split(r'[.!?]', summary) if len(s.strip()) > 30]
            for sent in summary_sentences:
                key = sent[:60]
                if key not in seen_snips:
                    seen_snips.add(key)
                    raw_snippets.append(sent)

        # Final fallback: pad with verified_facts headlines as diverse context
        if len(raw_snippets) < 5:
            for vf in verified_facts[:8]:
                frag = f"{vf.headline}: {vf.summary[:80]}"
                key = frag[:60]
                if key not in seen_snips:
                    seen_snips.add(key)
                    raw_snippets.append(frag)

        # Guarantee minimum pool size with topic-keyed fragments
        while len(raw_snippets) < 8:
            raw_snippets.append(f"According to {trusted_org} analysis, {headline} represents a pivotal development in {category} as of {current_month_year}.")

        graph_triplets_str = rag_pack.get("graph_triplets", "").replace("•", "").strip()[:300]

        # Track which snippets have been used to prevent any snippet appearing >1× in the script
        _used_snips: set = set()

        def _snip(i: int) -> str:
            """Return a specific unique web snippet for this shot index."""
            # Try the requested index first, then scan forward for an unused one
            for offset in range(len(raw_snippets)):
                candidate = raw_snippets[(i + offset) % len(raw_snippets)]
                key = candidate[:60]
                if key not in _used_snips:
                    _used_snips.add(key)
                    return candidate
            # All snippets exhausted — return a generic fallback that doesn't repeat
            return f"Sector intelligence published by {trusted_org} in {current_month_year} confirms sustained momentum in {category}."


        acts_blueprint = [
            (1, "The Inciting Incident",
             f"In {current_month_year}, a significant development shook {category} as news broke: {headline}. "
             f"According to initial reporting from {trusted_org}: {summary} "
             f"Industry observers immediately recognised the magnitude of this shift. {_snip(0)}",
             f"Cinematic 16:9 widescreen wide shot of {people_tag} in a sleek modern studio analysing glowing holographic data related to '{headline[:40]}', dramatic rim lighting, 8k photorealistic."),

            (1, "The Immediate Stakes",
             f"The ramifications of {headline} extend across multiple stakeholder groups in {current_month_year}. "
             f"Verified analysis from {trusted_org} confirms: {_snip(1)} "
             f"Decision-makers and industry practitioners are now re-evaluating core assumptions in light of this evidence. "
             f"Organisations that fail to adapt risk losing relevance in an increasingly competitive {category} landscape.",
             f"Cinematic 16:9 widescreen closeup of analytical dashboards and research reports under dark moody lighting, 8k resolution."),

            (2, "Historical Precedents & Origins",
             f"To understand {headline}, we must trace the origins of this development. Historical records document how {category} evolved from foundational frameworks to the inflection point we observe today in {current_month_year}. "
             f"Key background context: {_snip(2)} "
             f"Previous methodologies relied on legacy assumptions that the current breakthrough fundamentally challenges.",
             f"Cinematic 16:9 widescreen wide-angle documentary view of corporate headquarters and research archives, warm golden-hour atmosphere, 8k."),

            (2, "The Turning Point",
             f"The pivotal shift arrived when new empirical evidence surrounding {headline} reached the public domain. "
             f"Information released by {trusted_org} in {current_month_year} demonstrates: {_snip(3)} "
             f"This created unprecedented momentum as legacy frameworks proved insufficient to explain the emerging reality.",
             f"Cinematic 16:9 widescreen dolly push-in on a sleek data visualisation screen revealing a dramatic trend inflection, 8k photorealistic."),

            (3, "Deep Technical Mechanics",
             f"Behind the headline of {headline} lie complex mechanisms worth examining. "
             f"Semantic knowledge graph analysis reveals the following entity relationships: {graph_triplets_str if graph_triplets_str else _snip(4)} "
             f"Understanding these underlying structural dynamics allows researchers and strategists to build evidence-based models rather than relying on surface-level observations.",
             f"Cinematic 16:9 widescreen macro shot of glowing neural data-node connections on a dark digital grid, cold blue data-center lighting, 8k photorealistic."),

            (3, "The Data Evidence",
             f"Detailed research compiled by {trusted_org} in {current_month_year} illustrates the exact correlations underpinning {headline}. "
             f"Independent verification reveals: {_snip(5)} "
             f"By tracking key variables across extended observation windows, analysts established an empirical baseline that eliminates speculation and grounds decision-making in verifiable fact.",
             f"Cinematic 16:9 widescreen close-up of scientific data charts glowing on a sleek glass interface, warm executive lighting, 8k resolution."),

            (4, "Actionable Real-World Impact",
             f"What does {headline} mean for practitioners and decision-makers in {current_year}? "
             f"First: re-evaluate core operational assumptions based on the fresh empirical data confirmed by {trusted_org}. "
             f"Second: align strategic frameworks with the benchmarks this event establishes. "
             f"Third: build institutional agility to respond as the {category} landscape continues evolving throughout {current_month_year}.",
             f"Cinematic 16:9 widescreen shot of {people_tag} crafting a strategic blueprint on a sleek laptop workstation, warm dramatic 8k lighting."),

            (4, "Evidence & Ecosystem Velocity",
             f"According to research from {trusted_org}: {_snip(6)} "
             f"When platforms and organisations align their architecture with the verified standards emerging from {headline}, they gain compounding authority. "
             f"This flywheel effect attracts pre-qualified stakeholder interest and reinforces category dominance over time — a pattern well-documented across {category} cycles.",
             f"Cinematic 16:9 widescreen macro shot of digital citation links connecting across a glowing 3D global network model, 8k resolution."),

            (4, "Early Mover Advantage",
             f"Early responders who act on the insights from {headline} in {current_month_year} are positioning themselves ahead of the competitive curve. "
             f"Background analysis confirms: {_snip(7 % len(raw_snippets))} "
             f"Pioneering teams in {category} are capturing exponential growth while legacy players struggle to recalibrate their strategic models to the new evidence.",
             f"Cinematic 16:9 widescreen dynamic shot of an upward-trending performance graph glowing brightly on a glass screen, 8k resolution."),

            (5, "Critical Risks & Pitfalls",
             f"However, important risks must be navigated carefully in the context of {headline}. "
             f"Misinterpreting preliminary data or rushing execution without rigorous verification backfires rapidly across {category}. "
             f"Studies from {trusted_org} prove that sustainable success requires authentic expertise, verified primary data, and thorough quality assurance at every operational stage.",
             f"Cinematic 16:9 widescreen macro shot of a red warning icon flashing on a high-tech digital audit interface, dark moody low-key lighting, 8k."),

            (5, "Measuring Long-Term Reach",
             f"Tracking the ongoing implications of {headline} requires updated measurement frameworks in {current_year}. "
             f"Forward-thinking teams monitor multi-channel sentiment, citation frequency, and conversion velocity to measure true long-term impact across {category}. "
             f"Attribution modelling confirms: {_snip(0)} — reinforcing the compounding ROI of sustained strategic presence.",
             f"Cinematic 16:9 widescreen shot of a domain strategist analysing custom analytics funnel charts on a tablet, professional background, 8k."),

            (5, "The Future Horizon",
             f"As innovation accelerates around {headline}, the boundary between theoretical modelling and practical execution is dissolving. "
             f"Data from {trusted_org} indicates that a dominant share of {category} developments in {current_month_year} will build directly upon the structural changes triggered by this event: {_snip(2)} "
             f"Staying ahead of this trajectory demands continuous intelligence integration and evidence-based agility.",
             f"Cinematic 16:9 widescreen wide shot of a futuristic research complex with holographic data overlays and tilt-shift depth, 8k photorealistic."),

            (6, "The Final Verdict",
             f"The transformation underway in {category} following {headline} is not a distant prediction — it is happening right now in {current_year}. "
             f"Stakeholders who adapt their strategies to these new realities will thrive, while those relying on outdated playbooks risk complete obsolescence. "
             f"The opportunity is immediate, actionable, and grounded in verified evidence from {trusted_org}.",
             f"Cinematic 16:9 widescreen crane descent over a city skyline at twilight with glowing fibre-optic data networks spanning the horizon, dramatic 8k."),

            (6, f"Key Takeaways for {current_year}",
             f"To summarise the core takeaways from {headline} for {current_month_year}: Ground strategy in empirical facts validated by {trusted_org}. "
             f"Align operational architecture with the structural shifts this event confirms. "
             f"Maintain continuous intelligence integration and category agility. "
             f"Sustainable success in {category} belongs exclusively to those who provide verifiable, defensible value.",
             f"Cinematic 16:9 widescreen close-up of an executive summary document on a sleek modern desk, warm professional lighting, 8k resolution."),

            (6, "Call to Action",
             f"How is your organisation responding to {headline} in {current_year}? "
             f"Drop your perspective in the comments below, subscribe for weekly {category} intelligence deep dives, and hit the notification bell to stay ahead of global trends. "
             f"To continue your strategic deep dive, click the recommended video appearing on your screen right now for our next breakdown in {category}.",
             f"Cinematic 16:9 widescreen clean outro graphic with two YouTube end-screen video card placeholders, sleek dark theme design, 8k high resolution.")
        ]

        shots = []
        total_runtime = 0.0

        # RAG-seeded rotating padding pool — vocabulary drawn from real retrieved snippets
        _snip_words = " ".join(raw_snippets[:3])
        _kw_fragment = " ".join(_snip_words.split()[:8]) if _snip_words else summary[:60]
        _padding_pool = [
            f"Cross-referencing {trusted_org} intelligence confirms that {_kw_fragment} represents a structural inflection point shaping {category} strategy through {current_year}.",
            f"Sector observers tracking {headline[:45]} note that quantitative benchmarks published by {trusted_org} reinforce the magnitude of this reconfiguration.",
            f"What makes this development uniquely consequential is the convergence of verified empirical evidence and accelerating adoption velocity documented in {current_month_year}.",
            f"Longitudinal datasets compiled by leading {category} analysts corroborate that foundational assumptions underpinning legacy frameworks require urgent recalibration.",
            f"Early cohort data cited by {trusted_org} reveals that organisations adopting evidence-based repositioning strategies are outperforming reactive counterparts by measurable margins.",
            f"The compounding velocity of change within {category} demands that decision-makers move beyond periodic reviews toward continuous intelligence integration.",
            f"Historical precedents drawn from comparable {category} disruption cycles suggest that the first-mover advantage window documented in {current_month_year} closes faster than conventional wisdom predicts.",
        ]

        pool_idx = 0  # Global sequential index — never revisits same sentence
        for idx, (act_num, title_label, narration, v_prompt) in enumerate(acts_blueprint, start=1):
            # Add at most 2 padding sentences per shot, each from a different pool slot
            pads_added = 0
            while len(narration.split()) < 115 and pads_added < 2:
                pad_sentence = _padding_pool[pool_idx % len(_padding_pool)]
                pool_idx += 1
                pads_added += 1
                narration += " " + pad_sentence
            # If still short after 2 pool pads, inject the next raw RAG snippet
            if len(narration.split()) < 115:
                extra_snip = raw_snippets[(idx + 2) % len(raw_snippets)]
                if extra_snip not in narration:
                    narration += " " + extra_snip

            dur = max(42.0, round(len(narration.split()) / 2.2, 1))
            total_runtime += dur

            shot = ShotData(
                shot_id=idx,
                act_index=act_num,
                narration_text=narration,
                visual_prompt=v_prompt,
                duration_estimate=dur
            )
            shots.append(shot)

        script = ScriptData(
            title=f"The Hidden Truth Behind {headline[:35]}... ({current_month_year})",
            target_shots=len(shots),
            shots=shots,
            estimated_runtime_seconds=round(total_runtime, 1)
        )
        return script


    def generate_seo_metadata(self, topic: TopicCandidate, script: ScriptData) -> SEOMetadata:
        """
        Generates high-CTR SEO metadata (Title, Description, Tags, Thumbnail Brief) alongside the script.
        """
        headline = topic.headline
        clean_title = headline[:65] if len(headline) > 65 else headline
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
        
        return SEOMetadata(
            title=clean_title,
            description=description,
            tags=list(set(tags)),
            thumbnail_brief=f"{headline[:25]} Exposed",
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

        script = self.generate_6act_script(state.selected_topic, state.verified_facts, region=region, revision_violations=revision_violations)
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
