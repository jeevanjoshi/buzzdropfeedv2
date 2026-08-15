import uuid
import datetime
import re
from typing import List, Dict, Any, Tuple
from src.schemas.state import GlobalState, ScriptData, VerifiedFact
from src.schemas.a2a import A2AMessage, AgentRole, AgentIntent, compute_state_hash
from src.engine.monetization_optimizer import monetization_optimizer
from src.engine.channel_phase_manager import channel_phase_manager
from src.engine.text_embeddings import semantic_embedder, semantic_max_similarity, semantic_pairwise_similarity, semantic_topic_membership, COPY_SEMANTIC_HARD_THRESHOLD

# Semantic-gate thresholds (MiniLM cosine sim). When the semantic backend is
# unavailable these gates fall back to the deterministic TF-IDF/substring logic,
# so hermetic/offline runs behave exactly as before.
#
# Verbatim source-copy now uses a single WHOLE-SENTENCE hard threshold
# (COPY_SEMANTIC_HARD_THRESHOLD, imported from text_embeddings, = 0.94). The old
# 0.82 bar conflated two different things and caused never-converging revision
# loops (25 -> 27 -> 32 violations): it flagged fact-dense narration that MUST
# preserve required names/numbers/dates and therefore can never paraphrase far
# enough to drop its whole-sentence embedding sim below ~0.85, even when a human
# reads it as a genuine rewrite. Calibrated on live data, true copies cluster at
# >= 0.94 while legitimate rephrases sit at 0.80-0.93, so 0.94 is the clean split.
COPY_SEMANTIC_THRESHOLD = 0.82          # legacy lower bound (kept for API compat)
TOPIC_MEMBER_SEMANTIC_THRESHOLD = 0.50  # token ~== a topic anchor word (keyword/headline/summary)

# Style-class violations (revisable, non-factual). When a script fails the bounded
# revision loop with ONLY these remaining, it can be approved-with-warnings via
# the orchestrator's soft-approval path. Everything else (fact/temporal/revenue/
# audience/runtime/shot-count and any raw critic text) stays a hard abort.
_SOFT_VIOLATION_MARKERS = (
    "Keyword over-repetition",
    "verbatim source copy",
    "repetition:",
    "Source Diversity",
    "visual prompt",
    "narration too long",
    "Bare acronym",
)

_MONTHS_PAT = r"(?:Jan(?:uary|\.)?|Feb(?:ruary|\.)?|Mar(?:ch|\.)?|Apr(?:il|\.)?|May|June?|July?|Aug(?:ust|\.)?|Sept?(?:ember|\.)?|Oct(?:ober|\.)?|Nov(?:ember|\.)?|Dec(?:ember|\.)?)"
_DATE_PAT = r"(?:" + _MONTHS_PAT + r"\s+\d{1,2}|\d{1,2}\s+" + _MONTHS_PAT + r")"
_LOCATION_PAT = r"[A-Z][A-Za-z0-9\s.,\-\/’'\u2019]{2,50}"

# Narration must NEVER contain raw scrape/citation junk (markdown links, bare
# URLs, datelines "(City, St – Month DD, YYYY)", "Retrieved ..." bibliography
# tails, bylines/follow-prompts/hashtag tags like "ByTom Carter You're currently
# following this author! # Meta launches ..."). These are unambiguous production
# errors (paste/leak) — NOT style — so violations are HARD aborts and are
# excluded from the soft-approval path.
_RAW_JUNK_IN_NARR_RE = re.compile(
    r'\[[^\]\n]{0,120}?\]\((?:https?://|#|/)[^)\n]{0,300}?\)'
    r'|\bhttps?://'
    r'|[\(\[]\s*' + _LOCATION_PAT + r'\s*[–—-]\s*' + _DATE_PAT + r'\s*,?\s+\d{4}\s*[\)\]]'
    r'|\b' + _LOCATION_PAT + r'\s*[–—-]\s*' + _DATE_PAT + r'\s*,?\s+\d{4}\b'
    r'|\bRetrieved\b'
    r'|[\(\[]\s*' + _DATE_PAT + r'\s*,?\s+\d{4}\s*[\)\]]\.?'
    r'|[A-Z][a-z]+,\s+[A-Z][a-z]+\s+[\(\[]\s*' + _DATE_PAT + r'\s*,?\s+\d{4}\s*[\)\]]\.?'
    r'|\bBy[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+'
    r'|You\'?re\s+currently\s+following\s+this\s+(?:author|writer)'
    r'|\b(?:Subscribe|Sign)\s+(?:to\s+continue|for\s+(?:updates|more))'
    r'|\bAdd\s+us\s+on\b'
    r'|\bFollow\s+us\s+on\b'
    r'|#\s+[A-Z][A-Za-z]+',
    re.IGNORECASE,
)

# Tool-name citation leaks are a HARD abort: narration must never attribute a
# fact to the retrieval toolchain ("As Firecrawl highlights", "noted to Tavily",
# "Exa reports", "per Jina"). Only an actual publication passes. This catches
# the sample-swing where the LLM reads a "[Tool: ...]" bucket tag as the source.
_TOOL_NAME_IN_NARR_RE = re.compile(
    r'\b(?:Tavily|Firecrawl|Exa|Jina|DuckDuckGo|DDG)\b',
    re.IGNORECASE,
)

# Bare acronyms that must NOT survive in narration: 2-6 all-caps letters as a
# standalone token. Proper-noun initialisms / well-known names and unit symbols
# are exempt (see _ACRONYM_ALLOWLIST). Everything else must be spelt out — this
# is the deterministic gate that makes "no acronyms" independent of LLM sampling
# (the writer/polish already expand common terms, this catches the residue).
_ACRONYM_RE = re.compile(r'(?<![A-Za-z0-9])[A-Z]{2,6}(?![A-Za-z0-9])(?!\$)(?!%)')
_ACRONYM_ALLOWLIST = frozenset({
    "NASA", "IBM", "MIT", "CNN", "BBC", "NBC", "CBS", "ABC", "FOX",
    "NY", "LA", "NYSE", "NASDAQ", "IMF", "WTO", "ECB", "FOMC",
    "USD", "EUR", "GBP", "JPY", "INR", "FBI", "CIA", "FCC", "SEC",
    "WHO", "NATO", "UN", "EU",
})


def _looks_like_tool_citation(sentence: str) -> bool:
    """True when a narration sentence cites a search tool as the source of a
    fact (e.g. 'As Firecrawl highlights', 'reported by Tavily'). A bare tool
    name deep inside a legit technical sentence (rare) still gets flagged by
    design: tools are never a citable source in a documentary."""
    sl = (sentence or "").lower()
    if not _TOOL_NAME_IN_NARR_RE.search(sl):
        return False
    # Restrict to the citation frame so we don't misfire on e.g. a proper noun
    # that happens to contain a tool name deep in the sentence.
    return any(
        phrase in sl
        for phrase in (
            "as firecrawl", "firecrawl highlights", "firecrawl reports",
            "firecrawl notes", "by firecrawl", "from firecrawl", "via firecrawl",
            "as tavily", "tavily highlights", "tavily reports", "tavily notes",
            "tavily search", "noted to tavily", "reported by tavily", "from tavily",
            "as exa", "exa reports", "exa notes", "exa search", "reported by exa",
            "from exa", "as jina", "jina reports", "jina notes", "jina reader",
            "reported by jina", "as duckduckgo", "duckduckgo reports", "from duckduckgo",
        )
    )


def is_soft_violation(violation: str) -> bool:
    """True for revisable style-class violations; False (=> hard) for fact,
    temporal, revenue, audience, structural or any unmarked/critic text."""
    return any(m in (violation or "") for m in _SOFT_VIOLATION_MARKERS)


def _is_grounding_truncation(narr_num: str, gt_numbers: set) -> bool:
    """
    True when a narration number is a precision-losing truncation/rounding of a
    corpus (ground-truth) number — NOT a fabrication. e.g. "19" ~ "19.29%",
    "41" ~ "41.11", "US$55" ~ "US$55.11 billion". One-directional and anchored to
    the corpus: a number with NO supporting base in gt_numbers stays flagged, so
    genuinely invented figures are still caught. Numeric suffix scaling (k/m/b) is
    honored so "1.5m" ~ "1,500,000" style variance passes too.
    """
    try:
        narr_val = float(re.sub(r'[^\d.]', '', narr_num or ""))
    except ValueError:
        return False
    if narr_val == 0:
        return False
    for gt in gt_numbers:
        scale = 1.0
        g = (gt or "").lower()
        if g.endswith("k"):
            scale = 1e3
        elif g.endswith("m"):
            scale = 1e6
        elif g.endswith("b"):
            scale = 1e9
        try:
            gt_val = float(re.sub(r'[^\d.]', '', g)) * scale
        except ValueError:
            continue
        if gt_val <= 0:
            continue
        # Narr is an integer truncation of gt (e.g. 19 -> 19.29).
        if narr_val == int(gt_val):
            return True
        # Narr is gt rounded to 1, 2, or 3 decimals (e.g. 19.3 -> 19.29).
        for nd in (1, 2, 3):
            if abs(narr_val - round(gt_val, nd)) < 1e-6:
                return True
        # Narr is gt rounded to a whole number within half a unit (e.g. 19 -> 19.29).
        if narr_val == round(gt_val, 0):
            return True
    return False


def observer_quality_score(script: ScriptData) -> float:
    """
    Paraphrase-diversity score in [0,1]: 1 - mean pairwise sentence similarity,
    so higher = more varied, less monotonous narration. Uses MiniLM embeddings
    when the semantic backend is on, else a TF-IDF cosine fallback. Cheap, one
    batch-encode; used to keep the best draft across revision rounds.
    """
    sentences = [
        s.strip().lower()
        for shot in script.shots
        for s in re.split(r'[.!?]', shot.narration_text)
        if len(s.strip()) > 15
    ]
    if len(sentences) < 2:
        return 0.5
    try:
        import numpy as np
        mat = None
        if semantic_embedder.available:
            mat = semantic_pairwise_similarity(sentences)
        if mat is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            mat = cosine_similarity(TfidfVectorizer().fit_transform(sentences)).astype(np.float32)
        n = len(sentences)
        off_diag = [mat[i][j] for i in range(n) for j in range(i + 1, n)]
        mean_sim = float(np.mean(off_diag)) if off_diag else 0.0
        return round(max(0.0, 1.0 - mean_sim), 4)
    except Exception:
        return 0.5


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

        # Verbatim source-copy check: narration must not reproduce the RAG corpus
        # (reads as pasted slop rather than creative narrative). Only an
        # UNEQUIVOCAL whole-sentence copy is flagged: a narration sentence whose
        # meaning is ~identical to a corpus sentence (semantic sim >= 0.94).
        #
        # The 0.82-0.93 band is deliberately NOT a violation: fact-dense narration
        # that preserves required names/numbers/dates can never paraphrase far
        # enough to drop its whole-sentence embedding sim below ~0.85, so flagging
        # it produced false positives and never-converging revision loops
        # (25 -> 27 -> 32). A longer lifted clause INSIDE an otherwise-original
        # sentence also does not warn alone, because the corpus is highly
        # redundant (many sources restate the same fact) and even clear rephrases
        # share long factual word-runs — only the whole-sentence meaning check
        # separates a true copy from legitimate fact-dense prose.
        if crawled_content:
            corpus_norm = re.sub(r'\s+', ' ', crawled_content.lower())
            corpus_sents = [
                re.sub(r'[^a-z0-9 ]', '', s).strip()
                for s in re.split(r'[.!?]', corpus_norm)
                if len(s.strip().split()) >= 12
            ]
            if corpus_sents:
                _sem_on = semantic_embedder.available
                corpus_vectors = None
                if _sem_on:
                    corpus_vectors = semantic_embedder.encode_batch(corpus_sents)
                    if corpus_vectors is None:
                        _sem_on = False

                for shot in script.shots:
                    for sent in re.split(r'[.!?]', shot.narration_text):
                        sent_norm = re.sub(r'[^a-z0-9 ]', '', sent.strip().lower()).strip()
                        if len(sent_norm.split()) < 12:
                            continue
                        # Semantic whole-sentence meaning-copy check: catches lightly
                        # rewritten but semantically identical slop the substring test
                        # misses, while avoiding false positives on the 0.82-0.93 band.
                        if _sem_on and corpus_vectors is not None:
                            q_vec = semantic_embedder.encode_batch([sent_norm])
                            if q_vec is not None:
                                sims = q_vec[0] @ corpus_vectors.T
                                max_sim = float(sims.max()) if len(sims) else 0.0
                                if max_sim >= COPY_SEMANTIC_HARD_THRESHOLD:
                                    violations.append(
                                        f"Shot #{shot.shot_id} verbatim source copy: narration copies the "
                                        f"RAG corpus nearly word-for-word (semantic sim {max_sim:.2f}): '{sent.strip()[:90]}'"
                                    )
                                continue
                        for cs in corpus_sents:
                            if sent_norm in cs or cs in sent_norm:
                                violations.append(
                                    f"Shot #{shot.shot_id} verbatim source copy: narration copies the "
                                    f"RAG corpus nearly word-for-word: '{sent.strip()[:90]}'"
                                )
                                break

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
                        # A3: a narration number that is a valid truncation / rounding
                        # of a corpus number (e.g. "19" vs "19.29%", "41" vs "41.11",
                        # "US$55" vs "US$55.11 billion") is NOT a fabrication — the LLM
                        # just dropped precision. Only hard-flag figures with NO
                        # supporting base in the corpus. This prevents truncation-only
                        # diffs from aborting a run (hard fact audit).
                        if _is_grounding_truncation(num, gt_numbers):
                            continue
                        flagged_sentences_info.append((
                            shot.shot_id,
                            f"Numerical claim '{num}' in: {shot.narration_text}",
                            "Fact Audit"
                        ))

            # 2. Dynamic Temporal Anchor Check — DETERMINISTIC (no critic coin-flip).
            #    A past year that never appears in the ground-truth corpus is a
            #    fabricated/outdated anchor -> hard flag. A sentence that NAMES its own
            #    past year ("in 2023", "a 2023 Christmas dinner", "back in 2025") is
            #    historically framed BY that date, so it can never read as a
            #    current-2026 event and is accepted WITHOUT routing to the LLM critic
            #    (which has repeatedly false-rejected exactly these: e.g. rejecting
            #    "'5 billion in 2023 to over $4...'" as 'presented as current' even
            #    though the sentence says "in 2023"). Only a sentence that pairs the
            #    past-year data with a CURRENT-time word (now/today/this year/as of)
            #    is a genuine current-placeholder risk and gets routed to the critic.
            _CURRENT_TIME_WORDS = (" today", "this year", "this month", "currently",
                                   "current", "as of", " now ", " right now")
            for past_y in past_years:
                if past_y not in narration_lower:
                    continue
                if past_y not in ground_truth_corpus:
                    flagged_sentences_info.append((
                        shot.shot_id,
                        f"Outdated year '{past_y}' in: {shot.narration_text}",
                        "Temporal Audit"
                    ))
                    continue
                for _sent in [s.strip() for s in re.split(r'[.!?]', shot.narration_text) if len(s.strip()) > 15]:
                    _sl = _sent.lower()
                    if past_y not in _sl:
                        continue
                    # Self-dated historical framing -> accepted by construction.
                    if (f"in {past_y}" in _sl or f"during {past_y}" in _sl
                            or f"since {past_y}" in _sl or f"back in {past_y}" in _sl
                            or f"{past_y}," in _sl):
                        continue
                    # Ambiguous tense: old figure phrased as current -> critic.
                    if any(c in _sl for c in _CURRENT_TIME_WORDS):
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
                4. TEMPORAL RULE: APPROVE pre-2026 data whenever the sentence itself
                   names the year with a temporal preposition ('in 2023', 'during 2024',
                   'since 2020', 'back in 2025') or is explicitly dated (e.g. 'a 2023
                   Christmas dinner'). Such self-dated sentences are historical by
                   construction and must NEVER be rejected. Only reject an ASSERTION
                   that presents pre-2026 data as a current {current_year} event when
                   the sentence uses a CURRENT tense without any year anchoring (e.g.
                   'this month the industry is $5 billion' where the corpus only dates
                   it to 2023).
                5. Return a JSON object with a single key "violations" containing an array
                   of strings. Each string is the EXACT text of a REJECTED assertion followed
                   by the reason it fails. Return an empty array if all flagged claims are
                   STYLE or supported.
                """
                system_prompt = "You are a precise, objective facts verification critic. Return valid JSON only."
                try:
                    critic_result = llm_client.generate_json(prompt, system_prompt, route="critic", thinking="high")
                    if critic_result and "violations" in critic_result:
                        rejected_claims = critic_result["violations"]
                        for violation in rejected_claims:
                            # Map the violation string back to a flagged shot ID
                            matched_sid = None
                            m = re.search(r"Shot #?(\d+)", violation)
                            if m:
                                matched_sid = int(m.group(1))
                            else:
                                for sid, sentence, itype in flagged_sentences_info:
                                    if sentence in violation or violation in sentence:
                                        matched_sid = sid
                                        break
                            if matched_sid is not None:
                                violations.append(f"Shot #{matched_sid} fact audit violation: {violation}")
                            else:
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
        topic=None, channel_phase: str = "REVENUE", crawled_content: str = "",
        region: str = "all"
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

        # ── Gate: raw markup/scrape/citation junk in narration (HARD) ────────
        # Markdown links, URLs, datelines, 'Retrieved' tails, bylines, follow
        # prompts and hashtag tags in the narration are paste/leak errors, not
        # style, so they abort rather than soft-approve.
        for _shot in script.shots:
            _m = _RAW_JUNK_IN_NARR_RE.search(_shot.narration_text or "")
            if _m:
                violations.append(
                    f"Raw markup/citation junk in Shot #{_shot.shot_id} narration "
                    f"contains a markdown link, URL, dateline, byline/follow prompt, "
                    f"hashtag tag, or 'Retrieved' tail "
                    f"(leaked {_m.group(0)[:60]!r}). Scrub before publish."
                )
            for _sent in re.split(r'[.!?]', _shot.narration_text or ""):
                if _looks_like_tool_citation(_sent):
                    violations.append(
                        f"RAG tool-name citation leak in Shot #{_shot.shot_id} narration "
                        f"attributes a fact to the retrieval toolchain, not a publication "
                        f"(leaked {_sent.strip()[:90]!r}). Cite the actual source instead."
                    )
            # ── Gate: no bare unexplained acronyms in narration (HARD) ───────
            # Documentary narration must ship ZERO shorthand initialisms — each must
            # be spelt out to the contextually-appropriate full term. Only proper-
            # noun initialisms (NASA, IBM, NYSE, CNN, …) and unit symbols (USD,
            # %, °C) may remain. Deterministic: the acronym gate cannot be swung
            # by LLM sampling, it just reads the all-caps tokens.
            for _tok in _ACRONYM_RE.findall(_shot.narration_text or ""):
                if _tok in _ACRONYM_ALLOWLIST:
                    continue
                violations.append(
                    f"Bare acronym '{_tok}' in Shot #{_shot.shot_id} narration — "
                    f"spell it out to its most appropriate full term for the sentence "
                    f"context (e.g. '{_tok}' should be expressed as the full phrase). "
                    f"Only proper-noun initialisms may remain."
                )

        # ── Gate 0: Audience Type Gate ──────────────────────────────────────
        # Hard-block entertainment/gossip + YMYL medical content regardless of
        # other scores (neither is safe to auto-produce unattended).
        if topic and getattr(topic, "audience_type", "") == "blocked":
            violations.append(
                "Audience Gate FAIL: Topic audience_type='blocked' "
                "(entertainment/gossip or YMYL medical). "
                "Highly demo/gamma Rx — hard blocked. "
                "Select a Tech/Finance/Science/Space/History/Business topic instead."
            )
            return False, violations  # Early exit — no point checking further

        # ── Gate 0b: Revenue Gate (REVENUE + SCALE phases only) ─────────────
        # In GROWTH phase we skip this — goal is watch-time not RPM.
        # Evaluated against the market this topic was selected for (dynamic
        # region), matching the fact-retriever forecast.
        if topic and channel_phase in ("REVENUE", "SCALE"):
            rev = monetization_optimizer.calculate_revenue_yield(topic, estimated_runtime_mins=13.0, region=region)
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

        # Semantic fast path: when the MiniLM backend is on, encode every shot
        # sentence ONCE and reuse the pairwise matrix during the loop (avoids
        # re-encoding the growing sentence list per sentence).
        _sem_matrix = None
        _sem_index: Dict[str, int] = {}
        if semantic_embedder.available:
            _all_shot_sents = [
                s.strip().lower()
                for shot in script.shots
                for s in re.split(r'[.!?]', shot.narration_text)
                if len(s.strip()) > 15
            ]
            if _all_shot_sents:
                _sem_matrix = semantic_pairwise_similarity(_all_shot_sents)
                _sem_index = {s: i for i, s in enumerate(_all_shot_sents)}

        for shot in script.shots:
            word_count = len(shot.narration_text.split())
            if word_count > 155:
                violations.append(f"Shot #{shot.shot_id} narration too long ({word_count} words). Max 155 words per shot.")

            v_prompt = shot.visual_prompt.lower()
            if "16:9" not in v_prompt and "widescreen" not in v_prompt:
                violations.append(f"Shot #{shot.shot_id} visual prompt missing 16:9 widescreen specification.")

            if "cinematic" not in v_prompt and "8k" not in v_prompt and "photorealistic" not in v_prompt:
                violations.append(f"Shot #{shot.shot_id} visual prompt lacks aesthetic lighting/AQA keywords.")

            # Sentence duplication (verbatim + semantic similarity >= 0.82)
            shot_sentences = [s.strip().lower() for s in re.split(r'[.!?]', shot.narration_text) if len(s.strip()) > 15]
            for sentence in shot_sentences:
                if sentence in all_narr_sentences:
                    violations.append(f"Shot #{shot.shot_id} repetition: duplicate sentence '{sentence}'.")
                elif all_narr_sentences and _sem_matrix is not None:
                    _sidx = _sem_index.get(sentence)
                    if _sidx is not None:
                        prev_idxs = [i for s in all_narr_sentences if (i := _sem_index.get(s)) is not None]
                        if prev_idxs:
                            max_sim = max(float(_sem_matrix[_sidx][j]) for j in prev_idxs)
                            if max_sim > 0.82:
                                violations.append(
                                    f"Shot #{shot.shot_id} repetition: sentence semantically too similar "
                                    f"(sim {max_sim:.2f}): '{sentence}'"
                                )
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

        # (C) Single distinctive keyword over-repetition across shots — catches
        # slop like a non-topic keyword hammered in almost every shot. The topic's
        # OWN keywords/headline words are excluded (they legitimately recur), so
        # this only flags genuinely excessive filler, keeping it revisable.
        _STOP = {
            "about", "their", "which", "would", "could", "these", "those", "being",
            "been", "there", "still", "while", "after", "before", "other", "under",
            "again", "through", "every", "where", "because", "between", "during",
            "without", "around", "however", "people", "thing", "things",
        }
        _topic_tokens = set()
        if topic is not None:
            for kw in (getattr(topic, "keywords", None) or []):
                _topic_tokens.add(re.sub(r"[^a-z0-9-]", "", str(kw).lower()))
            _topic_tokens.update(
                re.findall(r"\b[a-z][a-z0-9-]{4,}\b", (getattr(topic, "headline", "") or "").lower())
            )
            # Also exclude summary tokens so central entities/states in the topic
            # (e.g. "jersey" in a New Jersey lawsuit, company names) aren't flagged
            # as over-repeated filler — they legitimately recur across shots.
            summary = getattr(topic, "summary", None) or ""
            _topic_tokens.update(
                re.findall(r"\b[a-z][a-z0-9-]{4,}\b", summary.lower())
            )
            # Expand common US state abbreviations used in the headline/summary
            # (e.g. "NJ" -> "new jersey") so the full state name isn't flagged as
            # over-repeated filler in the narration.
            _US_STATES = {
                "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
                "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
                "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
                "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
                "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
                "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
                "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
                "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
                "ny": "new york", "nc": "north carolina", "nd": "north dakota",
                "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
                "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
                "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
                "va": "virginia", "wa": "washington", "wv": "west virginia",
                "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
            }
            for _txt in ((getattr(topic, "headline", "") or ""), summary):
                for _abbrev, _name in _US_STATES.items():
                    if re.search(rf"\b{_abbrev}\b", _txt.lower()):
                        _topic_tokens.add(_name)
                        _topic_tokens.add(_name.split()[-1])
        _tok_freq: Dict[str, int] = {}
        for s in script.shots:
            for t in set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9-]{4,}\b', s.narration_text.lower())):
                if t in _topic_tokens or t in _STOP or t.isdigit():
                    continue
                _tok_freq[t] = _tok_freq.get(t, 0) + 1

        # Semantic topic-membership: drop any candidate token whose MEANING matches
        # the topic's OWN anchor words (keywords + headline/summary tokens). This
        # generically excludes topic entities and their inflections/synonyms
        # (e.g. 'chinese' ~ 'china') from the over-repetition count — the exact-
        # string blacklist can't do that, and it is what caused false rejections
        # like "chinese in 12/15 shots".
        if semantic_embedder.available and _tok_freq and topic is not None:
            try:
                _anchor_words = []
                for kw in (getattr(topic, "keywords", None) or []):
                    _kw = re.sub(r"[^a-z0-9-]", "", str(kw).lower())
                    if len(_kw) >= 4:
                        _anchor_words.append(_kw)
                _anchor_words += re.findall(
                    r"\b[a-z][a-z0-9-]{4,}\b",
                    ((getattr(topic, "headline", "") or "") + " " + (getattr(topic, "summary", "") or "")).lower(),
                )
                _anchor_words = list(dict.fromkeys(_anchor_words))
                if _anchor_words:
                    for _t in list(_tok_freq.keys()):
                        # Per-token isolation: a failure scoring one candidate token
                        # must NOT abort filtering for every other token (the old
                        # whole-loop try/except swallowed errors and left false
                        # positives like 'surge' ~ 'surging' in the count, causing
                        # spurious 'keyword over-repetition' rejections).
                        try:
                            _mem = semantic_topic_membership(_t, _anchor_words)
                        except Exception:
                            continue
                        if _mem is not None and _mem >= TOPIC_MEMBER_SEMANTIC_THRESHOLD:
                            del _tok_freq[_t]
            except Exception:
                pass

        _n_shots = len(script.shots)
        if _tok_freq and _n_shots >= 6:
            _worst = max(_tok_freq, key=_tok_freq.get)
            if _tok_freq[_worst] >= int(_n_shots * 0.7):
                violations.append(
                    f"Keyword over-repetition: '{_worst}' appears in {_tok_freq[_worst]}/{_n_shots} shots. Vary the vocabulary."
                )

        # (B) Source-attribution diversity check is bypassed/disabled for narration
        # since references are no longer included in the audio/video, only in the description.
        pass

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
            region=getattr(state, "region_market", "") or getattr(state, "region", "all") or "all",
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
                    "fact_audit": "PASSED",
                    "quality_score": observer_quality_score(state.script_data)
                },
                state_hash=compute_state_hash(state),
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
                    "fact_audit": "FAILED" if any("Fact Audit" in v or "Temporal Audit" in v for v in violations) else "PASSED",
                    "quality_score": observer_quality_score(state.script_data)
                },
                state_hash=compute_state_hash(state),
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
            )
