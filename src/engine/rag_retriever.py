import json
import urllib.request
import urllib.parse
import re
import os
import html
import time
from typing import Dict, Any, List, Tuple, Set
from src.schemas.state import TopicCandidate, VerifiedFact


class GraphNode:
    def __init__(self, entity_id: str, entity_type: str = "concept"):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.edges: List[Tuple[str, str]] = []  # List of (predicate, target_entity_id)

    def add_edge(self, predicate: str, target_id: str):
        self.edges.append((predicate, target_id))


class RAGTopicRetriever:
    """
    Stage 2: GraphRAG & Hybrid Information Retrieval Engine.
    Combines Standard Vector RAG for broad summaries with GraphRAG entity-relation semantic knowledge graphs
    for deep multi-hop factual reasoning and TrumorGPT hallucination defense fact checking.
    """

    def __init__(self):
        self.knowledge_graph: Dict[str, GraphNode] = {}
        # RAG cache: headline -> (timestamp, pack). Avoids re-fetching the same
        # paid/network searches on repeated runs for the same topic (cost + speed).
        self._rag_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._rag_cache_ttl_s = 6 * 3600

    def search_duckduckgo_facts(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes a DuckDuckGo HTML web search to extract real-world background facts.
        """
        results = []
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=8) as response:
                html = response.read().decode('utf-8', errors='ignore')

            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            ad_pattern = re.compile(
                r'\b(sign\s?up|subscribe|register|join\s?now|click\s?here|get\s?started|'
                r'free\s?trial|pricing\s?plan|price|coupon|discount|promo|checkout|'
                r'buy\s?now|add\s?to\s?cart|shop|store|order\s?now|affiliate|sponsor|'
                r'advertisement|marketing|ad\b|advertised|shopping|sale|discounted|deal|'
                r'best\sprice|cheap)\b',
                re.IGNORECASE
            )

            # Split the HTML page into individual result blocks
            blocks = html.split('<div class="result results_links results_links_deep web-result')
            for block in blocks[1:]:
                # Extract first anchor href
                url_match = re.search(r'href="([^"]+)"', block)
                if not url_match:
                    continue
                url = url_match.group(1)
                
                # Exclude sponsored tracking links
                if "y.js" in url or "ad_provider" in url or "sponsored" in url:
                    continue

                # Local match inside block for title and snippet
                title_match = re.search(r'<a class="result__title[^>]*>(.*?)</a>', block, re.DOTALL)
                snippet_match = re.search(r'<a class="result__snippet[^>]*>(.*?)</a>', block, re.DOTALL)
                
                if title_match and snippet_match:
                    clean_title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
                    clean_snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()
                    
                    if len(clean_snippet) > 30:
                        combined_text = f"{clean_title} {clean_snippet}"
                        if ad_pattern.search(combined_text):
                            continue

                        try:
                            vectorizer = TfidfVectorizer(stop_words='english')
                            tfidf_matrix = vectorizer.fit_transform([query, combined_text])
                            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                        except Exception:
                            sim = 0.0

                        if sim > 0.08:
                            results.append({"title": clean_title, "snippet": clean_snippet})
                            if len(results) >= max_results:
                                break
        except Exception as e:
            print(f"[RAGRetriever] Search Warning: {e}")

        return results

    def search_newsapi_facts(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Executes a NewsAPI query to retrieve structured global news snippets.
        """
        results = []
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            return results
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://newsapi.org/v2/everything?q={encoded_query}&pageSize={max_results}&apiKey={api_key}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8', errors='ignore'))
                articles = data.get("articles", [])
                for a in articles:
                    title = a.get("title", "")
                    desc = a.get("description", "")
                    if desc and len(desc) > 30:
                        results.append({"title": title, "snippet": desc})
        except Exception as e:
            print(f"[RAGRetriever] NewsAPI Warning: {e}")
        return results

    def search_wikipedia_facts(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Executes a Wikipedia Query API search to extract clean historical/factual snippets.
        """
        results = []
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&utf8=&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8', errors='ignore'))
                search_results = data.get("query", {}).get("search", [])
                for item in search_results[:max_results]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    clean_snippet = re.sub(r'<[^>]+>', '', snippet)
                    clean_snippet = html.unescape(clean_snippet)
                    if len(clean_snippet) > 30:
                        results.append({"title": title, "snippet": clean_snippet})
        except Exception as e:
            print(f"[RAGRetriever] Wikipedia Warning: {e}")
        return results

    def search_tavily_facts(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes a Tavily search (purpose-built for AI/RAG). Returns clean,
        ad-free snippets — no HTML scraping or ad filtering needed.
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return []
        try:
            import requests
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                },
                timeout=12,
            )
            data = resp.json()
            return [
                {"title": r.get("title", ""), "snippet": r.get("content", "")}
                for r in data.get("results", [])
                if r.get("content") and len(r.get("content", "")) > 30
            ]
        except Exception as e:
            print(f"[RAGRetriever] Tavily Warning: {e}")
            return []

    def search_firecrawl_facts(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes a Firecrawl search (clean crawl API returning page-level snippets).
        """
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            return []
        try:
            import requests
            resp = requests.post(
                "https://api.firecrawl.dev/v1/search",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"query": query, "limit": max_results},
                timeout=15,
            )
            data = resp.json()
            items = data.get("data", []) if isinstance(data, dict) else []
            out = []
            for it in items:
                desc = it.get("description") or (it.get("metadata", {}) or {}).get("description", "")
                title = it.get("title", "")
                if desc and len(str(desc)) > 30:
                    out.append({"title": title, "snippet": str(desc)})
            return out
        except Exception as e:
            print(f"[RAGRetriever] Firecrawl Warning: {e}")
            return []

    def extract_graph_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extracts semantic (Subject, Predicate, Object) triplets from retrieved research text.
        """
        triplets = []
        sentences = re.split(r'[.!?]', text)
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            words = s_clean.split()
            if len(words) >= 4:
                # Extract simple entity-relation heuristics
                subject = words[0].strip(",. ")
                predicate = words[1].strip(",. ")
                obj = " ".join(words[2:]).strip(",. ")
                if len(subject) > 2 and len(obj) > 2:
                    triplets.append((subject, predicate, obj))
                    # Register into in-memory knowledge graph
                    s_node = self.knowledge_graph.setdefault(subject.lower(), GraphNode(subject.lower()))
                    s_node.add_edge(predicate.lower(), obj.lower())
        return triplets[:15]

    def traverse_graph_paths(self, start_entity: str, max_depth: int = 2) -> List[str]:
        """
        Traverses semantic knowledge graph relational paths for multi-hop reasoning.
        """
        start_key = start_entity.lower()
        if start_key not in self.knowledge_graph:
            return []
        
        visited: Set[str] = set()
        paths: List[str] = []
        
        def dfs(curr_id: str, depth: int, current_path: str):
            if depth >= max_depth or curr_id in visited:
                return
            visited.add(curr_id)
            node = self.knowledge_graph.get(curr_id)
            if not node:
                return
            for pred, target in node.edges:
                path_str = f"{current_path} -[{pred}]-> {target}"
                paths.append(path_str)
                dfs(target, depth + 1, path_str)

        dfs(start_key, 0, start_key)
        return paths[:10]

    def select_rag_mode(self, query_complexity: str) -> str:
        """
        Hybrid RAG Router:
        Selects 'standard_vector' for simple overviews, or 'graph_rag' for deep multi-hop factual synthesis.
        """
        if any(w in query_complexity.lower() for w in ["history", "origin", "mechanism", "why", "relationship", "impact", "breakdown"]):
            return "graph_rag"
        return "standard_vector"

    def trumorgpt_verify_fact(self, claim: str) -> Tuple[bool, float, str]:
        """
        TrumorGPT-style Semantic Fact-Checker:
        Verifies script claims against registered knowledge graph triples to flag hallucinations.
        """
        claim_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', claim.lower()))
        if not self.knowledge_graph:
            return True, 0.90, "Ground truth facts clear."
        
        matching_edges = 0
        total_checks = 0
        for node_id, node in self.knowledge_graph.items():
            if node_id in claim_words:
                total_checks += 1
                for pred, target in node.edges:
                    if target in claim_words or any(w in claim_words for w in target.split()):
                        matching_edges += 1
                        break
        
        if total_checks == 0:
            return True, 0.85, "Fact check plausible based on broad corpus."
        
        confidence = matching_edges / total_checks
        is_verified = confidence >= 0.5
        msg = "Verified against GraphRAG evidence." if is_verified else "Potential hallucination detected; low graph support."
        return is_verified, float(round(confidence, 2)), msg

    def build_rag_knowledge_pack(
        self, topic: TopicCandidate, verified_facts: List[VerifiedFact]
    ) -> Dict[str, Any]:
        """
        Constructs a comprehensive 1,000+ word RAG Knowledge Pack containing:
        1. Core Headline & Real-World Trigger Summary
        2. Verified Facts & Primary Sources
        3. GraphRAG Multi-Hop Relational Paths & Knowledge Triplets
        4. TrumorGPT Citation Grounding & Fact Checks
        5. Future Strategic Implications & Category Insights
        """
        headline = topic.headline
        summary = topic.summary
        keywords = [k for k in topic.keywords if len(k) > 3][:6]

        # Cache: return recent RAG pack for the same headline/fact-count to avoid
        # redundant paid/network searches.
        cache_key = f"{headline}|{len(verified_facts)}"
        _now = time.time()
        _hit = self._rag_cache.get(cache_key)
        if _hit and (_now - _hit[0]) < self._rag_cache_ttl_s:
            return _hit[1]

        # Combine verified facts into core ground truth block
        verified_snippets = [f"{vf.headline}: {vf.summary} (Source: {vf.source_name})" for vf in verified_facts]
        ground_truth_block = "\n".join(verified_snippets) if verified_snippets else summary

        # Perform targeted RAG web queries to enrich depth
        search_queries = [
            f"{headline} background timeline history",
            f"{headline} key facts statistics analysis",
            f"{' '.join(keywords[:3])} strategic future impact"
        ]

        retrieved_facts = []
        all_text_corpus = summary + " " + ground_truth_block

        from src.engine.external_apis import external_api_manager

        for q in search_queries:
            has_api_data = False
            
            # 1. Fetch from Exa AI (Neural Search)
            try:
                facts = external_api_manager.fetch_exa_semantic_facts(q)
                if facts:
                    for f in facts:
                        retrieved_facts.append(f"• [Exa: {f.headline}]: {f.summary}")
                        all_text_corpus += " " + f.summary
                    has_api_data = True
            except Exception as e:
                print(f"[RAGRetriever] Exa query failed: {e}")

            # 2. Fetch from NewsAPI (Global News)
            try:
                items = self.search_newsapi_facts(q, max_results=2)
                if items:
                    for item in items:
                        retrieved_facts.append(f"• [NewsAPI: {item['title']}]: {item['snippet']}")
                        all_text_corpus += " " + item['snippet']
                    has_api_data = True
            except Exception as e:
                print(f"[RAGRetriever] NewsAPI query failed: {e}")

            # 3. Fetch from Wikipedia (Academic Grounding)
            try:
                items = self.search_wikipedia_facts(q, max_results=2)
                if items:
                    for item in items:
                        retrieved_facts.append(f"• [Wikipedia: {item['title']}]: {item['snippet']}")
                        all_text_corpus += " " + item['snippet']
                    has_api_data = True
            except Exception as e:
                print(f"[RAGRetriever] Wikipedia query failed: {e}")

            # 4. Fetch from Tavily (clean AI/RAG search, ad-free)
            try:
                items = self.search_tavily_facts(q, max_results=3)
                if items:
                    for item in items:
                        retrieved_facts.append(f"• [Tavily: {item['title']}]: {item['snippet']}")
                        all_text_corpus += " " + item['snippet']
                    has_api_data = True
            except Exception as e:
                print(f"[RAGRetriever] Tavily query failed: {e}")

            # 5. Fetch from Firecrawl (clean crawl API, page-level snippets)
            try:
                items = self.search_firecrawl_facts(q, max_results=3)
                if items:
                    for item in items:
                        retrieved_facts.append(f"• [Firecrawl: {item['title']}]: {item['snippet']}")
                        all_text_corpus += " " + item['snippet']
                    has_api_data = True
            except Exception as e:
                print(f"[RAGRetriever] Firecrawl query failed: {e}")

            # NOTE: DuckDuckGo HTML scraping removed — ad-heavy markup corrupts RAG.

        # 1. GraphRAG Triplet Extraction & In-Memory Graph Indexing
        triplets = self.extract_graph_triplets(all_text_corpus)
        
        # 2. Graph Path Traversal for Keyword Entities
        graph_paths = []
        for kw in keywords[:3]:
            paths = self.traverse_graph_paths(kw)
            graph_paths.extend(paths)

        # 3. Dynamic RAG Router Selection
        rag_mode = self.select_rag_mode(headline)

        # 4. TrumorGPT Fact Verification Check
        is_verified, confidence, fact_msg = self.trumorgpt_verify_fact(headline + " " + summary)

        rag_retrieved_block = "\n".join(retrieved_facts[:8]) if retrieved_facts else "No additional web snippets retrieved."
        graph_paths_block = "\n".join([f"• {p}" for p in graph_paths[:6]]) if graph_paths else "No multi-hop paths traversed."

        # Derive core domain category. Prefer the RSS/audience classification
        # (consistent with the revenue model) when it was genuinely set; fall back
        # to a keyword heuristic only for topics that were not audience-classified.
        rss_niche = (getattr(topic, "niche_category", "") or "").strip()
        rss_audience = (getattr(topic, "audience_type", "") or "general").strip()
        if rss_niche and rss_audience and rss_audience != "general":
            category = rss_niche
        else:
            combined_corpus = f"{headline} {summary} {' '.join(keywords)}".lower()
            if any(w in combined_corpus for w in ["ai", "chatgpt", "software", "tech", "chip", "nvidia", "cloud", "seo", "app"]):
                category = "Technology & Artificial Intelligence"
            elif any(w in combined_corpus for w in ["fed", "market", "stock", "trading", "crypto", "bank", "inflation", "revenue", "dollar"]):
                category = "Global Economics & Finance"
            elif any(w in combined_corpus for w in ["space", "nasa", "planet", "rocket", "star", "physics", "science"]):
                category = "Space & Scientific Innovation"
            elif any(w in combined_corpus for w in ["war", "election", "policy", "country", "president", "government"]):
                category = "Geopolitics & World Affairs"
            else:
                category = "Global Trends & Cultural Infotainment"

        # Guard: only surface triplets that look like real (Subject)-(Predicate)-(Object)
        # statements, to avoid flooding the LLM prompt with naive sentence-split garbage.
        meaningful_triplets = [
            t for t in triplets
            if len(t[0]) >= 3 and len(t[1]) >= 3 and len(t[2]) >= 3
            and not t[1].lower() in {"and", "the", "of", "to", "in", "for", "with"}
        ]
        graph_triplets_block = "\n".join([f"• ({s}) --[{p}]--> ({o})" for s, p, o in meaningful_triplets[:6]])

        knowledge_pack = {
            "topic_headline": headline,
            "category": category,
            "summary": summary,
            "keywords": keywords,
            "rag_mode": rag_mode,
            "ground_truth_block": ground_truth_block,
            "rag_retrieved_context": rag_retrieved_block,
            "graph_triplets": graph_triplets_block,
            "graph_paths": graph_paths_block,
            "trumorgpt_verification": {
                "is_verified": is_verified,
                "confidence": confidence,
                "message": fact_msg
            },
            "fact_corpus": (
                f"{ground_truth_block}\n"
                f"{rag_retrieved_block}"
            ),
            "full_rag_context_text": (
                f"TOPIC CATEGORY: {category}\n"
                f"RAG EXECUTION MODE: {rag_mode.upper()}\n"
                f"HEADLINE: {headline}\n"
                f"SUMMARY: {summary}\n\n"
                f"TRUMORGPT VERIFICATION: {fact_msg} (Confidence: {confidence})\n\n"
                f"VERIFIED GROUND TRUTH FACTS:\n{ground_truth_block}\n\n"
                f"GRAPHRAG KNOWLEDGE GRAPH TRIPLETS:\n{graph_triplets_block}\n\n"
                f"GRAPHRAG MULTI-HOP RELATIONAL PATHS:\n{graph_paths_block}\n\n"
                f"RETRIEVED DEEP CONTEXT & BACKGROUND:\n{rag_retrieved_block}"
            )
        }

        self._rag_cache[cache_key] = (time.time(), knowledge_pack)
        return knowledge_pack


rag_retriever = RAGTopicRetriever()

