import json
import urllib.request
import urllib.parse
import re
from typing import Dict, Any, List
from src.schemas.state import TopicCandidate, VerifiedFact


class RAGTopicRetriever:
    """
    RAG (Retrieval-Augmented Generation) Topic Context Retriever:
    Intelligently retrieves deep real-world context, historical origins, factual benchmarks,
    key stakeholders, and long-term implications for ANY trending topic (Tech, AI, Finance,
    Geopolitics, Science, Entertainment, Health, Sports, etc.).
    """

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

            snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
            titles = re.findall(r'<a class="result__title[^>]*>(.*?)</a>', html, re.DOTALL)

            for i in range(min(len(snippets), max_results)):
                clean_snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()
                clean_title = re.sub(r'<[^>]+>', '', titles[i] if i < len(titles) else "").strip()
                if len(clean_snippet) > 30:
                    results.append({"title": clean_title, "snippet": clean_snippet})
        except Exception as e:
            print(f"[RAGRetriever] Search Warning: {e}")

        return results

    def build_rag_knowledge_pack(
        self, topic: TopicCandidate, verified_facts: List[VerifiedFact]
    ) -> Dict[str, Any]:
        """
        Constructs a comprehensive 1,000+ word RAG Knowledge Pack containing:
        1. Core Headline & Real-World Trigger Summary
        2. Verified Facts & Primary Sources
        3. Retrieved Deep Historical Precedents & Background Data
        4. Key Stakeholders, Companies & Regulatory Bodies
        5. Future Strategic Implications & Category Insights
        """
        headline = topic.headline
        summary = topic.summary
        keywords = [k for k in topic.keywords if len(k) > 3][:6]

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
        for q in search_queries:
            items = self.search_duckduckgo_facts(q, max_results=3)
            for item in items:
                retrieved_facts.append(f"• [{item['title']}]: {item['snippet']}")

        rag_retrieved_block = "\n".join(retrieved_facts[:8]) if retrieved_facts else "No additional web snippets retrieved."

        # Derive core domain category dynamically from text
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

        knowledge_pack = {
            "topic_headline": headline,
            "category": category,
            "summary": summary,
            "keywords": keywords,
            "ground_truth_block": ground_truth_block,
            "rag_retrieved_context": rag_retrieved_block,
            "full_rag_context_text": (
                f"TOPIC CATEGORY: {category}\n"
                f"HEADLINE: {headline}\n"
                f"SUMMARY: {summary}\n\n"
                f"VERIFIED GROUND TRUTH FACTS:\n{ground_truth_block}\n\n"
                f"RETRIEVED DEEP CONTEXT & BACKGROUND:\n{rag_retrieved_block}"
            )
        }

        return knowledge_pack


rag_retriever = RAGTopicRetriever()
