import re
import json

from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document

from retrieval.retriever import semantic_search, filtered_search

_CLASSIFIER_PROMPT = """You are a query routing agent for a privacy policy database of 115 real websites.

Classify the user question into exactly ONE type:

SIMPLE   - No specific website mentioned. General question across all policies.
           Example: "What data do websites collect?"

FILTERED - ONE specific website is named.
           Example: "What does amazon.com say about third parties?"

COMPARE  - Explicitly compares TWO OR MORE named websites.
           Example: "Compare how amazon and nytimes handle data deletion."

RULES:
- If NO specific website is named, always use SIMPLE. Never invent website names.
- For FILTERED: put the mentioned domain keyword in filters.url (e.g. "amazon").
- For COMPARE: one sub_query per site, each with its domain keyword in filters.url.
- Never use placeholder URLs like "example.com" or "sample.org".

Respond ONLY with valid JSON, no markdown:
{{
  "type": "SIMPLE" | "FILTERED" | "COMPARE",
  "sub_queries": [
    {{"query": "search string", "filters": {{}}, "k": 5}}
  ],
  "reasoning": "one sentence"
}}

User question: {question}
"""

_PLACEHOLDER_DOMAINS = {"example.com", "sample.org", "website.com", "site.com"}

def _extract_json(text: str) -> dict:
    """Strip markdown fences and parse the first JSON object found."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)

def _has_hallucinated_urls(plan: dict) -> bool:
    """Return True if the plan contains invented placeholder URLs."""
    for sq in plan.get("sub_queries", []):
        url = sq.get("filters", {}).get("url", "")
        if url.startswith("contains:"):
            return True
        if any(p in url.lower() for p in _PLACEHOLDER_DOMAINS):
            return True
    return False

def _fallback_plan(question: str) -> dict:
    return {
        "type": "SIMPLE",
        "sub_queries": [{"query": question, "filters": {}, "k": 5}],
        "reasoning": "Fallback: no specific site detected.",
    }

def _post_filter_by_url(
    vectorstore: Chroma,
    query: str,
    url_keyword: str,
    k: int,
) -> list[Document]:
    """
    ChromaDB does not support substring matching, so we fetch broadly
    then post-filter by checking whether url_keyword appears in the
    document's URL metadata field.
    """
    candidates = semantic_search(vectorstore, query, k=80)
    filtered = [
        d for d in candidates
        if url_keyword.lower() in d.metadata.get("url", "").lower()
    ][:k]

    if not filtered:
        print(f"[Planner] No matches for url keyword '{url_keyword}'. Using semantic fallback.")
        filtered = semantic_search(vectorstore, query, k=k)

    return filtered

class PlannerAgent:
    """Routes queries to the appropriate retrieval strategy."""

    def __init__(self, vectorstore: Chroma, llm_model: str = "qwen2.5:7b"):
        self.vectorstore = vectorstore
        self.llm = OllamaLLM(model=llm_model, temperature=0)

    def classify(self, question: str) -> dict:
        """Classify the question and return a retrieval plan dict."""
        raw = self.llm.invoke(_CLASSIFIER_PROMPT.format(question=question))
        try:
            plan = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            print("[Planner] JSON parse failed — falling back to SIMPLE.")
            return _fallback_plan(question)

        if _has_hallucinated_urls(plan):
            print("[Planner] Hallucinated URLs detected — overriding to SIMPLE.")
            return _fallback_plan(question)

        return plan

    def retrieve(self, question: str) -> tuple[list[Document], dict]:
        """
        Classify the question and execute the appropriate search strategy.

        Returns:
            Tuple of (retrieved Documents, plan metadata dict).
        """
        plan = self.classify(question)
        query_type = plan.get("type", "SIMPLE")
        sub_queries = plan.get("sub_queries") or [{"query": question, "filters": {}, "k": 5}]

        print(f"\n[Planner] Type     : {query_type}")
        print(f"[Planner] Reasoning: {plan.get('reasoning', '')}")
        print(f"[Planner] Sub-queries: {json.dumps(sub_queries, indent=2)}")

        if query_type == "SIMPLE":
            q = sub_queries[0]
            docs = semantic_search(self.vectorstore, q["query"], k=q.get("k", 5))

        elif query_type == "FILTERED":
            q = sub_queries[0]
            url_kw = q.get("filters", {}).get("url", "")
            date   = q.get("filters", {}).get("collection_date", "")

            if url_kw:
                docs = _post_filter_by_url(self.vectorstore, q["query"], url_kw, q.get("k", 5))
            elif date:
                docs = filtered_search(
                    self.vectorstore, q["query"],
                    filters={"collection_date": date},
                    k=q.get("k", 5),
                )
            else:
                docs = semantic_search(self.vectorstore, q["query"], k=q.get("k", 5))

        elif query_type == "COMPARE":
            seen: set[str] = set()
            docs = []
            for sq in sub_queries:
                url_kw = sq.get("filters", {}).get("url", "")
                k = sq.get("k", 4)
                batch = (
                    _post_filter_by_url(self.vectorstore, sq["query"], url_kw, k)
                    if url_kw
                    else semantic_search(self.vectorstore, sq["query"], k=k)
                )
                for doc in batch:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        docs.append(doc)

        else:
            docs = semantic_search(self.vectorstore, question, k=5)

        print(f"[Planner] Retrieved {len(docs)} documents.")
        return docs, plan
