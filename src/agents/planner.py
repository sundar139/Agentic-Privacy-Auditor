import re
import json

from langchain_chroma import Chroma
from langchain_core.documents import Document

from retrieval.retriever import (
    semantic_search,
    filtered_search,
    threshold_search,
    section_search,
    year_filtered_search,
    detect_section_slug,
)


# ── Dynamic k by query type ───────────────────────────────────────────────────
K_BY_TYPE: dict[str, int] = {
    "SIMPLE":    5,
    "FILTERED":  8,   # one site → need broader policy coverage
    "COMPARE":   5,   # per site → 5 × n_sites total
    "AMBIGUOUS": 0,   # never retrieve, ask user first
}
_QUOTE_K_BOOST = 10  # override for verbatim-quote queries

# Legal-term pairs requiring two independent sub-queries
_LEGAL_TERM_PAIRS: list[tuple[str, str]] = [
    ("sell", "share"),
    ("sell", "disclose"),
    ("collect", "use"),
    ("store", "retain"),
    ("share", "disclose"),
]


# ─────────────────────────────────────────────
# LLM backend selector
# ─────────────────────────────────────────────

def _get_llm(model: str = "qwen2.5:7b"):
    import os
    if os.environ.get("HF_TOKEN"):
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="conversational",
            huggingfacehub_api_token=os.environ["HF_TOKEN"],
            temperature=0.01,
            max_new_tokens=512,
        )
        return ChatHuggingFace(llm=endpoint)
    else:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=model, temperature=0)


def _llm_invoke(llm, prompt: str) -> str:
    raw = llm.invoke(prompt)
    return raw.content if hasattr(raw, "content") else raw


# ─────────────────────────────────────────────
# Classifier prompt
# ─────────────────────────────────────────────

_CLASSIFIER_PROMPT = """You are a query routing agent for a privacy policy database of 115 real websites.

Classify the user question into exactly ONE type:

SIMPLE    - No specific website mentioned. Genuine broad question across all policies.
            Example: "What data do websites collect?"

FILTERED  - ONE specific website is named.
            Example: "What does amazon.com say about third parties?"

COMPARE   - Explicitly compares TWO OR MORE named websites.
            Example: "Compare how amazon and nytimes handle data deletion."

AMBIGUOUS - The question uses vague pronouns ("they", "it", "the company",
            "their", "the site") as the ONLY reference to a company AND asks
            about a real, recognisable privacy topic (cookies, tracking, data
            sharing, deletion, etc.).
            Example: "Do they sell my data?" / "What does it say about cookies?"
            NOT AMBIGUOUS: questions about nonsensical or clearly fictional data
            types (e.g. "neural-link data", "brainwave data", "dream logs") —
            use SIMPLE instead so the Auditor can handle it.
            NOT AMBIGUOUS: "this policy" or "this document" — treat as SIMPLE.

RULES:
- Use AMBIGUOUS ONLY when a real privacy topic is present but no website can
  be identified. For fictional/nonsensical data types, always use SIMPLE.
- COMPARE fires ONLY when TWO DIFFERENT website domain names are both present.
  "Summarize X and tell me Y about site.com" → FILTERED, NOT COMPARE.
  The word "and" within a question about ONE website is NOT a COMPARE trigger.
- For FILTERED: put the domain keyword in filters.url (e.g. "nytimes").
- For COMPARE: one sub_query per site, each with its domain keyword in filters.url.
- Never use placeholder URLs like "example.com" or "sample.org".
- If the question compares legally distinct terms (e.g. "sell" vs "share",
  "collect" vs "use"), generate two sub_queries — one per term.

Respond ONLY with valid JSON, no markdown:
{{
  "type": "SIMPLE" | "FILTERED" | "COMPARE" | "AMBIGUOUS",
  "sub_queries": [
    {{"query": "search string", "filters": {{}}, "k": 5}}
  ],
  "reasoning": "one sentence"
}}

User question: {question}
"""

_PLACEHOLDER_DOMAINS = {"example.com", "sample.org", "website.com", "site.com"}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def _has_hallucinated_urls(plan: dict) -> bool:
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
        "sub_queries": [{"query": question, "filters": {}, "k": K_BY_TYPE["SIMPLE"]}],
        "reasoning": "Fallback: classification failed.",
    }


def _extract_year(question: str) -> int | None:
    match = re.search(r"\b(199\d|20[012]\d|2030)\b", question)
    return int(match.group(1)) if match else None


def _is_quote_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in ("quote", "exact wording", "verbatim", "exact text", "copy of"))


def _has_legal_term_pair(question: str) -> tuple[str, str] | None:
    q = question.lower()
    for a, b in _LEGAL_TERM_PAIRS:
        if a in q and b in q:
            return (a, b)
    return None


def _post_filter_by_url(
    vectorstore: Chroma,
    query: str,
    url_keyword: str,
    k: int,
) -> tuple[list[Document], bool]:
    """
    Over-fetches with threshold_search, then post-filters by URL metadata.
    Returns (docs, in_corpus). in_corpus=False signals the site isn't in OPP-115.
    """
    candidates = threshold_search(vectorstore, query, k=80)
    matched = [
        d for d in candidates
        if url_keyword.lower() in d.metadata.get("url", "").lower()
    ][:k]
    if matched:
        return matched, True
    print(f"[Planner] '{url_keyword}' not found in corpus. Returning semantic fallback.")
    return semantic_search(vectorstore, query, k=k), False


def _interleave_site_docs(site_results: dict[str, list[Document]]) -> list[Document]:
    """
    FIX: Balanced per-site dedup for COMPARE.
    Round-robins across sites so neither side dominates the doc list.
    """
    seen: set[str] = set()
    docs: list[Document] = []
    max_len = max((len(v) for v in site_results.values()), default=0)
    for i in range(max_len):
        for site_docs in site_results.values():
            if i < len(site_docs):
                doc = site_docs[i]
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    docs.append(doc)
    return docs


# ─────────────────────────────────────────────
# Planner agent
# ─────────────────────────────────────────────

class PlannerAgent:
    """Routes queries to the appropriate retrieval strategy."""

    def __init__(self, vectorstore: Chroma, llm_model: str = "qwen2.5:7b"):
        self.vectorstore = vectorstore
        self.llm = _get_llm(llm_model)

    def classify(self, question: str) -> dict:
        raw = _llm_invoke(self.llm, _CLASSIFIER_PROMPT.format(question=question))
        try:
            plan = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            print("[Planner] JSON parse failed — falling back to SIMPLE.")
            return _fallback_plan(question)
        if _has_hallucinated_urls(plan):
            print("[Planner] Hallucinated URLs — overriding to SIMPLE.")
            return _fallback_plan(question)
        return plan

    def retrieve(self, question: str) -> tuple[list[Document], dict]:
        """
        Classify and execute retrieval.

        Plan diagnostic keys:
            ambiguous        : bool  — query needs clarification (no retrieval)
            corpus_miss      : bool  — FILTERED site not in OPP-115
            corpus_miss_site : str
            compare_misses   : list  — sites missing in COMPARE
            year_filter      : int
            section_filter   : str  — OPP-115 slug used for section_search
            quote_query      : bool  — verbatim quote was requested
        """
        plan = self.classify(question)
        query_type = plan.get("type", "SIMPLE")
        year = _extract_year(question)
        is_quote = _is_quote_query(question)
        legal_pair = _has_legal_term_pair(question)
        section_slug = detect_section_slug(question) if is_quote else None

        # ── AMBIGUOUS: stop immediately, ask for clarification ────────────────
        if query_type == "AMBIGUOUS":
            plan["ambiguous"] = True
            print("[Planner] AMBIGUOUS query — skipping retrieval.")
            return [], plan

        sub_queries = plan.get("sub_queries") or [
            {"query": question, "filters": {}, "k": K_BY_TYPE.get(query_type, 5)}
        ]
        base_k = _QUOTE_K_BOOST if is_quote else K_BY_TYPE.get(query_type, 5)
        for sq in sub_queries:
            sq["k"] = base_k

        if is_quote:
            plan["quote_query"] = True
        if section_slug:
            plan["section_filter"] = section_slug

        print(f"\n[Planner] Type       : {query_type}")
        print(f"[Planner] Reasoning  : {plan.get('reasoning', '')}")
        print(f"[Planner] Sub-queries: {json.dumps(sub_queries, indent=2)}")
        if year:
            print(f"[Planner] Year filter: {year}")
        if section_slug:
            print(f"[Planner] Section    : {section_slug}")
        if legal_pair:
            print(f"[Planner] Legal pair : {legal_pair}")

        # ── SIMPLE ───────────────────────────────────────────────────────────
        if query_type == "SIMPLE":
            q = sub_queries[0]

            if section_slug:
                docs = section_search(self.vectorstore, q["query"], section_slug, k=base_k)

            elif legal_pair and len(sub_queries) < 2:
                # Priority 4: two independent fetches for legal term pair
                term_a, term_b = legal_pair
                docs_a = semantic_search(self.vectorstore, f"{q['query']} {term_a}", k=base_k)
                docs_b = semantic_search(self.vectorstore, f"{q['query']} {term_b}", k=base_k)
                seen: set[str] = set()
                docs = []
                for d in docs_a + docs_b:
                    if d.page_content not in seen:
                        seen.add(d.page_content)
                        docs.append(d)

            elif year:
                docs = year_filtered_search(self.vectorstore, q["query"], year=year, k=base_k)
                plan["year_filter"] = year

            else:
                docs = semantic_search(self.vectorstore, q["query"], k=base_k)

        # ── FILTERED ─────────────────────────────────────────────────────────
        elif query_type == "FILTERED":
            q = sub_queries[0]
            url_kw = q.get("filters", {}).get("url", "")
            date   = q.get("filters", {}).get("collection_date", "")

            if url_kw:
                if section_slug:
                    candidates = section_search(
                        self.vectorstore, q["query"], section_slug,
                        k=base_k * 3, url_kw=url_kw,   # N7: pass url_kw for site-scoped fallback
                    )
                    matched = [
                        d for d in candidates
                        if url_kw.lower() in d.metadata.get("url", "").lower()
                    ][:base_k]
                    in_corpus = bool(matched)
                    docs = matched if matched else candidates[:base_k]
                else:
                    docs, in_corpus = _post_filter_by_url(
                        self.vectorstore, q["query"], url_kw, base_k
                    )

                if not in_corpus:
                    plan["corpus_miss"] = True
                    plan["corpus_miss_site"] = url_kw

                if year and in_corpus:
                    year_filtered = [d for d in docs if d.metadata.get("year") == year]
                    if year_filtered:
                        docs = year_filtered
                        plan["year_filter"] = year
                    else:
                        print(f"[Planner] Year filter {year} removed all docs — relaxing.")

                if legal_pair and in_corpus:
                    term_a, term_b = legal_pair
                    extra, _ = _post_filter_by_url(
                        self.vectorstore, f"{q['query']} {term_b}", url_kw, base_k
                    )
                    seen_pc = {d.page_content for d in docs}
                    docs += [d for d in extra if d.page_content not in seen_pc]

            elif date:
                docs = filtered_search(
                    self.vectorstore, q["query"],
                    filters={"collection_date": date},
                    k=base_k,
                )
            else:
                docs = semantic_search(self.vectorstore, q["query"], k=base_k)

        # ── COMPARE ──────────────────────────────────────────────────────────
        elif query_type == "COMPARE":
            compare_misses: list[str] = []
            site_results: dict[str, list[Document]] = {}

            for sq in sub_queries:
                url_kw = sq.get("filters", {}).get("url", "")
                if url_kw:
                    site_docs, in_corpus = _post_filter_by_url(
                        self.vectorstore, sq["query"], url_kw, base_k
                    )
                    if not in_corpus:
                        compare_misses.append(url_kw)
                        print(f"[Planner] COMPARE: '{url_kw}' not in corpus — skipping.")
                        continue
                    site_results[url_kw] = site_docs
                else:
                    site_results[sq["query"]] = semantic_search(
                        self.vectorstore, sq["query"], k=base_k
                    )

            # FIX: balanced interleaving instead of flat append
            docs = _interleave_site_docs(site_results)
            if compare_misses:
                plan["compare_misses"] = compare_misses

        # ── FALLBACK ─────────────────────────────────────────────────────────
        else:
            docs = semantic_search(self.vectorstore, question, k=K_BY_TYPE["SIMPLE"])

        print(f"[Planner] Retrieved {len(docs)} documents.")
        return docs, plan