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
    "FILTERED":  15,  # one site — need broad coverage; raised from 8 to surface section chunks
    "COMPARE":   6,   # per site — slightly raised from 5
    "AMBIGUOUS": 0,   # never retrieve, ask user first
}
_QUOTE_K_BOOST = 15  # verbatim-quote queries — raised from 10

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


def _llm_invoke(llm, prompt: str, _retries: int = 3, _base_delay: float = 8.0) -> str:
    """BUG #1 fix: exponential backoff for quota/rate-limit errors on LangChain LLM calls."""
    import time
    last_exc = None
    for attempt in range(_retries):
        try:
            raw = llm.invoke(prompt)
            return raw.content if hasattr(raw, "content") else raw
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_quota = any(c in msg for c in ("402", "429", "payment required", "too many requests", "rate limit"))
            is_server = any(c in msg for c in ("500", "502", "503", "504", "service unavailable"))
            if is_quota or is_server:
                delay = _base_delay * (2 ** attempt)
                print(f"[Planner] LLM error attempt {attempt+1}/{_retries} — "
                      f"waiting {delay:.0f}s. Error: {type(exc).__name__}")
                time.sleep(delay)
                continue
            raise
    # All retries exhausted
    msg = str(last_exc).lower()
    if any(c in msg for c in ("402", "429", "payment required", "too many requests", "rate limit")):
        raise RuntimeError(
            "⚠️ The HuggingFace inference service is temporarily unavailable "
            "(quota or rate limit reached). Please top up your HF credits or "
            "run locally with Ollama."
        ) from last_exc
    raise RuntimeError(
        "⚠️ The inference service returned a server error. Please try again."
    ) from last_exc


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

CRITICAL — COMPARE vs FILTERED:
  COMPARE requires EXACTLY TWO OR MORE DIFFERENT domain names. The word "and"
  NEVER makes a single-site question into a COMPARE.
  FAILING EXAMPLES that must route to FILTERED, not COMPARE:
    ✗ "Summarize security measures of washingtonpost.com and tell me if they encrypt passwords"
       → ONE site (washingtonpost.com) → FILTERED
    ✗ "Does nytimes.com collect data and share it with third parties?"
       → ONE site (nytimes.com) → FILTERED
    ✗ "What does reddit.com say about cookies and tracking?"
       → ONE site (reddit.com) → FILTERED
  CORRECT COMPARE examples (two distinct domains both present):
    ✓ "Compare nytimes.com and washingtonpost.com on data retention"  → COMPARE
    ✓ "How do reddit.com and twitter.com differ on cookies?"          → COMPARE

  VERIFY before outputting COMPARE: count distinct domain names in the question.
  If count < 2, output FILTERED.

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
    return any(kw in q for kw in (
        "quote", "exact wording", "verbatim", "exact text", "copy of",
        "first two sentences", "first sentence", "word for word", "exact words",
    ))


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
    candidates = threshold_search(vectorstore, query, k=250)  # large pool for URL post-filter
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

    # Known OPP-115 domain keywords (first segment of domain, lowercased).
    # If any of these appear in the question, the LLM must not classify as AMBIGUOUS.
    # Unambiguous domain tokens — safe to match as substrings (long or unique enough).
    _CORPUS_DOMAIN_HINTS = frozenset([
        "aol", "apple", "att", "cbsnews", "chase", "cnet", "comcast", "craigslist",
        "ebay", "espn", "facebook", "foxnews", "huffingtonpost", "imdb", "instagram",
        "linkedin", "mapquest", "mediafire", "microsoft", "msnbc", "nytimes",
        "paypal", "pinterest", "reddit", "salesforce", "scribd", "shutterstock",
        "snapchat", "spotify", "theatlantic", "ticketmaster", "tmz", "tripadvisor",
        "tumblr", "twitter", "usnews", "verizon", "vevo", "vimeo", "washingtonpost",
        "webmd", "whitepages", "wikia", "wikipedia", "wordpress", "yahoo", "yelp",
        "youtube", "accuweather", "bankofamerica", "bestbuy", "bing", "blogger",
        "booking", "businessinsider", "buzzfeed", "capitalone", "citibank",
        "classmates", "coldwellbanker", "costco", "creditkarma", "dailymotion",
        "dealnews", "deviantart", "digg", "directv", "dropbox", "drugstore",
        "ehow", "etsy", "expedia", "fanfiction", "fandango", "flickr", "foodnetwork",
        "foxsports", "gamefaqs", "gamespot", "genius", "gofundme", "goodreads",
        "groupon", "homedepot", "houzz", "hulu", "icloud", "iheartradio",
        "investopedia", "kmart", "kohls", "livestrong", "lowes", "macys",
        "mayoclinic", "merriam-webster", "metacritic", "opentable", "pandora",
        "photobucket", "quora", "realtor", "theweek", "tripadvisor", "walgreens",
        "wellsfargo", "sci-news",
    ])

    # Ambiguous tokens that are also common English words or substrings of other domains.
    # These require an explicit TLD suffix (e.g. "time.com") to match safely.
    _CORPUS_DOMAIN_HINTS_TLD_REQUIRED = frozenset([
        "time", "match", "target", "cars", "last", "weather", "discovery",
        "reference", "monster", "hotels", "mlb", "msn", "nba", "nfl", "nhl",
        "npr", "pbs", "bbc", "cnn", "aol", "irs", "tmz", "dictionary",
        "vikings", "genius",
    ])
    # TLD pattern for the above
    _TLD_PATTERN = re.compile(
        r"\b(" + "|".join(_CORPUS_DOMAIN_HINTS_TLD_REQUIRED) + r")\.(com|org|net|gov|edu|io)\b"
    )

    def _domain_in_question(self, question: str) -> str | None:
        """
        BUG #3 fix: use exact substring match only for unambiguous long tokens.
        Short/ambiguous tokens (time, cars, match…) require an explicit TLD suffix
        (time.com, cars.com) to avoid false-positive matches inside words like
        "nytimes" → "time" or "immediately" → "time".
        """
        q = question.lower()
        # First: check unambiguous domain hints as substrings
        for domain in self._CORPUS_DOMAIN_HINTS:
            if domain in q:
                return domain
        # Second: check ambiguous tokens only when followed by a TLD
        m = self._TLD_PATTERN.search(q)
        if m:
            return m.group(1)
        return None

    def classify(self, question: str) -> dict:
        raw = _llm_invoke(self.llm, _CLASSIFIER_PROMPT.format(question=question))
        try:
            plan = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            # Q15 fix: JSON parse failed — use domain-aware fallback instead of
            # blindly defaulting to SIMPLE. If a corpus domain is in the question,
            # return FILTERED so the answer isn't lost.
            domain_hit = self._domain_in_question(question)
            if domain_hit:
                print(f"[Planner] JSON parse failed — domain '{domain_hit}' found → FILTERED fallback.")
                return {
                    "type": "FILTERED",
                    "sub_queries": [{
                        "query": question,
                        "filters": {"url": domain_hit},
                        "k": K_BY_TYPE["FILTERED"],
                    }],
                    "reasoning": f"Fallback: classification failed but domain '{domain_hit}' found.",
                }
            print("[Planner] JSON parse failed — no domain found → SIMPLE fallback.")
            return _fallback_plan(question)

        if _has_hallucinated_urls(plan):
            print("[Planner] Hallucinated URLs — overriding to SIMPLE.")
            return _fallback_plan(question)

        # Q14 fix: if the LLM returned AMBIGUOUS but a real corpus domain is
        # present in the question, override to FILTERED. Domain presence always
        # wins over pronoun ambiguity detection.
        # Q6 fix: same override for SIMPLE — a query naming a real site should
        # always route to FILTERED even if the topic seems fictional/nonsensical.
        if plan.get("type") in ("AMBIGUOUS", "SIMPLE"):
            domain_hit = self._domain_in_question(question)
            if domain_hit:
                old_type = plan["type"]
                print(f"[Planner] {old_type} overridden → FILTERED (domain '{domain_hit}' found).")
                plan["type"] = "FILTERED"
                plan["sub_queries"] = [{
                    "query": question,
                    "filters": {"url": domain_hit},
                    "k": K_BY_TYPE["FILTERED"],
                }]
                plan["reasoning"] = f"Auto-corrected from {old_type}: domain '{domain_hit}' found in question."

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
        # Fix Q1/Q4/Q11/Q16: detect section for ALL queries — section_search
        # is used for any section-specific question, not just verbatim quotes.
        section_slug = detect_section_slug(question)

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