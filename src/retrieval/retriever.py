from langchain_chroma import Chroma
from langchain_core.documents import Document

# ── Section keyword → OPP-115 metadata slug ──────────────────────────────────
_SECTION_KEYWORDS: dict[str, str] = {
    # Data security — Q4/Q16
    "security":               "data_security",
    "data security":          "data_security",
    "information security":   "data_security",
    "encrypt":                "data_security",
    "password":               "data_security",
    "safeguard":              "data_security",
    "protect":                "data_security",
    # Data retention — Q1
    "retention":              "data_retention",
    "data retention":         "data_retention",
    "how long":               "data_retention",
    "retain":                 "data_retention",
    "retention period":       "data_retention",
    "stored for":             "data_retention",
    # Third party sharing
    "sharing":                "third_party_sharing",
    "third party":            "third_party_sharing",
    "third-party":            "third_party_sharing",
    "share with":             "third_party_sharing",
    "disclose":               "third_party_sharing",
    # First party collection — Q11
    "collection":             "first_party_collection",
    "first party":            "first_party_collection",
    "first-party":            "first_party_collection",
    "personally identifiable":"first_party_collection",
    "pii":                    "first_party_collection",
    "personal information":   "first_party_collection",
    "what data":              "first_party_collection",
    "what information":       "first_party_collection",
    # User choice
    "user choice":            "user_choice",
    "opt out":                "user_choice",
    "opt-out":                "user_choice",
    # User access / deletion
    "access":                 "user_access",
    "deletion":               "user_access",
    "delete":                 "user_access",
    "remove my data":         "user_access",
    # Policy change
    "policy change":          "policy_change",
    # DNT
    "do not track":           "do_not_track",
    "dnt":                    "do_not_track",
}

# L2 distance threshold for normalised embeddings (normalize_embeddings=True).
# score < 0.75 ≈ cosine_sim > 0.72 — reasonably relevant.
# Docs above this threshold are returned only as a fallback so corpus-miss
# logic can still fire; they are NOT treated as confident matches.
_RELEVANCE_THRESHOLD = 0.75


_LEGAL_SYNONYMS: dict[str, list[str]] = {
    # acronym AND spelled-out form both as keys so expansion fires either way
    "pii":                           ["personal information", "personally identifiable", "registration data", "user data", "contact details", "account information"],
    "personally identifiable":       ["personal information", "registration data", "user data", "contact details", "account information"],
    "dnt":                           ["do not track", "browser signals", "opt-out of tracking"],
    "do not track":                  ["browser signals", "opt-out tracking", "tracking preference"],
    "encrypt":                       ["secure", "protect", "safeguard", "SSL", "TLS", "encryption"],
    "delete":                        ["remove", "erase", "purge", "deactivate", "cancel account", "right to erasure"],
    "retention":                     ["how long", "stored", "keep data", "data storage period"],
    "gdpr":                          ["data protection", "right to access", "right to erasure", "data subject"],
    "ccpa":                          ["California", "opt out of sale", "do not sell"],
    # Q2 fix: prioritise advertising-specific terms that appear in real policy text
    "opt out":                       ["advertising preferences", "network advertising initiative", "unsubscribe", "withdraw consent", "preference", "choice"],
    # Q2 fix: "tracking" expands to advertising-flavoured synonyms used by real policies
    "tracking":                      ["behavioral advertising", "advertising preferences", "interest-based advertising", "network advertising initiative", "targeted ads"],
    "third party":                   ["partners", "affiliates", "advertisers", "service providers", "vendors"],
}


def expand_query(query: str) -> str:
    """
    N2 fix: append synonym expansions for known legal/privacy terms so the
    embedding space covers policy language that avoids formal acronyms.
    Checks both acronym and spelled-out forms. Only adds terms not already present.
    """
    q_lower = query.lower()
    seen: set[str] = set()
    additions: list[str] = []
    for term, synonyms in _LEGAL_SYNONYMS.items():
        if term in q_lower:
            for syn in synonyms:
                syn_lower = syn.lower()
                if syn_lower not in q_lower and syn_lower not in seen:
                    seen.add(syn_lower)
                    additions.append(syn)
    if additions:
        expanded = query + " " + " ".join(additions[:6])  # cap at 6 to avoid token bloat
        print(f"[Retriever] Query expanded: '{query[:60]}' + {len(additions)} synonyms")
        return expanded
    return query


def detect_section_slug(text: str) -> str | None:
    """Return the first matching OPP-115 section slug found in `text`, or None."""
    lower = text.lower()
    for keyword, slug in _SECTION_KEYWORDS.items():
        if keyword in lower:
            return slug
    return None


def semantic_search(vectorstore: Chroma, query: str, k: int = 5) -> list[Document]:
    return vectorstore.similarity_search(expand_query(query), k=k)


def threshold_search(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
    threshold: float = _RELEVANCE_THRESHOLD,
) -> list[Document]:
    """
    Semantic search with a minimum relevance filter.
    Over-fetches then drops results whose L2 distance exceeds `threshold`,
    preventing confident answers when the site/topic isn't in the corpus.
    Falls back to top-k unfiltered so corpus-miss logic in the planner can fire.
    """
    candidates = vectorstore.similarity_search_with_score(expand_query(query), k=max(k * 6, 40))
    passed = [doc for doc, score in candidates if score < threshold][:k]
    return passed if passed else [doc for doc, _ in candidates[:k]]


def filtered_search(
    vectorstore: Chroma,
    query: str,
    filters: dict,
    k: int = 5,
) -> list[Document]:
    if not filters:
        return semantic_search(vectorstore, query, k=k)

    if len(filters) == 1:
        key, val = next(iter(filters.items()))
        where = {key: {"$eq": val}}
    else:
        where = {"$and": [{fk: {"$eq": fv}} for fk, fv in filters.items()]}

    return vectorstore.similarity_search(query, k=k, filter=where)


def section_search(
    vectorstore: Chroma,
    query: str,
    section_slug: str,
    k: int = 10,
    url_kw: str = "",
) -> list[Document]:
    """
    Section-targeted retrieval for 'quote the X section' queries.

    Step 1: Hard metadata filter on section slug, url_kw post-filter if given.
    Step 2: Keyword scan of page_content (url-scoped when url_kw is set).
    Step 3 (N7): URL-only fallback when url_kw is set — returns from the right
                 site even when section metadata is missing after re-ingestion.
    Step 4: Global unfiltered semantic fallback (last resort).

    Q11 fix: expand_query is applied upfront so PII→"personal information" and
    encrypt→"SSL TLS encryption" synonyms are active in all embedding lookups.
    """
    # Apply synonym expansion once — used for all embedding calls below
    expanded = expand_query(query)

    # Step 1
    raw = filtered_search(vectorstore, expanded, {"section": section_slug}, k=k * 3 if url_kw else k)
    if url_kw:
        results = [d for d in raw if url_kw.lower() in d.metadata.get("url", "").lower()][:k]
    else:
        results = raw[:k]

    if results:
        return results

    # Step 2
    print(f"[Retriever] section_search: no metadata match for '{section_slug}'"
          f"{' @ ' + url_kw if url_kw else ''}, scanning content.")
    candidates = vectorstore.similarity_search(expanded, k=150)
    keyword = section_slug.replace("_", " ")
    content_matched = [
        d for d in candidates
        if keyword in d.page_content.lower()
        and (not url_kw or url_kw.lower() in d.metadata.get("url", "").lower())
    ][:k]
    if content_matched:
        return content_matched

    # Step 3 (N7 fix)
    if url_kw:
        print(f"[Retriever] section_search: falling back to URL-only for '{url_kw}'.")
        url_only = [d for d in candidates if url_kw.lower() in d.metadata.get("url", "").lower()][:k]
        if url_only:
            return url_only

    # Step 4
    print("[Retriever] section_search: falling back to unfiltered semantic search.")
    return vectorstore.similarity_search(query, k=k)


def year_filtered_search(
    vectorstore: Chroma,
    query: str,
    year: int,
    k: int = 5,
) -> list[Document]:
    """
    Semantic search with a hard Chroma metadata pre-filter on integer year.
    Falls back to unfiltered search if no documents carry that year.
    """
    try:
        results = vectorstore.similarity_search(
            query, k=k, filter={"year": {"$eq": year}},  # type: ignore
        )
    except Exception as exc:
        print(f"[Retriever] year_filtered_search({year}) error: {exc}")
        results = []

    if not results:
        print(f"[Retriever] No docs for year={year} — falling back to unfiltered search.")
        results = vectorstore.similarity_search(query, k=k)

    return results


def multi_query_search(
    vectorstore: Chroma,
    queries: list[dict],
) -> list[Document]:
    seen: set[str] = set()
    results: list[Document] = []
    for q in queries:
        docs = (
            filtered_search(vectorstore, q["query"], q["filters"], k=q.get("k", 5))
            if q.get("filters")
            else semantic_search(vectorstore, q["query"], k=q.get("k", 5))
        )
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                results.append(doc)
    return results