from langchain_chroma import Chroma
from langchain_core.documents import Document

# ── Section keyword → OPP-115 metadata slug ──────────────────────────────────
_SECTION_KEYWORDS: dict[str, str] = {
    "security":        "data_security",
    "data security":   "data_security",
    "retention":       "data_retention",
    "data retention":  "data_retention",
    "sharing":         "third_party_sharing",
    "third party":     "third_party_sharing",
    "third-party":     "third_party_sharing",
    "collection":      "first_party_collection",
    "first party":     "first_party_collection",
    "first-party":     "first_party_collection",
    "user choice":     "user_choice",
    "opt out":         "user_choice",
    "opt-out":         "user_choice",
    "access":          "user_access",
    "deletion":        "user_access",
    "policy change":   "policy_change",
    "do not track":    "do_not_track",
    "dnt":             "do_not_track",
}

# L2 distance threshold for normalised embeddings (normalize_embeddings=True).
# score < 0.75 ≈ cosine_sim > 0.72 — reasonably relevant.
# Docs above this threshold are returned only as a fallback so corpus-miss
# logic can still fire; they are NOT treated as confident matches.
_RELEVANCE_THRESHOLD = 0.75


_LEGAL_SYNONYMS: dict[str, list[str]] = {
    "pii":       ["personal information", "personally identifiable", "registration data", "user data", "contact details"],
    "dnt":       ["do not track", "browser signals", "opt-out of tracking"],
    "encrypt":   ["secure", "protect", "safeguard", "SSL", "TLS", "encryption"],
    "delete":    ["remove", "erase", "purge", "right to erasure"],
    "retention": ["how long", "stored", "keep data", "data storage period"],
    "gdpr":      ["data protection", "right to access", "right to erasure", "data subject"],
    "ccpa":      ["California", "opt out of sale", "do not sell"],
}


def expand_query(query: str) -> str:
    """
    N2 fix: append synonym expansions for known legal/privacy terms so the
    embedding space covers policy language that avoids the formal acronym.
    Only adds terms not already present in the query.
    """
    q_lower = query.lower()
    additions: list[str] = []
    for term, synonyms in _LEGAL_SYNONYMS.items():
        if term in q_lower:
            for syn in synonyms:
                if syn.lower() not in q_lower:
                    additions.append(syn)
    if additions:
        expanded = query + " " + " ".join(additions[:4])  # cap expansion to 4 terms
        print(f"[Retriever] Query expanded: '{query}' → '{expanded}'")
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
    candidates = vectorstore.similarity_search_with_score(expand_query(query), k=max(k * 4, 20))
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
    """
    # Step 1
    raw = filtered_search(vectorstore, query, {"section": section_slug}, k=k * 3 if url_kw else k)
    if url_kw:
        results = [d for d in raw if url_kw.lower() in d.metadata.get("url", "").lower()][:k]
    else:
        results = raw[:k]

    if results:
        return results

    # Step 2
    print(f"[Retriever] section_search: no metadata match for '{section_slug}'"
          f"{' @ ' + url_kw if url_kw else ''}, scanning content.")
    candidates = vectorstore.similarity_search(query, k=80)
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
            query, k=k, filter={"year": {"$eq": year}},
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