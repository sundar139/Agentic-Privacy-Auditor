from langchain_chroma import Chroma
from langchain_core.documents import Document


def semantic_search(vectorstore: Chroma, query: str, k: int = 5) -> list[Document]:
    return vectorstore.similarity_search(query, k=k)


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
        where = {key: {"$eq": str(val)}}
    else:
        where = {"$and": [{k: {"$eq": str(v)}} for k, v in filters.items()]}

    return vectorstore.similarity_search(query, k=k, filter=where)


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
