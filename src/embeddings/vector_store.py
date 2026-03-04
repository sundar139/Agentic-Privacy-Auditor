import os
import json
import glob

from langchain_chroma import Chroma
from langchain_core.documents import Document

def build_vector_store_enriched(
    processed_dir: str,
    embedding_model,
    persist_dir: str,
) -> Chroma:
    json_files = glob.glob(os.path.join(processed_dir, "*.json"))
    print(f"Found {len(json_files)} enriched JSON files.")

    documents = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            segs = json.load(f)

        for seg in segs:
            content = seg["text"]
            if seg.get("expert_summary"):
                content += f"\n\n[Expert Interpretation]: {seg['expert_summary']}"

            documents.append(Document(
                page_content=content,
                metadata={
                    "policy_id":        seg["policy_id"],
                    "policy_uid":       seg["policy_uid"],
                    "segment_id":       str(seg["segment_id"]),
                    "url":              seg["url"],
                    "collection_date":  seg["collection_date"],
                    "last_updated":     seg["last_updated"],
                    "has_annotations":  str(seg["has_annotations"]),
                    "annotation_count": str(seg["annotation_count"]),
                    "source_file":      seg["source_file"],
                },
            ))

    print(f"Total documents to embed: {len(documents)}")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    print(f"Vector store saved to: {persist_dir}")
    return vectorstore

def load_vector_store(embedding_model, persist_dir: str) -> Chroma:
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"No vector store found at: {persist_dir}\n"
            "Run `python src/ingestion/ingest.py` first."
        )
    print(f"Loading existing vector store from: {persist_dir}")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
    )
    print("Vector store loaded.")
    return vectorstore
