import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.document_loader import load_policy_segments
from ingestion.metadata_fuser import fuse_and_save
from embeddings.embedding_manager import get_embedding_model
from embeddings.vector_store import build_vector_store_enriched

BASE_DIR         = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SANITIZED_DIR    = os.path.join(BASE_DIR, "data", "raw", "sanitized_policies")
PRETTY_DIR       = os.path.join(BASE_DIR, "data", "raw", "pretty_print")
DOCS_DIR         = os.path.join(BASE_DIR, "data", "raw", "documentation")
PROCESSED_DIR    = os.path.join(BASE_DIR, "data", "processed")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vector_store")

if __name__ == "__main__":
    print("=== Agentic Privacy Auditor — Ingestion Pipeline ===\n")

    # 1. Load and parse raw segments
    segments = load_policy_segments(SANITIZED_DIR)

    # 2. Enrich with expert annotations and policy metadata
    fuse_and_save(
        segments=segments,
        pretty_print_dir=PRETTY_DIR,
        documentation_dir=DOCS_DIR,
        output_dir=PROCESSED_DIR,
    )

    # 3. Clear any existing vector store and rebuild
    if os.path.exists(VECTOR_STORE_DIR):
        print(f"\nClearing existing vector store at: {VECTOR_STORE_DIR}")
        shutil.rmtree(VECTOR_STORE_DIR)

    embeddings = get_embedding_model()
    build_vector_store_enriched(PROCESSED_DIR, embeddings, VECTOR_STORE_DIR)

    print("\n  Ingestion complete!")
