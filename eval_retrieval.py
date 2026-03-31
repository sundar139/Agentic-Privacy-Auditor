#!/usr/bin/env python3
"""
Retrieval Evaluation  â€”  Precision@K, Recall@K, MRR
=====================================================
Ground truth is derived directly from the OPP-115 expert annotations
embedded in every processed JSON file.  No LLM is needed at eval time;
only the embedding model + ChromaDB are exercised.

Two modes per query
-------------------
  semantic  : pure vector similarity (BAAI/bge-small-en-v1.5 + query expansion)
  section   : metadata-pre-filtered search (section queries only)

Usage
-----
  python eval_retrieval.py              # runs both modes, prints tables
  python eval_retrieval.py --save       # also writes eval_results.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from embeddings.embedding_manager import get_embedding_model   # noqa: E402
from embeddings.vector_store import load_vector_store         # noqa: E402
from retrieval.retriever import semantic_search, section_search  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
K_VALUES = [1, 3, 5, 10]

# â”€â”€ Eval query suite â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# type "section" â†’ ground truth = all corpus docs where section == relevant_key
# type "policy"  â†’ ground truth = all docs where policy_id  == relevant_key
EVAL_QUERIES: list[dict] = [
    # â”€â”€ Section queries (one per non-"other" OPP-115 category) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    {
        "query": "What personal data is collected directly from users?",
        "type": "section", "relevant_key": "first_party_collection",
        "label": "First-party collection",
    },
    {
        "query": "Does the policy share data with advertisers or third parties?",
        "type": "section", "relevant_key": "third_party_sharing",
        "label": "Third-party sharing",
    },
    {
        "query": "How does the website protect user data and information security?",
        "type": "section", "relevant_key": "data_security",
        "label": "Data security",
    },
    {
        "query": "How long does the website retain user data?",
        "type": "section", "relevant_key": "data_retention",
        "label": "Data retention",
    },
    {
        "query": "Can users opt out or control how their personal data is used?",
        "type": "section", "relevant_key": "user_choice",
        "label": "User choice / opt-out",
    },
    {
        "query": "Can users access, correct, or delete their personal account data?",
        "type": "section", "relevant_key": "user_access",
        "label": "User access / deletion",
    },
    {
        "query": "How does the site notify users when the privacy policy changes?",
        "type": "section", "relevant_key": "policy_change",
        "label": "Policy change notification",
    },
    {
        "query": "Does the site honor Do Not Track browser signals from users?",
        "type": "section", "relevant_key": "do_not_track",
        "label": "Do Not Track",
    },
    {
        "query": "What international jurisdiction laws govern this privacy policy?",
        "type": "section", "relevant_key": "international",
        "label": "International / jurisdiction",
    },
    # â”€â”€ Policy identity queries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    {
        "query": "What does Amazon collect and how does it use customer data?",
        "type": "policy", "relevant_key": "105_amazon.com",
        "label": "Amazon",
    },
    {
        "query": "What personal information does Instagram collect from its users?",
        "type": "policy", "relevant_key": "135_instagram.com",
        "label": "Instagram",
    },
    {
        "query": "How does the New York Times handle subscriber and reader privacy?",
        "type": "policy", "relevant_key": "26_nytimes.com",
        "label": "New York Times",
    },
    {
        "query": "Describe Reddit data privacy practices and user information usage",
        "type": "policy", "relevant_key": "303_reddit.com",
        "label": "Reddit",
    },
    {
        "query": "How does Walmart use and protect customer personal information?",
        "type": "policy", "relevant_key": "348_walmart.com",
        "label": "Walmart",
    },
    {
        "query": "What privacy protections does Bank of America provide to customers?",
        "type": "policy", "relevant_key": "1300_bankofamerica.com",
        "label": "Bank of America",
    },
    {
        "query": "What does IMDb privacy notice say about data collection?",
        "type": "policy", "relevant_key": "21_imdb.com",
        "label": "IMDb",
    },
    {
        "query": "How does Ticketmaster use and protect event attendee data?",
        "type": "policy", "relevant_key": "1498_ticketmaster.com",
        "label": "Ticketmaster",
    },
    {
        "query": "What information does Fox Sports collect about sports fans?",
        "type": "policy", "relevant_key": "1708_foxsports.com",
        "label": "Fox Sports",
    },
    {
        "query": "How does the Washington Post handle reader and subscriber data?",
        "type": "policy", "relevant_key": "200_washingtonpost.com",
        "label": "Washington Post",
    },
    {
        "query": "What personal information does Steam collect from gamers?",
        "type": "policy", "relevant_key": "1470_steampowered.com",
        "label": "Steam",
    },
]


# â”€â”€ Ground truth â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_ground_truth(processed_dir: Path) -> tuple[dict, dict]:
    """
    Parse every processed JSON and return:
      section_gt  : {section_slug: set of (policy_id, segment_id)}
      policy_gt   : {policy_id:    set of (policy_id, segment_id)}
    """
    section_gt: dict[str, set] = defaultdict(set)
    policy_gt: dict[str, set] = defaultdict(set)

    for jf in sorted(processed_dir.glob("*.json")):
        segments = json.loads(jf.read_text(encoding="utf-8"))
        for seg in segments:
            doc_id = (seg["policy_id"], int(seg["segment_id"]))
            section_gt[seg.get("section", "other")].add(doc_id)
            policy_gt[seg["policy_id"]].add(doc_id)

    return dict(section_gt), dict(policy_gt)


# â”€â”€ Metric helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def extract_ids(docs) -> list[tuple]:
    """Return (policy_id, segment_id) tuples.  segment_id coerced to int."""
    ids = []
    for d in docs:
        pid = d.metadata.get("policy_id", "")
        raw_sid = d.metadata.get("segment_id", -1)
        try:
            sid = int(raw_sid)
        except (TypeError, ValueError):
            sid = -1
        ids.append((pid, sid))
    return ids


def precision_at_k(retrieved: list[tuple], relevant: set, k: int) -> float:
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / k if k else 0.0


def recall_at_k(retrieved: list[tuple], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[tuple], relevant: set) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _mean(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


# â”€â”€ Core evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def evaluate(vs, section_gt: dict, policy_gt: dict) -> list[dict]:
    """
    Run all queries; for each return a result dict with per-K metrics.

    Two retrieval modes are exercised:
      semantic_search : used for all queries          (tests pure embedding quality)
      section_search  : used for section queries only (tests metadata-enhanced retrieval)
    """
    K_MAX = max(K_VALUES)
    results = []

    for q in EVAL_QUERIES:
        q_type = q["type"]
        key = q["relevant_key"]
        query_text = q["query"]

        relevant = (section_gt if q_type == "section" else policy_gt).get(key, set())

        # â”€â”€ Mode 1: pure semantic (all queries) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        sem_docs = semantic_search(vs, query_text, k=K_MAX)
        sem_ids = extract_ids(sem_docs)

        # â”€â”€ Mode 2: section_search (section queries only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        sec_ids: list[tuple] = []
        if q_type == "section":
            sec_docs = section_search(vs, query_text, key, k=K_MAX)
            sec_ids = extract_ids(sec_docs)

        row: dict = {
            "label": q["label"],
            "type": q_type,
            "relevant_key": key,
            "n_relevant": len(relevant),
            "rr_semantic": reciprocal_rank(sem_ids, relevant),
            "rr_section": reciprocal_rank(sec_ids, relevant) if sec_ids else None,
        }
        for k in K_VALUES:
            row[f"sem_P@{k}"] = precision_at_k(sem_ids, relevant, min(k, len(sem_ids)))
            row[f"sem_R@{k}"] = recall_at_k(sem_ids, relevant, min(k, len(sem_ids)))
            if sec_ids:
                row[f"sec_P@{k}"] = precision_at_k(sec_ids, relevant, min(k, len(sec_ids)))
                row[f"sec_R@{k}"] = recall_at_k(sec_ids, relevant, min(k, len(sec_ids)))
        results.append(row)

    return results


# â”€â”€ Reporting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def print_table(results: list[dict]) -> dict:
    """Print formatted tables and return aggregate dict."""
    section_rows = [r for r in results if r["type"] == "section"]
    policy_rows = [r for r in results if r["type"] == "policy"]

    print("\n" + "â•" * 80)
    print("  PER-QUERY RESULTS")
    print("â•" * 80)
    hdr = f"{'Query':<32} {'Type':<4} {'|Rel|':>6}  {'RR':>5}  {'P@1':>5} {'P@3':>5} {'P@5':>5} {'P@10':>5}  {'R@5':>6}"
    print(hdr)
    print("â”€" * 80)
    for r in results:
        rr = r["rr_semantic"]
        print(
            f"  {r['label']:<30} {r['type'][:3].upper():<4} {r['n_relevant']:>6}"
            f"  {rr:>5.2f}"
            f"  {r['sem_P@1']:>5.2f} {r['sem_P@3']:>5.2f} {r['sem_P@5']:>5.2f} {r['sem_P@10']:>5.2f}"
            f"  {r['sem_R@5']:>6.3f}"
        )

    # -- Aggregates
    aggregate = {}
    print("\n" + "â•" * 80)
    for group_label, rows, mode in [
        ("SECTION queries â€” semantic_search (no metadata filter)", section_rows, "sem"),
        ("SECTION queries â€” section_search  (metadata pre-filter)", section_rows, "sec"),
        ("POLICY queries  â€” semantic_search", policy_rows, "sem"),
        ("ALL queries     â€” semantic_search", results, "sem"),
    ]:
        # rr key names use full words ("semantic"/"section") in the data dict
        rr_key = "rr_semantic" if mode == "sem" else "rr_section"
        if mode == "sec" and all(r.get(f"sec_P@5") is None for r in rows):
            continue   # section_search data not available
        pfx = mode
        valid_rows = [r for r in rows if r.get(f"{pfx}_P@5") is not None]
        if not valid_rows:
            continue

        mrr = _mean([r[rr_key] for r in valid_rows if r.get(rr_key) is not None])
        print(f"\n  â”€â”€ {group_label}  (n={len(valid_rows)}) â”€â”€")
        print(f"  {'Metric':<14} {'K=1':>7} {'K=3':>7} {'K=5':>7} {'K=10':>7}   MRR")
        print("  " + "â”€" * 54)
        p_vals = {k: _mean([r[f"{pfx}_P@{k}"] for r in valid_rows]) for k in K_VALUES}
        r_vals = {k: _mean([r[f"{pfx}_R@{k}"] for r in valid_rows]) for k in K_VALUES}
        p_row = "  " + f"{'Precision@K':<14}" + "".join(f"{p_vals[k]:>7.3f}" for k in K_VALUES)
        r_row = "  " + f"{'Recall@K':<14}" + "".join(f"{r_vals[k]:>7.3f}" for k in K_VALUES)
        print(p_row + f"   {mrr:.3f}")
        print(r_row)

        agg_key = f"{group_label.split('â€”')[0].strip().split()[0].lower()}_{pfx}"
        aggregate[agg_key] = {
            **{f"P@{k}": p_vals[k] for k in K_VALUES},
            **{f"R@{k}": r_vals[k] for k in K_VALUES},
            "MRR": mrr,
        }

    print("\n" + "â•" * 80)
    return aggregate


# â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser(description="RAG retrieval evaluation")
    parser.add_argument("--save", action="store_true",
                        help="write detailed results to eval_results.json")
    args = parser.parse_args()

    print("Loading embedding model â€¦")
    embed = get_embedding_model()
    persist_dir = str(ROOT / "data" / "vector_store")
    print("Loading vector store â€¦")
    vs = load_vector_store(embed, persist_dir)

    print(f"Building ground truth from {PROCESSED_DIR} â€¦")
    section_gt, policy_gt = build_ground_truth(PROCESSED_DIR)

    # Print ground-truth corpus stats
    print(f"\n  Ground truth corpus")
    print(f"  {'section':<30} {'n_segments':>10}")
    print("  " + "â”€" * 42)
    for slug in sorted(section_gt):
        print(f"  {slug:<30} {len(section_gt[slug]):>10}")
    print(f"\n  Total policies: {len(policy_gt)}")
    print(f"  Total segments: {sum(len(v) for v in policy_gt.values())}\n")

    print("Running retrieval â€¦  (this takes ~30 s on CPU)")
    results = evaluate(vs, section_gt, policy_gt)
    aggregate = print_table(results)

    if args.save:
        out = ROOT / "eval_results.json"
        with open(out, "w") as f:
            json.dump({"per_query": results, "aggregate": aggregate}, f, indent=2)
        print(f"\n  Detailed results saved â†’ {out}")


if __name__ == "__main__":
    main()

