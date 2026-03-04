import os
import csv
import json

def load_policy_metadata(documentation_dir: str) -> dict:
    metadata_path = os.path.join(documentation_dir, "policies_opp115.csv")
    policy_meta = {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            policy_meta[row[0].strip()] = {
                "url": row[1].strip(),
                "collection_date": row[2].strip(),
                "last_updated": row[3].strip(),
            }

    print(f"Loaded metadata for {len(policy_meta)} policies.")
    return policy_meta

def load_pretty_print(pretty_print_dir: str, policy_filename: str) -> dict:
    # '1017_sci-news.com.html' -> 'sci-news.com.csv'
    name_without_prefix = "_".join(policy_filename.split("_")[1:])
    csv_path = os.path.join(pretty_print_dir, name_without_prefix.replace(".html", ".csv"))

    segment_annotations: dict[int, list[str]] = {}

    if not os.path.exists(csv_path):
        return segment_annotations

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                seg_id = int(row[1].strip())
            except ValueError:
                continue
            pretty_text = row[3].strip()
            if pretty_text:
                segment_annotations.setdefault(seg_id, []).append(pretty_text)

    return segment_annotations

def fuse_and_save(
    segments: list[dict],
    pretty_print_dir: str,
    documentation_dir: str,
    output_dir: str,
) -> int:
    os.makedirs(output_dir, exist_ok=True)
    policy_meta = load_policy_metadata(documentation_dir)

    # Group segments by source file
    policies: dict[str, list[dict]] = {}
    for seg in segments:
        policies.setdefault(seg["source_file"], []).append(seg)

    enriched_total = 0

    for filename, segs in policies.items():
        policy_uid = filename.split("_")[0]
        policy_id = filename.replace(".html", "")
        meta = policy_meta.get(policy_uid, {
            "url": "unknown",
            "collection_date": "unknown",
            "last_updated": "unknown",
        })
        pp_map = load_pretty_print(pretty_print_dir, filename)

        enriched_segments = []
        for seg in segs:
            interpretations = pp_map.get(seg["segment_id"], [])
            enriched_segments.append({
                "policy_id":        policy_id,
                "policy_uid":       policy_uid,
                "segment_id":       seg["segment_id"],
                "text":             seg["text"],
                "expert_summary":   " | ".join(interpretations),
                "has_annotations":  len(interpretations) > 0,
                "annotation_count": len(interpretations),
                "url":              meta["url"],
                "collection_date":  meta["collection_date"],
                "last_updated":     meta["last_updated"],
                "source_file":      filename,
            })

        out_path = os.path.join(output_dir, f"{policy_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched_segments, f, indent=2, ensure_ascii=False)

        enriched_total += len(enriched_segments)

    print(f"   Fusion complete. {enriched_total} enriched segments saved to: {output_dir}")
    print(f"   JSON files created: {len(policies)}")
    return enriched_total
