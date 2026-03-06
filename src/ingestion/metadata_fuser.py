import os
import csv
import json

# ── OPP-115 category → slug mapping ──────────────────────────────────────────
# Row[2] in the pretty_print CSV is the OPP-115 annotation category label.
# We normalise it to a lowercase slug stored as metadata["section"].

_CATEGORY_SLUG = {
    "First Party Collection/Use":          "first_party_collection",
    "Third Party Sharing/Collection":       "third_party_sharing",
    "User Choice/Control":                  "user_choice",
    "User Access, Edit and Deletion":       "user_access",
    "Data Retention":                       "data_retention",
    "Data Security":                        "data_security",
    "Policy Change":                        "policy_change",
    "Do Not Track":                         "do_not_track",
    "International and Specific Audiences": "international",
    "Other":                                "other",
}

def _slug(category: str) -> str:
    return _CATEGORY_SLUG.get(category.strip(), "other")


def load_policy_metadata(documentation_dir: str) -> dict:
    metadata_path = os.path.join(documentation_dir, "policies_opp115.csv")
    policy_meta = {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            uid = row[0].strip()
            collection_date = row[2].strip()
            policy_meta[uid] = {
                "url":              row[1].strip(),
                "collection_date":  collection_date,
                "last_updated":     row[3].strip(),
                # FIX: derive integer year for hard metadata filtering
                "year": int(collection_date[:4]) if collection_date and collection_date[:4].isdigit() else None,
            }

    print(f"Loaded metadata for {len(policy_meta)} policies.")
    return policy_meta


def load_pretty_print(pretty_print_dir: str, policy_filename: str) -> tuple[dict, dict]:
    """
    Returns:
        segment_annotations : {seg_id: [annotation_text, ...]}
        segment_sections    : {seg_id: primary_section_slug}
    """
    name_without_prefix = "_".join(policy_filename.split("_")[1:])
    csv_path = os.path.join(pretty_print_dir, name_without_prefix.replace(".html", ".csv"))

    segment_annotations: dict[int, list[str]] = {}
    segment_sections: dict[int, str] = {}

    if not os.path.exists(csv_path):
        return segment_annotations, segment_sections

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                seg_id = int(row[1].strip())
            except ValueError:
                continue

            # row[2] = OPP-115 category label
            category = row[2].strip() if len(row) > 2 else ""
            pretty_text = row[3].strip()

            if pretty_text:
                segment_annotations.setdefault(seg_id, []).append(pretty_text)
            if category and seg_id not in segment_sections:
                # Use the first (primary) category seen for this segment
                segment_sections[seg_id] = _slug(category)

    return segment_annotations, segment_sections


def fuse_and_save(
    segments: list[dict],
    pretty_print_dir: str,
    documentation_dir: str,
    output_dir: str,
) -> int:
    os.makedirs(output_dir, exist_ok=True)
    policy_meta = load_policy_metadata(documentation_dir)

    policies: dict[str, list[dict]] = {}
    for seg in segments:
        policies.setdefault(seg["source_file"], []).append(seg)

    enriched_total = 0

    for filename, segs in policies.items():
        policy_uid = filename.split("_")[0]
        policy_id  = filename.replace(".html", "")
        meta = policy_meta.get(policy_uid, {
            "url": "unknown", "collection_date": "unknown",
            "last_updated": "unknown", "year": None,
        })
        pp_map, section_map = load_pretty_print(pretty_print_dir, filename)

        enriched_segments = []
        for seg in segs:
            sid = seg["segment_id"]
            interpretations = pp_map.get(sid, [])
            enriched_segments.append({
                "policy_id":        policy_id,
                "policy_uid":       policy_uid,
                "segment_id":       sid,
                "text":             seg["text"],
                "expert_summary":   " | ".join(interpretations),
                "section":          section_map.get(sid, "other"),   # ← NEW: OPP-115 category slug
                "has_annotations":  len(interpretations) > 0,
                "annotation_count": len(interpretations),
                "url":              meta["url"],
                "collection_date":  meta["collection_date"],
                "last_updated":     meta["last_updated"],
                "year":             meta["year"],                     # ← NEW: integer year
                "source_file":      filename,
            })

        out_path = os.path.join(output_dir, f"{policy_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched_segments, f, indent=2, ensure_ascii=False)

        enriched_total += len(enriched_segments)

    print(f"   Fusion complete. {enriched_total} enriched segments saved to: {output_dir}")
    print(f"   JSON files created: {len(policies)}")
    return enriched_total