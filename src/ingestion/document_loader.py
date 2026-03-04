import os
from bs4 import BeautifulSoup


def load_policy_segments(sanitized_policies_dir: str) -> list[dict]:
    
    html_files = [f for f in os.listdir(sanitized_policies_dir) if f.endswith(".html")]
    print(f"Found {len(html_files)} policy files.")

    all_segments = []

    for filename in html_files:
        policy_id = filename.replace(".html", "")
        filepath = os.path.join(sanitized_policies_dir, filename)

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw_html = f.read()

        soup = BeautifulSoup(raw_html, "lxml")
        full_text = soup.get_text(separator=" ")

        for seg_index, segment_text in enumerate(full_text.split("|||")):
            cleaned = segment_text.strip()
            if not cleaned:
                continue

            all_segments.append({
                "policy_id": policy_id,
                "segment_id": seg_index,
                "text": cleaned,
                "source_file": filename,
            })

    print(f"Total segments extracted: {len(all_segments)}")
    return all_segments
