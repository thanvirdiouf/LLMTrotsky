#!/usr/bin/env python3
"""Apply logged, conservative text repairs without changing raw evidence."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

PUNCTUATION = {"8211": "–", "8212": "—", "8216": "‘", "8217": "’", "8220": "“", "8221": "”", "8230": "…"}
MALFORMED_ENTITY = re.compile(r"(?<!&)(?:#(8211|8212|8216|8217|8220|8221|8230);)")


def repair_text(text, url, corrections):
    changes = []
    counts = Counter(MALFORMED_ENTITY.findall(text))
    for number, count in counts.items():
        changes.append({"rule_id": "malformed_numeric_punctuation_entity", "before": "#" + number + ";", "after": PUNCTUATION[number], "count": count})
    text = MALFORMED_ENTITY.sub(lambda m: PUNCTUATION[m[1]], text)
    for item in corrections:
        if item["source_url"] != url:
            continue
        count = text.count(item["before"])
        if count:
            text = text.replace(item["before"], item["after"])
            changes.append({"rule_id": item["id"], "before": item["before"], "after": item["after"], "count": count, "reason": item["reason"]})
    return text, changes


def clean(project):
    root = project / "corpus_mia"
    corrections = json.loads((project / "reviews/text_corrections.json").read_text())
    splits = {}
    manifest = project / "evaluation/source_split_manifest.jsonl"
    if manifest.exists():
        splits = {r["document_id"]: r["split"] for r in map(json.loads, manifest.read_text().splitlines())}
    counts = Counter()
    words = 0
    docs_count = 0
    repeats = 0
    output = root / "cleaned_candidates.jsonl.tmp"
    edits = root / "text_repairs.jsonl.tmp"
    with (root / "documents.jsonl").open() as source, output.open("w") as destination, edits.open("w") as log:
        for line in source:
            doc = json.loads(line)
            if doc["page_kind"] != "text_candidate":
                continue
            blocks = []
            previous = None
            quality = list(doc["quality_flags"])
            for block in doc["blocks"]:
                if block["role"] not in {"body_candidate", "heading"}:
                    continue
                text, changes = repair_text(block["text"], doc["url"], corrections)
                for change in changes:
                    counts[change["rule_id"]] += change["count"]
                    log.write(json.dumps({"document_id": doc["document_id"], "source_url": doc["url"], "source_sha256": doc["source_sha256"], "block_id": block["block_id"], **change}, ensure_ascii=False) + "\n")
                if len(text) > 100 and text == previous:
                    quality.append("consecutive_duplicate_block_requires_review")
                    repeats += 1
                previous = text
                blocks.append({"source_block_id": block["block_id"], "role": block["role"], "source_text": block["text"], "text": text, "repairs": changes})
            text = "\n\n".join(b["text"] for b in blocks)
            docs_count += 1
            words += len(text.split())
            destination.write(json.dumps({"document_id": doc["document_id"], "source_url": doc["url"], "title": doc["title"], "source_sha256": doc["source_sha256"], "composition_date_interval": doc.get("composition_date_interval"), "metadata_fields_raw": doc["metadata_fields_raw"], "split": splits.get(doc["document_id"], "unassigned_pending_review"), "text": text, "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "blocks": blocks, "quality_flags": sorted(set(quality)), "authorship_status": "unverified", "review_status": "pending", "approved_for_training": False}, ensure_ascii=False) + "\n")
    for item in corrections:
        if counts[item["id"]] != item["expected_occurrences"]:
            raise ValueError(f"Reviewed correction count changed for {item['id']}: {counts[item['id']]}")
    output.replace(root / "cleaned_candidates.jsonl")
    edits.replace(root / "text_repairs.jsonl")
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "documents": docs_count, "candidate_words_including_duplicates_and_unreviewed_material": words, "repair_counts": counts, "consecutive_duplicate_blocks_flagged_not_deleted": repeats, "approved_bulk_training_documents": 0, "limitations": ["This is a conservative cleaned candidate corpus, not a blanket authorship approval.", "Raw HTML and source text are preserved; corrections are logged.", "No speculative word-boundary reconstruction, stylistic rewriting, or automatic removal of repeated content was performed.", "Unlabelled editorial content, mixed authorship, quotations, and difficult OCR still require review."]}
    (root / "cleaning_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    clean(parser.parse_args().project)
