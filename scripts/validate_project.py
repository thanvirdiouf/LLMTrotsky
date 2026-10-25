#!/usr/bin/env python3
"""Check final artifact integrity, provenance, dates and held-out families."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

try:
    from .build_seed_dataset import date_evidence, normalized, validate_verbatim, validate_source_record, evidence_lock, reserved_evaluation_families
    from .clean_candidates import repair_text
    from .fresh_holdout_guard import load_fresh_holdout_guard, check_training_records
except ImportError:
    from build_seed_dataset import date_evidence, normalized, validate_verbatim, validate_source_record, evidence_lock, reserved_evaluation_families
    from clean_candidates import repair_text
    from fresh_holdout_guard import load_fresh_holdout_guard, check_training_records


def read_rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate(project):
    root = project / "corpus_mia"
    state = json.loads((root / "crawl_state.json").read_text())
    check(not state["queue"] and not state.get("in_flight"), "Crawl is not complete")
    downloads = list(read_rows(root / "downloads.jsonl"))
    check(len({r["url"] for r in downloads}) == len(downloads), "Download manifest has duplicate URLs")
    for row in downloads:
        raw = (root / row["raw_path"]).read_bytes()
        check(hashlib.sha256(raw).hexdigest() == row["sha256"], "Cached HTML hash mismatch: " + row["url"])
    docs = {d["document_id"]: d for d in read_rows(root / "documents.jsonl")}
    check(len(docs) == len(downloads), "Extraction and download counts differ")
    for doc in docs.values():
        check([b["block_id"] for b in doc["blocks"]] == list(range(len(doc["blocks"]))), "Non-contiguous source block IDs")
    catalog = {r["document_id"]: r for r in read_rows(root / "catalog.jsonl")}
    check(set(catalog) == set(docs), "Catalog is stale")
    splits = {r["document_id"]: r for r in read_rows(project / "evaluation/source_split_manifest.jsonl")}
    check(set(splits) == set(docs), "Split manifest is stale")
    persona = json.loads((project / "config/persona.json").read_text())
    seed = list(read_rows(project / "datasets/seed_records.jsonl"))
    frozen = json.loads((project / "reviews/seed_evidence_lock.json").read_text())
    check(frozen == evidence_lock((project / "reviews/seed_specifications.json").read_bytes(), (project / "config/persona.json").read_bytes(), seed), "Seed records, persona, or specifications differ from the frozen review")
    chat = list(read_rows(project / "datasets/train_seed.jsonl"))
    check(len(chat) == len({r["example_id"] for r in chat}), "Duplicate training chat example IDs")
    check({r["example_id"]: r["messages"] for r in seed} == {r["example_id"]: r["messages"] for r in chat}, "Chat export differs from reviewed seed records")
    seed_families = set()
    for row in seed:
        doc = docs[row["source_document_id"]]
        validate_source_record(row, doc, catalog[doc["document_id"]], persona["historical_as_of"])
        check(row["source_sha256"] == doc["source_sha256"], "Seed source hash mismatch")
        evidence = "\n\n".join(doc["blocks"][i]["text"] for i in row["source_block_ids"])
        check(evidence == row["source_evidence"], "Seed evidence differs from source blocks")
        check(hashlib.sha256(evidence.encode()).hexdigest() == row["evidence_sha256"], "Seed evidence checksum mismatch")
        date_evidence(doc, persona["historical_as_of"])
        if row["answer_origin"] in {"verbatim_transcript", "verbatim_source"}:
            for message in row["messages"]:
                if message["role"] == "assistant":
                    validate_verbatim(message["content"], evidence)
        family = catalog[doc["document_id"]]["source_family_id"]
        check(row["source_family_id"] == family, "Seed family metadata is stale")
        seed_families.add(family)
    fresh_holdouts = load_fresh_holdout_guard(project, catalog)
    check_training_records(seed, catalog, fresh_holdouts)
    cases = list(read_rows(project / "evaluation/cases.jsonl"))
    held_out = {r["source_family_id"] for r in cases if r["source_family_id"]}
    current_reserved = reserved_evaluation_families(docs, catalog, json.loads((project / "reviews/evaluation_specifications.json").read_text()))
    check(current_reserved == held_out, "Evaluation reservations changed; rebuild the evaluation artifacts")
    check(not current_reserved & seed_families, "Training seed overlaps current evaluation specifications")
    check(not held_out & seed_families, "Training/evaluation source-family leakage")
    for case in cases:
        if case["source_document_id"]:
            doc = docs[case["source_document_id"]]
            evidence = "\n\n".join(doc["blocks"][i]["text"] for i in case["source_block_ids"])
            check(evidence == case["reference_evidence"], "Evaluation evidence mismatch")
            check(hashlib.sha256(evidence.encode()).hexdigest() == case["evidence_sha256"], "Evaluation evidence checksum mismatch")
            for item in catalog.values():
                if item["source_family_id"] == case["source_family_id"]:
                    check(splits[item["document_id"]]["split"] == case["split"], "Whole-family holdout was not preserved")
    corrections = json.loads((project / "reviews/text_corrections.json").read_text())
    repair_counts = Counter()
    cleaned_count = 0
    for clean in read_rows(root / "cleaned_candidates.jsonl"):
        doc = docs[clean["document_id"]]
        check(not clean["approved_for_training"], "Bulk unreviewed document is unexpectedly training-approved")
        check(clean["split"] == splits[doc["document_id"]]["split"], "Cleaned-corpus split is stale")
        for block in clean["blocks"]:
            source = doc["blocks"][block["source_block_id"]]
            check(source["text"] == block["source_text"], "Cleaned block lost its original source text")
            text, edits = repair_text(source["text"], doc["url"], corrections)
            check(text == block["text"] and edits == block["repairs"], "Unlogged text repair")
            for edit in edits:
                repair_counts[edit["rule_id"]] += edit["count"]
        check(clean["text"] == "\n\n".join(b["text"] for b in clean["blocks"]), "Cleaned document/block text mismatch")
        check(hashlib.sha256(clean["text"].encode()).hexdigest() == clean["cleaned_text_sha256"], "Cleaned text checksum mismatch")
        cleaned_count += 1
    qa_count = 0
    for item in read_rows(root / "qa_candidates.jsonl"):
        doc = docs[item["document_id"]]
        evidence = "\n\n".join(doc["blocks"][i]["text"] for i in item["answer_block_ids"])
        validate_verbatim(item["answer"], evidence)
        qa_count += 1
    cleaning_audit = json.loads((root / "cleaning_audit.json").read_text())
    check(dict(repair_counts) == cleaning_audit["repair_counts"], "Cleaning and validation repair counts differ")
    report = {"validated_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "raw_html_files_verified": len(downloads), "documents": len(docs), "cleaned_candidate_documents_verified": cleaned_count, "qa_candidate_answer_spans_verified": qa_count, "seed_examples_verified": len(seed), "evaluation_cases_verified": len(cases), "seed_evaluation_family_overlap": 0, "fresh_holdout_manifests": list(fresh_holdouts.manifests), "seed_fresh_holdout_family_overlap": 0, "repairs_verified": repair_counts, "original_input_sha256": hashlib.sha256((project / "trotsky_corpus.txt").read_bytes()).hexdigest(), "checks": ["Raw HTML checksums", "Document/block references", "Seed and evaluation evidence checksums", "Verbatim source spans", "Seed temporal eligibility", "Whole-source-family holdout", "Cleaned text and logged repairs", "Chat export consistency"], "limits": "Integrity validation does not establish historical truth, complete authorship curation, or model quality."}
    (root / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    validate(parser.parse_args().project)
