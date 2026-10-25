#!/usr/bin/env python3
"""Freeze source-backed evaluation cases and enforce family separation."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def compile_evaluation(project):
    root = project / "corpus_mia"
    state = json.loads((root / "crawl_state.json").read_text())
    if state["queue"] or state.get("in_flight"):
        raise ValueError("Finish the current crawl before freezing evaluation source families")
    docs = {d["document_id"]: d for d in rows(root / "documents.jsonl")}
    catalog = {d["document_id"]: d for d in rows(root / "catalog.jsonl")}
    if set(docs) != set(catalog):
        raise ValueError("Catalog is stale; rebuild it after the final extraction")
    persona = json.loads((project / "config" / "persona.json").read_text())
    specifications = json.loads((project / "reviews" / "evaluation_specifications.json").read_text())
    seed = rows(project / "datasets" / "seed_records.jsonl")
    seed_families = set()
    for item in seed:
        current = catalog[item["source_document_id"]]["source_family_id"]
        if item["source_family_id"] != current:
            raise ValueError("Seed family provenance is stale; rebuild the seed after the catalog")
        seed_families.add(current)
    reserved = {}
    output = []
    for spec in specifications["source_backed"]:
        matches = [d for d in docs.values() if d["url"].endswith(spec["source_suffix"])]
        if len(matches) != 1:
            raise ValueError(f"Missing/non-unique evaluation source: {spec['source_suffix']}")
        doc = matches[0]
        by_id = {b["block_id"]: b for b in doc["blocks"]}
        selected = [by_id[i] for i in spec["block_ids"]]
        if any(b["role"] != "body_candidate" for b in selected):
            raise ValueError(f"Evaluation evidence includes an excluded block: {spec['id']}")
        family = catalog[doc["document_id"]]["source_family_id"]
        if family in seed_families:
            raise ValueError(f"Evaluation source family overlaps training seed: {spec['id']}")
        if family in reserved and reserved[family] != spec["split"]:
            raise ValueError("One family cannot be shared between validation and test")
        reserved[family] = spec["split"]
        evidence = "\n\n".join(b["text"] for b in selected)
        output.append({"case_id": spec["id"], "split": spec["split"], "category": "source_backed", "messages": [{"role": "system", "content": persona["system_prompt"]}, {"role": "user", "content": spec["question"]}], "reference_checks": spec["checks"], "source_document_id": doc["document_id"], "source_family_id": family, "source_url": doc["url"], "source_sha256": doc["source_sha256"], "source_block_ids": spec["block_ids"], "reference_evidence": evidence, "evidence_sha256": hashlib.sha256(evidence.encode()).hexdigest(), "review_status": "agent_authored_initial_rubric"})
    for spec in specifications["behavioral"]:
        messages = spec.get("messages", [{"role": "user", "content": spec.get("question", "")}])
        output.append({"case_id": spec["id"], "split": "behavioral", "category": spec["category"], "messages": [{"role": "system", "content": persona["system_prompt"]}] + messages, "reference_checks": spec["checks"], "source_document_id": None, "source_family_id": None, "review_status": "agent_authored_initial_rubric"})
    if len({r["case_id"] for r in output}) != len(output):
        raise ValueError("Duplicate evaluation IDs")
    # References are kept outside the model input. RAG evaluation may retrieve
    # held-out sources; supervised training must not consume their target text.
    directory = project / "evaluation"
    directory.mkdir(exist_ok=True)
    (directory / "cases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
    assignments = []
    for item in catalog.values():
        family = item["source_family_id"]
        split = reserved.get(family, "train_seed_source" if family in seed_families else "unassigned_pending_review")
        assignments.append({"document_id": item["document_id"], "url": item["url"], "source_family_id": family, "split": split, "content_approval": "individual_passages_only; full documents remain unreviewed"})
    (directory / "source_split_manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in assignments))
    with (directory / "score_sheet.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model_label", "case_id", "response", "temporal_consistency_0_4", "source_faithfulness_0_4", "attribution_0_4", "persona_voice_0_4", "conversation_0_4", "calibration_0_4", "critical_failure", "notes"])
        for item in output:
            writer.writerow(["", item["case_id"]] + [""] * 9)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "historical_cutoff": persona["historical_as_of"], "cases": len(output), "categories": Counter(r["category"] for r in output), "case_splits": Counter(r["split"] for r in output), "reserved_families": reserved, "source_document_assignments": Counter(r["split"] for r in assignments), "seed_source_family_overlap": 0, "model_responses_generated": 0, "limitations": ["Source family holdout does not establish absence from the base model's pretraining.", "Rubrics are agent-authored and require empirical checking; no benchmark scores are claimed.", "Automatic grouping can miss alternate translations or partial overlaps; review reservations before scaling training data."]}
    (directory / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    compile_evaluation(parser.parse_args().project)
