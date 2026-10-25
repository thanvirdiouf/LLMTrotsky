#!/usr/bin/env python3
"""Compile explicitly reviewed seed specifications into traceable chat JSONL."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

try:
    from .corpus_dates import cutoff_status
    from .fresh_holdout_guard import load_fresh_holdout_guard
except ImportError:
    from corpus_dates import cutoff_status
    from fresh_holdout_guard import load_fresh_holdout_guard


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def select_qa(rows, session, question_fragment):
    matches = [r for r in rows if r["source_url"].endswith(f"1937/dewey/session{session}.htm") and question_fragment.casefold() in r["question"].casefold()]
    if len(matches) != 1:
        raise ValueError(f"Expected one reviewed source question, found {len(matches)}: {question_fragment}")
    return matches[0]


def date_evidence(doc, cutoff):
    interval = doc.get("composition_date_interval")
    basis = "explicit_composition_metadata"
    kinds = {item["kind"] for item in doc.get("composition_date_evidence", [])}
    if kinds - {"explicit_composition_metadata"}:
        basis = "explicit_composition_metadata_and_source_dateline" if "explicit_composition_metadata" in kinds else "explicit_source_dateline"
    if interval is None and re.search(r"/1937/dewey/session\d+\.htm$", doc["url"]):
        hints = doc.get("heading_date_hints", [])
        if hints:
            interval = hints[0]["interval"]
            basis = "explicit_hearing_session_dateline"
    if cutoff_status(interval, cutoff) != "on_or_before_cutoff":
        raise ValueError(f"Source does not have verified pre-cutoff date evidence: {doc['url']}")
    return {"basis": basis, "interval": interval}


def source_blocks(doc, ids):
    if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids) or ids != sorted(set(ids)):
        raise ValueError("Source blocks must be a nonempty, ordered list of unique integer IDs")
    by_id = {b["block_id"]: b for b in doc["blocks"]}
    if any(i not in by_id for i in ids):
        raise ValueError("Source selection contains an unknown block ID")
    selected = [by_id[i] for i in ids]
    if any(b["role"] not in {"body_candidate", "heading"} for b in selected):
        raise ValueError(f"Reviewed selection touches excluded editorial/note/metadata blocks: {doc['url']}")
    return selected


def validate_source_record(item, doc, catalog_entry, cutoff):
    """Bind a reviewed record to authoritative extraction, not its own checksum."""
    for field, expected in [("source_document_id", doc["document_id"]), ("source_url", doc["url"]),
                            ("source_sha256", doc["source_sha256"]),
                            ("source_family_id", catalog_entry["source_family_id"])]:
        if item.get(field) != expected:
            raise ValueError("Reviewed source provenance mismatch: " + field)
    if catalog_entry["source_sha256"] != doc["source_sha256"]:
        raise ValueError("Catalog source checksum is stale")
    evidence = "\n\n".join(b["text"] for b in source_blocks(doc, item["source_block_ids"]))
    if item.get("source_evidence") != evidence:
        raise ValueError("Reviewed evidence differs from cached source blocks")
    if hashlib.sha256(evidence.encode()).hexdigest() != item.get("evidence_sha256"):
        raise ValueError("Reviewed evidence checksum mismatch")
    if item.get("source_date_evidence") != date_evidence(doc, cutoff):
        raise ValueError("Reviewed date evidence differs from the authoritative source")


def validate_verbatim(answer, evidence):
    if normalized(answer) not in normalized(evidence):
        raise ValueError("Verbatim answer is not an exact normalized substring of its source evidence")


def reserved_evaluation_families(documents, catalog, specifications):
    reserved = set()
    for spec in specifications["source_backed"]:
        matches = [d for d in documents.values() if d["url"].endswith(spec["source_suffix"])]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous reserved evaluation source: " + spec["source_suffix"])
        reserved.add(catalog[matches[0]["document_id"]]["source_family_id"])
    return reserved


def evidence_lock(spec_bytes, persona_bytes, records):
    examples = {}
    for row in records:
        if row["example_id"] in examples:
            raise ValueError("Duplicate example ID in reviewed records")
        examples[row["example_id"]] = {k: row[k] for k in ["source_sha256", "evidence_sha256", "source_block_ids", "source_date_evidence"]}
        examples[row["example_id"]]["messages_sha256"] = hashlib.sha256(json.dumps(row["messages"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"specifications_sha256": hashlib.sha256(spec_bytes).hexdigest(), "persona_sha256": hashlib.sha256(persona_bytes).hexdigest(), "examples": examples}


def conversation_messages(persona, question, answer, turns=None):
    turns = turns if turns is not None else [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    if not isinstance(turns, list) or len(turns) < 2 or len(turns) % 2:
        raise ValueError("Reviewed conversation must contain complete user/assistant pairs")
    output = [{"role": "system", "content": persona["system_prompt"]}]
    for i, turn in enumerate(turns):
        role = "user" if i % 2 == 0 else "assistant"
        if not isinstance(turn, dict) or turn.get("role") != role or not isinstance(turn.get("content"), str) or not turn["content"].strip():
            raise ValueError("Reviewed conversation roles must alternate with nonempty content")
        output.append({"role": role, "content": turn["content"]})
    if turns[0]["content"] != question or turns[-1]["content"] != answer:
        raise ValueError("Conversation question/answer summary differs from its first/last turns")
    return output


def compile_seed(project, review_update_note=None):
    root = project / "corpus_mia"
    out = project / "datasets"
    out.mkdir(exist_ok=True)
    specs_path = project / "reviews" / "seed_specifications.json"
    spec_bytes = specs_path.read_bytes()
    specs = json.loads(spec_bytes)
    persona = json.loads((project / "config" / "persona.json").read_text())
    docs = {d["url"]: d for d in read_rows(root / "documents.jsonl")}
    qa = read_rows(root / "qa_candidates.jsonl")
    catalog = {d["document_id"]: d for d in read_rows(root / "catalog.jsonl")} if (root / "catalog.jsonl").exists() else {}
    if {d["document_id"] for d in docs.values()} != set(catalog) or any(catalog[d["document_id"]]["source_sha256"] != d["source_sha256"] for d in docs.values()):
        raise ValueError("Catalog is missing or stale; rebuild it after extraction")
    evaluation_specs = json.loads((project / "reviews/evaluation_specifications.json").read_text())
    reserved = reserved_evaluation_families(docs, catalog, evaluation_specs)
    fresh_holdouts = load_fresh_holdout_guard(project, catalog)
    records = []
    for origin, items in [("verbatim_transcript", specs["native_dialogue"]), ("verbatim_source", specs.get("verbatim_passages", [])), ("adapted_source_grounded", specs["adapted_passages"] + specs.get("adapted_conversations", []))]:
        for item in items:
            source_question = None
            if "session" in item or "source_session" in item:
                candidate = select_qa(qa, item.get("session", item.get("source_session")), item["source_question_contains"])
                doc = docs[candidate["source_url"]]
                selected = source_blocks(doc, candidate["answer_block_ids"])
                source_question = candidate["question"]
                answer = item.get("answer", item.get("answer_span", candidate["answer"]))
                if origin == "verbatim_transcript":
                    validate_verbatim(answer, candidate["answer"])
            else:
                matches = [d for u, d in docs.items() if u.endswith(item["source_suffix"])]
                if len(matches) != 1:
                    raise ValueError(f"Non-unique reviewed document suffix: {item['source_suffix']}")
                doc = matches[0]
                selected = source_blocks(doc, item["block_ids"])
                answer = item["answer"]
            evidence = "\n\n".join(b["text"] for b in selected)
            family = catalog[doc["document_id"]]["source_family_id"]
            fresh_holdouts.check_source(doc["document_id"], family)
            if family in reserved:
                raise ValueError("Reviewed seed would consume a reserved evaluation family: " + doc["url"])
            if origin in {"verbatim_transcript", "verbatim_source"}:
                validate_verbatim(answer, evidence)
            raw = (root / doc["source_html"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != doc["source_sha256"]:
                raise ValueError("Raw source changed after extraction")
            temporal = date_evidence(doc, persona["historical_as_of"])
            record_id = "seed_" + hashlib.sha256((doc["url"] + "\n" + item["question"]).encode()).hexdigest()[:16]
            messages = conversation_messages(persona, item["question"], answer, item.get("turns"))
            if origin in {"verbatim_transcript", "verbatim_source"}:
                for message in messages:
                    if message["role"] == "assistant":
                        validate_verbatim(message["content"], evidence)
            records.append({
                "example_id": record_id,
                "messages": messages,
                "answer_origin": origin, "source_document_id": doc["document_id"], "source_url": doc["url"],
                "source_sha256": doc["source_sha256"], "source_block_ids": [b["block_id"] for b in selected],
                "source_evidence": evidence, "evidence_sha256": hashlib.sha256(evidence.encode()).hexdigest(),
                "source_question": source_question, "source_date_evidence": temporal,
                "source_family_id": family,
                "genre": item.get("genre", "hearing" if "session" in item or "source_session" in item else "unclassified"),
                "temporal_frame": item.get("temporal_frame", "Review note describes any adaptation of the source date to the persona date."),
                "persona_id": persona["persona_id"], "historical_as_of": persona["historical_as_of"],
                "review_method": specs["review_method"], "review_note": item.get("review_note", "Source attribution, question adaptation, and temporal context checked by coding agent."),
                "review_status": "agent_reviewed_seed", "approved_for_seed_training": True,
                "limitations": ["Not independently reviewed by a historian.", "Small bootstrap set; source claims and translated phrasing are not a universal historical gold standard."],
            })
    if len({r["example_id"] for r in records}) != len(records):
        raise ValueError("Duplicate seed questions")
    lock = evidence_lock(spec_bytes, (project / "config/persona.json").read_bytes(), records)
    lock_path = project / "reviews" / "seed_evidence_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        if not review_update_note:
            raise ValueError("Seed evidence/specifications changed: inspect and pass --review-update-note with the review reason before rebuilding")
        with (project / "reviews" / "seed_review_history.jsonl").open("a") as history:
            history.write(json.dumps({"updated_utc": datetime.now(timezone.utc).isoformat(), "reason": review_update_note, "previous_lock": json.loads(lock_path.read_text())}, ensure_ascii=False) + "\n")
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    for name, rows in [("seed_records.jsonl", records), ("train_seed.jsonl", [{"example_id": r["example_id"], "messages": r["messages"]} for r in records])]:
        temp = out / (name + ".tmp")
        temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        temp.replace(out / name)
    preview = ["# Agent-reviewed seed examples", "", "Small bootstrap set, not an independently validated historical benchmark. Full provenance is in seed_records.jsonl.", ""]
    for i, row in enumerate(records, 1):
        preview.extend([f"## {i}. {row['example_id']}", ""])
        for message in row["messages"][1:]:
            preview.extend([f"**{message['role'].capitalize()}:** {message['content']}", ""])
        preview.extend([f"Source: {row['source_url']} — {row['answer_origin']}.", ""])
    (out / "seed_preview.md").write_text("\n".join(preview))
    lengths = Counter("short_1_40_words" if len(r["messages"][-1]["content"].split()) <= 40 else "medium_41_120_words" if len(r["messages"][-1]["content"].split()) <= 120 else "long_over_120_words" for r in records)
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(), "examples": len(records), "multi_turn_conversations": sum(len(r["messages"]) > 3 for r in records), "answer_origins": Counter(r["answer_origin"] for r in records), "genres": Counter(r["genre"] for r in records), "final_answer_length_bands": lengths, "source_documents": len({r["source_document_id"] for r in records}), "source_families": len({r["source_family_id"] for r in records}), "largest_source_document_example_count": max(Counter(r["source_document_id"] for r in records).values(), default=0), "historical_cutoff": persona["historical_as_of"], "validation": "Source hashes, evidence-role boundaries, verbatim spans, pre-cutoff dates, reserved evaluation families, alternating conversation roles, and unique IDs verified. Adapted answers rely on the documented coding-agent review, not an automated entailment proof."}
    (out / "seed_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--review-update-note", help="Required explanation when explicitly revising a previously frozen seed review")
    args = parser.parse_args()
    compile_seed(args.project, args.review_update_note)
