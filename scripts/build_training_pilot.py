#!/usr/bin/env python3
"""Freeze the reviewed pilot inputs without importing a model or training library.

The notebook may embed this stdlib-only module and call
``validate_training_payload(train_jsonl_bytes, lock)``. Evaluation material is
inspected only by the local lock builder; the portable guard contains hashes.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata


SCHEMA_VERSION = 1
MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
BASELINE_PATH = "evaluation/runs/kaggle_361242522/run_manifest.json"
REQUIRED_PATHS = (
    "datasets/train_seed.jsonl", "datasets/seed_records.jsonl",
    "config/persona.json", "reviews/seed_specifications.json",
    "reviews/seed_evidence_lock.json", "reviews/evaluation_specifications.json",
    "corpus_mia/catalog.jsonl", "evaluation/cases.jsonl",
    "evaluation/source_split_manifest.jsonl",
)
SPAN_WORDS = 16


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value):
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")).encode("utf-8"))


def parse_jsonl(payload):
    """Decode exact uploaded bytes; blank rows and non-object rows are errors."""
    result = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid JSONL row {number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {number} must be an object")
        result.append(row)
    return result


def normalized_words(text):
    """Case/punctuation/Unicode-insensitive words for copy detection."""
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def text_sha256(text):
    return sha256_bytes(" ".join(normalized_words(text)).encode("utf-8"))


def span_hashes(text, size=SPAN_WORDS):
    words = normalized_words(text)
    return {sha256_bytes(" ".join(words[start:start + size]).encode("utf-8"))
            for start in range(len(words) - size + 1)}


def validate_contamination(rows, guard):
    """Reject exact excluded texts and copied contiguous passages, without text references."""
    forbidden_text = set(guard["text_sha256"])
    forbidden_spans = set(guard["span_sha256"])
    for row in rows:
        for message in row["messages"]:
            if message["role"] == "system":
                continue  # The shared persona is deliberately identical in evaluation.
            text = message["content"]
            if text_sha256(text) in forbidden_text:
                raise ValueError(f"Excluded evaluation/generated text in {row['example_id']}")
            if span_hashes(text, guard["span_words"]) & forbidden_spans:
                raise ValueError(f"Excluded evaluation/generated passage in {row['example_id']}")


def validate_training_rows(rows, lock):
    """Check reviewed content, persona, complete alternating turns, and exclusions."""
    dataset = lock["dataset"]
    if len(rows) != dataset["records"]:
        raise ValueError("Training row count differs from the frozen pilot")
    expected = dataset["example_sha256"]
    seen = set()
    for row in rows:
        if set(row) != {"example_id", "messages"}:
            raise ValueError("Training rows may contain only example_id and messages")
        key = row["example_id"]
        if not isinstance(key, str) or key in seen:
            raise ValueError("Invalid or duplicate training example ID")
        seen.add(key)
        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 != 1:
            raise ValueError(f"Incomplete training conversation: {key}")
        for index, message in enumerate(messages):
            role = "system" if index == 0 else "user" if index % 2 else "assistant"
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message.get("role") != role
                    or not isinstance(message.get("content"), str)
                    or not message["content"].strip()):
                raise ValueError(f"Invalid training message/role: {key}")
        if sha256_bytes(messages[0]["content"].encode("utf-8")) != lock["persona"]["system_prompt_sha256"]:
            raise ValueError(f"Persona differs from the frozen pilot: {key}")
        if canonical_sha256(row) != expected.get(key):
            raise ValueError(f"Training example differs from the reviewed pilot: {key}")
    if seen != set(expected):
        raise ValueError("Training example IDs differ from the frozen pilot")
    validate_contamination(rows, lock["contamination_guard"])
    return rows


def validate_training_payload(payload, lock):
    """Notebook entry point: return only rows from the byte-identical reviewed JSONL."""
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported pilot lock schema")
    if not isinstance(payload, bytes):
        raise TypeError("Pass the original train_seed.jsonl bytes")
    if sha256_bytes(payload) != lock["dataset"]["sha256"]:
        raise ValueError("Training dataset checksum differs from the frozen pilot")
    return validate_training_rows(parse_jsonl(payload), lock)


def read_rows(path):
    return parse_jsonl(path.read_bytes())


def unique_index(rows, key, label):
    output = {row[key]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"Duplicate {label}")
    return output


def benchmark_inventory(project):
    # Inventory all saved artifacts, including prose reviews and source prompts.
    # Only text found in known response columns is treated as a generated answer.
    return sorted(path.relative_to(project).as_posix()
                  for path in (project / "evaluation").rglob("*") if path.is_file())


def generated_answers(project, paths):
    answers = set()
    response_keys = {"response", "answer", "generated_text", "prompt_only_response", "provided_source_response"}

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in response_keys and isinstance(child, str) and child.strip():
                    answers.add(child)
                elif key == "messages" and isinstance(child, list):
                    for message in child:
                        if message.get("role") == "assistant" and message.get("content", "").strip():
                            answers.add(message["content"])
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for name in paths:
        if not name.startswith("evaluation/runs/"):
            continue
        path = project / name
        if path.suffix == ".jsonl":
            visit(read_rows(path))
        elif path.suffix == ".json":
            visit(json.loads(path.read_bytes()))
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as stream:
                visit(list(csv.DictReader(stream)))
    return answers


def build_lock(project, expected_records=500):
    """Validate local provenance/exclusions and return a deterministic lock dict."""
    project = Path(project)
    payload = (project / "datasets/train_seed.jsonl").read_bytes()
    training = parse_jsonl(payload)
    if len(training) != expected_records:
        raise ValueError(f"Pilot requires the existing {expected_records} reviewed examples")
    records = read_rows(project / "datasets/seed_records.jsonl")
    record_index = unique_index(records, "example_id", "reviewed example ID")
    if len(records) != len(training):
        raise ValueError("Training rows and reviewed seed records differ")
    persona = json.loads((project / "config/persona.json").read_bytes())
    reviewed_lock = json.loads((project / "reviews/seed_evidence_lock.json").read_bytes())
    for name, path in [("specifications_sha256", "reviews/seed_specifications.json"),
                       ("persona_sha256", "config/persona.json")]:
        if reviewed_lock[name] != sha256_bytes((project / path).read_bytes()):
            raise ValueError(f"Reviewed evidence lock is stale: {path}")
    assignments = unique_index(read_rows(project / "evaluation/source_split_manifest.jsonl"),
                               "document_id", "source assignment")
    catalog = unique_index(read_rows(project / "corpus_mia/catalog.jsonl"),
                           "document_id", "catalog document")
    if set(assignments) != set(catalog):
        raise ValueError("Source split manifest and catalog document inventories differ")
    family_splits = {}
    for key, assignment in assignments.items():
        if assignment["source_family_id"] != catalog[key]["source_family_id"]:
            raise ValueError("Source family assignment differs from catalog")
        family = assignment["source_family_id"]
        if family in family_splits and family_splits[family] != assignment["split"]:
            raise ValueError("Source family assigned to conflicting splits")
        family_splits[family] = assignment["split"]
    train_families, train_documents = set(), set()
    evidence_examples = {}
    for row in training:
        key = row["example_id"]
        record = record_index.get(key, {})
        if record.get("messages") != row["messages"]:
            raise ValueError(f"Training content differs from reviewed provenance: {key}")
        if record.get("approved_for_seed_training") is not True or record.get("review_status") != "agent_reviewed_seed":
            raise ValueError(f"Training example lacks reviewed approval: {key}")
        if record.get("persona_id") != persona["persona_id"] or record.get("historical_as_of") != persona["historical_as_of"]:
            raise ValueError("Reviewed record persona differs from the current persona")
        doc = record["source_document_id"]
        family = record["source_family_id"]
        if (doc not in assignments or assignments[doc]["split"] != "train_seed_source"
                or assignments[doc]["source_family_id"] != family):
            raise ValueError("Training source is outside its reserved training family")
        if (record["source_sha256"] != catalog[doc]["source_sha256"]
                or record["source_url"] != catalog[doc]["url"]):
            raise ValueError("Reviewed source provenance differs from catalog")
        if sha256_bytes(record["source_evidence"].encode("utf-8")) != record["evidence_sha256"]:
            raise ValueError("Reviewed source evidence checksum mismatch")
        evidence_examples[key] = {field: record[field] for field in
                                  ("source_sha256", "evidence_sha256", "source_block_ids", "source_date_evidence")}
        evidence_examples[key]["messages_sha256"] = sha256_bytes(json.dumps(
            row["messages"], sort_keys=True, ensure_ascii=False).encode("utf-8"))
        train_families.add(family)
        train_documents.add(doc)
    if evidence_examples != reviewed_lock["examples"]:
        raise ValueError("Reviewed source/message evidence lock differs from the seed records")
    cases = read_rows(project / "evaluation/cases.jsonl")
    unique_index(cases, "case_id", "evaluation case ID")
    eval_families = set()
    excluded_texts = set()
    for case in cases:
        family = case.get("source_family_id")
        if family:
            doc = case["source_document_id"]
            if (doc not in assignments or assignments[doc]["source_family_id"] != family
                    or assignments[doc]["split"] != case["split"]
                    or case["source_sha256"] != catalog[doc]["source_sha256"]):
                raise ValueError("Evaluation source disagrees with the reserved family split")
            eval_families.add(family)
        if case.get("reference_evidence"):
            if sha256_bytes(case["reference_evidence"].encode("utf-8")) != case["evidence_sha256"]:
                raise ValueError("Evaluation evidence checksum mismatch")
            excluded_texts.add(case["reference_evidence"])
        # Never embed these texts in a notebook; portable checks use hashes only.
        excluded_texts.update(m["content"] for m in case["messages"] if m["role"] != "system")
    if train_families & eval_families:
        raise ValueError("Training and evaluation source families overlap")
    inventory = benchmark_inventory(project)
    answers = generated_answers(project, inventory)
    excluded_texts.update(answers)
    all_spans = set()
    for text in excluded_texts:
        all_spans.update(span_hashes(text))
    baseline = json.loads((project / BASELINE_PATH).read_bytes())
    identity = baseline["identity"]
    if identity["model_id"] != MODEL_ID or identity["revision"] != MODEL_REVISION:
        raise ValueError("Saved baseline checkpoint differs from the intended pilot")
    if (identity["quantization"] != "bnb_nf4_double" or identity["enable_thinking"] is not False
            or identity["generation"]["do_sample"] is not False
            or identity["generation"]["num_beams"] != 1
            or identity["generation"]["max_new_tokens"] != 512):
        raise ValueError("Saved baseline no longer matches NF4 non-thinking greedy 512 settings")
    inputs = sorted(set(REQUIRED_PATHS) | set(inventory))
    lock = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "Training compatibility preflight and small pilot; no scaling claim",
        "dataset": {
            "path": "datasets/train_seed.jsonl", "sha256": sha256_bytes(payload),
            "records": len(training),
            "example_sha256": {row["example_id"]: canonical_sha256(row) for row in training},
            "multi_turn_records": sum(len(row["messages"]) > 3 for row in training),
            "training_payload_keys": ["example_id", "messages"],
        },
        "persona": {"persona_id": persona["persona_id"], "historical_as_of": persona["historical_as_of"],
                    "path": "config/persona.json", "sha256": sha256_bytes((project / "config/persona.json").read_bytes()),
                    "system_prompt_sha256": sha256_bytes(persona["system_prompt"].encode("utf-8"))},
        "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "comparison": {"baseline_manifest": BASELINE_PATH, "baseline_run_id": baseline["run_id"],
                       "quantization": identity["quantization"], "dtype": identity["dtype"],
                       "enable_thinking": identity["enable_thinking"], "seed": identity["seed"],
                       "max_input_tokens": identity["max_input_tokens"],
                       "generation": identity["generation"], "experiment": identity["experiment"],
                       "chat_template_sha256": identity["chat_template_sha256"],
                       "packages": identity["packages"]},
        "provenance": {"review_status": "agent_reviewed_seed", "independent_historian_review": False,
                       "source_documents": sorted(train_documents), "source_families": sorted(train_families),
                       "answer_origins": dict(Counter(row["answer_origin"] for row in records)),
                       "source_family_overlap": 0},
        "evaluation": {"case_ids": [row["case_id"] for row in cases],
                       "source_families": sorted(eval_families),
                       "development_only_case_ids": identity["experiment"]["original_case_ids"],
                       "fresh_other_works_and_failure_types_required_before_scaling": True,
                       "locally_archived_unique_generated_answers": len(answers),
                       "earlier_raw_answer_archives_complete": False,
                       "exclusion_policy": "Only frozen train_seed.jsonl may enter training; all evaluation/**, generated benchmark answers, review references and unreviewed corpus text are excluded.",
                       "guard_limitations": "Copy guards catch normalized exact texts and contiguous 16-word spans, not paraphrases. Earlier runs have local inspection/review metadata but no complete local raw-answer archive; the training allowlist and reviewed dataset checksum exclude those artifacts."},
        "contamination_guard": {"normalization": "NFKC, casefold, Unicode word tokens, whitespace join",
                                "span_words": SPAN_WORDS,
                                "text_sha256": sorted({text_sha256(text) for text in excluded_texts}),
                                "span_sha256": sorted(all_spans)},
        "input_sha256": {name: sha256_bytes((project / name).read_bytes()) for name in inputs},
        "evaluation_artifact_inventory": inventory,
    }
    validate_training_payload(payload, lock)
    return lock


def validate_lock(project, lock):
    """Check that every frozen input and all exclusion/provenance decisions still match."""
    current = build_lock(project, expected_records=lock["dataset"]["records"])
    if current != lock:
        raise ValueError("Pilot lock is stale; inspect changed inputs before explicitly rebuilding it")
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-lock", action="store_true", help="Explicitly freeze current validated inputs")
    args = parser.parse_args()
    path = args.project / "training/pilot_lock.json"
    if args.write_lock:
        lock = build_lock(args.project)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    else:
        lock = validate_lock(args.project, json.loads(path.read_bytes()))
    print(json.dumps({"records": lock["dataset"]["records"], "dataset_sha256": lock["dataset"]["sha256"],
                      "lock_sha256": canonical_sha256(lock), "training_source_family_overlap": 0}, indent=2))


if __name__ == "__main__":
    main()
