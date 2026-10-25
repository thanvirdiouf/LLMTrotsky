#!/usr/bin/env python3
"""Bounded HTTP client for a configured remote teacher or evaluation model."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

import requests

try:
    from .corpus_dates import cutoff_status
    from .fresh_holdout_guard import load_fresh_holdout_guard
    from .build_seed_dataset import validate_source_record, reserved_evaluation_families
except ImportError:
    from corpus_dates import cutoff_status
    from fresh_holdout_guard import load_fresh_holdout_guard
    from build_seed_dataset import validate_source_record, reserved_evaluation_families


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def review_key(item):
    return (item["source_sha256"], item["evidence_sha256"], tuple(item["source_block_ids"]), json.dumps(item["source_date_evidence"], sort_keys=True))


def generation_sources(records, assignments, cutoff, documents, catalog, approved_evidence, reserved_families, fresh_holdouts=None):
    unique = {}
    for item in records:
        if not (item.get("approved_for_generation") is True or item.get("approved_for_seed_training") is True):
            raise ValueError("Generation requires explicitly reviewed passages")
        document_id = item["source_document_id"]
        if fresh_holdouts is not None:
            fresh_holdouts.check_source(document_id, item["source_family_id"])
        assignment = assignments.get(document_id, {})
        if assignment.get("split") not in {"train", "train_seed_source"}:
            raise ValueError("Held-out or unassigned sources cannot be used for training generation")
        if document_id not in documents or document_id not in catalog:
            raise ValueError("Generation source is absent from the authoritative corpus")
        validate_source_record(item, documents[document_id], catalog[document_id], cutoff)
        if item["source_family_id"] in reserved_families:
            raise ValueError("Current evaluation specifications reserve this source family")
        if review_key(item) not in approved_evidence:
            raise ValueError("Generation passage is absent from the frozen evidence review")
        if assignment.get("source_family_id") != item["source_family_id"] or assignment.get("url") != item["source_url"]:
            raise ValueError("Source split manifest is stale or mismatched")
        if cutoff_status(item["source_date_evidence"]["interval"], cutoff) != "on_or_before_cutoff":
            raise ValueError("Generation source is outside the verified historical date boundary")
        if hashlib.sha256(item["source_evidence"].encode()).hexdigest() != item["evidence_sha256"]:
            raise ValueError("Generation evidence checksum mismatch")
        unique.setdefault((document_id, item["evidence_sha256"]), item)
    return list(unique.values())


def validate_generated(content, source, maximum):
    # Fences are presentation, not executable code. No eval or expression parsing.
    content = content.strip()
    if content.startswith("```json\n") and content.endswith("\n```"):
        content = content[8:-4]
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    examples = data.get("examples")
    if not isinstance(examples, list) or len(examples) > maximum:
        raise ValueError("Expected an examples array within the requested count")
    result = []
    seen = set()
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("Each generated example must be an object")
        question, answer = example.get("question"), example.get("answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError("Generated question and answer must be nonempty strings")
        identity = normalized(question).casefold()
        if identity in seen:
            raise ValueError("Duplicate generated questions")
        seen.add(identity)
        origin = example.get("answer_origin")
        if origin not in {"verbatim_source", "adapted_source_grounded"}:
            raise ValueError("Missing/invalid answer origin")
        spans = example.get("evidence_spans")
        if not isinstance(spans, list) or not spans or any(not isinstance(span, str) or len(span.split()) < 5 or normalized(span) not in normalized(source["source_evidence"]) for span in spans):
            raise ValueError("Generated evidence spans must match the supplied source")
        if origin == "verbatim_source" and normalized(answer) not in normalized(source["source_evidence"]):
            raise ValueError("Verbatim generated answer does not match source")
        for field in ["source_url", "source_sha256"]:
            if field in example and example[field] != source[field]:
                raise ValueError("Teacher changed immutable source provenance")
        result.append({"question": question.strip(), "answer": answer.strip(), "answer_origin": origin, "evidence_spans": spans,
                       **{key: source[key] for key in ["source_document_id", "source_url", "source_sha256", "source_block_ids", "source_family_id", "source_date_evidence", "evidence_sha256"]},
                       "review_status": "pending", "approved_for_training": False, "validation_status": "mechanical_checks_only; adapted claims still require source review"})
    return result


def chat_endpoint(base_url):
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Use an HTTP(S) API base URL without credentials, query parameters or fragments")
    return base_url.rstrip("/") + "/chat/completions"


def call_chat(session, base_url, payload, api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    response = session.post(chat_endpoint(base_url), json=payload, headers=headers, timeout=(15, 180), allow_redirects=False)
    if 300 <= response.status_code < 400:
        raise ValueError("Refusing inference-server redirect")
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    content = choice["message"].get("content")
    if not isinstance(content, str):
        raise ValueError("Server did not return text assistant content")
    # Do not store provider reasoning_content or request authorization headers.
    return content, choice.get("finish_reason"), data.get("usage")


def run(args):
    if args.limit < 1 or args.examples_per_source < 1 or args.max_tokens < 1:
        raise ValueError("Request and output limits must be positive")
    project = args.project
    persona = json.loads((project / "config/persona.json").read_text())
    options = json.loads(args.request_options.read_text()) if args.request_options else {}
    if not isinstance(options, dict):
        raise ValueError("Request options must be a JSON object")
    if set(options) & {"model", "messages", "max_tokens", "temperature", "stream", "n"}:
        raise ValueError("Request options cannot override the model, messages, output limit or request count")
    if args.mode == "generate":
        source_path = args.input or project / "datasets/seed_records.jsonl"
        assignments = {r["document_id"]: r for r in read_rows(project / "evaluation/source_split_manifest.jsonl")}
        documents = {r["document_id"]: r for r in read_rows(project / "corpus_mia/documents.jsonl")}
        catalog = {r["document_id"]: r for r in read_rows(project / "corpus_mia/catalog.jsonl")}
        lock = json.loads((project / "reviews/seed_evidence_lock.json").read_text())
        for key, path in [("specifications_sha256", "reviews/seed_specifications.json"), ("persona_sha256", "config/persona.json")]:
            if lock.get(key) != hashlib.sha256((project / path).read_bytes()).hexdigest():
                raise ValueError("Frozen review is stale; rebuild the reviewed seed before generation")
        approved_evidence = {review_key(item) for item in lock["examples"].values()}
        evaluation_specs = json.loads((project / "reviews/evaluation_specifications.json").read_text())
        reserved = reserved_evaluation_families(documents, catalog, evaluation_specs)
        fresh_holdouts = load_fresh_holdout_guard(project, catalog)
        sources = generation_sources(read_rows(source_path), assignments, persona["historical_as_of"], documents, catalog, approved_evidence, reserved, fresh_holdouts)
        for source in sources:
            doc = documents[source["source_document_id"]]
            raw = (project / "corpus_mia" / doc["source_html"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != doc["source_sha256"]:
                raise ValueError("Cached HTML changed after source extraction")
        instruction = (project / "prompts/generate_examples.md").read_text()
        instruction += '\nReturn exactly one JSON object with an "examples" array. Empty is valid when the source is unsuitable. Never set approval flags.'
        jobs = []
        for source in sources:
            user = {"historical_as_of": persona["historical_as_of"], "persona": persona["system_prompt"], "requested_examples": args.examples_per_source, "source": {k: source[k] for k in ["source_url", "source_sha256", "source_block_ids", "source_date_evidence"]}, "evidence_text": source["source_evidence"]}
            jobs.append((source["evidence_sha256"], [{"role": "system", "content": instruction}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}], source))
    else:
        source_path = args.input or project / "evaluation/cases.jsonl"
        jobs = [(r["case_id"], r["messages"], None) for r in read_rows(source_path)]
    if args.output.resolve() == source_path.resolve():
        raise ValueError("Output must not overwrite/append to the input dataset")
    if not args.dry_run and not args.base_url:
        raise ValueError("A remote API base URL is required; use --dry-run to prepare requests without model calls")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    with args.output.open("a+") as stream, requests.Session() as session:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        for line in stream:
            row = json.loads(line)
            if row.get("status") == "completed":
                completed.add(row["job_id"])
        calls = 0
        for item_id, messages, source in jobs:
            payload = {"model": args.model, "messages": messages, "temperature": args.temperature, "max_tokens": args.max_tokens, "stream": False, **options}
            signature = json.dumps({"mode": args.mode, "base_url": args.base_url, "payload": payload, "item_id": item_id}, sort_keys=True)
            job_id = hashlib.sha256(signature.encode()).hexdigest()
            if job_id in completed:
                continue
            if calls >= args.limit:
                break
            record = {"job_id": job_id, "item_id": item_id, "mode": args.mode, "model": args.model, "created_utc": datetime.now(timezone.utc).isoformat()}
            if args.dry_run:
                record.update({"status": "dry_run", "request": payload, "network_call": False})
            else:
                # One bounded request per item, no automatic retry of an uncertain
                # paid/inference request. A later invocation resumes completed IDs.
                content, finish, usage = call_chat(session, args.base_url, payload, os.getenv(args.api_key_env, ""))
                record.update({"status": "completed", "response": content, "finish_reason": finish, "usage": usage})
                if source:
                    try:
                        if finish not in {"stop", None}:
                            raise ValueError("Teacher response was truncated or did not finish normally")
                        record["candidates"] = validate_generated(content, source, args.examples_per_source)
                    except (ValueError, TypeError, KeyError) as exc:
                        record["candidates"] = []
                        record["validation_error"] = str(exc)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            calls += 1
    print(json.dumps({"mode": args.mode, "dry_run": args.dry_run, "requests_prepared_or_completed": calls, "output": str(args.output), "automatic_training_approvals": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "evaluate"])
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", help="Remote API root including /v1 when required by the server")
    parser.add_argument("--api-key-env", default="TROTSKY_MODEL_API_KEY")
    parser.add_argument("--request-options", type=Path)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--examples-per-source", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())
