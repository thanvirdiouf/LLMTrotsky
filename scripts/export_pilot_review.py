#!/usr/bin/env python3
"""Export a validated, condition-blinded review of the frozen QLoRA pilot.

This is a standard-library-only postprocessing step. It never calls a model,
changes training inputs, or assigns quality scores. Give reviewers the CSV and
Markdown; keep the key and provenance manifest separate until scoring is done.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_CASES = 36
INPUTS_SHA256 = "3a7a38f679af94f27571248e88ba2a528129e5f613d6803a032b93c6ba6b37a0"
CASES_SHA256 = "036a898ac83a37058fc7770d9a5c43559a8f468f0ae05cd050b6763cfbd8b86f"
TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
GENERATION = {"do_sample": False, "eos_token_id": 248046,
              "max_new_tokens": 512, "num_beams": 1,
              "pad_token_id": 248044, "use_cache": True}
SCORE_COLUMNS = ["temporal_consistency_0_4", "source_faithfulness_0_4", "attribution_0_4",
                 "persona_voice_0_4", "conversation_0_4", "calibration_0_4"]
FLAG_COLUMNS = ["fabricated_quotation_or_citation_flag", "invented_intimate_memory_flag",
                "post_cutoff_hindsight_as_memory_flag", "speaker_or_editor_misattribution_flag"]
REVIEW_FIELDS = ["review_id", "pair_id", "answer_label", "case_id", "category", "split",
                 "messages", "response", "finish_reason", "token_count", "source_url",
                 "source_evidence", "reference_checks", *SCORE_COLUMNS,
                 "critical_failure", *FLAG_COLUMNS, "notes"]
RUBRIC = """Score each applicable dimension from 0 to 4: 0 = clear failure, 1 = major
problems, 2 = mixed, 3 = good with minor issues, 4 = fully supported and appropriate.
Leave an inapplicable dimension blank.

- **Temporal consistency:** respects 19 August 1940; distinguishes predictions and
  supplied hypotheticals from independently known outcomes; allows events known in 1940.
- **Source faithfulness:** answers accurately within the evidence and preserves
  conditions and uncertainty.
- **Attribution:** separates Trotsky, opponents, translators, editors and later commentary.
- **Persona voice:** coherent first-person historical viewpoint without forced speeches
  or generic encyclopedia narration.
- **Conversation:** follows the question, requested length and preceding turns.
- **Calibration:** admits limits without invented memories, quotes, citations or precision.

Flag critical failures separately: fabricated direct quotations/citations, invented
intimate historical memories, definite post-cutoff hindsight presented as memory,
and systematic speaker/editor misattribution. Mark each applicable flag and explain
it in notes; leave flags blank until reviewed. Do not let style averages hide them.

Checklists describe constraints, not mandatory wording. Reasoning conditionally from
a user-supplied later event can be valid. Historical fidelity does not require
agreement with Trotsky's claims; reference passages can be mistaken or partisan.
Inspect `finish_reason=length` answers for truncation before interpreting omissions.
No automatic quality scores are assigned.

These 36 previously inspected cases are development evidence. They are a holdout
from this project's supervised training, not proof of novel historical knowledge.
Add fresh evaluation across other works and failure types before scaling or broad claims.
"""


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode())


def _json(payload):
    def reject_constant(value):
        raise ValueError("Non-finite JSON number: " + value)
    return json.loads(payload, parse_constant=reject_constant)


def _jsonl(payload):
    rows = [_json(line) for line in payload.decode("utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected JSONL objects")
    return rows


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _load_inputs(inputs_path, cases_path):
    inputs_bytes, cases_bytes = inputs_path.read_bytes(), cases_path.read_bytes()
    if digest(inputs_bytes) != INPUTS_SHA256 or digest(cases_bytes) != CASES_SHA256:
        raise ValueError("Frozen evaluation input or reviewer case checksum mismatch")
    inputs, cases = _jsonl(inputs_bytes), _jsonl(cases_bytes)
    if (len(inputs) != EXPECTED_CASES or len(cases) != EXPECTED_CASES
            or len({row["case_id"] for row in inputs}) != EXPECTED_CASES
            or [row["case_id"] for row in inputs] != [row["case_id"] for row in cases]):
        raise ValueError("Expected all 36 distinct cases in frozen order")
    for item, case in zip(inputs, cases):
        if (set(item) != {"case_id", "category", "split", "messages"}
                or any(item[key] != case[key] for key in item)):
            raise ValueError("Model prompt and reviewer case differ")
        evidence = case.get("reference_evidence")
        if evidence is not None and digest(evidence.encode()) != case.get("evidence_sha256"):
            raise ValueError("Reviewer reference evidence checksum mismatch")
    return inputs, cases


def _load_comparison(path):
    comparison = _json(path.read_bytes())
    if not isinstance(comparison, dict):
        raise ValueError("Expected comparison manifest object")
    generation = comparison.get("generation")
    if (not _is_sha256(comparison.get("identity_sha256"))
            or comparison.get("conditions") != ["baseline", "adapter"]
            or comparison.get("evaluation_inputs_sha256") != INPUTS_SHA256
            or comparison.get("chat_template_sha256") != TEMPLATE_SHA256
            or comparison.get("enable_thinking") is not False
            or not isinstance(generation, dict)
            or any(type(generation.get(key)) is not type(value) or generation[key] != value
                   for key, value in GENERATION.items())
            or comparison.get("generation_sha256") != fingerprint(generation)):
        raise ValueError("Comparison identity, prompts, template or decoding checksum mismatch")
    return comparison


def _load_condition(path, condition, inputs, comparison):
    expected = {case["case_id"]: case for case in inputs}
    rows = _jsonl(path.read_bytes())
    seen = {}
    for row in rows:
        key = row.get("case_id")
        case = expected.get(key)
        if (case is None or key in seen or row.get("condition") != condition
                or row.get("identity_sha256") != comparison["identity_sha256"]
                or row.get("generation_sha256") != comparison["generation_sha256"]
                or row.get("prompt_sha256") != fingerprint(case["messages"])
                or row.get("category") != case["category"] or row.get("split") != case["split"]
                or not _is_sha256(row.get("input_sha256"))
                or not isinstance(row.get("response"), str)):
            raise ValueError(f"Unknown, duplicate or mismatched {condition} response: {key}")
        tokens = row.get("token_ids")
        if (not isinstance(tokens, list) or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)
                or type(row.get("token_count")) is not int or row["token_count"] != len(tokens)
                or len(tokens) > GENERATION["max_new_tokens"]
                or GENERATION["eos_token_id"] in tokens[:-1]
                or row.get("finish_reason") != ("stop" if tokens[-1] == GENERATION["eos_token_id"] else "length")
                or (row["finish_reason"] == "length" and len(tokens) != GENERATION["max_new_tokens"])):
            raise ValueError(f"Invalid token count or stop evidence for {condition}: {key}")
        seen[key] = row
    if set(seen) != set(expected):
        raise ValueError(f"Incomplete {condition} evaluation: expected all 36 cases")
    return seen


def _assignment(seed, case_id):
    """Stable per-case assignment independent of row order and global random state."""
    bit = hashlib.sha256(f"{seed}\0answer-label\0{case_id}".encode()).digest()[0] & 1
    return ("baseline", "adapter") if bit == 0 else ("adapter", "baseline")


def _literal(text):
    # Indentation preserves source/answer prose literally, including Markdown fences.
    return "\n".join("    " + line for line in str(text).splitlines()) + "\n"


def export_review(run_dir, *, inputs_path=PROJECT / "training/pilot_evaluation_inputs.jsonl",
                  cases_path=PROJECT / "evaluation/cases.jsonl", output_dir=None, seed=20261023):
    """Validate all 72 responses before publishing new, blank review artifacts.

    Existing output directories are refused to protect scored copies. A response
    hash cannot authenticate model output: input and journal hashes here detect
    accidental mixups, and raw response token IDs remain in the original run.
    """
    if type(seed) is not int:
        raise ValueError("Review seed must be an integer")
    run_dir, inputs_path, cases_path = Path(run_dir), Path(inputs_path), Path(cases_path)
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "review"
    if output_dir.exists():
        raise FileExistsError("Review directory exists; preserve scored copies and choose a new output directory")
    inputs, cases = _load_inputs(inputs_path, cases_path)
    comparison = _load_comparison(run_dir / "comparison_manifest.json")
    by_condition = {condition: _load_condition(run_dir / f"{condition}_responses.jsonl", condition,
                                              inputs, comparison)
                    for condition in ["baseline", "adapter"]}
    for case in inputs:
        key = case["case_id"]
        if by_condition["baseline"][key]["input_sha256"] != by_condition["adapter"][key]["input_sha256"]:
            raise ValueError("Baseline and adapter tokenized input hashes differ: " + key)
    ordered = sorted(cases, key=lambda case: digest(f"{seed}\0case-order\0{case['case_id']}".encode()))
    reviews, key_rows = [], []
    markdown = ["# Pilot answer review\n", RUBRIC,
                "Review A and B independently before comparing their scores. Labels are assigned\n"
                "separately for each case. Keep the condition key separate until scoring is complete.\n"
                "Reference evidence and checklists below were withheld from model prompts and training.\n"]
    for index, case in enumerate(ordered, 1):
        pair_id = f"P{index:03d}"
        markdown += [f"## {pair_id}: {case['case_id']}\n", f"Category: {case['category']}; split: {case['split']}.\n",
                     "### Conversation\n"]
        for message in case["messages"]:
            markdown += [f"**{message['role']}**\n", _literal(message["content"])]
        markdown += ["### Review-only evidence and checks\n"]
        if case.get("source_url"):
            markdown += ["Source: " + case["source_url"] + "\n", _literal(case.get("reference_evidence", ""))]
        markdown += ["\n".join("- " + check for check in case["reference_checks"]) + "\n"]
        for label, condition in zip(["A", "B"], _assignment(seed, case["case_id"])):
            row = by_condition[condition][case["case_id"]]
            review_id = pair_id + label
            review = {"review_id": review_id, "pair_id": pair_id, "answer_label": label,
                      "case_id": case["case_id"], "category": case["category"], "split": case["split"],
                      "messages": json.dumps(case["messages"], ensure_ascii=False), "response": row["response"],
                      "finish_reason": row["finish_reason"], "token_count": row["token_count"],
                      "source_url": case.get("source_url") or "", "source_evidence": case.get("reference_evidence") or "",
                      "reference_checks": " | ".join(case["reference_checks"])}
            reviews.append(review)
            key_rows.append({"review_id": review_id, "pair_id": pair_id, "answer_label": label,
                             "case_id": case["case_id"], "condition": condition,
                             "identity_sha256": row["identity_sha256"], "prompt_sha256": row["prompt_sha256"],
                             "response_sha256": digest(row["response"].encode())})
            markdown += [f"### Answer {label} ({review_id})\n",
                         f"Finish reason: {row['finish_reason']}; output tokens: {row['token_count']}.\n",
                         _literal(row["response"]),
                         "Scores (0–4 or blank): temporal ___; source ___; attribution ___; voice ___; conversation ___; calibration ___.\n",
                         "Critical failure ___; quote/citation ___; intimate memory ___; hindsight ___; misattribution ___.\n",
                         "Notes: ________________________________________________________________\n"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pilot-review-", dir=output_dir.parent))
    try:
        for name, fields, rows in [("pilot_review.csv", REVIEW_FIELDS, reviews),
                                   ("pilot_review_key.csv", list(key_rows[0]), key_rows)]:
            with (temporary / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        (temporary / "pilot_review.md").write_text("\n".join(markdown), encoding="utf-8")
        source_paths = {"comparison_manifest.json": run_dir / "comparison_manifest.json",
                        "baseline_responses.jsonl": run_dir / "baseline_responses.jsonl",
                        "adapter_responses.jsonl": run_dir / "adapter_responses.jsonl",
                        "pilot_evaluation_inputs.jsonl": inputs_path, "cases.jsonl": cases_path}
        manifest = {"schema_version": 1, "identity_sha256": comparison["identity_sha256"],
                    "seed": seed, "assignment": "SHA-256 of seed, domain and case ID; per-case A/B",
                    "cases": EXPECTED_CASES, "answers": len(reviews), "quality_scores_assigned": False,
                    "evidence_status": "Previously inspected development evaluation; fresh evaluation required before scaling",
                    "rubric_source": "evaluation/README.md", "review_only_evidence": True,
                    "source_sha256": {name: digest(path.read_bytes()) for name, path in source_paths.items()},
                    "output_sha256": {path.name: digest(path.read_bytes()) for path in sorted(temporary.iterdir())}}
        (temporary / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError("Review directory appeared during export; refusing overwrite")
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--inputs", type=Path, default=PROJECT / "training/pilot_evaluation_inputs.jsonl")
    parser.add_argument("--cases", type=Path, default=PROJECT / "evaluation/cases.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    print(export_review(args.run_dir, inputs_path=args.inputs, cases_path=args.cases,
                        output_dir=args.output, seed=args.seed))


if __name__ == "__main__":
    main()
