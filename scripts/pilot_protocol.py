#!/usr/bin/env python3
"""Frozen one-epoch pilot and matched development evaluation; stdlib only.

The portable payload validator checks explicit training/evaluation allowlists.
It never rebuilds an artifact inventory: creating a new evaluation result must
not change the inputs of an already frozen pilot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

if __package__:
    from .build_training_pilot import validate_training_payload
else:
    from build_training_pilot import validate_training_payload


PROJECT = Path(__file__).resolve().parents[1]
SUCCESS_RUN = "training/runs/kaggle_361399211"
CASES_SHA256 = "036a898ac83a37058fc7770d9a5c43559a8f468f0ae05cd050b6763cfbd8b86f"
DATASET_SHA256 = "b2701476c7dc8e9f0da811efb0417eb380af64e53666cdb526bd6500cc68f5f1"
LOCK_SHA256 = "14ba0b85eba2210e440d52019d13085efa65f83c5213fe171587537e73962439"
TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
SUCCESS_MANIFEST_SHA256 = "7733ccf5f2a2d7bad5fdc15334b24387ee2ac407f0aaaf0d1b46ddcbf90e3312"
PACKAGES = {"torch": "2.11.0+cu128", "transformers": "5.20.2", "peft": "0.21.2",
            "accelerate": "1.15.0", "bitsandbytes": "0.50.2",
            "huggingface-hub": "1.33.0", "tokenizers": "0.23.1", "jinja2": "3.1.6"}
DEFAULT_SETTINGS = {
    "model_id": "Qwen/Qwen3.5-9B",
    "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "seed": 42, "max_length": 1024, "steps": 500, "epochs": 1,
    "micro_batch_size": 1, "gradient_accumulation_steps": 1,
    "learning_rate": 0.0001, "weight_decay": 0.0, "max_grad_norm": 1.0,
    "amp_initial_scale": 1.0, "rank": 8, "lora_alpha": 16, "lora_dropout": 0.0,
    "optimizer": "torch.optim.AdamW", "packing": False,
    "truncation": "drop_whole_example", "gradient_checkpointing": "non_reentrant",
    "quantization": "nf4_double", "attention_implementation": "sdpa",
    "smoke_max_new_tokens": 64,
    "smoke_question": "Please introduce yourself in two sentences.",
    "save_steps": 50, "logging_steps": 10, "lr_scheduler": "constant",
    "fresh_adapter": True, "run_evaluation": True,
    "evaluation_max_new_tokens": 512, "evaluation_eos_token_id": 248046,
}


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value):
    return sha256_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8"))


def validate_settings(settings):
    """Reject silent experiment changes; revisions require a new protocol."""
    if not isinstance(settings, dict) or set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError("Settings keys must match the frozen pilot schema")
    for key, expected in DEFAULT_SETTINGS.items():
        if type(settings[key]) is not type(expected) or settings[key] != expected:
            raise ValueError(f"Setting differs from the frozen one-epoch pilot: {key}")


def build_order(encoded, seed=42):
    """Longest example first, then seeded shuffle of every other example once.

    The first item is the actual first training update, not an extra warm-up.
    Ties use original dataset order. No global random state is modified.
    """
    if type(seed) is not int:
        raise ValueError("Training order seed must be an integer")
    if not isinstance(encoded, (list, tuple)) or not encoded:
        raise ValueError("Expected nonempty encoded training examples")
    seen = set()
    for item in encoded:
        key = item.get("example_id")
        if not isinstance(key, str) or not key.strip() or key in seen:
            raise ValueError("Missing or duplicate encoded training example ID")
        seen.add(key)
        if item.get("dropped") is not False:
            raise ValueError(f"Pilot cannot omit a dropped training example: {key}")
        if type(item.get("token_count")) is not int or item["token_count"] <= 0:
            raise ValueError(f"Invalid token count: {key}")
    longest = max(range(len(encoded)), key=lambda index: encoded[index]["token_count"])
    remaining = [index for index in range(len(encoded)) if index != longest]
    random.Random(seed).shuffle(remaining)
    return [longest] + remaining


def _parse_jsonl(payload):
    rows = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except ValueError as error:
            raise ValueError(f"Invalid JSONL row {number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {number} must be an object")
        rows.append(value)
    return rows


def _validate_evaluation_rows(rows, persona_sha256):
    if len(rows) != 36:
        raise ValueError("The matched development comparison requires all 36 existing cases")
    seen = set()
    for row in rows:
        if set(row) != {"case_id", "category", "split", "messages"}:
            raise ValueError("Evaluation payload may contain only case_id, category, split and messages")
        key = row["case_id"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Missing or duplicate evaluation case ID")
        seen.add(key)
        if any(not isinstance(row[field], str) or not row[field] for field in ("category", "split")):
            raise ValueError("Invalid evaluation classification metadata")
        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
            raise ValueError(f"Evaluation must end with a user turn: {key}")
        for index, message in enumerate(messages):
            role = "system" if index == 0 else "user" if index % 2 else "assistant"
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message.get("role") != role
                    or not isinstance(message.get("content"), str) or not message["content"].strip()):
                raise ValueError(f"Invalid evaluation conversation: {key}")
        if sha256_bytes(messages[0]["content"].encode("utf-8")) != persona_sha256:
            raise ValueError(f"Evaluation persona differs from training: {key}")
    return rows


def prepare_evaluation_payload(project=PROJECT):
    """Extract model messages only; reference answers/passages never enter bundle."""
    project = Path(project)
    source = (project / "evaluation/cases.jsonl").read_bytes()
    if sha256_bytes(source) != CASES_SHA256:
        raise ValueError("Existing evaluation cases checksum changed")
    lock = json.loads((project / "training/pilot_lock.json").read_bytes())
    rows = [{key: row[key] for key in ("case_id", "category", "split", "messages")}
            for row in _parse_jsonl(source)]
    _validate_evaluation_rows(rows, lock["persona"]["system_prompt_sha256"])
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    metadata = {
        "path": "training/pilot_evaluation_inputs.jsonl", "sha256": sha256_bytes(payload),
        "source_cases_path": "evaluation/cases.jsonl", "source_cases_sha256": CASES_SHA256,
        "records": 36, "case_ids": [row["case_id"] for row in rows],
        "payload_keys": ["case_id", "category", "split", "messages"],
        "evidence_status": "All 36 existing cases are development/previously inspected evidence; no fresh quality claim.",
        "fresh_evaluation_required_before_scaling": True,
        "training_exclusion": "All evaluation messages, reference material and generated answers stay outside training.",
        "prompt_policy": "Original model messages, shared persona, no supplied reference passages or extra 120-word instruction.",
        "assistant_context": "Existing assistant turns are fixed prompt context only; never training targets.",
        "enable_thinking": False, "seed": 42, "max_input_tokens": 8192,
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 512,
                       "eos_token_id": 248046, "pad_token_id": 248044, "use_cache": True},
        "conditions": ["fresh_base", "reloaded_pilot_adapter"],
    }
    return payload, metadata


def build_protocol(project=PROJECT):
    project = Path(project)
    lock_bytes = (project / "training/pilot_lock.json").read_bytes()
    if sha256_bytes(lock_bytes) != LOCK_SHA256:
        raise ValueError("Pilot lock differs from the successful smoke test")
    lock = json.loads(lock_bytes)
    train_bytes = (project / "datasets/train_seed.jsonl").read_bytes()
    if sha256_bytes(train_bytes) != DATASET_SHA256:
        raise ValueError("Training dataset differs from the successful smoke test")
    validate_training_payload(train_bytes, lock)
    manifest_bytes = (project / SUCCESS_RUN / "run_manifest.json").read_bytes()
    if sha256_bytes(manifest_bytes) != SUCCESS_MANIFEST_SHA256:
        raise ValueError("Successful smoke-test manifest changed")
    manifest = json.loads(manifest_bytes)
    if manifest["status"] != "passed" or not all(manifest["checks"].values()):
        raise ValueError("Pilot requires the passing archived smoke test")
    identity = manifest["identity"]
    if identity["packages"] != PACKAGES or identity["chat_template_sha256"] != TEMPLATE_SHA256:
        raise ValueError("Successful training package/template identity changed")
    report = json.loads((project / SUCCESS_RUN / "tokenization_report.json").read_bytes())
    encoded = report["examples"]
    if len(encoded) != 500 or {row["example_id"] for row in encoded} != set(lock["dataset"]["example_sha256"]):
        raise ValueError("Archived tokenization does not cover the complete training dataset")
    order = build_order(encoded, DEFAULT_SETTINGS["seed"])
    first = encoded[order[0]]
    if first["example_id"] != "seed_9f8f80b0bc55ce95" or first["token_count"] != 457:
        raise ValueError("Longest-example memory check differs from the successful tokenization")
    _, evaluation = prepare_evaluation_payload(project)
    return {
        "schema_version": 1, "purpose": "Small one-epoch QLoRA pilot; no scaling or fresh quality claim",
        "settings": dict(DEFAULT_SETTINGS), "packages": dict(PACKAGES),
        "dataset": {"path": "datasets/train_seed.jsonl", "sha256": DATASET_SHA256,
                    "records": 500, "multi_turn_records": 65,
                    "training_payload_keys": ["example_id", "messages"]},
        "pilot_lock": {"path": "training/pilot_lock.json", "sha256": LOCK_SHA256,
                       "canonical_sha256": canonical_sha256(lock)},
        "model": dict(lock["model"]), "persona": dict(lock["persona"]),
        "chat_template_sha256": TEMPLATE_SHA256,
        "smoke_test": {"run": SUCCESS_RUN, "manifest_sha256": SUCCESS_MANIFEST_SHA256,
                       "status": "passed", "settings": identity["settings"],
                       "tokenization_report_sha256": sha256_bytes((project / SUCCESS_RUN / "tokenization_report.json").read_bytes())},
        "training_order": {"policy": "Longest example is first actual optimizer update; seeded shuffle of remaining 499; each exactly once",
                           "seed": 42, "records": 500, "longest_example_id": first["example_id"],
                           "longest_token_count": first["token_count"],
                           "ordered_example_ids_sha256": canonical_sha256([encoded[i]["example_id"] for i in order])},
        "checkpoint_policy": {"every_optimizer_steps": 50, "save_final": True,
                              "resumable_state": ["adapter", "optimizer", "grad_scaler", "rng", "next_training_index", "protocol_identity"],
                              "new_run_adapter": "fresh; never initialize from the two-step smoke adapter"},
        "evaluation": evaluation,
        "training_allowlist": ["datasets/train_seed.jsonl"],
        "exclusion_policy": "Only the exact reviewed training payload enters training; evaluation inputs and outputs are a separate path and never concatenated.",
    }


def validate_evaluation_payload(payload, protocol):
    if not isinstance(payload, bytes) or sha256_bytes(payload) != protocol["evaluation"]["sha256"]:
        raise ValueError("Evaluation payload checksum differs from the frozen pilot")
    rows = _validate_evaluation_rows(_parse_jsonl(payload), protocol["persona"]["system_prompt_sha256"])
    if [row["case_id"] for row in rows] != protocol["evaluation"]["case_ids"]:
        raise ValueError("Evaluation case order differs from the frozen pilot")
    return rows


def validate_protocol_payloads(train_bytes, eval_bytes, lock, protocol):
    """Portable pre-model check; returns (training_rows, evaluation_rows)."""
    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported pilot protocol schema")
    validate_settings(protocol["settings"])
    if (protocol["dataset"]["sha256"] != DATASET_SHA256
            or protocol["pilot_lock"]["sha256"] != LOCK_SHA256
            or protocol["pilot_lock"]["canonical_sha256"] != canonical_sha256(lock)
            or protocol["model"] != lock["model"]
            or protocol["persona"] != lock["persona"]
            or protocol["chat_template_sha256"] != TEMPLATE_SHA256
            or protocol["evaluation"]["source_cases_sha256"] != CASES_SHA256
            or protocol["packages"] != PACKAGES
            or protocol["training_allowlist"] != ["datasets/train_seed.jsonl"]):
        raise ValueError("Frozen protocol identity/provenance differs from the passing pilot inputs")
    train_rows = validate_training_payload(train_bytes, lock)
    if sha256_bytes(train_bytes) != protocol["dataset"]["sha256"]:
        raise ValueError("Protocol training checksum mismatch")
    return train_rows, validate_evaluation_payload(eval_bytes, protocol)


def validate_protocol(project, protocol):
    if protocol != build_protocol(project):
        raise ValueError("Pilot protocol is stale; freeze an explicit new experiment")
    project = Path(project)
    lock = json.loads((project / "training/pilot_lock.json").read_bytes())
    validate_protocol_payloads((project / "datasets/train_seed.jsonl").read_bytes(),
                               (project / protocol["evaluation"]["path"]).read_bytes(), lock, protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--write-protocol", action="store_true")
    args = parser.parse_args()
    path = args.project / "training/pilot_protocol.json"
    if args.write_protocol:
        protocol = build_protocol(args.project)
        evaluation_bytes, _ = prepare_evaluation_payload(args.project)
        (args.project / protocol["evaluation"]["path"]).write_bytes(evaluation_bytes)
        path.write_text(json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    protocol = validate_protocol(args.project, json.loads(path.read_bytes()))
    print(json.dumps({"training_examples": protocol["dataset"]["records"], "epochs": 1,
                      "optimizer_steps": 500, "evaluation_cases": 36,
                      "protocol_sha256": sha256_bytes(path.read_bytes())}, indent=2))


if __name__ == "__main__":
    main()
