#!/usr/bin/env python3
"""Reproduce the real-tokenizer CPU preflight; never load model weights or a GPU.

Download only tokenizer.json, tokenizer_config.json and chat_template.jinja from
Qwen/Qwen3.5-9B at c202236235762e1c871ad0ccb60c8ee5ba337b9a, then run:

    python scripts/verify_training_masks.py --tokenizer-path /tmp/trotsky-tokenizer-assets

Requires Transformers/tokenizers in the executing environment. No torch package
is needed. The repository's locked training allowlist is the only input text.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

try:
    from .build_training_pilot import validate_training_payload
    from .training_masks import collate_examples, encode_assistant_example, inspect_tokens
except ImportError:
    from build_training_pilot import validate_training_payload
    from training_masks import collate_examples, encode_assistant_example, inspect_tokens

MODEL_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ASSET_SHA256 = {
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def distribution(values):
    values = sorted(values)
    return {"min": min(values), "p50": values[math.ceil(len(values) * .50) - 1],
            "p95": values[math.ceil(len(values) * .95) - 1], "max": max(values),
            "total": sum(values)}


def verify(project, tokenizer_path, max_length=1024):
    """Verify every locked row and write training-only token evidence."""
    from transformers import AutoTokenizer

    project, tokenizer_path = Path(project), Path(tokenizer_path)
    for name, expected in ASSET_SHA256.items():
        if digest((tokenizer_path / name).read_bytes()) != expected:
            raise ValueError("Tokenizer asset differs from the pinned official checkpoint: " + name)
    payload = (project / "datasets/train_seed.jsonl").read_bytes()
    lock_bytes = (project / "training/pilot_lock.json").read_bytes()
    rows = validate_training_payload(payload, json.loads(lock_bytes))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    if digest(tokenizer.chat_template.encode()) != ASSET_SHA256["chat_template.jinja"]:
        raise ValueError("Loaded template differs from its pinned asset")
    encoded = [encode_assistant_example(tokenizer, row["messages"], max_length, row["example_id"])
               for row in rows]
    kept = [(row, enc) for row, enc in zip(rows, encoded) if not enc["dropped"]]
    # Exact target decoding checks every assistant turn, not just the samples.
    for row, enc in kept:
        expected = "".join(m["content"] + "<|im_end|>" for m in row["messages"] if m["role"] == "assistant")
        actual = tokenizer.decode([i for i in enc["labels"] if i != -100], skip_special_tokens=False)
        if actual != expected:
            raise ValueError("Supervised decoding differs from assistant answers: " + row["example_id"])
        positions = {i for span in enc["assistant_spans"] for i in range(span["start"], span["stop"])}
        for index, (token_id, label) in enumerate(zip(enc["input_ids"], enc["labels"])):
            if label != (token_id if index in positions else -100):
                raise ValueError("Mask differs from verified assistant spans")
        for span in enc["assistant_spans"]:
            expected_header = "empty_think" if span["message_index"] == len(row["messages"]) - 1 else "plain"
            if span["header_kind"] != expected_header:
                raise ValueError("Historical/final assistant template layout differs from the pinned checkpoint")
            if enc["input_ids"][span["end_token_position"]] != tokenizer.convert_tokens_to_ids("<|im_end|>"):
                raise ValueError("Assistant end token differs from official <|im_end|>")
    singles = [pair for pair in kept if len(pair[1]["assistant_spans"]) == 1]
    multis = [pair for pair in kept if len(pair[1]["assistant_spans"]) > 1]
    if not singles or not multis:
        raise ValueError("Need a complete single-turn and multi-turn example within max_length")
    selected = [min(singles, key=lambda pair: pair[1]["token_count"]),
                min(multis, key=lambda pair: pair[1]["token_count"])]
    evidence = []
    output = project / "training"
    for kind, (row, enc) in zip(["single", "multi"], selected):
        exact = encode_assistant_example(tokenizer, row["messages"], enc["token_count"], row["example_id"])
        overflow = encode_assistant_example(tokenizer, row["messages"], enc["token_count"] - 1, row["example_id"])
        if exact["dropped"] or not overflow["dropped"] or overflow["reason"] != "overflow":
            raise ValueError("Exact-length/one-token-overflow policy check failed")
        if any(overflow[key] is not None for key in ["input_ids", "attention_mask", "labels"]):
            raise ValueError("Dropped overflow unexpectedly has trainable token arrays")
        name = f"token_mask_inspection_{kind}.json"
        value = {"example_id": row["example_id"], "token_count": enc["token_count"],
                 "supervised_tokens": enc["supervised_tokens"], "rendered_sha256": enc["rendered_sha256"],
                 "assistant_spans": enc["assistant_spans"], "tokens": inspect_tokens(tokenizer, enc)}
        write_json(output / name, value)
        evidence.append({"kind": kind, "example_id": row["example_id"], "token_count": enc["token_count"],
                         "supervised_tokens": enc["supervised_tokens"], "assistant_spans": enc["assistant_spans"],
                         "path": "training/" + name, "sha256": digest((output / name).read_bytes())})
    # Force padding to use EOS, even if this tokenizer's configured pad differs.
    for pad_id in [tokenizer.pad_token_id, tokenizer.eos_token_id]:
        padded = collate_examples([enc for _, enc in selected], pad_id)
        for index, (_, enc) in enumerate(selected):
            length = enc["token_count"]
            if (padded["labels"][index][:length] != enc["labels"]
                    or any(label != -100 for label in padded["labels"][index][length:])
                    or any(flag != 0 for flag in padded["attention_mask"][index][length:])):
                raise ValueError("Length-based padding corrupted labels/attention")
            for span in enc["assistant_spans"]:
                if padded["labels"][index][span["end_token_position"]] != tokenizer.eos_token_id:
                    raise ValueError("EOS label was masked as padding")
    report = {
        "schema_version": 1, "status": "passed", "scope": "CPU tokenizer/masks only; GPU training remains untested",
        "model_id": MODEL_ID, "revision": REVISION, "model_weights_loaded": False, "gpu_used": False,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["transformers", "tokenizers", "huggingface-hub", "jinja2"]},
        "dataset_path": "datasets/train_seed.jsonl", "dataset_sha256": digest(payload),
        "pilot_lock_sha256": digest(lock_bytes), "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_files_sha256": ASSET_SHA256, "chat_template_sha256": digest(tokenizer.chat_template.encode()),
        "helper_sha256": digest((project / "scripts/training_masks.py").read_bytes()),
        "verifier_sha256": digest(Path(__file__).read_bytes()),
        "tokenization": {"enable_thinking": False, "add_generation_prompt": False, "truncation": False,
                         "max_length": max_length, "overflow_policy": "drop whole example; no synthetic EOS",
                         "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
                         "forced_eos_padding_tested": True},
        "counts": {"records": len(rows), "kept": len(kept), "dropped": len(rows) - len(kept),
                   "single_turn": sum(len(e["assistant_spans"]) == 1 for e in encoded),
                   "multi_turn": sum(len(e["assistant_spans"]) > 1 for e in encoded),
                   "assistant_answers": sum(len(e["assistant_spans"]) for e in encoded)},
        "input_token_distribution": distribution([e["token_count"] for e in encoded]),
        "supervised_token_distribution": distribution([e["supervised_tokens"] for e in encoded]),
        "dropped_examples": [{"example_id": r["example_id"], "token_count": e["token_count"], "reason": e["reason"]}
                             for r, e in zip(rows, encoded) if e["dropped"]],
        "checks": {"official_full_encoding_equality": True, "official_answer_boundary_prefix_equality": True,
                   "all_kept_assistant_answers_and_im_end_only": True,
                   "system_user_headers_empty_think_and_separator_masked": True,
                   "historical_and_final_assistant_templates": True,
                   "exact_length_and_one_token_overflow": True, "length_based_eos_padding": True},
        "selected_inspections": evidence,
        "reproduce": "python scripts/verify_training_masks.py --tokenizer-path /tmp/trotsky-tokenizer-assets --max-length " + str(max_length),
        "tokenizer_download": {"repo_id": MODEL_ID, "revision": REVISION, "token": False,
                               "allow_patterns": list(ASSET_SHA256), "local_dir": "/tmp/trotsky-tokenizer-assets"},
    }
    write_json(output / "token_mask_preflight.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    report = verify(args.project, args.tokenizer_path, args.max_length)
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "input_token_distribution": report["input_token_distribution"]}, indent=2))


if __name__ == "__main__":
    main()
