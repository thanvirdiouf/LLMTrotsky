"""Strict, CPU-testable assistant-answer masks for the pinned Qwen3.5 template.

No model or training packages are imported here.  Call with the *official*
checkpoint tokenizer, not a replacement chat template.  Completed assistant
turns are located by replacing one content field with a unique probe in the
whole official render.  Replacing that probe with the original field must
reproduce the unmodified render exactly.  Token boundaries are then checked
against the official full tokenization; content substring searches never set
labels.  This matters because Qwen3.5 renders historical and final assistant
turns differently.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

IGNORE_INDEX = -100
IM_END = "<|im_end|>"
ASSISTANT_HEADER = "<|im_start|>assistant\n"
NONTHINKING_PREFIX = "<think>\n\n</think>\n\n"
_CONTROL = re.compile(r"<\|[^<>\n]*\|>|</?(?:think|tool_call|tool_response)(?:\s[^<>]*)?>")


def _token_ids(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or any(type(i) is not int for i in value):
        raise ValueError("Tokenizer must return one flat sequence of integer token IDs")
    return list(value)


def _encode(tokenizer, text):
    return _token_ids(tokenizer.encode(text, add_special_tokens=False))


def validate_messages(tokenizer, messages):
    """Accept text-only, complete user/assistant pairs with an optional system turn."""
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("Expected nonempty complete conversation messages")
    output = []
    special = tuple(t for t in getattr(tokenizer, "all_special_tokens", []) if t)
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError("Training messages must contain only role and content")
        role, content = message["role"], message["content"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError("Only text system/user/assistant messages are supported")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Every training message needs nonempty text content")
        if _CONTROL.search(content) or any(token in content for token in special):
            raise ValueError(f"Injected control token in message {index}")
        output.append({"role": role, "content": content})
    offset = int(output[0]["role"] == "system")
    conversation = output[offset:]
    if not conversation or len(conversation) % 2:
        raise ValueError("Conversation must contain complete user/assistant pairs")
    for index, message in enumerate(conversation):
        if message["role"] != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("Conversation roles must alternate user and assistant")
    return output


def _render(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    if not isinstance(text, str):
        raise ValueError("Official template must render a text string")
    return text


def _boundary(tokenizer, prefix, full_ids, name):
    ids = _encode(tokenizer, prefix)
    if full_ids[:len(ids)] != ids:
        raise ValueError(f"Official tokenization does not preserve the {name} boundary")
    return len(ids)


def encode_assistant_example(tokenizer, messages, max_length, example_id=None):
    """Return explicit labels, or a reported whole-example overflow drop.

    Only answer text and its following single ``<|im_end|>`` token get labels.
    Headers, the empty thinking wrapper, separators and all other roles receive
    -100.  No sequence is truncated and no end token is manufactured.  Dropped
    records retain counts/spans for auditing but have no trainable arrays.
    """
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    messages = validate_messages(tokenizer, messages)
    rendered = _render(tokenizer, messages)
    full_ids = _token_ids(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False,
        truncation=False, padding=False, return_dict=False,
    ))
    if full_ids != _encode(tokenizer, rendered):
        raise ValueError("Official full encoding differs from its unmodified rendered text")
    end_ids = _encode(tokenizer, IM_END)
    if len(end_ids) != 1 or IM_END not in getattr(tokenizer, "all_special_tokens", []):
        raise ValueError("Expected official single special <|im_end|> token")
    end_id = end_ids[0]
    labels = [IGNORE_INDEX] * len(full_ids)
    spans = []
    for message_index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        # A probe is placed in the structured assistant field, never searched
        # among the original answer/user text to infer its location.
        marker = f"MASK_BOUNDARY_{message_index}_{hashlib.sha256(rendered.encode()).hexdigest()}"
        if marker in rendered:
            raise ValueError("Unexpected mask probe collision")
        probed = [dict(item) for item in messages]
        probed[message_index]["content"] = marker
        probe_render = _render(tokenizer, probed)
        if probe_render.count(marker) != 1:
            raise ValueError("Official template did not preserve one assistant content probe")
        before, after = probe_render.split(marker)
        if before + message["content"] + after != rendered:
            raise ValueError("Official template changes assistant content or context; cannot mask safely")
        header_kind = None
        for header in (ASSISTANT_HEADER + NONTHINKING_PREFIX, ASSISTANT_HEADER):
            if before.endswith(header):
                header_kind = "empty_think" if header.endswith(NONTHINKING_PREFIX) else "plain"
                break
        if header_kind is None or not after.startswith(IM_END + "\n"):
            raise ValueError("Unexpected official assistant header/end-of-answer layout")
        start = _boundary(tokenizer, before, full_ids, "answer-start")
        answer_stop = _boundary(tokenizer, before + message["content"], full_ids, "answer-end")
        stop = _boundary(tokenizer, before + message["content"] + IM_END, full_ids, "end-of-answer")
        if not (start < answer_stop and stop == answer_stop + 1 and full_ids[answer_stop] == end_id):
            raise ValueError("Answer must have text followed by exactly one official end-of-answer token")
        if any(label != IGNORE_INDEX for label in labels[start:stop]):
            raise ValueError("Assistant target spans overlap")
        labels[start:stop] = full_ids[start:stop]
        spans.append({"message_index": message_index, "start": start, "stop": stop,
                      "end_token_position": answer_stop, "header_kind": header_kind,
                      "answer_tokens": answer_stop - start})
    dropped = len(full_ids) > max_length
    return {"example_id": example_id, "dropped": dropped,
            "reason": "overflow" if dropped else None,
            "token_count": len(full_ids), "max_length": max_length,
            "supervised_tokens": sum(s["stop"] - s["start"] for s in spans),
            "assistant_spans": spans,
            "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "input_ids": None if dropped else full_ids,
            "attention_mask": None if dropped else [1] * len(full_ids),
            "labels": None if dropped else labels}


def inspect_tokens(tokenizer, encoded):
    """Produce serializable token-level evidence for inspection JSON/CSV outputs."""
    if encoded.get("dropped"):
        raise ValueError("Overflow examples are dropped whole; inspect an example within max_length")
    rows = []
    for index, (token_id, label) in enumerate(zip(encoded["input_ids"], encoded["labels"])):
        span = next((s for s in encoded["assistant_spans"] if s["start"] <= index < s["stop"]), None)
        token = tokenizer.convert_ids_to_tokens(token_id)
        rows.append({"index": index, "token_id": token_id, "token": token,
                     "decoded": tokenizer.decode([token_id], skip_special_tokens=False),
                     "label": label, "supervised": label != IGNORE_INDEX,
                     "assistant_message_index": span["message_index"] if span else None,
                     "end_of_answer": bool(span and index == span["end_token_position"])})
    return rows


def collate_examples(features, pad_token_id, as_tensors=False):
    """Right-pad by *sequence length*, preserving targets even when pad == EOS."""
    if type(pad_token_id) is not int or not features:
        raise ValueError("Need a padding token ID and at least one example")
    if any(feature.get("dropped") for feature in features):
        raise ValueError("Dropped overflow examples cannot enter a training batch")
    width = max(len(feature["input_ids"]) for feature in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        ids, mask, labels = feature["input_ids"], feature["attention_mask"], feature["labels"]
        if not ids or len(ids) != len(labels) or mask != [1] * len(ids):
            raise ValueError("Expected nonempty, unpadded explicit-mask examples")
        if not any(label != IGNORE_INDEX for label in labels):
            raise ValueError("Example has no supervised answer tokens")
        if any(label != IGNORE_INDEX and label != token_id for token_id, label in zip(ids, labels)):
            raise ValueError("Labels must be original token IDs or -100")
        padding = width - len(ids)
        batch["input_ids"].append(list(ids) + [pad_token_id] * padding)
        batch["attention_mask"].append(list(mask) + [0] * padding)
        batch["labels"].append(list(labels) + [IGNORE_INDEX] * padding)
    if as_tensors:
        import torch
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
    return batch
