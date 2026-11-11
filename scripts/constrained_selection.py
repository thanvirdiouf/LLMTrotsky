"""Token prefixes for bounded selections from the supplied reviewed records."""

import json
from itertools import combinations


def allowed_sequences(tokenizer, record_ids, eos, *, decisions=False):
    ids = tuple(record_ids)
    if len(ids) > 12 or len(set(ids)) != len(ids):
        raise ValueError('Expected at most twelve distinct reviewed records')
    for count in range(len(ids) + 1):
        for selection in combinations(ids, count):
            if decisions:
                chosen = set(selection)
                value = {'include': ''.join('1' if record_id in chosen else '0' for record_id in ids)}
            else:
                value = {'claim_ids': selection}
            text = json.dumps(value, separators=(',', ':'))
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if eos in tokens or tokenizer.decode(tokens, skip_special_tokens=True) != text:
                raise ValueError('Selection does not round-trip through the tokenizer')
            yield (*tokens, eos)


class SelectionConstraint:
    def __init__(self, tokenizer, record_ids, eos, prompt_length, *, decisions=False):
        self.prompt_length = prompt_length
        self.prefixes = {}
        for tokens in allowed_sequences(tokenizer, record_ids, eos, decisions=decisions):
            for i, token in enumerate(tokens):
                self.prefixes.setdefault(tokens[:i], set()).add(token)

    def __call__(self, input_ids, scores):
        prefix = tuple(input_ids[0, self.prompt_length:].tolist())
        allowed = sorted(self.prefixes[prefix])
        masked = scores.new_full(scores.shape, float('-inf'))
        masked[:, allowed] = scores[:, allowed]
        return masked
