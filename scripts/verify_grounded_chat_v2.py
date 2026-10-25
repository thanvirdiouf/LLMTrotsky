#!/usr/bin/env python3
"""Verify exact v2 bundle, cached calls, rendered answers and state transitions."""
import argparse
import ast
import base64
from collections import Counter
import contextlib
import hashlib
import io
import json
import lzma
from pathlib import Path
import tempfile

try:
    from .grounded_chat import ROOT, MODEL
    from .evaluate_grounded_chat_v2 import evaluate, read_journal
except ImportError:
    from grounded_chat import ROOT, MODEL
    from evaluate_grounded_chat_v2 import evaluate, read_journal


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def frozen_payloads(root=ROOT):
    folder = root/'evaluation/grounded_chat_v2'
    lock = json.loads((folder/'bundle_lock.json').read_text())
    stage = root/'training/kaggle_cli/grounded_chat_v2'
    raw = (stage/'kaggle_grounded_chat_v2.ipynb').read_bytes()
    if sha(raw) != lock['notebook_sha256'] or sha((stage/'kernel-metadata.json').read_bytes()) != lock['metadata_sha256']:
        raise ValueError('V2 submission differs from frozen notebook/metadata')
    notebook = json.loads(raw)
    cell = next(c for c in notebook['cells'] if c['id'] == 'unpack')
    tree = ast.parse(''.join(cell['source']))
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'packed' for t in n.targets))
    packed = ast.literal_eval(assignment.value)
    payloads = {}
    for name, value in packed.items():
        raw = lzma.decompress(base64.b85decode(value['data']))
        if sha(raw) != value['sha256'] or sha(raw) != lock['payload_sha256'][name]:
            raise ValueError('Frozen payload changed: '+name)
        payloads[name] = raw
    if set(payloads) != set(lock['payload_sha256']):
        raise ValueError('Frozen payload inventory differs')
    return payloads


def verify(download, *, root=ROOT, tokenizer=None):
    download = Path(download)
    payloads = frozen_payloads(root)
    protocol = json.loads(payloads['protocol.json'])
    for name, raw in payloads.items():
        if (download/'grounded_chat_v2_bundle'/name).read_bytes() != raw:
            raise ValueError('Downloaded payload differs: '+name)
    for name, expected in protocol['files'].items():
        if sha(payloads[name]) != expected:
            raise ValueError('Runtime protocol hash mismatch')
        if name.endswith('.py') and (root/'scripts'/name).read_bytes() != payloads[name]:
            raise ValueError('Local replay code differs from frozen run: '+name)
    for name, expected in protocol['preserved_files'].items():
        if sha((root/name).read_bytes()) != expected:
            raise ValueError('V1 artifacts or saved adapters changed: '+name)
    if sha((root/'evaluation/grounded_chat_v2/evaluation_spec.json').read_bytes()) != protocol['evaluation_spec_sha256']:
        raise ValueError('Frozen references changed')
    results = download/'grounded_chat_v2_results'
    manifest = json.loads((results/'run_manifest.json').read_text())
    identity = manifest['identity']
    if (manifest['status'] != 'complete' or manifest['answers'] != protocol['answers']
            or manifest['model_calls'] > protocol['maximum_model_calls']
            or not manifest['all_base_parameters_frozen']
            or identity['model'] != MODEL or identity['packages'] != protocol['packages']
            or identity['protocol_sha256'] != sha(payloads['protocol.json'])
            or identity['optimizer_updates'] != 0 or identity['adapters_loaded'] != []
            or identity['enable_thinking'] is not False or identity['gpu_ids_used'] != [0]
            or identity['gpu'] != 'Tesla T4' or identity['seed'] != 42):
        raise ValueError('Incomplete/mixed v2 execution identity')
    for name, value in protocol['generation'].items():
        if identity['generation'].get(name) != value:
            raise ValueError('Generation policy differs')
    for name, field in [('answers.jsonl', 'answers_sha256'), ('model_calls.jsonl', 'model_calls_sha256')]:
        if sha((results/name).read_bytes()) != manifest[field]:
            raise ValueError('Downloaded result journal checksum mismatch')
    answers, calls = read_journal(results/'answers.jsonl'), read_journal(results/'model_calls.jsonl')
    if len(answers) != protocol['answers'] or len(calls) != manifest['model_calls']:
        raise ValueError('Incomplete journal coverage')
    def no_inference(_):
        raise AssertionError('Missing cached generation; no inference permitted in verification')
    with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
        path = Path(temp)
        for name in ('answers.jsonl', 'model_calls.jsonl'):
            (path/name).write_bytes((results/name).read_bytes())
        replay, replay_calls = evaluate(json.loads(payloads['collection.json']), json.loads(payloads['claims.json']),
                                       json.loads(payloads['model_cases.json']), no_inference, path, identity)
        if replay != answers or replay_calls != calls:
            raise ValueError('State/render/call replay differs')
    if tokenizer and sha(tokenizer.chat_template.encode()) != protocol['chat_template_sha256']:
        raise ValueError('Local tokenizer template differs')
    for call in calls:
        out = call['output']; ids, tokens = out['input_token_ids'], out['token_ids']
        if (not ids or not tokens or any(type(t) is not int or t < 0 for t in ids+tokens)
                or len(ids) > protocol['max_input_tokens'] or out['token_count'] != len(tokens)
                or len(tokens) > protocol['generation']['max_new_tokens']):
            raise ValueError('Invalid prompt/output token record')
        finish = 'stop' if tokens[-1] == protocol['generation']['eos_token_id'] else 'length'
        if out['finish_reason'] != finish or (finish == 'length' and len(tokens) != protocol['generation']['max_new_tokens']):
            raise ValueError('Completion status differs from tokens')
        if tokenizer:
            actual = tokenizer.apply_chat_template(call['messages'], tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False, return_dict=False)
            if actual != ids or tokenizer.decode(tokens, skip_special_tokens=True) != out['response']:
                raise ValueError('Local prompt/output reconstruction differs')
    return {'status': 'verified', 'answers': len(answers), 'model_calls': len(calls),
        'conditions': dict(Counter(a['condition'] for a in answers)),
        'pipeline_status': dict(Counter(a['status'] for a in answers if a['condition'] == 'evidence_first_checked')),
        'truncated_model_calls': sum(c['output']['finish_reason'] != 'stop' for c in calls),
        'max_input_tokens': max(len(c['output']['input_token_ids']) for c in calls),
        'generation_seconds': sum(c['output']['seconds'] for c in calls),
        'tokenizer_reconstructed_locally': tokenizer is not None,
        'answers_sha256': manifest['answers_sha256'], 'model_calls_sha256': manifest['model_calls_sha256'],
        'v1_and_adapters_unchanged': True, 'optimizer_updates': 0, 'adapters_loaded': [],
        'quality_validation': 'Not established by integrity, rendering or model-checker approval'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', type=Path, default=ROOT/'evaluation/runs/kaggle_grounded_chat_v2/download')
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/runs/kaggle_grounded_chat_v2/verification.json')
    parser.add_argument('--tokenizer', type=Path)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False, local_files_only=True)
    result = verify(args.download, tokenizer=tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
