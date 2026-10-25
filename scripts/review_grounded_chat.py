#!/usr/bin/env python3
"""Verify the frozen base-chatbot run and validate evidence-backed human/agent scores."""
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
    from .grounded_chat import ROOT, MODEL, fingerprint, response_diagnostics
    from .evaluate_grounded_chat import evaluate
except ImportError:
    from grounded_chat import ROOT, MODEL, fingerprint, response_diagnostics
    from evaluate_grounded_chat import evaluate


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_jsonl(path):
    payload = Path(path).read_bytes()
    if payload and not payload.endswith(b'\n'):
        raise ValueError('Incomplete JSONL: '+str(path))
    return [json.loads(line) for line in payload.splitlines()]


def frozen_payloads(root=ROOT):
    folder = root/'evaluation/grounded_chat_v1'
    lock = json.loads((folder/'bundle_lock.json').read_text())
    notebook_path = root/'training/kaggle_cli/grounded_chat_v1/kaggle_grounded_chat.ipynb'
    if sha(notebook_path.read_bytes()) != lock['notebook_sha256']:
        raise ValueError('Submitted notebook changed')
    if sha((notebook_path.parent/'kernel-metadata.json').read_bytes()) != lock['metadata_sha256']:
        raise ValueError('Submitted notebook metadata changed')
    notebook = json.loads(notebook_path.read_text())
    unpack = next(c for c in notebook['cells'] if c['id'] == 'unpack')
    tree = ast.parse(''.join(unpack['source']))
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'packed' for t in n.targets))
    packed = ast.literal_eval(assignment.value)
    payloads = {}
    for name, value in packed.items():
        raw = lzma.decompress(base64.b85decode(value['data']))
        if sha(raw) != value['sha256'] or sha(raw) != lock['payload_sha256'][name]:
            raise ValueError('Frozen payload hash changed: '+name)
        payloads[name] = raw
    if set(payloads) != set(lock['payload_sha256']):
        raise ValueError('Missing/extra frozen payloads')
    return payloads


def verify(download, *, root=ROOT, tokenizer=None):
    download = Path(download)
    payloads = frozen_payloads(root)
    protocol = json.loads(payloads['protocol.json'])
    collection, spec = json.loads(payloads['collection.json']), json.loads(payloads['model_cases.json'])
    for name, raw in payloads.items():
        if (download/'grounded_chat_bundle'/name).read_bytes() != raw:
            raise ValueError('Downloaded payload differs from submitted bundle: '+name)
    for name, expected in protocol['files'].items():
        if sha(payloads[name]) != expected:
            raise ValueError('Runtime protocol payload mismatch')
        if name.endswith('.py') and (root/'scripts'/name).read_bytes() != payloads[name]:
            raise ValueError('Local replay helper differs from frozen run: '+name)
    full_spec_bytes = (root/'evaluation/grounded_chat_v1/evaluation_spec.json').read_bytes()
    if sha(full_spec_bytes) != protocol['evaluation_spec_sha256']:
        raise ValueError('Local frozen evaluation references changed')
    results = download/'grounded_chat_results'
    manifest = json.loads((results/'run_manifest.json').read_text())
    identity = manifest['identity']
    if (identity['model'] != MODEL or identity['packages'] != protocol['packages']
            or identity['adapters_loaded'] != [] or identity['optimizer_updates'] != 0
            or identity['enable_thinking'] is not False or identity['seed'] != 42
            or identity['gpu'] != 'Tesla T4' or identity['gpu_ids_used'] != [0]
            or identity['protocol_sha256'] != sha(payloads['protocol.json'])
            or manifest['status'] != 'complete' or manifest['answers'] != protocol['answers']
            or manifest['all_base_parameters_frozen'] is not True):
        raise ValueError('Incomplete run or mismatched base/runtime identity')
    for key, value in protocol['generation'].items():
        if identity['generation'].get(key) != value:
            raise ValueError('Generation setting changed: '+key)
    response_path = results/'responses.jsonl'
    if sha(response_path.read_bytes()) != manifest['responses_sha256']:
        raise ValueError('Response journal checksum mismatch')
    rows = read_jsonl(response_path)
    if len(rows) != protocol['answers']:
        raise ValueError('Incomplete response count')
    # Replay only local checks, refusing any attempt to generate an answer.
    def no_inference(_):
        raise AssertionError('Journal missing a required response')
    with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
        replay = Path(temp)/'responses.jsonl'
        replay.write_bytes(response_path.read_bytes())
        evaluate(collection, spec, no_inference, replay, identity)
    if tokenizer:
        if sha(tokenizer.chat_template.encode()) != protocol['chat_template_sha256']:
            raise ValueError('Local verification tokenizer template differs')
    for row in rows:
        ids, tokens = row['input_token_ids'], row['token_ids']
        if (row['identity_sha256'] != fingerprint(identity)
                or not isinstance(ids, list) or not ids or len(ids) > protocol['max_input_tokens']
                or not isinstance(tokens, list) or not tokens
                or any(type(i) is not int or i < 0 for i in ids + tokens)
                or row['token_count'] != len(tokens)
                or len(tokens) > protocol['generation']['max_new_tokens']
                or tokens[-1] != protocol['generation']['eos_token_id']
                or row['finish_reason'] != 'stop'
                or row['diagnostics'] != response_diagnostics(row['response'], row['request']['retrieval'])):
            raise ValueError('Invalid token/finish/diagnostic record: '+row['key'])
        if tokenizer:
            actual_ids = tokenizer.apply_chat_template(row['request']['messages'], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            if actual_ids != ids or tokenizer.decode(tokens, skip_special_tokens=True) != row['response']:
                raise ValueError('Prompt tokenization or response decoding differs: '+row['key'])
    return {'status': 'verified', 'answers': len(rows), 'conditions': dict(Counter(r['condition'] for r in rows)),
            'responses_sha256': sha(response_path.read_bytes()), 'base_runtime_identity_sha256': fingerprint(identity),
            'optimizer_updates': 0, 'adapters_loaded': [], 'tokenizer_reconstructed_locally': tokenizer is not None,
            'normal_finishes': len(rows), 'unknown_citation_rows': sum(bool(r['diagnostics']['unknown_citation_ids']) for r in rows),
            'missing_citation_rows': sum(r['diagnostics']['no_citations_with_evidence'] for r in rows),
            'max_input_tokens': max(len(r['input_token_ids']) for r in rows),
            'total_output_tokens': sum(r['token_count'] for r in rows),
            'quality_validation': 'Not established by integrity or citation checks'}, rows, collection


def validate_scores(scores, rows, collection, rubric):
    if len(scores) != len(rows) or [s['key'] for s in scores] != [r['key'] for r in rows]:
        raise ValueError('Scores must cover the exact journal order once')
    packets = {p['id']: {b['block_id']: b['text'] for b in p['blocks']} for p in collection['packets']}
    allowed_flags = set(rubric['critical_flags'])
    for s, row in zip(scores, rows):
        if s['response_sha256'] != row['response_sha256'] or not s.get('notes'):
            raise ValueError('Unbound or unexplained judgment')
        for field in ('response_score', 'conversation_score'):
            if s[field] is not None and (type(s[field]) is not int or not 0 <= s[field] <= 4):
                raise ValueError('Invalid score')
        if s['response_score'] is None or ((row['condition'] == 'conversation') != (s['conversation_score'] is not None)):
            raise ValueError('Response/conversation score applicability changed')
        if s['retrieval_sufficient'] not in {'yes', 'partial', 'no'}:
            raise ValueError('Invalid retrieval sufficiency')
        if s['retrieval_sufficient'] == 'yes':
            if type(s['interpretation_score']) is not int or not 0 <= s['interpretation_score'] <= 4:
                raise ValueError('Interpretation requires score for sufficient evidence')
        elif s['interpretation_score'] is not None:
            raise ValueError('Do not assign interpretation score without sufficient evidence')
        if set(s['critical_flags']) - allowed_flags or (s['critical_flags'] and s['response_score'] != 0):
            raise ValueError('Unknown critical flag or critical-zero violation')
        if not set(s['failure_stages']) <= {'retrieval', 'interpretation', 'conversation'}:
            raise ValueError('Invalid failure stage')
        if not s['response_spans'] or any(not span or span not in row['response'] for span in s['response_spans']):
            raise ValueError('Response evidence must be exact nonempty spans')
        retrieved = {m['packet']['id'] for m in row['request']['retrieval']['matches']}
        for span in s['source_spans']:
            if not span['text'] or span['packet_id'] not in packets or span['text'] not in packets[span['packet_id']].get(span['block_id'], ''):
                raise ValueError('Source evidence must match frozen source block')
            if span.get('in_current_retrieval') != (span['packet_id'] in retrieved):
                raise ValueError('Source availability annotation differs from trace')
        if s['retrieval_sufficient'] == 'yes' and not s['source_spans']:
            raise ValueError('Sufficiency judgment needs source evidence')
    return {'answers_scored': len(scores), 'critical_cases': sum(bool(s['critical_flags']) for s in scores),
            'critical_types': dict(Counter(f for s in scores for f in s['critical_flags'])),
            'failure_stages': dict(Counter(f for s in scores for f in s['failure_stages'])),
            'conditions': {condition: {'answers': len(subset), 'score_at_least_3': sum(s['response_score'] >= 3 for s in subset),
                'critical_cases': sum(bool(s['critical_flags']) for s in subset)}
                for condition in ('retrieved', 'oracle', 'conversation')
                for subset in [[s for s, r in zip(scores, rows) if r['condition'] == condition]]}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', type=Path, default=ROOT/'evaluation/runs/kaggle_grounded_chat_v1/download')
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/runs/kaggle_grounded_chat_v1/verification.json')
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--scores', type=Path)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False, local_files_only=True)
    summary, rows, collection = verify(args.download, tokenizer=tokenizer)
    if args.scores:
        lock_path = args.scores.parent/'score_lock.json'
        lock = json.loads(lock_path.read_text())
        if (lock['scores_sha256'] != sha(args.scores.read_bytes())
                or lock['responses_sha256'] != summary['responses_sha256']
                or lock['review_protocol_sha256'] != sha((args.scores.parent/'review_protocol.json').read_bytes())):
            raise ValueError('Locked score/response/review protocol identity differs')
        spec = json.loads((ROOT/'evaluation/grounded_chat_v1/evaluation_spec.json').read_text())
        summary['quality_review'] = validate_scores(read_jsonl(args.scores), rows, collection, spec['rubric'])
        summary['scores_sha256'] = sha(args.scores.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(summary, indent=2))
