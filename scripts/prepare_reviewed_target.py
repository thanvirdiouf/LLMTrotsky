"""Freeze focused literal-span selection inputs; no model calls or answer keys."""
import argparse
import json
from pathlib import Path
from build_ghost_candidate_probe import ROOT, enc, sha
from build_reviewed_selection import frozen_inputs as parent_inputs
from candidate_audit import input_ids

INPUTS = 'evaluation/ghost_reviewed_target_inputs_v1'
DESIGN = 'config/ghost_reviewed_target_design_v1.json'


def make_jobs(parent, design, tokenizer=None):
    if (design['purpose'] != 'literal_target_selection_diagnostic' or len(parent) != 5
            or [j['case_id'] for j in parent] != [t['case_id'] for t in design['targets']]):
        raise ValueError('Target diagnostic parent/case scope differs')
    jobs = []
    for old, target in zip(parent, design['targets']):
        span = target['target_span']
        if not isinstance(span, str) or not span.strip() or old['question'].count(span) != 1:
            raise ValueError('Target must be a unique exact span of the original question')
        data = json.loads(old['messages'][1]['content'])
        if data['current_user_request'] != old['question'] or 'selection_target_span' in data:
            raise ValueError('Original request differs')
        data['selection_target_span'] = span
        messages = [{'role': 'system', 'content': design['system_prompt']},
                    {'role': 'user', 'content': json.dumps(data, ensure_ascii=False, separators=(',', ':'))}]
        job = {k: old[k] for k in ('case_id', 'question', 'source_packet_ids', 'available_record_fingerprints')}
        job.update(target_span=span, messages=messages)
        if tokenizer is not None: job['input_token_ids'] = input_ids(tokenizer, messages)
        jobs.append(job)
    return jobs


def prepare(output, root=ROOT, tokenizer=None):
    parent, metadata, binding = parent_inputs(root)
    raw_design = (root/DESIGN).read_bytes(); design = json.loads(raw_design)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(root/'evaluation/ghost_reviewed_selector_inputs_v1/tokenizer_assets', local_files_only=True, trust_remote_code=False)
    if sha(tokenizer.chat_template.encode()) != metadata['chat_template_sha256']:
        raise ValueError('Candidate template differs')
    jobs = make_jobs(json.loads(parent['selection_requests.json']), design, tokenizer)
    info = {**metadata, 'purpose': 'reviewed_target_inputs_only', 'parent_inputs': binding,
            'parent_requests_sha256': sha(parent['selection_requests.json']), 'design_sha256': sha(raw_design),
            'request_input_tokens': [len(j['input_token_ids']) for j in jobs],
            'target_selection_is_semantic_validation': False, 'source_representation_changed': False}
    values = {n: raw for n, raw in parent.items() if n not in ('selection_requests.json', 'input_protocol.json')}
    values.update({'selection_requests.json': enc(jobs), 'input_protocol.json': enc(info), 'selection_design.json': raw_design})
    lock = {'purpose': 'input_freeze_only', 'file_sha256': {n: sha(raw) for n, raw in values.items()},
            'preparer_sha256': sha((root/'scripts/prepare_reviewed_target.py').read_bytes()),
            'renderer_sha256': sha((root/'scripts/reviewed_source_display.py').read_bytes())}
    values['input_lock.json'] = enc(lock); output = Path(output)
    for name, raw in values.items():
        if (output/name).exists() and (output/name).read_bytes() != raw:
            raise ValueError('Existing target inputs differ; use a new directory')
    for name, raw in values.items():
        path = output/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    return info


def frozen_inputs(root=ROOT, renderer=None):
    directory = root/INPUTS; lock_raw = (directory/'input_lock.json').read_bytes(); lock = json.loads(lock_raw)
    values = {n: (directory/n).read_bytes() for n in lock['file_sha256']}
    if any(sha(raw) != lock['file_sha256'][n] for n, raw in values.items()):
        raise ValueError('Frozen target input changed')
    if renderer is None: renderer = (root/'scripts/reviewed_source_display.py').read_bytes()
    if sha(renderer) != lock['renderer_sha256']: raise ValueError('Frozen renderer changed')
    parent, _, binding = parent_inputs(root, renderer=renderer)
    info = json.loads(values['input_protocol.json']); design = json.loads(values['selection_design.json'])
    if (info['purpose'] != 'reviewed_target_inputs_only' or info['parent_inputs'] != binding
            or info['parent_requests_sha256'] != sha(parent['selection_requests.json'])
            or info['design_sha256'] != sha(values['selection_design.json'])
            or values['selection_design.json'] != (root/DESIGN).read_bytes()
            or info['model_calls'] != 0 or info['submitted_to_kaggle'] is not False
            or info['target_selection_is_semantic_validation'] is not False):
        raise ValueError('Target diagnostic provenance differs')
    for name in parent:
        if name not in ('selection_requests.json', 'input_protocol.json') and values[name] != parent[name]:
            raise ValueError('Original source/ledger/probe/tokenizer changed: '+name)
    jobs = json.loads(values['selection_requests.json'])
    if [{k: v for k, v in j.items() if k != 'input_token_ids'} for j in jobs] != make_jobs(json.loads(parent['selection_requests.json']), design):
        raise ValueError('Original question/source/menu or target span differs')
    return values, info, {'path': INPUTS, 'input_lock_sha256': sha(lock_raw), 'parent_inputs': binding,
                          'design_sha256': sha(values['selection_design.json'])}


def score_targets(answers, expectations):
    if [r['case_id'] for r in answers] != [e['case_id'] for e in expectations['cases']]:
        raise ValueError('Local diagnostic expectations differ')
    checks = []
    for row, expected in zip(answers, expectations['cases']):
        valid = row['status'] in ('reviewed_summaries_selected', 'no_summaries_selected')
        selected = row['selected_claim_ids']
        if expected['selection_kind'] == 'empty':
            passed = valid and not selected
        elif expected['selection_kind'] == 'factual_request':
            passed = valid and set(expected['required_record_ids']) <= set(selected)
        else: raise ValueError('Unknown local diagnostic expectation')
        checks.append({'case_id': row['case_id'], 'expectation': expected['selection_kind'],
                       'diagnostic_check_passed': passed, 'valid_selection': valid,
                       'selected_count': len(selected),
                       'missing_required_record_ids': [cid for cid in expected.get('required_record_ids', []) if cid not in selected]})
    return {'checks': checks, 'diagnostic_checks_passed': sum(c['diagnostic_check_passed'] for c in checks),
            'checks_total': len(checks), 'semantic_quality_validated': False,
            'independent_or_human_validation': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    info = prepare(p.parse_args().output)
    print(json.dumps({k: info[k] for k in ('purpose', 'model_calls', 'request_input_tokens')}, indent=2))
