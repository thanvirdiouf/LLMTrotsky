"""Freeze a literal-question-part selection comparison against saved43."""
import argparse
import json
from pathlib import Path
import re
from build_ghost_candidate_probe import ROOT, enc, sha
from build_reviewed_selection import frozen_inputs as parent_inputs
from candidate_audit import input_ids

INPUTS = 'evaluation/ghost_reviewed_coverage_inputs_v1'
DESIGN = 'config/ghost_reviewed_coverage_design_v1.json'


def question_parts(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Require a nonempty original question')
    parts = []; start = 0
    for match in re.finditer(r'[.!?;](?=\s|$)', question):
        parts.append(question[start:match.end()]); start = match.end()
    if start < len(question): parts.append(question[start:])
    if ''.join(parts) != question: raise ValueError('Question text was lost')
    return parts


def make_jobs(parent, design, tokenizer=None):
    if (design['purpose'] != 'reviewed_question_part_selection'
            or design['part_boundary_regex'] != r'[.!?;](?=\s|$)'
            or len(parent) != 5 or len({j['case_id'] for j in parent}) != 5):
        raise ValueError('Question-part design or parent scope differs')
    jobs = []
    for old in parent:
        parts = question_parts(old['question'])
        data = json.loads(old['messages'][1]['content'])
        if data['current_user_request'] != old['question'] or 'request_parts' in data:
            raise ValueError('Original request differs')
        data['request_parts'] = parts
        messages = [{'role': 'system', 'content': design['system_prompt']},
                    {'role': 'user', 'content': json.dumps(data, ensure_ascii=False, separators=(',', ':'))}]
        job = {k: old[k] for k in ('case_id', 'question', 'source_packet_ids', 'available_record_fingerprints')}
        job.update(request_parts=parts, messages=messages)
        if tokenizer is not None: job['input_token_ids'] = input_ids(tokenizer, messages)
        jobs.append(job)
    return jobs


def prepare(output, root=ROOT, tokenizer=None):
    parent, metadata, binding = parent_inputs(root)
    design_raw = (root/DESIGN).read_bytes(); design = json.loads(design_raw)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(root/'evaluation/ghost_reviewed_selector_inputs_v1/tokenizer_assets', local_files_only=True, trust_remote_code=False)
    if sha(tokenizer.chat_template.encode()) != metadata['chat_template_sha256']:
        raise ValueError('Candidate template differs')
    jobs = make_jobs(json.loads(parent['selection_requests.json']), design, tokenizer)
    info = {**metadata, 'purpose': 'reviewed_question_part_inputs_only', 'parent_inputs': binding,
            'parent_requests_sha256': sha(parent['selection_requests.json']), 'design_sha256': sha(design_raw),
            'request_input_tokens': [len(j['input_token_ids']) for j in jobs],
            'request_part_counts': [len(j['request_parts']) for j in jobs],
            'part_presence_is_semantic_coverage': False, 'source_representation_changed': False}
    values = {n: raw for n, raw in parent.items() if n not in ('selection_requests.json', 'input_protocol.json')}
    values.update({'selection_requests.json': enc(jobs), 'input_protocol.json': enc(info), 'selection_design.json': design_raw})
    lock = {'purpose': 'input_freeze_only', 'file_sha256': {n: sha(raw) for n, raw in values.items()},
            'preparer_sha256': sha((root/'scripts/prepare_reviewed_coverage.py').read_bytes()),
            'renderer_sha256': sha((root/'scripts/reviewed_source_display.py').read_bytes())}
    values['input_lock.json'] = enc(lock); output = Path(output)
    for n, raw in values.items():
        if (output/n).exists() and (output/n).read_bytes() != raw:
            raise ValueError('Existing coverage input differs; use a new directory')
    for n, raw in values.items():
        path = output/n; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    return info


def frozen_inputs(root=ROOT, renderer=None):
    directory = root/INPUTS; lock_raw = (directory/'input_lock.json').read_bytes(); lock = json.loads(lock_raw)
    values = {n: (directory/n).read_bytes() for n in lock['file_sha256']}
    if any(sha(raw) != lock['file_sha256'][n] for n, raw in values.items()):
        raise ValueError('Frozen coverage input changed')
    if renderer is None: renderer = (root/'scripts/reviewed_source_display.py').read_bytes()
    if sha(renderer) != lock['renderer_sha256']: raise ValueError('Frozen renderer changed')
    parent, old, binding = parent_inputs(root, renderer=renderer)
    info = json.loads(values['input_protocol.json']); design = json.loads(values['selection_design.json'])
    if (info['purpose'] != 'reviewed_question_part_inputs_only' or info['parent_inputs'] != binding
            or info['parent_requests_sha256'] != sha(parent['selection_requests.json'])
            or info['design_sha256'] != sha(values['selection_design.json'])
            or values['selection_design.json'] != (root/DESIGN).read_bytes()
            or info['model_calls'] != 0 or info['submitted_to_kaggle'] is not False
            or info['part_presence_is_semantic_coverage'] is not False):
        raise ValueError('Coverage design provenance differs')
    for name in parent:
        if name not in ('selection_requests.json', 'input_protocol.json') and values[name] != parent[name]:
            raise ValueError('Original source/ledger/probe/tokenizer changed: '+name)
    jobs = json.loads(values['selection_requests.json'])
    expected = make_jobs(json.loads(parent['selection_requests.json']), design)
    if [{k: v for k, v in j.items() if k != 'input_token_ids'} for j in jobs] != expected:
        raise ValueError('Original question/source/menu or literal decomposition differs')
    if info['request_part_counts'] != [len(j['request_parts']) for j in jobs]:
        raise ValueError('Part counts differ')
    return values, info, {'path': INPUTS, 'input_lock_sha256': sha(lock_raw), 'parent_inputs': binding,
                          'design_sha256': sha(values['selection_design.json'])}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    info = prepare(p.parse_args().output)
    print(json.dumps({k: info[k] for k in ('purpose', 'model_calls', 'request_input_tokens', 'request_part_counts')}, indent=2))
