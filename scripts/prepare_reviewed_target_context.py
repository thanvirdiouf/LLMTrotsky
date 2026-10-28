"""One input-key counterfactual against saved45; all values and targets fixed."""
import argparse
import json
from pathlib import Path
from build_ghost_candidate_probe import ROOT, enc, sha
from candidate_audit import input_ids
from prepare_reviewed_target import frozen_inputs as parent_inputs

INPUTS = 'evaluation/ghost_reviewed_target_inputs_v2'
DESIGN = 'config/ghost_reviewed_target_context_v1.json'


def make_jobs(parent, design, tokenizer=None):
    old, new = design['old_context_key'], design['new_context_key']
    if (design['purpose'] != 'literal_target_context_label_diagnostic'
            or (old, new) != ('current_user_request', 'original_question_context')
            or len(parent) != 5 or len({j['case_id'] for j in parent}) != 5):
        raise ValueError('Context-label intervention differs')
    jobs = []
    for parent_job in parent:
        job = {k: v for k, v in parent_job.items() if k not in ('messages', 'input_token_ids')}
        system, user = parent_job['messages']
        data = json.loads(user['content'])
        if (system['role'] != 'system' or user['role'] != 'user' or new in data
                or data.get(old) != job['question'] or data.get('selection_target_span') != job['target_span']
                or system['content'].count(old) != 1):
            raise ValueError('Original context/target or system reference differs')
        # Retain key order, all source text and every value; rename one key only.
        renamed = {new if key == old else key: value for key, value in data.items()}
        messages = [{'role': 'system', 'content': system['content'].replace(old, new)},
                    {'role': 'user', 'content': json.dumps(renamed, ensure_ascii=False, separators=(',', ':'))}]
        job['messages'] = messages
        if tokenizer is not None: job['input_token_ids'] = input_ids(tokenizer, messages)
        jobs.append(job)
    return jobs


def prepare(output, root=ROOT, tokenizer=None):
    parent, metadata, binding = parent_inputs(root)
    design_raw = (root/DESIGN).read_bytes(); design = json.loads(design_raw)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(root/'evaluation/ghost_reviewed_target_inputs_v1/tokenizer_assets', local_files_only=True, trust_remote_code=False)
    if sha(tokenizer.chat_template.encode()) != metadata['chat_template_sha256']:
        raise ValueError('Candidate template differs')
    jobs = make_jobs(json.loads(parent['selection_requests.json']), design, tokenizer)
    info = {**metadata, 'purpose': 'reviewed_target_context_inputs_only', 'parent_inputs': binding,
            'parent_requests_sha256': sha(parent['selection_requests.json']), 'context_design_sha256': sha(design_raw),
            'request_input_tokens': [len(j['input_token_ids']) for j in jobs]}
    values = {n: raw for n, raw in parent.items() if n not in ('selection_requests.json', 'input_protocol.json')}
    values.update({'selection_requests.json': enc(jobs), 'input_protocol.json': enc(info), 'context_design.json': design_raw})
    lock = {'purpose': 'input_freeze_only', 'file_sha256': {n: sha(raw) for n, raw in values.items()},
            'preparer_sha256': sha((root/'scripts/prepare_reviewed_target_context.py').read_bytes()),
            'renderer_sha256': sha((root/'scripts/reviewed_source_display.py').read_bytes())}
    values['input_lock.json'] = enc(lock); output = Path(output)
    for name, raw in values.items():
        if (output/name).exists() and (output/name).read_bytes() != raw:
            raise ValueError('Existing counterfactual inputs differ; use a new directory')
    for name, raw in values.items():
        path = output/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    return info


def frozen_inputs(root=ROOT, renderer=None):
    directory = root/INPUTS; lock_raw = (directory/'input_lock.json').read_bytes(); lock = json.loads(lock_raw)
    values = {n: (directory/n).read_bytes() for n in lock['file_sha256']}
    if any(sha(raw) != lock['file_sha256'][n] for n, raw in values.items()):
        raise ValueError('Counterfactual frozen inputs differ')
    if renderer is None: renderer = (root/'scripts/reviewed_source_display.py').read_bytes()
    if sha(renderer) != lock['renderer_sha256']: raise ValueError('Frozen renderer changed')
    parent, _, binding = parent_inputs(root, renderer=renderer)
    info = json.loads(values['input_protocol.json']); design = json.loads(values['context_design.json'])
    if (info['purpose'] != 'reviewed_target_context_inputs_only' or info['parent_inputs'] != binding
            or info['parent_requests_sha256'] != sha(parent['selection_requests.json'])
            or info['context_design_sha256'] != sha(values['context_design.json'])
            or values['context_design.json'] != (root/DESIGN).read_bytes()
            or info['model_calls'] != 0 or info['submitted_to_kaggle'] is not False):
        raise ValueError('Counterfactual design provenance differs')
    for name in parent:
        if name not in ('selection_requests.json', 'input_protocol.json') and values[name] != parent[name]:
            raise ValueError('Original target/source/tokenizer value changed: '+name)
    expected = make_jobs(json.loads(parent['selection_requests.json']), design)
    jobs = json.loads(values['selection_requests.json'])
    if [{k: v for k, v in j.items() if k != 'input_token_ids'} for j in jobs] != expected:
        raise ValueError('More than the context key/reference changed')
    return values, info, {'path': INPUTS, 'input_lock_sha256': sha(lock_raw), 'parent_inputs': binding,
                          'context_design_sha256': sha(values['context_design.json'])}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    info = prepare(p.parse_args().output)
    print(json.dumps({k: info[k] for k in ('purpose', 'model_calls', 'request_input_tokens')}, indent=2))
