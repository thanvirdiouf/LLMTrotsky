#!/usr/bin/env python3
"""Validate all fresh comparison outputs and publish blank condition-blinded review rows.

This local postprocessing never calls a model, edits the frozen holdout, or assigns
quality scores. Response hashes detect accidental mixups; they do not independently
authenticate GPU execution or historical accuracy.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile

if __package__:
    from . import kaggle_checkpoint_compare as runner
    from .build_kaggle_checkpoint_compare import RUNTIME_FIELDS
else:
    import kaggle_checkpoint_compare as runner
    from build_kaggle_checkpoint_compare import RUNTIME_FIELDS

PROJECT = Path(__file__).resolve().parents[1]
SCORE_FIELDS = ('chronology', 'correction', 'evidence_calibration', 'answer_detail',
                'critical_failure', 'review_notes')
REQUIRED_CHECKS = ('fresh_base_has_no_adapter', 'fresh_base_complete',
                   'step_50_exact_adapter_tensors', 'step_50_complete',
                   'step_500_fresh_base_reload', 'step_500_exact_adapter_tensors',
                   'step_500_complete')


def _json(payload):
    def reject(value):
        raise ValueError('Non-finite JSON: ' + value)
    return json.loads(payload, parse_constant=reject)


def _jsonl(payload):
    values = [_json(line) for line in payload.splitlines()]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError('Expected JSONL objects')
    return values


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def validate_run(run_dir, *, project=PROJECT):
    """Return verified local cases and all three full response journals."""
    run_dir, project = Path(run_dir), Path(project)
    fresh = project / 'evaluation/fresh_holdout_v1'
    protocol_bytes = (fresh / 'protocol.json').read_bytes()
    protocol = _json(protocol_bytes)
    for name, expected in protocol['files'].items():
        if Path(name).name != name or runner.digest((fresh / name).read_bytes()) != expected:
            raise ValueError('Frozen fresh evaluation differs: ' + name)
    if runner.digest((fresh / 'specifications.json').read_bytes()) != protocol['specifications_sha256']:
        raise ValueError('Frozen fresh specifications differ')
    input_bytes = (fresh / 'model_inputs.jsonl').read_bytes()
    if (run_dir / 'model_inputs.jsonl').read_bytes() != input_bytes:
        raise ValueError('Downloaded model inputs differ from the frozen local inputs')
    runtime_bytes = (run_dir / 'runtime_protocol.json').read_bytes()
    runtime = _json(runtime_bytes)
    pilot_bytes = (project / 'training/runs/kaggle_361474126/run_manifest.json').read_bytes()
    expected_runtime = {key: protocol[key] for key in RUNTIME_FIELDS}
    expected_runtime.update(files={'model_inputs.jsonl': protocol['files']['model_inputs.jsonl']},
                            source_protocol_sha256=runner.digest(protocol_bytes),
                            pilot_run_manifest_sha256=runner.digest(pilot_bytes))
    if runtime != expected_runtime:
        raise ValueError('Runtime protocol differs from the frozen local projection')
    inputs = runner.validate_payloads(input_bytes, runtime, _json(pilot_bytes))
    cases = _jsonl((fresh / 'cases.jsonl').read_bytes())
    if len(cases) != len(inputs) or [r['case_id'] for r in cases] != [r['case_id'] for r in inputs]:
        raise ValueError('Reviewer cases differ from the frozen model-input order')
    for case, model_input in zip(cases, inputs):
        if any(case.get(key) != value for key, value in model_input.items()):
            raise ValueError('Reviewer prompt differs from model input')
        evidence = case.get('reference_evidence')
        if (not isinstance(evidence, str) or runner.digest(evidence.encode()) != case.get('evidence_sha256')
                or not isinstance(case.get('reference_checks'), list)
                or any(not isinstance(check, str) for check in case['reference_checks'])):
            raise ValueError('Reviewer evidence/checklist differs from frozen provenance')
    manifest = _json((run_dir / 'run_manifest.json').read_bytes())
    comparison = _json((run_dir / 'comparison_manifest.json').read_bytes())
    identity = manifest.get('identity')
    if not isinstance(identity, dict):
        raise ValueError('Missing run identity')
    identity_sha = runner.fingerprint(identity)
    if (manifest.get('status') != 'complete' or manifest.get('stage') != 'complete'
            or manifest.get('identity_sha256') != identity_sha
            or any(manifest.get('checks', {}).get(key) is not True for key in REQUIRED_CHECKS)
            or not all(value is True for value in manifest.get('checks', {}).values())
            or manifest.get('quality_scores_assigned') is not False):
        raise ValueError('Run is incomplete or identity/checks differ')
    generation = identity.get('generation')
    if (identity.get('conditions') != list(runner.CONDITIONS)
            or identity.get('model') != protocol['model']
            or identity.get('packages') != protocol['packages']
            or identity.get('chat_template_sha256') != protocol['chat_template_sha256']
            or identity.get('pilot_identity_sha256') != protocol['pilot_identity_sha256']
            or identity.get('source_protocol_sha256') != runner.digest(protocol_bytes)
            or identity.get('runtime_protocol_sha256') != runner.digest(runtime_bytes)
            or identity.get('model_inputs_sha256') != runner.digest(input_bytes)
            or identity.get('pilot_run_manifest_sha256') != runner.digest(pilot_bytes)
            or identity.get('enable_thinking') is not False or identity.get('seed') != 42
            or identity.get('dtype') != 'torch.float16' or identity.get('gpu_ids_used') != [0]
            or identity.get('kbit_preparation_training') is not False
            or identity.get('attention_implementation') != 'sdpa'
            or identity.get('gpu') != protocol['pilot_identity']['gpu']
            or identity.get('quantization') != {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_use_double_quant': True, 'compute_dtype': 'torch.float16'}
            or not isinstance(generation, dict)
            or any(type(generation.get(key)) is not type(value) or generation[key] != value
                   for key, value in protocol['generation'].items())
            or identity.get('generation_sha256') != runner.fingerprint(generation)):
        raise ValueError('Run policy/inputs/generation differ from the frozen comparison')
    # Recovery checkpoint manifests use a canonical JSON newline, unlike the
    # runtime fingerprint helper. Verify their declared pilot identity too.
    checkpoint_identity_sha = runner.digest((json.dumps(protocol['pilot_identity'],
            sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode())
    for condition, step in [('step_50', 50), ('step_500', 500)]:
        checkpoint = identity.get('checkpoints', {}).get(condition, {})
        if (checkpoint.get('step') != step
                or checkpoint.get('identity_sha256') != checkpoint_identity_sha
                or any(not _sha256(checkpoint.get(key)) for key in ['manifest_sha256', 'adapter_sha256', 'identity_sha256'])):
            raise ValueError('Invalid checkpoint identity: ' + condition)
    if (comparison.get('identity_sha256') != identity_sha
            or comparison.get('conditions') != list(runner.CONDITIONS)
            or comparison.get('generation') != generation
            or comparison.get('generation_sha256') != runner.fingerprint(generation)
            or comparison.get('model_inputs_sha256') != runner.digest(input_bytes)
            or comparison.get('source_protocol_sha256') != runner.digest(protocol_bytes)
            or comparison.get('enable_thinking') is not False
            or comparison.get('fresh_base_before_adapter_attachment') is not True
            or comparison.get('step_500_reloads_base') is not True
            or comparison.get('quality_scores_assigned') is not False):
        raise ValueError('Comparison manifest differs from the verified run identity')
    first_path = run_dir / f'{runner.CONDITIONS[0]}_responses.jsonl'
    first_rows = _jsonl(first_path.read_bytes())
    if [r.get('case_id') for r in first_rows] != [r['case_id'] for r in inputs]:
        raise ValueError('Incomplete fresh baseline response coverage')
    prompt_tokens = {r['case_id']: r.get('input_token_ids') for r in first_rows}
    if (any(not isinstance(tokens, list) or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
            or len(tokens) > protocol['max_input_tokens'] for tokens in prompt_tokens.values())
            or identity.get('prompt_token_ids_sha256') != runner.fingerprint(prompt_tokens)):
        raise ValueError('Recorded tokenized input identity differs')
    results = {}
    for condition in runner.CONDITIONS:
        values = runner.read_evaluation_journal(run_dir / f'{condition}_responses.jsonl',
                        identity_sha, condition, inputs, generation, prompt_tokens)
        if len(values) != len(inputs):
            raise ValueError('Incomplete response coverage: ' + condition)
        for value, case in zip(values, inputs):
            if (value.get('category') != case['category'] or value.get('split') != case['split']
                    or generation['eos_token_id'] in value['token_ids'][:-1]):
                raise ValueError('Response category/split/stop evidence differs')
        results[condition] = values
    source_paths = {'run/' + name: run_dir / name for name in ['run_manifest.json', 'comparison_manifest.json',
                   'runtime_protocol.json', 'model_inputs.jsonl', *[f'{c}_responses.jsonl' for c in runner.CONDITIONS]]}
    source_paths.update({'fresh/' + name: fresh / name for name in ['protocol.json', 'cases.jsonl',
                         'model_inputs.jsonl', 'specifications.json']})
    return cases, results, manifest, source_paths


def _lines(values):
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in values)


def export_review(run_dir, *, project=PROJECT, output_dir=None, seed=20261024, shards=3):
    """Validate every answer first; publish new review files without overwriting scores."""
    if type(seed) is not int or type(shards) is not int or not 1 <= shards <= 24:
        raise ValueError('Review seed must be an integer; shards must be in 1..24')
    run_dir, project = Path(run_dir), Path(project)
    target = Path(output_dir) if output_dir is not None else run_dir / 'review'
    if target.exists():
        raise FileExistsError('Review directory already exists; preserve scores and choose a new directory')
    cases, results, manifest, source_paths = validate_run(run_dir, project=project)
    by_condition = {condition: {r['case_id']: r for r in values} for condition, values in results.items()}
    case_order = sorted(cases, key=lambda case: runner.digest(f'{seed}\0case-order\0{case["case_id"]}'.encode()))
    shard_for = {case['case_id']: min(index * shards // len(case_order), shards - 1) + 1
                 for index, case in enumerate(case_order)}
    # Per-case labels and a separate global shuffle avoid a fixed condition order.
    items = []
    for case in cases:
        for condition in runner.CONDITIONS:
            order = runner.digest(f'{seed}\0answer-order\0{case["case_id"]}\0{condition}'.encode())
            items.append((order, case, condition))
    reviews, keys = [], []
    for index, (_, case, condition) in enumerate(sorted(items, key=lambda item: item[0]), 1):
        value = by_condition[condition][case['case_id']]
        review_id = f'R{index:03d}'
        row = {'review_id': review_id, 'case_id': case['case_id'], 'category': case['category'],
               'messages': case['messages'], 'response': value['response'],
               'finish_reason': value['finish_reason'], 'token_count': value['token_count'],
               'source_url': case['source_url'], 'source_block_ids': case['source_block_ids'],
               'reference_checks': case['reference_checks'], 'reference_evidence': case['reference_evidence'],
               'composition_date_interval': case['composition_date_interval'],
               'composition_date_qualification': case.get('composition_date_qualification', ''),
               'reference_role_notes': case.get('reference_role_notes', ''),
               **{name: '' for name in SCORE_FIELDS}}
        reviews.append(row)
        keys.append({'review_id': review_id, 'case_id': case['case_id'], 'condition': condition,
                     'shard': shard_for[case['case_id']], 'identity_sha256': manifest['identity_sha256'],
                     'response_sha256': runner.digest(value['response'].encode()),
                     'prompt_sha256': value['prompt_sha256']})
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.checkpoint-review-', dir=target.parent))
    try:
        (temporary / 'review_rows.jsonl').write_text(_lines(reviews))
        (temporary / 'review_key.jsonl').write_text(_lines(keys))
        for number in range(1, shards + 1):
            subset = [row for row in reviews if shard_for[row['case_id']] == number]
            (temporary / f'review_shard_{number:02d}.jsonl').write_text(_lines(subset))
        csv_rows=[]
        for row in reviews:
            csv_rows.append({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
        with (temporary / 'review_rows.csv').open('w', newline='') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(reviews[0]))
            writer.writeheader();writer.writerows(csv_rows)
        review_manifest = {'schema_version': 1, 'identity_sha256': manifest['identity_sha256'],
            'seed': seed, 'shards': shards, 'cases': len(cases), 'answers': len(reviews),
            'quality_scores_assigned': False, 'assignment': 'SHA256 seed/domain/case/condition shuffle; neutral IDs; all three answers for each case stay in one shard.',
            'limitations': 'Condition-blinded coding-agent review is not independent historical validation; answer content/style may reveal condition. No automatic quality score is assigned.',
            'review_only_evidence': True,
            'source_sha256': {name: runner.digest(path.read_bytes()) for name, path in source_paths.items()},
            'output_sha256': {path.name: runner.digest(path.read_bytes()) for path in sorted(temporary.iterdir())}}
        (temporary / 'review_manifest.json').write_text(json.dumps(review_manifest, ensure_ascii=False, indent=2) + '\n')
        if target.exists():
            raise FileExistsError('Review directory appeared during export; refusing overwrite')
        temporary.rename(target)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--shards', type=int, default=3)
    args=parser.parse_args()
    print(export_review(args.run_dir, project=args.project, output_dir=args.output,
                        seed=args.seed, shards=args.shards))


if __name__ == '__main__':
    main()
