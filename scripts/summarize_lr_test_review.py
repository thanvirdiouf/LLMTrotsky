#!/usr/bin/env python3
"""Validate all controlled-LR neutral agent scores before publishing descriptive diagnostics.

This postprocessor assigns no quality scores. It preserves the blinded inputs and
refuses to replace either output; use new output paths for another summary.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


DIM = {'chronology': 'chronology', 'false_premise': 'correction',
       'invented_memory_citation': 'evidence_calibration', 'answer_detail': 'answer_detail'}
CONDITIONS = ('fresh_base', 'revised_lr1e4_step50', 'revised_lr1e5_step50')
CONTROLS = ('positive', 'negative', 'conditional')
SCORE_FIELDS = (*DIM.values(), 'critical_failure', 'critical_failure_types', 'review_notes')
PAIRS = (('revised_lr1e4_step50', 'revised_lr1e5_step50'),
         ('fresh_base', 'revised_lr1e5_step50'), ('fresh_base', 'revised_lr1e4_step50'))


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def _json(payload):
    def reject_constant(value):
        raise ValueError('Non-finite JSON: ' + value)

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate JSON field: ' + key)
            value[key] = item
        return value

    return json.loads(payload, parse_constant=reject_constant, object_pairs_hook=unique_object)


def _rows(payload, name):
    try:
        values = [_json(line) for line in payload.splitlines()]
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError('Invalid JSONL in ' + name + ': ' + str(error)) from error
    if not all(isinstance(value, dict) for value in values):
        raise ValueError('Expected JSONL objects in ' + name)
    return values


def _index(values, name, expected=144):
    if len(values) != expected:
        raise ValueError(f'Expected exactly {expected} rows in {name}')
    indexed = {}
    for row in values:
        review_id = row.get('review_id')
        if not isinstance(review_id, str) or not review_id.strip() or review_id in indexed:
            raise ValueError('Missing or duplicate review_id in ' + name)
        indexed[review_id] = row
    return indexed


def _metrics(values):
    """Keep means within categories: their ordinal rubrics differ."""
    return {
        'cases': len(values),
        'critical_failures': sum(row['critical_failure'] for row in values),
        'critical_failure_type_counts': dict(sorted(Counter(
            kind for row in values for kind in row['critical_failure_types']).items())),
        'mean_output_tokens': round(statistics.mean(row['token_count'] for row in values), 2)
        if values else None,
        'truncations': sum(row['finish_reason'] == 'length' for row in values),
        'category_scores': {
            category: {
                'cases': len(subset), 'main_dimension': dimension,
                'mean_main_score_0_4': round(statistics.mean(
                    row['main_score'] for row in subset), 3) if subset else None,
                'scores': dict(sorted(Counter(row['main_score'] for row in subset).items())),
                'critical_failures': sum(row['critical_failure'] for row in subset),
            }
            for category, dimension in DIM.items()
            for subset in [[row for row in values if row['category'] == category]]
        },
    }


def _paired(left, right):
    case_ids = sorted(left)
    if set(left) != set(right):
        raise ValueError('Paired comparison has different case IDs')
    return {
        'cases': len(case_ids),
        'score_improved': sum(right[k]['main_score'] > left[k]['main_score'] for k in case_ids),
        'score_degraded': sum(right[k]['main_score'] < left[k]['main_score'] for k in case_ids),
        'score_tied': sum(right[k]['main_score'] == left[k]['main_score'] for k in case_ids),
        'new_critical_cases': [k for k in case_ids
                               if right[k]['critical_failure'] and not left[k]['critical_failure']],
        'removed_critical_cases': [k for k in case_ids
                                   if left[k]['critical_failure'] and not right[k]['critical_failure']],
        'persistent_critical_cases': [k for k in case_ids
                                      if left[k]['critical_failure'] and right[k]['critical_failure']],
    }


def validate_review(run_dir):
    """Return checked unblinded rows and exact hashes of the files consumed."""
    review = Path(run_dir) / 'review'
    payloads = {}

    def read(name):
        if name not in payloads:
            payloads[name] = (review / name).read_bytes()
        return payloads[name]

    score_lock=_json(read('score_lock_before_unblinding.json'))
    required={'review_rows.jsonl','review_key.jsonl',*[f'agent_scores_shard_{i:02d}.jsonl' for i in range(1,4)]}
    if (score_lock.get('reviewer_count')!=1 or score_lock.get('locked_before_unblinding') is not True
        or set(score_lock.get('files',{}))!=required or not score_lock.get('completed_utc')):
        raise ValueError('Missing complete pre-unblinding score lock')
    for name,expected in score_lock['files'].items():
        if _digest(read(name))!=expected:raise ValueError('Pre-unblinding score lock hash mismatch: '+name)

    original = _index(_rows(read('review_rows.jsonl'), 'review_rows.jsonl'), 'review_rows.jsonl')
    keys = _index(_rows(read('review_key.jsonl'), 'review_key.jsonl'), 'review_key.jsonl')
    scores, shard_for_score = [], {}
    for number in range(1, 4):
        name = f'agent_scores_shard_{number:02d}.jsonl'
        shard = _rows(read(name), name)
        _index(shard, name, expected=48)
        scores.extend(shard)
        for row in shard:
            if row['review_id'] in shard_for_score:
                raise ValueError('Duplicate review_id across score shards')
            shard_for_score[row['review_id']] = number
    indexed_scores = _index(scores, 'agent scores')
    if set(original) != set(keys) or set(original) != set(indexed_scores):
        raise ValueError('Unknown or missing review IDs in keys or agent scores')

    # Identity/prompt hashes are optional for synthetic/imported reviews. When
    # available, require complete, well-formed evidence rather than partial IDs.
    for field in ('identity_sha256', 'prompt_sha256'):
        present = [field in key for key in keys.values()]
        if any(present) and not all(present):
            raise ValueError('Incomplete key ' + field)
    identities = set()
    for key in keys.values():
        for field, value in key.items():
            if field.endswith('_sha256') and not _sha256(value):
                raise ValueError('Invalid key SHA256: ' + field)
        if 'identity_sha256' in key:
            identities.add(key['identity_sha256'])
    if len(identities) > 1:
        raise ValueError('Review keys have different run identities')
    identity = next(iter(identities), None)

    source_artifacts = {}
    if (review / 'review_manifest.json').exists():
        manifest = _json(read('review_manifest.json'))
        if not isinstance(manifest, dict):
            raise ValueError('Review manifest must be an object')
        if 'identity_sha256' in manifest and (
                not _sha256(manifest['identity_sha256'])
                or manifest['identity_sha256'] != identity):
            raise ValueError('Review manifest identity differs from keys')
        for field in ('source_sha256', 'output_sha256'):
            hashes = manifest.get(field, {})
            if (not isinstance(hashes, dict)
                    or any(not isinstance(name, str) or not name or not _sha256(value)
                           for name, value in hashes.items())):
                raise ValueError('Invalid manifest ' + field)
            if field == 'source_sha256':
                source_artifacts = dict(hashes)
            else:
                for name, expected in hashes.items():
                    if Path(name).name != name or name in ('.', '..'):
                        raise ValueError('Invalid manifest output path')
                    if _digest(read(name)) != expected:
                        raise ValueError('Review manifest output hash mismatch: ' + name)

    combined, case_metadata, case_shards, prompts = [], {}, {}, {}
    condition_cases = {condition: set() for condition in CONDITIONS}
    for review_id in sorted(original):
        source, key, score = original[review_id], keys[review_id], indexed_scores[review_id]
        case_id, category, control = (source.get(field)
                                      for field in ('case_id', 'category', 'control_type'))
        if (not isinstance(case_id, str) or not case_id.strip()
                or not isinstance(category, str) or category not in DIM
                or not isinstance(control, str) or control not in CONTROLS):
            raise ValueError('Invalid source case/category/control_type: ' + review_id)
        if any(source.get(field, '') not in ('', None) for field in SCORE_FIELDS):
            raise ValueError('Blinded source contains assigned scores: ' + review_id)
        if 'condition' in source:
            raise ValueError('Blinded source contains a condition label')
        if (not isinstance(source.get('response'), str)
                or type(source.get('token_count')) is not int or source['token_count'] < 0
                or source.get('finish_reason') not in ('stop', 'length')):
            raise ValueError('Invalid source response/token/finish evidence: ' + review_id)
        condition = key.get('condition')
        if not isinstance(condition, str) or condition not in CONDITIONS:
            raise ValueError('Unknown condition in review key: ' + review_id)
        if case_id in condition_cases[condition]:
            raise ValueError('Duplicate case within condition: ' + condition)
        condition_cases[condition].add(case_id)
        if key.get('case_id') != case_id:
            raise ValueError('Key/source case mismatch: ' + review_id)
        for field, expected in [('category', category), ('control_type', control)]:
            if field in key and key[field] != expected:
                raise ValueError('Key/source ' + field + ' mismatch: ' + review_id)
            if field in score and score[field] != expected:
                raise ValueError('Score/source ' + field + ' mismatch: ' + review_id)
        if score.get('case_id') != case_id or score.get('category') != category:
            raise ValueError('Score/source case or category mismatch: ' + review_id)
        response_hash = _digest(source['response'].encode('utf-8'))
        if (not _sha256(key.get('response_sha256'))
                or not _sha256(score.get('response_sha256'))
                or score['response_sha256'] != response_hash
                or key['response_sha256'] != response_hash
                or ('response_sha256' in source and source['response_sha256'] != response_hash)):
            raise ValueError('Score/key/response hash mismatch: ' + review_id)
        if 'identity_sha256' in score and score['identity_sha256'] != identity:
            raise ValueError('Score/key identity mismatch: ' + review_id)
        if 'prompt_sha256' in key:
            prompt_hash = key['prompt_sha256']
            if case_id in prompts and prompts[case_id] != prompt_hash:
                raise ValueError('Prompt hash differs across conditions: ' + case_id)
            prompts[case_id] = prompt_hash
        if source.get('split') not in ('fresh_diagnostic_holdout_v3','fresh_diagnostic_holdout_v2'):
            raise ValueError('Unknown fresh/regression split')
        metadata = (category, control, source['split'])
        if case_id in case_metadata and case_metadata[case_id] != metadata:
            raise ValueError('Source category/control_type differs across conditions: ' + case_id)
        case_metadata[case_id] = metadata
        shard = shard_for_score[review_id]
        if 'shard' in key and (type(key['shard']) is not int or key['shard'] != shard):
            raise ValueError('Score is assigned to the wrong shard: ' + review_id)
        if case_id in case_shards and case_shards[case_id] != shard:
            raise ValueError('A case must be scored by the same reviewer in all conditions')
        case_shards[case_id] = shard
        for dimension in DIM.values():
            if dimension not in score or (score[dimension] is not None and (
                    type(score[dimension]) is not int or not 0 <= score[dimension] <= 4)):
                raise ValueError('Invalid dimension score: ' + dimension)
        if score[DIM[category]] is None:
            raise ValueError('Missing main dimension score: ' + review_id)
        critical, kinds = score.get('critical_failure'), score.get('critical_failure_types')
        if type(critical) is not int or critical not in (0, 1):
            raise ValueError('Critical failure flag must be integer 0 or 1')
        if (not isinstance(kinds, list)
                or any(not isinstance(kind, str) or not kind.strip() for kind in kinds)
                or len(kinds) != len(set(kinds)) or bool(kinds) != bool(critical)):
            raise ValueError('Critical failure types must agree with the critical flag')
        if not isinstance(score.get('review_notes'), str) or not score['review_notes'].strip():
            raise ValueError('Missing evidence review notes: ' + review_id)
        combined.append({**score, 'condition': condition, 'control_type': control,
                         'main_score': score[DIM[category]], 'split': source['split'], 'response_sha256': response_hash,
                         'finish_reason': source['finish_reason'], 'token_count': source['token_count']})

    case_ids = set(case_metadata)
    for condition in CONDITIONS:
        values = [row for row in combined if row['condition'] == condition]
        if (len(values) != 48 or condition_cases[condition] != case_ids
                or Counter(row['category'] for row in values) != dict.fromkeys(DIM, 12)):
            raise ValueError('Expected the same 48 cases, twelve per category, in each condition')
    return (combined, {name: _digest(payload) for name, payload in sorted(payloads.items())},
            identity, source_artifacts)


def _write_new_outputs(outputs):
    created = []
    try:
        for path, payload in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as stream:
                created.append(path)
                stream.write(payload)
    except BaseException:
        for path in created:
            path.unlink()
        raise


def summarize_subset(combined):
    models = {}
    for condition in CONDITIONS:
        values = [row for row in combined if row['condition'] == condition]
        models[condition] = _metrics(values)
        models[condition]['control_diagnostics'] = {
            control: _metrics([row for row in values if row['control_type'] == control])
            for control in CONTROLS}
    paired = {}
    for a, b in PAIRS:
        left = {row['case_id']: row for row in combined if row['condition'] == a}
        right = {row['case_id']: row for row in combined if row['condition'] == b}
        comparison = _paired(left, right)
        comparison['category_scores'] = {}
        for category, dimension in DIM.items():
            subset = {k: row for k, row in left.items() if row['category'] == category}
            matched = {k: right[k] for k in subset}
            comparison['category_scores'][category] = {
                **_paired(subset, matched), 'main_dimension': dimension,
                'mean_main_score_delta_0_4': round(statistics.mean(
                    matched[k]['main_score'] - subset[k]['main_score'] for k in subset), 3)}
        comparison['control_diagnostics'] = {}
        for control in CONTROLS:
            subset = {k: row for k, row in left.items() if row['control_type'] == control}
            matched = {k: right[k] for k in subset}
            comparison['control_diagnostics'][control] = {
                **_paired(subset, matched),
                'category_scores': {
                    category: _paired(
                        {k: row for k, row in subset.items() if row['category'] == category},
                        {k: row for k, row in matched.items() if row['category'] == category})
                    for category in DIM}}
        paired[f'{a}_vs_{b}'] = comparison
    return {'models': models, 'paired': paired}


def build(run_dir, *, summary_path=None, unblinded_path=None, reviewer_count=1):
    """Validate and write two new outputs; never modify the blinded review files."""
    if type(reviewer_count) is not int or reviewer_count < 1:
        raise ValueError('reviewer_count must be a positive integer')
    reviewers = ('one coding-agent reviewer' if reviewer_count == 1
                 else f'{reviewer_count} coding-agent reviewers')
    run = Path(run_dir)
    summary_path = Path(summary_path) if summary_path is not None else run.parent / 'quality_summary.json'
    unblinded_path = (Path(unblinded_path) if unblinded_path is not None
                      else run / 'review' / 'agent_scores_unblinded.jsonl')
    if summary_path.resolve() == unblinded_path.resolve():
        raise ValueError('Summary and unblinded output paths must differ')
    for path in (summary_path, unblinded_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError('Output already exists; choose a new output path: ' + str(path))
    combined, source_hashes, identity, source_artifacts = validate_review(run)
    groups={}
    for label,split in [('fresh_primary','fresh_diagnostic_holdout_v3'),('development_regression','fresh_diagnostic_holdout_v2')]:
        subset=[r for r in combined if r['split']==split]
        if len(subset)!=72 or any(sum(r['condition']==c for r in subset)!=24 for c in CONDITIONS):
            raise ValueError('Incomplete fresh/regression group')
        groups[label]=summarize_subset(subset)
    summary = {
        'schema_version': 2,
        'reviewer_count': reviewer_count,
        'review_method': f'Condition labels hidden from {reviewers}; all three '
                         'answers to each case scored in one reviewer shard; mapping revealed only '
                         'after complete score validation. No independent historian/human validation.',
        'identity_sha256': identity, 'main_dimensions': DIM,
        'score_interpretation': 'Ordinal 0–4 category rubrics; means and paired changes are '
                                'descriptive within a category, not a global model rating. '
                                'Paired improvement means the second named condition scored higher.',
        'groups': groups,
        'limitations': [
            'Coding-agent judgments and source review are not independent historian validation; '
            'answer content or style may reveal the condition despite neutral labels.',
            'Only 24 fresh diagnostic cases plus 24 previously inspected regressions; results do not '
            'establish population performance or a global quality rating.',
            'Both adapters use identical data/order/4407 supervised tokens/50 updates and matched initialization; only learning rate changes. One seed does not establish robust causality.',
            'Positive, negative, and conditional controls are reported separately; their category '
            'mixes and small counts must be considered when interpreting differences.',
            'File hashes bind this summary to the supplied review artifacts; they do not '
            'independently authenticate GPU execution or historical accuracy.',
            'Once inspected, this holdout is development evidence; reserve fresh cases for final claims.',
        ],
        'review_source_sha256': source_hashes,
        'score_source_sha256': {name: value for name, value in source_hashes.items()
                                if name.startswith('agent_scores_shard_')},
        'source_artifact_sha256': source_artifacts,
    }
    _write_new_outputs([
        (unblinded_path, ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                                for row in combined)),
        (summary_path, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
    ])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--reviewer-count', type=int, default=1,
                        help='Actual number of coding-agent reviewers (default: 1; shards are not reviewers)')
    parser.add_argument('--summary-output', '--summary', dest='summary_path', type=Path,
                        help='New summary path (default: RUN_DIR.parent/quality_summary.json)')
    parser.add_argument('--unblinded-output', dest='unblinded_path', type=Path,
                        help='New unblinded score path (default: RUN_DIR/review/agent_scores_unblinded.jsonl)')
    args = parser.parse_args()
    summary = build(args.run_dir, summary_path=args.summary_path, unblinded_path=args.unblinded_path,
                    reviewer_count=args.reviewer_count)
    print(json.dumps(summary['groups'], indent=2))


if __name__ == '__main__':
    main()
