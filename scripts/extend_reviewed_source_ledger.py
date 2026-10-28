"""Offline opt-in extension of a pinned existing reviewed ledger snapshot."""
import argparse
import json
from pathlib import Path
from build_ghost_candidate_probe import ROOT, enc, sha
from build_reviewed_source_ledger import extend, save
from grounded_chat import fingerprint
from grounded_chat_v2 import EvidenceError

PLAN = 'config/ghost_reviewed_precision_plan_v1.json'


def build(root=ROOT, plan_path=PLAN):
    raw_plan = (root/plan_path).read_bytes(); plan = json.loads(raw_plan)
    required = {'schema_version', 'status', 'parent_directory', 'parent_file_sha256',
                'supplements_file', 'supplements_sha256', 'added_claim_count'}
    if (set(plan) != required or type(plan['schema_version']) is not int or plan['schema_version'] != 1
            or type(plan['added_claim_count']) is not int or plan['added_claim_count'] < 1
            or set(plan['parent_file_sha256']) != {'profile.json', 'collection.json', 'ledger.json', 'build_manifest.json'}):
        raise EvidenceError('Invalid pinned reviewed-ledger extension plan')
    parent = root/plan['parent_directory']
    files = {n: (parent/n).read_bytes() for n in plan['parent_file_sha256']}
    if any(sha(raw) != plan['parent_file_sha256'][n] for n, raw in files.items()):
        raise EvidenceError('Pinned parent source/ledger/manifest changed')
    profile, collection, ledger = [json.loads(files[n]) for n in ('profile.json', 'collection.json', 'ledger.json')]
    # Saving must reproduce the original source/profile bytes, not reformat them.
    if enc(profile) != files['profile.json'] or enc(collection) != files['collection.json']:
        raise EvidenceError('Parent profile/collection encoding differs')
    raw_supplements = (root/plan['supplements_file']).read_bytes()
    if sha(raw_supplements) != plan['supplements_sha256']:
        raise EvidenceError('Reviewed supplement snapshot changed')
    supplements = json.loads(raw_supplements); new = extend(collection, ledger, supplements)
    if len(new['claims']) - len(ledger['claims']) != plan['added_claim_count']:
        raise EvidenceError('Reviewed addition count differs')
    manifest = {'purpose': 'offline_reviewed_precision_extension', 'model_calls': 0,
                'active_model_changed': False, 'selector_or_runtime_changed': False,
                'source_packets_changed': False, 'original_questions_or_expectations_changed': False,
                'plan_path': str(plan_path), 'plan_sha256': sha(raw_plan),
                'parent_directory': plan['parent_directory'], 'parent_file_sha256': plan['parent_file_sha256'],
                'supplements_sha256': sha(raw_supplements),
                'builder_sha256': sha((root/'scripts/extend_reviewed_source_ledger.py').read_bytes()),
                'profile_sha256': sha(files['profile.json']), 'collection_sha256': sha(files['collection.json']),
                'base_ledger_fingerprint': fingerprint(ledger), 'ledger_fingerprint': fingerprint(new),
                'original_claims_preserved': len(ledger['claims']),
                'added_claim_ids': new['coverage_extension']['added_claim_ids'],
                'semantic_support_independently_verified': False,
                'review': 'Unblinded implementing-agent source curation; not independent or human validation'}
    return profile, collection, new, manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--plan', default=PLAN)
    p.add_argument('--output', required=True, type=Path); args = p.parse_args()
    values = build(plan_path=args.plan); save(args.output, *values)
    print(json.dumps({'original_claims_preserved': values[3]['original_claims_preserved'],
                      'added_claims': len(values[3]['added_claim_ids']), 'model_calls': 0,
                      'active_model_changed': False}, indent=2))
