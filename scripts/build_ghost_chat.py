#!/usr/bin/env python3
"""Build the active present-day ghost evidence collection without changing archives."""
import argparse
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
try:
    from .grounded_chat import ROOT, load_collection, fingerprint
    from .grounded_chat_v2 import validate_ledger
except ImportError:
    from grounded_chat import ROOT, load_collection, fingerprint
    from grounded_chat_v2 import validate_ledger


def build(root=ROOT):
    profile = json.loads((root/'config/persona_ghost.json').read_text())
    if profile['mode'] != 'present_day_ghost' or profile['historical_knowledge_cutoff'] is not None or not profile['fictional_identity']:
        raise ValueError('Expected a clearly fictional present-day persona without a 1940 cutoff')
    reference = date.fromisoformat(profile['reference_date'])
    old = load_collection(root/'evaluation/grounded_chat_v1/collection.json')
    old_ledger = json.loads((root/'config/grounded_chat_claims_v2.json').read_text())
    validate_ledger(old, old_ledger)
    packets, claims = deepcopy(old['packets']), deepcopy(old_ledger['claims'])
    for packet in packets:
        packet['domain'] = 'historical_record'
    for claim in claims:
        claim['domain'] = 'historical_record'
    aliases = deepcopy(old_ledger['topic_aliases'])
    modern = json.loads((root/'config/ghost_contemporary_sources.json').read_text())
    for source in modern['sources']:
        if (source['id'] in aliases or date.fromisoformat(source['updated_date']) > reference
                or date.fromisoformat(source['observed_date']) > reference):
            raise ValueError('Duplicate or future-dated source snapshot')
        interval = {'earliest': source['updated_date'], 'latest': source['updated_date'], 'precision': 'day'}
        snapshot = '\n\n'.join(b['text'] for b in source['blocks'])
        packets.append({'id': source['id'], 'title': source['title'], 'author': source['author'],
            'domain': 'contemporary_evidence', 'url': source['url'], 'document_id': source['id'],
            'source_family_id': source['id'], 'composition_date_interval': interval,
            'composition_date_evidence': [{'kind': 'page_update_date', 'text': source['date_basis']}],
            'observed_date': source['observed_date'], 'blocks': deepcopy(source['blocks']),
            'snapshot_kind': source['snapshot_kind'],
            'snapshot_sha256': hashlib.sha256(snapshot.encode()).hexdigest(),
            'claim_status_annotations': [], 'context_boundary': 'Bounded vendor listing; not evidence of availability or a timeless price.'})
        aliases[source['id']] = source['aliases']
        for claim in source['claims']:
            claims.append({**deepcopy(claim), 'packet_id': source['id'], 'composition_date_interval': interval,
                           'observed_date': source['observed_date'], 'domain': 'contemporary_evidence'})
    collection = {'schema_version': 3, 'mode': profile['mode'], 'reference_date': profile['reference_date'],
        'historical_knowledge_cutoff': None, 'persona_prompt': profile['system_prompt'],
        'display_label': profile['display_label'], 'packets': packets, 'packets_sha256': fingerprint(packets),
        'historical_archive_sha256': old['packets_sha256'], 'profile_sha256': fingerprint(profile)}
    ledger = {'schema_version': 3, 'collection_sha256': collection['packets_sha256'],
        'review': old_ledger['review']+' Contemporary source: '+modern['review'],
        'claims': claims, 'topic_aliases': aliases,
        'reflection_basis_claim_ids': ['newleader_principles', 'weisbord_propaganda']}
    validate_ledger(collection, ledger)
    return profile, collection, ledger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/ghost_chat_v1')
    args = parser.parse_args()
    profile, collection, ledger = build()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('collection.json', collection), ('claims.json', ledger)]:
        path = args.output/name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Existing ghost evidence differs; preserve it and use a new output version')
        path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n')
    print(json.dumps({'mode': profile['mode'], 'reference_date': profile['reference_date'],
        'historical_knowledge_cutoff': None, 'packets': len(collection['packets']), 'claims': len(ledger['claims'])}))
