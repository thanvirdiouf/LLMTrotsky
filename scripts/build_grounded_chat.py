#!/usr/bin/env python3
"""Build only the allowlisted, source-locked context packets from the local archive."""
import argparse
import hashlib
import json
from pathlib import Path
import re
try:
    from .grounded_chat import ROOT, MODEL, fingerprint
except ImportError:
    from grounded_chat import ROOT, MODEL, fingerprint


def build(root=ROOT):
    config = json.loads((root/'config/grounded_chat_sources.json').read_text())
    persona = json.loads((root/'config/persona.json').read_text())
    docs = {d['document_id']: d for d in map(json.loads, (root/'corpus_mia/documents.jsonl').read_text().splitlines())}
    catalog = {d['document_id']: d for d in map(json.loads, (root/'corpus_mia/catalog.jsonl').read_text().splitlines())}
    packets = []
    for s in config['sources']:
        d = docs[s['document_id']]
        if d['source_sha256'] != s['source_sha256'] or hashlib.sha256((root/'corpus_mia'/d['source_html']).read_bytes()).hexdigest() != s['source_sha256']:
            raise ValueError('Raw source changed: '+s['id'])
        blocks = [b for b in d['blocks'] if b['block_id'] in s['block_ids']]
        if [b['block_id'] for b in blocks] != s['block_ids'] or fingerprint(blocks) != s['blocks_sha256']:
            raise ValueError('Reviewed context changed: '+s['id'])
        if any(b['role'] not in {'body_candidate', 'heading'} for b in blocks):
            raise ValueError('Unexpected source role')
        interval, evidence = d['composition_date_interval'], d['composition_date_evidence']
        if 'date_override' in s:
            override = s['date_override']
            if d['blocks'][override['evidence_block_id']]['text'] != override['evidence_text']:
                raise ValueError('Session dateline changed')
            interval, evidence = override['interval'], [override]
        if not interval or not interval.get('earliest') or not interval.get('latest') or interval['latest'] > persona['historical_as_of']:
            raise ValueError('Unverified/out-of-bounds composition date')
        for a in s['annotations']:
            if a['span'] not in next(b['text'] for b in blocks if b['block_id'] == a['block_id']):
                raise ValueError('Claim-status annotation lost its anchor')
        context = []
        for b in blocks:
            speaker = re.match(r'^([A-Z][A-Z .-]+):', b['text']) if s['id'] == 'hearing' else None
            context.append({'block_id': b['block_id'], 'role': b['role'],
                            'speaker': speaker.group(1) if speaker else ('quoted signatories' if s['id'] == 'newleader' and b['block_id'] == 9 else s['author']),
                            'text': b['text']})
        packets.append({'id': s['id'], 'title': d['title'], 'author': s['author'],
            'document_id': d['document_id'], 'source_family_id': catalog[d['document_id']]['source_family_id'],
            'url': d['url'], 'source_sha256': d['source_sha256'],
            'composition_date_interval': interval, 'composition_date_evidence': evidence,
            'publication_metadata_external_only': d['metadata_fields_raw'].get('first published', []),
            'blocks': context, 'claim_status_annotations': s['annotations'],
            'context_boundary': 'Complete allowlisted contiguous excerpt; omitted material is not approved evidence.'})
    return {'schema_version': 1, 'historical_as_of': persona['historical_as_of'],
            'persona_prompt': persona['system_prompt'], 'model': MODEL,
            'source_config_sha256': fingerprint(config), 'review': config['review'],
            'scope': config['scope'], 'packets': packets, 'packets_sha256': fingerprint(packets)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/grounded_chat_v1/collection.json')
    args = parser.parse_args()
    value = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n'
    if args.output.exists() and json.loads(args.output.read_text()) != value:
        raise ValueError('Existing collection differs; select a new version/output path')
    args.output.write_text(raw)
    print(json.dumps({'packets': len(value['packets']), 'collection_sha256': value['packets_sha256'], 'output': str(args.output)}))
