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


def historical_additions(root, path, reference):
    """Only contiguous reviewed original blocks, with verified raw provenance."""
    spec=json.loads(path.read_text())
    docs={d['document_id']:d for d in map(json.loads,(root/'corpus_mia/documents.jsonl').read_text().splitlines())}
    catalog={d['document_id']:d for d in map(json.loads,(root/'corpus_mia/catalog.jsonl').read_text().splitlines())}
    packets,claims,aliases=[],[],{}
    for source in spec['sources']:
        doc=docs[source['document_id']]
        if doc['source_sha256']!=source['source_sha256'] or hashlib.sha256((root/'corpus_mia'/doc['source_html']).read_bytes()).hexdigest()!=source['source_sha256']:
            raise ValueError('Additional historical raw source changed')
        ids=source['block_ids']
        if not ids or ids!=list(range(ids[0],ids[-1]+1)):
            raise ValueError('Additional historical context must be contiguous')
        blocks=[b for b in doc['blocks'] if b['block_id'] in ids]
        if [b['block_id'] for b in blocks]!=ids or fingerprint(blocks)!=source['blocks_sha256']:
            raise ValueError('Additional historical blocks changed')
        interval=doc['composition_date_interval']
        if not interval or date.fromisoformat(interval['latest'])>reference:
            raise ValueError('Additional historical date is unknown or future')
        if any(b['role'] not in {'heading','body_candidate'} for b in blocks):
            raise ValueError('Unexpected additional source role')
        packet={'id':source['id'],'title':doc['title'],'author':source['author'],
            'domain':'historical_record','url':doc['url'],'document_id':doc['document_id'],
            'source_family_id':catalog[doc['document_id']]['source_family_id'],
            'source_sha256':source['source_sha256'],'composition_date_interval':interval,
            'composition_date_evidence':doc['composition_date_evidence'],
            'publication_metadata_external_only':doc['metadata_fields_raw'].get('first published',[]),
            'blocks':[{'block_id':b['block_id'],'role':b['role'],'speaker':source['author'],'text':b['text']} for b in blocks],
            'claim_status_annotations':source['annotations'],'context_boundary':source['context_boundary']}
        for annotation in source['annotations']:
            if annotation['span'] not in next(b['text'] for b in blocks if b['block_id']==annotation['block_id']):
                raise ValueError('Additional source annotation lost its anchor')
        packets.append(packet);aliases[source['id']]=source['aliases']
        claims.extend({**deepcopy(c),'packet_id':source['id'],'composition_date_interval':interval,'domain':'historical_record'} for c in source['claims'])
    return packets,claims,aliases,spec['review']


def historical_claim_supplements(packets,existing_claims,path):
    """Add reviewed claims to identical existing packets, never rewrite history."""
    spec=json.loads(path.read_text())
    if (set(spec)!={'schema_version','review','sources'} or type(spec['schema_version']) is not int or spec['schema_version']!=1
            or not isinstance(spec['review'],str) or not spec['review'].strip()
            or not isinstance(spec['sources'],list) or not spec['sources']):
        raise ValueError('Expected reviewed historical claim supplements')
    by_packet={p['id']:p for p in packets};seen={c['id'] for c in existing_claims}
    sources=set();added=[]
    for source in spec['sources']:
        if not isinstance(source,dict) or set(source)!={'packet_id','packet_fingerprint','claims'}:
            raise ValueError('Invalid supplement source binding')
        pid=source['packet_id'];packet=by_packet.get(pid)
        if (pid in sources or packet is None or packet['domain']!='historical_record'
                or fingerprint(packet)!=source['packet_fingerprint']):
            raise ValueError('Supplement source differs from reviewed historical packet')
        sources.add(pid)
        if not isinstance(source['claims'],list) or not source['claims']:raise ValueError('Supplement claims required')
        for claim in source['claims']:
            if not isinstance(claim,dict) or set(claim)!={'id','statement','status','attribution','anchors'}:
                raise ValueError('Supplement cannot override source metadata')
            if not isinstance(claim['id'],str) or claim['id'] in seen:
                raise ValueError('Supplement claim IDs must be new and unique')
            seen.add(claim['id'])
            added.append({**deepcopy(claim),'packet_id':pid,
                'composition_date_interval':deepcopy(packet['composition_date_interval']),'domain':'historical_record'})
    # Reuse the established statement/anchor/speaker/status validator as well as
    # the packet fingerprint. A source binding alone is not a semantic review.
    collection={'packets':packets,'packets_sha256':fingerprint(packets)}
    validate_ledger(collection,{'collection_sha256':collection['packets_sha256'],
        'claims':deepcopy(existing_claims)+added,'topic_aliases':{pid:[] for pid in by_packet}})
    return added,spec['review']


def build(root=ROOT, source_plan=None):
    if source_plan is None:
        active=root/'config/active_chat.json'
        source_plan=json.loads(active.read_text()).get('source_plan_file') if active.exists() else None
    plan=json.loads((root/source_plan).read_text()) if source_plan else {}
    profile = json.loads((root/plan.get('persona_file','config/persona_ghost.json')).read_text())
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
    extra_review=''
    if plan.get('historical_sources_file'):
        extra_packets,extra_claims,extra_aliases,extra_review=historical_additions(root,root/plan['historical_sources_file'],reference)
        if set(extra_aliases)&set(aliases):raise ValueError('Duplicate additional source ID')
        packets.extend(extra_packets);claims.extend(extra_claims);aliases.update(extra_aliases)
    modern = json.loads((root/plan.get('contemporary_sources_file','config/ghost_contemporary_sources.json')).read_text())
    for source in modern['sources']:
        source_date=source.get('publication_date',source.get('updated_date'))
        if bool(source.get('publication_date'))==bool(source.get('updated_date')):
            raise ValueError('Require one explicit publication or update date')
        if (source['id'] in aliases or date.fromisoformat(source_date) > reference
                or date.fromisoformat(source['observed_date']) > reference):
            raise ValueError('Duplicate or future-dated source snapshot')
        if source_date>source['observed_date']:raise ValueError('Source date is later than observation')
        interval = {'earliest': source_date, 'latest': source_date, 'precision': 'day'}
        snapshot = '\n\n'.join(b['text'] for b in source['blocks'])
        packets.append({'id': source['id'], 'title': source['title'], 'author': source['author'],
            'domain': 'contemporary_evidence', 'url': source['url'], 'document_id': source['id'],
            'source_family_id': source['id'], 'composition_date_interval': interval,
            'composition_date_evidence': [{'kind': 'publication_date' if 'publication_date' in source else 'page_update_date', 'text': source['date_basis']}],
            'observed_date': source['observed_date'], 'blocks': deepcopy(source['blocks']),
            'snapshot_kind': source['snapshot_kind'],
            'snapshot_sha256': hashlib.sha256(snapshot.encode()).hexdigest(),
            'claim_status_annotations': source.get('annotations',[]),
            'context_boundary': source.get('context_boundary','Bounded vendor listing; not evidence of availability or a timeless price.')})
        aliases[source['id']] = source['aliases']
        for claim in source['claims']:
            claims.append({**deepcopy(claim), 'packet_id': source['id'], 'composition_date_interval': interval,
                           'observed_date': source['observed_date'], 'domain': 'contemporary_evidence'})
    if plan.get('historical_claim_supplements_file'):
        supplements,supplement_review=historical_claim_supplements(packets,claims,root/plan['historical_claim_supplements_file'])
        claims.extend(supplements);extra_review+=' Historical claim supplements: '+supplement_review
    collection = {'schema_version': 3, 'mode': profile['mode'], 'reference_date': profile['reference_date'],
        'historical_knowledge_cutoff': None, 'persona_prompt': profile['system_prompt'],
        'display_label': profile['display_label'], 'packets': packets, 'packets_sha256': fingerprint(packets),
        'historical_archive_sha256': old['packets_sha256'], 'profile_sha256': fingerprint(profile)}
    ledger = {'schema_version': 3, 'collection_sha256': collection['packets_sha256'],
        'review': old_ledger['review']+' Contemporary source: '+modern['review']+extra_review,
        'claims': claims, 'topic_aliases': aliases,
        'reflection_basis_claim_ids': plan.get('reflection_basis_claim_ids',['newleader_principles', 'weisbord_propaganda'])}
    if 'selection_anchor_offsets' in plan:
        if type(plan['selection_anchor_offsets']) is not bool:raise ValueError('Selection offset policy must be Boolean')
        ledger['selection_anchor_offsets']=plan['selection_anchor_offsets']
    validate_ledger(collection, ledger)
    by_id={c['id']:c for c in claims}
    if not ledger['reflection_basis_claim_ids'] or any(c not in by_id or by_id[c]['domain']!='historical_record' for c in ledger['reflection_basis_claim_ids']):
        raise ValueError('Interpretive basis must reference reviewed historical claims')
    return profile, collection, ledger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New version directory; defaults to active evidence when no explicit plan is supplied')
    parser.add_argument('--source-plan', type=Path, help='Explicit optional evidence expansion; never changes the active pointer')
    args = parser.parse_args()
    if args.source_plan and not args.output:parser.error('An explicit source plan requires --output to avoid overwriting active evidence')
    output=args.output or ROOT/Path(json.loads((ROOT/'config/active_chat.json').read_text())['collection_file']).parent
    profile, collection, ledger = build(source_plan=args.source_plan)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('collection.json', collection), ('claims.json', ledger)]:
        path = output/name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Existing ghost evidence differs; preserve it and use a new output version')
        path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n')
    print(json.dumps({'mode': profile['mode'], 'reference_date': profile['reference_date'],
        'historical_knowledge_cutoff': None, 'packets': len(collection['packets']), 'claims': len(ledger['claims'])}))
