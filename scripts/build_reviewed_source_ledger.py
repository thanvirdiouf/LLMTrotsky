"""New opt-in mixed-domain ledger; original source/claim snapshots stay intact."""
import argparse
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import re
from build_ghost_candidate_probe import ROOT,enc,sha
from build_ghost_chat import build as build_base
from grounded_chat import fingerprint
from grounded_chat_v2 import EvidenceError,validate_ledger

PLAN='config/ghost_reviewed_ledger_plan_v1.json'


def validate_arithmetic(claim):
    d=claim['derivation']
    if (not isinstance(d,dict) or set(d)!={'kind','anchor_index','operands','result','denominator_phrase','result_unit'}
            or d['kind']!='add_percentage_shares' or d['result_unit']!='percentage_points'
            or type(d['anchor_index']) is not int or not 0<=d['anchor_index']<len(claim['anchors'])
            or not isinstance(d['operands'],list) or len(d['operands'])!=2
            or not isinstance(d['denominator_phrase'],str)
            or any(not isinstance(v,str) or not re.fullmatch(r'\d+(?:\.\d+)?',v) for v in [*d['operands'],d['result']])):
        raise EvidenceError('Invalid percentage derivation')
    text=claim['anchors'][d['anchor_index']]['text']
    if not d['denominator_phrase'] or d['denominator_phrase'] not in text or any(v+'%' not in text for v in d['operands']):
        raise EvidenceError('Arithmetic operands/denominator absent from source anchor')
    values=[Decimal(v) for v in d['operands']];result=Decimal(d['result'])
    if any(not 0<=v<=100 for v in values) or sum(values)!=result or not 0<=result<=100:
        raise EvidenceError('Arithmetic result differs from source operands')
    # Machine arithmetic does not establish that categories are disjoint or
    # that applying the shares to another population is empirically justified.


def extend(collection,ledger,supplements):
    if (set(supplements)!={'schema_version','review','sources'} or type(supplements['schema_version']) is not int or supplements['schema_version']!=1
            or not isinstance(supplements['review'],str) or not supplements['review'].strip()
            or not isinstance(supplements['sources'],list) or not supplements['sources']):
        raise EvidenceError('Reviewed supplements required')
    if fingerprint(collection['packets'])!=collection['packets_sha256']:raise EvidenceError('Base packet content changed')
    validate_ledger(collection,ledger)
    packets={p['id']:p for p in collection['packets']};seen={c['id'] for c in ledger['claims']};source_ids=set();added=[]
    required={'id','statement','status','attribution','anchors','review_kind','review_note'}
    for source in supplements['sources']:
        if set(source)!={'packet_id','packet_fingerprint','claims'} or source['packet_id'] in source_ids:raise EvidenceError('Supplement source differs')
        pid=source['packet_id'];packet=packets.get(pid)
        if packet is None or packet['domain'] not in ('historical_record','contemporary_evidence') or fingerprint(packet)!=source['packet_fingerprint']:
            raise EvidenceError('Supplement source fingerprint/domain differs')
        source_ids.add(pid)
        if not isinstance(source['claims'],list) or not source['claims']:raise EvidenceError('Supplement source claims required')
        for claim in source['claims']:
            if (not isinstance(claim,dict) or not required<=set(claim) or set(claim)-required-{'derivation','scope_block_ids'}
                    or not isinstance(claim['id'],str) or not claim['id'] or claim['id'] in seen
                    or any(not isinstance(claim[k],str) or not claim[k].strip() for k in ('statement','status','attribution','review_note'))
                    or not isinstance(claim['anchors'],list) or not claim['anchors']
                    or any(not isinstance(a,dict) or set(a)!={'block_id','speaker','text'} or not isinstance(a['text'],str) or not a['text'] for a in claim['anchors'])):
                raise EvidenceError('New claim fields/ID/review differ')
            kind=claim['review_kind']
            if kind=='source_paraphrase':
                if set(claim)!=required:raise EvidenceError('Source paraphrase has unexpected derivation metadata')
            elif kind=='bounded_absence':
                if set(claim)!=required|{'scope_block_ids'} or claim['status']!='inference' or claim['scope_block_ids']!=[b['block_id'] for b in packet['blocks']]:
                    raise EvidenceError('Bounded source assessment must retain full reviewed scope and inference status')
            elif kind=='arithmetic_derivation':
                if set(claim)!=required|{'derivation'} or claim['status']!='inference':raise EvidenceError('Arithmetic must be labelled as a derivation')
                validate_arithmetic(claim)
            else:raise EvidenceError('Unknown review kind')
            seen.add(claim['id'])
            item={**deepcopy(claim),'packet_id':pid,'domain':packet['domain'],
                'composition_date_interval':deepcopy(packet['composition_date_interval'])}
            if 'observed_date' in packet:item['observed_date']=packet['observed_date']
            added.append(item)
    new=deepcopy(ledger);new['claims'].extend(added)
    new['review']=new.get('review','')+' Additional opt-in coverage: '+supplements['review']
    new['coverage_extension']={'base_ledger_fingerprint':fingerprint(ledger),'supplements_fingerprint':fingerprint(supplements),
        'added_claim_ids':[c['id'] for c in added],'semantic_support_independently_verified':False}
    validate_ledger(collection,new)
    if new['claims'][:len(ledger['claims'])]!=ledger['claims']:raise EvidenceError('Original claims were rewritten')
    return new


def build(root=ROOT,plan_path=PLAN):
    plan=json.loads((root/plan_path).read_text())
    if (set(plan)!={'schema_version','status','base_source_plan','supplements_file','base_collection_fingerprint','base_ledger_fingerprint','added_claim_count'}
            or type(plan['schema_version']) is not int or plan['schema_version']!=1 or type(plan['added_claim_count']) is not int or plan['added_claim_count']<1):
        raise EvidenceError('Invalid opt-in ledger plan')
    profile,collection,ledger=build_base(root,source_plan=plan['base_source_plan'])
    if fingerprint(collection)!=plan['base_collection_fingerprint'] or fingerprint(ledger)!=plan['base_ledger_fingerprint']:
        raise EvidenceError('Base coverage snapshot differs')
    raw=(root/plan['supplements_file']).read_bytes();supplements=json.loads(raw);new=extend(collection,ledger,supplements)
    if len(new['claims'])-len(ledger['claims'])!=plan['added_claim_count']:raise EvidenceError('Expected additions differ')
    manifest={'status':'opt_in_reviewed_coverage','model_calls':0,'active_model_changed':False,
        'plan_path':str(plan_path),'plan_sha256':sha((root/plan_path).read_bytes()),'supplements_sha256':sha(raw),
        'builder_sha256':sha((root/'scripts/build_reviewed_source_ledger.py').read_bytes()),
        'collection_fingerprint':fingerprint(collection),'base_ledger_fingerprint':fingerprint(ledger),'ledger_fingerprint':fingerprint(new),
        'original_claims_preserved':len(ledger['claims']),'added_claim_ids':new['coverage_extension']['added_claim_ids'],
        'active_pointer_sha256':sha((root/'config/active_chat.json').read_bytes()),
        'review':'Implementing coding-agent unblinded source review, not independent or human validation',
        'source_paraphrases':sum(c.get('review_kind')=='source_paraphrase' for c in new['claims']),
        'bounded_source_assessments':sum(c.get('review_kind')=='bounded_absence' for c in new['claims']),
        'arithmetic_derivations':sum(c.get('review_kind')=='arithmetic_derivation' for c in new['claims'])}
    return profile,collection,new,manifest


def save(output,profile,collection,ledger,manifest):
    output=Path(output)
    files={'profile.json':enc(profile),'collection.json':enc(collection),'ledger.json':enc(ledger),'build_manifest.json':enc(manifest)}
    for name,raw in files.items():
        path=output/name
        if path.exists() and path.read_bytes()!=raw:raise EvidenceError('Existing reviewed coverage differs; use a new directory')
    output.mkdir(parents=True,exist_ok=True)
    for name,raw in files.items():(output/name).write_bytes(raw)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',default=PLAN);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();values=build(plan_path=args.plan);save(args.output,*values)
    print(json.dumps({'original_claims_preserved':values[3]['original_claims_preserved'],'added_claims':len(values[3]['added_claim_ids']),'model_calls':0,'active_model_changed':False},indent=2))
