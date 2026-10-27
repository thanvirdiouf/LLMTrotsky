"""Opt-in source-record rendering; no generated factual wording or audit gate."""
from copy import deepcopy
from urllib.parse import urlparse
from grounded_chat import fingerprint
from grounded_chat_v2 import EvidenceError,parse_object,validate_ledger

MAX_SELECTED=12


def records_for_packets(collection,ledger,packet_ids):
    if fingerprint(collection['packets'])!=collection['packets_sha256']:raise EvidenceError('Source packet content changed')
    validate_ledger(collection,ledger)
    if (not isinstance(packet_ids,list) or not 1<=len(packet_ids)<=2 or any(not isinstance(i,str) for i in packet_ids)
            or len(set(packet_ids))!=len(packet_ids)):
        raise EvidenceError('Require one or two distinct source packet IDs')
    packets={p['id']:p for p in collection['packets']}
    if not set(packet_ids)<=packets.keys():raise EvidenceError('Unknown source packet')
    result={}
    for claim in ledger['claims']:
        if claim['packet_id'] not in packet_ids:continue
        packet=packets[claim['packet_id']];blocks={b['block_id']:b for b in packet['blocks']}
        if urlparse(packet['url']).scheme not in ('http','https'):raise EvidenceError('Unsupported source URL')
        refs=[]
        for anchor in claim['anchors']:
            block=blocks[anchor['block_id']]
            refs.append({'block_id':block['block_id'],'speaker':block['speaker'],'role':block['role'],
                'anchor_text':anchor['text'],'anchor_start':block['text'].index(anchor['text']),
                'block_text_fingerprint':fingerprint(block['text'])})
        item={'id':claim['id'],'statement':claim['statement'],'statement_fingerprint':fingerprint(claim['statement']),
            'status':claim['status'],'attribution':claim['attribution'],'domain':claim['domain'],
            'source':{'packet_id':packet['id'],'title':packet['title'],'author':packet['author'],'url':packet['url'],
                'composition_date_interval':deepcopy(packet['composition_date_interval']),
                'composition_date_evidence':deepcopy(packet['composition_date_evidence']),
                'context_boundary':packet['context_boundary'],'packet_fingerprint':fingerprint(packet),'block_refs':refs},
            'semantic_support_independently_verified':False}
        if 'observed_date' in packet:item['source']['observed_date']=packet['observed_date']
        item['record_fingerprint']=fingerprint(item);result[item['id']]=item
    return result


def select_ids(text,records):
    selection=parse_object(text)
    if set(selection)!={'claim_ids'} or not isinstance(selection['claim_ids'],list):raise EvidenceError('Selection may contain claim IDs only')
    ids=selection['claim_ids']
    if (len(ids)>MAX_SELECTED or any(not isinstance(i,str) or i not in records for i in ids)
            or len(ids)!=len(set(ids))):raise EvidenceError('Unknown, duplicate or over-limit source-record selection')
    return list(ids)


def render_selection(ids,records):
    # Round-trip the narrow selection validator even for callers supplying IDs.
    import json
    ids=select_ids(json.dumps({'claim_ids':ids}),records);parts=[]
    for cid in ids:
        record=records[cid];bound={k:v for k,v in record.items() if k!='record_fingerprint'}
        if fingerprint(bound)!=record['record_fingerprint'] or fingerprint(record['statement'])!=record['statement_fingerprint']:
            raise EvidenceError('Reviewed source record changed')
        source=record['source'];date=source['composition_date_interval'];when=date['earliest']
        if when!=date['latest']:when+=' to '+date['latest']
        speakers=list(dict.fromkeys(r['speaker'] for r in source['block_refs']))
        parts.append(record['statement']+' ['+source['packet_id']+']\n'
            +'Source: '+source['author']+'; '+when+'. Attribution: '+record['attribution']
            +'. Status: '+record['status']+'. Anchor speakers: '+', '.join(speakers)+'.')
    response='Reviewed source summaries:\n\n'+'\n\n'.join(parts) if parts else 'No reviewed source summaries were selected. The passages may contain additional relevant evidence.'
    return {'response':response,'response_fingerprint':fingerprint(response),'selected_claim_ids':ids,
        'rendered_claim_ids':list(ids),'omitted_selected_claim_ids':[],
        'source_record_fingerprints':{i:records[i]['record_fingerprint'] for i in ids},
        'source_records':[deepcopy(records[i]) for i in ids],
        'generated_factual_text_used':False,'model_audit_performed':False,'selection_relevance_verified':False,
        'requested_task_coverage_verified':False,'semantic_guarantee':False,'accepted_claim_ids':[],'memory_updates':[]}
