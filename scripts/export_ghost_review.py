#!/usr/bin/env python3
"""Export verified source/response packets without model verdicts or prior scores."""
import argparse
import hashlib
import json
from pathlib import Path
from grounded_chat import ROOT,fingerprint
from verify_ghost_smoke import frozen_payloads


def build_packet(version,root=ROOT):
    payloads=frozen_payloads(root=root,version=version)
    run=root/'evaluation/runs'/version
    verified=json.loads((run/'verification.json').read_text())
    raw=(run/'download/ghost_smoke_results/answers.jsonl').read_bytes()
    if verified['status']!='verified' or hashlib.sha256(raw).hexdigest()!=verified['answers_sha256']:
        raise ValueError('Review requires the exact verified answer journal')
    for name in ('collection.json','model_cases.json'):
        if (run/'download/ghost_smoke_bundle'/name).read_bytes()!=payloads[name]:
            raise ValueError('Review source bundle differs from frozen submission')
    collection=json.loads(payloads['collection.json']);spec=json.loads(payloads['model_cases.json'])
    packets={p['id']:p for p in collection['packets']}
    questions={c['id']:[c['question']] for c in spec['cases']}
    questions['ghost_conversation']=spec['conversation']
    groups={};used={};mapping=[]
    for number,line in enumerate(raw.decode().splitlines(),1):
        row=json.loads(line)
        if row.get('mode')=='reviewer_calibration':
            raise ValueError('Calibration judgments are not chatbot replies')
        if fingerprint(row['response'])!=row['response_sha256']:
            raise ValueError('Review response hash differs')
        group=('arm',row['arm']) if row.get('arm') else ('dialogue',) if row['case_id']=='ghost_conversation' else ('single',row['case_id'])
        if group not in groups:groups[group]={'group_id':f'G{len(groups)+1:02d}','turns':[]}
        source_ids=[]
        for match in row['retrieval']['matches']:
            source=match['packet'];pid=source['id']
            if pid not in packets or source!=packets[pid]:raise ValueError('Review retrieval source differs from frozen source')
            source_ids.append(pid)
            used[pid]={k:source[k] for k in ('id','title','author','domain','url','composition_date_interval','composition_date_evidence','observed_date','context_boundary','blocks') if k in source}
        rid=f'R{number:02d}'
        groups[group]['turns'].append({'review_id':rid,'question':questions[row['case_id']][row['turn']-1],
            'displayed_response':row['response'],'response_sha256':row['response_sha256'],'source_packet_ids':source_ids})
        mapping.append({'review_id':rid,'run_key':row['key'],'response_sha256':row['response_sha256']})
    packet={'schema_version':1,'purpose':'Source-based review, blinded to pipeline status, model verdicts and prior scores; source identities and questions remain visible.',
        'persona':'Clearly fictional present-day Leon Trotsky; new normative views are allowed, descriptive premises need evidence or an explicit conditional scenario.',
        'reference_date':collection['reference_date'],'source_journal_sha256':verified['answers_sha256'],
        'review_instructions':['Review displayed replies first, using the exact supplied context and actual prior turns within each group.',
            'Separate descriptive support, certainty/authorship/date, relevance, necessary refusal, conversational depth/shortening and voice.',
            'A political preference is not a factual error merely because you disagree. Record ambiguous rhetoric rather than inventing a unique authentic modern opinion.',
            'Save judgments before inspecting drafts, model verdicts or prior scores. Disclose reviewer identity and AI/human limitations.'],
        'source_packets':list(used.values()),'groups':list(groups.values())}
    return packet,{'version':version,'mapping':mapping,'source_journal_sha256':verified['answers_sha256']}


def export(version,root=ROOT):
    packet,mapping=build_packet(version,root)
    out=root/'evaluation/runs'/version/'blind_review';out.mkdir(exist_ok=True)
    for name,value in [('review_packet.json',packet),('private_mapping.json',mapping)]:
        raw=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode();path=out/name
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Existing review packet differs; preserve it')
        path.write_bytes(raw)
    return out/'review_packet.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',required=True)
    print(export(parser.parse_args().version))
