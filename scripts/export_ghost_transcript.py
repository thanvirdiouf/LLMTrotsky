#!/usr/bin/env python3
"""Export verified displayed answers for human review, without model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from grounded_chat import ROOT,fingerprint
from verify_ghost_smoke import frozen_payloads


def export(version,root=ROOT):
    if not re.fullmatch(r'ghost_smoke_v[1-9][0-9]*',version):raise ValueError('Invalid smoke version')
    run=root/'evaluation/runs'/version
    verified=json.loads((run/'verification.json').read_text())
    result=run/'download/ghost_smoke_results'
    raw=(result/'answers.jsonl').read_bytes()
    if verified['status']!='verified' or hashlib.sha256(raw).hexdigest()!=verified['answers_sha256']:
        raise ValueError('Displayed answer journal is not the verified one')
    bundle=run/'download/ghost_smoke_bundle'
    payloads=frozen_payloads(root=root,version=version)
    for name in ('collection.json','model_cases.json'):
        if (bundle/name).read_bytes()!=payloads[name]:raise ValueError('Transcript source bundle differs from frozen submission')
    collection=json.loads((bundle/'collection.json').read_text())
    spec=json.loads((bundle/'model_cases.json').read_text())
    packets={p['id']:p for p in collection['packets']}
    questions={c['id']:[c['question']] for c in spec['cases']}
    questions['ghost_conversation']=spec['conversation']
    rows=[json.loads(line) for line in raw.decode().splitlines()]
    lines=[f'# {version}: displayed-answer transcript','',
        'Research output for source-based review. Model approval is not proof of factual accuracy.',
        'Questions and replies below are the exact frozen evaluation text, including failures.',
        f"Persona reference date: {collection['reference_date']}. Historical cutoff: none.",
        f"Verified answer journal SHA-256: `{verified['answers_sha256']}`.",'']
    for row in rows:
        if fingerprint(row['response'])!=row['response_sha256']:raise ValueError('Answer text hash differs')
        question=questions[row['case_id']][row['turn']-1]
        audit=row.get('mode')=='reviewer_calibration'
        lines.extend([f"## {row['key']}",'',f"Status: `{row['status']}`; stage calls: {row['model_calls']}.",'',
                      'User: '+question,''])
        if audit:lines.extend(['Candidate test input (may be deliberately false; never a trusted chatbot reply):','',row['audit_candidate_display'],''])
        lines.extend(['Auditor diagnostic output:' if audit else 'Displayed reply:','',row['response'],''])
        if row.get('validation_error'):lines.extend(['Application error: '+row['validation_error'],''])
        for rejected in row.get('rejected_segments',[]):lines.extend(['Filtered candidate: '+rejected['reason'],''])
        cited=[pid for pid in packets if '['+pid+']' in (row['audit_candidate_display'] if audit else row['response'])]
        direct=row.get('arm')=='direct'
        supplied={m['packet']['id'] for m in row['retrieval']['matches']}
        if cited:
            lines.extend(['Sources supplied for the test candidate (citation does not validate its text):' if audit else 'Source references written by the unchecked model (not verified):' if direct else 'Supplied source context for generated reading (not sentence-level proof):' if row.get('application_mode')=='experimental_source_reading' else 'Sources cited by the renderer:',''])
            for pid in cited:
                packet=packets[pid];interval=packet['composition_date_interval']
                observed='; observed '+packet['observed_date'] if 'observed_date' in packet else ''
                kinds=', '.join(e.get('kind') or ('dated source block '+str(e['evidence_block_id'])
                    if 'evidence_block_id' in e else 'source date evidence') for e in packet['composition_date_evidence'])
                outside=' **Not in the supplied context.**' if pid not in supplied else ''
                lines.append(f"- [{pid}]({packet['url']}): {packet['author']}; {interval['earliest']}–{interval['latest']} ({kinds}){observed}.{outside}")
            lines.append('')
        unknown=sorted(set(re.findall(r'\[([a-zA-Z0-9_]+)\]',row['response']))-packets.keys())
        if direct and unknown:lines.extend(['Unknown source references: '+', '.join(unknown)+'.',''])
    for pid,packet in packets.items():lines.append(f"[{pid}]: {packet['url']}")
    path=run/'TRANSCRIPT.md';path.write_text('\n'.join(lines)+'\n')
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',required=True)
    print(export(parser.parse_args().version))
