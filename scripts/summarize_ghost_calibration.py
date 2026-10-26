#!/usr/bin/env python3
"""Score verified checker diagnostics against frozen AI-reviewed expectations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from build_ghost_smoke import model_cases
from grounded_chat import ROOT,fingerprint
from grounded_chat_v2 import EvidenceError,parse_object,validate_review
from verify_ghost_smoke import frozen_payloads


def classify(case,row):
    if type(case['expected_approved']) is not bool:raise ValueError('Expected decision must be Boolean')
    if row.get('status')=='blocked':return 'blocked',None,[]
    try:
        review=parse_object(row['response'])
        accepted=validate_review(review,case['segments'])
    except EvidenceError:
        # An empty/missing/invalid verdict is not a semantic rejection, even
        # when the application safely rejects the candidate on that basis.
        return 'unscored_judgment',None,[]
    if sorted(accepted)!=row['accepted_segment_ids']:raise ValueError('Parsed decision differs from verified application result')
    approved=len(accepted)==len(case['segments'])
    if approved:outcome='correct_approval' if case['expected_approved'] else 'false_acceptance'
    else:outcome='false_rejection' if case['expected_approved'] else 'correct_rejection'
    return outcome,approved,[v['reason'] for v in review['verdicts']]


def summarize(version,root=ROOT):
    payload=frozen_payloads(root=root,version=version);protocol=json.loads(payload['protocol.json'])
    if protocol['purpose']!='ghost_audit_calibration':raise ValueError('Expected a checker calibration')
    spec_raw=(root/'evaluation'/version/'evaluation_spec.json').read_bytes()
    if hashlib.sha256(spec_raw).hexdigest()!=protocol['evaluation_spec_sha256']:raise ValueError('Frozen expectations changed')
    spec=json.loads(spec_raw)
    if model_cases(spec)!=json.loads(payload['model_cases.json']):raise ValueError('Candidate inputs differ from frozen expectations')
    run=root/'evaluation/runs'/version
    verified=json.loads((run/'verification.json').read_text())
    raw=(run/'download/ghost_smoke_results/answers.jsonl').read_bytes()
    if (verified['status']!='verified' or not verified.get('tokenizer_reconstructed_locally')
            or hashlib.sha256(raw).hexdigest()!=verified['answers_sha256']):
        raise ValueError('Scoring requires exact token-verified results')
    answers=[json.loads(line) for line in raw.decode().splitlines()]
    if [r['case_id'] for r in answers]!=[c['id'] for c in spec['cases']]:raise ValueError('Candidate coverage/order differs')
    rows=[]
    for case,row in zip(spec['cases'],answers):
        if (row['mode']!='reviewer_calibration' or row['turn']!=1 or row['segments']!=case['segments']
                or fingerprint(row['response'])!=row['response_sha256']):raise ValueError('Calibration binding differs')
        outcome,approved,reasons=classify(case,row)
        rows.append({'case_id':case['id'],'expected_approved':case['expected_approved'],
            'actual_approved':approved,'outcome':outcome,'reasons':reasons,
            'candidate_segments_fingerprint':fingerprint(case['segments']),
            'raw_output_sha256':hashlib.sha256(row['response'].encode()).hexdigest(),
            'response_fingerprint':row['response_sha256']})
    byid={r['case_id']:r for r in rows};pairs=[]
    for pair in spec.get('reviewer_only_pairs',[]):
        ids=pair['case_ids']
        if (len(ids)!=2 or len(set(ids))!=2 or any(i not in byid for i in ids)
                or {byid[i]['expected_approved'] for i in ids}!={True,False}):raise ValueError('Invalid pair coverage')
        pairs.append({'pair_id':pair['pair_id'],'case_ids':ids,'both_decisions_match':all(byid[i]['outcome'] in ('correct_approval','correct_rejection') for i in ids)})
    if pairs and sorted(i for p in pairs for i in p['case_ids'])!=sorted(byid):raise ValueError('Pairs must cover every candidate exactly once')
    return {'version':version,'purpose':'Fixed development diagnostic against AI source-review judgments, not human gold or general reliability',
        'answers_sha256':verified['answers_sha256'],'evaluation_spec_sha256':protocol['evaluation_spec_sha256'],
        'counts':dict(Counter(r['outcome'] for r in rows)),'pairs_both_correct':sum(p['both_decisions_match'] for p in pairs),
        'pairs_total':len(pairs),'rows':rows,'pairs':pairs}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',required=True)
    result=summarize(parser.parse_args().version)
    path=ROOT/'evaluation/runs'/result['version']/'CALIBRATION_REVIEW.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('version','counts','pairs_both_correct','pairs_total')},indent=2))
