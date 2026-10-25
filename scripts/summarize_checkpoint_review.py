#!/usr/bin/env python3
"""Validate complete blinded agent scores, then reveal condition-level diagnostics."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import argparse

DIM={'chronology':'chronology','false_premise':'correction','invented_memory_citation':'evidence_calibration','answer_detail':'answer_detail'}
CONDITIONS=['fresh_base','step_50','step_500']

def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]

def build(run):
    run=Path(run);review=run/'review'
    original={r['review_id']:r for r in rows(review/'review_rows.jsonl')}
    keys={r['review_id']:r for r in rows(review/'review_key.jsonl')}
    scores=[r for i in range(1,4) for r in rows(review/f'agent_scores_shard_{i:02d}.jsonl')]
    if len(original)!=72 or len(scores)!=72 or len({r['review_id'] for r in scores})!=72 or set(keys)!=set(original):
        raise ValueError('Expected exactly72 unique review rows and scores')
    combined=[]
    for r in scores:
        source=original[r['review_id']];key=keys[r['review_id']]
        if (r['case_id']!=source['case_id'] or r['category']!=source['category']
            or r['response_sha256']!=hashlib.sha256(source['response'].encode()).hexdigest()
            or r['response_sha256']!=key['response_sha256']):raise ValueError('Score/response mismatch')
        for d in DIM.values():
            if r.get(d) is not None and (type(r[d]) is not int or not 0<=r[d]<=4):raise ValueError('Invalid dimension score')
        if r.get(DIM[r['category']]) is None or type(r['critical_failure']) is not int or r['critical_failure'] not in [0,1] or not r['review_notes']:
            raise ValueError('Missing main score/critical flag/evidence notes')
        combined.append({**r,'condition':key['condition'],'main_score':r[DIM[r['category']]],'finish_reason':source['finish_reason'],'token_count':source['token_count']})
    case_ids={r['case_id'] for r in combined}
    models={}
    for c in CONDITIONS:
        rs=[r for r in combined if r['condition']==c]
        if len(rs)!=24 or {r['case_id'] for r in rs}!=case_ids:raise ValueError('Incomplete condition')
        models[c]={'cases':24,'critical_failures':sum(r['critical_failure'] for r in rs),'mean_output_tokens':round(statistics.mean(r['token_count'] for r in rs),2),'truncations':sum(r['finish_reason']=='length' for r in rs),'category_scores':{cat:{'cases':len(ys:=[r for r in rs if r['category']==cat]),'mean_main_score_0_4':round(statistics.mean(r['main_score'] for r in ys),3),'scores':dict(Counter(r['main_score'] for r in ys)),'critical_failures':sum(r['critical_failure'] for r in ys)} for cat in DIM}}
    paired={}
    for a,b in [('fresh_base','step_50'),('step_50','step_500'),('fresh_base','step_500')]:
        left={r['case_id']:r for r in combined if r['condition']==a};right={r['case_id']:r for r in combined if r['condition']==b}
        paired[f'{a}_vs_{b}']={'score_improved':sum(right[k]['main_score']>left[k]['main_score'] for k in case_ids),'score_degraded':sum(right[k]['main_score']<left[k]['main_score'] for k in case_ids),'score_tied':sum(right[k]['main_score']==left[k]['main_score'] for k in case_ids),'new_critical_cases':[k for k in sorted(case_ids) if right[k]['critical_failure'] and not left[k]['critical_failure']],'removed_critical_cases':[k for k in sorted(case_ids) if left[k]['critical_failure'] and not right[k]['critical_failure']]}
    summary={'schema_version':1,'review_method':'Condition labels hidden from three coding-agent reviewers; each case’s three answers scored by the same reviewer; root reveals mapping after all scores. No independent historian/human validation.','main_dimensions':DIM,'models':models,'paired':paired,'limitations':['Small diagnostic set of24cases/sixworks; these scores are descriptive judgments, not a global model rating.','Several implicit premise accommodations were scored0 but conservatively not called explicit critical failures. Critical totals understate the full set of defects.','Answer style can reveal condition despite neutral labels.','Step50and500 differ in both update count and examples seen; trajectory does not isolate causal data/intensity effects.','The inspected holdout is now development evidence for subsequent iteration; reserve fresh cases for final claims.'],'score_source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(review.glob('agent_scores_shard_*.jsonl'))}}
    (review/'agent_scores_unblinded.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(combined,key=lambda r:r['review_id'])))
    (run.parent/'quality_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'models':models,'paired':paired},indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir',type=Path);build(p.parse_args().run_dir)
