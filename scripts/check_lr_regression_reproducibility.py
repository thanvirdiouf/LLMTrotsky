#!/usr/bin/env python3
"""After score lock, compare repeated v2 prompts/responses and review consistency."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
 run=ROOT/'evaluation/runs/kaggle_lr1e5_v1';review=run/'results/review';lock=json.loads((review/'score_lock_before_unblinding.json').read_bytes())
 for name,h in lock['files'].items():
  if hashlib.sha256((review/name).read_bytes()).hexdigest()!=h:raise ValueError('Score lock changed')
 new_scores=rows(review/'agent_scores_unblinded.jsonl');old=ROOT/'evaluation/runs/kaggle_data_revision_v2_v1/results';old_scores=rows(old/'review/agent_scores_unblinded.jsonl');reports={}
 for current,previous in [('fresh_base','fresh_base'),('revised_lr1e4_step50','revised_step_50')]:
  a={r['case_id']:r for r in rows(run/'results'/f'{current}_responses.jsonl') if r['split']=='fresh_diagnostic_holdout_v2'};b={r['case_id']:r for r in rows(old/f'{previous}_responses.jsonl')}
  sa={r['case_id']:r for r in new_scores if r['condition']==current and r['split']=='fresh_diagnostic_holdout_v2'};sb={r['case_id']:r for r in old_scores if r['condition']==previous}
  if set(a)!=set(b) or len(a)!=24:raise ValueError('Repeated regression IDs differ')
  fields=['prompt_sha256','generation_sha256','input_token_ids','token_ids','response']
  reports[current]={'cases':24,'exact_match_counts':{f:sum(a[k][f]==b[k][f] for k in a) for f in fields},'changed_response_cases':[k for k in a if a[k]['response']!=b[k]['response']],
   'review_consistency_on_identical_responses':{'cases':sum(a[k]['response']==b[k]['response'] for k in a),
    'main_score_matches':sum(sa[k]['main_score']==sb[k]['main_score'] for k in a if a[k]['response']==b[k]['response']),
    'critical_flag_matches':sum(sa[k]['critical_failure']==sb[k]['critical_failure'] for k in a if a[k]['response']==b[k]['response']),
    'score_differences':[{'case_id':k,'prior_score':sb[k]['main_score'],'current_score':sa[k]['main_score'],'prior_critical':sb[k]['critical_failure'],'current_critical':sa[k]['critical_failure']} for k in a if a[k]['response']==b[k]['response'] and (sa[k]['main_score']!=sb[k]['main_score'] or sa[k]['critical_failure']!=sb[k]['critical_failure'])]}}
 result={'comparisons':reports,'interpretation':'Repeated v2 responses check execution stability; repeat scoring agreement is a descriptive reviewer-consistency diagnostic. Do not overwrite locked scores or pool v2 into fresh evidence.'}
 target=run/'regression_reproducibility.json'
 with target.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
