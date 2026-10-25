#!/usr/bin/env python3
"""Persist explicitly assigned reviewer judgments under response hashes; no auto-scoring."""
import argparse,hashlib,json
from pathlib import Path
DIM={'chronology':'chronology','false_premise':'correction','invented_memory_citation':'evidence_calibration','answer_detail':'answer_detail'}
ROOT=Path(__file__).resolve().parents[1]/'evaluation/runs/kaggle_lr1e5_v1/results/review'
def write_scores(assignments_path,shard):
 assignments=json.loads(Path(assignments_path).read_bytes())
 rows=[json.loads(s) for s in (ROOT/f'review_shard_{shard:02d}.jsonl').read_text().splitlines()]
 if set(assignments)!={r['review_id'] for r in rows}:raise ValueError('Every neutral shard row must have an explicit judgment')
 scored=[]
 for r in rows:
  value=assignments[r['review_id']]
  if len(value)!=4:raise ValueError('Need main score, critical flag, critical types, evidence note')
  main,critical,types,note=value
  if type(main) is not int or not 0<=main<=4 or type(critical) is not int or critical not in (0,1) or bool(types)!=bool(critical) or not isinstance(note,str) or len(note)<20:raise ValueError('Invalid explicit judgment')
  scored.append(dict(review_id=r['review_id'],case_id=r['case_id'],category=r['category'],**{d:(main if d==DIM[r['category']] else None) for d in DIM.values()},critical_failure=critical,critical_failure_types=types,review_notes=note,response_sha256=hashlib.sha256(r['response'].encode()).hexdigest()))
 target=ROOT/f'agent_scores_shard_{shard:02d}.jsonl'
 with target.open('x') as f:
  f.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored))
 print('Preserved explicit judgments:',len(scored),'shard',shard)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('assignments');p.add_argument('shard',type=int);a=p.parse_args();write_scores(a.assignments,a.shard)
