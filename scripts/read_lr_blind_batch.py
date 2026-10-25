#!/usr/bin/env python3
"""Display only neutral reviewer rows; never open condition keys/raw journals."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('shard',type=int);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=8);a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'evaluation/runs/kaggle_lr1e5_v1/results/review'
rows=[json.loads(s) for s in (root/f'review_shard_{a.shard:02d}.jsonl').read_text().splitlines()]
assert all('condition' not in r for r in rows)
ids=sorted({r['case_id'] for r in rows})[a.start:a.start+a.count]
for key in ids:
 group=[r for r in rows if r['case_id']==key];first=group[0]
 print('\nCASE',key,first['category'],first['control_type'])
 print('QUESTION',first['messages'][-1]['content'])
 print('CHECKS',json.dumps(first['reference_checks'],ensure_ascii=False))
 for r in group:print('\n',r['review_id'],r['finish_reason'],r['token_count'],'tokens\n'+r['response'])
