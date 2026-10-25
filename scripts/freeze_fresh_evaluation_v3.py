#!/usr/bin/env python3
"""Freeze fresh source-separated LR diagnostic cases, without changing prior freezes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from freeze_fresh_evaluation_v2 import shingles

ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=['fresh_base','revised_lr1e4_step50','revised_lr1e5_step50']
CATEGORIES={'chronology','false_premise','invented_memory_citation','answer_detail'}
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(v):return (json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode()
def jl(v):return ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in v).encode()

def prepare(root=ROOT):
 out=root/'evaluation/fresh_holdout_v3';specs=json.loads((out/'specifications.json').read_bytes())['cases']
 docs={r['document_id']:r for r in rows(root/'corpus_mia/documents.jsonl')};cat={r['document_id']:r for r in rows(root/'corpus_mia/catalog.jsonl')}
 prior=rows(root/'evaluation/cases.jsonl')
 for name in ['fresh_holdout_v1','fresh_holdout_v2']:prior+=rows(root/'evaluation'/name/'cases.jsonl')
 training=rows(root/'datasets/seed_records.jsonl')
 for path in (root/'reviews/data_revision_v2').glob('*records.jsonl'):training+=rows(path)
 forbidden={r['source_family_id'] for r in prior+training if r.get('source_family_id')}
 selected={r['source_document_id'] for r in specs}
 if len(specs)!=24 or len({r['case_id'] for r in specs})!=24 or len(selected)!=6 or Counter(r['category'] for r in specs)!=Counter(dict.fromkeys(CATEGORIES,6)):raise ValueError('Expected 24 cases, six works and six per category')
 for key in selected:
  group=[r for r in specs if r['source_document_id']==key]
  if len(group)!=4 or {r['category'] for r in group}!=CATEGORIES:raise ValueError('Each work must cover every dimension')
 reserved={cat[key]['source_family_id'] for key in selected}
 selected_sh={key:shingles('\n'.join(b['text'] for b in docs[key]['blocks'] if b['role']=='body_candidate')) for key in selected}
 matches=[]
 for key,doc in docs.items():
  current=shingles('\n'.join(b['text'] for b in doc['blocks'] if b['role']=='body_candidate'))
  for chosen,target in selected_sh.items():
   if key==chosen or min(len(current),len(target))<60:continue
   common=len(current&target);fraction=common/min(len(current),len(target))
   if common>=60 and fraction>=.4:
    reserved.add(cat[key]['source_family_id']);matches.append(dict(selected=chosen,matched=key,common_8grams=common,containment=fraction))
 changed=True
 while changed:
  before=set(reserved)
  for edge in rows(root/'corpus_mia/duplicate_edges.jsonl'):
   families={cat[edge[k]]['source_family_id'] for k in ['left','right']}
   if families&reserved:reserved|=families
  changed=reserved!=before
 if reserved&forbidden:raise ValueError('Source-family/near-duplicate overlap: '+str(reserved&forbidden))
 persona=json.loads((root/'config/persona.json').read_bytes());cases=[]
 old_questions={m['content'].strip().casefold() for r in prior+training for m in r.get('messages',[]) if m['role']=='user'}
 for spec in specs:
  doc=docs[spec['source_document_id']];blocks={b['block_id']:b for b in doc['blocks']};date=doc['composition_date_interval']
  if date['latest']>persona['historical_as_of'] or not spec['reference_checks'] or not spec['critical_failure_conditions'] or spec['control_type'] not in {'positive','negative','conditional'}:raise ValueError('Date/checks/control invalid')
  chosen=[blocks[key] for key in spec['source_block_ids']]
  if any(b['role']!='body_candidate' for b in chosen):raise ValueError('Evidence includes non-authored block')
  if spec['question'].strip().casefold() in old_questions:raise ValueError('Question reused')
  evidence='\n\n'.join(b['text'] for b in chosen)
  cases.append(dict(**spec,split='fresh_diagnostic_holdout_v3',messages=[{'role':'system','content':persona['system_prompt']},{'role':'user','content':spec['question']}],source_url=doc['url'],source_family_id=cat[doc['document_id']]['source_family_id'],source_sha256=doc['source_sha256'],composition_date_interval=date,composition_date_evidence=doc['composition_date_evidence'],reference_evidence=evidence,evidence_sha256=sha(evidence.encode()),review_status='single_coding_agent_source_review_not_independent_historian'))
 exclusions=[dict(document_id=r['document_id'],source_family_id=r['source_family_id'],url=r['url'],split='fresh_holdout_v3',training_prohibited=True) for r in cat.values() if r['source_family_id'] in reserved]
 payloads={'cases.jsonl':jl(cases),'model_inputs.jsonl':jl([{k:r[k] for k in ['case_id','category','split','messages']} for r in cases]),'source_exclusions.jsonl':jl(exclusions),'overlap_review.json':enc(dict(selected_documents=sorted(selected),reserved_families=sorted(reserved),near_duplicate_matches=matches,training_and_prior_evaluation_family_overlap=0,documents_scanned=len(docs),limitations='Eight-word-shingle containment plus catalog edge closure cannot rule out semantic/translation leakage or pretraining exposure.'))}
 old=json.loads((root/'evaluation/fresh_holdout_v2/protocol.json').read_bytes())
 protocol={k:old[k] for k in ['model','packages','generation','enable_thinking','seed','max_input_tokens','chat_template_sha256']}
 protocol.update(schema_version=3,purpose='Fresh primary evaluation for controlled 1e-5 versus saved revised-data 1e-4 adapter; v2 is regression only.',cases=24,works=6,conditions=CONDITIONS,primary_comparison=CONDITIONS[1:],category_counts=dict(Counter(r['category'] for r in cases)),control_counts_by_category={c:dict(Counter(r['control_type'] for r in cases if r['category']==c)) for c in sorted(CATEGORIES)},files={k:sha(v) for k,v in payloads.items()},specifications_sha256=sha((out/'specifications.json').read_bytes()),source_review_sha256=sha((out/'source_review.md').read_bytes()),preserved_files={name:sha((root/name).read_bytes()) for name in ['datasets/revision_v2/train_slice.jsonl','datasets/revision_v2/training_lock.json','datasets/train_seed.jsonl','datasets/seed_records.jsonl','training/data_revision_v2/protocol.json','evaluation/fresh_holdout_v1/protocol.json','evaluation/fresh_holdout_v2/protocol.json']},rubric=old['rubric'],interpretation_rules=['Freeze before generating any response. After inspection these cases become development evidence.','Primary results use only v3; report v2 regressions separately and do not pool them as fresh.','Compare main-dimension 0-4 scores, critical cases, positive-control refusals and paired wins/losses.','Useful knowledge requires supported factual/detail answers; shorter outputs and lower loss are not quality evidence.','One seed and one coding-agent reviewer do not establish robust learning-rate causality or general superiority.'],training_exclusion='All reserved families, evaluation prompts/reference checks/evidence and generated answers stay outside training.')
 return payloads,protocol

def verify(root=ROOT):
 out=root/'evaluation/fresh_holdout_v3';p=json.loads((out/'protocol.json').read_bytes());payloads,expected=prepare(root)
 if {k:v for k,v in p.items() if k!='frozen_utc'}!=expected:raise ValueError('Frozen v3 protocol changed')
 for name,b in payloads.items():
  if (out/name).read_bytes()!=b:raise ValueError('Frozen v3 source/projection mismatch: '+name)
 return p

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args();out=ROOT/'evaluation/fresh_holdout_v3'
 if args.verify:p=verify()
 else:
  if (out/'protocol.json').exists():raise FileExistsError('Already frozen')
  payloads,p=prepare();p['frozen_utc']=datetime.now(timezone.utc).isoformat()
  for name,b in payloads.items():
   if (out/name).exists():raise FileExistsError(name)
   (out/name).write_bytes(b)
  (out/'protocol.json').write_bytes(enc(p));verify()
 print(json.dumps({'cases':p['cases'],'conditions':p['conditions'],'control_counts':p['control_counts_by_category']},indent=2))
if __name__=='__main__':main()
