#!/usr/bin/env python3
"""Verify full seed-audit coverage and consolidate review-only findings."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]

def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(project=PROJECT):
    project=Path(project);out=project/'reviews/targeted_audit_500'
    seed=rows(project/'datasets/seed_records.jsonl')
    audits=[r for f in sorted(out.glob('shard_*.jsonl')) for r in rows(f)]
    audits.sort(key=lambda r:r['index_1based'])
    if len(audits)!=500 or len(seed)!=500 or [r['index_1based'] for r in audits]!=list(range(1,501)):
        raise ValueError('Audit must cover every example exactly once')
    for i,(source,audit) in enumerate(zip(seed,audits),1):
        if (audit['example_id']!=source['example_id'] or not audit['reviewed_all_turns'] or not audit['source_evidence_checked']):
            raise ValueError(f'Missing or mismatched semantic/source review: {i}')
        count=sum(m['role']=='assistant' for m in source['messages'])
        checked=audit.get('assistant_turns_reviewed',audit.get('reviewed_assistant_turn_count'))
        if checked!=count: raise ValueError(f'Unreviewed assistant turn: {i}')
    outputs=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audits)
    (out/'example_audit.jsonl').write_text(outputs)
    proposals=[{'example_id':r['example_id'],'index_1based':r['index_1based'],'disposition':r['disposition'],'findings':r['findings'],'status':'proposal_only_original_training_unchanged','canonical_record_sha256':hashlib.sha256(json.dumps(source,ensure_ascii=False,sort_keys=True).encode()).hexdigest()} for source,r in zip(seed,audits) if r['findings']]
    (out/'correction_proposals.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in proposals))
    words=sorted(len(m['content'].split()) for r in seed for m in r['messages'] if m['role']=='assistant')
    findings=[f for r in audits for f in r['findings']]
    report={'schema_version':1,'generated_utc':datetime.now(timezone.utc).isoformat(),'examples_reviewed':500,'assistant_turns_reviewed':len(words),'all_turns_and_source_evidence_checked':True,'dispositions':dict(Counter(r['disposition'] for r in audits)),'flagged_examples':len(proposals),'findings':len(findings),'finding_categories':dict(Counter(f['category'] for f in findings)),'review_category_status_counts':{k:dict(Counter(r['categories'][k] for r in audits)) for k in audits[0]['categories']},'assistant_word_counts':{'min':min(words),'median':words[len(words)//2],'mean':round(sum(words)/len(words),2),'max':max(words),'bands':dict(Counter('1_40' if n<=40 else '41_120' if n<=120 else 'over_120' for n in words))},'preserved_inputs':{p:sha(project/p) for p in ['datasets/train_seed.jsonl','datasets/seed_records.jsonl','reviews/seed_evidence_lock.json','training/pilot_lock.json','training/pilot_protocol.json']},'shard_sha256':{f.name:sha(f) for f in sorted(out.glob('shard_*.jsonl'))},'limitations':['Parallel coding-agent review, integrated and coverage-checked; not independent historian review.','Flags include uncertain source-date/provenance conflicts and contextual omissions; not all are factual errors in generated answer text.','Category pass/not_tested judgments use different breadth across shards and are not calibrated model scores or estimates of adversarial coverage.','No observed unsupported private-memory invention or post-cutoff target knowledge does not prove that none exists. Two memory/citation category flags concern an inherited publication-name slip and historical addressee leakage.','Short-answer distribution is descriptive; it is not proof that training length caused output collapse.','Proposed corrections remain separate; original training data, evidence lock, pilot protocol and adapter are preserved.','No benchmark responses have been copied into training.']}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['examples_reviewed','assistant_turns_reviewed','dispositions','flagged_examples','findings','finding_categories','assistant_word_counts']},indent=2))
    return report

if __name__=='__main__':build()
