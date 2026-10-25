#!/usr/bin/env python3
"""Compile an additive, immutable diagnostic holdout; never rewrite pilot artifacts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[1]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def dump(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()

def lines(values):
    return ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values).encode()

def shingles(text, size=8):
    words = re.findall(r'[a-z0-9]+', text.lower())
    return {tuple(words[i:i+size]) for i in range(len(words)-size+1)}

def verify(project=PROJECT):
    """Read-only recheck of the freeze before uploading or reviewing responses."""
    project=Path(project); out=project/'evaluation/fresh_holdout_v1'
    protocol=json.loads((out/'protocol.json').read_text())
    for name,expected in protocol['files'].items():
        if sha((out/name).read_bytes())!=expected:
            raise ValueError('Frozen evaluation checksum mismatch: '+name)
    for relative,key in [('datasets/train_seed.jsonl','training_dataset_sha256'),
                         ('datasets/seed_records.jsonl','seed_records_sha256'),
                         ('training/pilot_lock.json','pilot_lock_sha256')]:
        if sha((project/relative).read_bytes())!=protocol[key]:
            raise ValueError('Original pilot input changed: '+relative)
    if sha((out/'specifications.json').read_bytes())!=protocol['specifications_sha256']:
        raise ValueError('Frozen evaluation specifications changed')
    cases=rows(out/'cases.jsonl'); inputs=rows(out/'model_inputs.jsonl')
    if len(cases)!=24 or len(inputs)!=24 or len({r['case_id'] for r in cases})!=24:
        raise ValueError('Fresh evaluation coverage changed')
    expected_inputs=[{k:r[k] for k in ['case_id','category','split','messages']} for r in cases]
    if inputs!=expected_inputs:
        raise ValueError('Model input projection differs or includes references')
    docs={r['document_id']:r for r in rows(project/'corpus_mia/documents.jsonl')}
    cats={r['document_id']:r for r in rows(project/'corpus_mia/catalog.jsonl')}
    excluded=rows(out/'source_exclusions.jsonl')
    reserved={r['source_family_id'] for r in excluded}
    used={r['source_family_id'] for r in rows(project/'datasets/seed_records.jsonl')}
    old={r['source_family_id'] for r in rows(project/'evaluation/cases.jsonl') if r.get('source_family_id')}
    if reserved & (used|old): raise ValueError('Fresh holdout source leakage')
    for case in cases:
        doc=docs[case['source_document_id']]
        evidence='\n\n'.join(next(b['text'] for b in doc['blocks'] if b['block_id']==i)
                            for i in case['source_block_ids'])
        if (case['source_sha256']!=doc['source_sha256'] or case['reference_evidence']!=evidence
                or sha(evidence.encode())!=case['evidence_sha256']
                or case['source_family_id']!=cats[doc['document_id']]['source_family_id']
                or case['source_family_id'] not in reserved):
            raise ValueError('Fresh evidence/provenance differs: '+case['case_id'])
    return protocol

def build(project=PROJECT):
    project = Path(project)
    out = project / 'evaluation/fresh_holdout_v1'
    if (out / 'protocol.json').exists():
        raise FileExistsError('Holdout v1 is frozen; use a new version rather than overwrite it.')
    specs = json.loads((out / 'specifications.json').read_text())['cases']
    docs = {r['document_id']:r for r in rows(project/'corpus_mia/documents.jsonl')}
    cats = {r['document_id']:r for r in rows(project/'corpus_mia/catalog.jsonl')}
    seed = rows(project/'datasets/seed_records.jsonl')
    old = rows(project/'evaluation/cases.jsonl')
    persona = json.loads((project/'config/persona.json').read_text())
    families = {r['source_family_id'] for r in seed}
    old_families = {r['source_family_id'] for r in old if r.get('source_family_id')}
    selected_ids = {s['source_document_id'] for s in specs}
    reserved = {cats[key]['source_family_id'] for key in selected_ids}
    if reserved & (families|old_families):
        raise ValueError('Fresh sources overlap training or existing evaluation families')
    # Review automatic grouping with an extra eight-word-shingle scan and known
    # near-duplicate edges. It is a conservative curation aid, not an exhaustive
    # semantic/translation overlap proof.
    text = {key:'\n'.join(b['text'] for b in docs[key]['blocks'] if b['role']=='body_candidate') for key in docs}
    selected_shingles = {key:shingles(text[key]) for key in selected_ids}
    matches=[]
    for key in docs:
        current=shingles(text[key])
        if len(current)<60: continue
        for chosen, sh in selected_shingles.items():
            if key==chosen or len(sh)<60: continue
            common=len(sh & current)
            containment=common/min(len(sh),len(current))
            if common>=60 and containment>=0.6:
                reserved.add(cats[key]['source_family_id'])
                matches.append({'selected':chosen,'matched':key,'shorter_containment_8gram':round(containment,4),'common_shingles':common})
    edges=rows(project/'corpus_mia/duplicate_edges.jsonl')
    changed=True
    while changed:
        changed=False
        for edge in edges:
            left,right=cats[edge['left']]['source_family_id'],cats[edge['right']]['source_family_id']
            if left in reserved or right in reserved:
                before=len(reserved); reserved.update([left,right]);changed |= len(reserved)!=before
    if reserved & (families|old_families):
        raise ValueError('Fresh source duplicate closure overlaps training or old evaluation')
    cases=[]
    for spec in specs:
        doc=docs[spec['source_document_id']]; cat=cats[doc['document_id']]
        parsed_date=doc['composition_date_interval']
        date=parsed_date
        date_qualification='As documented in cached source metadata.'
        if doc['document_id']=='mia_f90cf291da6b39369c2b':
            # The editor explicitly calls this a probable May-or-June date;
            # the old parser narrowed that uncertainty to June incorrectly.
            date={'earliest':'1938-05-01','latest':'1938-06-30','precision':'range','raw':'probable May or June 1938'}
            date_qualification='Probable editorial dating of an undated manuscript; not an exact author dateline. Cached source block 7.'
        if not date or date['latest']>persona['historical_as_of']:
            raise ValueError('Missing or post-cutoff authored composition evidence')
        blocks={b['block_id']:b for b in doc['blocks']}
        evidence='\n\n'.join(blocks[i]['text'] for i in spec['source_block_ids'])
        messages=[{'role':'system','content':persona['system_prompt']},{'role':'user','content':spec['question']}]
        if any(messages==r['messages'] for r in old+seed): raise ValueError('Reused messages')
        cases.append({**spec,'split':'fresh_diagnostic_holdout','messages':messages,'source_url':doc['url'],'source_sha256':doc['source_sha256'],'source_family_id':cat['source_family_id'],'composition_date_interval':date,'cached_parsed_date_interval':parsed_date,'composition_date_qualification':date_qualification,'composition_date_evidence':doc.get('composition_date_evidence'),'reference_evidence':evidence,'evidence_sha256':sha(evidence.encode()),'review_status':'coding_agent_reviewed_not_independent_historian','reference_role_notes':'In industry blocks3–7 and imperialism blocks5–10, later editors speak, despite parser body_candidate labels. Those blocks are reference-only attribution/chronology traps; never Trotsky training targets.'})
    if len(cases)!=24 or len({r['case_id'] for r in cases})!=24:
        raise ValueError('Expected 24 unique cases')
    inputs=[{k:r[k] for k in ['case_id','category','split','messages']} for r in cases]
    exclusions=[{'document_id':c['document_id'],'source_family_id':c['source_family_id'],'url':c['url'],'split':'fresh_holdout_v1','training_prohibited':True} for c in cats.values() if c['source_family_id'] in reserved]
    payloads={'cases.jsonl':lines(cases),'model_inputs.jsonl':lines(inputs),'source_exclusions.jsonl':lines(exclusions),'overlap_review.json':dump({'known_edge_and_8gram_matches':matches,'seed_family_overlap':0,'old_evaluation_family_overlap':0,'limits':'Eight-word-shingle scan + catalog edge closure; not exhaustive alternate-translation detection or a base-pretraining holdout.'})}
    pilot=json.loads((project/'training/pilot_protocol.json').read_text())
    run=json.loads((project/'training/runs/kaggle_361474126/run_manifest.json').read_text())
    protocol={'schema_version':1,'frozen_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Diagnostic attribution of data defects vs update intensity; no further training authorized by this protocol.','cases':len(inputs),'works':len(selected_ids),'category_counts':dict(Counter(r['category'] for r in cases)),'conditions':['fresh_base','step_50','step_500'],'primary_comparison':['fresh_base','step_50'],'secondary_comparison':['step_50','step_500'],'step_100_policy':'Downloaded and verified but not selected after seeing results; reserved for a separately declared follow-up.','model':pilot['model'],'packages':pilot['packages'],'generation':pilot['evaluation']['generation'],'enable_thinking':False,'seed':42,'max_input_tokens':8192,'chat_template_sha256':pilot['chat_template_sha256'],'pilot_identity':run['identity'],'pilot_identity_sha256':run['identity_sha256'],'pilot_kernel_source':'thanvirdiouf/trotsky-qlora-500-example-pilot/2','files':{k:sha(v) for k,v in payloads.items()},'specifications_sha256':sha((out/'specifications.json').read_bytes()),'training_dataset_sha256':sha((project/'datasets/train_seed.jsonl').read_bytes()),'seed_records_sha256':sha((project/'datasets/seed_records.jsonl').read_bytes()),'pilot_lock_sha256':sha((project/'training/pilot_lock.json').read_bytes()),'training_exclusion':'All fresh prompts, reference evidence, source families/near duplicates, rubrics, generated answers are permanently held out from training.','rubric':{'scores':'Per-case critical_failure 0/1; chronology, correction, evidence_calibration and answer_detail as applicable 0–4. Blank until reviewed.','scale':{'0':'Fails main requirement or invents critical facts.','1':'Major defects; little useful response.','2':'Partly correct but a material omission/error.','3':'Meets main checks with minor issue.','4':'Meets every applicable check with supported detail.'},'critical_failure':'Invented memory/direct quotation/citation; post-cutoff event claimed as personal knowledge; explicitly accepting a false premise central to the answer. Mark evidence, not automatic keyword scoring.','detail':'Score fulfillment of requested components and explanation; do not equate length or EOS with quality.','review':'Use neutral row IDs and separate identity key; disclose that coding-agent review is not independent historian validation.'},'decision_rule':['If step50 materially improves on step500 while retaining base calibration, prioritize fewer updates or a lower learning rate in the next controlled fresh-adapter pilot after data repairs.','If both trained checkpoints introduce similar critical failures relative to base, prioritize data correction/coverage and do not assume another epoch will help.','If base and adapters share failures, investigate source-conditioned prompting/retrieval and evaluation coverage; do not attribute every failure to SFT.','Data corrections and optimizer changes must be tested separately; do not change multiple factors and claim causality.','Do not promote an adapter on brevity or training loss. Inspect paired critical failures and per-work detail.'],'limitations':['24 related cases across six works are a small diagnostic set, not a general historical benchmark.','Checkpoint50 and checkpoint500 differ in examples seen as well as update count; this is a diagnostic learning trajectory, not a randomized learning-rate/data ablation.','Rubrics are agent-authored; no independent historical gold standard claimed.','As soon as responses are inspected, these are development evidence for later decisions; use another reserved evaluation for final claims.']}
    for name,data in payloads.items(): (out/name).write_bytes(data)
    (out/'protocol.json').write_bytes(dump(protocol))
    print(json.dumps({'cases':24,'works':6,'excluded_documents':len(exclusions),'reserved_families':len(reserved),'overlap_matches':matches},indent=2))
    return protocol

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,default=PROJECT)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:
        p=verify(args.project); print('Verified frozen cases:',p['cases'])
    else:
        build(args.project)
