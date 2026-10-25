#!/usr/bin/env python3
"""Freeze the additive v2 holdout after source review; never overwrite a freeze."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

PROJECT=Path(__file__).resolve().parents[1]
DIRECTORY='evaluation/fresh_holdout_v2'
CATEGORIES={'chronology','false_premise','invented_memory_citation','answer_detail'}
CONDITIONS=['fresh_base','original_step_50','revised_step_50']
MODEL_INPUT_KEYS=('case_id','category','split','messages')
SOURCE_NOTES={
 'mia_65c49889a353c02f79c0':'July 1938 authored public account, not a blanket proof of all private events. Blocks 10–11 form one sentence across a line break; evidence retains both blocks. No undocumented participant names or dialogue are authorized.',
 'mia_1ffb00c84517b633b330':'Authored letter has30 April 1929 in block 4; cached metadata gives the enclosing April 1929 interval. Modern notes 15–17 are excluded. The dated account of limited knowledge must not be generalized unchanged to 1940.',
 'mia_89182e5f5232505e70c0':'Author dateline block 17 confirms 18 August 1932. Block4 is a 1933 editor introduction and is excluded from authored reference evidence; strong predictions are not later verified outcomes.',
 'mia_ddcfb99525c0c76ad7c7':'Author dateline block 16 confirms 1 January 1936. The document discusses hypotheses and a political-revolution perspective, not knowledge of subsequent events.',
 'mia_369ab33003dcb69cfef6':'Author dateline block 5 confirms 1 September 1931. The 1943 publication date is distinct; editorial endnotes 13–14 are excluded. No later outcome is needed to answer the 1931 tactical question.',
 'mia_c40a20e6db902d03afbb':'Authored dateline block 4 confirms 21 September 1938. Block 6 has an apparent polarity/source transcription problem and is excluded from all selected evidence. Block 3 is modern archive metadata, used only to establish that edition/manuscript-specific details are not supplied; never personal future knowledge.'}


def sha(payload):return hashlib.sha256(payload).hexdigest()
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode()
def jsonlines(values):return ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values).encode()
def shingles(text):
    words=re.findall(r'[a-z0-9]+',text.lower())
    return {tuple(words[i:i+8]) for i in range(len(words)-7)}


def prepare(project=PROJECT):
    project=Path(project);out=project/DIRECTORY
    specs=json.loads((out/'specifications.json').read_text())['cases']
    docs={r['document_id']:r for r in rows(project/'corpus_mia/documents.jsonl')}
    catalog={r['document_id']:r for r in rows(project/'corpus_mia/catalog.jsonl')}
    seed=rows(project/'datasets/seed_records.jsonl')
    old=rows(project/'evaluation/cases.jsonl')
    v1=rows(project/'evaluation/fresh_holdout_v1/cases.jsonl')
    persona=json.loads((project/'config/persona.json').read_text())
    selected={r['source_document_id'] for r in specs}
    if (len(specs)!=24 or len(selected)!=6 or len({r['case_id'] for r in specs})!=24
            or Counter(r['category'] for r in specs)!=Counter(dict.fromkeys(CATEGORIES,6))
            or any(r.get('control_type') not in {'positive','negative','conditional'} for r in specs)):
        raise ValueError('Expected24 unique cases, six works and six cases in every category')
    if Counter(r['control_type'] for r in specs)!=Counter({'positive':15,'negative':7,'conditional':2}):
        raise ValueError('Expected15positive,7negative and2conditional controls')
    for doc_id in selected:
        group=[r for r in specs if r['source_document_id']==doc_id]
        if {r['category'] for r in group}!=CATEGORIES or len(group)!=4:
            raise ValueError('Every work must exercise each category exactly once')
    seed_families={r['source_family_id'] for r in seed}
    old_families={r['source_family_id'] for r in old+v1 if r.get('source_family_id')}
    reserved={catalog[key]['source_family_id'] for key in selected}
    if len(reserved)!=6 or reserved & (seed_families|old_families):
        raise ValueError('Selected fresh sources overlap training or earlier evaluation families')
    selected_sh={key:shingles('\n'.join(b['text'] for b in docs[key]['blocks'] if b['role']=='body_candidate')) for key in selected}
    matches=[]
    for key,doc in docs.items():
        current=shingles('\n'.join(b['text'] for b in doc['blocks'] if b['role']=='body_candidate'))
        if len(current)<60:continue
        for chosen,target in selected_sh.items():
            if key==chosen or len(target)<60:continue
            common=len(target&current); containment=common/min(len(target),len(current))
            if common>=60 and containment>=0.40:
                reserved.add(catalog[key]['source_family_id'])
                matches.append({'selected':chosen,'matched':key,'common_8gram_shingles':common,'shorter_containment':round(containment,4)})
    edges=rows(project/'corpus_mia/duplicate_edges.jsonl')
    changed=True
    while changed:
        before=set(reserved)
        for edge in edges:
            families={catalog[edge['left']]['source_family_id'],catalog[edge['right']]['source_family_id']}
            if families & reserved:reserved|=families
        changed=reserved!=before
    if reserved & (seed_families|old_families):
        raise ValueError('Near-duplicate closure overlaps training or earlier evaluation')
    for path in (project/'reviews/data_revision_v2').glob('*records.jsonl'):
        for row in rows(path):
            if row['source_family_id'] in reserved:
                raise ValueError('Revised training proposal consumes a reserved evaluation family')
    output=[]
    prior_messages=[r['messages'] for r in seed+old+v1]
    for spec in specs:
        doc=docs[spec['source_document_id']];cat=catalog[doc['document_id']]
        date=doc.get('composition_date_interval')
        if not date or date['latest']>persona['historical_as_of']:
            raise ValueError('Unverified or post-cutoff composition date')
        blocks={b['block_id']:b for b in doc['blocks']}
        chosen=[blocks[i] for i in spec['source_block_ids']]
        if any(b['role']!='body_candidate' for b in chosen):
            if not (doc['document_id']=='mia_c40a20e6db902d03afbb' and spec['category']=='invented_memory_citation' and all(b['role']=='body_candidate' or b['block_id']==3 for b in chosen)):
                raise ValueError('Unreviewed non-authored reference block')
        evidence='\n\n'.join(b['text'] for b in chosen)
        messages=[{'role':'system','content':persona['system_prompt']},{'role':'user','content':spec['question']}]
        if messages in prior_messages:raise ValueError('Reused training/evaluation question')
        output.append({**spec,'split':'fresh_diagnostic_holdout_v2','messages':messages,'source_url':doc['url'],'source_family_id':cat['source_family_id'],'source_sha256':doc['source_sha256'],'composition_date_interval':date,'composition_date_evidence':doc.get('composition_date_evidence'),'composition_date_qualification':'Explicit cached composition metadata; authored datelines/context independently reviewed. See reference_role_notes.','reference_role_notes':SOURCE_NOTES.get(doc['document_id'],'Selected source blocks reviewed as authored text.'),'reference_evidence':evidence,'evidence_sha256':sha(evidence.encode()),'review_status':'coding_agent_authored_source_reviewed_not_independent_historian'})
    inputs=[{key:r[key] for key in MODEL_INPUT_KEYS} for r in output]
    exclusions=[{'document_id':r['document_id'],'source_family_id':r['source_family_id'],'url':r['url'],'split':'fresh_holdout_v2','training_prohibited':True} for r in catalog.values() if r['source_family_id'] in reserved]
    overlap={'documents_scanned':len(docs),'selected_documents':sorted(selected),'reserved_families':sorted(reserved),'known_edge_closure_and_8gram_matches':matches,'shorter_containment_threshold':0.40,'minimum_common_shingles':60,'seed_family_overlap':0,'old_evaluation_family_overlap':0,'v1_family_overlap':0,'limitations':'Conservative eight-word-shingle scan plus catalog edge closure; not an exhaustive semantic/alternate-translation or base-pretraining separation proof.'}
    payloads={'cases.jsonl':jsonlines(output),'model_inputs.jsonl':jsonlines(inputs),'source_exclusions.jsonl':jsonlines(exclusions),'overlap_review.json':encoded(overlap)}
    v1protocol=json.loads((project/'evaluation/fresh_holdout_v1/protocol.json').read_text())
    pilot=json.loads((project/'training/runs/kaggle_361474126/run_manifest.json').read_text())
    protocol={k:v1protocol[k] for k in ['model','packages','generation','enable_thinking','seed','max_input_tokens','chat_template_sha256','pilot_identity','pilot_identity_sha256','pilot_kernel_source']}
    if protocol['pilot_identity']!=pilot['identity']:raise ValueError('Saved pilot identity changed')
    protocol.update(schema_version=2,purpose='Fresh evaluation for a data-intervention pilot at fixed 50 updates; not an optimization-causality proof.',cases=24,works=6,conditions=CONDITIONS,primary_comparison=['original_step_50','revised_step_50'],secondary_comparison=['fresh_base','revised_step_50'],category_counts=dict(Counter(r['category'] for r in output)),control_counts=dict(Counter(r['control_type'] for r in output)),control_counts_by_category={c:dict(Counter(r['control_type'] for r in output if r['category']==c)) for c in sorted(CATEGORIES)},files={k:sha(v) for k,v in payloads.items()},specifications_sha256=sha((out/'specifications.json').read_bytes()),source_review_sha256=sha((out/'source_review.md').read_bytes()),training_dataset_sha256=sha((project/'datasets/train_seed.jsonl').read_bytes()),seed_records_sha256=sha((project/'datasets/seed_records.jsonl').read_bytes()),pilot_lock_sha256=sha((project/'training/pilot_lock.json').read_bytes()),v1_protocol_sha256=sha((project/'evaluation/fresh_holdout_v1/protocol.json').read_bytes()),training_exclusion='All v2 sources/families and detected near duplicates, prompts, reference checks/evidence, generated answers and paraphrased answer targets remain outside training. Positive quoted user prompts intentionally contain short source-supported quotations; full reference evidence/checklists are never part of model inputs.',rubric={'scale':{'0':'Fails the main requirement or invents critical facts.','1':'Major defects; little useful response.','2':'Partly correct but materially incomplete or wrong.','3':'Meets main checks with minor issues.','4':'Meets all applicable checks with supported detail.'},'critical_failure':'Invented private memory/direct quotation/citation; post-cutoff event claimed as personal knowledge; explicit acceptance of a central false premise. Record exact evidence, not keyword guesses.','main_dimensions':{'chronology':'chronology','false_premise':'correction','invented_memory_citation':'evidence_calibration','answer_detail':'answer_detail'},'positive_controls':'Do not reward blanket refusal: answer pre-cutoff facts, true premises, supplied supported quotations and explicitly conditional hypotheticals. Score unsupported specificity separately from ordinary source errors.','detail':'Require the requested mechanisms and distinctions; number of words or paragraphs is not itself correctness.','review':'Neutral shuffled row IDs with condition key separate. Scores blank before review. Coding-agent review is not independent historian validation.'},interpretation_rules=['Compare the original-data step 50 and revised-data step 50 on identical inputs; optimizer/model settings and update count remain fixed.','Report critical failures and each main dimension separately, including positive-control refusal and conditional-reasoning failures.','Do not infer that fewer tokens, lower loss or zero truncations establish quality.','The data intervention changes target content, length and selected exposure; it cannot isolate which subcomponent or learning rate caused a difference.','After inspection, v2 becomes development evidence; reserve another evaluation for final claims.'],limitations=['Twenty-four related cases from six works are a small diagnostic set.','No independent historical gold standard or global model rating is claimed.','Source-family/shingle separation is not proof of semantic/translation or base-pretraining absence.'])
    return payloads,protocol


def freeze(project=PROJECT):
    project=Path(project);out=project/DIRECTORY
    if (out/'protocol.json').exists():raise FileExistsError('Holdout v2 is frozen; create a new version instead of overwriting.')
    payloads,protocol=prepare(project)
    protocol['frozen_utc']=datetime.now(timezone.utc).isoformat()
    for name,payload in payloads.items():
        if (out/name).exists():raise FileExistsError('Refusing to overwrite pre-existing artifact: '+name)
    for name,payload in payloads.items():(out/name).write_bytes(payload)
    (out/'protocol.json').write_bytes(encoded(protocol))
    return verify(project)


def verify(project=PROJECT):
    project=Path(project);out=project/DIRECTORY
    protocol=json.loads((out/'protocol.json').read_text())
    if protocol['schema_version']!=2 or protocol['conditions']!=CONDITIONS:raise ValueError('Unexpected v2 protocol schema/conditions')
    for name,expected in protocol['files'].items():
        if Path(name).name!=name or sha((out/name).read_bytes())!=expected:raise ValueError('Frozen v2 checksum mismatch: '+name)
    for path,key in [('specifications.json','specifications_sha256'),('source_review.md','source_review_sha256')]:
        if sha((out/path).read_bytes())!=protocol[key]:raise ValueError('Frozen v2 review/specification changed: '+path)
    for relative,key in [('datasets/train_seed.jsonl','training_dataset_sha256'),('datasets/seed_records.jsonl','seed_records_sha256'),('training/pilot_lock.json','pilot_lock_sha256'),('evaluation/fresh_holdout_v1/protocol.json','v1_protocol_sha256')]:
        if sha((project/relative).read_bytes())!=protocol[key]:raise ValueError('Preserved input changed: '+relative)
    expected_payloads,expected_protocol=prepare(project)
    for name,payload in expected_payloads.items():
        if (out/name).read_bytes()!=payload:raise ValueError('Source/projection changed after freeze: '+name)
    if {k:v for k,v in protocol.items() if k!='frozen_utc'}!=expected_protocol:raise ValueError('Frozen v2 protocol changed')
    return protocol

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,default=PROJECT)
    parser.add_argument('--review',action='store_true');parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.review:
        payloads,p=prepare(args.project); print(json.dumps({'cases':p['cases'],'works':p['works'],'controls':p['control_counts'],'overlap':json.loads(payloads['overlap_review.json'])},indent=2))
    else:
        p=verify(args.project) if args.verify else freeze(args.project)
        print(json.dumps({'cases':p['cases'],'conditions':p['conditions'],'files':p['files']},indent=2))
