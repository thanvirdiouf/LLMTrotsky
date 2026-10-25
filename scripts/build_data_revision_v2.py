#!/usr/bin/env python3
"""Build separately versioned data; preserve original examples and all historical locks."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

if __package__:
    from .build_training_pilot import canonical_sha256, validate_training_payload, validate_contamination, span_hashes, text_sha256, generated_answers, benchmark_inventory
    from .fresh_holdout_guard import guard_training_project,load_fresh_holdout_guard,check_training_records
    from .build_seed_dataset import date_evidence
else:
    from build_training_pilot import canonical_sha256, validate_training_payload, validate_contamination, span_hashes, text_sha256, generated_answers, benchmark_inventory
    from fresh_holdout_guard import guard_training_project,load_fresh_holdout_guard,check_training_records
    from build_seed_dataset import date_evidence

PROJECT=Path(__file__).resolve().parents[1]
SELECTED_REPAIR_INDICES=[36,55,56,66,78,81,82,127,190,488]

def digest(data):return hashlib.sha256(data).hexdigest()
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
def jsonl(rs):return ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs).encode()

def validate_integration_review(project, repairs, new):
    """Approval is an exact frozen review, not an inferred Boolean on a row."""
    project=Path(project);review_dir=project/'reviews/data_revision_v2'
    review=json.loads((review_dir/'integration_review.json').read_bytes())
    expected={r['example_id'] for r in repairs+new}
    approved=review.get('approved_example_ids')
    if (len(repairs)!=21 or len(new)!=40 or len(expected)!=61
            or not isinstance(approved,list) or len(approved)!=61
            or len(set(approved))!=61 or set(approved)!=expected
            or review.get('repair_records_reviewed')!=21
            or review.get('new_records_reviewed')!=40
            or review.get('preserved_original_inputs') is not True):
        raise ValueError('Root integration approval does not cover the exact 61 records')
    names={'repaired_records.jsonl','new_records.jsonl','repair_log.jsonl','source_review.md'}
    hashes=review.get('source_proposals_sha256',{})
    if set(hashes)!=names or any(digest((review_dir/n).read_bytes())!=hashes[n] for n in names):
        raise ValueError('Approved source proposals or review text changed')
    cross=json.loads((review_dir/'repair_cross_review.json').read_bytes())
    cross_rows=cross.get('reviews',[])
    cross_by={r['example_id']:r for r in cross_rows}
    if (cross.get('records')!=21 or cross.get('approved')!=21
            or cross.get('actionable_issues')!=0 or len(cross_rows)!=21
            or set(cross_by)!={r['example_id'] for r in repairs}
            or cross.get('inputs',{}).get('repaired_records_sha256')!=digest((review_dir/'repaired_records.jsonl').read_bytes())):
        raise ValueError('Independent repair cross-review is missing or stale')
    for row in repairs:
        checked=cross_by[row['example_id']]
        # The independent reviewer hashed sorted, Unicode JSON with default
        # separators; the training lock's compact canonical hash is distinct.
        reviewed_hash=digest(json.dumps(row,sort_keys=True,ensure_ascii=False).encode())
        if (checked.get('record_sha256')!=reviewed_hash
                or checked.get('decision')!='approve_for_root_integration'
                or checked.get('reviewed_all_assistant_turns') is not True
                or checked.get('exact_source_evidence_and_hash_checked') is not True
                or checked.get('actionable_issues')!=[]):
            raise ValueError('Repair cross-review does not approve exact record: '+row['example_id'])
    return review

def validate_version(project=PROJECT):
    """Read-only reconstruction of lineage, review, projection and exclusions."""
    project=Path(project);out=project/'datasets/revision_v2';review_dir=project/'reviews/data_revision_v2'
    manifest=json.loads((out/'version_manifest.json').read_bytes())
    counts={'original_records':500,'retained':465,'repaired':21,'quarantined':14,'added':40,'records':526,'slice_records':50,'source_family_overlap':0}
    if any(manifest.get(k)!=v for k,v in counts.items()):raise ValueError('Unexpected revised version counts')
    required={'seed_records.jsonl','train_all.jsonl','slice_records.jsonl','train_slice.jsonl','quarantine_records.jsonl','change_log.jsonl','training_lock.json'}
    if set(manifest.get('files',{}))!=required:raise ValueError('Unexpected revised file inventory')
    for name,expected in manifest['files'].items():
        if Path(name).name!=name or digest((out/name).read_bytes())!=expected:raise ValueError('Versioned data changed: '+name)
    audit=json.loads((project/'reviews/targeted_audit_500/summary.json').read_bytes())
    if manifest['original_input_hashes']!=audit['preserved_inputs']:raise ValueError('Original input inventory changed')
    for name,expected in manifest['original_input_hashes'].items():
        if digest((project/name).read_bytes())!=expected:raise ValueError('Historical input changed: '+name)
    seed=rows(project/'datasets/seed_records.jsonl')
    repairs=rows(review_dir/'repaired_records.jsonl');new=rows(review_dir/'new_records.jsonl')
    validate_integration_review(project,repairs,new)
    repaired={r['parent_example_id']:r for r in repairs}
    proposals=rows(project/'reviews/targeted_audit_500/correction_proposals.jsonl')
    revisions={r['example_id'] for r in proposals if r['disposition']=='revise'}
    quarantined={r['example_id']:r for r in proposals if r['disposition']=='review'}
    if set(repaired)!=revisions or len(quarantined)!=14:raise ValueError('Audit dispositions changed')
    expected=[]
    for r in seed:
        parent=r['example_id']
        if parent in quarantined:continue
        current=deepcopy(repaired.get(parent,r))
        current.update(data_version='revision_v2',parent_example_id=parent,parent_record_sha256=canonical_sha256(r))
        if parent in repaired:current.update(approved_for_seed_training=True,review_status='agent_reviewed_revision_v2')
        expected.append(current)
    for r in new:
        current=deepcopy(r);current.update(data_version='revision_v2',approved_for_seed_training=True,review_status='agent_reviewed_revision_v2');expected.append(current)
    records=rows(out/'seed_records.jsonl')
    if records!=expected or len({r['example_id'] for r in records})!=526:raise ValueError('Revised records differ from approved lineage')
    projection=lambda rs:[{'example_id':r['example_id'],'messages':r['messages']} for r in rs]
    if rows(out/'train_all.jsonl')!=projection(records):raise ValueError('Full training projection contains altered content or provenance')
    wanted={seed[i-1]['example_id'] for i in SELECTED_REPAIR_INDICES};new_ids={r['example_id'] for r in new}
    selected=[r for r in records if r.get('parent_example_id') in wanted or r['example_id'] in new_ids]
    if len(selected)!=50 or rows(out/'slice_records.jsonl')!=selected or rows(out/'train_slice.jsonl')!=projection(selected):
        raise ValueError('Slice is not exactly the selected 40 additions and 10 repairs')
    expected_q=[{'original_record':r,'audit':quarantined[r['example_id']]} for r in seed if r['example_id'] in quarantined]
    if rows(out/'quarantine_records.jsonl')!=expected_q:raise ValueError('Quarantine differs from original unresolved records')
    catalog={r['document_id']:r for r in rows(project/'corpus_mia/catalog.jsonl')}
    docs={r['document_id']:r for r in rows(project/'corpus_mia/documents.jsonl')}
    persona=json.loads((project/'config/persona.json').read_bytes())
    for r in records:validate_record(r,docs,catalog,persona)
    guard=load_fresh_holdout_guard(project,catalog)
    if 'evaluation/fresh_holdout_v2/source_exclusions.jsonl' not in guard.manifests:raise ValueError('New v2 holdout must be frozen before training')
    check_training_records(records,catalog,guard)
    if manifest['reserved_documents']!=len(guard.document_ids):raise ValueError('Holdout inventory expanded; freeze a new experiment')
    lock=json.loads((out/'training_lock.json').read_bytes())
    validate_training_payload((out/'train_slice.jsonl').read_bytes(),lock)
    if (lock['data_review']['root_review_sha256']!=digest((review_dir/'integration_review.json').read_bytes())
            or lock['data_review']['new_record_reviews_sha256']!=digest((review_dir/'source_review.md').read_bytes())
            or lock['data_review']['repair_log_sha256']!=digest((review_dir/'repair_log.jsonl').read_bytes())):
        raise ValueError('Portable training lock does not bind the current semantic review')
    validate_contamination(projection(records),lock['contamination_guard'])
    return manifest

def validate_record(r,docs,catalog,persona):
    doc=docs[r['source_document_id']];cat=catalog[r['source_document_id']]
    bs={b['block_id']:b for b in doc['blocks']}
    evidence='\n\n'.join(bs[i]['text'] for i in r['source_block_ids'])
    if r['source_sha256']!=doc['source_sha256'] or r['source_url']!=doc['url'] or r['source_family_id']!=cat['source_family_id']:
        raise ValueError('Source provenance mismatch: '+r['example_id'])
    if r['source_evidence']!=evidence or digest(evidence.encode())!=r['evidence_sha256']:
        raise ValueError('Evidence mismatch: '+r['example_id'])
    if any(bs[i]['role']!='body_candidate' for i in r['source_block_ids']):raise ValueError('Excluded source block')
    date=r['source_date_evidence']['interval']
    if date['latest']>persona['historical_as_of'] or date['earliest']>date['latest']:raise ValueError('Unresolved source date')
    authoritative=date_evidence(doc,persona['historical_as_of'])
    if r['source_date_evidence']!=authoritative:
        a=r.get('date_adjudication',{})
        if (r.get('parent_example_id')!='seed_b3e0ee14607cefe6'
            or a.get('original_parsed_source_date_evidence')!=authoritative
            or a.get('adjudicated_source_date_evidence')!=r['source_date_evidence']
            or a.get('source_metadata_text')!=bs[3]['text']
            or date['earliest']!='1938-05-01' or date['latest']!='1938-06-30'):
            raise ValueError('Date differs without reviewed exact-source adjudication')
    if r['historical_as_of']!=persona['historical_as_of'] or r['messages'][0] != {'role':'system','content':persona['system_prompt']}:
        raise ValueError('Persona mismatch')
    if len(r['messages'])<3 or len(r['messages'])%2!=1:raise ValueError('Incomplete conversation')
    for i,m in enumerate(r['messages']):
        role='system' if i==0 else 'user' if i%2 else 'assistant'
        if set(m)!={'role','content'} or m['role']!=role or not m['content'].strip():raise ValueError('Invalid role/text')
    if not r.get('review_note'):raise ValueError('Missing semantic source review')


def build(project=PROJECT):
    project=Path(project);out=project/'datasets/revision_v2';out.mkdir(parents=True,exist_ok=True)
    if (out/'version_manifest.json').exists():raise FileExistsError('v2 dataset already built; preserve version')
    audit=json.loads((project/'reviews/targeted_audit_500/summary.json').read_text())
    for p,h in audit['preserved_inputs'].items():
        if digest((project/p).read_bytes())!=h:raise ValueError('Original input changed: '+p)
    seed=rows(project/'datasets/seed_records.jsonl');by={r['example_id']:r for r in seed}
    proposals=rows(project/'reviews/targeted_audit_500/correction_proposals.jsonl')
    revise={r['example_id']:r for r in proposals if r['disposition']=='revise'}
    quarantine={r['example_id']:r for r in proposals if r['disposition']=='review'}
    repairs=rows(project/'reviews/data_revision_v2/repaired_records.jsonl');new=rows(project/'reviews/data_revision_v2/new_records.jsonl')
    validate_integration_review(project,repairs,new)
    repair_by={r['parent_example_id']:r for r in repairs}
    if len(repairs)!=21 or set(repair_by)!=set(revise) or len(new)!=40:raise ValueError('Expected21 repairs and40 additions')
    if Counter(r['behavior_category'] for r in new)!=dict.fromkeys(['chronology','false_premise','invented_memory_citation','answer_detail'],10):raise ValueError('Unbalanced new behaviors')
    catalog={r['document_id']:r for r in rows(project/'corpus_mia/catalog.jsonl')}
    docs={r['document_id']:r for r in rows(project/'corpus_mia/documents.jsonl')}
    persona=json.loads((project/'config/persona.json').read_text())
    forbidden_docs={by[k]['source_document_id'] for k in set(revise)|set(quarantine)}
    original_families={r['source_family_id'] for r in seed}
    for r in repairs+new:validate_record(r,docs,catalog,persona)
    for r in new:
        if r['source_family_id'] not in original_families or r['source_document_id'] in forbidden_docs:raise ValueError('New examples use unapproved/conflicted source pool')
        if r['messages'] in [s['messages'] for s in seed]:raise ValueError('Duplicate original conversation')
    records=[];changes=[]
    for i,r in enumerate(seed,1):
        parent=r['example_id'];old_hash=canonical_sha256(r)
        if parent in quarantine:
            changes.append({'parent_example_id':parent,'index_1based':i,'action':'quarantine','original_record_sha256':old_hash,'findings':quarantine[parent]['findings']});continue
        current=deepcopy(repair_by.get(parent,r))
        current['data_version']='revision_v2';current['parent_example_id']=parent;current['parent_record_sha256']=old_hash
        if parent in repair_by:
            current['approved_for_seed_training']=True;current['review_status']='agent_reviewed_revision_v2'
        records.append(current)
        changes.append({'parent_example_id':parent,'example_id':current['example_id'],'index_1based':i,'action':'repair' if parent in repair_by else 'retain','original_record_sha256':old_hash,'revised_record_sha256':canonical_sha256(current)})
    for r in new:
        current=deepcopy(r);current['data_version']='revision_v2';current['approved_for_seed_training']=True;current['review_status']='agent_reviewed_revision_v2';records.append(current)
        changes.append({'example_id':r['example_id'],'action':'add','record_sha256':canonical_sha256(current),'behavior_category':r['behavior_category'],'control_type':r['control_type']})
    if len(records)!=526 or len({r['example_id'] for r in records})!=526:raise ValueError('New version must have526 unique rows')
    guard=load_fresh_holdout_guard(project,catalog);check_training_records(records,catalog,guard)
    wanted={seed[i-1]['example_id'] for i in SELECTED_REPAIR_INDICES}
    selected=[r for r in records if r.get('parent_example_id') in wanted or r['example_id'] in {n['example_id'] for n in new}]
    if len(selected)!=50:raise ValueError('Expected40 new +10 repaired slice rows')
    payload=jsonl([{'example_id':r['example_id'],'messages':r['messages']} for r in selected])
    # Exclude all known benchmark text. Exact spans already in the original
    # pre-benchmark seed are not new leakage; record this explicit exception.
    original_texts={m['content'] for r in seed for m in r['messages'] if m['role']!='system'}
    pre_text_hashes={text_sha256(t) for t in original_texts};pre_spans=set().union(*(span_hashes(t) for t in original_texts))
    excluded=set(generated_answers(project,benchmark_inventory(project)))
    for folder in [project/'evaluation']+list((project/'evaluation').glob('fresh_holdout_*')):
        f=folder/'cases.jsonl'
        if f.exists():
            for c in rows(f):
                if c.get('reference_evidence'):excluded.add(c['reference_evidence'])
                excluded.update(m['content'] for m in c['messages'] if m['role']!='system')
    forbidden_texts={text_sha256(t) for t in excluded}-pre_text_hashes
    forbidden_spans=set().union(*(span_hashes(t) for t in excluded))-pre_spans
    contamination={'normalization':'NFKC/casefold/Unicodewords','span_words':16,'text_sha256':sorted(forbidden_texts),'span_sha256':sorted(forbidden_spans)}
    lock={'schema_version':1,'dataset':{'path':'datasets/revision_v2/train_slice.jsonl','sha256':digest(payload),'records':50,'example_sha256':{r['example_id']:canonical_sha256({'example_id':r['example_id'],'messages':r['messages']}) for r in selected}},'persona':{'system_prompt_sha256':digest(persona['system_prompt'].encode()),'historical_as_of':persona['historical_as_of']},'contamination_guard':contamination,'data_review':{'original_sha256':digest((project/'datasets/seed_records.jsonl').read_bytes()),'new_record_reviews_sha256':digest((project/'reviews/data_revision_v2/source_review.md').read_bytes()),'repair_log_sha256':digest((project/'reviews/data_revision_v2/repair_log.jsonl').read_bytes()),'root_review_sha256':digest((project/'reviews/data_revision_v2/integration_review.json').read_bytes())}}
    validate_training_payload(payload,lock)
    validate_contamination([{'example_id':r['example_id'],'messages':r['messages']} for r in records],contamination)
    files={'seed_records.jsonl':jsonl(records),'train_all.jsonl':jsonl([{'example_id':r['example_id'],'messages':r['messages']} for r in records]),'slice_records.jsonl':jsonl(selected),'train_slice.jsonl':payload,'quarantine_records.jsonl':jsonl([{'original_record':r,'audit':quarantine[r['example_id']]} for r in seed if r['example_id'] in quarantine]),'change_log.jsonl':jsonl(changes),'training_lock.json':(json.dumps(lock,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()}
    for name,data in files.items():(out/name).write_bytes(data)
    manifest={'schema_version':1,'version':'revision_v2','created_utc':datetime.now(timezone.utc).isoformat(),'original_records':500,'retained':465,'repaired':21,'quarantined':14,'added':40,'records':526,'slice_records':50,'selection':{'new_examples':40,'repaired_original_indices':SELECTED_REPAIR_INDICES,'policy':'One longest example then seed42 shuffle; exact optimizer order frozen by tokenizer preflight.'},'source_review':'Coding-agent source/context review plus root integration; not independent historian validation.','reserved_documents':len(guard.document_ids),'source_family_overlap':0,'copy_guard':{'excluded_texts':len(excluded),'preexisting_seed_text_exemptions':len(pre_text_hashes),'preexisting_seed_span_exemptions':len(pre_spans),'limitation':'Detects exact normalized text/16-word spans, not semantic paraphrases. Preexisting original seed text cannot be new benchmark leakage.'},'original_input_hashes':audit['preserved_inputs'],'files':{k:digest(v) for k,v in files.items()}}
    write_json(out/'version_manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['retained','repaired','quarantined','added','records','slice_records','source_family_overlap']},indent=2))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=PROJECT);build(p.parse_args().project)
