"""Verify the grounding delivery, export hidden labels, lock scores, then analyze."""
import argparse, csv, hashlib, io, json, random, secrets, stat, zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from prepare_grounding_cases import verify, ROOT, OUT, rows, enc, jl
from kaggle_grounding import CONDITIONS, digest, fingerprint, validate_payloads, read_evaluation_journal

RUN=ROOT/'evaluation/runs/kaggle_grounding_v1'

def checkpoint_identity_fingerprint(identity):
    # The preserved checkpoint helper hashes canonical JSON WITH a final newline;
    # run identities use canonical JSON without it. These are distinct schemas.
    return digest((json.dumps(identity,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode())

def map_judgment(score, key, case):
    return {**score,'condition':key['condition'],'case_id':key['case_id'],
            'category':case['category'],'control_type':case['control_type'],
            'source_family_id':case['source_family_id'],
            'critical_zero_score':0 if score['critical_types'] else score['dimension_score']}

def receive():
    p=verify();lock=json.loads((OUT/'bundle_lock.json').read_bytes())
    archives=list((RUN/'download').rglob('comparison_delivery.zip'))
    if len(archives)!=1:raise ValueError('Expected exactly one delivery ZIP')
    target=RUN/'results'
    if target.exists():raise FileExistsError('Preserve extracted results')
    with zipfile.ZipFile(archives[0]) as z:
        names=z.namelist()
        if len(set(names))!=len(names) or z.testzip():raise ValueError('Duplicate path/CRC failure')
        for info in z.infolist():
            n=Path(info.filename)
            if n.is_absolute() or '..' in n.parts or '\\' in str(n) or info.is_dir() or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Unsafe archive path/type')
        z.extractall(target)
    for n,h in lock['payloads'].items():
        if digest((target/'inputs'/n).read_bytes())!=h:raise ValueError('Bundled input changed: '+n)
    runtime=json.loads((target/'runtime_protocol.json').read_bytes());m=json.loads((target/'run_manifest.json').read_bytes());ident=m['identity']
    if (target/'runtime_protocol.json').read_bytes()!=(target/'inputs/runtime_protocol.json').read_bytes():raise ValueError('Runtime protocol differs')
    if (target/'model_inputs.jsonl').read_bytes()!=(OUT/'model_inputs.jsonl').read_bytes():raise ValueError('Model input differs')
    if m['status']!='complete' or m['stage']!='complete' or m['identity_sha256']!=fingerprint(ident):raise ValueError('Incomplete/invalid run identity')
    required={'fresh_base_has_no_adapter','saved_lr1e4_step50_exact_adapter_tensors','matched_treatment_tokens_equal','zero_optimizer_updates',*[c+'_complete' for c in CONDITIONS]}
    if not required<=set(m['checks']) or not all(v is True for v in m['checks'].values()):raise ValueError('Runtime checks incomplete')
    if ident['optimizer_updates']!=0 or ident['seed']!=42 or ident['enable_thinking'] is not False or ident['packages']!=p['packages'] or ident['model']!=p['model'] or ident['source_protocol_sha256']!=digest((OUT/'protocol.json').read_bytes()):raise ValueError('Runtime identity differs')
    if ident['runtime_protocol_sha256']!=digest((target/'runtime_protocol.json').read_bytes()) or ident['model_inputs_sha256']!=p['files']['model_inputs.jsonl']:raise ValueError('Input provenance differs')
    if ident['control_identity_sha256']!=p['control_identity_sha256'] or ident['control_run_manifest_sha256']!=p['control_run_manifest_sha256']:raise ValueError('Control provenance differs')
    for n,h in ident['helpers'].items():
        if digest((target/'inputs'/n).read_bytes())!=h or lock['payloads'][n]!=h:raise ValueError('Runtime helper differs')
    cp=ident['checkpoints']['saved_lr1e4_step50']
    if cp['adapter_sha256']!=p['adapter_sha256'] or cp['manifest_sha256']!=p['checkpoint_manifest_sha256'] or cp['identity_sha256']!=checkpoint_identity_fingerprint(p['control_identity']):raise ValueError('Wrong saved adapter')
    if ident['chat_template_sha256']!=p['chat_template_sha256'] or any(ident['generation'][k]!=v for k,v in p['generation'].items()) or ident['generation_sha256']!=fingerprint(ident['generation']):raise ValueError('Decoding differs')
    control=json.loads((target/'inputs/control_run_manifest.json').read_bytes())
    if ident['gpu']!=control['identity']['gpu'] or ident['gpu_ids_used']!=[0] or ident['dtype']!='torch.float16' or ident['attention_implementation']!='sdpa' or ident['kbit_preparation_training'] is not False:raise ValueError('Runtime policy differs')
    expected_quant={'load_in_4bit':True,'bnb_4bit_quant_type':'nf4','bnb_4bit_use_double_quant':True,'compute_dtype':'torch.float16'}
    if ident['quantization']!=expected_quant:raise ValueError('Quantization differs')
    inputs=validate_payloads((target/'model_inputs.jsonl').read_bytes(),runtime,control);data={};token_map={}
    for c in CONDITIONS:
        selected=[r for r in inputs if r['treatment']==('with_passages' if c.endswith('_with_passages') else 'without_passages')]
        raw=rows(target/(c+'_responses.jsonl'));token_map[c]={r['case_id']:r['input_token_ids'] for r in raw}
        data[c]=read_evaluation_journal(target/(c+'_responses.jsonl'),m['identity_sha256'],c,selected,ident['generation'],token_map[c])
        if len(data[c])!=24:raise ValueError('Incomplete answers')
        if any(len(v)>p['max_input_tokens'] or not v for v in token_map[c].values()):raise ValueError('Input truncation/budget')
    if token_map[CONDITIONS[0]]!=token_map[CONDITIONS[2]] or token_map[CONDITIONS[1]]!=token_map[CONDITIONS[3]] or fingerprint(token_map)!=ident['prompt_token_ids_sha256']:raise ValueError('Matched token identity failed')
    if any(token_map[CONDITIONS[0]][c]==token_map[CONDITIONS[1]][c] for c in token_map[CONDITIONS[0]]):raise ValueError('Treatment missing')
    for name,h in p['preserved_files'].items():
        if digest((ROOT/name).read_bytes())!=h:raise ValueError('Preserved artifact changed: '+name)
    report=dict(status='verified',answers=96,training_steps=0,source_protocol_sha256=ident['source_protocol_sha256'],identity_sha256=m['identity_sha256'],archive_sha256=digest(archives[0].read_bytes()),archive_entries=len(names),all_runtime_checks=m['checks'],matched_treatment_tokens_equal=True,max_input_tokens=max(len(v) for t in token_map.values() for v in t.values()),preserved_training_artifacts_verified=True,limitations='Checks bind downloaded bytes and runtime claims; not independent remote execution attestation.')
    (RUN/'verification.json').write_bytes(enc(report))
    review=RUN/'review';review.mkdir(exist_ok=False)
    cases={c['case_id']:c for c in rows(OUT/'cases.jsonl')}
    shuffled=[(c,r) for c,answers in data.items() for r in answers];random.Random(secrets.randbits(128)).shuffle(shuffled)
    neutral=[];key=[]
    for i,(c,r) in enumerate(shuffled,1):
        rid=f'R{i:03d}';response_hash=digest(r['response'].encode())
        neutral.append(dict(review_id=rid,case_id=r['case_id'],response=r['response'],response_sha256=response_hash))
        key.append(dict(review_id=rid,case_id=r['case_id'],condition=c,response_sha256=response_hash))
    (review/'neutral_answers.jsonl').write_bytes(jl(neutral));(review/'condition_key.jsonl').write_bytes(jl(key))
    (review/'cases.jsonl').write_bytes((OUT/'cases.jsonl').read_bytes());(review/'rubric.json').write_bytes((OUT/'rubric.json').read_bytes())
    (review/'review_manifest.json').write_bytes(enc(dict(created_utc=datetime.now(timezone.utc).isoformat(),answers=96,neutral_sha256=digest(jl(neutral)),key_sha256=digest(jl(key)),scores_assigned=False,reviewer='separate coding agent; condition-hidden, no historian validation')))
    print(json.dumps(report,indent=2))

def validate_scores():
    review=RUN/'review';scores=rows(review/'scores.jsonl');neutral={r['review_id']:r for r in rows(review/'neutral_answers.jsonl')};cases={r['case_id']:r for r in rows(review/'cases.jsonl')};rubric=json.loads((review/'rubric.json').read_bytes())
    manifest=json.loads((review/'review_manifest.json').read_bytes())
    if len(neutral)!=96 or digest((review/'neutral_answers.jsonl').read_bytes())!=manifest['neutral_sha256'] or digest((review/'condition_key.jsonl').read_bytes())!=manifest['key_sha256']:raise ValueError('Review export changed')
    if (review/'cases.jsonl').read_bytes()!=(OUT/'cases.jsonl').read_bytes() or (review/'rubric.json').read_bytes()!=(OUT/'rubric.json').read_bytes():raise ValueError('Review sources/rubric changed')
    if any(digest(n['response'].encode())!=n['response_sha256'] for n in neutral.values()):raise ValueError('Neutral answer hash differs')
    if len(scores)!=96 or len({s['review_id'] for s in scores})!=96 or {s['review_id'] for s in scores}!=set(neutral):raise ValueError('All96 explicit unique judgments required')
    required=set(rubric['required_per_answer'])|{'review_id'}
    for s in scores:
        n=neutral[s['review_id']];case=cases[n['case_id']]
        if not required<=set(s) or s['response_sha256']!=n['response_sha256']:raise ValueError('Score coverage/hash')
        if type(s['dimension_score']) is not int or not 0<=s['dimension_score']<=4 or not s['notes']:raise ValueError('Invalid dimension/notes')
        if not isinstance(s['critical_types'],list) or len(set(s['critical_types']))!=len(s['critical_types']) or not set(s['critical_types'])<=set(rubric['critical_types']):raise ValueError('Invalid critical flag')
        if 'critical_zero_score' in s and (type(s['critical_zero_score']) is not int or s['critical_zero_score']!=(0 if s['critical_types'] else s['dimension_score'])):raise ValueError('Critical-zero rule differs')
        if not isinstance(s['checklist_met'],list) or len(set(s['checklist_met']))!=len(s['checklist_met']) or any(type(i)is not int or not 1<=i<=len(case['reference_checks']) for i in s['checklist_met']):raise ValueError('Invalid checklist')
        for field in ['fabrication_present','unsupported_replacement_present']:
            if type(s[field]) is not bool:raise ValueError('Invalid diagnostic bool')
        for field,applies in [('central_premise_corrected',case['category']=='false_premise' and case['control_type']=='negative'),('supported_answer',case['control_type']=='positive'),('blanket_refusal',case['control_type']=='positive')]:
            if (applies and type(s[field]) is not bool) or (not applies and s[field] is not None):raise ValueError('Diagnostic applicability: '+field)
        if s['fabrication_present']!=bool({'invented_private_scene','invented_direct_quote_or_citation'}&set(s['critical_types'])) or s['unsupported_replacement_present']!=('unsupported_replacement_explanation' in s['critical_types']):raise ValueError('Critical diagnostic mismatch')
    return scores

def lock_scores():
    scores=validate_scores();review=RUN/'review';path=review/'score_lock.json'
    if path.exists():raise FileExistsError('Scores already locked')
    names=['scores.jsonl','neutral_answers.jsonl','cases.jsonl','rubric.json','condition_key.jsonl']
    lock=dict(locked_utc=datetime.now(timezone.utc).isoformat(),answers=len(scores),scores_locked_before_unblinding=True,files={n:digest((review/n).read_bytes()) for n in names})
    path.write_bytes(enc(lock));print(json.dumps(lock,indent=2))

def analyze():
    scores=validate_scores();review=RUN/'review';lock=json.loads((review/'score_lock.json').read_bytes())
    for n,h in lock['files'].items():
        if digest((review/n).read_bytes())!=h:raise ValueError('Locked review changed')
    key={r['review_id']:r for r in rows(review/'condition_key.jsonl')};cases={r['case_id']:r for r in rows(review/'cases.jsonl')};neutral={r['review_id']:r for r in rows(review/'neutral_answers.jsonl')}
    mapped=[]
    for s in scores:
        k=key[s['review_id']];case=cases[k['case_id']]
        if k['response_sha256']!=s['response_sha256']:raise ValueError('Key/score mismatch')
        mapped.append(map_judgment(s,k,case))
    def summarize(group):
        mean=lambda a:round(sum(a)/len(a),4) if a else None
        return dict(n=len(group),critical_cases=sum(bool(s['critical_types']) for s in group),critical_types=dict(Counter(t for s in group for t in s['critical_types'])),category_means={cat:mean([s['critical_zero_score'] for s in group if s['category']==cat]) for cat in sorted({s['category'] for s in group})},dimension_means={cat:mean([s['dimension_score'] for s in group if s['category']==cat]) for cat in sorted({s['category'] for s in group})},premises_corrected=sum(s['central_premise_corrected'] is True for s in group),negative_premise_n=sum(s['central_premise_corrected'] is not None for s in group),fabrication_cases=sum(s['fabrication_present'] for s in group),unsupported_replacements=sum(s['unsupported_replacement_present'] for s in group),positive_n=sum(s['control_type']=='positive' for s in group),positive_supported=sum(s['supported_answer'] is True for s in group),positive_score_ge3=sum(s['control_type']=='positive' and s['critical_zero_score']>=3 for s in group),positive_refusals=sum(s['blanket_refusal'] is True for s in group),detail_score_ge3=sum(s['category']=='answer_detail' and s['critical_zero_score']>=3 for s in group),checklist_checks_met=sum(len(s['checklist_met']) for s in group))
    summary=dict(conditions={c:summarize([s for s in mapped if s['condition']==c]) for c in CONDITIONS},source_level={f:{c:summarize([s for s in mapped if s['condition']==c and s['source_family_id']==f]) for c in CONDITIONS} for f in sorted({s['source_family_id'] for s in mapped})},paired={},limitations=['24 cases clustered within six works; descriptive ordinal scores, no pooled overall-quality scale.','One separate coding-agent reviewer; style may reveal treatment/model, no independent historian validation.','Curated sufficient passages test evidence availability, not retrieval quality or pretraining absence.'])
    by={(s['condition'],s['case_id']):s for s in mapped};paired=[]
    for name,a,b in [('base_passage_effect',CONDITIONS[0],CONDITIONS[1]),('adapter_passage_effect',CONDITIONS[2],CONDITIONS[3]),('adapter_vs_base_without',CONDITIONS[0],CONDITIONS[2]),('adapter_vs_base_with',CONDITIONS[1],CONDITIONS[3])]:
        counts=Counter();details=[]
        for case_id,case in cases.items():
            x,y=by[a,case_id],by[b,case_id];delta=y['critical_zero_score']-x['critical_zero_score'];verdict='improved' if delta>0 else 'degraded' if delta<0 else 'tied';counts[verdict]+=1
            record=dict(comparison=name,case_id=case_id,category=case['category'],source_family_id=case['source_family_id'],before=x['critical_zero_score'],after=y['critical_zero_score'],delta=delta,verdict=verdict,critical_before=bool(x['critical_types']),critical_after=bool(y['critical_types']))
            paired.append(record);details.append(record)
        summary['paired'][name]=dict(before=a,after=b,**{k:counts[k] for k in ['improved','degraded','tied']},by_category={cat:dict(Counter(r['verdict'] for r in details if r['category']==cat)) for cat in sorted({r['category'] for r in details})})
    summary['length_diagnostics']={}
    for c in CONDITIONS:
        raw=rows(RUN/'results'/(c+'_responses.jsonl'));summary['length_diagnostics'][c]=dict(mean_tokens=round(sum(r['token_count'] for r in raw)/24,2),truncations=sum(r['finish_reason']=='length' for r in raw))
    (review/'mapped_scores.jsonl').write_bytes(jl(mapped));(RUN/'quality_summary.json').write_bytes(enc(summary))
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired);(RUN/'paired_cases.csv').write_text(stream.getvalue())
    audit=['# Complete case and failure audit','', 'All judgments were locked before condition mapping. Exact responses are retained; primary score zeros every critical answer.','']
    for case_id,case in cases.items():
        audit.extend(['## '+case_id,'',case['question'],'','Source: '+case['source_url'],''])
        for c in CONDITIONS:
            s=by[c,case_id];n=neutral[s['review_id']]
            audit.extend(['### '+c,'',f"Score {s['critical_zero_score']}; dimension {s['dimension_score']}; critical {s['critical_types']}; checks {s['checklist_met']}.",'',s['notes'],'',n['response'],''])
    (RUN/'CASE_AUDIT.md').write_text('\n'.join(audit))
    print(json.dumps({'conditions':summary['conditions'],'paired':summary['paired'],'length_diagnostics':summary['length_diagnostics']},indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['receive','validate-scores','lock','analyze']);action=a.parse_args().action
    {'receive':receive,'validate-scores':validate_scores,'lock':lock_scores,'analyze':analyze}[action]()
