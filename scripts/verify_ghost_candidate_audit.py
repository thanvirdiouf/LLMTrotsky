"""Verify frozen27B audit tokens and replay the archived9B decision parser."""
import argparse
import ast
import base64
from collections import Counter
import json
import lzma
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from build_ghost_candidate_probe import ROOT,sha
from build_ghost_candidate_audit import control,PROBE,freeze_inputs
from candidate_audit import CANDIDATE,GIB,validate_protocol,input_ids,device_map_for,resource_gate
from summarize_ghost_calibration import classify,judgment_hashes
from grounded_chat import fingerprint


def payloads(version,root=ROOT):
    if not re.fullmatch(r'ghost_candidate_audit_v[1-9][0-9]*',version):raise ValueError('Invalid candidate audit version')
    stage=root/'training/kaggle_cli'/version;lock=json.loads((root/'evaluation'/version/'bundle_lock.json').read_text())
    raw=(stage/'kaggle_ghost_candidate_audit.ipynb').read_bytes();metadata=(stage/'kernel-metadata.json').read_bytes()
    if sha(raw)!=lock['notebook_sha256'] or sha(metadata)!=lock['metadata_sha256']:raise ValueError('Frozen notebook changed')
    m=json.loads(metadata)
    if m['is_private'] is not True or m['id_no']!=140792344:raise ValueError('Wrong private target')
    code=''.join(next(c for c in json.loads(raw)['cells'] if c['id']=='run')['source']);tree=ast.parse(code)
    packed=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='packed' for t in n.targets)))
    values={name:lzma.decompress(base64.b85decode(raw)) for name,raw in packed.items()}
    if {n:sha(v) for n,v in values.items()}!=lock['payload_sha256']:raise ValueError('Frozen payload changed')
    p=json.loads(values['protocol.json']);validate_protocol(p)
    if any(sha(values[n])!=digest for n,digest in p['files'].items()):raise ValueError('Protocol payload changed')
    return values


def verify_generated(job,out,cap,generation,tokenizer):
    ids=out['token_ids']
    if (not isinstance(ids,list) or not 1<=len(ids)<=cap or any(type(i) is not int or i<0 for i in ids)
            or out['input_token_ids']!=job['input_token_ids'] or not 1<=len(out['input_token_ids'])<=5120
            or out['token_count']!=len(ids) or out.get('finite_logits') is not True or out.get('finite_logit_steps')!=len(ids)
            or out.get('efficient_attention_calls')!=16*len(ids) or out.get('expanded_kv_calls')!=16*len(ids)):
        raise ValueError('Output bounds, input identity, finite logits or backend coverage differ')
    finish='stop' if ids[-1]==generation['eos_token_id'] else 'length'
    if out['finish_reason']!=finish or (finish=='length' and len(ids)!=cap):raise ValueError('Output stop differs')
    if tokenizer is not None:
        if input_ids(tokenizer,job['messages'])!=out['input_token_ids'] or tokenizer.decode(ids,skip_special_tokens=True)!=out['response']:
            raise ValueError('Tokenizer reconstruction differs')


def replay(values,jobs,rows):
    """Use exact archived source, context, parser and coverage behavior in isolation."""
    code='''import contextlib,io,json
from pathlib import Path
from evaluate_ghost_audit import evaluate
p=Path('.')
calls=json.loads((p/'candidate_rows.json').read_text());jobs=json.loads((p/'candidate_jobs.json').read_text());index=0
def generate(messages):
 global index
 if messages!=jobs[index]['messages']:raise ValueError('Frozen prompt replay differs')
 out=calls[index]['output'];index+=1
 return {**out,'model_call_made':True}
identity={'model':json.loads((p/'protocol.json').read_text())['model'],'enable_thinking':False,'purpose':'parser_replay_only'}
with contextlib.redirect_stdout(io.StringIO()):
 rows,_=evaluate(*[json.loads((p/n).read_text()) for n in ('profile.json','collection.json','claims.json','model_cases.json')],generate,p/'replay_results',identity)
if index!=10:raise ValueError('Replay coverage differs')
(p/'replayed.json').write_text(json.dumps(rows))
'''
    with tempfile.TemporaryDirectory(prefix='candidate_audit_replay_') as temp:
        d=Path(temp)
        for name,raw in values.items():(d/name).write_bytes(raw)
        (d/'candidate_rows.json').write_text(json.dumps(rows));(d/'candidate_jobs.json').write_text(json.dumps(jobs))
        subprocess.run([sys.executable,'-c',code],cwd=d,check=True,capture_output=True,text=True,timeout=60)
        return json.loads((d/'replayed.json').read_text())


def score_rows(spec,rows):
    if len(rows)!=len(spec['cases']):raise ValueError('Scoring coverage differs')
    scored=[]
    for case,row in zip(spec['cases'],rows):
        if row['case_id']!=case['id'] or row['segments']!=case['segments']:raise ValueError('Scoring candidate differs')
        outcome,approved,reasons=classify(case,row)
        scored.append({'case_id':case['id'],'expected_approved':case['expected_approved'],'actual_approved':approved,
            'outcome':outcome,'reasons':reasons,'response_fingerprint':fingerprint(row['response']),**judgment_hashes(row)})
    byid={r['case_id']:r for r in scored};pairs=[]
    for pair in spec['reviewer_only_pairs']:
        ids=pair['case_ids']
        if len(ids)!=2 or len(set(ids))!=2 or {byid[i]['expected_approved'] for i in ids}!={True,False}:raise ValueError('Pair design differs')
        pairs.append({'pair_id':pair['pair_id'],'case_ids':ids,'both_decisions_match':all(byid[i]['outcome'] in ('correct_approval','correct_rejection') for i in ids)})
    if len(pairs)!=5 or sorted(i for pair in pairs for i in pair['case_ids'])!=sorted(byid):raise ValueError('Pair coverage differs')
    return {'counts':dict(Counter(r['outcome'] for r in scored)),'pairs_both_correct':sum(p['both_decisions_match'] for p in pairs),'pairs_total':5,'rows':scored,'pairs':pairs}


def verify(version='ghost_candidate_audit_v1',root=ROOT,tokenizer=None):
    frozen=payloads(version,root);p=json.loads(frozen['protocol.json']);run=root/'evaluation/runs'/version;download=run/'download'
    for name,raw in frozen.items():
        if (download/'ghost_candidate_audit_bundle'/name).read_bytes()!=raw:raise ValueError('Downloaded payload differs: '+name)
    for name,digest in p['preserved_files'].items():
        if sha((root/name).read_bytes())!=digest:raise ValueError('Protected archive changed')
    if sha((root/'scripts/summarize_ghost_calibration.py').read_bytes())!=p['local_scorer_sha256']:raise ValueError('Scorer changed')
    values,old,spec,expected,calls,binding=control(root)
    if binding!=p['control'] or sha((root/'evaluation'/version/'evaluation_spec.json').read_bytes())!=p['evaluation_spec_sha256'] or expected!=(root/'evaluation'/version/'evaluation_spec.json').read_bytes():raise ValueError('Control/expectations changed')
    jobs=json.loads(frozen['audit_inputs.json'])
    if tokenizer is not None and jobs!=freeze_inputs(spec,calls,tokenizer):raise ValueError('Control prompt messages differ')
    if [j['messages'] for j in jobs]!=[c['messages'] for c in calls] or [j['case_id'] for j in jobs]!=[c['id'] for c in spec['cases']]:raise ValueError('Fixed candidate inputs differ')
    result_path=download/'ghost_candidate_audit_results/result.json';result=json.loads(result_path.read_text())
    if any(result.get(k)!=p[k] for k in ('model','purpose','training_steps','adapters_loaded','active_model_changed','enable_thinking')) or result.get('quality_validated') is not False or result['protocol_sha256']!=sha(frozen['protocol.json']):raise ValueError('Result identity differs')
    if type(result['evaluation_calls']) is not int or not 0<=result['evaluation_calls']<=10 or type(result['memory_probe_calls']) is not int or not 0<=result['memory_probe_calls']<=1:raise ValueError('Run call limits differ')
    report={'version':version,'status':'verified_artifacts','model':CANDIDATE,'result_sha256':sha(result_path.read_bytes()),'run_status':result['status'],
        'evaluation_calls':result['evaluation_calls'],'memory_probe_calls':result['memory_probe_calls'],'quality_validated':False,'active_model_changed':False}
    if result['status']=='blocked':
        if not result.get('error_type') or not result.get('error'):raise ValueError('Blocked reason absent')
        return {**report,'error_type':result['error_type'],'error':result['error'],'phase':result['phase']}
    if result['status']!='complete' or result['evaluation_calls']!=10 or result['memory_probe_calls']!=1 or result['packages']!=p['packages']:raise ValueError('Incomplete result')
    expected_map={k:str(v) for k,v in device_map_for('fixed_34_30').items()}
    if result['parameter_devices']!=['cuda:0','cuda:1'] or result['device_map']!=expected_map or result['generation']!=p['generation']:raise ValueError('Runtime placement or generation differs')
    if set(result['memory_after_load'])!={'0','1'} or any(not 0<v['allocated_bytes']<=13*GIB for v in result['memory_after_load'].values()):raise ValueError('Weight memory differs')
    resources=result['resources_before'];resource_gate(result['devices_before'],resources['cache_free_bytes'],resources['ram_available_bytes'],p['weight_file_bytes'])
    checks=result['attention_equivalence_checks'];wanted={(d,c) for d in (0,1) for c in ('prefill_causal','decode','masked_prefill')}
    if len(checks)!=6 or {(c['device'],c['case']) for c in checks}!=wanted or any(c.get('passed') is not True or c['atol']!=0.002 or c['rtol']!=0.01 for c in checks):raise ValueError('Kernel checks differ')
    result_dir=result_path.parent;raw=(result_dir/'judgments.jsonl').read_bytes();probe_raw=(result_dir/'memory_probe.json').read_bytes()
    if sha(raw)!=result['judgments_sha256'] or sha(probe_raw)!=result['memory_probe_sha256']:raise ValueError('Output journals changed')
    rows=[json.loads(line) for line in raw.decode().splitlines()]
    if len(rows)!=10 or [r['case_id'] for r in rows]!=[j['case_id'] for j in jobs]:raise ValueError('Judgment order differs')
    probe_out=json.loads(probe_raw);verify_generated(json.loads(frozen['memory_probe.json']),probe_out,16,p['generation'],tokenizer)
    for job,row in zip(jobs,rows):
        digest=sha(json.dumps(job['messages'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
        if row['model']!=CANDIDATE or row['input_sha256']!=digest:raise ValueError('Judgment identity differs')
        verify_generated(job,row['output'],1024,p['generation'],tokenizer)
    steps=len(probe_out['token_ids'])+sum(len(r['output']['token_ids']) for r in rows);stats=result['attention_runtime_stats']
    if stats.get('backend')!='EFFICIENT_ATTENTION' or stats.get('native_gqa') is not False or stats.get('mask_function')!='sdpa_mask' or stats.get('calls')!=16*steps or stats.get('expanded_kv_calls')!=16*steps:raise ValueError('Total backend coverage differs')
    peaks=result['peak_allocated_bytes']
    if set(peaks)!={'0','1'} or any(type(peaks[str(d['index'])]) is not int or not 0<peaks[str(d['index'])]<=d['total_memory_bytes'] for d in result['devices_before']):raise ValueError('GPU peaks differ')
    replayed=replay(values,jobs,rows);scores=score_rows(spec,replayed)
    if tokenizer is None:raise ValueError('Scoring requires local tokenizer reconstruction')
    if sha(tokenizer.chat_template.encode())!=p['chat_template_sha256']:raise ValueError('Candidate template differs')
    report.update(status='verified',tokenizer_reconstructed_locally=True,frozen_parser_replayed=True,judgments_sha256=sha(raw),
        control=p['control'],peak_gib={k:v/GIB for k,v in peaks.items()},evaluation_input_tokens=[len(j['input_token_ids']) for j in jobs],
        truncated_judgments=sum(r['output']['finish_reason']=='length' for r in rows),scores=scores,
        quality_policy=p['quality_policy'])
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',default='ghost_candidate_audit_v1');parser.add_argument('--tokenizer',type=Path,required=True)
    args=parser.parse_args();from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    report=verify(args.version,tokenizer=tokenizer);path=ROOT/'evaluation/runs'/args.version/'verification.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('version','status','run_status','evaluation_calls','memory_probe_calls')},indent=2))
    if 'scores' in report:print(json.dumps({k:report['scores'][k] for k in ('counts','pairs_both_correct','pairs_total')},indent=2))
