"""Verify exact27B synthesis execution and replay frozen dependent stages."""
import argparse
import ast
import base64
from collections import Counter
import json
import lzma
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from build_ghost_candidate_probe import ROOT,sha
from build_ghost_candidate_synthesis import control
from candidate_synthesis import validate_protocol
from candidate_audit import CANDIDATE,GIB,device_map_for,resource_gate
from verify_ghost_candidate_audit import verify_generated


def payloads(version,root=ROOT):
    if not re.fullmatch(r'ghost_candidate_synthesis_v[1-9][0-9]*',version):raise ValueError('Invalid synthesis version')
    stage=root/'training/kaggle_cli'/version;record=root/'evaluation'/version
    lock=json.loads((record/'bundle_lock.json').read_text());raw=(stage/'kaggle_ghost_candidate_synthesis.ipynb').read_bytes();meta=(stage/'kernel-metadata.json').read_bytes()
    if sha(raw)!=lock['notebook_sha256'] or sha(meta)!=lock['metadata_sha256']:raise ValueError('Frozen notebook differs')
    m=json.loads(meta)
    if m['is_private'] is not True or m['id_no']!=140792344:raise ValueError('Wrong target')
    code=''.join(next(c for c in json.loads(raw)['cells'] if c['id']=='run')['source']);tree=ast.parse(code)
    packed=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='packed' for t in n.targets)))
    values={n:lzma.decompress(base64.b85decode(v)) for n,v in packed.items()}
    if {n:sha(v) for n,v in values.items()}!=lock['payload_sha256']:raise ValueError('Frozen bundle differs')
    p=json.loads(values['protocol.json']);validate_protocol(p)
    if any(sha(values[n])!=digest for n,digest in p['files'].items()):raise ValueError('Protocol files differ')
    return values


def replay(frozen,jobs,calls):
    code='''import contextlib,io,json
from pathlib import Path
from candidate_synthesis import evaluate
p=Path('.');calls=json.loads((p/'recorded_calls.json').read_text());jobs=json.loads((p/'audit_inputs.json').read_text());index=0
def generate(messages):
 global index
 if index>=len(calls) or messages!=calls[index]['messages']:raise ValueError('Frozen synthesis request replay differs')
 out=calls[index]['output'];index+=1;return out
with contextlib.redirect_stdout(io.StringIO()):evaluate(jobs,generate,p/'replayed')
if index!=len(calls):raise ValueError('Unexpected generation tail')
'''
    with tempfile.TemporaryDirectory(prefix='candidate_synthesis_replay_') as temp:
        d=Path(temp)
        for n,raw in frozen.items():(d/n).write_bytes(raw)
        (d/'recorded_calls.json').write_text(json.dumps(calls));(d/'audit_inputs.json').write_text(json.dumps(jobs))
        subprocess.run([sys.executable,'-c',code],cwd=d,check=True,capture_output=True,text=True,timeout=60)
        return {n:(d/'replayed'/n).read_bytes() for n in ('answers.jsonl','model_calls.jsonl')}


def verify(version='ghost_candidate_synthesis_v1',root=ROOT,tokenizer=None):
    frozen=payloads(version,root);p=json.loads(frozen['protocol.json']);run=root/'evaluation/runs'/version;d=run/'download'
    for n,raw in frozen.items():
        if (d/'ghost_candidate_synthesis_bundle'/n).read_bytes()!=raw:raise ValueError('Downloaded bundle differs: '+n)
    for n,digest in p['preserved_files'].items():
        if sha((root/n).read_bytes())!=digest:raise ValueError('Protected archive differs')
    cases,old_calls,expected,binding=control(root)
    if binding!=p['control'] or sha(expected)!=p['evaluation_spec_sha256'] or expected!=(root/'evaluation'/version/'evaluation_spec.json').read_bytes():raise ValueError('Original control or expectations differ')
    jobs=json.loads(frozen['audit_inputs.json'])
    if len(jobs)!=5 or [j['messages'] for j in jobs]!=[c['messages'] for c in old_calls] or [j['case_id'] for j in jobs]!=[c['id'] for c in cases] or [j['question'] for j in jobs]!=[c['question'] for c in cases]:raise ValueError('Exact writer inputs differ')
    output=d/'ghost_candidate_synthesis_results';result_path=output/'result.json';r=json.loads(result_path.read_text())
    if any(r.get(k)!=p[k] for k in ('model','purpose','training_steps','adapters_loaded','active_model_changed','enable_thinking')) or r.get('quality_validated') is not False or r['protocol_sha256']!=sha(frozen['protocol.json']):raise ValueError('Candidate identity differs')
    if type(r['evaluation_calls']) is not int or not 0<=r['evaluation_calls']<=10 or type(r['memory_probe_calls']) is not int or not 0<=r['memory_probe_calls']<=1:raise ValueError('Call limits differ')
    report={'version':version,'status':'verified_artifacts','run_status':r['status'],'model':CANDIDATE,'result_sha256':sha(result_path.read_bytes()),'evaluation_calls':r['evaluation_calls'],
        'memory_probe_calls':r['memory_probe_calls'],'quality_validated':False,'active_model_changed':False}
    if r['status']=='blocked':
        if not r.get('error') or not r.get('error_type'):raise ValueError('Blocked reason absent')
        return {**report,'error_type':r['error_type'],'error':r['error'],'phase':r['phase']}
    if r['status']!='complete' or not 5<=r['evaluation_calls']<=10 or r['memory_probe_calls']!=1 or r['packages']!=p['packages'] or r['generation']!=p['generation']:raise ValueError('Incomplete or different execution')
    if r['device_map']!={k:str(v) for k,v in device_map_for('fixed_34_30').items()} or r['parameter_devices']!=['cuda:0','cuda:1']:raise ValueError('Placement differs')
    if set(r['memory_after_load'])!={'0','1'} or any(not 0<v['allocated_bytes']<=13*GIB for v in r['memory_after_load'].values()):raise ValueError('Weight allocation differs')
    resource_gate(r['devices_before'],r['resources_before']['cache_free_bytes'],r['resources_before']['ram_available_bytes'],p['weight_file_bytes'])
    checks=r['attention_equivalence_checks'];wanted={(d,c) for d in (0,1) for c in ('prefill_causal','decode','masked_prefill')}
    if len(checks)!=6 or {(c['device'],c['case']) for c in checks}!=wanted or any(c.get('passed') is not True or c['atol']!=0.002 or c['rtol']!=0.01 for c in checks):raise ValueError('Kernel checks differ')
    raw={n:(output/n).read_bytes() for n in ('answers.jsonl','model_calls.jsonl','memory_probe.json')}
    for name,key in [('answers.jsonl','answers_sha256'),('model_calls.jsonl','model_calls_sha256'),('memory_probe.json','memory_probe_sha256')]:
        if sha(raw[name])!=r[key]:raise ValueError('Output journal differs')
    calls=[json.loads(l) for l in raw['model_calls.jsonl'].decode().splitlines()];answers=[json.loads(l) for l in raw['answers.jsonl'].decode().splitlines()]
    if len(calls)!=r['evaluation_calls'] or len(calls)!=r['completed_evaluation_calls'] or len(answers)!=5 or r['displayed_answers']!=5 or [a['case_id'] for a in answers]!=[j['case_id'] for j in jobs]:raise ValueError('Journal coverage differs')
    probe=json.loads(raw['memory_probe.json']);verify_generated(json.loads(frozen['memory_probe.json']),probe,16,p['generation'],tokenizer)
    for call in calls:
        out=call['output'];verify_generated({'messages':call['messages'],'input_token_ids':out['input_token_ids']},out,1024,p['generation'],tokenizer)
    replayed=replay(frozen,jobs,calls)
    if any(raw[n]!=data for n,data in replayed.items()):raise ValueError('Frozen synthesis answer/call replay differs')
    steps=len(probe['token_ids'])+sum(len(c['output']['token_ids']) for c in calls);stats=r['attention_runtime_stats']
    if stats.get('backend')!='EFFICIENT_ATTENTION' or stats.get('native_gqa') is not False or stats.get('mask_function')!='sdpa_mask' or stats.get('calls')!=16*steps or stats.get('expanded_kv_calls')!=16*steps:raise ValueError('Backend coverage differs')
    if set(r['peak_allocated_bytes'])!={'0','1'} or any(type(r['peak_allocated_bytes'][str(d['index'])]) is not int or not 0<r['peak_allocated_bytes'][str(d['index'])]<=d['total_memory_bytes'] for d in r['devices_before']):raise ValueError('Impossible peaks')
    if tokenizer is None or sha(tokenizer.chat_template.encode())!=p['chat_template_sha256']:raise ValueError('Local tokenizer required')
    for a in answers:
        if a['model']!=CANDIDATE or a['accepted_claim_ids']!=[] or a['memory_updates']!=[] or a['semantic_guarantee'] is not False:raise ValueError('Unvalidated factual promotion')
    return {**report,'status':'verified','tokenizer_reconstructed_locally':True,'frozen_pipeline_replayed':True,
        'answers_sha256':sha(raw['answers.jsonl']),'model_calls_sha256':sha(raw['model_calls.jsonl']),
        'answer_statuses':dict(Counter(a['status'] for a in answers)),'truncated_generations':sum(c['output']['finish_reason']=='length' for c in calls),
        'peak_gib':{k:v/GIB for k,v in r['peak_allocated_bytes'].items()},'quality_policy':p['quality_policy']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',default='ghost_candidate_synthesis_v1');parser.add_argument('--tokenizer',required=True,type=Path)
    args=parser.parse_args();from transformers import AutoTokenizer
    report=verify(args.version,tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False))
    (ROOT/'evaluation/runs'/args.version/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
