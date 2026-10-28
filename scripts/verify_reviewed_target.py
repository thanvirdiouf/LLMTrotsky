"""Verify focused target-span selection with unchanged runtime and local diagnostic checks."""
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
from prepare_reviewed_target import frozen_inputs,score_targets
from candidate_reviewed_selection import validate_protocol
from candidate_base_probe import CANDIDATE,GIB,device_map_for,resource_gate
from verify_ghost_candidate_audit import verify_generated
from grounded_chat import fingerprint


def payloads(version,root=ROOT):
    if not re.fullmatch(r'ghost_reviewed_target_v[1-9][0-9]*',version):raise ValueError('Invalid reviewed selector version')
    stage=root/'training/kaggle_cli'/version;lock=json.loads((root/'evaluation'/version/'bundle_lock.json').read_text())
    raw=(stage/'kaggle_reviewed_target.ipynb').read_bytes();meta=(stage/'kernel-metadata.json').read_bytes()
    if sha(raw)!=lock['notebook_sha256'] or sha(meta)!=lock['metadata_sha256']:raise ValueError('Frozen notebook differs')
    m=json.loads(meta)
    if m['is_private'] is not True or m['id_no']!=140792344:raise ValueError('Wrong private target')
    tree=ast.parse(''.join(next(c for c in json.loads(raw)['cells'] if c['id']=='run')['source']))
    packed=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='packed' for t in n.targets)))
    values={n:lzma.decompress(base64.b85decode(v)) for n,v in packed.items()}
    if {n:sha(v) for n,v in values.items()}!=lock['payload_sha256']:raise ValueError('Frozen payload differs')
    p=json.loads(values['protocol.json']);validate_protocol(p)
    if any(sha(values[n])!=digest for n,digest in p['files'].items()):raise ValueError('Protocol files differ')
    return values


def replay(frozen,calls):
    code='''import contextlib,io,json
from pathlib import Path
from candidate_reviewed_selection import evaluate
p=Path('.');calls=json.loads((p/'recorded_calls.json').read_text());jobs=json.loads((p/'audit_inputs.json').read_text());index=0
def generate(messages):
 global index
 if index>=len(calls) or messages!=calls[index]['messages']:raise ValueError('Selector request replay differs')
 out=calls[index]['output'];index+=1;return out
with contextlib.redirect_stdout(io.StringIO()):
 evaluate(jobs,generate,p/'replayed',json.loads((p/'collection.json').read_text()),json.loads((p/'ledger.json').read_text()))
if index!=len(calls):raise ValueError('Unexpected selector call tail')
'''
    with tempfile.TemporaryDirectory(prefix='reviewed_selection_replay_') as temp:
        d=Path(temp)
        for n,raw in frozen.items():(d/n).write_bytes(raw)
        (d/'recorded_calls.json').write_text(json.dumps(calls))
        subprocess.run([sys.executable,'-c',code],cwd=d,check=True,capture_output=True,text=True,timeout=60)
        return {n:(d/'replayed'/n).read_bytes() for n in ('answers.jsonl','model_calls.jsonl')}


def verify(version='ghost_reviewed_target_v1',root=ROOT,tokenizer=None):
    f=payloads(version,root);p=json.loads(f['protocol.json']);run=root/'evaluation/runs'/version;d=run/'download'
    if p.get('experiment_kind')!='literal_target_diagnostic' or p.get('target_selection_is_semantic_validation') is not False:raise ValueError('Target diagnostic scope differs')
    expected_raw=(run/'expectations.json').read_bytes()
    if sha(expected_raw)!=p['local_expectations_sha256']:raise ValueError('Frozen local diagnostic expectations changed')
    from verify_reviewed_selection import payloads as parent_payloads
    parent=parent_payloads('ghost_reviewed_selection_v1',root)
    if any(raw!=parent[name] for name,raw in f.items() if name not in ('audit_inputs.json','protocol.json')):raise ValueError('Shared runtime/source control changed')
    for n,raw in f.items():
        if (d/'ghost_reviewed_target_bundle'/n).read_bytes()!=raw:raise ValueError('Downloaded bundle differs: '+n)
    inputs,_,binding=frozen_inputs(root,renderer=f['reviewed_source_display.py'])
    if binding!=p['frozen_inputs'] or f['audit_inputs.json']!=inputs['selection_requests.json']:raise ValueError('Frozen selector requests differ')
    for n in ('profile.json','collection.json','ledger.json','memory_probe.json'):
        if f[n]!=inputs[n]:raise ValueError('Frozen source/ledger/probe differs')
    if fingerprint(json.loads(f['ledger.json']))!=p['ledger_fingerprint']:raise ValueError('Reviewed ledger identity differs')
    for n,digest in p['preserved_files'].items():
        if sha((root/n).read_bytes())!=digest:raise ValueError('Protected archive changed')
    output=d/'ghost_reviewed_target_results';path=output/'result.json';r=json.loads(path.read_text())
    if any(r.get(k)!=p[k] for k in ('model','purpose','training_steps','adapters_loaded','active_model_changed','enable_thinking')) or r.get('quality_validated') is not False or r['protocol_sha256']!=sha(f['protocol.json']):raise ValueError('Result scope differs')
    if type(r['evaluation_calls']) is not int or not 0<=r['evaluation_calls']<=5 or type(r['memory_probe_calls']) is not int or not 0<=r['memory_probe_calls']<=1:raise ValueError('Call bounds differ')
    report={'version':version,'model':CANDIDATE,'status':'verified_artifacts','run_status':r['status'],'result_sha256':sha(path.read_bytes()),
        'experiment_kind':'literal_target_diagnostic','answers_are_target_span_diagnostics':True,
        'evaluation_calls':r['evaluation_calls'],'memory_probe_calls':r['memory_probe_calls'],'quality_validated':False,'active_model_changed':False}
    if r['status']=='blocked':
        if not r.get('error') or not r.get('error_type'):raise ValueError('Blocked reason absent')
        return {**report,'error_type':r['error_type'],'error':r['error'],'phase':r['phase']}
    if (r['status']!='complete' or r['evaluation_calls']!=5 or r['memory_probe_calls']!=1 or r['packages']!=p['packages'] or r['generation']!=p['generation']
            or r.get('generated_factual_text_used') is not False or r.get('model_audit_performed') is not False):raise ValueError('Incomplete or different execution')
    if r['parameter_devices']!=['cuda:0','cuda:1'] or r['device_map']!={k:str(v) for k,v in device_map_for('fixed_34_30').items()}:raise ValueError('Placement differs')
    if set(r['memory_after_load'])!={'0','1'} or any(not 0<v['allocated_bytes']<=13*GIB for v in r['memory_after_load'].values()):raise ValueError('Weight memory differs')
    resource_gate(r['devices_before'],r['resources_before']['cache_free_bytes'],r['resources_before']['ram_available_bytes'],p['weight_file_bytes'])
    wanted={(d,c) for d in (0,1) for c in ('prefill_causal','decode','masked_prefill')};checks=r['attention_equivalence_checks']
    if len(checks)!=6 or {(c['device'],c['case']) for c in checks}!=wanted or any(c.get('passed') is not True or c['atol']!=0.002 or c['rtol']!=0.01 for c in checks):raise ValueError('Kernel checks differ')
    raw={n:(output/n).read_bytes() for n in ('answers.jsonl','model_calls.jsonl','memory_probe.json')}
    for n,k in [('answers.jsonl','answers_sha256'),('model_calls.jsonl','model_calls_sha256'),('memory_probe.json','memory_probe_sha256')]:
        if sha(raw[n])!=r[k]:raise ValueError('Journal hashes differ')
    calls=[json.loads(l) for l in raw['model_calls.jsonl'].decode().splitlines()];answers=[json.loads(l) for l in raw['answers.jsonl'].decode().splitlines()];jobs=json.loads(f['audit_inputs.json'])
    if len(calls)!=5 or len(answers)!=5 or r['completed_evaluation_calls']!=5 or r['displayed_answers']!=5 or [a['case_id'] for a in answers]!=[j['case_id'] for j in jobs]:raise ValueError('Coverage differs')
    probe=json.loads(raw['memory_probe.json']);verify_generated(json.loads(f['memory_probe.json']),probe,16,p['generation'],tokenizer)
    for job,call in zip(jobs,calls):
        if call['case_id']!=job['case_id'] or call['stage']!='select' or call['messages']!=job['messages']:raise ValueError('Selector stage/request differs')
        verify_generated(job,call['output'],1024,p['generation'],tokenizer)
    repeated=replay(f,calls)
    if any(raw[n]!=data for n,data in repeated.items()):raise ValueError('Frozen display/selection replay differs')
    steps=len(probe['token_ids'])+sum(len(c['output']['token_ids']) for c in calls);stats=r['attention_runtime_stats']
    if stats.get('backend')!='EFFICIENT_ATTENTION' or stats.get('native_gqa') is not False or stats.get('mask_function')!='sdpa_mask' or stats.get('calls')!=16*steps or stats.get('expanded_kv_calls')!=16*steps:raise ValueError('Backend coverage differs')
    peaks=r['peak_allocated_bytes']
    if set(peaks)!={'0','1'} or any(type(peaks[str(d['index'])]) is not int or not 0<peaks[str(d['index'])]<=d['total_memory_bytes'] for d in r['devices_before']):raise ValueError('Impossible peaks')
    if tokenizer is None or sha(tokenizer.chat_template.encode())!=p['chat_template_sha256']:raise ValueError('Local tokenizer reconstruction required')
    for a in answers:
        if (a['model']!=CANDIDATE or a['generated_factual_text_used'] is not False or a['model_audit_performed'] is not False
                or a['requested_task_coverage_verified'] is not False or a['selection_relevance_verified'] is not False or a['semantic_guarantee'] is not False
                or a['accepted_claim_ids']!=[] or a['memory_updates']!=[] or a['selected_claim_ids']!=a['rendered_claim_ids'] or a['omitted_selected_claim_ids']!=[]):
            raise ValueError('Unvalidated promotion, rewriting or omitted selected records')
    return {**report,'target_diagnostic':score_targets(answers,json.loads(expected_raw)),'status':'verified','tokenizer_reconstructed_locally':True,'frozen_pipeline_replayed':True,
        'answers_sha256':sha(raw['answers.jsonl']),'model_calls_sha256':sha(raw['model_calls.jsonl']),
        'answer_statuses':dict(Counter(a['status'] for a in answers)),'selected_record_counts':[len(a['selected_claim_ids']) for a in answers],
        'truncated_generations':sum(c['output']['finish_reason']=='length' for c in calls),'peak_gib':{k:v/GIB for k,v in peaks.items()},
        'quality_policy':p['quality_policy']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',default='ghost_reviewed_target_v1');parser.add_argument('--tokenizer',required=True,type=Path)
    args=parser.parse_args();from transformers import AutoTokenizer
    report=verify(args.version,tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False))
    (ROOT/'evaluation/runs'/args.version/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
