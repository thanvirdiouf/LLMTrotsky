"""Verify real conversational review replies and dependent retrieval/history replay."""
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
from candidate_review_chat import validate_protocol
from candidate_base_probe import CANDIDATE,GIB,device_map_for,resource_gate
from verify_ghost_candidate_audit import verify_generated
from grounded_chat import fingerprint


def payloads(version,root=ROOT):
    if not re.fullmatch(r'ghost_review_chat_v[1-9][0-9]*',version):raise ValueError('Invalid reviewed selector version')
    stage=root/'training/kaggle_cli'/version;lock=json.loads((root/'evaluation'/version/'bundle_lock.json').read_text())
    raw=(stage/'kaggle_review_chat.ipynb').read_bytes();meta=(stage/'kernel-metadata.json').read_bytes()
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


def replay(frozen,calls,tokenizer_path):
    code='''import contextlib,io,json,sys
from pathlib import Path
from transformers import AutoTokenizer
from candidate_review_chat import evaluate
p=Path('.');calls=json.loads((p/'recorded_calls.json').read_text());index=0
def generate(messages):
 global index
 if index>=len(calls) or messages!=calls[index]['messages']:raise ValueError('Conversation request replay differs')
 out=calls[index]['output'];index+=1;return out
tok=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True,trust_remote_code=False)
with contextlib.redirect_stdout(io.StringIO()):
 evaluate(json.loads((p/'audit_inputs.json').read_text()),generate,p/'replayed',*[json.loads((p/n).read_text()) for n in ('profile.json','collection.json','ledger.json')],tok,json.loads((p/'review_design.json').read_text())['current_date'])
if index!=len(calls):raise ValueError('Unexpected conversation call tail')
'''
    with tempfile.TemporaryDirectory(prefix='review_chat_replay_') as temp:
        directory=Path(temp)
        for n,raw in frozen.items():(directory/n).write_bytes(raw)
        (directory/'recorded_calls.json').write_text(json.dumps(calls))
        subprocess.run([sys.executable,'-c',code,str(Path(tokenizer_path).resolve())],cwd=directory,check=True,capture_output=True,text=True,timeout=90)
        return {n:(directory/'replayed'/n).read_bytes() for n in ('answers.jsonl','model_calls.jsonl')}


def verify(version='ghost_review_chat_v4',root=ROOT,tokenizer=None,tokenizer_path=None):
    f=payloads(version,root);p=json.loads(f['protocol.json']);run=root/'evaluation/runs'/version;d=run/'download'
    for n,raw in f.items():
        if (d/'ghost_review_chat_bundle'/n).read_bytes()!=raw:raise ValueError('Downloaded conversation payload differs: '+n)
    design=json.loads(f['review_design.json']);source=root/design['source_snapshot']
    if sha(f['review_design.json'])!=p['design_sha256'] or sha((source/'build_manifest.json').read_bytes())!=p['source_manifest_sha256'] or sha((source/'SOURCE_REVIEW.json').read_bytes())!=p['source_review_sha256']:raise ValueError('Conversation source/design provenance differs')
    for n in ('profile.json','collection.json','ledger.json'):
        if f[n]!=(source/n).read_bytes():raise ValueError('Source snapshot changed')
    for n,digest in p['preserved_files'].items():
        if sha((root/n).read_bytes())!=digest:raise ValueError('Protected archive changed')
    output=d/'ghost_review_chat_results';r=json.loads((output/'result.json').read_text())
    if any(r.get(k)!=p[k] for k in ('model','purpose','training_steps','adapters_loaded','active_model_changed','enable_thinking')) or r.get('quality_validated') is not False or r['protocol_sha256']!=sha(f['protocol.json']):raise ValueError('Review execution scope differs')
    if type(r['evaluation_calls']) is not int or not 0<=r['evaluation_calls']<=p['evaluation_calls'] or type(r['memory_probe_calls']) is not int or not 0<=r['memory_probe_calls']<=1:raise ValueError('Review call budgets differ')
    report={'version':version,'status':'verified_artifacts','run_status':r['status'],'result_sha256':sha((output/'result.json').read_bytes()),'evaluation_calls':r['evaluation_calls'],'memory_probe_calls':r['memory_probe_calls'],'quality_validated':False,'active_model_changed':False}
    if r['status']=='blocked':return {**report,'error_type':r.get('error_type'),'error':r.get('error'),'phase':r.get('phase')}
    if r['status']!='complete' or r['memory_probe_calls']!=1 or r['packages']!=p['packages'] or r['generation']!=p['generation'] or r.get('model_audit_performed') is not False:raise ValueError('Review incomplete or different settings')
    if r['parameter_devices']!=['cuda:0','cuda:1'] or r['device_map']!={k:str(v) for k,v in device_map_for('fixed_34_30').items()}:raise ValueError('GPU placement differs')
    if set(r['memory_after_load'])!={'0','1'} or any(not 0<v['allocated_bytes']<=13*GIB for v in r['memory_after_load'].values()):raise ValueError('Weight memory differs')
    resource_gate(r['devices_before'],r['resources_before']['cache_free_bytes'],r['resources_before']['ram_available_bytes'],p['weight_file_bytes'])
    checks=r['attention_equivalence_checks'];wanted={(dev,case) for dev in (0,1) for case in ('prefill_causal','decode','masked_prefill')}
    if len(checks)!=6 or {(c['device'],c['case']) for c in checks}!=wanted or any(c.get('passed') is not True or c['atol']!=0.002 or c['rtol']!=0.01 for c in checks):raise ValueError('Kernel checks differ')
    raw={n:(output/n).read_bytes() for n in ('answers.jsonl','model_calls.jsonl','memory_probe.json')}
    for n,key in [('answers.jsonl','answers_sha256'),('model_calls.jsonl','model_calls_sha256'),('memory_probe.json','memory_probe_sha256')]:
        if sha(raw[n])!=r[key]:raise ValueError('Conversation journal hash differs')
    calls=[json.loads(l) for l in raw['model_calls.jsonl'].decode().splitlines()];answers=[json.loads(l) for l in raw['answers.jsonl'].decode().splitlines()]
    jobs=json.loads(f['audit_inputs.json']);expected=[(j['case_id'],i+1) for j in jobs for i in range(len(j['questions']))]
    if p.get('seeded_reply_count'):
        control=p['seed_control'];parent=root/'evaluation/runs'/control['local_version'];parent_output=parent/'download/ghost_review_chat_results'
        prior=json.loads((parent/'verification.json').read_text());prior_calls=(parent_output/'model_calls.jsonl').read_bytes()
        if (control['saved_version']!=47 or prior['status']!='verified' or prior['result_sha256']!=control['result_sha256']
                or sha((parent_output/'result.json').read_bytes())!=control['result_sha256']
                or sha(prior_calls)!=control['model_calls_sha256'] or prior['model_calls_sha256']!=control['model_calls_sha256']):raise ValueError('Archived seed provenance differs')
        archived=[json.loads(line) for line in prior_calls.decode().splitlines()]
        if p['seeded_reply_count']!=5 or len([j for j in jobs if 'seed_reply' in j])!=5:raise ValueError('Expected five archived first replies')
        for job in jobs:
            matched=[c for c in archived if c['case_id']==job['case_id'] and c['turn']==1]
            if len(matched)!=1 or job['seed_reply']!=matched[0] or job['seed_reply']['messages']!=job['messages']:raise ValueError('Archived reply or prompt changed')
    if len(calls)!=r['evaluation_calls'] or len(answers)!=10 or r['completed_evaluation_calls']!=len(calls) or r['displayed_answers']!=10 or [(a['case_id'],a['turn']) for a in answers]!=expected:raise ValueError('Conversation coverage differs')
    probe=json.loads(raw['memory_probe.json']);verify_generated(json.loads(f['memory_probe.json']),probe,16,p['generation'],tokenizer)
    for call in calls:
        if call['stage']!='conversation':raise ValueError('Unexpected model stage')
        verify_generated({'messages':call['messages'],'input_token_ids':call['output']['input_token_ids']},call['output'],1024,p['generation'],tokenizer)
    repeated=replay(f,calls,tokenizer_path)
    if any(raw[n]!=data for n,data in repeated.items()):raise ValueError('Dependent conversation/retrieval/parser replay differs')
    steps=len(probe['token_ids'])+sum(len(c['output']['token_ids']) for c in calls);stats=r['attention_runtime_stats']
    if stats.get('backend')!='EFFICIENT_ATTENTION' or stats.get('native_gqa') is not False or stats.get('mask_function')!='sdpa_mask' or stats.get('calls')!=16*steps or stats.get('expanded_kv_calls')!=16*steps:raise ValueError('Backend coverage differs')
    peaks=r['peak_allocated_bytes']
    if set(peaks)!={'0','1'} or any(not 0<peaks[str(dev['index'])]<=dev['total_memory_bytes'] for dev in r['devices_before']):raise ValueError('Impossible GPU peaks')
    if tokenizer is None or sha(tokenizer.chat_template.encode())!=p['chat_template_sha256']:raise ValueError('Exact local tokenizer required')
    if any(a['quality_validated'] is not False or a['model_audit_performed'] is not False or a['memory_updates']!=[] for a in answers):raise ValueError('Unvalidated quality or memory promotion')
    return {**report,'status':'verified','tokenizer_reconstructed_locally':True,'frozen_pipeline_replayed':True,
        'answers_sha256':sha(raw['answers.jsonl']),'model_calls_sha256':sha(raw['model_calls.jsonl']),
        'answer_statuses':dict(Counter(a['status'] for a in answers)),'generation_origins':dict(Counter(a.get('generation_origin','original_run') for a in answers)),
        'reused_model_replies':p.get('seeded_reply_count',0),'new_model_calls':len(calls),
        'truncated_generations':sum(c['output']['finish_reason']=='length' for c in calls),
        'peak_gib':{k:v/GIB for k,v in peaks.items()},'human_review_status':'unreviewed'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',default='ghost_review_chat_v4');parser.add_argument('--tokenizer',required=True,type=Path);args=parser.parse_args()
    from transformers import AutoTokenizer
    report=verify(args.version,tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False),tokenizer_path=args.tokenizer)
    (ROOT/'evaluation/runs'/args.version).mkdir(parents=True,exist_ok=True)
    (ROOT/'evaluation/runs'/args.version/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
