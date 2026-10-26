#!/usr/bin/env python3
"""Validate ghost smoke artifacts and replay frozen code without model inference."""
import argparse
import ast
from collections import Counter
import base64
import hashlib
import json
import lzma
import re
from pathlib import Path
import subprocess
import sys
import tempfile
from grounded_chat import ROOT,MODEL


def sha(raw):return hashlib.sha256(raw).hexdigest()


def frozen_payloads(root=ROOT,version='ghost_smoke_v1'):
    if not re.fullmatch(r'ghost_smoke_v[1-9][0-9]*',version):raise ValueError('Invalid smoke version')
    lock=json.loads((root/'evaluation'/version/'bundle_lock.json').read_text())
    stage=root/'training/kaggle_cli'/version
    raw=(stage/'kaggle_ghost_smoke.ipynb').read_bytes()
    if sha(raw)!=lock['notebook_sha256'] or sha((stage/'kernel-metadata.json').read_bytes())!=lock['metadata_sha256']:
        raise ValueError('Frozen ghost submission changed')
    notebook=json.loads(raw);cell=next(c for c in notebook['cells'] if c['id']=='unpack')
    tree=ast.parse(''.join(cell['source']))
    packed=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='packed' for t in n.targets)))
    payloads={n:lzma.decompress(base64.b85decode(v['data'])) for n,v in packed.items()}
    if {n:sha(b) for n,b in payloads.items()}!=lock['payload_sha256']:raise ValueError('Frozen payload mismatch')
    return payloads


def verify(download,tokenizer=None,version='ghost_smoke_v1'):
    download=Path(download);payloads=frozen_payloads(version=version);protocol=json.loads(payloads['protocol.json'])
    bundle=download/'ghost_smoke_bundle';results=download/'ghost_smoke_results'
    for n,b in payloads.items():
        if (bundle/n).read_bytes()!=b:raise ValueError('Downloaded bundle differs: '+n)
    for n,h in protocol['files'].items():
        if sha(payloads[n])!=h:raise ValueError('Runtime bundle hash differs')
    for n,h in protocol['preserved_files'].items():
        if sha((ROOT/n).read_bytes())!=h:raise ValueError('Archived code or adapter changed: '+n)
    manifest=json.loads((results/'run_manifest.json').read_text());identity=manifest['identity']
    if (manifest['status']!='complete' or manifest['answers']!=protocol['answers']
        or manifest['model_calls']>protocol['maximum_model_calls'] or identity['model']!=MODEL
        or identity['packages']!=protocol['packages'] or identity['adapters_loaded']!=[]
        or identity['optimizer_updates']!=0 or not manifest['all_base_parameters_frozen']
        or identity['protocol_sha256']!=sha(payloads['protocol.json']) or identity['enable_thinking'] is not False):
        raise ValueError('Incomplete or mixed ghost execution')
    for k,v in protocol['generation'].items():
        if identity['generation'].get(k)!=v:raise ValueError('Generation changed')
    for name,field in [('answers.jsonl','answers_sha256'),('model_calls.jsonl','model_calls_sha256')]:
        if sha((results/name).read_bytes())!=manifest[field]:raise ValueError('Journal hash differs')
    answers=[json.loads(s) for s in (results/'answers.jsonl').read_text().splitlines()]
    calls=[json.loads(s) for s in (results/'model_calls.jsonl').read_text().splitlines()]
    if len(answers)!=protocol['answers'] or len(calls)!=manifest['model_calls']:raise ValueError('Journal coverage differs')
    # Frozen helper code is replayed in a separate local Python process, making
    # this verifier usable even after the active application evolves.
    replay="""import json,contextlib,io
from pathlib import Path
p=Path('.')
if json.loads((p/'protocol.json').read_text())['purpose']=='ghost_direct_checked_comparison':
 from evaluate_ghost_comparison import evaluate
elif json.loads((p/'protocol.json').read_text())['purpose']=='ghost_audit_calibration':
 from evaluate_ghost_audit import evaluate
else:
 from evaluate_ghost_smoke import evaluate
def no_inference(_): raise AssertionError('Missing cached call; inference forbidden')
with contextlib.redirect_stdout(io.StringIO()):
 a,c=evaluate(json.loads((p/'profile.json').read_text()),json.loads((p/'collection.json').read_text()),json.loads((p/'claims.json').read_text()),json.loads((p/'model_cases.json').read_text()),no_inference,p,json.loads((p/'identity.json').read_text()))
assert len(a)==len(json.loads((p/'coverage.json').read_text())['answers'])
"""
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)
        for n,b in payloads.items():(path/n).write_bytes(b)
        for name in ('answers.jsonl','model_calls.jsonl'):(path/name).write_bytes((results/name).read_bytes())
        (path/'identity.json').write_text(json.dumps(identity));(path/'coverage.json').write_text(json.dumps({'answers':answers}))
        result=subprocess.run([sys.executable,'-c',replay],cwd=path,capture_output=True,text=True)
        if result.returncode:raise ValueError('Frozen replay failed: '+result.stderr[-2000:])
    probe=json.loads((results/'memory_probe.json').read_text())
    if probe['result']['input_token_ids']!=json.loads(payloads['memory_probe.json'])['input_token_ids']:
        raise ValueError('Probe inputs differ')
    outputs=[(c['messages'],c['output'],protocol['generation']['max_new_tokens']) for c in calls]
    outputs.append((json.loads(payloads['memory_probe.json'])['messages'],probe['result'],16))
    for messages,out,limit in outputs:
        ids,tokens=out['input_token_ids'],out['token_ids']
        if any(type(i)is not int or i<0 for i in ids+tokens):raise ValueError('Invalid tokens')
        if out.get('model_call_made') is False:
            if out['finish_reason']!='input_budget' or len(ids)<=protocol['max_input_tokens'] or tokens or out['response']:
                raise ValueError('Invalid pre-inference budget rejection')
        else:
            if not tokens or len(ids)>protocol['max_input_tokens'] or len(tokens)>limit or len(tokens)!=out['token_count']:
                raise ValueError('Invalid generation token bounds')
            finish='stop' if tokens[-1]==protocol['generation']['eos_token_id'] else 'length'
            if out['finish_reason']!=finish or (finish=='length' and len(tokens)!=limit):raise ValueError('Invalid stop record')
        if tokenizer:
            if tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)!=ids:
                raise ValueError('Prompt reconstruction differs')
            if tokenizer.decode(tokens,skip_special_tokens=True)!=out['response']:raise ValueError('Decoded output differs')
    return {'status':'verified','answers':len(answers),'evaluation_stage_requests':len(calls),
        'evaluation_generations':sum(c['output'].get('model_call_made') is True for c in calls),
        'input_budget_rejections':sum(c['output'].get('model_call_made') is False for c in calls),
        'truncated_evaluation_generations':sum(c['output']['finish_reason']=='length' for c in calls),
        'memory_probe_tokens':len(probe['result']['input_token_ids']),
        'probe_peak_gib':probe['peak_allocated_bytes']/1024**3,'run_peak_gib':manifest['peak_allocated_bytes']/1024**3,
        'answer_statuses':dict(Counter(a['status'] for a in answers)),
        'tokenizer_reconstructed_locally':tokenizer is not None,
        'answers_sha256':manifest['answers_sha256'],'model_calls_sha256':manifest['model_calls_sha256'],
        'training_steps':0,'adapters_loaded':[],'quality_validation':'Not established by execution or model-checker approval'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',default='ghost_smoke_v1')
    parser.add_argument('--download',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--tokenizer',type=Path)
    args=parser.parse_args();tokenizer=None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,trust_remote_code=False,local_files_only=True)
    if not re.fullmatch(r'ghost_smoke_v[1-9][0-9]*',args.version):parser.error('Invalid smoke version')
    download=args.download or ROOT/'evaluation/runs'/args.version/'download'
    output=args.output or ROOT/'evaluation/runs'/args.version/'verification.json'
    result=verify(download,tokenizer,args.version);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
