"""Freeze ten exact archived9B neutral-audit messages for the approved27B path."""
import argparse
import ast
import base64
import json
import lzma
from pathlib import Path
import re
from build_ghost_candidate_probe import ROOT,ASSETS,enc,sha
from candidate_audit import validate_protocol,input_ids
from verify_ghost_candidate_probe import payloads as probe_payloads,verify as verify_probe
from verify_ghost_smoke import frozen_payloads
from build_ghost_smoke import model_cases

CONTROL='ghost_smoke_v30'
PROBE='ghost_candidate_probe_v3'


def freeze_inputs(spec,calls,tokenizer):
    if len(spec['cases'])!=10 or len(calls)!=10 or spec.get('conversation'):raise ValueError('Expected ten fixed independent cases')
    jobs=[]
    for case,call in zip(spec['cases'],calls):
        if call['key']!=case['id']+'/1' or call['step']!=1:raise ValueError('Control call order differs')
        data=json.loads(call['messages'][1]['content'])
        if data['candidate_segments']!=case['segments'] or data['current_user_request']!=case['question']:
            raise ValueError('Control candidate or question differs')
        jobs.append({'case_id':case['id'],'messages':call['messages'],'input_token_ids':input_ids(tokenizer,call['messages'])})
    if len({j['case_id'] for j in jobs})!=10:raise ValueError('Duplicate case IDs')
    return jobs


def control(root=ROOT):
    values=frozen_payloads(root=root,version=CONTROL);p=json.loads(values['protocol.json'])
    record=root/'evaluation'/CONTROL;run=root/'evaluation/runs'/CONTROL;download=run/'download/ghost_smoke_results'
    verification=json.loads((run/'verification.json').read_text())
    manifest=json.loads((download/'run_manifest.json').read_text())
    if verification['status']!='verified' or verification.get('tokenizer_reconstructed_locally') is not True:raise ValueError('Control must be token verified')
    for name,key in [('model_calls.jsonl','model_calls_sha256'),('answers.jsonl','answers_sha256')]:
        if sha((download/name).read_bytes())!=verification[key] or verification[key]!=manifest[key]:raise ValueError('Control journal changed')
    if manifest['identity']['protocol_sha256']!=sha(values['protocol.json']) or manifest['identity']['model']!=p['model'] or manifest['status']!='complete':raise ValueError('Control runtime identity differs')
    if any(manifest['identity']['generation'].get(k)!=v for k,v in p['generation'].items()):raise ValueError('Control generation differs')
    expected=(record/'evaluation_spec.json').read_bytes();spec=json.loads(expected)
    if sha(expected)!=p['evaluation_spec_sha256'] or model_cases(spec)!=json.loads(values['model_cases.json']):raise ValueError('Control expectations changed')
    if p['purpose']!='ghost_audit_calibration' or json.loads(values['model_cases.json'])['audit_calibration'].get('policy')!='neutral_source' or p.get('enable_thinking',False) is not False:raise ValueError('Wrong control')
    calls=[json.loads(line) for line in (download/'model_calls.jsonl').read_text().splitlines()]
    binding={'local_version':CONTROL,'saved_version':32,'model':p['model'],'evaluation_spec_sha256':sha(expected),
        'model_calls_sha256':verification['model_calls_sha256'],'answers_sha256':verification['answers_sha256'],
        'protocol_sha256':sha(values['protocol.json'])}
    return values,{**p,'generation':manifest['identity']['generation']},spec,expected,calls,binding


def build(version='ghost_candidate_audit_v1',root=ROOT,tokenizer=None,generated_diagnostic=False):
    if not re.fullmatch(r'ghost_candidate_audit_v[1-9][0-9]*',version):raise ValueError('Invalid candidate audit version')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(root/'evaluation'/PROBE/'tokenizer_assets',local_files_only=True,trust_remote_code=False)
    verified=verify_probe(PROBE,root,tokenizer)
    if verified.get('memory_success') is not True or not verified.get('tokenizer_reconstructed_locally'):raise ValueError('Verified candidate memory success required')
    probed=probe_payloads(PROBE,root);probe_p=json.loads(probed['protocol.json'])
    values,old,spec,expected,calls,binding=control(root)
    diagnostic=None
    if generated_diagnostic:
        from generated_audit_design import load
        spec,expected,prepared,diagnostic=load(root,values,spec,tokenizer)
        jobs=[{**j,'input_token_ids':input_ids(tokenizer,j['messages'])} for j in prepared]
    else:jobs=freeze_inputs(spec,calls,tokenizer)
    names=['candidate_audit.py','candidate_base_probe.py','candidate_efficient_attention.py']
    payload={name:(root/'scripts'/name).read_bytes() for name in names}
    for name in names[1:]:
        if payload[name]!=probed[name]:raise ValueError('Successful candidate backend/placement helper changed')
    payload['audit_inputs.json']=enc(jobs);payload['memory_probe.json']=probed['memory_probe.json']
    p={key:probe_p[key] for key in ('model','packages','gpu_ids','placement_policy','attention_policy','tokenizer_files','chat_template_sha256',
        'per_gpu_weight_budget_bytes','weight_file_bytes','preserved_files','runtime_timeout_seconds','installation_timeout_seconds')}
    p.update(purpose='candidate_neutral_audit',evaluation_calls=10,memory_probe_calls=1,max_input_tokens=5120,max_output_tokens=1024,
        memory_probe_output_tokens=16,enable_thinking=False,training_steps=0,adapters_loaded=[],active_model_changed=False,
        generation=old['generation'],control=binding,prerequisite_probe={'local_version':PROBE,'saved_version':38,'result_sha256':verified['result_sha256']},
        evaluation_spec_sha256=sha(expected),files={name:sha(raw) for name,raw in payload.items()},
        local_scorer_sha256=sha((root/'scripts/summarize_ghost_calibration.py').read_bytes()),
        quality_policy='Fixed development diagnostic; coding-agent expectations are not independent or human validation. No automatic activation.')
    if diagnostic:p['generated_diagnostic']=diagnostic
    validate_protocol(p);payload['protocol.json']=enc(p)
    packed={name:base64.b85encode(lzma.compress(raw)).decode() for name,raw in payload.items()}
    install='import subprocess,sys\nsubprocess.run([sys.executable,"-m","pip","install","-q",'+','.join(repr(k+'=='+v) for k,v in p['packages'].items() if k!='torch')+'],check=True,timeout=600)\n'
    unpack='import base64,lzma,sys\nfrom pathlib import Path\npacked='+repr(packed)+'\nbundle=Path("/kaggle/working/ghost_candidate_audit_bundle")\nbundle.mkdir(exist_ok=True)\nfor name,data in packed.items():\n    (bundle/name).write_bytes(lzma.decompress(base64.b85decode(data)))\nsys.path.insert(0,str(bundle))\nfrom candidate_audit import run\nrun(bundle,"/kaggle/working/ghost_candidate_audit_results")\n'
    for code in (install,unpack):ast.parse(code)
    intro='# Separate27B neutral-source audit comparison\nTen exact archived9B source/question/candidate prompts. Non-thinking, greedy, at most10 evaluation calls plus one16-token memory probe. Expected decisions stay local. Two free T4s, no training/adapters/paid compute/active9B replacement. Install cap10minutes, workload cap45minutes. Fixed development diagnostic, not independent validation.\n'
    if diagnostic:intro=intro.replace('Ten exact archived9B source/question/candidate prompts.','Ten post-hoc source-checker candidates: two hearing failure pairs and three familiar controls. All have the same three-source context; no exact-control or fresh-validation claim.')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},'cells':[
        {'cell_type':'markdown','id':'scope','metadata':{},'source':[intro]},
        *[{'cell_type':'code','id':cid,'metadata':{},'source':code.splitlines(True),'execution_count':None,'outputs':[]} for cid,code in [('install',install),('run',unpack)]]]}
    stage=root/'training/kaggle_cli'/version;record=root/'evaluation'/version
    metadata=json.loads((root/'training/kaggle_cli'/PROBE/'kernel-metadata.json').read_text());metadata['code_file']='kaggle_ghost_candidate_audit.ipynb'
    if metadata['is_private'] is not True or metadata['id_no']!=140792344:raise ValueError('Wrong private target')
    notebook_raw=enc(notebook);metadata_raw=enc(metadata)
    lock={'notebook_sha256':sha(notebook_raw),'metadata_sha256':sha(metadata_raw),'payload_sha256':{name:sha(raw) for name,raw in payload.items()}}
    files={stage/metadata['code_file']:notebook_raw,stage/'kernel-metadata.json':metadata_raw,
        record/'bundle_lock.json':enc(lock),record/'protocol.json':payload['protocol.json'],record/'evaluation_spec.json':expected}
    for name,digest in p['tokenizer_files'].items():
        raw=(root/'evaluation'/PROBE/'tokenizer_assets'/name).read_bytes()
        if sha(raw)!=digest:raise ValueError('Verified candidate assets changed')
        files[record/'tokenizer_assets'/name]=raw
    for path,raw in files.items():
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Frozen candidate audit changed')
    for path,raw in files.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return stage


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',default='ghost_candidate_audit_v1');parser.add_argument('--generated-diagnostic',action='store_true')
    args=parser.parse_args();print(build(args.version,generated_diagnostic=args.generated_diagnostic))
