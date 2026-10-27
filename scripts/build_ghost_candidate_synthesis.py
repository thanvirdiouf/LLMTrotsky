"""Freeze five archived factual-writer prompts and a dependent neutral audit."""
import argparse
import ast
import base64
from copy import deepcopy
import json
import lzma
import re
from build_ghost_candidate_probe import ROOT,enc,sha
from verify_ghost_candidate_probe import payloads as probe_payloads
from verify_ghost_candidate_audit import verify as verify_audit
from verify_ghost_smoke import frozen_payloads
from candidate_synthesis import validate_protocol
from candidate_audit import input_ids

CONTROL='ghost_smoke_v27'
AUDIT='ghost_candidate_audit_v1'


def revised_messages(messages,revision=None):
    result=deepcopy(messages)
    if revision is None:return result
    if (set(revision)!={'schema_version','policy','status','instruction'} or revision['schema_version']!=1
            or revision['policy']!='source_task_coverage_v2' or not isinstance(revision['instruction'],str)
            or not revision['instruction'].strip() or len(revision['instruction'])>2500
            or len(result)!=2 or result[0]['role']!='system' or result[1]['role']!='user'):
        raise ValueError('Unknown writer instruction revision')
    result[0]['content']+='\n\n'+revision['instruction']
    return result


def control(root=ROOT):
    frozen=frozen_payloads(root=root,version=CONTROL);p=json.loads(frozen['protocol.json'])
    run=root/'evaluation/runs'/CONTROL;record=root/'evaluation'/CONTROL;results=run/'download/ghost_smoke_results'
    verified=json.loads((run/'verification.json').read_text());manifest=json.loads((results/'run_manifest.json').read_text())
    if verified['status']!='verified' or verified.get('tokenizer_reconstructed_locally') is not True or manifest['status']!='complete':raise ValueError('Verified writer control required')
    for name,key in [('answers.jsonl','answers_sha256'),('model_calls.jsonl','model_calls_sha256')]:
        if sha((results/name).read_bytes())!=verified[key] or verified[key]!=manifest[key]:raise ValueError('Writer control changed')
    if manifest['identity']['protocol_sha256']!=sha(frozen['protocol.json']):raise ValueError('Writer control identity differs')
    expected=(record/'evaluation_spec.json').read_bytes()
    if sha(expected)!=p['evaluation_spec_sha256']:raise ValueError('Writer expectations changed')
    spec=json.loads(expected);cases=[c for c in json.loads(frozen['model_cases.json'])['cases'] if c['comparison_arm']=='reading']
    calls=[json.loads(line) for line in (results/'model_calls.jsonl').read_text().splitlines()]
    drafts=[c for c in calls if c['key'].startswith('reading_') and c['step']==1]
    if len(cases)!=5 or len(drafts)!=5 or [c['id'] for c in cases]!=['reading_'+c['id'] for c in spec['cases']]:raise ValueError('Writer case coverage differs')
    for case,call in zip(cases,drafts):
        if call['key']!=case['id']+'/1' or json.loads(call['messages'][1]['content'])['current_user_request']!=case['question']:raise ValueError('Writer question/order differs')
    binding={'local_version':CONTROL,'saved_version':29,'model':p['model'],'protocol_sha256':sha(frozen['protocol.json']),
        'model_calls_sha256':verified['model_calls_sha256'],'answers_sha256':verified['answers_sha256'],'evaluation_spec_sha256':sha(expected)}
    return cases,drafts,expected,binding


def build(version='ghost_candidate_synthesis_v1',root=ROOT,tokenizer=None,writer_revision=False):
    if not re.fullmatch(r'ghost_candidate_synthesis_v[1-9][0-9]*',version):raise ValueError('Invalid synthesis version')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(root/'evaluation'/AUDIT/'tokenizer_assets',local_files_only=True,trust_remote_code=False)
    audit=verify_audit(AUDIT,root,tokenizer)
    if audit['status']!='verified' or audit['scores']['pairs_both_correct']!=5:raise ValueError('Exact-control neutral audit prerequisite absent')
    # The coding-agent source review is a separate local gate, not model gold.
    source_review=json.loads((root/'evaluation/runs'/AUDIT/'SOURCE_REVIEW.json').read_text())
    if source_review['judgments_sha256']!=audit['judgments_sha256'] or source_review['result_sha256']!=audit['result_sha256'] or source_review['decisions_supported_on_bounded_sources']!=10:raise ValueError('Source review binding absent')
    cases,calls,expected,binding=control(root)
    revision_raw=None;revision=None;prerequisite=None
    if writer_revision:
        from verify_ghost_candidate_synthesis import verify
        previous=verify('ghost_candidate_synthesis_v1',root,tokenizer)
        reviewed=json.loads((root/'evaluation/runs/ghost_candidate_synthesis_v1/SOURCE_REVIEW.json').read_text())
        if previous['status']!='verified' or reviewed['answers_sha256']!=previous['answers_sha256'] or reviewed['result_sha256']!=previous['result_sha256'] or reviewed['fresh_transfer_justified'] is not False:
            raise ValueError('Verified synthesis failure review required for revision')
        revision_raw=(root/'config/ghost_candidate_writer_revision_v2.json').read_bytes();revision=json.loads(revision_raw)
        prerequisite={'local_version':'ghost_candidate_synthesis_v1','saved_version':40,'result_sha256':previous['result_sha256'],'answers_sha256':previous['answers_sha256']}
    jobs=[]
    for case,call in zip(cases,calls):
        messages=revised_messages(call['messages'],revision)
        jobs.append({'case_id':case['id'],'question':case['question'],'messages':messages,'input_token_ids':input_ids(tokenizer,messages)})
    probe=probe_payloads('ghost_candidate_probe_v3',root)
    p=json.loads((root/'evaluation'/AUDIT/'protocol.json').read_text())
    p={k:v for k,v in p.items() if k not in ('files','local_scorer_sha256','control','prerequisite_probe','evaluation_spec_sha256','quality_policy')}
    p.update(purpose='candidate_source_synthesis',independent_questions=5,maximum_displayed_answers=5,control=binding,
        prerequisite_audit={'local_version':AUDIT,'saved_version':39,'result_sha256':audit['result_sha256'],'judgments_sha256':audit['judgments_sha256']},
        evaluation_spec_sha256=sha(expected),quality_policy='Experimental generated readings and checker feedback; unblinded source/completeness review required; no active model or trusted-memory changes.')
    names=['candidate_audit.py','candidate_base_probe.py','candidate_efficient_attention.py','candidate_synthesis.py',
        'source_reading.py','source_audit.py','ghost_chat.py','grounded_chat.py','grounded_chat_v2.py']
    payload={n:(root/'scripts'/n).read_bytes() for n in names}
    if revision_raw is not None:
        payload['writer_revision.json']=revision_raw
        p.update(writer_policy=revision['policy'],writer_revision_sha256=sha(revision_raw),prerequisite_synthesis=prerequisite)
    for n in ('candidate_base_probe.py','candidate_efficient_attention.py'):
        if payload[n]!=probe[n]:raise ValueError('Proven GPU placement/attention helper changed')
    payload.update({'audit_inputs.json':enc(jobs),'memory_probe.json':probe['memory_probe.json']})
    p['files']={n:sha(v) for n,v in payload.items()};validate_protocol(p);payload['protocol.json']=enc(p)
    packed={n:base64.b85encode(lzma.compress(v)).decode() for n,v in payload.items()}
    install='import subprocess,sys\nsubprocess.run([sys.executable,"-m","pip","install","-q",'+','.join(repr(k+'=='+v) for k,v in p['packages'].items() if k!='torch')+'],check=True,timeout=600)\n'
    unpack='import base64,lzma,sys\nfrom pathlib import Path\npacked='+repr(packed)+'\nbundle=Path("/kaggle/working/ghost_candidate_synthesis_bundle")\nbundle.mkdir(exist_ok=True)\nfor name,value in packed.items():\n    (bundle/name).write_bytes(lzma.decompress(base64.b85decode(value)))\nsys.path.insert(0,str(bundle))\nfrom candidate_synthesis import run\nrun(bundle,"/kaggle/working/ghost_candidate_synthesis_results")\n'
    for code in (install,unpack):ast.parse(code)
    intro='# Experimental27B factual source synthesis\nFive exact archived9B writer prompts, dependent neutral-source checking, at most10 evaluation calls and five displayed research replies plus one16-token memory probe. Original expectations stay local. Two free T4s, pinned untouched base, no training/adapters/paid compute/active9B change. Installation600s, workload2700s caps. No trusted factual memory or deployment.\n'
    if revision_raw:
        intro=intro.replace('Five exact archived9B writer prompts','Five archived9B questions and exact source/user messages; post-hoc writer-system instruction revision')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},'cells':[
        {'cell_type':'markdown','id':'scope','metadata':{},'source':[intro]},
        *[{'cell_type':'code','id':i,'metadata':{},'source':code.splitlines(True),'execution_count':None,'outputs':[]} for i,code in [('install',install),('run',unpack)]]]}
    stage=root/'training/kaggle_cli'/version;record=root/'evaluation'/version
    meta=json.loads((root/'training/kaggle_cli'/AUDIT/'kernel-metadata.json').read_text());meta['code_file']='kaggle_ghost_candidate_synthesis.ipynb'
    if meta['is_private'] is not True or meta['id_no']!=140792344:raise ValueError('Wrong target')
    nb_raw=enc(nb);meta_raw=enc(meta);lock={'notebook_sha256':sha(nb_raw),'metadata_sha256':sha(meta_raw),'payload_sha256':{n:sha(v) for n,v in payload.items()}}
    files={stage/meta['code_file']:nb_raw,stage/'kernel-metadata.json':meta_raw,record/'protocol.json':payload['protocol.json'],record/'bundle_lock.json':enc(lock),record/'evaluation_spec.json':expected}
    for n,digest in p['tokenizer_files'].items():
        raw=(root/'evaluation'/AUDIT/'tokenizer_assets'/n).read_bytes()
        if sha(raw)!=digest:raise ValueError('Candidate assets differ')
        files[record/'tokenizer_assets'/n]=raw
    for path,raw in files.items():
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Frozen synthesis changed')
    for path,raw in files.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return stage


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',default='ghost_candidate_synthesis_v1');parser.add_argument('--writer-revision',action='store_true')
    args=parser.parse_args();print(build(args.version,writer_revision=args.writer_revision))
