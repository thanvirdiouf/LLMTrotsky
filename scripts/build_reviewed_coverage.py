"""Freeze a separate question-part selector comparison; source records unchanged."""
import argparse
import ast
import base64
import json
import lzma
import re
from pathlib import Path
from build_ghost_candidate_probe import ROOT,enc,sha
from candidate_base_probe import CANDIDATE
from candidate_reviewed_coverage import validate_protocol
from prepare_reviewed_coverage import frozen_inputs
from grounded_chat import fingerprint


def build(version='ghost_reviewed_coverage_v1',root=ROOT):
    if not re.fullmatch(r'ghost_reviewed_coverage_v[1-9][0-9]*',version):raise ValueError('Invalid reviewed selection version')
    inputs,metadata,binding=frozen_inputs(root)
    old=json.loads((root/'evaluation/ghost_candidate_audit_v2/protocol.json').read_text())
    p={k:old[k] for k in ('model','packages','generation','gpu_ids','placement_policy','attention_policy','tokenizer_files','chat_template_sha256',
        'per_gpu_weight_budget_bytes','weight_file_bytes','preserved_files','runtime_timeout_seconds','installation_timeout_seconds')}
    p.update(purpose='candidate_reviewed_coverage',evaluation_calls=5,memory_probe_calls=1,independent_questions=5,maximum_displayed_answers=5,
        part_presence_is_semantic_coverage=False,maximum_selected_records=12,max_input_tokens=5120,max_output_tokens=1024,memory_probe_output_tokens=16,
        enable_thinking=False,training_steps=0,adapters_loaded=[],active_model_changed=False,generated_factual_text_used=False,model_audit_performed=False,
        frozen_inputs=binding,ledger_fingerprint=metadata['source_ledger_fingerprint'],evaluation_spec_sha256=metadata['evaluation_spec_sha256'],
        quality_policy='Reviewed wording is immutable; model selection relevance and displayed task completeness are unverified. No trusted memory or automatic activation.')
    names=('candidate_audit.py','candidate_base_probe.py','candidate_efficient_attention.py','candidate_reviewed_coverage.py',
        'reviewed_source_display.py','grounded_chat.py','grounded_chat_v2.py')
    payload={n:(root/'scripts'/n).read_bytes() for n in names}
    payload.update({n:inputs[n] for n in ('profile.json','collection.json','ledger.json','memory_probe.json')})
    from verify_reviewed_selection import payloads as parent_payloads
    parent = parent_payloads('ghost_reviewed_selection_v1', root)
    for name, raw in payload.items():
        if name != 'candidate_reviewed_coverage.py' and raw != parent[name]:
            raise ValueError('Shared runtime/source control changed: '+name)
    payload['audit_inputs.json']=inputs['selection_requests.json'];p['files']={n:sha(raw) for n,raw in payload.items()}
    validate_protocol(p);payload['protocol.json']=enc(p)
    packed={n:base64.b85encode(lzma.compress(raw)).decode() for n,raw in payload.items()}
    install='import subprocess,sys\nsubprocess.run([sys.executable,"-m","pip","install","-q",'+','.join(repr(k+'=='+v) for k,v in p['packages'].items() if k!='torch')+'],check=True,timeout=600)\n'
    unpack='import base64,lzma,sys\nfrom pathlib import Path\npacked='+repr(packed)+'\nbundle=Path("/kaggle/working/ghost_reviewed_coverage_bundle")\nbundle.mkdir(exist_ok=True)\nfor name,value in packed.items():\n    (bundle/name).write_bytes(lzma.decompress(base64.b85decode(value)))\nsys.path.insert(0,str(bundle))\nfrom candidate_reviewed_coverage import run\nrun(bundle,"/kaggle/working/ghost_reviewed_coverage_results")\n'
    for code in (install,unpack):ast.parse(code)
    intro='# Experimental reviewed-source selector\nFive original controlled-source questions; model chooses IDs separately for literal question parts. Part-array presence is not semantic coverage. Application preserves reviewed wording/provenance with no factual rewriting or self-audit gate. At most5 evaluation calls/five research replies plus one16-token memory probe. Expected answers/oracle selections are not supplied. Pinned27B, two free T4s, private notebook, no training/adapters/paid compute/active9B change. Setup600s/workload2700s caps.\n'
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},'cells':[
        {'cell_type':'markdown','id':'scope','metadata':{},'source':[intro]},
        *[{'cell_type':'code','id':i,'metadata':{},'source':code.splitlines(True),'execution_count':None,'outputs':[]} for i,code in [('install',install),('run',unpack)]]]}
    meta=json.loads((root/'training/kaggle_cli/ghost_candidate_audit_v2/kernel-metadata.json').read_text());meta['code_file']='kaggle_reviewed_coverage.ipynb'
    if meta['is_private'] is not True or meta['id_no']!=140792344:raise ValueError('Wrong private target')
    raw=enc(nb);meta_raw=enc(meta);lock={'notebook_sha256':sha(raw),'metadata_sha256':sha(meta_raw),'payload_sha256':{n:sha(v) for n,v in payload.items()}}
    stage=root/'training/kaggle_cli'/version;record=root/'evaluation'/version
    files={stage/meta['code_file']:raw,stage/'kernel-metadata.json':meta_raw,record/'protocol.json':payload['protocol.json'],record/'bundle_lock.json':enc(lock)}
    for n,raw in files.items():
        if n.exists() and n.read_bytes()!=raw:raise ValueError('Frozen reviewed selector changed')
    for n,raw in files.items():n.parent.mkdir(parents=True,exist_ok=True);n.write_bytes(raw)
    return stage


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--version',default='ghost_reviewed_coverage_v1');print(build(p.parse_args().version))
