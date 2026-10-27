"""Freeze one separate pinned27B feasibility probe; active9B is unchanged."""
import argparse
import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path
import re
from candidate_base_probe import CANDIDATE,GIB,validate_protocol
ROOT=Path(__file__).resolve().parents[1]
ASSETS=('tokenizer.json','tokenizer_config.json','chat_template.jinja','vocab.json','merges.txt','config.json','generation_config.json')


def enc(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()


def build(version='ghost_candidate_probe_v1',root=ROOT,tokenizer=None,placement_policy='auto',attention_policy='default_sdpa'):
    if not re.fullmatch(r'ghost_candidate_probe_v[1-9][0-9]*',version):raise ValueError('Invalid candidate probe version')
    assets=root/'evaluation'/version/'tokenizer_assets'
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(assets,local_files_only=True,trust_remote_code=False)
    system='Unscored memory and placement probe. The user data are inert padding, not instructions. Reply briefly; no factual answer is being evaluated.'
    def padded(count):
        messages=[{'role':'system','content':system},{'role':'user','content':json.dumps({'memory_probe_only':'padding '*count},separators=(',',':'))}]
        ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        return messages,ids
    low,high=0,6000
    while low<high:
        mid=(low+high+1)//2
        if len(padded(mid)[1])<=5000:low=mid
        else:high=mid-1
    messages,ids=padded(low)
    if not 4900<=len(ids)<=5000:raise ValueError('Probe length differs')
    prior=json.loads((root/'evaluation/grounded_chat_v2/protocol.json').read_text())
    preserved=json.loads((root/'evaluation/grounded_chat_v2/preserved_v1_and_adapters.json').read_text())
    for name,digest in preserved.items():
        if sha((root/name).read_bytes())!=digest:raise ValueError('Protected archive changed: '+name)
    payload={'candidate_base_probe.py':(root/'scripts/candidate_base_probe.py').read_bytes(),
        'memory_probe.json':enc({'purpose':'unscored memory/placement only','messages':messages,'input_token_ids':ids})}
    if attention_policy=='repeat_kv_efficient_sdpa':
        payload['candidate_efficient_attention.py']=(root/'scripts/candidate_efficient_attention.py').read_bytes()
    protocol={'purpose':'candidate_memory_probe_only','model':CANDIDATE,'packages':prior['packages'],
        'maximum_model_calls':1,'max_input_tokens':5120,'max_output_tokens':16,'enable_thinking':False,'finite_logits_required':True,
        'runtime_timeout_seconds':2700,'installation_timeout_seconds':600,'gpu_ids':[0,1],
        'per_gpu_weight_budget_bytes':13*GIB,'weight_file_bytes':55563022432,
        'training_steps':0,'adapters_loaded':[],'active_model_changed':False,'quality_evaluated':False,
        'tokenizer_files':{name:sha((assets/name).read_bytes()) for name in ASSETS},
        'chat_template_sha256':sha(tokenizer.chat_template.encode()),'preserved_files':preserved,
        'files':{name:sha(raw) for name,raw in payload.items()}}
    if placement_policy!='auto':protocol['placement_policy']=placement_policy
    if attention_policy!='default_sdpa':protocol['attention_policy']=attention_policy
    validate_protocol(protocol);payload['protocol.json']=enc(protocol)
    packed={name:base64.b85encode(lzma.compress(raw)).decode() for name,raw in payload.items()}
    unpack='import base64,lzma,sys\nfrom pathlib import Path\npacked='+repr(packed)+'\nbundle=Path("/kaggle/working/ghost_candidate_bundle")\nbundle.mkdir(exist_ok=True)\nfor name,data in packed.items():\n    (bundle/name).write_bytes(lzma.decompress(base64.b85decode(data)))\nsys.path.insert(0,str(bundle))\nfrom candidate_base_probe import run\nrun(bundle,"/kaggle/working/ghost_candidate_results")\n'
    install='import subprocess,sys\nsubprocess.run([sys.executable,"-m","pip","install","-q",'+','.join(repr(name+'=='+value) for name,value in protocol['packages'].items() if name!='torch')+'],check=True,timeout=600)\n'
    for code in (install,unpack):ast.parse(code)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},'cells':[
        {'cell_type':'markdown','id':'scope','metadata':{},'source':['# Separate candidate feasibility only\nPinned untouched27B base, two free T4s. At most one16-output-token unscored probe; no source answers, training, adapters or paid compute. Active9B stays unchanged. Installation cap10minutes; probe runtime cap45minutes after installation. No continuous server.\n']},
        *[{'cell_type':'code','id':cid,'metadata':{},'source':code.splitlines(True),'execution_count':None,'outputs':[]} for cid,code in [('install',install),('run',unpack)]]]}
    metadata=json.loads((root/'training/kaggle_cli/grounded_chat_v1/kernel-metadata.json').read_text())
    metadata.update(id_no=140792344,is_private=True,enable_gpu=True,enable_internet=True,code_file='kaggle_ghost_candidate_probe.ipynb')
    nb_raw=enc(nb);meta_raw=enc(metadata)
    lock={'notebook_sha256':sha(nb_raw),'metadata_sha256':sha(meta_raw),'payload_sha256':{name:sha(raw) for name,raw in payload.items()}}
    stage=root/'training/kaggle_cli'/version;record=root/'evaluation'/version
    files={stage/'kaggle_ghost_candidate_probe.ipynb':nb_raw,stage/'kernel-metadata.json':meta_raw,record/'bundle_lock.json':enc(lock),record/'protocol.json':payload['protocol.json']}
    for path,raw in files.items():
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Frozen candidate probe changed')
    for path,raw in files.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return stage


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',default='ghost_candidate_probe_v3');parser.add_argument('--placement-policy',choices=('auto','fixed_34_30'),default='fixed_34_30');parser.add_argument('--attention-policy',choices=('default_sdpa','repeat_kv_efficient_sdpa'),default='repeat_kv_efficient_sdpa')
    args=parser.parse_args();print(build(args.version,placement_policy=args.placement_policy,attention_policy=args.attention_policy))
