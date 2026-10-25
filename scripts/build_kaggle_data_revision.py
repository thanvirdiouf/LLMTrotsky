#!/usr/bin/env python3
"""Freeze token exposure/protocol and stage the authorized data-only Kaggle pilot."""
import argparse,base64,hashlib,json,textwrap,zlib
from collections import Counter
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
HELPERS=['build_training_pilot.py','training_masks.py','kbit_preparation.py','kaggle_training.py','pilot_checkpoints.py','kaggle_checkpoint_compare.py','data_revision_protocol.py','kaggle_data_revision_pilot.py']
ASSET_SHA={'chat_template.jinja':'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715','tokenizer.json':'5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42','tokenizer_config.json':'316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8'}

def digest(x):return hashlib.sha256(x).hexdigest()
def fingerprint(x):return digest(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n')

def verify_local_inputs(project):
    """Recheck local review and holdout evidence, beyond self-reported hashes."""
    if __package__:
        from .build_data_revision_v2 import validate_version
        from .freeze_fresh_evaluation_v2 import verify as verify_evaluation
    else:
        from build_data_revision_v2 import validate_version
        from freeze_fresh_evaluation_v2 import verify as verify_evaluation
    return validate_version(project),verify_evaluation(project)

def validate_frozen_payloads(project,protocol):
    if __package__:
        from .data_revision_protocol import validate_payloads
    else:
        from data_revision_protocol import validate_payloads
    project=Path(project);data=project/'datasets/revision_v2';ev=project/'evaluation/fresh_holdout_v2'
    verify_local_inputs(project)
    pilot_bytes=(project/'training/runs/kaggle_361474126/run_manifest.json').read_bytes()
    if digest(pilot_bytes)!=protocol['pilot_run_manifest_sha256']:raise ValueError('Frozen original pilot manifest changed')
    if (digest((data/'version_manifest.json').read_bytes())!=protocol['training']['version_manifest_sha256']
            or digest((ev/'protocol.json').read_bytes())!=protocol['evaluation']['source_protocol_sha256']
            or digest((project/'training/data_revision_v2/token_mask_preflight.json').read_bytes())!=protocol['exposure_report_sha256']):
        raise ValueError('Frozen data/evaluation/exposure snapshot changed')
    for relative,expected in protocol['local_authoring_sources'].items():
        if digest((project/relative).read_bytes())!=expected:raise ValueError('Frozen local authoring source changed: '+relative)
    validate_payloads((data/'train_slice.jsonl').read_bytes(),(ev/'model_inputs.jsonl').read_bytes(),json.loads((data/'training_lock.json').read_bytes()),protocol,json.loads(pilot_bytes))

def freeze(project=PROJECT):
    if __package__:
        from .training_masks import encode_assistant_example,inspect_tokens
        from .data_revision_protocol import build_order,validate_payloads
    else:
        from training_masks import encode_assistant_example,inspect_tokens
        from data_revision_protocol import build_order,validate_payloads
    project=Path(project);out=project/'training/data_revision_v2';data=project/'datasets/revision_v2';ev=project/'evaluation/fresh_holdout_v2'
    if (out/'protocol.json').exists():raise FileExistsError('Training protocol already frozen')
    verify_local_inputs(project)
    from transformers import AutoTokenizer
    version=json.loads((data/'version_manifest.json').read_bytes())
    for name,h in version['files'].items():
        if digest((data/name).read_bytes())!=h:raise ValueError('Versioned data changed: '+name)
    eval_protocol_bytes=(ev/'protocol.json').read_bytes();eval_protocol=json.loads(eval_protocol_bytes)
    for name,h in eval_protocol['files'].items():
        if digest((ev/name).read_bytes())!=h:raise ValueError('New evaluation changed: '+name)
    tokpath=project/'training/data_revision_v2/tokenizer_assets'
    for name,h in ASSET_SHA.items():
        if digest((tokpath/name).read_bytes())!=h:raise ValueError('Pinned tokenizer asset changed')
    tok=AutoTokenizer.from_pretrained(tokpath,local_files_only=True,trust_remote_code=False)
    training=rows(data/'train_slice.jsonl');full=rows(data/'train_all.jsonl')
    enc=[encode_assistant_example(tok,r['messages'],1024,r['example_id']) for r in training]
    full_enc=[encode_assistant_example(tok,r['messages'],1024,r['example_id']) for r in full]
    if any(e['dropped'] for e in enc+full_enc):raise ValueError('No revised example may be truncated or dropped')
    for r,e in zip(full,full_enc):
        expected=''.join(m['content']+'<|im_end|>' for m in r['messages'] if m['role']=='assistant')
        if tok.decode([x for x in e['labels'] if x!=-100],skip_special_tokens=False)!=expected:raise ValueError('Assistant masks/end tokens differ')
    order=build_order(enc,42);ids=[training[i]['example_id'] for i in order]
    oldmetrics=rows(project/'training/runs/kaggle_361474126/training_metrics.jsonl')[:50]
    exposure={'model_weights_loaded':False,'gpu_used':False,'tokenizer_assets':ASSET_SHA,'slice_records':50,'parent_records':526,'all_parent_masks_verified':True,'parent_token_range':[min(e['token_count'] for e in full_enc),max(e['token_count'] for e in full_enc)],'slice_token_range':[min(e['token_count'] for e in enc),max(e['token_count'] for e in enc)],'slice_supervised_tokens':sum(e['supervised_tokens'] for e in enc),'original_first50_supervised_tokens':sum(r['supervised_tokens'] for r in oldmetrics),'slice_assistant_answers':sum(len(e['assistant_spans']) for e in enc),'original_first50_examples':[r['example_id'] for r in oldmetrics],'ordered_examples':[{'update':n,'example_id':training[i]['example_id'],'input_tokens':enc[i]['token_count'],'supervised_tokens':enc[i]['supervised_tokens'],'assistant_answers':len(enc[i]['assistant_spans'])} for n,i in enumerate(order,1)],'limitation':'Update count is matched; source examples, target length, control mix and supervised token exposure change as part of the data intervention.'}
    exposure['supervised_token_ratio']=exposure['slice_supervised_tokens']/exposure['original_first50_supervised_tokens']
    write(out/'token_mask_preflight.json',exposure)
    write(out/'mask_inspection_longest.json',{'example_id':ids[0],'tokens':inspect_tokens(tok,enc[order[0]])})
    pilot=json.loads((project/'training/runs/kaggle_361474126/run_manifest.json').read_bytes());settings=dict(pilot['identity']['settings']);settings['steps']=50
    lock=json.loads((data/'training_lock.json').read_bytes())
    cp=next((project/'training/kaggle_cli/pilot_checkpoints').rglob('checkpoint-000050/checkpoint_manifest.json'))
    p={'schema_version':1,'purpose':'Data revision v2, fixed50update intervention; no parent-dataset full epoch','conditions':['fresh_base','original_step_50','revised_step_50'],'settings':settings,'packages':pilot['identity']['packages'],'pilot_run_manifest_sha256':digest((project/'training/runs/kaggle_361474126/run_manifest.json').read_bytes()),'pilot_identity':pilot['identity'],'pilot_identity_sha256':pilot['identity_sha256'],'chat_template_sha256':digest(tok.chat_template.encode()),'training':{'sha256':digest((data/'train_slice.jsonl').read_bytes()),'records':50,'parent_records':526,'version_manifest_sha256':digest((data/'version_manifest.json').read_bytes())},'training_lock_fingerprint':fingerprint(lock),'training_order':{'ordered_example_ids_sha256':fingerprint(ids),'example_ids':ids,'policy':'Longest first, seed42 shuffle of remaining49 exactly once.'},'evaluation':{'sha256':digest((ev/'model_inputs.jsonl').read_bytes()),'source_protocol_sha256':digest(eval_protocol_bytes),'records':24,'generation':eval_protocol['generation']},'original_checkpoint_manifest_sha256':digest(cp.read_bytes()),'exposure_report_sha256':digest((out/'token_mask_preflight.json').read_bytes()),'frozen_helpers':{n:digest((project/'scripts'/n).read_bytes()) for n in HELPERS},'quality_scoring':'References and checks stay local; no quality score generated by training. Preserve blank blinded materials before coding-agent assessment.'}
    p['local_authoring_sources']={relative:digest((project/relative).read_bytes()) for relative in ['scripts/build_data_revision_v2.py','scripts/build_kaggle_data_revision.py','scripts/freeze_fresh_evaluation_v2.py']}
    validate_payloads((data/'train_slice.jsonl').read_bytes(),(ev/'model_inputs.jsonl').read_bytes(),lock,p,pilot)
    write(out/'protocol.json',p)
    print(json.dumps({'frozen':True,'updates':50,'supervised_tokens':exposure['slice_supervised_tokens'],'old_supervised_tokens':exposure['original_first50_supervised_tokens'],'ratio':exposure['supervised_token_ratio'],'max_parent_tokens':max(e['token_count'] for e in full_enc)},indent=2))
    return p

def build(project=PROJECT):
    project=Path(project);out=project/'training/data_revision_v2';p=json.loads((out/'protocol.json').read_bytes())
    validate_frozen_payloads(project,p)
    for n,h in p['frozen_helpers'].items():
        if digest((project/'scripts'/n).read_bytes())!=h:raise ValueError('Frozen runtime helper changed: '+n)
    payload={n:(project/'scripts'/n).read_bytes() for n in HELPERS}
    payload.update({'train_slice.jsonl':(project/'datasets/revision_v2/train_slice.jsonl').read_bytes(),'training_lock.json':(project/'datasets/revision_v2/training_lock.json').read_bytes(),'protocol.json':(out/'protocol.json').read_bytes(),'model_inputs.jsonl':(project/'evaluation/fresh_holdout_v2/model_inputs.jsonl').read_bytes(),'pilot_run_manifest.json':(project/'training/runs/kaggle_361474126/run_manifest.json').read_bytes()})
    cells=[]
    def add(kind,id,source):
        c={'cell_type':kind,'id':id,'metadata':{},'source':textwrap.dedent(source).lstrip().splitlines(True)}
        if kind=='code':c.update(execution_count=None,outputs=[])
        cells.append(c)
    add('markdown','intro','''# Trotsky data revision v2: 50 updates
    Fresh adapter; exactly50 actual updates on the frozen revised slice. All optimizer/model/adapter settings retain the original pilot policy. This is one pass over a50-example diagnostic slice of the526-record revised parent, not another epoch over the original500. The input pilot version2 supplies the original step50 control. All training source/evidence reviews and evaluation reference checks remain local; only50 approved chat rows and24 model-input-only evaluation prompts are bundled.
    After verified training/checkpoint save/restore and fresh-base adapter reload, compare fresh_base, original_step_50 and revised_step_50 with identical frozen prompts and decoding. OnlyGPU0 is used. Preserve all saved outputs; mechanical success is not historical quality.
    ''')
    add('code','install','%pip install -q '+' '.join(f'{k}=={v}' for k,v in p['packages'].items() if k!='torch')+'\n')
    packed={n:{'sha256':digest(b),'zlib_base64':base64.b64encode(zlib.compress(b,9)).decode()} for n,b in payload.items()}
    add('code','bundle',"import base64,hashlib,json,sys,zlib\nfrom pathlib import Path\nBUNDLE=Path('/kaggle/working/data_revision_v2_bundle')\nBUNDLE.mkdir(parents=True,exist_ok=True)\nBUNDLED_FILES = "+repr(packed)+"\n"+textwrap.dedent('''
    for name,item in BUNDLED_FILES.items():
        b=zlib.decompress(base64.b64decode(item['zlib_base64']))
        if hashlib.sha256(b).hexdigest()!=item['sha256']:raise ValueError('Bundle mismatch: '+name)
        (BUNDLE/name).write_bytes(b)
    sys.path.insert(0,str(BUNDLE))
    from data_revision_protocol import validate_payloads
    protocol=json.loads((BUNDLE/'protocol.json').read_bytes())
    rows,eval_inputs=validate_payloads((BUNDLE/'train_slice.jsonl').read_bytes(),(BUNDLE/'model_inputs.jsonl').read_bytes(),json.loads((BUNDLE/'training_lock.json').read_bytes()),protocol,json.loads((BUNDLE/'pilot_run_manifest.json').read_bytes()))
    print('Frozen approved slice:',len(rows),'Fresh model inputs:',len(eval_inputs))
    print('Protocol SHA256:',BUNDLED_FILES['protocol.json']['sha256'])
    '''))
    add('code','train-compare',"from kaggle_data_revision_pilot import run_data_revision_pilot\nRUN_DIR=run_data_revision_pilot(BUNDLE,Path('/kaggle/working/trotsky_data_revision_v2'))\n")
    add('code','inspect',"print((RUN_DIR/'run_manifest.json').read_text())\nprint((RUN_DIR/'evaluation_summary.json').read_text())\nprint('Download:',RUN_DIR/'data_revision_delivery.zip')\n")
    for cell in cells:
        if cell['cell_type']=='code' and cell['id']!='install':
            compile(''.join(cell['source']),cell['id'],'exec')
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    code=json.dumps(nb,ensure_ascii=False,indent=1)+'\n'
    (project/'notebooks/kaggle_data_revision_v2.ipynb').write_text(code)
    stage=project/'training/kaggle_cli/data_revision_v2';stage.mkdir(parents=True,exist_ok=True)
    (stage/'kaggle_data_revision_v2.ipynb').write_text(code)
    meta={'id':'thanvirdiouf/trotsky-data-revision-v2-50-updates','title':'Trotsky data revision v2 50 updates','code_file':'kaggle_data_revision_v2.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','docker_image':'gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435','docker_image_pinning_type':'original','kernel_sources':['thanvirdiouf/trotsky-qlora-500-example-pilot/2'],'dataset_sources':[],'model_sources':[],'competition_sources':[]}
    write(stage/'kernel-metadata.json',meta)
    print(stage)
    return nb

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--project',type=Path,default=PROJECT);a.add_argument('--freeze',action='store_true');args=a.parse_args()
    if args.freeze:freeze(args.project)
    build(args.project)
