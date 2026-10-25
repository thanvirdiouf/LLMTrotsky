#!/usr/bin/env python3
"""Freeze and package the user-authorized LR-only 50-update test."""
import argparse,base64,hashlib,json,lzma,textwrap
from pathlib import Path
from lr_test_protocol import validate_payloads, fingerprint
from freeze_fresh_evaluation_v3 import verify as verify_fresh
ROOT=Path(__file__).resolve().parents[1]
HELPERS=['build_training_pilot.py','training_masks.py','kbit_preparation.py','kaggle_training.py','pilot_checkpoints.py','kaggle_checkpoint_compare.py','lr_test_protocol.py','kaggle_lr_test.py']
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
def frozen_inputs():
 verify_fresh()
 olddir=ROOT/'evaluation/runs/kaggle_data_revision_v2_v1';control=olddir/'results'
 old=json.loads((ROOT/'training/data_revision_v2/protocol.json').read_bytes())
 p=dict(old);p['purpose']='controlled_lr_1e5_50_updates';p['settings']={**old['settings'],'learning_rate':1e-5}
 p['conditions']=['fresh_base','revised_lr1e4_step50','revised_lr1e5_step50']
 p['control_identity']=json.loads((control/'run_manifest.json').read_bytes())['identity']
 p['control_identity_sha256']=fingerprint(p['control_identity'])
 p['control_run_manifest_sha256']=sha((control/'run_manifest.json').read_bytes())
 cp=list((olddir/'download').rglob('checkpoint-000050/checkpoint_manifest.json'))
 if len(cp)!=1:raise ValueError('Missing/ambiguous control checkpoint')
 p['original_checkpoint_manifest_sha256']=sha(cp[0].read_bytes())
 p['control_adapter_sha256']=sha((control/'final_adapter/adapter_model.safetensors').read_bytes())
 p['control_initial_adapter_sha256']=json.loads(next((olddir/'download').rglob('checkpoint-000000/checkpoint_manifest.json')).read_bytes())['files']['adapter/adapter_model.safetensors']
 ev=(ROOT/'evaluation/fresh_holdout_v3/model_inputs.jsonl').read_bytes()+(ROOT/'evaluation/fresh_holdout_v2/model_inputs.jsonl').read_bytes()
 p['evaluation']={**old['evaluation'],'sha256':sha(ev),'source_protocol_sha256':sha((ROOT/'evaluation/fresh_holdout_v3/protocol.json').read_bytes()),'records':48,'fresh_records':24,'regression_records':24,'regression_protocol_sha256':sha((ROOT/'evaluation/fresh_holdout_v2/protocol.json').read_bytes())}
 p['frozen_helpers']={n:sha((ROOT/'scripts'/n).read_bytes()) for n in HELPERS}
 p['local_authoring_sources']={name:sha((ROOT/name).read_bytes()) for name in ['scripts/freeze_fresh_evaluation_v3.py','scripts/build_kaggle_lr_test.py','evaluation/fresh_holdout_v3/source_review.md']}
 p['parent_control_protocol_sha256']=sha((ROOT/'training/data_revision_v2/protocol.json').read_bytes())
 p['authorization']='User authorized controlled lower-learning-rate test on 25 October 2026: same 50 revised examples/order/updates, fresh adapter; no full epoch over parent data or adapter promotion.'
 p['hypothesis']='Smaller updates may preserve useful knowledge while retaining calibration gains; not an established fix.'
 train=(ROOT/'datasets/revision_v2/train_slice.jsonl').read_bytes();lock=(ROOT/'datasets/revision_v2/training_lock.json').read_bytes();pilot=(ROOT/'training/runs/kaggle_361474126/run_manifest.json').read_bytes()
 validate_payloads(train,ev,json.loads(lock),p,json.loads(pilot))
 payload={n:(ROOT/'scripts'/n).read_bytes() for n in HELPERS}
 payload.update({'train_slice.jsonl':train,'training_lock.json':lock,'model_inputs.jsonl':ev,'pilot_run_manifest.json':pilot,'control_run_manifest.json':(control/'run_manifest.json').read_bytes()})
 return p,payload

def build(freeze=False):
 out=ROOT/'training/lr_1e5';p,payload=frozen_inputs()
 if freeze:
  if (out/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
  write(out/'protocol.json',p);(out/'model_inputs.jsonl').write_bytes(payload['model_inputs.jsonl'])
 else:
  if json.loads((out/'protocol.json').read_bytes())!=p:raise ValueError('Frozen inputs/helpers changed')
 payload['protocol.json']=(out/'protocol.json').read_bytes()
 cells=[]
 def add(kind,id,s):
  cell=dict(cell_type=kind,id=id,metadata={},source=textwrap.dedent(s).lstrip().splitlines(True))
  if kind=='code':cell.update(execution_count=None,outputs=[])
  cells.append(cell)
 add('markdown','intro','# Controlled lower-learning-rate test\nFresh adapter on exactly the same revised 50 examples, order and optimizer-update budget. Only learning rate changes from 1e-4 to 1e-5. Compare untouched base, saved revised 1e-4 adapter and new 1e-5 adapter. The 24 v3 cases are fresh primary evidence; 24 inspected v2 cases are development regressions. References stay local. Both adapters remain experimental.\n')
 add('code','install','%pip install -q '+' '.join(f'{k}=={v}' for k,v in p['packages'].items() if k!='torch')+'\n')
 packed={n:dict(sha256=sha(b),lzma_base85=base64.b85encode(lzma.compress(b,preset=9)).decode()) for n,b in payload.items()}
 add('code','bundle',"import base64,hashlib,json,sys,lzma\nfrom pathlib import Path\nBUNDLE=Path('/kaggle/working/lr_test_bundle')\nBUNDLE.mkdir(parents=True,exist_ok=True)\nBUNDLED_FILES="+repr(packed)+"\n"+textwrap.dedent('''
 for name,item in BUNDLED_FILES.items():
     raw=lzma.decompress(base64.b85decode(item['lzma_base85']))
     if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Bundle mismatch: '+name)
     (BUNDLE/name).write_bytes(raw)
 sys.path.insert(0,str(BUNDLE))
 from lr_test_protocol import validate_payloads
 protocol=json.loads((BUNDLE/'protocol.json').read_bytes())
 rows,inputs=validate_payloads((BUNDLE/'train_slice.jsonl').read_bytes(),(BUNDLE/'model_inputs.jsonl').read_bytes(),json.loads((BUNDLE/'training_lock.json').read_bytes()),protocol,json.loads((BUNDLE/'pilot_run_manifest.json').read_bytes()))
 print('Approved training rows:',len(rows),'Fresh + regression cases:',len(inputs),'Learning rate:',protocol['settings']['learning_rate'])
 '''))
 add('code','train-compare',"from kaggle_lr_test import run_lr_test\nRUN_DIR=run_lr_test(BUNDLE,Path('/kaggle/working/trotsky_lr_1e5'))\n")
 add('code','inspect',"print((RUN_DIR/'run_manifest.json').read_text())\nprint((RUN_DIR/'evaluation_summary.json').read_text())\nprint('Download:',RUN_DIR/'lr_test_delivery.zip')\n")
 for c in cells:
  if c['cell_type']=='code' and c['id']!='install':compile(''.join(c['source']),c['id'],'exec')
 nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},nbformat=4,nbformat_minor=5)
 encoded=(json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode()
 normal=json.loads(encoded)
 for c in normal['cells']:c['source']=''.join(c['source'])
 if len(json.dumps(normal).encode())>=1_000_000:raise ValueError('Kaggle source limit exceeded')
 stage=ROOT/'training/kaggle_cli/lr_1e5';stage.mkdir(exist_ok=True)
 for path in [ROOT/'notebooks/kaggle_lr_1e5.ipynb',stage/'kaggle_lr_1e5.ipynb']:path.write_bytes(encoded)
 meta=json.loads((ROOT/'training/kaggle_cli/data_revision_v2/kernel-metadata.json').read_bytes())
 meta.update(id='thanvirdiouf/trotsky-revised-lr1e5-50-updates',title='Trotsky revised lr1e5 50 updates',code_file='kaggle_lr_1e5.ipynb',kernel_sources=['thanvirdiouf/trotsky-data-revision-v2-50-updates/1'])
 write(stage/'kernel-metadata.json',meta)
 report=dict(protocol_sha256=sha(payload['protocol.json']),notebook_sha256=sha(encoded),notebook_bytes=len(encoded),payloads={n:sha(b) for n,b in payload.items()},private=True,source_version=meta['kernel_sources'],learning_rate_only_changed=True,supervised_tokens=4407,fresh_cases=24,regression_cases=24)
 write(out/'bundle_verification.json',report)
 print(json.dumps(report,indent=2));return nb
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--freeze',action='store_true');args=a.parse_args();build(args.freeze)
