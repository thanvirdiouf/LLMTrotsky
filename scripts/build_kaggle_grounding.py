"""Package the frozen inference-only grounding comparison, without training data."""
import base64, hashlib, json, lzma
from pathlib import Path
from prepare_grounding_cases import verify, ROOT, OUT
from kaggle_grounding import validate_payloads

def sha(b):return hashlib.sha256(b).hexdigest()
def encoded(v):return (json.dumps(v,indent=2,ensure_ascii=False,sort_keys=True)+'\n').encode()

def build():
    p=verify()
    runtime={k:p[k] for k in ['model','packages','generation','enable_thinking','seed','max_input_tokens','chat_template_sha256','conditions','cases','answers','training_steps','control_identity','control_identity_sha256','control_run_manifest_sha256','checkpoint_manifest_sha256','adapter_sha256','adapter_config_sha256']}
    runtime['files']={'model_inputs.jsonl':p['files']['model_inputs.jsonl']}
    runtime['source_protocol_sha256']=sha((OUT/'protocol.json').read_bytes())
    payload={n:(ROOT/'scripts'/n).read_bytes() for n in ['kaggle_grounding.py','kbit_preparation.py','training_masks.py','pilot_checkpoints.py']}
    payload['model_inputs.jsonl']=(OUT/'model_inputs.jsonl').read_bytes()
    payload['control_run_manifest.json']=(ROOT/'evaluation/runs/kaggle_data_revision_v2_v1/results/run_manifest.json').read_bytes()
    payload['runtime_protocol.json']=encoded(runtime)
    validate_payloads(payload['model_inputs.jsonl'],runtime,json.loads(payload['control_run_manifest.json']))
    cells=[]
    def add(kind,name,text):
        c=dict(cell_type=kind,id=name,metadata={},source=text.splitlines(True))
        if kind=='code':
            c.update(execution_count=None,outputs=[])
            if name!='install':compile(text,name,'exec')
        cells.append(c)
    add('markdown','intro','# Frozen source-grounding comparison\nInference only: 24 fresh cases, 96 answers. Untouched base and saved revised1e-4 step50 adapter each with/without identical passages. No training, optimizer, new adapter or epoch. Both adapters remain experimental. Only Kaggle GPU0.\n')
    add('code','install','%pip install -q '+' '.join(f'{k}=={v}' for k,v in p['packages'].items() if k!='torch')+'\n')
    packed={n:dict(sha256=sha(b),lzma_base85=base64.b85encode(lzma.compress(b,preset=9)).decode()) for n,b in payload.items()}
    add('code','bundle',"import base64,hashlib,json,lzma,sys\nfrom pathlib import Path\nBUNDLE=Path('/kaggle/working/grounding_bundle')\nBUNDLE.mkdir(exist_ok=True)\nBUNDLED_FILES="+repr(packed)+"\nfor name,item in BUNDLED_FILES.items():\n    raw=lzma.decompress(base64.b85decode(item['lzma_base85']))\n    if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Bundle mismatch: '+name)\n    (BUNDLE/name).write_bytes(raw)\nsys.path.insert(0,str(BUNDLE))\nprint('Verified frozen input-only bundle',len(BUNDLED_FILES),'files')\n")
    add('code','infer',"from kaggle_grounding import run_comparison\nRUN_DIR=run_comparison(BUNDLE,Path('/kaggle/working/trotsky_grounding'))\n")
    add('code','inspect',"print((RUN_DIR/'run_manifest.json').read_text())\nprint((RUN_DIR/'evaluation_summary.json').read_text())\nprint('Delivery:',RUN_DIR/'comparison_delivery.zip')\n")
    nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},nbformat=4,nbformat_minor=5)
    raw=encoded(nb)
    if len(raw)>950000:raise ValueError('Notebook too large')
    stage=ROOT/'training/kaggle_cli/grounding_v1';stage.mkdir(parents=True,exist_ok=True)
    for path in [ROOT/'notebooks/kaggle_grounding_v1.ipynb',stage/'kaggle_grounding_v1.ipynb']:path.write_bytes(raw)
    metadata=json.loads((ROOT/'training/kaggle_cli/data_revision_v2/kernel-metadata.json').read_bytes())
    metadata.update(id='thanvirdiouf/trotsky-source-grounding-comparison',title='Trotsky source grounding comparison',code_file='kaggle_grounding_v1.ipynb',kernel_sources=[p['control_kernel_source']])
    (stage/'kernel-metadata.json').write_bytes(encoded(metadata))
    lock=dict(protocol_sha256=runtime['source_protocol_sha256'],notebook_sha256=sha(raw),metadata_sha256=sha(encoded(metadata)),payloads={n:sha(b) for n,b in payload.items()},training_steps=0,answers=96)
    lockpath=OUT/'bundle_lock.json'
    if lockpath.exists() and json.loads(lockpath.read_bytes())!=lock:raise ValueError('Frozen bundle differs; preserve original, do not silently overwrite')
    lockpath.write_bytes(encoded(lock))
    print(json.dumps(lock,indent=2))
    return payload,runtime

if __name__=='__main__':build()
