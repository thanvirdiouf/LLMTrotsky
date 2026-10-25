#!/usr/bin/env python3
"""Freeze a separate v2 comparison; preserve every v1 artifact and adapter."""
import base64
import hashlib
import json
import lzma
from pathlib import Path
from grounded_chat import ROOT, MODEL, load_collection
from grounded_chat_v2 import validate_ledger
from evaluate_grounded_chat_v2 import model_spec


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build():
    out = ROOT/'evaluation/grounded_chat_v2'; out.mkdir(parents=True, exist_ok=True)
    preserved = json.loads((out/'preserved_v1_and_adapters.json').read_text())
    if any(sha((ROOT/name).read_bytes()) != value for name, value in preserved.items()):
        raise ValueError('V1 artifacts or saved adapters changed')
    collection = load_collection(ROOT/'evaluation/grounded_chat_v1/collection.json')
    ledger = validate_ledger(collection, json.loads((ROOT/'config/grounded_chat_claims_v2.json').read_text()))
    spec = json.loads((ROOT/'config/grounded_chat_evaluation_v2.json').read_text())
    old_spec = json.loads((ROOT/'config/grounded_chat_evaluation.json').read_text())
    def questions(s):
        return [r['question'] for r in s['singles']]+[t['question'] for c in s['conversations'] for t in c['turns']]
    if set(questions(spec)) & set(questions(old_spec)) or len(questions(spec)) != len(set(questions(spec))):
        raise ValueError('Duplicate/previously used v2 prompt')
    names = ['grounded_chat.py', 'grounded_chat_v2.py', 'evaluate_grounded_chat_v2.py',
             'kaggle_grounded_chat_v2.py', 'kaggle_grounding.py', 'kbit_preparation.py']
    payload = {name: (ROOT/'scripts'/name).read_bytes() for name in names}
    payload.update({'collection.json': encoded(collection), 'claims.json': encoded(ledger), 'model_cases.json': encoded(model_spec(spec))})
    prior = json.loads((ROOT/'evaluation/grounded_chat_v1/protocol.json').read_text())
    protocol = {k: prior[k] for k in ('model', 'packages', 'generation', 'max_input_tokens', 'chat_template_sha256')}
    protocol['generation']['max_new_tokens'] = 1024
    protocol.update(schema_version=2, purpose='grounded_chat_v2_matched_source_comparison',
        conditions=spec['conditions'], answers=2*len(questions(spec)), maximum_model_calls=4*len(questions(spec)),
        training_steps=0, adapters_loaded=[], files={n: sha(b) for n, b in payload.items()},
        evaluation_spec_sha256=sha(encoded(spec)), preserved_files=preserved,
        comparison=spec['comparison'], freshness=spec['freshness'])
    payload['protocol.json'] = encoded(protocol)
    local = {'protocol.json': encoded(protocol), 'evaluation_spec.json': encoded(spec)}
    cells = []
    def cell(kind, cid, source):
        item = {'cell_type': kind, 'id': cid, 'metadata': {}, 'source': source.splitlines(True)}
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
            if cid != 'install':
                compile(source, cid, 'exec')
        cells.append(item)
    cell('markdown', 'intro', '# Grounded chatbot v2: matched-source development comparison\nUntouched base only. Persistent source state and identical packets/propositions in both arms. Compare direct single-pass generation with evidence selection, drafting, semantic audit and deterministic rendering. 32 displayed answers, at most 64 generation calls. No training or adapters. Model auditing is not proof of support.\n')
    cell('code', 'install', '%pip install -q '+' '.join(f'{k}=={v}' for k, v in protocol['packages'].items() if k != 'torch')+'\n')
    packed = {name: {'sha256': sha(raw), 'data': base64.b85encode(lzma.compress(raw)).decode()} for name, raw in payload.items()}
    cell('code', 'unpack', 'import base64, hashlib, lzma, sys\nfrom pathlib import Path\nbundle=Path("/kaggle/working/grounded_chat_v2_bundle")\nbundle.mkdir(exist_ok=True)\npacked='+repr(packed)+'\nfor name,value in packed.items():\n    raw=lzma.decompress(base64.b85decode(value["data"]))\n    assert hashlib.sha256(raw).hexdigest()==value["sha256"]\n    (bundle/name).write_bytes(raw)\nsys.path.insert(0,str(bundle))\n')
    cell('code', 'run', 'from kaggle_grounded_chat_v2 import run\nrun(bundle,"/kaggle/working/grounded_chat_v2_results")\n')
    notebook = encoded({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells})
    metadata = json.loads((ROOT/'training/kaggle_cli/grounded_chat_v1/kernel-metadata.json').read_text())
    metadata.update(code_file='kaggle_grounded_chat_v2.ipynb', title='Trotsky grounded base chatbot v2')
    lock = {'notebook_sha256': sha(notebook), 'metadata_sha256': sha(encoded(metadata)), 'payload_sha256': {n: sha(b) for n,b in payload.items()}}
    local['bundle_lock.json'] = encoded(lock)
    for name, raw in local.items():
        if (out/name).exists() and (out/name).read_bytes() != raw:
            raise ValueError('Frozen v2 file changed; preserve and use a new version: '+name)
    for name, raw in local.items():
        (out/name).write_bytes(raw)
    stage = ROOT/'training/kaggle_cli/grounded_chat_v2'; stage.mkdir(parents=True, exist_ok=True)
    (stage/'kernel-metadata.json').write_bytes(encoded(metadata))
    (stage/'kaggle_grounded_chat_v2.ipynb').write_bytes(notebook)
    (ROOT/'notebooks/kaggle_grounded_chat_v2.ipynb').write_bytes(notebook)
    print(json.dumps({'stage':str(stage),'answers':protocol['answers'],'maximum_model_calls':protocol['maximum_model_calls'],'training_steps':0,'notebook_bytes':len(notebook)}))
    return payload, protocol


if __name__ == '__main__':
    build()
