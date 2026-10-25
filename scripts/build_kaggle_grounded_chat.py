#!/usr/bin/env python3
"""Freeze and package base-only retrieval/chat evaluation, with no training data."""
import base64
import hashlib
import json
import lzma
from pathlib import Path
from build_grounded_chat import build
from grounded_chat import ROOT, MODEL
from evaluate_grounded_chat import model_spec, retrieval_preflight


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def build_bundle():
    collection = build()
    spec = json.loads((ROOT/'config/grounded_chat_evaluation.json').read_text())
    out = ROOT/'evaluation/grounded_chat_v1'
    out.mkdir(parents=True, exist_ok=True)
    payload = {name: (ROOT/'scripts'/name).read_bytes() for name in [
        'grounded_chat.py', 'evaluate_grounded_chat.py', 'kaggle_grounded_chat.py',
        'kaggle_grounding.py', 'kbit_preparation.py']}
    payload['collection.json'] = encoded(collection)
    payload['model_cases.json'] = encoded(model_spec(spec))
    # Use the already verified local inference stack, without changing the older
    # locked protocols. No network/package/version discovery at build time.
    prior = json.loads((ROOT/'evaluation/fresh_holdout_grounding_v1/protocol.json').read_text())
    protocol = {k: prior[k] for k in ['packages', 'generation', 'max_input_tokens', 'chat_template_sha256']}
    protocol.update(model=MODEL, training_steps=0, adapters_loaded=[],
                    answers=2*len(spec['singles'])+sum(len(c['turns']) for c in spec['conversations']),
                    files={name: sha(raw) for name, raw in payload.items()},
                    evaluation_spec_sha256=sha(encoded(spec)),
                    freshness=spec['freshness'], rubric=spec['rubric'])
    # Rubric stays local; the remote runtime receives no expected answers.
    runtime = {k: v for k, v in protocol.items() if k not in {'rubric'}}
    payload['protocol.json'] = encoded(runtime)
    local = {'protocol.json': encoded(protocol), 'evaluation_spec.json': encoded(spec),
             'retrieval_preflight.json': encoded(retrieval_preflight(collection, spec))}
    for name, raw in local.items():
        path = out/name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('Frozen artifact changed; use a new version: '+name)
    for name, raw in local.items():
        (out/name).write_bytes(raw)
    cells = []
    def cell(kind, cid, source):
        c = {'cell_type': kind, 'id': cid, 'metadata': {}, 'source': source.splitlines(True)}
        if kind == 'code':
            c.update(execution_count=None, outputs=[])
            if cid != 'install':
                compile(source, cid, 'exec')
        cells.append(c)
    cell('markdown', 'intro', '# Grounded base chatbot prototype\nSeven allowlisted context packets; actual retrieval, eight fresh single questions with matched oracle diagnostics, four sustained four-turn conversations. No training or adapters. Source-exposed development evaluation, not an unseen-source benchmark.\n')
    cell('code', 'install', '%pip install -q '+' '.join(f'{k}=={v}' for k, v in protocol['packages'].items() if k != 'torch')+'\n')
    packed = {name: {'sha256': sha(raw), 'data': base64.b85encode(lzma.compress(raw)).decode()} for name, raw in payload.items()}
    cell('code', 'unpack', 'import base64, hashlib, json, lzma, sys\nfrom pathlib import Path\nbundle = Path("/kaggle/working/grounded_chat_bundle")\nbundle.mkdir(exist_ok=True)\npacked = '+repr(packed)+'\nfor name, value in packed.items():\n    raw = lzma.decompress(base64.b85decode(value["data"]))\n    assert hashlib.sha256(raw).hexdigest() == value["sha256"]\n    (bundle/name).write_bytes(raw)\nsys.path.insert(0, str(bundle))\n')
    cell('code', 'run', 'from kaggle_grounded_chat import run\nrun(bundle, "/kaggle/working/grounded_chat_results")\n')
    notebook = encoded({'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': cells})
    metadata = {'id': 'thanvirdiouf/trotsky-grounded-base-chatbot', 'title': 'Trotsky grounded base chatbot',
        'code_file': 'kaggle_grounded_chat.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': True, 'enable_internet': True,
        'dataset_sources': [], 'competition_sources': [], 'kernel_sources': []}
    previous_metadata = json.loads((ROOT/'training/kaggle_cli/grounding_v1/kernel-metadata.json').read_text())
    for key in ('docker_image', 'docker_image_pinning_type', 'machine_shape'):
        metadata[key] = previous_metadata[key]
    stage = ROOT/'training/kaggle_cli/grounded_chat_v1'
    lock = {'notebook_sha256': sha(notebook), 'metadata_sha256': sha(encoded(metadata)),
            'payload_sha256': {name: sha(raw) for name, raw in payload.items()}, 'answers': protocol['answers']}
    lockpath = out/'bundle_lock.json'
    if lockpath.exists() and lockpath.read_bytes() != encoded(lock):
        raise ValueError('Frozen notebook bundle changed; preserve original')
    stage.mkdir(parents=True, exist_ok=True)
    (ROOT/'notebooks').mkdir(exist_ok=True)
    (ROOT/'notebooks/kaggle_grounded_chat.ipynb').write_bytes(notebook)
    (stage/'kaggle_grounded_chat.ipynb').write_bytes(notebook)
    (stage/'kernel-metadata.json').write_bytes(encoded(metadata))
    lockpath.write_bytes(encoded(lock))
    print(json.dumps({'stage': str(stage), 'answers': protocol['answers'], 'training_steps': 0, 'notebook_bytes': len(notebook)}))
    return payload, protocol


if __name__ == '__main__':
    build_bundle()
