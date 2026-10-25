#!/usr/bin/env python3
"""Build and stage the frozen inference-only Kaggle checkpoint comparison."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

if __package__:
    from .kaggle_checkpoint_compare import validate_payloads
else:
    from kaggle_checkpoint_compare import validate_payloads

PROJECT = Path(__file__).resolve().parents[1]
FRESH = 'evaluation/fresh_holdout_v1'
BUNDLED_PATHS = {
    'kaggle_checkpoint_compare.py': 'scripts/kaggle_checkpoint_compare.py',
    'kbit_preparation.py': 'scripts/kbit_preparation.py',
    'training_masks.py': 'scripts/training_masks.py',
    'pilot_checkpoints.py': 'scripts/pilot_checkpoints.py',
    'model_inputs.jsonl': FRESH + '/model_inputs.jsonl',
    'pilot_run_manifest.json': 'training/runs/kaggle_361474126/run_manifest.json',
}
RUNTIME_FIELDS = ('schema_version', 'cases', 'conditions', 'model', 'packages',
                  'generation', 'enable_thinking', 'seed', 'max_input_tokens',
                  'chat_template_sha256', 'pilot_identity',
                  'pilot_identity_sha256', 'pilot_kernel_source')
DOCKER_IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def prepare_payloads(project=PROJECT):
    project = Path(project)
    protocol_bytes = (project / FRESH / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    # Reference artifacts remain local, but must still match their frozen hash.
    # Changing the rubric or exclusions after inference must not silently build
    # another notebook with the same declared experiment.
    for name, expected in protocol['files'].items():
        if Path(name).name != name or digest((project / FRESH / name).read_bytes()) != expected:
            raise ValueError('Frozen holdout file differs: ' + name)
    if digest((project / FRESH / 'specifications.json').read_bytes()) != protocol['specifications_sha256']:
        raise ValueError('Frozen holdout specifications differ')
    payloads = {name: (project / relative).read_bytes() for name, relative in BUNDLED_PATHS.items()}
    runtime = {key: protocol[key] for key in RUNTIME_FIELDS}
    runtime['files'] = {'model_inputs.jsonl': protocol['files']['model_inputs.jsonl']}
    runtime['source_protocol_sha256'] = digest(protocol_bytes)
    runtime['pilot_run_manifest_sha256'] = digest(payloads['pilot_run_manifest.json'])
    validate_payloads(payloads['model_inputs.jsonl'], runtime,
                      json.loads(payloads['pilot_run_manifest.json']))
    pilot_identity = runtime['pilot_identity']
    for name in ('kbit_preparation.py', 'training_masks.py', 'pilot_checkpoints.py'):
        if digest(payloads[name]) != pilot_identity['helpers'][name]:
            raise ValueError('Frozen pilot helper differs: ' + name)
    payloads['runtime_protocol.json'] = (json.dumps(runtime, ensure_ascii=False,
                                                sort_keys=True, indent=2) + '\n').encode()
    return payloads, runtime


def make_notebook(project=PROJECT):
    payloads, protocol = prepare_payloads(project)
    cells = []

    def add(kind, identity, source):
        cell = {'cell_type': kind, 'id': identity, 'metadata': {},
                'source': textwrap.dedent(source).lstrip().splitlines(True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', 'intro', '''
    # Trotsky checkpoint comparison

    This notebook performs **inference only** on 24 frozen fresh cases across six other works. The primary comparison is the untouched base against saved step 50; step 500 is a predeclared secondary comparison on the identical prompts. It creates no adapter, optimizer, training step, or new epoch.

    The source is private pilot version 2, attached as a version-pinned notebook output. The runner verifies complete checkpoint inventories and the saved pilot identity before loading weights. It generates the untouched base first, then loads step 50, and releases/reloads the base before loading step 500. Both adapters are checked against their saved tensors.

    Use a fresh **GPU T4 x2** session with Internet; only GPU 0 is used. Packages, model revision, NF4/FP16 preparation, non-thinking greedy decoding, 512-token output ceiling, seed 42 and EOS 248046 retain the pilot policy. All model-input messages are identical between conditions. Reference passages, checklists, training examples and scores are excluded from the bundled model-input data.
    ''')
    packages = {key: version for key, version in protocol['packages'].items() if key != 'torch'}
    add('code', 'install', '%pip install -q ' + ' '.join(f'{key}=={version}' for key, version in packages.items()) + '\n')
    add('markdown', 'stack', '''
    Keep the pinned Kaggle container's CUDA PyTorch. Actual versions, GPU identity, exact template and tokenized inputs are checked before generation. A changed stack fails rather than silently creating a different comparison. No training-state deserialization occurs; checkpoint files are hashed for provenance.
    ''')
    bundle = {name: {'sha256': digest(value),
                     'zlib_base64': base64.b64encode(zlib.compress(value, 9)).decode()}
              for name, value in payloads.items()}
    add('code', 'bundle', textwrap.dedent('''
    import base64, hashlib, json, sys, zlib
    from pathlib import Path
    BUNDLE = Path('/kaggle/working/trotsky_checkpoint_comparison_bundle')
    BUNDLE.mkdir(parents=True, exist_ok=True)
    ''') + '\nBUNDLED_FILES = ' + repr(bundle) + '\n' + textwrap.dedent('''
    for name, item in BUNDLED_FILES.items():
        payload = zlib.decompress(base64.b64decode(item['zlib_base64']))
        if hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError('Bundled checksum mismatch: ' + name)
        (BUNDLE / name).write_bytes(payload)
    sys.path.insert(0, str(BUNDLE))
    from kaggle_checkpoint_compare import run_comparison, validate_payloads
    protocol = json.loads((BUNDLE / 'runtime_protocol.json').read_bytes())
    pilot = json.loads((BUNDLE / 'pilot_run_manifest.json').read_bytes())
    cases = validate_payloads((BUNDLE / 'model_inputs.jsonl').read_bytes(), protocol, pilot)
    print('Verified fresh model-input-only cases:', len(cases))
    print('Conditions:', protocol['conditions'])
    print('Source frozen protocol SHA-256:', protocol['source_protocol_sha256'])
    print('Pinned pilot notebook input:', protocol['pilot_kernel_source'])
    '''))
    add('code', 'compare', '''
    OUT = Path('/kaggle/working/trotsky_checkpoint_comparison')
    # Explicit resume only from this same comparison's complete journal folder.
    # This never resumes training or loads checkpoint optimizer/RNG state.
    RESUME_DIR = None
    RUN_DIR = run_comparison(BUNDLE, OUT, input_root='/kaggle/input', resume_dir=RESUME_DIR)
    ''')
    add('code', 'inspect', '''
    result = json.loads((RUN_DIR / 'run_manifest.json').read_bytes())
    print('Status:', result['status'])
    print(json.dumps(result['checks'], indent=2))
    print((RUN_DIR / 'evaluation_summary.json').read_text())
    print('Saved comparison:', RUN_DIR)
    print('Compact artifact:', RUN_DIR / 'comparison_delivery.zip')
    ''' )
    add('markdown', 'interpretation', '''
    The exports report completeness, exact input/decoding identities, finish reasons and token counts. Quality cells remain blank. Score paired critical failures and fulfillment of each requested explanation against the separately frozen local rubric. Brevity, EOS stopping and training loss do not establish quality.

    These are small diagnostic results, not independent historical validation. Step 50 and step 500 differ both in update count and in examples seen, so this comparison cannot isolate a causal learning-rate or data effect. Preserve all original training data, locks and pilot outputs.
    ''')
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
            'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 5}


def kernel_metadata():
    return {'id': 'thanvirdiouf/trotsky-checkpoint-comparison',
            'title': 'Trotsky checkpoint comparison',
            'code_file': 'kaggle_checkpoint_compare.ipynb', 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
            'enable_tpu': False, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
            'docker_image': DOCKER_IMAGE, 'docker_image_pinning_type': 'original',
            'dataset_sources': [], 'competition_sources': [], 'model_sources': [],
            'kernel_sources': ['thanvirdiouf/trotsky-qlora-500-example-pilot/2']}


def build(project=PROJECT):
    project = Path(project)
    notebook = make_notebook(project)
    encoded = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    path = project / 'notebooks/kaggle_checkpoint_compare.ipynb'
    path.write_text(encoded)
    stage = project / 'training/kaggle_cli/checkpoint_compare'
    stage.mkdir(parents=True, exist_ok=True)
    (stage / path.name).write_text(encoded)
    (stage / 'kernel-metadata.json').write_text(json.dumps(kernel_metadata(), indent=2) + '\n')
    print(path)
    print('Kaggle CLI staging:', stage)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    build(parser.parse_args().project)
