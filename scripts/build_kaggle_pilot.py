#!/usr/bin/env python3
"""Build the standalone, frozen 500-example Kaggle QLoRA pilot notebook."""
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

if __package__:
    from .build_kaggle_training_smoke import PACKAGES
    from .build_training_pilot import validate_lock
    from .fresh_holdout_guard import guard_training_project
else:
    from build_kaggle_training_smoke import PACKAGES
    from build_training_pilot import validate_lock
    from fresh_holdout_guard import guard_training_project

PROJECT = Path(__file__).resolve().parents[1]
BUNDLED_PATHS = {
    "training_masks.py": "scripts/training_masks.py",
    "build_training_pilot.py": "scripts/build_training_pilot.py",
    "kbit_preparation.py": "scripts/kbit_preparation.py",
    "kaggle_training.py": "scripts/kaggle_training.py",
    "kaggle_pilot.py": "scripts/kaggle_pilot.py",
    "pilot_checkpoints.py": "scripts/pilot_checkpoints.py",
    "pilot_protocol.py": "scripts/pilot_protocol.py",
    "pilot_lock.json": "training/pilot_lock.json",
    "train_seed.jsonl": "datasets/train_seed.jsonl",
    "pilot_protocol.json": "training/pilot_protocol.json",
    "pilot_evaluation_inputs.jsonl": "training/pilot_evaluation_inputs.jsonl",
}


def make_notebook(project=PROJECT):
    if __package__:
        from .pilot_protocol import validate_protocol
    else:
        from pilot_protocol import validate_protocol
    project = Path(project)
    guard_training_project(project)
    lock = json.loads((project / "training/pilot_lock.json").read_text())
    validate_lock(project, lock)
    protocol = json.loads((project / "training/pilot_protocol.json").read_text())
    validate_protocol(project, protocol)
    cells = []

    def add(kind, identity, source):
        cell = {"cell_type": kind, "id": identity, "metadata": {},
                "source": textwrap.dedent(source).lstrip().splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "intro", """
    # Trotsky: frozen 500-example QLoRA pilot

    Import into a **fresh Kaggle GPU T4 x2 session**, enable Internet, and Run All. This notebook uses **GPU 0 only**; GPU 1 stays idle. The exact reviewed training data, model-input-only evaluation cases, protocol and helpers are bundled. No dataset attachment is needed for a new run.

    The passing two-step smoke test established the masks, quantized training preparation, adapter targets, AMP settings and fresh-base adapter reload. This pilot starts a **new adapter**, trains for **one epoch over all 500 examples exactly once**, and checks the longest example first. Remaining examples have a seeded order. Recoverable checkpoints are written every **50 optimizer steps** and at the end. Complete outputs are saved under `/kaggle/working/trotsky_qlora_pilot/`.

    Before loading model weights, inspect CPU tokenization and real assistant masks below. System/user text, assistant headers, empty thinking prefixes and separators have label `-100`; each complete assistant answer and its `<|im_end|>` token are supervised. Explicit labels feed the training loop. The pilot aborts if any reviewed example would be dropped or silently truncated.

    After training, the final adapter is saved, loaded onto a **fresh quantized base**, and checked for exact adapter tensors and matching greedy probe tokens. A matched comparison then generates all **36 existing development cases** with the baseline and adapter, using identical prompts, greedy decoding, a 512-token output budget and the corrected `<|im_end|>` EOS ID **248046**. Reference passages, scoring checklists and previous benchmark answers are excluded from both the training data and these model-input-only evaluation payloads.
    """)
    add("code", "install", "%pip install -q " + " ".join(f"{p}=={v}" for p, v in PACKAGES.items()) + "\n")
    add("markdown", "verified-stack", """
    Keep Kaggle's installed CUDA PyTorch. These package pins and the single-GPU NF4 preparation match the passing smoke run. Start fresh or restart the session after changing already imported package versions. The frozen base, omitted vision tower, text adapter targets, actual package versions, GPU memory, finite loss and gradients, and adapter updates are checked and recorded. FP16 AMP starts at the verified loss scale of 1.

    Frozen input/output matrices are temporarily staged on CPU during preparation and restored before computation; all parameters must be on GPU 0 for training. Norms stay FP32. Preserve this preparation and the adapter settings for the first pilot.
    """)
    bundle = {}
    for name, relative in BUNDLED_PATHS.items():
        payload = (project / relative).read_bytes()
        bundle[name] = {"sha256": hashlib.sha256(payload).hexdigest(),
                        "zlib_base64": base64.b64encode(zlib.compress(payload, level=9)).decode()}
    add("code", "bundle", textwrap.dedent("""
    import base64, hashlib, json, sys, zlib
    from pathlib import Path
    BUNDLE = Path('/kaggle/working/trotsky_pilot_bundle')
    BUNDLE.mkdir(parents=True, exist_ok=True)
    """) + "\nBUNDLED_FILES = " + repr(bundle) + "\n" + textwrap.dedent("""
    for name, item in BUNDLED_FILES.items():
        payload = zlib.decompress(base64.b64decode(item['zlib_base64']))
        if hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError('Bundled file checksum mismatch: ' + name)
        (BUNDLE / name).write_bytes(payload)
    sys.path.insert(0, str(BUNDLE))
    from pilot_protocol import validate_protocol_payloads
    lock = json.loads((BUNDLE / 'pilot_lock.json').read_text())
    protocol = json.loads((BUNDLE / 'pilot_protocol.json').read_text())
    rows, evaluation_inputs = validate_protocol_payloads(
        (BUNDLE / 'train_seed.jsonl').read_bytes(),
        (BUNDLE / 'pilot_evaluation_inputs.jsonl').read_bytes(), lock, protocol)
    assert len(rows) == 500, 'The pilot requires all 500 reviewed examples.'
    print('Verified reviewed training rows:', len(rows))
    print('Verified model-input-only evaluation cases:', len(evaluation_inputs))
    print('Training SHA-256:', lock['dataset']['sha256'])
    print('Protocol SHA-256:', BUNDLED_FILES['pilot_protocol.json']['sha256'])
    print('Evaluation input SHA-256:', BUNDLED_FILES['pilot_evaluation_inputs.jsonl']['sha256'])
    """))
    add("code", "settings", """
    from copy import deepcopy
    from kaggle_pilot import run_pilot
    from pilot_protocol import DEFAULT_SETTINGS
    SETTINGS = deepcopy(DEFAULT_SETTINGS)
    OUT = Path('/kaggle/working/trotsky_qlora_pilot')
    # For an explicit resume, attach the saved run and set this to its complete
    # checkpoint folder, e.g. Path('/kaggle/input/saved-pilot/run/checkpoints/checkpoint-000050').
    RESUME_CHECKPOINT = None
    print(json.dumps(SETTINGS, indent=2))
    print('Resume checkpoint:', RESUME_CHECKPOINT)
    """)
    add("code", "inspect-masks", r'''
    # CPU tokenizer work only: no model weights are loaded here.
    import html
    from IPython.display import display, HTML
    from transformers import AutoTokenizer
    from training_masks import encode_assistant_example, inspect_tokens
    tokenizer = AutoTokenizer.from_pretrained(SETTINGS['model_id'], revision=SETTINGS['revision'])
    encoded = [encode_assistant_example(tokenizer, row['messages'], SETTINGS['max_length'], row['example_id'])
               for row in rows]
    retained = [example for example in encoded if not example['dropped']]
    print('Total / retained / overflow drops:', len(rows), len(retained), len(rows) - len(retained))
    if len(retained) != 500:
        raise ValueError('Pilot requires all 500 complete examples; inspect the tokenization before training.')
    longest = max(retained, key=lambda example: example['token_count'])
    print('Token range:', min(example['token_count'] for example in retained), 'to', longest['token_count'])
    print('Longest example is checked first:', longest['example_id'])
    for name, predicate in [('single', lambda example: len(example['assistant_spans']) == 1),
                            ('multi', lambda example: len(example['assistant_spans']) > 1)]:
        example = min((example for example in retained if predicate(example)), key=lambda example: example['token_count'])
        evidence = inspect_tokens(tokenizer, example)
        print(name, example['example_id'], 'tokens:', example['token_count'], 'targets:', example['supervised_tokens'])
        print('Assistant spans:', example['assistant_spans'])
        body = ''.join('<tr style="background:' + ('#d8f3dc' if token['supervised'] else '#eeeeee') + '">'
                       + ''.join('<td>' + html.escape(str(token[key])) + '</td>' for key in
                                 ['index', 'token_id', 'decoded', 'label', 'assistant_message_index', 'end_of_answer'])
                       + '</tr>' for token in evidence)
        display(HTML('<details open><summary>' + name + ': green = supervised; gray = ignored</summary>'
                     + '<div style="max-height:480px;overflow:auto"><table><tr>'
                     + '<th>Position</th><th>Token ID</th><th>Decoded</th><th>Label</th><th>Assistant turn</th><th>End token</th>'
                     + '</tr>' + body + '</table></div></details>'))
    del encoded, retained, longest, evidence, body, example
    print('Training independently rechecks the masks and writes complete inspection evidence.')
    ''')
    add("markdown", "checkpoint-resume", """
    ## Checkpoints and explicit resume

    Leave `RESUME_CHECKPOINT = None` for the first run. A new run initializes a fresh adapter and a new output folder. To resume after interruption, save/download the complete run folder, attach it to a fresh Kaggle session, and point `RESUME_CHECKPOINT` at a complete checkpoint folder. Keep the bundled protocol and settings identical. Resume restores the adapter, optimizer, AMP scaler, random state and next position in the frozen training order. It rejects incomplete checkpoints and changed settings/data/helpers. The next unfinished example is trained; completed examples are not repeated.

    On failure, inspect the saved manifest and memory/metric logs and preserve the latest complete checkpoint before ending the session. Do not repeatedly run the training cell in a session holding a previous model. Start a fresh session for a new run or an explicit resume. Kaggle session storage is temporary: save a private version **with outputs** or download the complete run folder.
    """)
    add("code", "train-save-reload-evaluate", "RUN_DIR = run_pilot(BUNDLE, OUT, SETTINGS, resume_checkpoint=RESUME_CHECKPOINT)\n")
    add("code", "inspect-results", r'''
    result = json.loads((RUN_DIR / 'run_manifest.json').read_text())
    print('Status:', result['status'])
    print(json.dumps(result.get('checks', {}), indent=2))
    metrics = RUN_DIR / 'training_metrics.jsonl'
    if metrics.exists():
        print('Last training metric rows:')
        print('\n'.join(metrics.read_text().splitlines()[-5:]))
    print('Matched evaluation summary:')
    print((RUN_DIR / 'evaluation_summary.json').read_text())
    print('Exact prompt/decoding comparison manifest:')
    print((RUN_DIR / 'comparison_manifest.json').read_text())
    print('Published checkpoint folders:')
    print('\n'.join(str(path.relative_to(RUN_DIR)) for path in sorted(RUN_DIR.rglob('checkpoint-*')) if path.is_dir()))
    print('Output files:')
    print('\n'.join(str(path.relative_to(RUN_DIR)) for path in sorted(RUN_DIR.rglob('*')) if path.is_file()))
    print('Save/download the complete folder:', RUN_DIR)
    ''')
    add("markdown", "interpretation", """
    ## Reading the pilot

    A completed run must pass training, frozen-base, finite-gradient, checkpoint, final adapter save/reload and matched generation checks. Compare baseline and adapter responses with the same evaluation protocol; inspect truncation and failure types before scoring an apparent gain. The corrected EOS baseline supersedes the old stop-token setting for this comparison; historical baseline outputs stay preserved.

    The 36 cases are **existing development evidence** and include repeatedly inspected questions. They can guide this first pilot, but they do not establish broad historical or persona quality. Add fresh evaluation across other works and failure types before broad conclusions or scaling. The 500 reviewed training examples are enough for this first pilot. Expanding the dataset, building retrieval and optimizing GPU utilization can wait.

    Sources: [PEFT quantization preparation](https://huggingface.co/docs/peft/en/developer_guides/quantization), [TRL assistant-only masking requirements](https://huggingface.co/docs/trl/en/sft_trainer#train-on-assistant-messages-only), [Qwen3.5 text architecture](https://huggingface.co/docs/transformers/model_doc/qwen3_5).
    """)
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def build(project=PROJECT):
    target = Path(project) / "notebooks/kaggle_qlora_pilot.ipynb"
    target.write_text(json.dumps(make_notebook(project), indent=1, ensure_ascii=False) + "\n")
    print(target)
    return target


if __name__ == "__main__":
    build()
