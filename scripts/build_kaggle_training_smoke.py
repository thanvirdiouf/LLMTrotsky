#!/usr/bin/env python3
"""Build a standalone, reviewed-data-only Kaggle training compatibility notebook."""
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

if __package__:
    from .build_training_pilot import validate_lock
    from .fresh_holdout_guard import guard_training_project
    from .kaggle_training import DEFAULT_SETTINGS
else:
    from build_training_pilot import validate_lock
    from fresh_holdout_guard import guard_training_project
    from kaggle_training import DEFAULT_SETTINGS

PROJECT = Path(__file__).resolve().parents[1]
PACKAGES = {"transformers": "5.20.2", "peft": "0.21.2", "accelerate": "1.15.0",
            "bitsandbytes": "0.50.2", "huggingface_hub": "1.33.0", "tokenizers": "0.23.1",
            "jinja2": "3.1.6"}
BUNDLED_PATHS = {
    "training_masks.py": "scripts/training_masks.py",
    "build_training_pilot.py": "scripts/build_training_pilot.py",
    "kaggle_training.py": "scripts/kaggle_training.py",
    "kbit_preparation.py": "scripts/kbit_preparation.py",
    "pilot_lock.json": "training/pilot_lock.json",
    "train_seed.jsonl": "datasets/train_seed.jsonl",
}


def make_notebook(project=PROJECT):
    project = Path(project)
    guard_training_project(project)
    lock = json.loads((project / "training/pilot_lock.json").read_text())
    validate_lock(project, lock)
    cells = []

    def add(kind, identity, source):
        cell = {"cell_type": kind, "id": identity, "metadata": {},
                "source": textwrap.dedent(source).lstrip().splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "intro", """
    # Trotsky: Qwen3.5-9B QLoRA training preflight

    Import into a fresh Kaggle notebook, select **GPU T4 x2** (only GPU 0 is used), enable Internet, and Run All. The reviewed **500-example dataset and helper code are bundled**; no data attachment is needed. This notebook takes **two optimizer steps**, covering one single-turn and one multi-turn conversation. It checks training compatibility; it does not estimate historical quality or run the full pilot.

    Before model loading, inspect the real tokenizer tables: labels cover every assistant answer and its `<|im_end|>`, with system/user text, headers, the empty thinking prefix and separators masked. Examples over 1,024 tokens are **reported and dropped whole**. The official template is unchanged. Explicit precomputed labels go directly to a PyTorch loop, so no trainer/template flag can silently change the masks.

    The base uses the successful inference run's immutable checkpoint and NF4 double quantization. Text-only loading omits the vision tower. PEFT prepares the quantized model before attaching adapters to all verified text decoder Linear projections, including both attention types and MLPs. Frozen input/output matrices are temporarily staged on CPU during preparation, converted to compute precision there, and restored to GPU before any computation. All parameters must be on GPU before training; norms stay FP32. This avoids the transient FP32 GPU allocation that exhausted T4 memory in version 1. Training is still unverified on Kaggle until this notebook passes.

    Outputs: `/kaggle/working/trotsky_qlora_smoke/`. Save a private version **with outputs**, or download the run folder. A pass requires finite loss/gradients, adapter updates, frozen base, exact saved/reloaded adapter tensors, and matching greedy output after a **fresh base reload**. Failures preserve their stage and partial reports. A rerun creates a new folder; no incomplete training state is silently resumed.
    """)
    add("code", "install", "%pip install -q " + " ".join(f"{p}=={v}" for p, v in PACKAGES.items()) + "\n")
    add("markdown", "stack-note", """
    Keep Kaggle's installed CUDA PyTorch. The Transformers/quantization pins come from the successful inference run; PEFT 0.21.2 is a **candidate training pin**, not a previously verified training stack. Start fresh or restart after changing imported package versions. Actual package, GPU, memory and dtype information is recorded in each run. FP16 AMP starts with a recorded loss scale of 1; the previous default scale produced non-finite gradients. Every optimizer step must still pass strict finite-gradient checks.
    """)
    bundle = {}
    for name, relative in BUNDLED_PATHS.items():
        payload = (project / relative).read_bytes()
        bundle[name] = {"sha256": hashlib.sha256(payload).hexdigest(),
                        "zlib_base64": base64.b64encode(zlib.compress(payload, level=9)).decode()}
    add("code", "bundle", textwrap.dedent("""
    import base64, hashlib, json, sys, zlib
    from pathlib import Path
    BUNDLE = Path('/kaggle/working/trotsky_qlora_bundle')
    BUNDLE.mkdir(parents=True, exist_ok=True)
    """) + "\nBUNDLED_FILES = " + repr(bundle) + "\n" + textwrap.dedent("""
    for name, item in BUNDLED_FILES.items():
        payload = zlib.decompress(base64.b64decode(item['zlib_base64']))
        if hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError('Bundled file checksum mismatch: ' + name)
        (BUNDLE / name).write_bytes(payload)
    sys.path.insert(0, str(BUNDLE))
    from build_training_pilot import validate_training_payload
    lock = json.loads((BUNDLE / 'pilot_lock.json').read_text())
    rows = validate_training_payload((BUNDLE / 'train_seed.jsonl').read_bytes(), lock)
    print('Verified reviewed training rows:', len(rows))
    print('Training SHA-256:', lock['dataset']['sha256'])
    """))
    add("code", "settings", "\nfrom kaggle_training import DEFAULT_SETTINGS, run_smoke\n"
        + "SETTINGS = " + repr(DEFAULT_SETTINGS) + "\n"
        + "OUTPUT_ROOT = Path('/kaggle/working/trotsky_qlora_smoke')\n"
        + "print(json.dumps(SETTINGS, indent=2))\n")
    add("code", "inspect-masks", r'''
    # CPU tokenizer work only; no model weights are loaded in this cell.
    import html
    from IPython.display import display, HTML
    from transformers import AutoTokenizer
    from training_masks import encode_assistant_example, inspect_tokens, collate_examples
    tokenizer = AutoTokenizer.from_pretrained(SETTINGS['model_id'], revision=SETTINGS['revision'])
    encoded = [encode_assistant_example(tokenizer, r['messages'], SETTINGS['max_length'], r['example_id']) for r in rows]
    kept = [e for e in encoded if not e['dropped']]
    print('Total / retained / whole-example overflow drops:', len(rows), len(kept), len(rows) - len(kept))
    for name, predicate in [('single', lambda e: len(e['assistant_spans']) == 1),
                            ('multi', lambda e: len(e['assistant_spans']) > 1)]:
        example = min((e for e in kept if predicate(e)), key=lambda e: e['token_count'])
        evidence = inspect_tokens(tokenizer, example)
        print(name, example['example_id'], 'tokens:', example['token_count'], 'targets:', example['supervised_tokens'])
        print('Assistant spans:', example['assistant_spans'])
        body = ''.join('<tr style="background:' + ('#d8f3dc' if t['supervised'] else '#eeeeee') + '">'
                       + ''.join('<td>' + html.escape(str(t[k])) + '</td>' for k in
                                 ['index', 'token_id', 'decoded', 'label', 'assistant_message_index', 'end_of_answer'])
                       + '</tr>' for t in evidence)
        display(HTML('<details open><summary>' + name + ': green = supervised; gray = ignored</summary>'
                     + '<div style="max-height:480px;overflow:auto"><table><tr>'
                     + '<th>Position</th><th>Token ID</th><th>Decoded</th><th>Label</th><th>Assistant turn</th><th>End token</th>'
                     + '</tr>' + body + '</table></div></details>'))
    print('The run below rechecks every mask and writes complete token inspection files.')
    ''')
    add("code", "train-save-reload", "RUN_DIR = run_smoke(BUNDLE, OUTPUT_ROOT, SETTINGS)\n")
    add("code", "inspect-results", """
    result = json.loads((RUN_DIR / 'run_manifest.json').read_text())
    print('Status:', result['status'])
    print(json.dumps(result['checks'], indent=2))
    print((RUN_DIR / 'training_metrics.jsonl').read_text())
    print('Reloaded answer:', json.loads((RUN_DIR / 'reloaded_probe.json').read_text())['answer'])
    print('Download/save this entire folder:', RUN_DIR)
    """)
    add("markdown", "interpretation", """
    ## Before the 500-example pilot

    A passing preflight establishes that these masks, adapters, precision and GPU settings can train and reload. The two examples are selected for small memory use, not representativeness. The 64-token introduction probe is operational; its answer is not a benchmark result. The full pilot should use the frozen 500-row dataset and explicitly record the final number of examples retained at its chosen length, train order, epochs/steps and settings. Keep its baseline and adapter prompt/decoding identical. The probe explicitly uses the tokenizer’s `<|im_end|>` ID 248046; earlier benchmark config used EOS 248044. Before pilot quality evaluation, run a fresh matched baseline with the corrected stop token. The historical baseline stays preserved in the lock.

    `pilot_lock.json` freezes dataset/provenance/persona, the baseline checkpoint, generation settings, and available benchmark artifact inventory. Evaluation passages and generated benchmark answers are excluded from training; only hash guards are bundled. The repeatedly inspected six questions are **development evidence**. Add fresh evaluation across other works and failure types before broad conclusions or larger training. Dataset expansion, retrieval and GPU utilization tuning can wait.

    If CUDA runs out of memory, inspect `run_manifest.json` and `memory.jsonl`. Reduce `max_length` only with an explicit new run, inspect its whole-example drops, and ensure it still contains both conversation types. Do not substitute model sharding, CPU offload, larger batches or untested kernels during this preflight. Inspect the saved stage on any exception; notebook completion alone does not imply a pass.

    Sources: [TRL assistant-only masking requirements](https://huggingface.co/docs/trl/en/sft_trainer#train-on-assistant-messages-only), [PEFT quantization preparation](https://huggingface.co/docs/peft/en/developer_guides/quantization), [Qwen3.5 text architecture](https://huggingface.co/docs/transformers/model_doc/qwen3_5).
    """)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def build(project=PROJECT):
    target = Path(project) / "notebooks/kaggle_qlora_smoke.ipynb"
    target.write_text(json.dumps(make_notebook(project), indent=1, ensure_ascii=False) + "\n")
    print(target)
    return target


if __name__ == "__main__":
    build()
