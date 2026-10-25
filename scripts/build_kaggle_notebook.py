#!/usr/bin/env python3
"""Bundle the standalone parallel worker in an uploadable Kaggle notebook."""
import json
from pathlib import Path
import textwrap

PROJECT = Path(__file__).resolve().parents[1]

def make_notebook():
    cells = []
    def add(kind, identity, source):
        cell = dict(cell_type=kind, id=identity, metadata={}, source=textwrap.dedent(source).lstrip().splitlines(True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', 'intro', '''
    # Trotsky baseline: one independent worker per GPU

    Import this notebook into Kaggle, attach the project's original `evaluation/cases.jsonl`, enable Internet and **GPU T4 x2**, then Run All. The default runs all 36 cases; set `LIMIT = 2` for a short parallel check. Each worker holds a full Qwen3.5-9B NF4 copy on its own GPU and processes different cases. Batch size stays one, with the same prompt, greedy decoding, token budget and non-thinking template as the previous baseline.

    Start with a fresh session, or interrupt the previous inference before running this version. The notebook downloads/caches one checkpoint snapshot and starts fresh Python worker processes. Two full model copies must fit separately: two T4s can run the 9B NF4 model, but are not a suitable configuration for two copies of the larger candidates. The preflight checks each selected GPU's available memory. No CPU/disk model offload is used.

    `/kaggle/working/trotsky_baseline_parallel` holds the results. Save a notebook version with outputs or download them before the session ends. To resume a **parallel** run in another session, attach its saved run folder and set `RESUME_DIR`. The parallel run has a separate identity; it does not silently reuse the earlier serial smoke answer. It downloads no model weights to your PC.

    The reserved evaluation set contains 6 validation cases, 6 test cases and 24 behavioral cases. Only the messages enter worker prompts; reference fields stay in the parent for review exports. Keep prompt/settings fixed while finishing this baseline. The serial and parallel 36-case runs both completed on Kaggle, with identical answer text. The parallel run recorded 747.849 seconds of execution plus 94.9 seconds of checkpoint preparation. Timing boundaries differ from the serial run, so no exact speedup factor is claimed.
    ''')
    add('code', 'install', '''
    # Versions observed in the successful serial Kaggle run. Keep Kaggle's installed CUDA torch.
    # If these libraries were already imported with other versions, restart after installing.
    %pip install -q transformers==5.20.2 accelerate==1.15.0 bitsandbytes==0.50.2 huggingface_hub==1.33.0 tokenizers==0.23.1
    ''')
    add('code', 'configuration', '''
    from pathlib import Path

    MODEL_ID = 'Qwen/Qwen3.5-9B'
    MODEL_REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'  # same as the successful serial run
    LOAD_IN_4BIT = True
    MAX_NEW_TOKENS = 512
    MAX_INPUT_TOKENS = 8192
    SEED = 42
    LIMIT = 36  # set 2 to check both workers first, then restore 36 and rerun inference
    GPU_IDS = None  # None uses all visible GPUs; [0] selects one worker for a speed comparison
    CASES_PATH = None  # set explicitly if multiple input datasets contain cases.jsonl
    RESUME_DIR = None  # e.g. Path('/kaggle/input/saved-parallel-run/Qwen3.5-9B-...')
    OUT = Path('/kaggle/working/trotsky_baseline_parallel')
    ''')
    runtime = (PROJECT / 'scripts/kaggle_parallel.py').read_text()
    add('code', 'worker-source', '%%writefile /kaggle/working/trotsky_parallel_runtime.py\n' + runtime)
    add('code', 'preflight', '''
    import importlib.util
    import importlib.metadata
    import json
    import os
    import subprocess
    import sys
    import time
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig

    WORKER_PATH = Path('/kaggle/working/trotsky_parallel_runtime.py')
    spec = importlib.util.spec_from_file_location('trotsky_parallel_runtime', WORKER_PATH)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    cases = runtime.load_cases(CASES_PATH)
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a Kaggle GPU accelerator.')
    gpu_ids = list(range(torch.cuda.device_count())) if GPU_IDS is None else list(GPU_IDS)
    runtime.make_jobs(cases, set(), gpu_ids, LIMIT)  # validate selection and limits
    if any(i >= torch.cuda.device_count() for i in gpu_ids):
        raise ValueError('GPU_IDS contains a device not visible in this session.')
    if MODEL_ID not in {'Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-27B', 'Qwen/Qwen3.6-35B-A3B'}:
        raise ValueError('Choose one of the three verified Qwen candidates.')
    token = os.environ.get('HF_TOKEN')
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:
            token = None  # these public repositories do not require a token
    revision = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, token=token).sha
    config = AutoConfig.from_pretrained(MODEL_ID, revision=revision, token=token, trust_remote_code=False)
    if type(config) not in AutoModelForCausalLM._model_mapping:
        raise RuntimeError('Installed Transformers lacks this Qwen text-only model mapping.')
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, token=token, trust_remote_code=False)
    if not isinstance(tok.chat_template, str) or 'enable_thinking' not in tok.chat_template:
        raise ValueError('Tokenizer has no supported non-thinking switch.')
    prompt = tok.apply_chat_template(runtime.model_messages(cases[0]), tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    if not prompt.endswith('<think>\\n\\n</think>\\n\\n'):
        raise ValueError('Unexpected non-thinking assistant prefix.')
    generation = runtime.generation_config(config, tok, GenerationConfig, MAX_NEW_TOKENS)
    dtype_name = 'bfloat16' if all(torch.cuda.get_device_capability(i)[0] >= 8 for i in gpu_ids) else 'float16'
    params = {'Qwen/Qwen3.5-9B': 9e9, 'Qwen/Qwen3.5-27B': 27e9, 'Qwen/Qwen3.6-35B-A3B': 35e9}[MODEL_ID]
    rough_weights = params * (0.7 if LOAD_IN_4BIT else 2.0)
    gpus = []
    for gpu in gpu_ids:
        free, total = torch.cuda.mem_get_info(gpu)
        if free * 0.85 < rough_weights:
            raise RuntimeError(f'GPU {gpu} cannot fit an estimated full model copy with headroom. Use 9B NF4 or more memory.')
        gpus.append(dict(device_id=gpu, name=torch.cuda.get_device_name(gpu), total_bytes=total))
    versions = {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'huggingface-hub', 'tokenizers']}
    identity = dict(model_id=MODEL_ID, revision=revision, cases_sha256=runtime.EXPECTED_CASES_SHA256,
        chat_template_sha256=runtime.digest(tok.chat_template.encode()), generation=generation.to_dict(),
        quantization='bnb_nf4_double' if LOAD_IN_4BIT else 'none', dtype='torch.' + dtype_name,
        enable_thinking=False, max_input_tokens=MAX_INPUT_TOKENS, seed=SEED, packages=versions, gpus=gpus,
        execution_mode='one_process_per_gpu', worker_sha256=runtime.digest(WORKER_PATH.read_bytes()), runner_version=4)
    run_id = runtime.fingerprint(identity)
    manifest = dict(run_id=run_id, identity=identity)
    run_dir = OUT / f'{MODEL_ID.split("/")[-1]}-{run_id[:12]}'
    print('Workers:', gpus)
    print('Revision:', revision, '| compute dtype:', dtype_name, '| NF4:', LOAD_IN_4BIT)
    print('Run:', run_dir)
    ''')
    add('code', 'inference', '''
    # All GPU/model work below runs only on Kaggle. Rerun to resume completed cases.
    # LIMIT may change without changing the run identity; keep other settings fixed.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    prepared = time.perf_counter()
    # One shared cached snapshot; workers load locally and do not redownload the weights.
    snapshot_path = snapshot_download(MODEL_ID, revision=revision, token=token,
        allow_patterns=['*.json', '*.jinja', '*.txt', '*.safetensors'], max_workers=2)
    print(f'Checkpoint preparation: {time.perf_counter() - prepared:.1f}s')
    rows, summary = runtime.execute_parallel(WORKER_PATH, run_dir, manifest, cases, gpu_ids, LIMIT,
        snapshot_path, MAX_NEW_TOKENS, MAX_INPUT_TOKENS, SEED, LOAD_IN_4BIT, dtype_name, resume_dir=RESUME_DIR)
    print(json.dumps(summary, indent=2))
    print('Saved:', run_dir)
    print('Length-limited responses:', sum(r['finish_reason'] == 'length' for r in rows))
    print('Last response:', rows[-1]['response'] if rows else '(none)')
    ''')
    add('markdown', 'scoring', '''
    ## Results and troubleshooting

    `responses.jsonl` is the validated merged output in original case order. Per-GPU journals preserve partial progress; rerunning skips completed cases. A torn final journal line is backed up before recovery. Conflicting duplicates, changed prompts, changed settings and incompatible resume directories fail explicitly. Interrupting the cell terminates its worker processes. Each run is locked against simultaneous writers.

    `worker.gpu-0.log` and `worker.gpu-1.log` show model loading, warnings and case progress. If a worker fails, the other can finish its assigned cases; partial results and review sheets are exported before the failure is reported. Inspect the log rather than assuming a completed Kaggle notebook means every worker succeeded. `execution_summary.json` reports the number of selected/completed/new cases and wall time including subprocess startup/model loading, excluding snapshot preparation. Each response records its worker GPU, synchronized generation time and output tokens per second. Compare a fresh `GPU_IDS=[0]` run with a fresh two-GPU run for an end-to-end speed measurement. Warm-up, CPU contention and model-loading overhead can reduce speedup; 100% GPU utilization is not a target by itself.

    `human_review.csv` includes questions/context, source evidence, checklists and blank scores. Keep `review_key.csv` separate for blind review, and preserve scored copies elsewhere: rerunning regenerates the blank export. Use the project `evaluation/README.md` rubric; flag invented memories, quotes/citations, post-cutoff hindsight and speaker misattribution separately from averages. No automatic model-quality scores are assigned. Preserve model/template/precision/decoding settings for baseline-versus-adapter comparisons.

    The implementation keeps the known reference kernels; it does not add unverified kernel dependencies to the working T4 setup. Both workers can be active, but sequential token generation within each answer still limits utilization. Full-precision or larger models need enough memory **on each GPU** for replication. The archived `kaggle_prompt_baseline_serial.ipynb` retains the previous `device_map='auto'` path for models that require sharding.

    This parallel version passed offline process tests and completed 36 real Kaggle cases, 18 per GPU, in [saved version 361155394](https://www.kaggle.com/code/thanvirdiouf/notebook8b9f7bfe23?scriptVersionId=361155394). All answer texts matched the serial baseline. The recorded execution time was 747.849 seconds, excluding 94.9 seconds of checkpoint preparation. Sources: [Accelerate distributed inference](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference), [Hub snapshot caching](https://huggingface.co/docs/huggingface_hub/en/guides/download).
    ''')
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                        language_info=dict(name='python')), nbformat=4, nbformat_minor=5)

def build():
    notebook = make_notebook()
    target = PROJECT / 'notebooks/kaggle_prompt_baseline.ipynb'
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
    print(target)

if __name__ == '__main__':
    build()
