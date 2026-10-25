#!/usr/bin/env python3
"""Bundle a paired six-question provided-source experiment for Kaggle."""
import argparse
import json
from pathlib import Path
import textwrap

try:
    from .build_kaggle_notebook import make_notebook as baseline_notebook
except ImportError:
    from build_kaggle_notebook import make_notebook as baseline_notebook

PROJECT = Path(__file__).resolve().parents[1]


def make_notebook(*, concise=False):
    notebook = baseline_notebook()
    cells = {c['id']: c for c in notebook['cells']}

    def source(identity, value):
        cells[identity]['source'] = textwrap.dedent(value).lstrip().splitlines(True)

    def replace(identity, old, new):
        value = ''.join(cells[identity]['source'])
        if value.count(old) != 1:
            raise ValueError(f'Baseline cell {identity} changed; review experiment bundling.')
        cells[identity]['source'] = value.replace(old, new).splitlines(True)

    source('intro', '''
    # Trotsky: paired provided-source validation

    Import this notebook into Kaggle, attach the original project `evaluation/cases.jsonl`, enable Internet and **GPU T4 x2**, and Run All in a fresh session. It runs **12 answers: six validation questions in two conditions**. `prompt_only` uses the exact original messages; `provided_source` appends a verified historical passage to the same question. The persona system message is unchanged. Questions from the test and behavioral splits are excluded.

    The checkpoint, NF4 quantization, non-thinking template, greedy decoding and 512-token output budget match the completed 9B baseline. Each GPU receives three questions from each condition. `LIMIT = 2` runs the first complete pair as a smoke check; restore 12 and rerun inference to finish. The two workers share one cached checkpoint and each holds a complete model copy. Stop any previous inference before starting.

    This is an **oracle/provided-source experiment**, not a retrieval benchmark: the passages are the reserved validation references, supplied deliberately. Scoring checklists never enter model prompts. The source arm also adds a short instruction to use the passage, so any difference measures that combined intervention. These six questions all come from one chapter of *The Revolution Betrayed*; success would not establish broad historical, autobiographical or temporal reliability. No training runs here.

    Results have a separate identity under `/kaggle/working/trotsky_source_validation`. Save a version **with outputs**, or download the run folder before ending the session. Only the notebook and existing evaluation dataset need uploading; both Python helpers are bundled. Model downloads and inference happen on Kaggle.
    ''')
    replace('configuration', 'LIMIT = 36  # set 2 to check both workers first, then restore 36 and rerun inference',
            'LIMIT = 12  # 2 = first complete pair; restore 12 and rerun inference to finish')
    replace('configuration', "Path('/kaggle/working/trotsky_baseline_parallel')",
            "Path('/kaggle/working/trotsky_source_validation')")
    replace('configuration', "# e.g. Path('/kaggle/input/saved-parallel-run/Qwen3.5-9B-...')",
            "# saved source-validation run folder only; baseline folders are incompatible")

    helper = dict(cell_type='code', id='experiment-source', metadata={}, execution_count=None, outputs=[],
        source=('%%writefile /kaggle/working/trotsky_source_validation.py\n'
                + (PROJECT / 'scripts/kaggle_source_validation.py').read_text()).splitlines(True))
    notebook['cells'].insert(4, helper)
    replace('preflight', 'cases = runtime.load_cases(CASES_PATH)', '''EXPERIMENT_PATH = Path('/kaggle/working/trotsky_source_validation.py')
spec_experiment = importlib.util.spec_from_file_location('trotsky_source_validation', EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(spec_experiment)
spec_experiment.loader.exec_module(experiment)
cases = experiment.prepare_validation(runtime.load_cases(CASES_PATH))''')
    replace('preflight', 'run_id = runtime.fingerprint(identity)', '''identity['experiment'] = experiment.experiment_identity(cases)
identity['experiment_helper_sha256'] = runtime.digest(EXPERIMENT_PATH.read_bytes())
run_id = runtime.fingerprint(identity)''')
    replace('preflight', "print('Workers:', gpus)", '''input_lengths = {}
for case in cases:
    encoded = tok.apply_chat_template(runtime.model_messages(case), tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False)
    input_lengths[case['case_id']] = len(encoded)
if max(input_lengths.values()) > MAX_INPUT_TOKENS:
    raise ValueError('At least one input exceeds MAX_INPUT_TOKENS; no input will be silently truncated.')
print('Validation questions: 6 | conditions: 2 | total answers: 12')
print('Input token range:', min(input_lengths.values()), 'to', max(input_lengths.values()))
print('Workers:', gpus)''')
    source('inference', '''
    # Only Kaggle runs GPU inference. Rerunning resumes this experiment's saved answers.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    experiment.write_inputs(runtime, cases, run_dir)
    prepared = time.perf_counter()
    snapshot_path = snapshot_download(MODEL_ID, revision=revision, token=token,
        allow_patterns=['*.json', '*.jinja', '*.txt', '*.safetensors'], max_workers=2)
    print(f'Checkpoint preparation: {time.perf_counter() - prepared:.1f}s')
    try:
        rows, summary = runtime.execute_parallel(WORKER_PATH, run_dir, manifest, cases, gpu_ids, LIMIT,
            snapshot_path, MAX_NEW_TOKENS, MAX_INPUT_TOKENS, SEED, LOAD_IN_4BIT, dtype_name, resume_dir=RESUME_DIR)
    finally:
        # Export condition-blinded review files even after a partial worker failure.
        if (run_dir / 'run_manifest.json').exists():
            saved_manifest = json.loads((run_dir / 'run_manifest.json').read_text())
            if saved_manifest == manifest:
                saved_rows = runtime.collect_rows(run_dir, run_id, cases)
                comparison = experiment.export_results(runtime, saved_rows, cases, run_dir)
                print(json.dumps(comparison, indent=2))
    print(json.dumps(summary, indent=2))
    print('Saved:', run_dir)
    print('Review validation_review.csv; keep validation_review_key.csv separate.')
    print('A LIMIT=2 smoke check leaves the full experiment incomplete; restore LIMIT=12 to finish.')
    ''')
    source('scoring', '''
    ## Inspect and review

    `comparison_summary.json` must show **completed_pairs: 6**, with six answers per condition. It reports completion and truncation counts, not answer-quality scores. `responses.jsonl` contains the validated outputs, with condition-prefixed case IDs. `model_inputs.jsonl` records exactly which messages each condition received. `experiment.json` and `run_manifest.json` bind the experiment, source references, checkpoint, settings and helper code to the run identity.

    Score **`validation_review.csv`** using the project `evaluation/README.md` rubric. Its rows are shuffled, labels are neutral, and each arm shows the same original question and review-only reference evidence. Keep **`validation_review_key.csv`** separate until scoring is complete; prose itself may still reveal the condition. Preserve a scored copy elsewhere because rerunning regenerates blank exports. The generic `human_review.csv` emitted by the shared runtime exposes condition IDs; use `validation_review.csv` for condition-blinded review instead.

    `paired_answers.csv` places the two answers side by side for later inspection and is **unblinded**. Compare source faithfulness, calibration and critical failures first; assess voice and conversation separately. Flag every `finish_reason=length` answer: truncation can affect both quality and apparent improvement. Keep the 512-token budget for this comparison; a larger budget would be a separate experiment.

    Each worker has its own journal and log. Partial progress is preserved; an interruption terminates workers. Rerun the inference cell to skip completed answers. In a new session, attach this experiment's saved run folder and set `RESUME_DIR`. Existing serial/parallel baseline folders belong to different identities and cannot be resumed into this run. Checkpoints and two full model copies must fit as in the successful 9B baseline.

    After scoring, check whether supplied evidence repairs the specific source errors. If it does, build a real retriever and evaluate its passage selection separately. If it does not, inspect the failed answers before choosing prompt changes, a larger model or fine-tuning. This local artifact has offline tests; its source-conditioned GPU execution still needs the Kaggle run.
    ''')
    if concise:
        source('intro', '''
        # Trotsky: concise paired source validation

        Import this notebook into a fresh Kaggle session, attach the original `evaluation/cases.jsonl`, enable Internet and **GPU T4 x2**, then Run All. It generates **12 answers: six validation questions, with and without supplied source passages**. Both conditions receive the identical instruction: **Answer directly in no more than 120 words. Do not add an introduction or a closing summary.**

        This is a separate prompt condition following saved run 361239681, where ten of twelve answers reached the 512-token ceiling. The original persona system message, questions, source passages, pinned Qwen3.5-9B checkpoint, NF4 precision, greedy non-thinking decoding and **512-token ceiling** are retained. Only the shared concise instruction is new. `prompt_only` now means persona/question plus that instruction; it is not the exact original baseline prompt. The notebook does not trim or retry answers that exceed 120 words.

        `LIMIT=12` completes all pairs; `LIMIT=2` checks the first pair, after which restoring 12 and rerunning inference resumes the remainder. Each of two GPUs initially receives three cases from each condition. Stop other GPU inference before starting. Both helpers are bundled; model loading and inference happen on Kaggle.

        Outputs have a separate identity under `/kaggle/working/trotsky_source_validation_concise`. Earlier run folders cannot resume into this condition. Save a version **with outputs** before ending the session. This is an oracle/provided-source test with six related questions from one chapter, not a retrieval benchmark. Checklists stay outside model input. No fine-tuning runs here.
        ''')
        replace('configuration', "Path('/kaggle/working/trotsky_source_validation')",
                "Path('/kaggle/working/trotsky_source_validation_concise')")
        replace('configuration', '# saved source-validation run folder only; baseline folders are incompatible',
                '# concise-run folder only; original source-validation/baseline folders are incompatible')
        replace('preflight', 'experiment.prepare_validation(runtime.load_cases(CASES_PATH))',
                'experiment.prepare_validation(runtime.load_cases(CASES_PATH), concise=True)')
        replace('preflight', "print('Validation questions: 6 | conditions: 2 | total answers: 12')",
                "print('Validation questions: 6 | conditions: 2 | total answers: 12')\nprint('Shared instruction:', experiment.CONCISE_INSTRUCTION)")
        scoring = ''.join(cells['scoring']['source'])
        source('scoring', scoring + '''

## Concise-condition review

`comparison_summary.json` reports `within_word_limit`, `over_word_limit` and `max_response_words` for each condition. Word counts use whitespace splitting. They are diagnostics, not quality scores. `validation_review.csv` shows the common instruction and word-limit compliance while keeping condition labels separate; `paired_answers.csv` includes both arms' counts. Over-limit and length-limited answers are preserved without trimming or retries. An empty or incomplete answer can meet the word limit and still fail the rubric.

Compare source faithfulness, calibration and critical failures within this concise pair first. Then compare each arm separately with its original long-answer counterpart from version 361239681. Report omissions as well as reduced hallucinations; six related pairs cannot establish statistical significance or general persona reliability. Changing the prompt makes this a new experiment, not an extension of the original run. The new concise GPU run remains pending until executed on Kaggle.
''')
    return notebook


def build(*, concise=False):
    name = 'kaggle_source_validation_concise.ipynb' if concise else 'kaggle_source_validation.ipynb'
    target = PROJECT / 'notebooks' / name
    target.write_text(json.dumps(make_notebook(concise=concise), indent=1, ensure_ascii=False) + '\n')
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--concise', action='store_true', help='Build the separate 120-word prompt condition.')
    build(concise=parser.parse_args().concise)
