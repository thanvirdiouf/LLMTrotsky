#!/usr/bin/env python3
"""Kaggle per-GPU inference workers; GPU libraries are imported only in workers."""
from pathlib import Path
import csv
import hashlib
import json
import random
import shutil
import fcntl
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

EXPECTED_CASES_SHA256 = '036a898ac83a37058fc7770d9a5c43559a8f468f0ae05cd050b6763cfbd8b86f'
SCORE_COLUMNS = ['temporal_consistency_0_4', 'source_faithfulness_0_4', 'attribution_0_4',
                 'persona_voice_0_4', 'conversation_0_4', 'calibration_0_4', 'critical_failure', 'notes']

def digest(data):
    return hashlib.sha256(data).hexdigest()

def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())

def load_cases(path=None, input_root=Path('/kaggle/input')):
    if path is None:
        matches = sorted(input_root.rglob('cases.jsonl')) if input_root.exists() else []
        if len(matches) != 1:
            raise ValueError(f'Found {len(matches)} cases.jsonl files. Set CASES_PATH explicitly.')
        path = matches[0]
    raw = Path(path).read_bytes()
    if digest(raw) != EXPECTED_CASES_SHA256:
        raise ValueError('Evaluation set differs from the frozen project file; attach the original cases.jsonl.')
    cases = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if len(cases) != 36 or len({c['case_id'] for c in cases}) != 36:
        raise ValueError('Expected 36 distinct cases.')
    return cases

def model_messages(case):
    # Whitelist fields; never serialize the whole case into a prompt.
    return [{'role': m['role'], 'content': m['content']} for m in case['messages']]

def append_jsonl(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    stream.flush()

def read_completed(path, run_id, cases):
    rows = []
    seen = set()
    case_map = {c['case_id']: c for c in cases}
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)  # fail clearly on a damaged/interrupted line
        case_id = row['case_id']
        if row['run_id'] != run_id or case_id not in case_map or case_id in seen:
            raise ValueError('Mismatched run, unknown case, or duplicate output; do not combine these files.')
        if row['prompt_sha256'] != fingerprint(model_messages(case_map[case_id])):
            raise ValueError('Saved output has a different prompt.')
        seen.add(case_id)
        rows.append(row)
    return rows

def finish_reason(tokens, eos_token_id):
    eos = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    return 'stop' if tokens and tokens[-1] in eos else 'length'

def export_review(rows, cases, run_dir):
    case_map = {c['case_id']: c for c in cases}
    ordered = list(rows)
    random.Random(2026).shuffle(ordered)
    fields = ['review_id', 'case_id', 'split', 'category', 'messages', 'response', 'finish_reason',
              'source_url', 'source_evidence', 'reference_checks'] + SCORE_COLUMNS
    with (run_dir / 'human_review.csv').open('w', newline='', encoding='utf-8') as f, \
         (run_dir / 'review_key.csv').open('w', newline='', encoding='utf-8') as key:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        mapping = csv.DictWriter(key, fieldnames=['review_id', 'run_id', 'model_id', 'case_id'])
        mapping.writeheader()
        for i, row in enumerate(ordered, 1):
            case = case_map[row['case_id']]
            review_id = f'R{i:03d}'
            writer.writerow(dict(review_id=review_id, case_id=case['case_id'], split=case['split'],
                                 category=case['category'], messages=json.dumps(model_messages(case), ensure_ascii=False),
                                 response=row['response'], finish_reason=row['finish_reason'],
                                 source_url=case.get('source_url', ''), source_evidence=case.get('reference_evidence', ''),
                                 reference_checks=' | '.join(case['reference_checks'])))
            mapping.writerow(dict(review_id=review_id, run_id=row['run_id'], model_id=row['model_id'], case_id=case['case_id']))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def collect_rows(run_dir, run_id, cases):
    """Merge the published file and worker journals; reject conflicting duplicates."""
    found = {}
    paths = [run_dir / 'responses.jsonl', *sorted(run_dir.glob('responses.gpu-*.jsonl'))]
    for path in paths:
        for row in read_completed(path, run_id, cases):
            case_id = row['case_id']
            if case_id in found and found[case_id] != row:
                raise ValueError(f'Conflicting saved responses for {case_id}')
            found[case_id] = row
    return [found[c['case_id']] for c in cases if c['case_id'] in found]


def publish_rows(run_dir, rows):
    temporary = run_dir / 'responses.jsonl.tmp'
    with temporary.open('w', encoding='utf-8') as stream:
        for row in rows:
            append_jsonl(stream, row)
    temporary.replace(run_dir / 'responses.jsonl')


def recover_journal_tail(path):
    """Preserve a backup before removing only a torn, invalid final JSON line."""
    raw = path.read_bytes()
    if not raw or raw.endswith(b'\n'):
        return
    end = raw.rfind(b'\n') + 1
    try:
        json.loads(raw[end:])
    except (ValueError, UnicodeDecodeError):
        backup = path.with_name(path.name + f'.damaged-{time.time_ns()}.bak')
        backup.write_bytes(raw)
        path.write_bytes(raw[:end])
        print(f'Recovered incomplete journal tail; original preserved at {backup}', flush=True)
    else:
        path.write_bytes(raw + b'\n')


def make_jobs(cases, completed, gpu_ids, limit):
    if type(limit) is not int or not 1 <= limit <= len(cases):
        raise ValueError('LIMIT must be between 1 and the evaluation case count.')
    if not gpu_ids or any(type(i) is not int or i < 0 for i in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('GPU_IDS must contain distinct nonnegative device indices.')
    pending = [c for c in cases[:limit] if c['case_id'] not in completed]
    shards = []
    for rank, gpu_id in enumerate(gpu_ids):
        # These are the only historical case fields sent to worker processes.
        jobs = [{'case_id': c['case_id'], 'messages': model_messages(c),
                 'prompt_sha256': fingerprint(model_messages(c))} for c in pending[rank::len(gpu_ids)]]
        if jobs:
            shards.append((gpu_id, jobs))
    return shards


def worker_environment(gpu_id):
    env = os.environ.copy()
    # Set before importing torch in a fresh interpreter. Each child sees one GPU as cuda:0.
    visible = env.get('CUDA_VISIBLE_DEVICES')
    physical = visible.split(',')[gpu_id].strip() if visible else str(gpu_id)
    env.update(CUDA_VISIBLE_DEVICES=physical, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    return env


def execute_parallel(worker_script, run_dir, manifest, cases, gpu_ids, limit,
                     snapshot_path, max_new_tokens, max_input_tokens, seed, load_in_4bit, dtype_name,
                     resume_dir=None):
    """Launch independent interpreters, preserve partial results, and publish a validated merge."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / '.run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = run_dir / 'run_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Run manifest differs; start a separate run.')
        if resume_dir is not None and Path(resume_dir).resolve() != run_dir:
            resume = Path(resume_dir)
            if json.loads((resume / 'run_manifest.json').read_text()) != manifest:
                raise ValueError('Resume directory belongs to different settings or execution mode.')
            if not list(run_dir.glob('responses*.jsonl')):
                for path in resume.glob('responses*.jsonl'):
                    shutil.copy2(path, run_dir / path.name)
        write_json(manifest_path, manifest)
        for path in run_dir.glob('responses.gpu-*.jsonl'):
            recover_journal_tail(path)
        rows = collect_rows(run_dir, manifest['run_id'], cases)
        prior = {r['case_id'] for r in rows}
        shards = make_jobs(cases, prior, gpu_ids, limit)
        processes = []
        failure = None
        started = time.perf_counter()
        try:
            for gpu_id, jobs in shards:
                job_path = run_dir / f'jobs.gpu-{gpu_id}.json'
                write_json(job_path, dict(run_id=manifest['run_id'], model_id=manifest['identity']['model_id'],
                    revision=manifest['identity']['revision'], gpu_id=gpu_id, snapshot_path=str(snapshot_path),
                    output_path=str(run_dir / f'responses.gpu-{gpu_id}.jsonl'), jobs=jobs,
                    max_new_tokens=max_new_tokens, max_input_tokens=max_input_tokens, seed=seed,
                    load_in_4bit=load_in_4bit, dtype_name=dtype_name,
                    expected_generation=manifest['identity']['generation'],
                    expected_template_sha256=manifest['identity']['chat_template_sha256']))
                log_path = run_dir / f'worker.gpu-{gpu_id}.log'
                # Popen starts a fresh Python interpreter; no CUDA state is forked from this kernel.
                with log_path.open('a', encoding='utf-8') as log:
                    process = subprocess.Popen([sys.executable, str(worker_script), '--job', str(job_path)],
                        env=worker_environment(gpu_id), stdout=log, stderr=subprocess.STDOUT)
                processes.append((gpu_id, process, log_path))
                print(f'Started GPU {gpu_id}: {len(jobs)} cases. Log: {log_path}', flush=True)
            reported = len(prior)
            while any(p.poll() is None for _, p, _ in processes):
                count = 0
                for path in run_dir.glob('responses.gpu-*.jsonl'):
                    # Only count complete journal lines; final merging performs strict validation.
                    count += path.read_bytes().count(b'\n')
                if count != reported:
                    print(f'Worker journals: {count} saved responses', flush=True)
                    reported = count
                time.sleep(0.5)
            failed = [(gpu, p.returncode, log) for gpu, p, log in processes if p.returncode != 0]
            if failed:
                details = []
                for gpu, status, log in failed:
                    details.append(f'GPU {gpu}, exit {status}, {log}:\n' + log.read_text(errors='replace')[-4000:])
                failure = RuntimeError('Worker failure; completed cases were preserved.\n' + '\n'.join(details))
        except BaseException as exc:
            failure = exc
            raise
        finally:
            # Interruptions must not leave orphan GPU workers running after the cell ends.
            for _, process, _ in processes:
                if process.poll() is None:
                    process.terminate()
            for _, process, _ in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for path in run_dir.glob('responses.gpu-*.jsonl'):
                recover_journal_tail(path)
            rows = collect_rows(run_dir, manifest['run_id'], cases)
            publish_rows(run_dir, rows)
            export_review(rows, cases, run_dir)
            elapsed = time.perf_counter() - started
            selected_ids = {c['case_id'] for c in cases[:limit]}
            completed_ids = {r['case_id'] for r in rows}
            new_rows = [r for r in rows if r['case_id'] not in prior]
            summary = dict(run_id=manifest['run_id'], selected_cases=limit,
                completed_selected_cases=len(selected_ids & completed_ids), new_cases=len(new_rows),
                status='completed' if selected_ids <= completed_ids and failure is None else 'incomplete',
                execution_seconds=elapsed, new_output_tokens=sum(r.get('output_tokens', 0) for r in new_rows),
                finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(run_dir / 'execution_summary.json', summary)
        if failure:
            raise failure
        if summary['status'] != 'completed':
            raise RuntimeError('Workers exited without completing all selected cases; inspect worker logs.')
        return rows, summary


def generation_config(config, tokenizer, generation_class, max_new_tokens):
    # A separate generation_config.json is optional and absent from the verified 9B revision.
    result = generation_class.from_model_config(config.get_text_config(decoder=True))
    result.do_sample = False
    result.temperature = 1.0
    result.top_p = 1.0
    result.top_k = 0  # preserve the existing baseline settings (ignored under greedy decoding)
    result.num_beams = 1
    result.max_new_tokens = max_new_tokens
    result.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if result.eos_token_id is None:
        raise ValueError('Checkpoint is missing generation EOS token IDs.')
    return result


def run_worker(job_path):
    """Each worker is explicitly pinned to its single visible GPU, with batch size one."""
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, set_seed

    job = json.loads(Path(job_path).read_text())
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Each worker must have exactly one visible CUDA device.')
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    source = job['snapshot_path']
    load = dict(local_files_only=True, trust_remote_code=False, token=False)
    tokenizer = AutoTokenizer.from_pretrained(source, **load)
    config = AutoConfig.from_pretrained(source, **load)
    generation = generation_config(config, tokenizer, GenerationConfig, job['max_new_tokens'])
    if generation.to_dict() != job['expected_generation'] or digest(tokenizer.chat_template.encode()) != job['expected_template_sha256']:
        raise ValueError('Worker generation defaults or tokenizer differ from the run manifest.')
    dtype = getattr(torch, job['dtype_name'])
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=dtype) if job['load_in_4bit'] else None
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(source, **load, dtype=dtype,
                quantization_config=quant, device_map={'': 0}, attn_implementation='sdpa')
    model.eval()
    print(f'GPU {job["gpu_id"]} ready; load seconds: {time.perf_counter() - started:.2f}', flush=True)
    with Path(job['output_path']).open('a', encoding='utf-8') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case in job['jobs']:
            if case['prompt_sha256'] != fingerprint(case['messages']):
                raise ValueError('Worker prompt checksum mismatch.')
            inputs = tokenizer.apply_chat_template(case['messages'], tokenize=True, return_dict=True,
                return_tensors='pt', add_generation_prompt=True, enable_thinking=False).to('cuda:0')
            input_count = inputs['input_ids'].shape[-1]
            if input_count > job['max_input_tokens']:
                raise ValueError(f'{case["case_id"]}: prompt too long; nothing was truncated.')
            set_seed(job['seed'])
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**inputs, generation_config=generation)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            generated = ids[0][input_count:].tolist()
            append_jsonl(stream, dict(run_id=job['run_id'], model_id=job['model_id'],
                case_id=case['case_id'], prompt_sha256=case['prompt_sha256'],
                response=tokenizer.decode(generated, skip_special_tokens=True).strip(),
                finish_reason=finish_reason(generated, generation.eos_token_id),
                input_tokens=input_count, output_tokens=len(generated), gpu_id=job['gpu_id'],
                generation_seconds=seconds, output_tokens_per_second=len(generated) / seconds if seconds else None,
                created_utc=datetime.now(timezone.utc).isoformat()))
            print(f'GPU {job["gpu_id"]}: {case["case_id"]} saved ({seconds:.2f}s)', flush=True)
            del inputs, ids


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    run_worker(parser.parse_args().job)
