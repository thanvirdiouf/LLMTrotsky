"""Test the actual process launcher with concurrent CPU subprocesses, never model inference."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import kaggle_parallel as runtime


FAKE_WORKER = r'''
import argparse
import json
import os
from pathlib import Path
import sys
import time
parser = argparse.ArgumentParser()
parser.add_argument('--job')
job = json.loads(Path(parser.parse_args().job).read_text())
root = Path(job['output_path']).parent
control = json.loads((root / 'control.json').read_text())
assert os.environ['CUDA_VISIBLE_DEVICES'] == str(job['gpu_id'])
assert os.environ['OMP_NUM_THREADS'] == '1'
(root / f"{os.getpid()}.started").write_text(str(job['gpu_id']))
deadline = time.monotonic() + 5
while len(list(root.glob('*.started'))) < control.get('require_peers', 1):
    if time.monotonic() >= deadline:
        raise RuntimeError('Other worker never started concurrently')
    time.sleep(0.02)
if control.get('omit_gpu') == job['gpu_id']:
    sys.exit(0)
with Path(job['output_path']).open('a') as stream:
    for item in job['jobs']:
        assert set(item) == {'case_id', 'messages', 'prompt_sha256'}
        row = dict(run_id=job['run_id'], model_id=job['model_id'], case_id=item['case_id'],
            prompt_sha256=item['prompt_sha256'], response='Fake test response', finish_reason='stop',
            output_tokens=2, generation_seconds=0.1, gpu_id=job['gpu_id'])
        stream.write(json.dumps(row) + '\n')
        stream.flush()
        if control.get('fail_gpu') == job['gpu_id']:
            raise RuntimeError('Deliberate worker failure after preserving one case')
'''


class ParallelRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.worker = self.root / 'fake_worker.py'
        self.worker.write_text(FAKE_WORKER)
        self.run_dir = self.root / 'run'
        self.run_dir.mkdir()
        self.cases = [dict(case_id=f'c{i}', messages=[dict(role='user', content=f'Question {i}')],
                           reference_checks=['private check'], reference_evidence='private passage',
                           split='validation', category='source_backed') for i in range(4)]
        self.manifest = dict(run_id='run', identity=dict(model_id='fake', revision='commit',
                              generation={}, chat_template_sha256='template'))
        self.control()

    def control(self, **settings):
        (self.run_dir / 'control.json').write_text(json.dumps(settings))

    def execute(self, limit=4, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return runtime.execute_parallel(self.worker, self.run_dir, self.manifest, self.cases,
                [0, 1], limit, '/fake/cache', 512, 8192, 42, True, 'float16', **kwargs)

    def test_two_workers_really_overlap_and_merge_once_in_original_order(self):
        self.control(require_peers=2)
        rows, summary = self.execute()
        self.assertEqual([r['case_id'] for r in rows], ['c0', 'c1', 'c2', 'c3'])
        self.assertEqual({r['gpu_id'] for r in rows}, {0, 1})
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['new_cases'], 4)
        self.assertEqual(summary['new_output_tokens'], 8)
        self.assertEqual(len(list(self.run_dir.glob('*.started'))), 2)
        self.assertTrue((self.run_dir / 'human_review.csv').exists())
        for path in self.run_dir.glob('jobs.gpu-*.json'):
            self.assertNotIn('private passage', path.read_text())
            self.assertNotIn('private check', path.read_text())
        # A repeated run neither launches workers nor duplicates rows.
        again, summary = self.execute()
        self.assertEqual(again, rows)
        self.assertEqual(summary['new_cases'], 0)
        self.assertEqual(len(list(self.run_dir.glob('*.started'))), 2)

    def test_increasing_limit_preserves_completed_outputs(self):
        first, _ = self.execute(limit=2)
        self.assertEqual(len(first), 2)
        rows, summary = self.execute(limit=4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[:2], first)
        self.assertEqual(summary['new_cases'], 2)

    def test_failure_exports_partial_results_then_resume_only_missing_cases(self):
        self.control(fail_gpu=0, require_peers=2)
        with self.assertRaisesRegex(RuntimeError, 'Worker failure'):
            self.execute()
        rows = runtime.collect_rows(self.run_dir, 'run', self.cases)
        self.assertEqual(len(rows), 3)
        self.assertTrue((self.run_dir / 'human_review.csv').exists())
        self.assertEqual(json.loads((self.run_dir / 'execution_summary.json').read_text())['status'], 'incomplete')
        self.control()
        done, summary = self.execute()
        self.assertEqual(len(done), 4)
        self.assertEqual(summary['new_cases'], 1)
        self.assertEqual(len(list(self.run_dir.glob('*.started'))), 3)

    def test_success_exit_without_all_answers_is_a_failure(self):
        self.control(omit_gpu=0)
        with self.assertRaisesRegex(RuntimeError, 'without completing'):
            self.execute()
        self.assertEqual(len(runtime.collect_rows(self.run_dir, 'run', self.cases)), 2)

    def test_changed_manifest_is_rejected_before_workers_start(self):
        self.execute()
        self.manifest = {**self.manifest, 'run_id': 'changed'}
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            self.execute()
        self.assertEqual(len(list(self.run_dir.glob('*.started'))), 2)

    def test_resume_from_saved_parallel_directory(self):
        first, _ = self.execute(limit=2)
        saved = self.run_dir
        self.run_dir = self.root / 'restored'
        self.run_dir.mkdir()
        self.control()
        rows, summary = self.execute(resume_dir=saved)
        self.assertEqual(rows[:2], first)
        self.assertEqual(summary['new_cases'], 2)

    def test_conflicting_duplicate_cannot_be_silently_merged(self):
        self.execute()
        path = self.run_dir / 'responses.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['response'] = 'Conflicting answer'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'Conflicting saved'):
            runtime.collect_rows(self.run_dir, 'run', self.cases)

    def test_torn_last_line_is_backed_up_and_recovered(self):
        self.execute(limit=2)
        path = self.run_dir / 'responses.gpu-0.jsonl'
        with path.open('ab') as stream:
            stream.write(b'{"case_id":"torn')
        rows, summary = self.execute()
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(list(self.run_dir.glob('*.damaged-*.bak'))), 1)
        self.assertEqual(summary['new_cases'], 2)

    def test_duplicate_device_ids_and_bad_limits_are_rejected(self):
        for gpu_ids, limit in [([0, 0], 4), ([], 4), ([-1], 4), ([0], 0), ([0], 5)]:
            with self.subTest(gpu_ids=gpu_ids, limit=limit), self.assertRaises(ValueError):
                runtime.make_jobs(self.cases, set(), gpu_ids, limit)

    def test_gpu_selection_preserves_inherited_visibility_mapping(self):
        with patch.dict(runtime.os.environ, {'CUDA_VISIBLE_DEVICES': '3,2'}):
            self.assertEqual(runtime.worker_environment(0)['CUDA_VISIBLE_DEVICES'], '3')
            self.assertEqual(runtime.worker_environment(1)['CUDA_VISIBLE_DEVICES'], '2')

    def test_interruption_terminates_real_worker_processes(self):
        popen = runtime.subprocess.Popen
        processes = []
        def capture(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process
        sleep = runtime.time.sleep
        interrupted = False
        def interrupt_once(delay):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            return sleep(delay)
        with patch.object(runtime.subprocess, 'Popen', side_effect=capture), \
             patch.object(runtime.time, 'sleep', side_effect=interrupt_once):
            with self.assertRaises(KeyboardInterrupt):
                self.execute()
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(p.poll() is not None for p in processes))
        self.assertTrue((self.run_dir / 'responses.jsonl').exists())

    def test_simultaneous_parent_cannot_acquire_run_lock(self):
        with (self.run_dir / '.run.lock').open('a') as lock:
            runtime.fcntl.flock(lock.fileno(), runtime.fcntl.LOCK_EX | runtime.fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.execute()
        self.assertEqual(list(self.run_dir.glob('*.started')), [])


if __name__ == '__main__':
    unittest.main()
