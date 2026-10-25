"""Check source isolation, matched controls, review blinding and real CPU dispatch."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts import kaggle_parallel as runtime
from scripts import kaggle_source_validation as experiment
from scripts.build_kaggle_notebook import make_notebook as baseline_notebook
from scripts.build_kaggle_source_validation import make_notebook

PROJECT = Path(__file__).resolve().parents[1]


class SourceValidationTests(unittest.TestCase):
    def setUp(self):
        self.original = runtime.load_cases(PROJECT / 'evaluation/cases.jsonl')
        self.cases = experiment.prepare_validation(self.original)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def rows(self):
        return [dict(run_id='run', model_id='fake', case_id=c['case_id'],
                     prompt_sha256=runtime.fingerprint(runtime.model_messages(c)),
                     response=f'Answer {i}', finish_reason='length' if i == 0 else 'stop',
                     input_tokens=100 + i, output_tokens=12) for i, c in enumerate(self.cases)]

    def read_csv(self, name):
        with (self.root / name).open(newline='') as stream:
            return list(csv.DictReader(stream))

    def test_only_validation_controls_exact_original_messages_and_no_input_mutation(self):
        untouched = copy.deepcopy(self.original)
        cases = experiment.prepare_validation(self.original)
        self.assertEqual(self.original, untouched)
        self.assertEqual(len({c['case_id'] for c in cases}), 12)
        original = {c['case_id']: c for c in self.original}
        for case in cases:
            self.assertEqual(case['split'], 'validation')
            base = original[case['original_case_id']]
            self.assertEqual(case['messages'][0], base['messages'][0])
            if case['condition'] == 'prompt_only':
                self.assertEqual(case['messages'], runtime.model_messages(base))
                self.assertEqual(runtime.fingerprint(case['messages']), runtime.fingerprint(base['messages']))
            else:
                content = case['messages'][-1]['content']
                prefix = base['messages'][-1]['content'] + experiment.SOURCE_PREAMBLE
                self.assertTrue(content.startswith(prefix))
                source = json.loads(content[len(prefix):])
                self.assertEqual(source['passage'], base['reference_evidence'])
                self.assertEqual(source['source_url'], base['source_url'])
                for check in base['reference_checks']:
                    self.assertNotIn(check, content)

    def test_gpu_assignment_balances_conditions_and_smoke_limit_is_a_complete_pair(self):
        shards = runtime.make_jobs(self.cases, set(), [0, 1], 12)
        for _, jobs in shards:
            self.assertEqual(sum(j['case_id'].startswith('prompt_only__') for j in jobs), 3)
            self.assertEqual(sum(j['case_id'].startswith('provided_source__') for j in jobs), 3)
            for job in jobs:
                self.assertEqual(set(job), {'case_id', 'messages', 'prompt_sha256'})
        smoke = runtime.make_jobs(self.cases, set(), [0, 1], 2)
        ids = [j['case_id'] for _, jobs in smoke for j in jobs]
        self.assertEqual({i.split('__')[1] for i in ids}, {experiment.VALIDATION_IDS[0]})
        self.assertEqual({i.split('__')[0] for i in ids}, set(experiment.CONDITIONS))

    def test_changed_evidence_missing_case_and_multiturn_input_are_rejected(self):
        first = next(c for c in self.original if c['split'] == 'validation')
        first['reference_evidence'] += ' Changed'
        with self.assertRaisesRegex(ValueError, 'source evidence'):
            experiment.prepare_validation(self.original)
        for mutation in ('missing', 'multiturn'):
            cases = copy.deepcopy(runtime.load_cases(PROJECT / 'evaluation/cases.jsonl'))
            index = next(i for i, c in enumerate(cases) if c['split'] == 'validation')
            if mutation == 'missing':
                cases.pop(index)
            else:
                cases[index]['messages'].append(dict(role='user', content='extra'))
            with self.assertRaises(ValueError):
                experiment.prepare_validation(cases)

    def test_experiment_identity_changes_when_prompt_or_review_reference_changes(self):
        identity = experiment.experiment_identity(self.cases)
        self.assertFalse(identity['retrieval_tested'])
        for field in ('messages', 'reference_checks'):
            cases = copy.deepcopy(self.cases)
            if field == 'messages':
                cases[0]['messages'][-1]['content'] += ' changed'
            else:
                cases[0]['reference_checks'].append('new review criterion')
            self.assertNotEqual(identity, experiment.experiment_identity(cases))

    def test_review_omits_condition_and_augmented_prompts_but_keeps_equal_references(self):
        summary = experiment.export_results(runtime, self.rows(), self.cases, self.root)
        self.assertEqual(summary['completed_pairs'], 6)
        self.assertEqual(summary['status'], 'completed')
        self.assertFalse(summary['quality_scores_assigned'])
        reviews = self.read_csv('validation_review.csv')
        keys = {r['review_id']: r for r in self.read_csv('validation_review_key.csv')}
        self.assertEqual(len(reviews), 12)
        for review in reviews:
            self.assertNotIn('condition', review)
            self.assertNotIn('case_id', review)
            self.assertNotIn(experiment.SOURCE_PREAMBLE, review['question'])
            for score in runtime.SCORE_COLUMNS:
                self.assertEqual(review[score], '')
        for original_id in experiment.VALIDATION_IDS:
            pair = [r for r in reviews if keys[r['review_id']]['original_case_id'] == original_id]
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0]['question'], pair[1]['question'])
            self.assertEqual(pair[0]['source_evidence'], pair[1]['source_evidence'])
        pairs = self.read_csv('paired_answers.csv')
        self.assertEqual(len(pairs), 6)
        self.assertEqual(pairs[0]['prompt_only_finish_reason'], 'length')
        self.assertEqual(summary['conditions']['prompt_only']['length_limited'], 1)

    def test_partial_review_ids_survive_resume_and_summary_stays_incomplete(self):
        rows = self.rows()
        partial = experiment.export_results(runtime, rows[:2], self.cases, self.root)
        keys = {r['case_id']: r['review_id'] for r in self.read_csv('validation_review_key.csv')}
        self.assertEqual(partial['status'], 'incomplete')
        self.assertEqual(partial['completed_pairs'], 1)
        experiment.export_results(runtime, rows, self.cases, self.root)
        full = {r['case_id']: r['review_id'] for r in self.read_csv('validation_review_key.csv')}
        self.assertTrue(all(full[case_id] == review_id for case_id, review_id in keys.items()))

    def test_review_rejects_duplicate_unknown_different_run_and_changed_prompt(self):
        for mode in ('duplicate', 'unknown', 'different_run', 'prompt'):
            rows = self.rows()
            if mode == 'duplicate':
                rows.append(rows[0])
            elif mode == 'unknown':
                rows[0]['case_id'] = 'unknown'
            elif mode == 'different_run':
                rows[0]['run_id'] = 'other'
            else:
                rows[0]['prompt_sha256'] = 'changed'
            with self.assertRaises(ValueError):
                experiment.export_results(runtime, rows, self.cases, self.root)

    def test_input_preview_and_notebook_are_reproducible_and_self_contained(self):
        experiment.write_inputs(runtime, self.cases, self.root)
        preview = [json.loads(line) for line in (self.root / 'model_inputs.jsonl').read_text().splitlines()]
        self.assertEqual(len(preview), 12)
        for row, case in zip(preview, self.cases):
            self.assertEqual(row['messages'], runtime.model_messages(case))
            self.assertNotIn('reference_checks', row)
        notebook = make_notebook()
        self.assertEqual(len({c['id'] for c in notebook['cells']}), len(notebook['cells']))
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code' or cell['id'] == 'install':
                continue
            source = ''.join(cell['source'])
            if cell['id'].endswith('-source'):
                source = source.split('\n', 1)[1]
                filename = 'kaggle_parallel.py' if cell['id'] == 'worker-source' else 'kaggle_source_validation.py'
                self.assertEqual(source, (PROJECT / 'scripts' / filename).read_text())
            compile(source, cell['id'], 'exec')
        saved = json.loads((PROJECT / 'notebooks/kaggle_source_validation.ipynb').read_text())
        self.assertEqual(saved, notebook)
        baseline = json.loads((PROJECT / 'notebooks/kaggle_prompt_baseline.ipynb').read_text())
        self.assertEqual(baseline, baseline_notebook())

    def test_preflight_counts_tokens_instead_of_tokenizer_dictionary_fields(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                ids = list(range(13))
                return ids if kwargs.get('return_dict') is False else dict(input_ids=ids, attention_mask=[1] * 13)
        cell = next(c for c in make_notebook()['cells'] if c['id'] == 'preflight')
        source = ''.join(cell['source'])
        check = source[source.index('input_lengths = {}'):source.index("print('Validation questions:")]
        namespace = dict(tok=Tokenizer(), runtime=runtime, cases=self.cases, MAX_INPUT_TOKENS=8192)
        exec(check, namespace)
        self.assertEqual(set(namespace['input_lengths'].values()), {13})
        namespace['MAX_INPUT_TOKENS'] = 12
        with self.assertRaisesRegex(ValueError, 'exceeds MAX_INPUT_TOKENS'):
            exec(check, namespace)

    def test_twelve_prepared_inputs_dispatch_resume_and_export_without_a_model(self):
        worker = self.root / 'fake_worker.py'
        worker.write_text('''
import argparse, json, os
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--job')
job = json.loads(Path(parser.parse_args().job).read_text())
assert os.environ['CUDA_VISIBLE_DEVICES'] == str(job['gpu_id'])
with Path(job['output_path']).open('a') as stream:
    for item in job['jobs']:
        assert set(item) == {'case_id', 'messages', 'prompt_sha256'}
        row = dict(run_id=job['run_id'], model_id=job['model_id'], case_id=item['case_id'],
            prompt_sha256=item['prompt_sha256'], response='Synthetic transport test only',
            finish_reason='stop', output_tokens=3, gpu_id=job['gpu_id'])
        stream.write(json.dumps(row) + '\\n')
        stream.flush()
''')
        identity = dict(model_id='fake', revision='commit', generation={}, chat_template_sha256='template',
                        experiment=experiment.experiment_identity(self.cases))
        manifest = dict(run_id=runtime.fingerprint(identity), identity=identity)
        with contextlib.redirect_stdout(io.StringIO()):
            first, _ = runtime.execute_parallel(worker, self.root, manifest, self.cases, [0, 1], 2,
                '/fake/cache', 512, 8192, 42, True, 'float16')
            rows, summary = runtime.execute_parallel(worker, self.root, manifest, self.cases, [0, 1], 12,
                '/fake/cache', 512, 8192, 42, True, 'float16')
        self.assertEqual(len(first), 2)
        self.assertEqual(len(rows), 12)
        self.assertEqual(summary['new_cases'], 10)
        result = experiment.export_results(runtime, rows, self.cases, self.root)
        self.assertEqual(result['completed_pairs'], 6)
        self.assertEqual({r['gpu_id'] for r in rows}, {0, 1})


if __name__ == '__main__':
    unittest.main()
