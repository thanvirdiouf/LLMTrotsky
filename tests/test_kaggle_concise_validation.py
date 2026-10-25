"""Check that concise prompting is a separate, symmetric, auditable intervention."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts import kaggle_parallel as runtime
from scripts import kaggle_source_validation as experiment
from scripts.build_kaggle_source_validation import make_notebook

PROJECT = Path(__file__).resolve().parents[1]


class ConciseValidationTests(unittest.TestCase):
    def setUp(self):
        self.original = runtime.load_cases(PROJECT / 'evaluation/cases.jsonl')
        self.normal = experiment.prepare_validation(self.original)
        self.concise = experiment.prepare_validation(self.original, concise=True)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def rows(self):
        return [dict(run_id='concise-run', model_id='fake', case_id=c['case_id'],
            prompt_sha256=runtime.fingerprint(runtime.model_messages(c)), response='short answer',
            finish_reason='stop', input_tokens=300, output_tokens=3) for c in self.concise]

    def read_csv(self, name):
        with (self.root / name).open(newline='') as stream:
            return list(csv.DictReader(stream))

    def test_only_change_to_both_model_inputs_is_identical_length_instruction(self):
        for old, new in zip(self.normal, self.concise):
            self.assertEqual(old['case_id'], new['case_id'])
            self.assertEqual(new['split'], 'validation')
            self.assertEqual(old['messages'][0], new['messages'][0])
            self.assertEqual(old['original_messages'], new['original_messages'])
            before, after = old['messages'][-1]['content'], new['messages'][-1]['content']
            addition = '\n\n' + experiment.CONCISE_INSTRUCTION
            self.assertEqual(after.count(experiment.CONCISE_INSTRUCTION), 1)
            self.assertEqual(after.replace(addition, ''), before)
            self.assertEqual(old['reference_evidence'], new['reference_evidence'])
            for criterion in new['reference_checks']:
                self.assertNotIn(criterion, after)

    def test_new_identity_rejects_old_outputs_and_mixed_length_policies(self):
        normal = experiment.experiment_identity(self.normal)
        concise = experiment.experiment_identity(self.concise)
        self.assertNotEqual(normal['name'], concise['name'])
        self.assertNotEqual(normal['prepared_cases_sha256'], concise['prepared_cases_sha256'])
        self.assertEqual(concise['requested_max_words'], 120)
        with (self.root / 'responses.jsonl').open('w') as stream:
            runtime.append_jsonl(stream, self.rows()[0])
        with self.assertRaisesRegex(ValueError, 'different prompt'):
            runtime.read_completed(self.root / 'responses.jsonl', 'concise-run', self.normal)
        mixed = copy.deepcopy(self.concise)
        mixed[0].pop('answer_instruction')
        with self.assertRaisesRegex(ValueError, 'Mixed or unknown'):
            experiment.experiment_identity(mixed)

    def test_word_limit_boundary_blinded_exports_and_quality_scores_remain_blank(self):
        rows = self.rows()
        rows[0]['response'] = ' '.join(['word'] * 120)
        rows[1]['response'] = ' '.join(['word'] * 121)
        rows[2]['response'] = ''
        rows[3]['finish_reason'] = 'length'
        summary = experiment.export_results(runtime, rows, self.concise, self.root)
        self.assertEqual(summary['experiment'], experiment.CONCISE_EXPERIMENT)
        self.assertFalse(summary['quality_scores_assigned'])
        self.assertEqual(summary['conditions']['provided_source']['over_word_limit'], 1)
        self.assertEqual(summary['conditions']['prompt_only']['length_limited'], 1)
        reviews = self.read_csv('validation_review.csv')
        self.assertEqual(len(reviews), 12)
        for row in reviews:
            self.assertEqual(row['answer_instruction'], experiment.CONCISE_INSTRUCTION)
            self.assertNotIn('condition', row)
            self.assertNotIn('case_id', row)
            for score in runtime.SCORE_COLUMNS:
                self.assertEqual(row[score], '')
        pairs = self.read_csv('paired_answers.csv')
        self.assertEqual(pairs[0]['prompt_only_word_count'], '120')
        self.assertEqual(pairs[0]['prompt_only_within_word_limit'], 'True')
        self.assertEqual(pairs[0]['provided_source_within_word_limit'], 'False')
        self.assertEqual(pairs[1]['provided_source_word_count'], '0')

    def test_missing_answers_stay_blank_instead_of_counting_as_zero_word_successes(self):
        summary = experiment.export_results(runtime, self.rows()[:1], self.concise, self.root)
        self.assertEqual(summary['status'], 'incomplete')
        self.assertEqual(summary['completed_pairs'], 0)
        self.assertEqual(summary['conditions']['provided_source']['within_word_limit'], 0)
        self.assertIsNone(summary['conditions']['provided_source']['max_response_words'])
        first = self.read_csv('paired_answers.csv')[0]
        self.assertEqual(first['provided_source_word_count'], '')
        self.assertEqual(first['provided_source_within_word_limit'], '')

    def test_notebook_keeps_generation_and_worker_fixed_and_changes_output_root(self):
        normal = {c['id']: ''.join(c['source']) for c in make_notebook()['cells']}
        notebook = make_notebook(concise=True)
        concise = {c['id']: ''.join(c['source']) for c in notebook['cells']}
        for identity in ('worker-source', 'install', 'experiment-source', 'inference'):
            self.assertEqual(normal[identity], concise[identity])
        old, new = {}, {}
        exec(normal['configuration'], old)
        exec(concise['configuration'], new)
        for setting in ('MODEL_ID', 'MODEL_REVISION', 'LOAD_IN_4BIT', 'MAX_NEW_TOKENS',
                        'MAX_INPUT_TOKENS', 'SEED', 'LIMIT', 'GPU_IDS'):
            self.assertEqual(old[setting], new[setting])
        self.assertNotEqual(old['OUT'], new['OUT'])
        self.assertIn('concise=True', concise['preflight'])
        self.assertIn('return_dict=False', concise['preflight'])
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code' or cell['id'] == 'install':
                continue
            code = ''.join(cell['source'])
            if cell['id'].endswith('-source'):
                code = code.split('\n', 1)[1]
            compile(code, cell['id'], 'exec')
        saved = json.loads((PROJECT / 'notebooks/kaggle_source_validation_concise.ipynb').read_text())
        self.assertEqual(saved, notebook)


if __name__ == '__main__':
    unittest.main()
