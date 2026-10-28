import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import test_ghost_reviewed_selection as fixtures
from prepare_reviewed_target import make_jobs, score_targets


class TargetSelectionTests(unittest.TestCase):
    def fixture(self):
        _, _, parent = fixtures.ReviewedSelectionTests().fixture()
        for j in parent:
            j['question'] = 'Use only this source. Explain the relevant facts.'
            data = json.loads(j['messages'][1]['content']); data['current_user_request'] = j['question']
            j['messages'][1]['content'] = json.dumps(data)
            j['gold_answer'] = 'SECRET_EXPECTED'
        design = {'purpose': 'literal_target_selection_diagnostic', 'system_prompt': 'Choose IDs for the target only.',
                  'targets': [{'case_id': j['case_id'], 'target_span': 'Use only this source.',
                               'required_record_ids': ['SECRET_ORACLE']} for j in parent]}
        return parent, design

    def test_context_sources_menu_and_questions_unchanged_and_no_expectation_leak(self):
        parent, design = self.fixture(); before = copy.deepcopy((parent, design))
        jobs = make_jobs(parent, design, fixtures.Tokenizer())
        self.assertEqual((parent, design), before)
        for old, job in zip(parent, jobs):
            data = json.loads(job['messages'][1]['content'])
            self.assertEqual(data.pop('selection_target_span'), 'Use only this source.')
            self.assertEqual(data, json.loads(old['messages'][1]['content']))
            for key in ('question', 'source_packet_ids', 'available_record_fingerprints'):
                self.assertEqual(job[key], old[key])
            self.assertNotIn('SECRET_', json.dumps(job['messages']))
            self.assertNotIn('gold_answer', job)

    def test_rewritten_absent_or_ambiguous_target_is_rejected(self):
        parent, design = self.fixture()
        for value in ('Use a different source.', '', ' '):
            bad = copy.deepcopy(design); bad['targets'][0]['target_span'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_jobs(parent, bad, fixtures.Tokenizer())
        parent[0]['question'] += ' Use only this source.'
        with self.assertRaises(ValueError): make_jobs(parent, design, fixtures.Tokenizer())

    def test_real_input_budget_is_enforced_without_source_truncation(self):
        parent, design = self.fixture()
        class Long:
            def apply_chat_template(self, *a, **k): return [1]*5121
        with self.assertRaises(ValueError): make_jobs(parent, design, Long())

    def test_blocked_is_not_successful_empty_and_positive_needs_required_records(self):
        expectations = {'cases': [{'case_id': 'scope', 'selection_kind': 'empty'},
                                  {'case_id': 'fact', 'selection_kind': 'factual_request',
                                   'required_record_ids': ['needed', 'qualifier']}]}
        rows = [{'case_id': 'scope', 'status': 'blocked', 'selected_claim_ids': []},
                {'case_id': 'fact', 'status': 'reviewed_summaries_selected', 'selected_claim_ids': ['needed']}]
        report = score_targets(rows, expectations)
        self.assertEqual(report['diagnostic_checks_passed'], 0)
        self.assertEqual(report['checks'][1]['missing_required_record_ids'], ['qualifier'])
        rows[0]['status'] = 'no_summaries_selected'
        rows[1]['selected_claim_ids'] = ['needed', 'qualifier', 'irrelevant_extra']
        report = score_targets(rows, expectations)
        self.assertEqual(report['diagnostic_checks_passed'], 2)
        self.assertFalse(report['semantic_quality_validated'])
        self.assertFalse(report['independent_or_human_validation'])


if __name__ == '__main__': unittest.main()
