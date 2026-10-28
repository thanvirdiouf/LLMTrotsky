import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import test_ghost_reviewed_target as fixtures
from prepare_reviewed_target import make_jobs as parent_jobs
from prepare_reviewed_target_context import make_jobs
from test_ghost_reviewed_selection import Tokenizer


class ContextLabelTests(unittest.TestCase):
    def fixture(self):
        parent, design = fixtures.TargetSelectionTests().fixture()
        design['system_prompt'] = 'Select selection_target_span within current_user_request only.'
        return parent_jobs(parent, design, Tokenizer()), json.loads((ROOT/'config/ghost_reviewed_target_context_v1.json').read_text())

    def test_only_key_name_and_single_system_reference_change(self):
        parent, design = self.fixture(); before = copy.deepcopy(parent)
        jobs = make_jobs(parent, design, Tokenizer())
        self.assertEqual(parent, before)
        for old, new in zip(parent, jobs):
            self.assertEqual(new['messages'][0]['content'].replace('original_question_context', 'current_user_request'), old['messages'][0]['content'])
            data = json.loads(new['messages'][1]['content'])
            restored = {'current_user_request' if k == 'original_question_context' else k: v for k, v in data.items()}
            self.assertEqual(restored, json.loads(old['messages'][1]['content']))
            self.assertEqual(list(restored), list(json.loads(old['messages'][1]['content'])))
            self.assertNotIn('current_user_request', data)
            self.assertEqual({k: v for k, v in new.items() if k not in ('messages', 'input_token_ids')},
                             {k: v for k, v in old.items() if k not in ('messages', 'input_token_ids')})
            self.assertNotIn('SECRET_', json.dumps(new['messages']))

    def test_changed_target_and_ambiguous_or_missing_context_key_rejected(self):
        parent, design = self.fixture()
        for change in ('target', 'duplicate_context', 'system_reference'):
            bad = copy.deepcopy(parent); data = json.loads(bad[0]['messages'][1]['content'])
            if change == 'target': data['selection_target_span'] = 'Another task'
            elif change == 'duplicate_context': data['original_question_context'] = 'Competing question'
            else: bad[0]['messages'][0]['content'] = 'No key reference'
            bad[0]['messages'][1]['content'] = json.dumps(data)
            with self.subTest(change=change), self.assertRaises(ValueError): make_jobs(bad, design, Tokenizer())

    def test_source_budget_rejected_without_truncation(self):
        parent, design = self.fixture()
        class Long:
            def apply_chat_template(self, *a, **k): return [1]*5121
        with self.assertRaises(ValueError): make_jobs(parent, design, Long())


if __name__ == '__main__': unittest.main()
