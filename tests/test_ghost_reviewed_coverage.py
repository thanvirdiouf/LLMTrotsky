import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import test_ghost_reviewed_selection as fixtures
from candidate_audit import reserve_call
from candidate_reviewed_coverage import evaluate, select_parts, validate_protocol
from grounded_chat_v2 import EvidenceError
from prepare_reviewed_coverage import make_jobs, question_parts
from reviewed_source_display import records_for_packets
from verify_reviewed_coverage import replay
from test_ghost_candidate_audit import protocol as common_protocol


class CoverageTests(unittest.TestCase):
    def fixture(self):
        c, l, parent = fixtures.ReviewedSelectionTests().fixture()
        design = json.loads((ROOT/'config/ghost_reviewed_coverage_design_v1.json').read_text())
        for j in parent:
            j['question'] = 'Check 5.7% and 4.7%; preserve the denominator. Explain uncertainty?'
            data = json.loads(j['messages'][1]['content']); data['current_user_request'] = j['question']
            j['messages'][1]['content'] = json.dumps(data)
            j['reviewer_only'] = 'SECRET_EXPECTED_ANSWER'
        return c, l, parent, make_jobs(parent, design, fixtures.Tokenizer())

    def test_literal_parts_preserve_decimals_and_all_original_characters(self):
        question = ' Check 5.7% and 4.7%; explain. Keep “four.” separate?  '
        parts = question_parts(question)
        self.assertEqual(''.join(parts), question)
        self.assertEqual(len(parts), 4)  # Includes final whitespace, without deleting it.
        self.assertIn('5.7% and 4.7%', parts[0])
        with self.assertRaises(ValueError): question_parts(' ')

    def test_only_literal_parts_and_system_schema_change_parent_sources_and_questions(self):
        _, _, parent, jobs = self.fixture()
        for old, job in zip(parent, jobs):
            original = json.loads(old['messages'][1]['content'])
            new = json.loads(job['messages'][1]['content'])
            self.assertEqual(new.pop('request_parts'), question_parts(old['question']))
            self.assertEqual(new, original)
            for key in ('question', 'source_packet_ids', 'available_record_fingerprints'):
                self.assertEqual(job[key], old[key])
            self.assertNotIn('SECRET_', json.dumps(job['messages']))
            self.assertNotIn('reviewer_only', job)

    def test_repeated_ids_across_parts_display_once_without_hidden_union_truncation(self):
        records = {str(i): {} for i in range(13)}
        groups, ids = select_parts('{"part_claim_ids":[["1","2"],["2","3"],[]]}', records, 3)
        self.assertEqual(groups, [['1', '2'], ['2', '3'], []]); self.assertEqual(ids, ['1', '2', '3'])
        for text, count in [('{"part_claim_ids":[["1","1"]]}', 1),
                            ('{"part_claim_ids":[["unknown"]]}', 1),
                            ('{"part_claim_ids":[[]],"facts":"INVENTED"}', 1),
                            ('{"part_claim_ids":[[]]}', 2),
                            (json.dumps({'part_claim_ids': [[str(i) for i in range(12)], ['12']]}), 2)]:
            with self.subTest(text=text), self.assertRaises(EvidenceError):
                select_parts(text, records, count)

    def test_actual_five_call_pipeline_preserves_union_and_exact_frozen_replay(self):
        c, l, _, jobs = self.fixture()
        response = {'response': '{"part_claim_ids":[["new_h"],["sum","limit"],["limit"]]}',
                    'finish_reason': 'stop', 'token_ids': [1], 'input_token_ids': [1, 2, 3]}
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            result = evaluate(jobs, lambda _: copy.deepcopy(response), d, c, l)
            self.assertEqual(result['completed_evaluation_calls'], 5)
            for row in map(json.loads, (d/'answers.jsonl').read_text().splitlines()):
                self.assertEqual(row['selected_claim_ids'], ['new_h', 'sum', 'limit'])
                self.assertEqual(row['rendered_claim_ids'], row['selected_claim_ids'])
                self.assertFalse(row['part_presence_is_semantic_coverage'])
                self.assertFalse(row['requested_task_coverage_verified'])
                self.assertEqual(row['memory_updates'], [])
            names = ('candidate_reviewed_coverage.py', 'candidate_audit.py', 'candidate_base_probe.py',
                     'reviewed_source_display.py', 'grounded_chat.py', 'grounded_chat_v2.py')
            frozen = {n: (ROOT/'scripts'/n).read_bytes() for n in names}
            frozen.update({n: json.dumps(value).encode() for n, value in
                           [('collection.json', c), ('ledger.json', l), ('audit_inputs.json', jobs)]})
            calls = [json.loads(line) for line in (d/'model_calls.jsonl').read_text().splitlines()]
            for name, raw in replay(frozen, calls).items(): self.assertEqual(raw, (d/name).read_bytes())

    def test_truncation_or_bad_part_schema_blocks_without_another_call(self):
        c, l, _, jobs = self.fixture()
        for text, stop in [('{"part_claim_ids":[[],[],[]]}', 'length'),
                           ('{"part_claim_ids":[[]]}', 'stop'),
                           ('{"part_claim_ids":[["unknown"],[],[]]}', 'stop')]:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as temp:
                d = Path(temp); outputs = []
                def generate(_):
                    outputs.append(1); return {'response': text, 'finish_reason': stop}
                evaluate(jobs, generate, d, c, l)
                rows = [json.loads(line) for line in (d/'answers.jsonl').read_text().splitlines()]
                self.assertEqual(len(outputs), 5)
                self.assertTrue(all(r['status'] == 'blocked' and not r['selected_claim_ids'] for r in rows))

    def test_five_eval_call_bound_and_no_truth_or_semantic_coverage_approval(self):
        p = {**common_protocol(), 'purpose': 'candidate_reviewed_coverage', 'evaluation_calls': 5,
             'independent_questions': 5, 'maximum_displayed_answers': 5, 'maximum_selected_records': 12,
             'generated_factual_text_used': False, 'model_audit_performed': False,
             'part_presence_is_semantic_coverage': False}
        validate_protocol(p)
        for key, value in [('evaluation_calls', 6), ('part_presence_is_semantic_coverage', True),
                           ('model_audit_performed', True), ('generated_factual_text_used', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError): validate_protocol({**p, key: value})
        result = {'memory_probe_calls': 0, 'evaluation_calls': 0}
        for _ in range(5): reserve_call(result, p, 1024)
        with self.assertRaises(ValueError): reserve_call(result, p, 1024)


if __name__ == '__main__': unittest.main()
