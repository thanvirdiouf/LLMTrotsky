import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from grounded_chat import load_collection, fingerprint
from grounded_chat_v2 import (EvidenceChat, EvidenceError, SourceStateRetriever, initial_state,
    validate_ledger, validate_plan, validate_draft, validate_review, render_segments, parse_object, CHECKS)


class ChatV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collection = load_collection(ROOT/'evaluation/grounded_chat_v1/collection.json')
        cls.ledger = json.loads((ROOT/'config/grounded_chat_claims_v2.json').read_text())
        cls.cards = {c['id']: c for c in cls.ledger['claims']}

    def test_ledger_anchors_speakers_and_dates_are_immutable(self):
        validate_ledger(self.collection, self.ledger)
        for field, value in [('text', 'invented source wording'), ('speaker', 'Wrong speaker')]:
            bad = copy.deepcopy(self.ledger); bad['claims'][0]['anchors'][0][field] = value
            with self.subTest(field=field), self.assertRaises(EvidenceError):
                validate_ledger(self.collection, bad)
        bad = copy.deepcopy(self.ledger); bad['claims'][0]['composition_date_interval']['latest'] = '1947-01-01'
        with self.assertRaises(EvidenceError):
            validate_ledger(self.collection, bad)

    def test_topic_survives_long_followup_and_explicit_switch_resets_it(self):
        retriever = SourceStateRetriever(self.collection, self.ledger)
        state = initial_state(self.collection, self.ledger)
        for q in ['What happened in the Rakovsky article?', 'Why that hypothesis?', 'Explain it more fully.',
                  'Now make that shorter.', 'Are you still calling it unverified?', 'Explain why.']:
            trace = retriever.retrieve(q, state)
            self.assertIn('rakovsky', [m['packet']['id'] for m in trace['matches']])
            state['active_packet_ids'] = trace['active_packet_ids']
        trace = retriever.retrieve('Change subject: whose address is New York in the Weisbord letter?', state)
        self.assertEqual(trace['active_packet_ids'], ['weisbord'])
        self.assertEqual(trace['matches'][0]['packet']['id'], 'weisbord')
        trace = retriever.retrieve('Switch to quantum entanglement and Einstein.', state)
        self.assertEqual(trace['active_packet_ids'], [])

    def test_budget_drops_whole_source_and_stale_state_is_rejected(self):
        r = SourceStateRetriever(self.collection, self.ledger)
        state = initial_state(self.collection, self.ledger)
        trace = r.retrieve('Rakovsky hospital', state, max_chars=1)
        self.assertFalse(trace['matches']); self.assertIn('rakovsky', trace['budget_omitted'])
        state['active_packet_ids'] = ['invented']
        with self.assertRaises(EvidenceError):
            r.retrieve('Explain that.', state)

    def test_selection_cannot_create_claims_or_upgrade_metadata(self):
        available = {'rakovsky_limit': self.cards['rakovsky_limit']}
        for plan in [{'answerable': True, 'claim_ids': ['invented']},
                     {'answerable': True, 'claim_ids': ['rakovsky_limit'], 'status': 'fact'},
                     {'answerable': False, 'claim_ids': ['rakovsky_limit']}]:
            with self.subTest(plan=plan), self.assertRaises(EvidenceError):
                validate_plan(plan, available)

    def test_raw_citations_and_direct_quotes_cannot_escape_renderer(self):
        selected = {'hearing_occupation': self.cards['hearing_occupation']}
        for text in ['Author [eastman]', 'He said "Author".', 'He said ‘Author’.', "He said 'Author'.", 'He said 「Author」.']:
            with self.subTest(text=text), self.assertRaises(EvidenceError):
                validate_draft({'segments': [{'id': 's1', 'text': text, 'claim_ids': ['hearing_occupation']}]}, selected)
        valid = [{'id': 's1', 'quote': {'claim_id': 'hearing_occupation', 'anchor_index': 1}}]
        validate_draft({'segments': valid}, selected)
        rendered, used = render_segments(valid, {'s1'}, selected)
        self.assertIn('“Author.” [hearing]', rendered)
        self.assertNotIn('[eastman]', rendered)
        self.assertEqual(used, ['hearing_occupation'])

    def test_review_requires_complete_coverage_and_every_check(self):
        segments = [{'id': 's1'}, {'id': 's2'}]
        verdicts = [{'id': s['id'], 'reason': 'Synthetic test only', **{k: True for k in CHECKS}} for s in segments]
        self.assertEqual(validate_review({'verdicts': verdicts}, segments), {'s1', 's2'})
        verdicts[1]['no_extra_claims'] = False
        self.assertEqual(validate_review({'verdicts': verdicts}, segments), {'s1'})
        with self.assertRaises(EvidenceError):
            validate_review({'verdicts': verdicts[:1]}, segments)
        verdicts[0]['supported'] = 'true'
        with self.assertRaises(EvidenceError):
            validate_review({'verdicts': verdicts}, segments)

    def test_untrusted_duplicate_json_and_nonfinite_values_are_rejected(self):
        for text in ['{"answerable": true, "answerable": false}', '{"value":NaN}', '```json\n{}\n```']:
            with self.subTest(text=text), self.assertRaises(EvidenceError):
                parse_object(text)

    def test_end_to_end_filters_bad_addition_before_display_and_memory(self):
        stages = []
        def generate(messages):
            stages.append(messages)
            index = len(stages)
            if index == 1:
                out = {'answerable': True, 'claim_ids': ['hearing_occupation', 'hearing_record']}
            elif index == 2:
                out = {'segments': [
                    {'id': 's1', 'text': 'My recorded occupation was Author.', 'claim_ids': ['hearing_occupation']},
                    {'id': 's2', 'text': 'Goldman was trying to portray my house as a spy headquarters.', 'claim_ids': ['hearing_record']}]}
            else:
                out = {'verdicts': [
                    {'id': 's1', 'reason': 'Occupation is recorded', **{k: True for k in CHECKS}},
                    {'id': 's2', 'reason': 'Hostile motive not in source', **{k: False for k in CHECKS}}]}
            return {'response': json.dumps(out), 'finish_reason': 'stop'}
        result = EvidenceChat(self.collection, self.ledger).answer('In the hearing, what was your occupation?', generate)
        self.assertEqual(len(stages), 3)
        self.assertIn('Author. [hearing]', result['response'])
        self.assertNotIn('spy headquarters', result['response'])
        self.assertEqual(result['state_after']['accepted_claim_ids'], ['hearing_occupation'])
        self.assertEqual(result['status'], 'partial')
        self.assertFalse(result['semantic_guarantee'])

    def test_no_support_or_failed_stage_returns_scoped_reply_without_extra_calls(self):
        for response, finish in [('{"answerable":false,"claim_ids":[]}', 'stop'),
                                 ('{"answerable":true', 'length'), ('not json', 'stop')]:
            calls = []
            def generate(messages):
                calls.append(messages); return {'response': response, 'finish_reason': finish}
            result = EvidenceChat(self.collection, self.ledger).answer('What did Einstein tell you in the Weisbord meeting?', generate)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['state_after']['accepted_claim_ids'], [])
            self.assertNotIn('never happened', result['response'])
            self.assertNotIn('1947', result['response'])

    def test_network_errors_propagate_without_retry_or_false_abstention(self):
        def generate(_):
            raise ConnectionError('Synthetic network failure')
        with self.assertRaises(ConnectionError):
            EvidenceChat(self.collection, self.ledger).answer('Rakovsky hospital?', generate)

    def test_v1_and_adapter_hashes_unchanged(self):
        import hashlib
        saved = json.loads((ROOT/'evaluation/grounded_chat_v2/preserved_v1_and_adapters.json').read_text())
        for path, expected in saved.items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), expected, path)

    def test_comparison_replays_cached_calls_and_matches_sources(self):
        import tempfile
        from evaluate_grounded_chat_v2 import evaluate
        spec = {'singles': [{'id': 'test', 'question': 'Rakovsky hypothesis?'}], 'conversations': []}
        calls = []
        def generate(messages):
            calls.append(messages)
            out = 'Synthetic direct response' if len(calls) == 1 else '{"answerable":false,"claim_ids":[]}'
            return {'response': out, 'finish_reason': 'stop'}
        with tempfile.TemporaryDirectory() as temp:
            answers, records = evaluate(self.collection, self.ledger, spec, generate, Path(temp), {'test': True})
            self.assertEqual(len(calls), 2)
            self.assertEqual(answers[0]['retrieval'], answers[1]['retrieval'])
            self.assertEqual(answers[1]['status'], 'abstained')
            replay, _ = evaluate(self.collection, self.ledger, spec, generate, Path(temp), {'test': True})
            self.assertEqual(replay, answers); self.assertEqual(len(calls), 2)
            records[-1]['output']['response'] = 'changed'
            path = Path(temp)/'model_calls.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(ValueError):
                evaluate(self.collection, self.ledger, spec, generate, Path(temp), {'test': True})


if __name__ == '__main__':
    unittest.main()
