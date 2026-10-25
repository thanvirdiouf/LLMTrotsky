import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_grounded_chat import build
from grounded_chat import Retriever, make_request, response_diagnostics, load_collection
from evaluate_grounded_chat import model_spec, evaluate


class GroundedChatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collection = build()
        cls.spec = json.loads((ROOT/'config/grounded_chat_evaluation.json').read_text())

    def setUp(self):
        self.retriever = Retriever(self.collection)

    def test_context_keeps_hypothesis_and_address_signature_together(self):
        for q, pid, phrase in [('Rakovsky hospital refusal proof', 'rakovsky', 'hypothesis which requires verification'),
                               ('Weisbord New York signature', 'weisbord', 'Kadikoy, October 10, 1931')]:
            matches = self.retriever.retrieve(q)['matches']
            packet = next(m['packet'] for m in matches if m['packet']['id'] == pid)
            self.assertIn(phrase, '\n'.join(b['text'] for b in packet['blocks']))
        hearing = next(p for p in self.collection['packets'] if p['id'] == 'hearing')
        b = {b['block_id']: b for b in hearing['blocks']}
        self.assertEqual(b[105]['speaker'], 'GOLDMAN')
        self.assertEqual(b[104]['speaker'], 'TROTSKY')
        self.assertIn('1897, I should have said', b[104]['text'])

    def test_no_hit_and_budget_drop_do_not_crop_context(self):
        self.assertEqual(self.retriever.retrieve('zygomorphic xylophone')['status'], 'insufficient_evidence')
        trace = self.retriever.retrieve('Rakovsky hospital', max_chars=1)
        self.assertEqual(trace['matches'], [])
        self.assertIn('rakovsky', trace['budget_omitted'])

    def test_followup_uses_human_context_and_topic_switch(self):
        history = [{'role': 'user', 'content': 'What about Rakovsky and the hospital?'},
                   {'role': 'assistant', 'content': 'Fabricated Atlantis private memory'}]
        r = self.retriever.retrieve('Explain that more fully', history)
        self.assertIn('rakovsky', [m['packet']['id'] for m in r['matches']])
        self.assertNotIn('atlantis', r['query_terms'])
        switch = self.retriever.retrieve('Where was the Weisbord reply signed and whose address was New York?', history)
        self.assertEqual(switch['matches'][0]['packet']['id'], 'weisbord')

    def test_reference_answers_never_enter_model_cases(self):
        spec = model_spec(self.spec)
        self.assertNotIn('expected', json.dumps(spec))
        self.assertNotIn('rubric', spec)
        q = spec['singles'][0]['question']
        req = make_request(self.collection, self.retriever, q)
        self.assertEqual(req['messages'][-1], {'role': 'user', 'content': q})
        self.assertNotIn('oracle_packet_ids', json.dumps(req['messages']))

    def test_actual_replies_flow_forward_and_resume_rejects_tampering(self):
        spec = {'singles': [], 'conversations': [{'id': 'test', 'turns': [{'question': 'Rakovsky?'}, {'question': 'Why that hypothesis?'}]}]}
        calls = []
        def generate(messages):
            calls.append(messages)
            return {'response': 'Synthetic backend reply [rakovsky]', 'finish_reason': 'stop'}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'responses.jsonl'
            rows = evaluate(self.collection, spec, generate, path, {'test_identity': True})
            self.assertEqual(calls[1][2]['content'], rows[0]['response'])
            evaluate(self.collection, spec, generate, path, {'test_identity': True})
            self.assertEqual(len(calls), 2)
            rows[0]['response'] = 'tampered'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaises(ValueError):
                evaluate(self.collection, spec, generate, path, {'test_identity': True})

    def test_truncation_never_becomes_conversation_history(self):
        spec = {'singles': [], 'conversations': [{'id': 'test', 'turns': [{'question': 'Rakovsky?'}, {'question': 'Why?'}]}]}
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                evaluate(self.collection, spec, lambda _: {'response': 'cut off', 'finish_reason': 'length'}, Path(temp)/'r.jsonl', {})
            self.assertEqual(len((Path(temp)/'r.jsonl').read_text().splitlines()), 1)

    def test_collection_and_citations_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            bad = copy.deepcopy(self.collection)
            bad['packets'][0]['blocks'][0]['text'] += 'tampered'
            path = Path(temp)/'collection.json'; path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                load_collection(path)
        trace = self.retriever.retrieve('Rakovsky hospital')
        self.assertEqual(response_diagnostics('Alleged [invented]', trace)['unknown_citation_ids'], ['invented'])


if __name__ == '__main__':
    unittest.main()
