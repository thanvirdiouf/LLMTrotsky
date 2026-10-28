import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from conversation_retrieval import ConversationRetriever
from grounded_request_boundary import ConversationState, PriorTurn
from review_chat import prepare_turn
import test_review_chat as fixtures


class ConversationRetrievalTests(unittest.TestCase):
    def fixture(self):
        return fixtures.ReviewChatTests().fixture()

    def test_implicit_followup_uses_human_context_not_assistant_inventions(self):
        p, c, l = self.fixture()
        history = (PriorTurn('user', 'solar wage control stellar union bargaining'),
                   PriorTurn('assistant', 'river cloud violet inventedneutron'))
        state = ConversationState(('h', 'm'), prior_turns=history)
        question = 'Would accepting the proposed arrangement settle every remaining objection?'
        r = prepare_turn(p, c, l, state, question, retrieval_policy='human_context_decay_v1')
        self.assertEqual(r['source_packets'][0]['id'], 'h')
        for word in ('river', 'cloud', 'violet', 'inventedneutron'):
            self.assertNotIn(word, r['retrieval']['query_terms'])
        self.assertEqual(r['messages'][-1]['content'], question)
        self.assertFalse(r['quality_validated'])

    def test_topic_reset_removes_earlier_context_including_on_later_turn(self):
        p, c, l = self.fixture()
        prior = [PriorTurn('user', 'solar wage control stellar union bargaining'), PriorTurn('assistant', 'ignored')]
        state = ConversationState(('h', 'm'), prior_turns=tuple(prior))
        r = prepare_turn(p, c, l, state, 'New topic: river cloud violet', retrieval_policy='human_context_decay_v1')
        self.assertEqual(r['retrieval']['context_contributions'], [])
        prior += [PriorTurn('user', 'New topic: river cloud violet'), PriorTurn('assistant', 'ignored')]
        r = prepare_turn(p, c, l, ConversationState(('h', 'm'), prior_turns=tuple(prior)),
                         'Could the arrangement still change?', retrieval_policy='human_context_decay_v1')
        self.assertIn('river', r['retrieval']['query_terms'])
        self.assertNotIn('solar', r['retrieval']['query_terms'])

    def test_scope_and_complete_packet_budget_still_apply(self):
        p, c, l = self.fixture()
        history = (PriorTurn('user', 'solar wage control stellar union bargaining'), PriorTurn('assistant', 'ignored'))
        r = prepare_turn(p, c, l, ConversationState(('m',), prior_turns=history), 'river cloud violet',
                         retrieval_policy='human_context_decay_v1', tokenizer=fixtures.Tokenizer())
        self.assertEqual([x['id'] for x in r['source_packets']], ['m'])
        self.assertEqual(r['source_packets'][0], c['packets'][1])
        class Budget:
            def apply_chat_template(self, messages, **kwargs):
                return [1]*(6000 if messages[0]['content'].count('"context_boundary"') > 1 else 100)
        r = prepare_turn(p, c, l, ConversationState(('h', 'm'), prior_turns=history), 'river cloud violet',
                         retrieval_policy='human_context_decay_v1', tokenizer=Budget())
        self.assertEqual(len(r['source_packets']), 1)
        self.assertEqual(len(r['token_budget_omitted_packets']), 1)

    def test_prior_context_is_bounded_and_current_question_never_rewritten(self):
        p, c, l = self.fixture()
        prior = [{'role': 'user', 'content': 'obsoleteunique'}]
        prior += [{'role': 'user', 'content': 'solar '+str(i)} for i in range(5)]
        r = ConversationRetriever(c).retrieve('river cloud violet', prior)
        self.assertEqual(len(r['context_contributions']), 4)
        self.assertNotIn('obsoleteunique', r['query_terms'])
        with self.assertRaises(ValueError):
            prepare_turn(p, c, l, ConversationState(('h',)), 'question', retrieval_policy='unknown')


if __name__ == '__main__':
    unittest.main()
