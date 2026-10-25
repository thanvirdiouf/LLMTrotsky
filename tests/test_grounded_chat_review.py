import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from grounded_chat import fingerprint
from review_grounded_chat import frozen_payloads, validate_scores


class GroundedChatReviewTest(unittest.TestCase):
    def test_frozen_bundle_has_no_answer_keys_or_training_data(self):
        payloads = frozen_payloads()
        protocol = json.loads(payloads['protocol.json'])
        self.assertEqual(protocol['answers'], 32)
        self.assertEqual(protocol['training_steps'], 0)
        self.assertEqual(protocol['adapters_loaded'], [])
        spec = json.loads(payloads['model_cases.json'])
        self.assertNotIn('expected', json.dumps(spec))
        self.assertNotIn('rubric', protocol)
        self.assertFalse(any('adapter' in name or 'train' in name for name in payloads))

    def test_score_binding_exact_spans_and_critical_zero(self):
        response = 'Synthetic hypothesis remains unverified.'
        row = {'key': 'conversation/test/1', 'condition': 'conversation', 'response': response,
               'response_sha256': fingerprint(response), 'request': {'retrieval': {'matches': [{'packet': {'id': 'test'}}]}}}
        collection = {'packets': [{'id': 'test', 'blocks': [{'block_id': 1, 'text': 'A hypothesis requires verification.'}]}]}
        rubric = {'critical_flags': ['certainty_upgrade']}
        score = {'key': row['key'], 'response_sha256': row['response_sha256'], 'notes': 'Synthetic fixture only',
                 'response_score': 4, 'conversation_score': 4, 'retrieval_sufficient': 'yes', 'interpretation_score': 4,
                 'critical_flags': [], 'failure_stages': [], 'response_spans': ['remains unverified'],
                 'source_spans': [{'packet_id': 'test', 'block_id': 1, 'text': 'requires verification', 'in_current_retrieval': True}]}
        self.assertEqual(validate_scores([score], [row], collection, rubric)['critical_cases'], 0)
        for field, value in [('response_sha256', 'changed'), ('critical_flags', ['certainty_upgrade']),
                             ('response_spans', ['invented wording']), ('response_spans', ['']), ('retrieval_sufficient', 'no')]:
            bad = copy.deepcopy(score); bad[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_scores([bad], [row], collection, rubric)
        bad = copy.deepcopy(score); bad['source_spans'][0]['in_current_retrieval'] = False
        with self.assertRaises(ValueError):
            validate_scores([bad], [row], collection, rubric)


if __name__ == '__main__':
    unittest.main()
