"""Frozen pilot order and evaluation separation checks; no model packages needed."""
import copy
import json
from pathlib import Path
import random
import unittest

from scripts import pilot_protocol as protocol
from scripts import kaggle_pilot as runtime

PROJECT = Path(__file__).resolve().parents[1]


class PilotProtocolTests(unittest.TestCase):
    def test_frozen_protocol_is_current_and_all500_inputs_are_valid(self):
        frozen = json.loads((PROJECT / 'training/pilot_protocol.json').read_text())
        self.assertEqual(protocol.validate_protocol(PROJECT, frozen), frozen)
        lock = json.loads((PROJECT / 'training/pilot_lock.json').read_text())
        train, evaluation = protocol.validate_protocol_payloads(
            (PROJECT / 'datasets/train_seed.jsonl').read_bytes(),
            (PROJECT / 'training/pilot_evaluation_inputs.jsonl').read_bytes(), lock, frozen)
        self.assertEqual(len(train), 500)
        self.assertEqual(len(evaluation), 36)
        self.assertTrue(all(set(row) == {'case_id', 'category', 'split', 'messages'} for row in evaluation))
        self.assertEqual(frozen['evaluation']['generation']['eos_token_id'], 248046)

    def test_order_uses_each_example_once_longest_first_with_stable_shuffle(self):
        encoded = json.loads((PROJECT / protocol.SUCCESS_RUN / 'tokenization_report.json').read_text())['examples']
        state = random.getstate()
        order = protocol.build_order(encoded)
        self.assertEqual(state, random.getstate())
        self.assertEqual(order, protocol.build_order(encoded))
        self.assertEqual(sorted(order), list(range(500)))
        self.assertEqual(encoded[order[0]]['token_count'], 457)
        self.assertNotEqual(order[1:], sorted(order[1:]))
        self.assertNotEqual(order, protocol.build_order(encoded, seed=43))

    def test_order_rejects_overflow_duplicate_ids_and_empty_labels_policy(self):
        examples = [{'example_id': 'a', 'token_count': 20, 'dropped': False},
                    {'example_id': 'b', 'token_count': 10, 'dropped': False}]
        for changed in [[{**examples[0], 'dropped': True}, examples[1]],
                        [examples[0], {**examples[1], 'example_id': 'a'}],
                        [{**examples[0], 'token_count': 0}]]:
            with self.assertRaises(ValueError):
                protocol.build_order(changed)

    def test_settings_drift_and_payload_changes_are_rejected(self):
        for update in [{'steps': 2}, {'epochs': 2}, {'run_evaluation': False}, {'amp_initial_scale': 65536.0},
                       {'evaluation_eos_token_id': 248044}, {'micro_batch_size': 2}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                protocol.validate_settings({**protocol.DEFAULT_SETTINGS, **update})
        frozen = json.loads((PROJECT / 'training/pilot_protocol.json').read_text())
        payload = (PROJECT / 'training/pilot_evaluation_inputs.jsonl').read_bytes()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            protocol.validate_evaluation_payload(payload + b'\n', frozen)
        rows = [json.loads(line) for line in payload.splitlines()]
        rows[0]['reference_evidence'] = 'must not reach the model'
        forged = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        altered_protocol = copy.deepcopy(frozen)
        altered_protocol['evaluation']['sha256'] = protocol.sha256_bytes(forged)
        with self.assertRaisesRegex(ValueError, 'only case_id'):
            protocol.validate_evaluation_payload(forged, altered_protocol)

    def test_evaluation_journal_rejects_wrong_prompts_settings_or_duplicate_cases(self):
        import tempfile
        cases = [{'case_id': 'one', 'messages': [{'role': 'user', 'content': 'question'}]}]
        generation = {'max_new_tokens': 512, 'eos_token_id': 248046}
        valid = {'case_id': 'one', 'condition': 'baseline', 'identity_sha256': 'run',
                 'prompt_sha256': runtime.fingerprint(cases[0]['messages']),
                 'generation_sha256': runtime.fingerprint(generation), 'response': 'answer',
                 'token_ids': [1, 248046], 'token_count': 2, 'finish_reason': 'stop'}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'responses.jsonl'
            path.write_text(json.dumps(valid) + '\n')
            self.assertEqual(runtime.read_evaluation_journal(path, 'run', 'baseline', cases, generation), [valid])
            for invalid in [{**valid, 'identity_sha256': 'different'},
                            {**valid, 'prompt_sha256': 'wrong'}, {**valid, 'token_count': 99}]:
                path.write_text(json.dumps(invalid) + '\n')
                with self.assertRaises(ValueError):
                    runtime.read_evaluation_journal(path, 'run', 'baseline', cases, generation)
            path.write_text((json.dumps(valid) + '\n') * 2)
            with self.assertRaises(ValueError):
                runtime.read_evaluation_journal(path, 'run', 'baseline', cases, generation)

    def test_local_gpu_work_is_blocked(self):
        if not Path('/kaggle/working').is_dir():
            with self.assertRaisesRegex(RuntimeError, 'only on Kaggle'):
                runtime.run_pilot('/missing', '/missing')


if __name__ == '__main__':
    unittest.main()
