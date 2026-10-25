"""Synthetic v2 reviews must be complete and response-bound before unblinding."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import summarize_data_revision_review as summarizer


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_jsonl(path, values):
    path.write_text(''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values),
                    encoding='utf-8')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


class DataRevisionSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'results'
        self.review = self.run / 'review'
        self.review.mkdir(parents=True)
        self.sources, self.keys, self.scores = [], [], {1: [], 2: [], 3: []}
        self.case_ids = [f'case_{index:02d}' for index in range(24)]
        for index, case_id in enumerate(self.case_ids):
            category = list(summarizer.DIM)[index // 6]
            control = ('positive', 'negative', 'conditional', 'positive', 'negative', 'positive')[index % 6]
            shard = index // 8 + 1
            for number, condition in enumerate(summarizer.CONDITIONS):
                review_id = f'R{len(self.sources) + 1:03d}'
                response = f'Synthetic answer {number} to {case_id}: dated evidence — not a quotation.'
                source = {'review_id': review_id, 'case_id': case_id, 'category': category,
                          'control_type': control, 'response': response,
                          'token_count': 80 + number * 10 + index, 'finish_reason': 'stop',
                          **dict.fromkeys(summarizer.SCORE_FIELDS, '')}
                if condition == 'revised_step_50' and index == 2:
                    source.update(token_count=512, finish_reason='length')
                key = {'review_id': review_id, 'case_id': case_id, 'category': category,
                       'control_type': control, 'condition': condition, 'shard': shard,
                       'identity_sha256': 'a' * 64, 'response_sha256': digest(response.encode()),
                       'prompt_sha256': digest(case_id.encode())}
                main_score = number + 1
                if (condition == 'original_step_50' and index == 0
                        or condition == 'revised_step_50' and index == 1):
                    main_score = 0
                critical = int(index in {'fresh_base': {3}, 'original_step_50': {0, 2},
                                         'revised_step_50': {1, 2}}[condition])
                score = {'review_id': review_id, 'case_id': case_id, 'category': category,
                         **dict.fromkeys(summarizer.DIM.values(), None),
                         summarizer.DIM[category]: main_score, 'critical_failure': critical,
                         'critical_failure_types': ['invented_fact'] if critical else [],
                         'review_notes': 'Fixture evidence: the response asserts a dated fact.',
                         'response_sha256': key['response_sha256']}
                self.sources.append(source)
                self.keys.append(key)
                self.scores[shard].append(score)
        self.write_inputs()

    def write_inputs(self):
        write_jsonl(self.review / 'review_rows.jsonl', self.sources)
        write_jsonl(self.review / 'review_key.jsonl', self.keys)
        for number, values in self.scores.items():
            write_jsonl(self.review / f'agent_scores_shard_{number:02d}.jsonl', values)
        self.refresh_manifest()

    def refresh_manifest(self):
        manifest = {'schema_version': 2, 'identity_sha256': 'a' * 64,
                    'source_sha256': {'run_manifest.json': 'b' * 64},
                    'output_sha256': {name: digest((self.review / name).read_bytes())
                                      for name in ('review_rows.jsonl', 'review_key.jsonl')}}
        (self.review / 'review_manifest.json').write_text(json.dumps(manifest) + '\n')

    def assert_rejected_without_outputs(self, regex=None):
        with self.assertRaisesRegex(ValueError, regex or '.'):
            summarizer.build(self.run)
        self.assertFalse((self.root / 'quality_summary.json').exists())
        self.assertFalse((self.review / 'agent_scores_unblinded.jsonl').exists())

    def test_explicit_reviewer_count_changes_method_without_changing_results(self):
        summary = summarizer.build(self.run, reviewer_count=3)
        self.assertEqual(summary['reviewer_count'], 3)
        self.assertIn('3 coding-agent reviewers;', summary['review_method'])
        self.assertEqual(summary['models']['revised_step_50']['critical_failures'], 2)

    def test_invalid_reviewer_counts_rejected_before_outputs(self):
        for count in (0, -1, True, 1.5, '3'):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    summarizer.build(self.run, reviewer_count=count)
                self.assertFalse((self.root / 'quality_summary.json').exists())
                self.assertFalse((self.review / 'agent_scores_unblinded.jsonl').exists())

    def test_complete_review_produces_bound_category_and_control_diagnostics(self):
        before = {path.name: path.read_bytes() for path in self.review.iterdir()}
        summary = summarizer.build(self.run)
        serialized = json.loads((self.root / 'quality_summary.json').read_text())
        self.assertEqual(serialized['schema_version'], 2)
        self.assertEqual(serialized['reviewer_count'], 1)
        self.assertIn('one coding-agent reviewer;', serialized['review_method'])
        self.assertEqual(set(summary['models']), set(summarizer.CONDITIONS))
        revised = summary['models']['revised_step_50']
        self.assertEqual(revised['cases'], 24)
        self.assertEqual(revised['critical_failures'], 2)
        self.assertEqual(revised['truncations'], 1)
        self.assertEqual(revised['critical_failure_type_counts'], {'invented_fact': 2})
        self.assertEqual(revised['category_scores']['chronology']['mean_main_score_0_4'], 2.5)
        self.assertEqual(revised['category_scores']['chronology']['scores'], {0: 1, 3: 5})
        self.assertEqual(set(revised['control_diagnostics']), set(summarizer.CONTROLS))
        self.assertEqual({control: result['cases']
                          for control, result in revised['control_diagnostics'].items()},
                         {'positive': 12, 'negative': 8, 'conditional': 4})
        self.assertEqual(revised['control_diagnostics']['negative']['critical_failures'], 1)
        self.assertEqual(revised['control_diagnostics']['conditional']['truncations'], 1)
        self.assertNotIn('mean_main_score_0_4', revised)
        self.assertNotIn('mean_main_score_0_4', revised['control_diagnostics']['positive'])
        self.assertEqual(revised['mean_output_tokens'], round(
            sum(row['token_count'] for row, key in zip(self.sources, self.keys)
                if key['condition'] == 'revised_step_50') / 24, 2))
        comparison = summary['paired']['original_step_50_vs_revised_step_50']
        self.assertEqual((comparison['score_improved'], comparison['score_degraded'],
                          comparison['score_tied']), (23, 1, 0))
        self.assertEqual(comparison['new_critical_cases'], ['case_01'])
        self.assertEqual(comparison['removed_critical_cases'], ['case_00'])
        self.assertEqual(comparison['persistent_critical_cases'], ['case_02'])
        self.assertEqual(comparison['category_scores']['chronology']['mean_main_score_delta_0_4'],
                         0.833)
        self.assertEqual(comparison['control_diagnostics']['negative']['new_critical_cases'], ['case_01'])
        self.assertEqual(set(summary['paired']), {'original_step_50_vs_revised_step_50',
                         'fresh_base_vs_revised_step_50', 'fresh_base_vs_original_step_50'})
        self.assertEqual(summary['identity_sha256'], 'a' * 64)
        self.assertEqual(summary['source_artifact_sha256'], {'run_manifest.json': 'b' * 64})
        self.assertEqual(len(summary['score_source_sha256']), 3)
        for name, payload in before.items():
            self.assertEqual((self.review / name).read_bytes(), payload)
            self.assertEqual(summary['review_source_sha256'][name], digest(payload))
        unblinded = read_jsonl(self.review / 'agent_scores_unblinded.jsonl')
        self.assertEqual(len(unblinded), 72)
        self.assertEqual([row['review_id'] for row in unblinded], sorted(row['review_id'] for row in unblinded))
        keyed = {row['review_id']: row for row in self.keys}
        for row in unblinded:
            self.assertEqual(row['condition'], keyed[row['review_id']]['condition'])
            self.assertEqual(row['control_type'], keyed[row['review_id']]['control_type'])
            self.assertEqual(row['main_score'], row[summarizer.DIM[row['category']]])
        limitations = ' '.join(summary['limitations'])
        for phrase in ('historian', '24 diagnostic', 'token exposure', 'global quality rating'):
            self.assertIn(phrase, limitations)

    def test_score_response_hash_malformed_or_altered_is_rejected(self):
        for value in ('d' * 64, 'not-a-hash', 'A' * 64, None):
            with self.subTest(value=value):
                original = self.scores[1][0]['response_sha256']
                self.scores[1][0]['response_sha256'] = value
                self.write_inputs()
                self.assert_rejected_without_outputs('hash mismatch')
                self.scores[1][0]['response_sha256'] = original

    def test_key_response_hash_and_changed_response_are_rejected(self):
        original = deepcopy(self.keys)
        self.keys[0]['response_sha256'] = 'd' * 64
        self.write_inputs()
        self.assert_rejected_without_outputs('hash mismatch')
        self.keys = original
        self.sources[0]['response'] += ' Altered after review.'
        self.write_inputs()
        self.assert_rejected_without_outputs('hash mismatch')

    def test_unknown_review_id_case_category_or_condition_is_rejected(self):
        original_scores, original_keys = deepcopy(self.scores), deepcopy(self.keys)
        for changes in ({'review_id': 'unknown'}, {'case_id': 'another_case'},
                        {'category': 'unknown'}, {'category': 'false_premise'}):
            with self.subTest(changes=changes):
                self.scores = deepcopy(original_scores)
                self.scores[1][0].update(changes)
                self.write_inputs()
                self.assert_rejected_without_outputs()
        self.scores = original_scores
        for changes in ({'condition': 'step_500'}, {'case_id': 'another_case'},
                        {'category': 'false_premise'}, {'control_type': 'negative'}):
            with self.subTest(key_changes=changes):
                self.keys = deepcopy(original_keys)
                self.keys[0].update(changes)
                self.write_inputs()
                self.assert_rejected_without_outputs()

    def test_duplicate_source_key_and_score_rows_are_rejected(self):
        sources, keys, scores = deepcopy(self.sources), deepcopy(self.keys), deepcopy(self.scores)
        for target in ('sources', 'keys', 'scores'):
            with self.subTest(target=target):
                self.sources, self.keys, self.scores = deepcopy(sources), deepcopy(keys), deepcopy(scores)
                values = self.scores[1] if target == 'scores' else getattr(self, target)
                values[1] = deepcopy(values[0])
                self.write_inputs()
                self.assert_rejected_without_outputs('duplicate')
        self.sources, self.keys, self.scores = sources, keys, scores
        # A duplicate split across files is not hidden by per-shard uniqueness.
        self.scores[2][0] = deepcopy(self.scores[1][0])
        self.write_inputs()
        self.assert_rejected_without_outputs('Duplicate')

    def test_missing_rows_and_unbalanced_score_shards_are_rejected(self):
        sources, keys, scores = deepcopy(self.sources), deepcopy(self.keys), deepcopy(self.scores)
        for target in ('sources', 'keys', 'scores'):
            with self.subTest(target=target):
                self.sources, self.keys, self.scores = deepcopy(sources), deepcopy(keys), deepcopy(scores)
                values = self.scores[1] if target == 'scores' else getattr(self, target)
                values.pop()
                self.write_inputs()
                self.assert_rejected_without_outputs('Expected exactly')
        self.sources, self.keys, self.scores = sources, keys, scores
        self.scores[2].append(self.scores[1].pop())
        self.write_inputs()
        self.assert_rejected_without_outputs('Expected exactly 24')

    def test_invalid_dimension_scores_and_missing_main_score_are_rejected(self):
        original = deepcopy(self.scores)
        for changes in ({'chronology': True}, {'chronology': -1}, {'chronology': 5},
                        {'chronology': 2.0}, {'chronology': '2'}, {'chronology': None},
                        {'correction': ''}, {'answer_detail': False}):
            with self.subTest(changes=changes):
                self.scores = deepcopy(original)
                self.scores[1][0].update(changes)
                self.write_inputs()
                self.assert_rejected_without_outputs('score')
        self.scores = original
        del self.scores[1][0]['answer_detail']
        self.write_inputs()
        self.assert_rejected_without_outputs('Invalid dimension')

    def test_invalid_flags_failure_types_and_evidence_notes_are_rejected(self):
        original = deepcopy(self.scores)
        mutations = [{'critical_failure': value} for value in (True, False, 2, '0', None)]
        mutations += [{'review_notes': value} for value in ('', '  \n', None, ['note'])]
        mutations += [{'critical_failure_types': value}
                      for value in ('invented_fact', None, [''], ['invented_fact'], [{}])]
        mutations += [{'critical_failure': 1, 'critical_failure_types': []},
                      {'critical_failure': 1, 'critical_failure_types': ['same', 'same']}]
        for changes in mutations:
            with self.subTest(changes=changes):
                self.scores = deepcopy(original)
                self.scores[1][0].update(changes)
                self.write_inputs()
                self.assert_rejected_without_outputs()

    def test_category_control_and_condition_case_coverage_are_consistent(self):
        sources, keys, scores = deepcopy(self.sources), deepcopy(self.keys), deepcopy(self.scores)
        # Matching score/key labels cannot make a case's source category vary by condition.
        self.sources[0]['category'] = self.keys[0]['category'] = self.scores[1][0]['category'] = 'false_premise'
        self.scores[1][0]['correction'] = 2
        self.write_inputs()
        self.assert_rejected_without_outputs('differs across conditions')
        self.sources, self.keys, self.scores = deepcopy(sources), deepcopy(keys), deepcopy(scores)
        self.sources[0]['control_type'] = self.keys[0]['control_type'] = 'conditional'
        self.write_inputs()
        self.assert_rejected_without_outputs('differs across conditions')
        self.sources, self.keys, self.scores = deepcopy(sources), deepcopy(keys), deepcopy(scores)
        # All three answers consistently relabeled still must retain six cases per category.
        for index in range(3):
            self.sources[index]['category'] = self.keys[index]['category'] = self.scores[1][index]['category'] = 'false_premise'
            self.scores[1][index]['correction'] = 2
        self.write_inputs()
        self.assert_rejected_without_outputs('six per category')
        self.sources, self.keys, self.scores = sources, keys, scores
        self.keys[0]['condition'] = 'original_step_50'
        self.write_inputs()
        self.assert_rejected_without_outputs('Duplicate case')

    def test_invalid_source_control_token_finish_or_nonblank_scores_are_rejected(self):
        original = deepcopy(self.sources)
        for changes in ({'control_type': 'unknown'}, {'token_count': True}, {'token_count': -1},
                        {'finish_reason': 'error'}, {'chronology': 1}, {'condition': 'fresh_base'}):
            with self.subTest(changes=changes):
                self.sources = deepcopy(original)
                self.sources[0].update(changes)
                self.write_inputs()
                self.assert_rejected_without_outputs()

    def test_key_identity_and_prompt_hash_evidence_are_validated(self):
        original = deepcopy(self.keys)
        for changes in ({'identity_sha256': 'b' * 64}, {'identity_sha256': 'bad'},
                        {'prompt_sha256': 'bad'}, {'prompt_sha256': 'c' * 64}):
            with self.subTest(changes=changes):
                self.keys = deepcopy(original)
                self.keys[0].update(changes)
                self.write_inputs()
                self.assert_rejected_without_outputs()
        for field in ('identity_sha256', 'prompt_sha256'):
            with self.subTest(missing=field):
                self.keys = deepcopy(original)
                del self.keys[0][field]
                self.write_inputs()
                self.assert_rejected_without_outputs('Incomplete key')

    def test_optional_key_identity_can_be_absent_everywhere(self):
        for row in self.keys:
            del row['identity_sha256']
            del row['prompt_sha256']
        self.write_inputs()
        manifest = json.loads((self.review / 'review_manifest.json').read_text())
        del manifest['identity_sha256']
        (self.review / 'review_manifest.json').write_text(json.dumps(manifest))
        summary = summarizer.build(self.run)
        self.assertIsNone(summary['identity_sha256'])

    def test_score_shards_must_match_key_and_keep_each_case_together(self):
        self.scores[1][0], self.scores[2][0] = self.scores[2][0], self.scores[1][0]
        self.write_inputs()
        self.assert_rejected_without_outputs('wrong shard')
        for key in self.keys:
            del key['shard']
        self.write_inputs()
        self.assert_rejected_without_outputs('same reviewer')

    def test_review_manifest_binds_inputs_and_run_identity(self):
        path = self.review / 'review_manifest.json'
        original = json.loads(path.read_text())
        for changes in ({'identity_sha256': 'c' * 64},
                        {'source_sha256': {'run_manifest.json': 'bad'}},
                        {'output_sha256': {'review_rows.jsonl': 'c' * 64}},
                        {'output_sha256': {'../outside.json': 'c' * 64}}):
            with self.subTest(changes=changes):
                manifest = deepcopy(original)
                manifest.update(changes)
                path.write_text(json.dumps(manifest))
                self.assert_rejected_without_outputs()
        path.write_text(json.dumps(original))
        with (self.review / 'review_rows.jsonl').open('a') as stream:
            stream.write('\n')
        self.assert_rejected_without_outputs()

    def test_default_outputs_are_never_overwritten_and_new_paths_are_supported(self):
        summary_path = self.root / 'quality_summary.json'
        unblinded_path = self.review / 'agent_scores_unblinded.jsonl'
        for path, other in ((summary_path, unblinded_path), (unblinded_path, summary_path)):
            with self.subTest(existing=path.name):
                path.write_text('preserve this output\n')
                with self.assertRaises(FileExistsError):
                    summarizer.build(self.run)
                self.assertEqual(path.read_text(), 'preserve this output\n')
                self.assertFalse(other.exists())
                path.unlink()
        first = summarizer.build(self.run)
        outputs = {path: path.read_bytes() for path in (summary_path, unblinded_path)}
        with self.assertRaises(FileExistsError):
            summarizer.build(self.run)
        alternate_summary = self.root / 'alternate' / 'quality.json'
        alternate_unblinded = self.root / 'alternate' / 'scores.jsonl'
        second = summarizer.build(self.run, summary_path=alternate_summary,
                                  unblinded_path=alternate_unblinded)
        self.assertEqual(first, second)
        self.assertTrue(alternate_summary.exists())
        self.assertTrue(alternate_unblinded.exists())
        for path, payload in outputs.items():
            self.assertEqual(path.read_bytes(), payload)
        with self.assertRaises(FileExistsError):
            summarizer.build(self.run, summary_path=alternate_summary,
                             unblinded_path=alternate_unblinded)

    def test_output_paths_must_differ(self):
        target = self.root / 'same-output'
        with self.assertRaisesRegex(ValueError, 'paths must differ'):
            summarizer.build(self.run, summary_path=target, unblinded_path=target)
        self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
