"""Guard frozen fresh source families before seed export, generation or staging."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.fresh_holdout_guard import (check_training_records,
    guard_training_project, load_fresh_holdout_guard)
from scripts.build_seed_dataset import compile_seed
from scripts.remote_model import generation_sources
from scripts import build_kaggle_pilot, build_kaggle_training_smoke, stage_training_pilot


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value) + '\n' for value in values))


class FreshHoldoutGuardTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.project = Path(temp.name)
        self.catalog = {key: {'document_id': key, 'source_family_id': family,
                        'source_sha256': 'raw', 'url': 'https://source.test/' + key}
                        for key, family in [('held', 'reserved'), ('sibling', 'reserved'),
                                            ('allowed', 'train'), ('second', 'other_reserved')]}
        write_rows(self.project / 'corpus_mia/catalog.jsonl', self.catalog.values())
        self.freeze('v1', 'held')

    def freeze(self, version, key, **updates):
        directory = self.project / ('evaluation/fresh_holdout_' + version)
        row = {**self.catalog[key], 'training_prohibited': True, **updates}
        path = directory / 'source_exclusions.jsonl'
        write_rows(path, [row])
        write_json(directory / 'protocol.json', {'files': {
            'source_exclusions.jsonl': hashlib.sha256(path.read_bytes()).hexdigest()}})

    def record(self, key):
        return {'source_document_id': key, 'source_family_id': self.catalog[key]['source_family_id'],
                'source_url': self.catalog[key]['url']}

    def test_all_frozen_holdouts_and_whole_families_are_blocked(self):
        self.freeze('v2', 'second')
        guard = load_fresh_holdout_guard(self.project, self.catalog)
        self.assertEqual(len(guard.manifests), 2)
        for key in ['held', 'sibling', 'second']:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Fresh evaluation holdout'):
                check_training_records([self.record(key)], self.catalog, guard)
        check_training_records([self.record('allowed')], self.catalog, guard)

    def test_stale_exclusion_family_or_url_cannot_silently_reassign_holdout(self):
        for updates in [{'source_family_id': 'old_family'}, {'url': 'https://wrong.test'}]:
            self.freeze('v1', 'held', **updates)
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, 'provenance is stale'):
                load_fresh_holdout_guard(self.project, self.catalog)

    def test_corruption_and_incomplete_freeze_fail_closed(self):
        path = self.project / 'evaluation/fresh_holdout_v1/source_exclusions.jsonl'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_fresh_holdout_guard(self.project, self.catalog)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete frozen'):
            load_fresh_holdout_guard(self.project, self.catalog)

    def test_draft_without_protocol_is_not_yet_frozen(self):
        write_json(self.project / 'evaluation/fresh_holdout_draft/specifications.json', {'draft': True})
        guard = load_fresh_holdout_guard(self.project, self.catalog)
        self.assertEqual(len(guard.manifests), 1)

    def test_record_cannot_evade_family_guard_using_stale_metadata(self):
        guard = load_fresh_holdout_guard(self.project, self.catalog)
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            check_training_records([{**self.record('sibling'), 'source_family_id': 'train'}], self.catalog, guard)

    def test_seed_compile_rejects_holdout_before_writing_dataset(self):
        doc = {**self.catalog['held'], 'blocks': [{'block_id': 0, 'role': 'body_candidate',
                'text': 'The source contains the reviewed answer.'}]}
        write_rows(self.project / 'corpus_mia/documents.jsonl', [doc])
        write_rows(self.project / 'corpus_mia/catalog.jsonl', [self.catalog['held']])
        write_rows(self.project / 'corpus_mia/qa_candidates.jsonl', [])
        write_json(self.project / 'config/persona.json', {'system_prompt': 'Persona'})
        write_json(self.project / 'reviews/evaluation_specifications.json', {'source_backed': []})
        write_json(self.project / 'reviews/seed_specifications.json', {'native_dialogue': [],
            'adapted_passages': [{'source_suffix': '/held', 'block_ids': [0],
                                 'question': 'Question?', 'answer': 'Reviewed answer.'}]})
        with self.assertRaisesRegex(ValueError, 'Fresh evaluation holdout'):
            compile_seed(self.project)
        self.assertFalse((self.project / 'datasets/train_seed.jsonl').exists())

    def test_generation_rejects_fresh_holdout_despite_training_assignment(self):
        guard = load_fresh_holdout_guard(self.project, self.catalog)
        source = {**self.record('sibling'), 'approved_for_seed_training': True}
        with self.assertRaisesRegex(ValueError, 'Fresh evaluation holdout'):
            generation_sources([source], {'sibling': {'split': 'train'}}, '1940-08-19',
                               {}, self.catalog, set(), set(), guard)

    def test_notebook_staging_rejects_before_old_lock_validation(self):
        write_rows(self.project / 'datasets/seed_records.jsonl', [self.record('sibling')])
        for builder in [build_kaggle_training_smoke, build_kaggle_pilot]:
            with self.subTest(builder=builder.__name__), patch.object(builder, 'validate_lock') as validate:
                with self.assertRaisesRegex(ValueError, 'Fresh evaluation holdout'):
                    builder.make_notebook(self.project)
                validate.assert_not_called()

    def test_new_training_lock_staging_blocks_holdout_and_preserves_old_lock(self):
        records_path = self.project / 'datasets/seed_records.jsonl'
        write_rows(records_path, [self.record('sibling')])
        with patch.object(stage_training_pilot, 'build_lock') as build:
            with self.assertRaisesRegex(ValueError, 'Fresh evaluation holdout'):
                stage_training_pilot.stage(self.project)
            build.assert_not_called()
        write_rows(records_path, [self.record('allowed')])
        old_lock = self.project / 'training/pilot_lock.json'
        write_json(old_lock, {'historical': True})
        before = old_lock.read_bytes()
        with patch.object(stage_training_pilot, 'build_lock', return_value={'candidate': True}):
            with self.assertRaisesRegex(ValueError, 'Preserve the historical'):
                stage_training_pilot.stage(self.project, old_lock)
        self.assertEqual(old_lock.read_bytes(), before)

    def test_allowed_project_staging_has_zero_overlap(self):
        write_rows(self.project / 'datasets/seed_records.jsonl', [self.record('allowed')])
        report = guard_training_project(self.project)
        self.assertEqual(report['records_checked'], 1)
        self.assertEqual(report['training_source_overlap'], 0)
        self.assertEqual(report['reserved_families'], 1)


if __name__ == '__main__':
    unittest.main()
