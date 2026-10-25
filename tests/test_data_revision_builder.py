"""Critical review, immutable-freeze and exact lineage guards for v2."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.build_data_revision_v2 import validate_integration_review, validate_version, rows
from scripts.build_kaggle_data_revision import freeze

ROOT=Path(__file__).resolve().parents[1]

class RevisionBuilderTests(unittest.TestCase):
    def test_current526_records_and50_slice_reconstruct_from_approved_lineage(self):
        # New additive holds must stale the old live inventory without rewriting it.
        with self.assertRaisesRegex(ValueError, 'Holdout inventory expanded'):
            validate_version(ROOT)
        # Reconstruct the historical v2 source view for exact lineage reproduction.
        with tempfile.TemporaryDirectory() as temporary:
            snapshot=Path(temporary)
            for path in ROOT.iterdir():
                if path.name!='evaluation':(snapshot/path.name).symlink_to(path)
            evaluation=snapshot/'evaluation';evaluation.mkdir()
            for name in ['cases.jsonl','fresh_holdout_v1','fresh_holdout_v2']:
                (evaluation/name).symlink_to(ROOT/'evaluation'/name)
            m=validate_version(snapshot)
        self.assertEqual((m['records'],m['slice_records'],m['reserved_documents']),(526,50,12))

    def review_fixture(self,root):
        target=root/'reviews/data_revision_v2';target.parent.mkdir(parents=True)
        shutil.copytree(ROOT/'reviews/data_revision_v2',target)
        return target

    def test_approval_requires_exact61_unique_ids(self):
        repairs=rows(ROOT/'reviews/data_revision_v2/repaired_records.jsonl')
        new=rows(ROOT/'reviews/data_revision_v2/new_records.jsonl')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);review=self.review_fixture(root)
            p=review/'integration_review.json';r=json.loads(p.read_bytes())
            r['approved_example_ids'][-1]=r['approved_example_ids'][0];p.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'exact 61'):
                validate_integration_review(root,repairs,new)

    def test_approval_does_not_survive_proposal_bytes_changing(self):
        repairs=rows(ROOT/'reviews/data_revision_v2/repaired_records.jsonl')
        new=rows(ROOT/'reviews/data_revision_v2/new_records.jsonl')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);review=self.review_fixture(root)
            with (review/'new_records.jsonl').open('a') as stream:stream.write('\n')
            with self.assertRaisesRegex(ValueError,'proposals or review text changed'):
                validate_integration_review(root,repairs,new)

    def test_recomputed_self_reported_hash_cannot_bypass_lineage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'datasets').mkdir();shutil.copytree(ROOT/'datasets/revision_v2',root/'datasets/revision_v2')
            for name in ['seed_records.jsonl','train_seed.jsonl']:(root/'datasets'/name).symlink_to(ROOT/'datasets'/name)
            (root/'reviews').mkdir()
            for name in ['targeted_audit_500','data_revision_v2','seed_evidence_lock.json']:(root/'reviews'/name).symlink_to(ROOT/'reviews'/name)
            (root/'training').mkdir()
            for name in ['pilot_lock.json','pilot_protocol.json']:(root/'training'/name).symlink_to(ROOT/'training'/name)
            p=root/'datasets/revision_v2/seed_records.jsonl';records=rows(p)
            records[-1]['messages'][-1]['content']='This invented target was never source reviewed.'
            p.write_text(''.join(json.dumps(r)+'\n' for r in records))
            mp=root/'datasets/revision_v2/version_manifest.json';m=json.loads(mp.read_bytes());m['files'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest();mp.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'approved lineage'):
                validate_version(root)

    def test_frozen_protocol_is_refused_before_optional_tokenizer_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'training/data_revision_v2';out.mkdir(parents=True)
            (out/'protocol.json').write_text('preserved')
            with self.assertRaisesRegex(FileExistsError,'already frozen'):freeze(root)
            self.assertEqual((out/'protocol.json').read_text(),'preserved')

if __name__=='__main__':unittest.main()
