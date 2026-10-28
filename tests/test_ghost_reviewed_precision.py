import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import test_ghost_reviewed_ledger as fixtures
from build_ghost_candidate_probe import enc, sha
from extend_reviewed_source_ledger import build
from grounded_chat import fingerprint
from grounded_chat_v2 import EvidenceError


class PrecisionExtensionTests(unittest.TestCase):
    def fixture(self, root):
        collection, ledger, supplements = fixtures.ReviewedLedgerTests().fixture()
        # A source paraphrase only; semantic support remains development review.
        supplements = copy.deepcopy(supplements); supplements['sources'] = [supplements['sources'][0]]
        profile = {'reference_date': '2026-10-26'}
        parent = root/'parent'; parent.mkdir()
        for name, value in [('profile.json', profile), ('collection.json', collection),
                            ('ledger.json', ledger), ('build_manifest.json', {'review': 'Original review'})]:
            (parent/name).write_bytes(enc(value))
        (root/'supplements.json').write_bytes(enc(supplements))
        (root/'scripts').mkdir(); (root/'scripts/extend_reviewed_source_ledger.py').write_bytes((ROOT/'scripts/extend_reviewed_source_ledger.py').read_bytes())
        plan = {'schema_version': 1, 'status': 'Offline opt-in', 'parent_directory': 'parent',
                'parent_file_sha256': {n: sha((parent/n).read_bytes()) for n in ('profile.json','collection.json','ledger.json','build_manifest.json')},
                'supplements_file': 'supplements.json', 'supplements_sha256': sha((root/'supplements.json').read_bytes()),
                'added_claim_count': len(supplements['sources'][0]['claims'])}
        (root/'plan.json').write_bytes(enc(plan))
        return profile, collection, ledger

    def test_pinned_parent_claims_and_source_bytes_preserved_without_activation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); profile, collection, old = self.fixture(root)
            p, c, new, manifest = build(root, 'plan.json')
            self.assertEqual(new['claims'][:len(old['claims'])], old['claims'])
            self.assertEqual(enc(p), (root/'parent/profile.json').read_bytes())
            self.assertEqual(enc(c), (root/'parent/collection.json').read_bytes())
            self.assertEqual(c, collection); self.assertEqual(p, profile)
            self.assertEqual(manifest['model_calls'], 0); self.assertFalse(manifest['active_model_changed'])
            self.assertFalse(manifest['semantic_support_independently_verified'])
            self.assertEqual(manifest['base_ledger_fingerprint'], fingerprint(old))

    def test_changed_parent_manifest_ledger_source_or_supplement_rejected(self):
        for filename in ('parent/profile.json','parent/collection.json','parent/ledger.json','parent/build_manifest.json','supplements.json'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); self.fixture(root)
                path = root/filename; path.write_bytes(path.read_bytes()+b'\n')
                with self.assertRaises(EvidenceError): build(root, 'plan.json')

    def test_wrong_addition_count_and_existing_claim_rewrite_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, _, ledger = self.fixture(root)
            plan = json.loads((root/'plan.json').read_text()); plan['added_claim_count'] += 1
            (root/'plan.json').write_bytes(enc(plan))
            with self.assertRaises(EvidenceError): build(root, 'plan.json')
            supplements = json.loads((root/'supplements.json').read_text())
            supplements['sources'][0]['claims'][0]['id'] = ledger['claims'][0]['id']
            (root/'supplements.json').write_bytes(enc(supplements))
            plan['supplements_sha256'] = sha((root/'supplements.json').read_bytes())
            (root/'plan.json').write_bytes(enc(plan))
            with self.assertRaises(EvidenceError): build(root, 'plan.json')


if __name__ == '__main__': unittest.main()
