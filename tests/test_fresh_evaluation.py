"""Protect the new diagnostic holdout and the original pilot artifacts."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.freeze_fresh_evaluation import build, verify

ROOT=Path(__file__).resolve().parents[1]

class FreshEvaluationTests(unittest.TestCase):
    def test_frozen_evidence_input_projection_and_original_pilot_still_match(self):
        p=verify(ROOT)
        self.assertEqual(p['category_counts'],dict.fromkeys(['chronology','false_premise','invented_memory_citation','answer_detail'],6))
        self.assertEqual(p['primary_comparison'],['fresh_base','step_50'])
        self.assertEqual(p['generation']['eos_token_id'],248046)

    def test_later_editorial_accounts_are_explicit_and_manuscript_uncertainty_preserved(self):
        cases=[json.loads(s) for s in (ROOT/'evaluation/fresh_holdout_v1/cases.jsonl').read_text().splitlines()]
        industry=next(c for c in cases if c['case_id']=='fresh_v1_industry_chronology')
        self.assertTrue({5,6,7}<=set(industry['source_block_ids']))
        self.assertEqual(industry['composition_date_interval']['earliest'],'1938-05-01')
        self.assertIn('Probable',industry['composition_date_qualification'])
        self.assertIn('editors',industry['reference_role_notes'])

    def test_freeze_refuses_to_overwrite_existing_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'evaluation/fresh_holdout_v1';out.mkdir(parents=True)
            p=out/'protocol.json';p.write_text('original frozen bytes')
            with self.assertRaisesRegex(FileExistsError,'frozen'):
                build(root)
            self.assertEqual(p.read_text(),'original frozen bytes')

    def test_checksum_corruption_rejected_before_source_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'evaluation/fresh_holdout_v1';out.mkdir(parents=True)
            (out/'protocol.json').write_bytes((ROOT/'evaluation/fresh_holdout_v1/protocol.json').read_bytes())
            # Dict iteration order is canonical in protocol output; corrupt first listed file.
            p=json.loads((out/'protocol.json').read_text())
            name=next(iter(p['files']))
            (out/name).write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                verify(root)

if __name__=='__main__':unittest.main()
