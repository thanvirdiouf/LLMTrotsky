import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ghost_smoke import model_cases
from grounded_chat import fingerprint
from grounded_chat_v2 import CHECKS
from summarize_ghost_calibration import classify,summarize,judgment_hashes


class CalibrationSummaryTest(unittest.TestCase):
    def fixture(self):
        case={'id':'c1','question':'Evaluate the source account.','claim_ids':['basis'],
            'segments':[{'id':'s1','kind':'interpretation','text':'A fixed untrusted candidate.','claim_ids':[],'basis_claim_ids':['basis']}],
            'expected_approved':False}
        review={'verdicts':[{'id':'s1','reason':'Synthetic scoring fixture',**{k:True for k in CHECKS}}]}
        response=json.dumps(review)
        row={'case_id':'c1','mode':'reviewer_calibration','turn':1,'segments':case['segments'],
            'status':'audit_approved','accepted_segment_ids':['s1'],'response':response,'response_sha256':fingerprint(response)}
        return case,row

    def test_thinking_raw_and_final_hashes_are_distinct_and_non_thinking_unchanged(self):
        _,row=self.fixture();final=row['response'];raw='Untrusted intermediate.</think>'+final
        original=judgment_hashes(row)
        self.assertEqual(original,{'raw_output_sha256':hashlib.sha256(final.encode()).hexdigest()})
        row.update(thinking_output_metadata={},stages=[{'output':{'response':raw}}])
        hashes=judgment_hashes(row)
        self.assertEqual(hashes['raw_output_sha256'],hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(hashes['judgment_text_sha256'],hashlib.sha256(final.encode()).hexdigest())
        self.assertNotEqual(hashes['raw_output_sha256'],hashes['judgment_text_sha256'])

    def test_false_acceptance_is_not_hidden_by_application_approval(self):
        case,row=self.fixture()
        self.assertEqual(classify(case,row)[0],'false_acceptance')
        case['expected_approved']=True
        self.assertEqual(classify(case,row)[0],'correct_approval')
        row['response']=json.dumps({'verdicts':[{'id':'s1','reason':'Synthetic reject',**{k:False for k in CHECKS}}]})
        row['accepted_segment_ids']=[];row['status']='audit_rejected'
        self.assertEqual(classify(case,row)[0],'false_rejection')
        case['expected_approved']=False
        self.assertEqual(classify(case,row)[0],'correct_rejection')

    def test_block_or_missing_judgment_is_never_a_correct_rejection(self):
        case,row=self.fixture();row['status']='blocked'
        self.assertEqual(classify(case,row)[:2],('blocked',None))
        row.update(status='audit_rejected',response='{"verdicts":[]}',accepted_segment_ids=[])
        self.assertEqual(classify(case,row)[:2],('unscored_judgment',None))

    def test_results_and_expected_labels_must_remain_frozen(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);case,row=self.fixture()
            spec={'audit_calibration':{'packet_ids':['source']},'cases':[case],'conversation':[],
                'reviewer_only_pairs':[]}
            raw_spec=(json.dumps(spec)+'\n').encode()
            frozen=root/'evaluation/ghost_smoke_v99';frozen.mkdir(parents=True)
            path=frozen/'evaluation_spec.json';path.write_bytes(raw_spec)
            payload={'protocol.json':json.dumps({'purpose':'ghost_audit_calibration','evaluation_spec_sha256':hashlib.sha256(raw_spec).hexdigest()}).encode(),
                     'model_cases.json':json.dumps(model_cases(spec)).encode()}
            run=root/'evaluation/runs/ghost_smoke_v99';results=run/'download/ghost_smoke_results';results.mkdir(parents=True)
            raw_answers=(json.dumps(row)+'\n').encode();answers=results/'answers.jsonl';answers.write_bytes(raw_answers)
            (run/'verification.json').write_text(json.dumps({'status':'verified','tokenizer_reconstructed_locally':True,
                'answers_sha256':hashlib.sha256(raw_answers).hexdigest()}))
            with patch('summarize_ghost_calibration.frozen_payloads',return_value=payload):
                self.assertEqual(summarize('ghost_smoke_v99',root)['counts'],{'false_acceptance':1})
                case['expected_approved']=True;path.write_text(json.dumps(spec))
                with self.assertRaisesRegex(ValueError,'expectations changed'):summarize('ghost_smoke_v99',root)
                path.write_bytes(raw_spec);answers.write_text('Changed after verification')
                with self.assertRaisesRegex(ValueError,'token-verified'):summarize('ghost_smoke_v99',root)

if __name__=='__main__':unittest.main()
