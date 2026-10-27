import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from ghost_chat import load_active
from build_ghost_smoke import model_cases
from evaluate_ghost_audit import evaluate
from grounded_chat_v2 import CHECKS

class NeutralSourceAuditTest(unittest.TestCase):
    def setUp(self):
        self.bot=load_active()
        self.source=json.loads((ROOT/'config/ghost_neutral_source_calibration_v1.json').read_text())
        self.spec=model_cases(self.source)

    def test_candidates_remain_test_data_and_labels_and_map_are_withheld(self):
        calls=[]
        def generate(messages):
            d=json.loads(messages[1]['content']);calls.append(d)
            self.assertEqual(d['candidate_speaker'],{'role':'assistant','task':'factual source analysis'})
            self.assertEqual(list(d)[-1],'current_user_request')
            self.assertEqual({p['id'] for p in d['source_packets']},{'rakovsky','bessedovsky'})
            self.assertEqual(d['candidate_segments'],self.spec['cases'][len(calls)-1]['segments'])
            for key in ('provisional_task_map','evidence_propositions','expected_approved','reviewer_only','previous_verified_fact_claim_ids'):
                self.assertNotIn(key,d)
            return {'finish_reason':'stop','response':json.dumps({'verdicts':[{'id':'s1','reason':'Synthetic contract only',**{k:True for k in CHECKS}}]})}
        with tempfile.TemporaryDirectory() as out:
            rows,requests=evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,self.spec,generate,out,{'fixture':True})
            self.assertEqual(len(rows),10);self.assertEqual(len(requests),10)
            for r in rows:
                self.assertEqual(r['mode'],'reviewer_calibration');self.assertEqual(r['state_after']['accepted_claim_ids'],[])
                self.assertIn(r['segments'][0]['text'],r['audit_candidate_display'])
            replay,_=evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,self.spec,lambda _:self.fail('Replay inferred'),out,{'fixture':True})
            self.assertEqual(rows,replay)
        self.assertNotIn('expected_approved',json.dumps(self.spec));self.assertNotIn('local_rationale',json.dumps(self.spec))

    def test_unknown_policy_and_malformed_or_truncated_verdict_do_not_approve(self):
        for output in ({'response':'{"verdicts":[]}','finish_reason':'stop'},
                       {'response':'{"verdicts":[],"verdicts":[]}','finish_reason':'stop'},
                       {'response':'{"verdicts":[]}','finish_reason':'length'}):
            with tempfile.TemporaryDirectory() as out:
                spec=copy.deepcopy(self.spec);spec['cases']=spec['cases'][:1]
                rows,_=evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,spec,lambda _:output,out,{'fixture':True})
                self.assertFalse(rows[0]['accepted_segment_ids'])
        bad=copy.deepcopy(self.source);bad['audit_calibration']['policy']='unknown'
        with self.assertRaises(ValueError):model_cases(bad)
        bad=copy.deepcopy(self.spec);bad['audit_calibration']['policy']='unknown'
        with tempfile.TemporaryDirectory() as out,self.assertRaises(ValueError):
            evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,bad,lambda _:self.fail('Must not infer'),out,{'fixture':True})

if __name__=='__main__':unittest.main()
