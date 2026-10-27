import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ghost_smoke import model_cases
from evaluate_ghost_audit import evaluate
from ghost_chat import load_active
from source_audit import thinking_final
from grounded_chat import fingerprint
from grounded_chat_v2 import CHECKS,EvidenceError

class ThinkingAuditTest(unittest.TestCase):
    def setUp(self):
        self.bot=load_active()
        self.source=json.loads((ROOT/'config/ghost_neutral_thinking_calibration_v1.json').read_text())
        self.spec=model_cases(self.source)

    def test_only_complete_final_channel_can_be_judged(self):
        final='{"verdicts":[]}'
        for prefix in ['Untrusted intermediate {"verdicts":["approve"]}.','<think>Untrusted intermediate.']:
            result,meta=thinking_final(prefix+'</think>\n'+final)
            self.assertEqual(result,final);self.assertEqual(meta['raw_output_fingerprint'],fingerprint(prefix+'</think>\n'+final))
        for raw in [final,'Unfinished reasoning','reason</think>','a</think>b</think>'+final,'x<think>y</think>'+final,'a</think><think>'+final]:
            with self.subTest(raw=raw),self.assertRaises(EvidenceError):thinking_final(raw)

    def test_raw_journal_and_final_verdict_remain_separate_and_replay_exactly(self):
        spec=copy.deepcopy(self.spec);spec['cases']=spec['cases'][:1]
        verdict={'verdicts':[{'id':'s1','reason':'Synthetic final rejection, not semantic validation',**{k:False for k in CHECKS}}]}
        final=json.dumps(verdict);raw='Intermediate text: {"verdicts": ["approve"]}.\n</think>\n'+final
        with tempfile.TemporaryDirectory() as out:
            rows,calls=evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,spec,
                lambda _:{'response':raw,'finish_reason':'stop'},out,{'enable_thinking':True})
            row=rows[0];self.assertEqual(row['response'],final);self.assertEqual(row['stages'][0]['output']['response'],raw)
            self.assertEqual(row['response_sha256'],fingerprint(final));self.assertEqual(calls[0]['output']['response'],raw)
            self.assertEqual(row['status'],'audit_rejected');self.assertFalse(row['accepted_segment_ids'])
            self.assertEqual(row['thinking_output_metadata']['final_output_fingerprint'],fingerprint(final))
            replay,_=evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,spec,lambda _:self.fail('Replay inferred'),out,{'enable_thinking':True})
            self.assertEqual(rows,replay)

    def test_incomplete_reasoning_or_final_and_identity_mismatch_block(self):
        spec=copy.deepcopy(self.spec);spec['cases']=spec['cases'][:1]
        for raw,finish in [('Unfinished reasoning','stop'),('a</think>{','stop'),('a</think>{"verdicts":[]}','length')]:
            with tempfile.TemporaryDirectory() as out:
                rows,_=evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,spec,lambda _:{'response':raw,'finish_reason':finish},out,{'enable_thinking':True})
                self.assertEqual(rows[0]['status'],'blocked');self.assertFalse(rows[0]['accepted_segment_ids'])
        with tempfile.TemporaryDirectory() as out,self.assertRaises(ValueError):
            evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,spec,lambda _:self.fail('Must not infer'),out,{'enable_thinking':False})

    def test_candidates_and_expectations_unchanged_and_thinking_scope_narrow(self):
        original=json.loads((ROOT/'config/ghost_neutral_source_calibration_v1.json').read_text())
        self.assertEqual(self.source['cases'],original['cases'])
        self.assertNotIn('expected_approved',json.dumps(self.spec))
        for change in ('wrong_policy','wrong_type'):
            v=copy.deepcopy(self.source)
            if change=='wrong_policy':v['audit_calibration']['policy']='ghost'
            else:v['audit_calibration']['thinking']='true'
            with self.subTest(change=change),self.assertRaises(ValueError):model_cases(v)

if __name__=='__main__':unittest.main()
