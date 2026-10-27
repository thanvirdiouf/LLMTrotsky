import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from candidate_reviewed_selection import evaluate,validate_protocol
from prepare_reviewed_selector import cases
from build_reviewed_source_ledger import extend
from candidate_audit import reserve_call
from verify_reviewed_selection import replay
import test_ghost_reviewed_ledger as fixtures
from test_ghost_candidate_audit import protocol as common_protocol


class Tokenizer:
    def apply_chat_template(self,*args,**kwargs):return [1,2,3]


class ReviewedSelectionTests(unittest.TestCase):
    def fixture(self):
        c,l,s=fixtures.ReviewedLedgerTests().fixture();l=extend(c,l,s);c['reference_date']='2026-10-26'
        spec={'conversation':[],'cases':[{'id':'q'+str(i),'question':'Inspect sources.','source_packet_ids':['h','m']} for i in range(5)]}
        jobs=cases({'reference_date':'2026-10-26'},c,l,spec,Tokenizer())
        return c,l,jobs

    def test_only_selected_immutable_summaries_display_and_frozen_pipeline_replays(self):
        c,l,jobs=self.fixture();out={'response':'{"claim_ids":["new_h","sum","limit"]}','finish_reason':'stop','token_ids':[1],'input_token_ids':[1,2,3]}
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);summary=evaluate(jobs,lambda _:copy.deepcopy(out),d,c,l)
            self.assertEqual(summary['completed_evaluation_calls'],5)
            rows=[json.loads(line) for line in (d/'answers.jsonl').read_text().splitlines()]
            for r in rows:
                self.assertEqual(r['selected_claim_ids'],['new_h','sum','limit']);self.assertEqual(r['rendered_claim_ids'],r['selected_claim_ids'])
                self.assertEqual(r['omitted_selected_claim_ids'],[]);self.assertEqual(r['accepted_claim_ids'],[]);self.assertEqual(r['memory_updates'],[])
                self.assertFalse(r['generated_factual_text_used']);self.assertFalse(r['model_audit_performed']);self.assertFalse(r['requested_task_coverage_verified'])
            names=('candidate_reviewed_selection.py','candidate_audit.py','candidate_base_probe.py','reviewed_source_display.py','grounded_chat.py','grounded_chat_v2.py')
            frozen={n:(ROOT/'scripts'/n).read_bytes() for n in names}
            frozen.update({'collection.json':json.dumps(c).encode(),'ledger.json':json.dumps(l).encode(),'audit_inputs.json':json.dumps(jobs).encode()})
            calls=[json.loads(line) for line in (d/'model_calls.jsonl').read_text().splitlines()]
            for n,raw in replay(frozen,calls).items():self.assertEqual(raw,(d/n).read_bytes())

    def test_bad_or_truncated_selection_blocks_without_rewriting_or_retry(self):
        c,l,jobs=self.fixture()
        for response,finish in [('{"claim_ids":["sum"],"text":"INVENTED FACT"}','stop'),('{"claim_ids":["unknown"]}','stop'),('{"claim_ids":["sum"]}','length')]:
            with tempfile.TemporaryDirectory() as temp:
                d=Path(temp);summary=evaluate(jobs,lambda _:{'response':response,'finish_reason':finish},d,c,l)
                self.assertEqual(summary['completed_evaluation_calls'],5)
                rows=[json.loads(line) for line in (d/'answers.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['status']=='blocked' and not r['selected_claim_ids'] for r in rows))
                self.assertTrue(all('INVENTED FACT' not in r['response'] for r in rows))

    def test_scope_and_actual_call_limit_are_five_without_audit_or_factual_generation(self):
        p={**common_protocol(),'purpose':'candidate_reviewed_selection','evaluation_calls':5,'independent_questions':5,
            'maximum_displayed_answers':5,'maximum_selected_records':12,'generated_factual_text_used':False,'model_audit_performed':False}
        validate_protocol(p)
        for key,value in [('evaluation_calls',6),('maximum_selected_records',13),('generated_factual_text_used',True),('model_audit_performed',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_protocol({**p,key:value})
        r={'memory_probe_calls':0,'evaluation_calls':0}
        for _ in range(5):reserve_call(r,p,1024)
        with self.assertRaises(ValueError):reserve_call(r,p,1024)


if __name__=='__main__':unittest.main()
