import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_reviewed_selector import cases
from build_reviewed_source_ledger import extend
import test_ghost_reviewed_ledger as fixtures


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['enable_thinking'] is False
        return [1,2,3]


class SelectorInputTests(unittest.TestCase):
    def inputs(self):
        collection,ledger,supplements=fixtures.ReviewedLedgerTests().fixture();ledger=extend(collection,ledger,supplements)
        spec={'cases':[{'id':'case'+str(i),'question':'Inspect the supplied sources.','source_packet_ids':['h','m'],
            'checks':[{'expected':'SECRET_EXPECTED_ANSWER'}],'oracle_claim_ids':['SECRET_ORACLE']} for i in range(5)],
            'conversation':[],'reviewer_only_rubric':'SECRET_REVIEW_RUBRIC'}
        return {'reference_date':'2026-10-26'},collection,ledger,spec

    def test_full_sources_and_all_available_summaries_preserve_inference_roles_without_answer_keys(self):
        profile,collection,ledger,spec=self.inputs();before=copy.deepcopy((profile,collection,ledger,spec))
        jobs=cases(profile,collection,ledger,spec,Tokenizer());self.assertEqual((profile,collection,ledger,spec),before)
        self.assertEqual(len(jobs),5)
        for job in jobs:
            serialized=json.dumps(job['messages']);self.assertNotIn('SECRET_',serialized)
            data=json.loads(job['messages'][1]['content'])
            self.assertEqual([p['id'] for p in data['source_packets']],['h','m'])
            self.assertEqual(len(data['reviewed_source_summaries']),4)
            summary=next(s for s in data['reviewed_source_summaries'] if s['id']=='sum')
            self.assertEqual(summary['status'],'inference');self.assertEqual(summary['review_kind'],'arithmetic_derivation')
            self.assertEqual(data['source_packets'][1]['observed_date'],'2026-10-26')
            self.assertNotIn('claim_ids',data)

    def test_input_budget_or_missing_source_blocks_preparation_without_silent_truncation(self):
        profile,c,l,spec=self.inputs()
        class Long(Tokenizer):
            def apply_chat_template(self,*args,**kwargs):return [1]*5121
        with self.assertRaises(ValueError):cases(profile,c,l,spec,Long())
        spec['cases'][0]['source_packet_ids']=['absent']
        with self.assertRaises(ValueError):cases(profile,c,l,spec,Tokenizer())


if __name__=='__main__':unittest.main()
