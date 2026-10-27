import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from candidate_synthesis import evaluate,validate_protocol
from candidate_audit import reserve_call,CANDIDATE
from verify_ghost_candidate_synthesis import replay
from test_ghost_candidate_audit import protocol


def jobs():
    packet={'id':'source','title':'Bounded excerpt','author':'Author','domain':'historical_record',
        'composition_date_interval':{'earliest':'1930-01-01','latest':'1930-01-01'},'composition_date_evidence':[],
        'context_boundary':'Only this excerpt','blocks':[{'block_id':1,'role':'body_candidate','speaker':'Author','text':'The author reports a claim but does not verify it.'}]}
    return [{'case_id':'q'+str(i),'question':'Explain the claim.', 'messages':[{'role':'system','content':'fixed writer'},
        {'role':'user','content':json.dumps({'reference_date':'2026-10-26','source_packets':[packet],'current_user_request':'Explain the claim.'})}]} for i in range(5)]


def result(text):return {'response':text,'finish_reason':'stop','token_ids':[1],'input_token_ids':[2],'token_count':1}


def verdict(messages,*,missing=False,reject=False):
    data=json.loads(messages[1]['content'])
    return result(json.dumps({'verdicts':[] if missing else [{'id':s['id'],**{k:not reject for k in ('supported','certainty_preserved','attribution_preserved','temporal_safe','relevant','no_extra_claims')},'reason':'Specific source comparison'} for s in data['candidate_segments']]}))


class CandidateSynthesisTests(unittest.TestCase):
    def test_same_source_context_dependent_audit_and_frozen_replay(self):
        requests=[]
        def generate(messages):
            requests.append(messages)
            if messages[0]['content']=='fixed writer':return result('The author reports the claim; the excerpt does not verify it.')
            self.assertEqual(json.loads(messages[1]['content'])['source_packets'],json.loads(jobs()[0]['messages'][1]['content'])['source_packets'])
            return verdict(messages)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);summary=evaluate(jobs(),generate,path)
            self.assertEqual(summary['displayed_answers'],5);self.assertEqual(summary['completed_evaluation_calls'],10)
            rows=[json.loads(s) for s in (path/'answers.jsonl').read_text().splitlines()]
            for row in rows:
                self.assertEqual(row['status'],'supported_by_model_review');self.assertEqual(row['model'],CANDIDATE)
                self.assertEqual(row['accepted_claim_ids'],[]);self.assertEqual(row['memory_updates'],[]);self.assertFalse(row['semantic_guarantee'])
            names=('candidate_synthesis.py','candidate_audit.py','candidate_base_probe.py','source_reading.py','source_audit.py','ghost_chat.py','grounded_chat.py','grounded_chat_v2.py')
            frozen={n:(ROOT/'scripts'/n).read_bytes() for n in names}
            calls=[json.loads(l) for l in (path/'model_calls.jsonl').read_text().splitlines()]
            reproduced=replay(frozen,jobs(),calls)
            for name,raw in reproduced.items():self.assertEqual(raw,(path/name).read_bytes())
            bad=copy.deepcopy(calls);bad[1]['messages'][1]['content']='changed source context'
            import subprocess
            with self.assertRaises(subprocess.CalledProcessError):replay(frozen,jobs(),bad)

    def test_bad_literal_quote_or_truncation_blocks_before_checker(self):
        for text,finish in [('The source says "invented quotation".','stop'),('Incomplete prose','length')]:
            requests=[]
            def generate(messages):
                requests.append(messages);return {**result(text),'finish_reason':finish}
            with tempfile.TemporaryDirectory() as temp:
                summary=evaluate(jobs(),generate,Path(temp))
                self.assertEqual(summary['completed_evaluation_calls'],5);self.assertEqual(len(requests),5)
                rows=[json.loads(l) for l in (Path(temp)/'answers.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['status']=='blocked' and not r['accepted_segment_ids'] for r in rows))

    def test_missing_or_negative_verdict_is_not_published_as_supported(self):
        for missing,reject in [(True,False),(False,True)]:
            def generate(messages):
                if messages[0]['content']=='fixed writer':return result('The author reports a claim.')
                return verdict(messages,missing=missing,reject=reject)
            with tempfile.TemporaryDirectory() as temp:
                evaluate(jobs(),generate,Path(temp));rows=[json.loads(l) for l in (Path(temp)/'answers.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['status']=='review_rejected' and not r['accepted_segment_ids'] for r in rows))
                self.assertTrue(all(r['missing_verdict_ids']==(['s1'] if missing else []) for r in rows))

    def test_protocol_and_actual_call_reservation_are_bounded(self):
        p={**protocol(),'purpose':'candidate_source_synthesis','independent_questions':5,'maximum_displayed_answers':5}
        validate_protocol(p)
        for key,val in [('independent_questions',10),('maximum_displayed_answers',10),('evaluation_calls',30),('enable_thinking',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_protocol({**p,key:val})
        r={'evaluation_calls':0,'memory_probe_calls':0};reserve_call(r,p,16,memory_probe=True)
        with self.assertRaises(ValueError):reserve_call(r,p,16,memory_probe=True)
        for _ in range(10):reserve_call(r,p,1024)
        with self.assertRaises(ValueError):reserve_call(r,p,1024)
        with self.assertRaises(ValueError):reserve_call({'evaluation_calls':0,'memory_probe_calls':0},p,4096)


if __name__=='__main__':unittest.main()
