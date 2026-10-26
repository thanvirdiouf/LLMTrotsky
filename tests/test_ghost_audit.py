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
from grounded_chat_v2 import CHECKS

class GhostAuditTest(unittest.TestCase):
    def setUp(self):
        self.bot=load_active()
        self.source=json.loads((ROOT/'config/ghost_audit_calibration_v1.json').read_text())
        self.spec=model_cases(self.source)

    def test_roles_unmodified_traps_no_trusted_memory_and_exact_replay(self):
        requests=[]
        def generate(messages):
            data=json.loads(messages[1]['content']);requests.append(data)
            self.assertEqual(list(data)[-1],'current_user_request')
            self.assertEqual(data['candidate_speaker']['role'],'assistant')
            self.assertTrue(data['candidate_speaker']['fictional'])
            self.assertEqual(data['previous_verified_fact_claim_ids'],[])
            self.assertNotIn('expected_approved',data)
            return {'finish_reason':'stop','response':json.dumps({'verdicts':[
                {'id':s['id'],'reason':'Synthetic replay fixture, no semantic claim',**{k:True for k in CHECKS}}
                for s in data['candidate_segments']]})}
        with tempfile.TemporaryDirectory() as temp:
            answers,calls=evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,self.spec,generate,temp,{'fixture':True})
            self.assertEqual(len(calls),10);self.assertEqual(len(answers),10)
            for a,c,r in zip(answers,self.spec['cases'],requests):
                self.assertEqual(r['candidate_segments'],c['segments'])
                self.assertEqual(a['state_after']['accepted_claim_ids'],[])
                self.assertEqual(a['state_after'].get('interpretation_basis_claim_ids',[]),[])
                self.assertEqual(a['response'],a['stages'][0]['output']['response'])
                self.assertIn(c['segments'][0]['text'],a['audit_candidate_display'])
            self.assertEqual(requests[0]['candidate_support_roles'][0]['historical_starting_points'],['radio_social_control'])
            self.assertIn('already been eliminated',requests[1]['candidate_segments'][0]['text'])
            def no_call(_):raise AssertionError('Replay must not infer')
            replay,_=evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,self.spec,no_call,temp,{'fixture':True})
            self.assertEqual(answers,replay)
            old=json.loads(Path(temp,'model_calls.jsonl').read_text().splitlines()[0]);old['output']['response']='tampered'
            rows=Path(temp,'model_calls.jsonl').read_text().splitlines();rows[0]=json.dumps(old)
            Path(temp,'model_calls.jsonl').write_text('\n'.join(rows)+'\n')
            with self.assertRaises(ValueError):evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,self.spec,no_call,temp,{'fixture':True})

    def test_expected_labels_withheld_and_bounds_checked(self):
        self.assertEqual(sum(c['expected_approved'] for c in self.source['cases']),5)
        self.assertNotIn('expected_approved',json.dumps(self.spec));self.assertNotIn('local_rationale',json.dumps(self.spec))
        def no_call(_):raise AssertionError('Invalid design must fail before inference')
        for change in ('excess','duplicate','conversation'):
            spec=copy.deepcopy(self.spec)
            if change=='excess':spec['cases'].append(spec['cases'][0])
            elif change=='duplicate':spec['cases'][1]['id']=spec['cases'][0]['id']
            else:spec['conversation']=['test']
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,spec,no_call,temp,{})

    def test_missing_and_false_verdicts_do_not_approve(self):
        spec=copy.deepcopy(self.spec);spec['cases']=spec['cases'][:2];index=0
        def generate(_):
            nonlocal index
            index+=1
            verdicts=[] if index==1 else [{'id':'s1','reason':'Fixture rejection',**{k:k!='supported' for k in CHECKS}}]
            return {'finish_reason':'stop','response':json.dumps({'verdicts':verdicts})}
        with tempfile.TemporaryDirectory() as temp:
            answers,_=evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,spec,generate,temp,{})
            self.assertTrue(all(a['status']=='audit_rejected' for a in answers))
            self.assertTrue(all(not a['accepted_segment_ids'] for a in answers))
            self.assertEqual(answers[0]['missing_verdict_ids'],['s1'])

if __name__=='__main__':unittest.main()
