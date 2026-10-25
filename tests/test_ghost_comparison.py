import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ghost_smoke import model_cases
from evaluate_ghost_comparison import evaluate
from ghost_chat import load_active
from grounded_chat_v2 import CHECKS


class GhostComparisonTest(unittest.TestCase):
    def test_matched_sources_real_separate_histories_and_exact_resume(self):
        bot=load_active()
        source=json.loads((ROOT/'config/ghost_comparison_evaluation_v1.json').read_text())
        spec=model_cases(source);calls=[]
        def generate(messages):
            calls.append(messages);instruction=messages[0]['content'];data=json.loads(messages[1]['content'])
            if 'Answer the current user naturally in plain prose' in instruction:
                text='Synthetic direct reply with no quality claim.'
            elif 'Select relevant supplied claim IDs' in instruction:
                text=json.dumps({'claim_ids':['ilo_exposure_estimate','radio_social_control']})
            elif 'Audit every segment' in instruction:
                text=json.dumps({'verdicts':[{'id':s['id'],'reason':'Synthetic comparison replay fixture',**{k:True for k in CHECKS}} for s in data['candidate_segments']]})
            else:
                text=json.dumps({'facts':['ilo_exposure_estimate'],'interpretation':'I would ask how workers could decide the use of these tools.','quotes':[]})
            return {'response':text,'finish_reason':'stop'}
        with tempfile.TemporaryDirectory() as temp:
            answers,records=evaluate(bot.profile,bot.collection,bot.ledger,spec,generate,temp,{'fixture':True})
            self.assertEqual(len(answers),10);self.assertLessEqual(len(records),20)
            direct=[a for a in answers if a['arm']=='direct'];checked=[a for a in answers if a['arm']=='checked']
            self.assertEqual([a['conversation_turn'] for a in direct],[1,2,3,4,5])
            self.assertTrue(all(a['status']=='direct_unchecked' for a in direct))
            self.assertTrue(all(not a['state_after']['accepted_claim_ids'] for a in direct))
            direct_first=json.loads(direct[0]['stages'][0]['messages'][1]['content'])
            checked_first=json.loads(checked[0]['stages'][0]['messages'][1]['content'])
            self.assertEqual(direct_first['source_packets'],checked_first['source_packets'])
            self.assertEqual(direct_first['evidence_propositions'],checked_first['evidence_propositions'])
            for arm_rows in (direct,checked):
                data=json.loads(arm_rows[1]['stages'][0]['messages'][1]['content'])
                self.assertEqual(data['conversation'][1]['content'],arm_rows[0]['response'])
                self.assertEqual({m['packet']['id'] for m in arm_rows[1]['retrieval']['matches']},{'ilo_genai_2025','radio_science'})
            count=len(calls)
            def no_call(_):raise AssertionError('Exact replay needs no generation')
            replay,_=evaluate(bot.profile,bot.collection,bot.ledger,spec,no_call,temp,{'fixture':True})
            self.assertEqual(replay,answers);self.assertEqual(count,len(calls))
            records[0]['output']['response']='changed'
            Path(temp,'model_calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(ValueError):evaluate(bot.profile,bot.collection,bot.ledger,spec,no_call,temp,{'fixture':True})

    def test_model_cases_withhold_checks_and_reject_unmatched_or_excessive_cases(self):
        source=json.loads((ROOT/'config/ghost_comparison_evaluation_v1.json').read_text());spec=model_cases(source)
        self.assertEqual(len(spec['cases']),10)
        self.assertFalse(any('checks' in c for c in spec['cases']))
        bot=load_active()
        def no_call(_):raise AssertionError('Malformed comparison must fail before inference')
        with tempfile.TemporaryDirectory() as temp:
            unmatched=copy.deepcopy(spec);unmatched['cases'][-1]['question']='different'
            with self.assertRaisesRegex(ValueError,'matched questions'):
                evaluate(bot.profile,bot.collection,bot.ledger,unmatched,no_call,temp,{})
            excess=copy.deepcopy(spec);extra=copy.deepcopy(excess['cases'][0]);extra['id']='extra';excess['cases'].append(extra)
            with self.assertRaisesRegex(ValueError,'limits'):
                evaluate(bot.profile,bot.collection,bot.ledger,excess,no_call,temp,{})


if __name__=='__main__':unittest.main()
