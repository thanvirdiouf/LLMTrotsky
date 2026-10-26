import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ghost_smoke import model_cases
from evaluate_ghost_comparison import FixedRetriever
from evaluate_ghost_factual_comparison import evaluate
from ghost_chat import load_active
from source_reading import SourceReadingChat,reading_parts
from grounded_chat_v2 import CHECKS,EvidenceError


class SourceReadingTest(unittest.TestCase):
    def bot(self):
        old=load_active();bot=SourceReadingChat(old.profile,old.collection,old.ledger)
        bot.retriever=FixedRetriever(bot,['ilo_genai_2025']);return bot

    def test_full_sources_and_factual_arithmetic_need_no_historical_basis_or_verified_memory(self):
        bot=self.bot();calls=[]
        draft='Adding 5.7% and 4.7% gives 10.4% of female employment in those exposure categories. This measures potential exposure, not observed job losses.\n\nThe supplied global figures do not establish how many people at a particular local office would be affected.'
        def generate(messages):
            data=json.loads(messages[1]['content']);calls.append(data)
            self.assertEqual(list(data)[-1],'current_user_request')
            self.assertEqual([p['id'] for p in data['source_packets']],['ilo_genai_2025'])
            if len(calls)==1:
                self.assertNotIn('evidence_propositions',data)
                return {'response':draft,'finish_reason':'stop'}
            self.assertEqual(data['candidate_segments'][0]['kind'],'source_analysis')
            self.assertEqual(data['previous_verified_fact_claim_ids'],[])
            self.assertEqual(data['requested_answer_mode'],'factual')
            return {'response':json.dumps({'verdicts':[{'id':s['id'],'reason':'Synthetic contract fixture, not semantic validation',**{k:True for k in CHECKS}} for s in data['candidate_segments']]}),'finish_reason':'stop'}
        result=bot.answer('Check 5.7 plus 4.7 and its denominator, then explain local limits.',generate)
        self.assertEqual(len(calls),2);self.assertIn('10.4%',result['response'])
        self.assertIn('[ilo_genai_2025]',result['response'])
        self.assertEqual(result['state_after']['accepted_claim_ids'],[])
        self.assertFalse(result['state_after'].get('interpretation_basis_claim_ids'))
        self.assertFalse(result['semantic_guarantee'])

    def test_quote_occurrence_does_not_authenticate_user_premise(self):
        trace={'matches':[{'packet':{'id':'record','blocks':[{'block_id':1,'speaker':'Witness','text':'Three or four secretaries.'}]}}]}
        _,origins=reading_parts('The witness says "Three or four".',trace,'Explain the count.')
        self.assertEqual(origins[0]['source_matches'][0]['speaker'],'Witness')
        _,origins=reading_parts('Your proposed label "exactly four" is not established.',trace,'Is "exactly four" supported?')
        self.assertFalse(origins[0]['source_matches']);self.assertTrue(origins[0]['user_match_is_unverified'])
        for text in ['The record says "exactly five".','The record says "unfinished.','[unknown] invented citation','Read https://example.invalid','<script>bad</script>']:
            with self.subTest(text=text),self.assertRaises(EvidenceError):reading_parts(text,trace,'Explain the count.')

    def test_negative_missing_and_truncated_reviews_never_promote_a_reading(self):
        for variant in ('negative','missing','truncated'):
            with self.subTest(variant=variant):
                calls=[]
                def generate(messages):
                    calls.append(messages)
                    if len(calls)==1:return {'response':'This is a synthetic source-reading fixture.','finish_reason':'stop'}
                    verdicts=[] if variant=='missing' else [{'id':'s1','reason':'Synthetic rejection',**{k:False for k in CHECKS}}]
                    return {'response':json.dumps({'verdicts':verdicts}),'finish_reason':'length' if variant=='truncated' else 'stop'}
                row=self.bot().answer('Explain the supplied source.',generate)
                self.assertEqual(row['accepted_segment_ids'],[]);self.assertEqual(row['state_after']['accepted_claim_ids'],[])
                self.assertNotIn('synthetic source-reading fixture',row['response'])
        with self.assertRaises(EvidenceError):self.bot().answer('A follow-up?',lambda _:self.fail('Must not infer'),history=[{'role':'user','content':'Prior'}])

    def test_paired_cases_reset_history_withhold_checks_and_replay_without_inference(self):
        bot=self.bot()
        source={'factual_comparison':{'arms':['canonical','reading']},'conversation':[],
            'cases':[{'id':f'c{i}','question':f'Explain the supplied ILO finding, case {i}.','source_packet_ids':['ilo_genai_2025'],
                'checks':['SECRET_EXPECTED_CHECK']} for i in range(5)]}
        spec=model_cases(source);self.assertNotIn('SECRET_EXPECTED_CHECK',json.dumps(spec))
        def generate(messages):
            data=json.loads(messages[1]['content']);system=messages[0]['content']
            self.assertEqual(data.get('conversation',[]),[])
            if 'Select relevant supplied claim IDs' in system:
                return {'response':'{"answer_mode":"facts","detail":"normal","claim_ids":["ilo_exposure_estimate"]}','finish_reason':'stop'}
            if 'candidate_segments' not in data:
                return {'response':'A synthetic independent reading, not a quality judgment.','finish_reason':'stop'}
            return {'response':json.dumps({'verdicts':[{'id':s['id'],'reason':'Synthetic fixture only',**{k:True for k in CHECKS}} for s in data['candidate_segments']]}),'finish_reason':'stop'}
        with tempfile.TemporaryDirectory() as temp:
            answers,calls=evaluate(bot.profile,bot.collection,bot.ledger,spec,generate,temp,{'fixture':True})
            self.assertEqual(len(answers),10);self.assertEqual(len(calls),20)
            self.assertTrue(all(a['state_before']['turns']==0 for a in answers))
            self.assertTrue(all(not a['state_after']['accepted_claim_ids'] for a in answers[5:]))
            replay,_=evaluate(bot.profile,bot.collection,bot.ledger,spec,lambda _:self.fail('No replay inference'),temp,{'fixture':True})
            self.assertEqual(replay,answers)
        wrong=copy.deepcopy(spec);wrong['cases'][5]['question']='Changed counterpart.'
        with tempfile.TemporaryDirectory() as temp,self.assertRaises(ValueError):evaluate(bot.profile,bot.collection,bot.ledger,wrong,lambda _:self.fail('No inference'),temp,{})

if __name__=='__main__':unittest.main()
