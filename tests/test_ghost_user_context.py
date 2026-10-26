import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ghost_smoke import model_cases
from evaluate_ghost_smoke import evaluate
from ghost_chat import load_active,retain_user_context,EvidenceError
from grounded_chat import fingerprint

class UserContextTest(unittest.TestCase):
    def test_original_and_revised_user_terms_survive_source_only_turns_without_becoming_facts(self):
        bot=load_active();source=json.loads((ROOT/'config/ghost_transfer_evaluation_v1.json').read_text());spec=model_cases(source)
        requests=[]
        def generate(messages):
            requests.append(json.loads(messages[1]['content']))
            return {'response':'{"answer_mode":"facts","detail":"brief","claim_ids":[]}','finish_reason':'stop'}
        with tempfile.TemporaryDirectory() as temp:
            answers,calls=evaluate(bot.profile,bot.collection,bot.ledger,spec,generate,temp,{'synthetic_context_fixture':True})
            late=next(d for d in requests if d['current_user_request']==spec['conversation'][6])
            self.assertEqual([r['turn'] for r in late['earlier_user_context']],[1,4])
            self.assertEqual([r['text'] for r in late['earlier_user_context']],[spec['conversation'][0],spec['conversation'][3]])
            self.assertTrue(all(r['role']=='user' and r['status']=='unverified_user_context' for r in late['earlier_user_context']))
            self.assertFalse(any(m['content']==spec['conversation'][3] for m in late['conversation']))
            self.assertEqual(list(late)[-1],'current_user_request')
            self.assertEqual(late['previous_verified_fact_claim_ids'],[])
            self.assertTrue(all(not a['state_after']['accepted_claim_ids'] for a in answers))
            self.assertFalse(any('earlier_user_context' in d for d in requests[:5]))
            self.assertEqual(len(calls),10)
            again,_=evaluate(bot.profile,bot.collection,bot.ledger,spec,lambda _:self.fail('Replay must not infer'),temp,{'synthetic_context_fixture':True})
            self.assertEqual(again,answers)

    def test_explicit_revision_of_an_existing_scenario_does_not_depend_on_its_noun(self):
        for verb in ('Revise','Now update','Modify','Amend'):
            with self.subTest(verb=verb):
                state,_=retain_user_context({'turns':0},'Imagine a committee with an advisory council.')
                state['turns']=1
                revised=verb+' the committee: elected delegates now decide and the chair has no veto.'
                state,events=retain_user_context(state,revised)
                self.assertEqual([r['turn'] for r in state['user_context_records']],[1,2])
                self.assertEqual(state['user_context_records'][-1]['text'],revised)
                self.assertEqual(state['user_context_records'][-1]['status'],'unverified_user_context')
                self.assertTrue(events)
        # Ordinary editing instructions do not start a hypothetical memory.
        state,events=retain_user_context({'turns':0},'Revise this sentence for grammar.')
        self.assertNotIn('user_context_records',state);self.assertEqual(events,[])

    def test_context_is_bounded_by_whole_messages_with_omission_provenance(self):
        state={'turns':0};original=copy.deepcopy(state)
        for i in range(4):
            state,events=retain_user_context(state,f'Assume proposal {i} remains hypothetical.')
            state={**state,'turns':state['turns']+1}
        self.assertEqual(original,{'turns':0})
        self.assertEqual([r['turn'] for r in state['user_context_records']],[2,3,4])
        self.assertEqual(state['user_context_omitted'],1)
        self.assertTrue(any(e['kind']=='whole_record_evicted' for e in events))
        old=copy.deepcopy(state)
        state,events=retain_user_context(state,'Assume '+('x'*2500))
        self.assertEqual(state['user_context_records'],old['user_context_records'])
        self.assertEqual(state['user_context_omitted'],2)
        self.assertEqual(events[0]['kind'],'not_retained_over_budget')
        for r in state['user_context_records']:self.assertEqual(fingerprint(r['text']),r['content_fingerprint'])

    def test_user_context_cannot_forge_verified_or_assistant_provenance(self):
        state,_=retain_user_context({'turns':0},'Imagine a proposed arrangement.')
        state['turns']=1
        for change in ('role','status','text','extra_key','future_turn'):
            bad=copy.deepcopy(state);r=bad['user_context_records'][0]
            if change=='role':r['role']='assistant'
            elif change=='status':r['status']='verified_fact'
            elif change=='text':r['text']='Changed without its fingerprint'
            elif change=='extra_key':r['verified']=True
            else:r['turn']=2
            with self.subTest(change=change),self.assertRaises(EvidenceError):retain_user_context(bad,'Continue.')

if __name__=='__main__':unittest.main()
