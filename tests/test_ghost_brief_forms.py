import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ghost_chat import build
from ghost_chat import GhostChat,EvidenceError,validate_segments,render,plan_response_budget
from grounded_chat import fingerprint
from grounded_chat_v2 import CHECKS

class BriefFormsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile,cls.collection,cls.ledger=build(source_plan=Path('config/ghost_source_plan_v3.json'))
        cls.cards={c['id']:c for c in cls.ledger['claims']}

    def test_brief_scope_retains_method_condition_and_one_sentence_is_real(self):
        for question,sentence_limit in [('Give the dated ILO finding and its implementation limits in two sentences.',2),('Give only the ILO evidence limit in one sentence.',1)]:
            calls=[];bot=GhostChat(self.profile,self.collection,self.ledger)
            def generate(messages):
                calls.append(messages);data=json.loads(messages[1]['content'])
                if len(calls)==1:out={'answer_mode':'facts','detail':'brief','claim_ids':['ilo_exposure_estimate','ilo_adoption_limit','ilo_transformation_assessment']}
                else:out={'verdicts':[{'id':s['id'],'reason':'Synthetic compact-render fixture, not semantic gold',**{k:True for k in CHECKS}} for s in data['candidate_segments']]}
                return {'response':json.dumps(out),'finish_reason':'stop'}
            result=bot.answer(question,generate)
            with self.subTest(sentences=sentence_limit):
                self.assertEqual([s['stage'] for s in result['stages']],['select','check'])
                self.assertIn('20 May 2025',result['response'])
                self.assertIn('not observed job losses',result['response'])
                self.assertEqual(result['response_budget']['displayed_sentences_estimate'],sentence_limit)
                self.assertFalse(result['response_budget']['over_sentence_target'])
                self.assertFalse(result['response_budget']['over_target'])
                if sentence_limit==2:
                    self.assertIn('upper threshold',result['response']);self.assertIn('if GenAI were fully implemented',result['response'])
                    self.assertEqual(result['state_after']['accepted_claim_ids'],['ilo_exposure_estimate','ilo_adoption_limit'])
                else:self.assertEqual(result['state_after']['accepted_claim_ids'],['ilo_exposure_estimate'])
                self.assertNotIn('_fact_form',result['state_after'])

    def test_tampered_brief_forms_or_changed_source_require_new_review(self):
        old=json.load(open(ROOT/'evaluation/ghost_chat_v2/claims.json'))
        self.assertEqual(self.collection,json.load(open(ROOT/'evaluation/ghost_chat_v2/collection.json')))
        by_id={c['id']:c for c in old['claims']}
        for c in self.ledger['claims']:
            self.assertTrue(all(c[k]==v for k,v in by_id[c['id']].items()))
        self.assertEqual(sum('brief_statement' in c for c in self.ledger['claims']),4)
        changed=copy.deepcopy(self.ledger);next(c for c in changed['claims'] if c['id']=='ilo_exposure_estimate')['brief_statement']='The study proves actual losses.'
        with self.assertRaises(EvidenceError):GhostChat(self.profile,self.collection,changed)
        collection=copy.deepcopy(self.collection);ledger=copy.deepcopy(self.ledger)
        next(p for p in collection['packets'] if p['id']=='ilo_genai_2025')['blocks'][1]['text']+=' Added context requiring review.'
        collection['packets_sha256']=fingerprint(collection['packets']);ledger['collection_sha256']=collection['packets_sha256']
        with self.assertRaises(EvidenceError):GhostChat(self.profile,collection,ledger)

    def test_model_cannot_rewrite_forms_and_exact_quotes_keep_original_anchors(self):
        selected={'ilo_exposure_estimate':self.cards['ilo_exposure_estimate']}
        segment={'id':'s1','kind':'fact','claim_ids':list(selected),'form':'brief','text':'Invented 2026 measured outcome.'}
        validated=validate_segments({'segments':[segment]},selected)
        self.assertEqual(validated[0]['text'],selected['ilo_exposure_estimate']['brief_statement'])
        text,_,_=render(validated,{'s1'},selected)
        self.assertNotIn('Invented',text)
        with self.assertRaises(EvidenceError):validate_segments({'segments':[{**segment,'form':'unknown'}]},selected)
        quote={'id':'s1','kind':'quote','quote':{'claim_id':'ilo_exposure_estimate','anchor_index':0}}
        validated=validate_segments({'segments':[quote]},selected)
        text,_,_=render(validated,{'s1'},selected)
        self.assertIn(selected['ilo_exposure_estimate']['anchors'][0]['text'],text)
        ids,text,budget=plan_response_budget('Explain the source finding.',selected,{'_answer_mode':'facts','_detail':'normal'})
        self.assertIn(selected['ilo_exposure_estimate']['statement'],text)
        self.assertNotIn('fact_form',budget)

if __name__=='__main__':unittest.main()
