import copy
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ghost_chat import build,historical_additions
from grounded_chat import fingerprint
from ghost_chat import GhostChat,initial_state,claims_for_trace,request
from grounded_chat_v2 import CHECKS


class GhostSourceTest(unittest.TestCase):
    def test_expansion_is_separate_and_preserves_dates_and_prediction_status(self):
        active_before=(ROOT/'config/active_chat.json').read_bytes()
        _,old,old_ledger=build(source_plan=Path('config/ghost_source_plan_v1.json'))
        profile,new,ledger=build(source_plan=Path('config/ghost_source_plan_v2.json'))
        self.assertEqual(profile['reference_date'],'2026-10-26')
        self.assertEqual((len(new['packets']),len(ledger['claims'])),(10,39))
        packets={p['id']:p for p in new['packets']}
        for packet in old['packets']:self.assertEqual(packet,packets[packet['id']])
        radio=packets['radio_science']
        self.assertEqual([b['block_id'] for b in radio['blocks']],[25,26,27,28])
        self.assertEqual(radio['composition_date_interval']['latest'],'1926-03-01')
        self.assertNotIn('Publisher’s Introduction',' '.join(b['text'] for b in radio['blocks']))
        ilo=packets['ilo_genai_2025']
        self.assertEqual(ilo['composition_date_evidence'][0]['kind'],'publication_date')
        self.assertEqual(ilo['composition_date_interval']['latest'],'2025-05-20')
        self.assertEqual(ilo['observed_date'],'2026-10-26')
        cards={c['id']:c for c in ledger['claims']}
        for claim in old_ledger['claims']:self.assertEqual(claim,cards[claim['id']])
        self.assertEqual(cards['radio_war_prediction']['status'],'prediction')
        self.assertEqual(cards['ilo_transformation_assessment']['status'],'prediction')
        self.assertEqual(cards['ilo_adoption_limit']['status'],'explicit_limit')
        self.assertEqual((ROOT/'config/active_chat.json').read_bytes(),active_before)
        bot=GhostChat(profile,new,ledger)
        trace=bot.retriever.retrieve('How would you assess the ILO GenAI exposure estimate?',initial_state(new,ledger))
        self.assertEqual({m['packet']['id'] for m in trace['matches']},{'ilo_genai_2025','radio_science'})
        data=json.loads(request('select',profile,'What does the ILO estimate?',trace,claims_for_trace(ledger,trace),[],initial_state(new,ledger))['messages'][1]['content'])
        ilo_input=next(p for p in data['source_packets'] if p['id']=='ilo_genai_2025')
        self.assertEqual(ilo_input['composition_date_evidence'][0]['kind'],'publication_date')
        self.assertIn('not an observed job-loss count',ilo_input['context_boundary'])

    def test_additional_historical_raw_bytes_must_match_the_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);corpus=root/'corpus_mia';corpus.mkdir()
            raw=b'reviewed fixture';digest=hashlib.sha256(raw).hexdigest()
            block={'block_id':1,'role':'body_candidate','text':'A source argument.'}
            doc={'document_id':'fixture','source_sha256':digest,'source_html':'fixture.html',
                 'blocks':[block],'composition_date_interval':{'latest':'1926-03-01'},
                 'composition_date_evidence':[],'metadata_fields_raw':{},'title':'Fixture','url':'https://example.org'}
            (corpus/'fixture.html').write_bytes(raw)
            (corpus/'documents.jsonl').write_text(json.dumps(doc)+'\n')
            (corpus/'catalog.jsonl').write_text(json.dumps({'document_id':'fixture','source_family_id':'fixture'})+'\n')
            source={'id':'fixture','document_id':'fixture','author':'Fixture author','source_sha256':digest,
                    'block_ids':[1],'blocks_sha256':fingerprint([block]),'annotations':[],
                    'context_boundary':'Complete fixture','aliases':['fixture'],'claims':[]}
            path=root/'extra.json';path.write_text(json.dumps({'review':'Fixture','sources':[source]}))
            self.assertEqual(len(historical_additions(root,path,date(2026,10,26))[0]),1)
            (corpus/'fixture.html').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'raw source changed'):
                historical_additions(root,path,date(2026,10,26))

    def test_publication_and_update_dates_cannot_be_conflated(self):
        modern=json.loads((ROOT/'config/ghost_contemporary_sources_v2.json').read_text())
        plan=json.loads((ROOT/'config/ghost_source_plan_v2.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp);path=temp/'modern.json';plan['contemporary_sources_file']=str(path)
            spec=temp/'plan.json';spec.write_text(json.dumps(plan))
            ambiguous=copy.deepcopy(modern)
            ambiguous['sources'][-1]['updated_date']='2025-05-20'
            path.write_text(json.dumps(ambiguous))
            with self.assertRaisesRegex(ValueError,'one explicit publication or update date'):
                build(source_plan=spec)
            future=copy.deepcopy(modern);future['sources'][-1]['observed_date']='2026-10-27'
            path.write_text(json.dumps(future))
            with self.assertRaisesRegex(ValueError,'future-dated'):
                build(source_plan=spec)

    def test_study_only_followup_excludes_old_political_context_from_display(self):
        profile,collection,ledger=build(source_plan=Path('config/ghost_source_plan_v2.json'))
        bot=GhostChat(profile,collection,ledger);state=initial_state(collection,ledger)
        state['active_packet_ids']=['ilo_genai_2025','radio_science']
        state['interpretation_basis_claim_ids']=['radio_social_control']
        calls=[]
        def generate(messages):
            calls.append(messages)
            if len(calls)==1:out={'claim_ids':['ilo_exposure_estimate','radio_social_control']}
            else:
                segment=json.loads(messages[1]['content'])['candidate_segments'][0]
                self.assertEqual(segment['claim_ids'],['ilo_exposure_estimate'])
                out={'verdicts':[{'id':'s1','reason':'Synthetic source-scope fixture',**{k:True for k in CHECKS}}]}
            return {'response':json.dumps(out),'finish_reason':'stop'}
        result=bot.answer('Now give only the dated study findings, without your political interpretation.',generate,state=state)
        self.assertEqual(result['application_excluded_context_claim_ids'],['radio_social_control'])
        self.assertNotIn('[radio_science]',result['response'])
        self.assertFalse(result['state_after']['interpretation_basis_claim_ids'])


if __name__=='__main__':unittest.main()
