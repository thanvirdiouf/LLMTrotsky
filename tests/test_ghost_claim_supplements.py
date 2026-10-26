import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ghost_chat import historical_claim_supplements
from grounded_chat import fingerprint
from grounded_chat_v2 import EvidenceError,CHECKS
from ghost_chat import GhostChat,load_active,initial_state,claims_for_trace,request


class ClaimSupplementTest(unittest.TestCase):
    def setUp(self):
        self.packet={'id':'record','domain':'historical_record','author':'Witness',
            'composition_date_interval':{'earliest':'1937-04-10','latest':'1937-04-10'},
            'blocks':[{'block_id':1,'speaker':'Witness','text':'Three or four; one does not live here.'}]}
        self.claim={'id':'new_count_limit','statement':'The witness gives an uncertain count and excludes one person from residence.',
            'status':'explicit_limit','attribution':'Witness testimony, not an exact census',
            'anchors':[{'block_id':1,'speaker':'Witness','text':'Three or four; one does not live here.'}]}
        self.spec={'schema_version':1,'review':'Synthetic source-binding fixture, not historical review',
            'sources':[{'packet_id':'record','packet_fingerprint':fingerprint(self.packet),'claims':[self.claim]}]}

    def apply(self,spec=None,packet=None,existing=()):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'reviewed.json';path.write_text(json.dumps(spec or self.spec))
            return historical_claim_supplements([packet or self.packet],list(existing),path)

    def test_append_preserves_original_sources_and_claims_and_inherits_date(self):
        old={**copy.deepcopy(self.claim),'id':'old_count','packet_id':'record','domain':'historical_record',
            'composition_date_interval':self.packet['composition_date_interval']}
        packets_before=copy.deepcopy(self.packet);old_before=copy.deepcopy(old)
        added,review=self.apply(existing=[old])
        self.assertEqual(self.packet,packets_before);self.assertEqual(old,old_before)
        self.assertEqual(added[0]['composition_date_interval'],self.packet['composition_date_interval'])
        self.assertEqual(added[0]['domain'],'historical_record')
        self.assertEqual(added[0]['anchors'],self.claim['anchors'])
        self.assertTrue(review)

    def test_changed_source_or_unsupported_anchor_speaker_cannot_be_added(self):
        for change in ('packet','text','speaker','status','unrenderable_format'):
            with self.subTest(change=change):
                spec=copy.deepcopy(self.spec);packet=copy.deepcopy(self.packet);c=spec['sources'][0]['claims'][0]
                if change=='packet':packet['blocks'][0]['text']='A changed source, with a different count.'
                elif change=='text':c['anchors'][0]['text']='Exactly four residents.'
                elif change=='speaker':c['anchors'][0]['speaker']='Counsel'
                elif change=='status':c['status']='independently_verified'
                else:c['statement']='The count is “three or four” according to the witness.'
                with self.assertRaises((ValueError,EvidenceError)):self.apply(spec,packet)

    def test_selector_offset_omission_preserves_full_sources_and_later_anchor_checks(self):
        original=load_active()
        bot=GhostChat(original.profile,original.collection,{**original.ledger,'selection_anchor_offsets':False})
        question='What hourly A40 compute rate is documented in the Runpod listing?'
        state=initial_state(bot.collection,bot.ledger);trace=bot.retriever.retrieve(question,state)
        control=json.loads(request('select',bot.profile,question,trace,claims_for_trace(bot.ledger,trace),[],state)['messages'][1]['content'])
        calls=[]
        def generate(messages):
            data=json.loads(messages[1]['content']);calls.append(data)
            if len(calls)==1:
                self.assertEqual(data['source_packets'],control['source_packets'])
                self.assertEqual(data['evidence_propositions'],[{k:v for k,v in c.items() if k!='anchors'} for c in control['evidence_propositions']])
                return {'finish_reason':'stop','response':'{"answer_mode":"facts","detail":"normal","claim_ids":["runpod_a40_price"]}'}
            self.assertTrue(all(c['anchors'] for c in data['evidence_propositions']))
            return {'finish_reason':'stop','response':json.dumps({'verdicts':[{'id':s['id'],'reason':'Synthetic contract fixture',**{k:True for k in CHECKS}} for s in data['candidate_segments']]})}
        result=bot.answer(question,generate)
        self.assertEqual(len(calls),2)
        self.assertIn('$0.59',result['response'])
        self.assertEqual(result['state_after']['accepted_claim_ids'],['runpod_a40_price'])

    def test_supplements_cannot_rewrite_ids_dates_or_modern_sources(self):
        for change in ('duplicate','date_override','modern','missing_review'):
            with self.subTest(change=change):
                spec=copy.deepcopy(self.spec);packet=copy.deepcopy(self.packet);existing=[]
                if change=='duplicate':existing=[{'id':self.claim['id']}]
                elif change=='date_override':spec['sources'][0]['claims'][0]['composition_date_interval']={'earliest':'1900-01-01'}
                elif change=='modern':packet['domain']='contemporary_evidence';spec['sources'][0]['packet_fingerprint']=fingerprint(packet)
                else:spec['review']=''
                with self.assertRaises(ValueError):self.apply(spec,packet,existing)

if __name__=='__main__':unittest.main()
