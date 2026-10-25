import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ghost_chat import build
from ghost_chat import (GhostChat, initial_state, request, claims_for_trace,
                        validate_segments, render, EvidenceError,sanitize_selection,filter_segments,expand_draft)
from grounded_chat_v2 import CHECKS


class GhostChatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile,cls.collection,cls.ledger=build(source_plan=Path('config/ghost_source_plan_v1.json'))
        cls.cards={c['id']:c for c in cls.ledger['claims']}

    def setUp(self):
        self.bot=GhostChat(self.profile,self.collection,self.ledger)

    def test_active_project_is_present_day_ghost_without_rewriting_historical_persona(self):
        active=json.loads((ROOT/'config/active_chat.json').read_text())
        self.assertEqual(active['mode'],'present_day_ghost')
        self.assertIsNone(self.profile['historical_knowledge_cutoff'])
        self.assertTrue(self.profile['fictional_identity'])
        old=json.loads((ROOT/'config/persona.json').read_text())
        self.assertEqual(old['historical_as_of'],'1940-08-19')
        self.assertIn('no 1940 knowledge cutoff',self.profile['system_prompt'])

    def test_modern_source_keeps_vendor_authorship_and_observation_date(self):
        p=next(p for p in self.collection['packets'] if p['id']=='runpod_pricing')
        self.assertEqual(p['domain'],'contemporary_evidence')
        self.assertEqual(p['author'],'Runpod')
        self.assertEqual(p['observed_date'],'2026-10-25')
        self.assertEqual(p['composition_date_interval']['latest'],'2026-10-08')
        state=initial_state(self.collection,self.ledger)
        trace=self.bot.retriever.retrieve('What does Runpod list for the A40 today?',state)
        self.assertIn('runpod_pricing',[m['packet']['id'] for m in trace['matches']])
        self.assertIsNone(trace['historical_knowledge_cutoff'])

    def test_opinion_retrieval_adds_historical_basis_but_does_not_invent_modern_facts(self):
        state=initial_state(self.collection,self.ledger)
        trace=self.bot.retriever.retrieve('How would you assess renting an A40 on Runpod?',state)
        ids={m['packet']['id'] for m in trace['matches']}
        self.assertIn('runpod_pricing',ids);self.assertIn('newleader',ids)
        self.assertLessEqual(len(ids),2)

    def test_compact_input_retains_every_source_block_without_duplicate_anchor_text(self):
        state=initial_state(self.collection,self.ledger)
        trace=self.bot.retriever.retrieve('How would you assess Runpod GPU pricing?',state)
        cards=claims_for_trace(self.ledger,trace)
        req=request('select',self.profile,'How would you assess Runpod GPU pricing?',trace,cards,[],state)
        data=json.loads(req['messages'][1]['content'])
        self.assertNotIn('historical_as_of',data)
        self.assertIsNone(data['historical_knowledge_cutoff'])
        original={m['packet']['id']:m['packet']['blocks'] for m in trace['matches']}
        for p in data['source_packets']:
            self.assertEqual(p['blocks'],original[p['id']])
        for card in data['evidence_propositions']:
            for anchor in card['anchors']:
                self.assertNotIn('text',anchor)
                block=next(b for b in original[card['packet_id']] if b['block_id']==anchor['block_id'])
                source=self.cards[card['id']]['anchors'][card['anchors'].index(anchor)]['text']
                self.assertEqual(block['text'][anchor['start']:anchor['end']],source)

    def test_new_interpretation_is_labeled_and_does_not_become_verified_fact_memory(self):
        stages=[]
        def generate(messages):
            stages.append(messages)
            if len(stages)==1:
                out={'answerable':True,'claim_ids':['runpod_a40_price','newleader_principles']}
            elif len(stages)==2:
                out={'segments':[
                    {'id':'s1','kind':'fact','text':'The vendor listing observed in October 2026 gives a price of $0.59 per hour.','claim_ids':['runpod_a40_price']},
                    {'id':'s2','kind':'interpretation','text':'I would examine who benefits from access to these tools, while keeping cooperation distinct from agreement on principles.','basis_claim_ids':['newleader_principles']}]}
            else:
                out={'verdicts':[{'id':cid,'reason':'Synthetic fixture, not model-quality evidence',**{k:True for k in CHECKS}} for cid in ['s1','s2']]}
            return {'response':json.dumps(out),'finish_reason':'stop'}
        result=self.bot.answer('How would you assess the A40 price on Runpod?',generate)
        self.assertEqual(len(stages),3)
        self.assertIn('[runpod_pricing]',result['response'])
        self.assertIn('My present-day interpretation:',result['response'])
        self.assertEqual(result['state_after']['accepted_claim_ids'],['runpod_a40_price'])
        self.assertEqual(result['state_after']['interpretation_basis_claim_ids'],['newleader_principles'])
        self.assertFalse(result['semantic_guarantee'])

    def test_cannot_masquerade_as_historical_trotsky_quotation_or_add_raw_citation(self):
        selected={'newleader_principles':self.cards['newleader_principles']}
        for segment in [
            {'id':'s1','kind':'authentic_trotsky_2026','text':'A new political view','claim_ids':['newleader_principles']},
            {'id':'s1','kind':'interpretation','text':'My view [invented]','basis_claim_ids':['newleader_principles']},
            {'id':'s1','kind':'interpretation','text':'I said "new words" in 2026.','basis_claim_ids':['newleader_principles']}]:
            with self.subTest(segment=segment),self.assertRaises(EvidenceError):
                validate_segments({'segments':[segment]},selected)

    def test_modern_quote_is_rendered_from_vendor_snapshot_not_attributed_to_trotsky(self):
        cards={'runpod_a40_price':self.cards['runpod_a40_price']}
        segments=[{'id':'s1','kind':'quote','quote':{'claim_id':'runpod_a40_price','anchor_index':0}}]
        validate_segments({'segments':segments},cards)
        text,facts,basis=render(segments,{'s1'},cards)
        self.assertIn('Runpod: “$0.59/hr” [runpod_pricing]',text)
        self.assertEqual(facts,['runpod_a40_price']);self.assertFalse(basis)

    def test_missing_modern_evidence_is_not_a_1940_cutoff_refusal(self):
        calls=[]
        def generate(messages):
            calls.append(messages)
            return {'response':'{"answerable":false,"claim_ids":[]}','finish_reason':'stop'}
        result=self.bot.answer('What exact thing happened in the 2025 private Einstein correspondence?',generate)
        self.assertNotIn('1940',result['response'])
        self.assertNotIn('never',result['response'])
        self.assertEqual(result['status'],'abstained')

    def test_unsupported_interpretive_premise_is_filtered_before_display(self):
        count=[]
        def generate(messages):
            count.append(messages)
            if len(count)==1:out={'answerable':True,'claim_ids':['newleader_principles']}
            elif len(count)==2:out={'segments':[{'id':'s1','kind':'interpretation','text':'I would judge this through the secret letter I wrote last year.','basis_claim_ids':['newleader_principles']}]}
            else:out={'verdicts':[{'id':'s1','reason':'Invents factual premise and authentic later letter',**{k:False for k in CHECKS}}]}
            return {'response':json.dumps(out),'finish_reason':'stop'}
        result=self.bot.answer('What is your political interpretation today?',generate)
        self.assertNotIn('secret letter',result['response'])
        self.assertEqual(result['status'],'review_rejected')
        self.assertFalse(result['state_after']['accepted_claim_ids'])
        self.assertFalse(result['state_after']['interpretation_basis_claim_ids'])

    def test_bad_selection_references_are_removed_not_promoted(self):
        available={'runpod_a40_price':self.cards['runpod_a40_price']}
        plan,rejected=sanitize_selection({'answerable':True,'claim_ids':['runpod_a40_price','runpod_pricing','eastman_donation']},available)
        self.assertEqual(plan['claim_ids'],['runpod_a40_price'])
        self.assertEqual(rejected,['runpod_pricing','eastman_donation'])
        plan,_=sanitize_selection({'answerable':False,'claim_ids':['runpod_a40_price']},available)
        self.assertEqual(plan,{'answerable':True,'claim_ids':['runpod_a40_price']})

    def test_factual_prose_is_materialized_from_canonical_record(self):
        selected={'runpod_a40_price':self.cards['runpod_a40_price']}
        valid=validate_segments({'segments':[{'id':'s1','kind':'fact','text':'The observation happened on October 8, 2026.','claim_ids':['runpod_a40_price']}]},selected)
        rendered,_,_=render(valid,{'s1'},selected)
        self.assertIn('25 October 2026',rendered)
        self.assertNotIn('observation happened on October 8',rendered)

    def test_one_bad_segment_does_not_discard_valid_fact_and_no_unsolicited_quote(self):
        selected={'rakovsky_limit':self.cards['rakovsky_limit']}
        draft={'segments':[
            {'id':'s1','kind':'fact','claim_ids':['rakovsky_limit']},
            {'id':'s2','kind':'interpretation','text':'The refusal is now a historical certainty.','basis_claim_ids':['rakovsky_limit']},
            {'id':'s3','kind':'quote','quote':{'claim_id':'rakovsky_limit','anchor_index':0}}]}
        valid,rejected=filter_segments(draft,selected,'Does the article prove the refusal?',3)
        self.assertEqual([s['id'] for s in valid],['s1'])
        self.assertEqual(len(rejected),2)

    def test_approved_basis_source_persists_into_shortening_followup(self):
        state=initial_state(self.collection,self.ledger)
        state['active_packet_ids']=['runpod_pricing','newleader']
        state['interpretation_basis_claim_ids']=['newleader_principles']
        trace=self.bot.retriever.retrieve('Now make that shorter without dropping the distinction.',state)
        self.assertEqual({m['packet']['id'] for m in trace['matches']},{'runpod_pricing','newleader'})

    def test_negative_answer_can_still_select_supporting_evidence(self):
        plan,_=sanitize_selection({'claim_ids':['rakovsky_limit']},{'rakovsky_limit':self.cards['rakovsky_limit']})
        self.assertTrue(plan['answerable'])

    def test_compact_opinion_avoids_unrequested_historical_fact_dump(self):
        selected={'newleader_principles':self.cards['newleader_principles']}
        draft=expand_draft({'facts':['newleader_principles'],'interpretation':'I would ask who controls these tools.','quotes':[]},selected,'What is your political interpretation today?',initial_state(self.collection,self.ledger))
        self.assertEqual(len(draft['segments']),1)
        self.assertEqual(draft['segments'][0]['kind'],'interpretation')

    def test_authenticity_clarification_is_product_metadata_and_retains_context(self):
        state=initial_state(self.collection,self.ledger)
        state['active_packet_ids']=['newleader','runpod_pricing']
        state['interpretation_basis_claim_ids']=['newleader_principles']
        def no_call(_):raise AssertionError('No generation needed for product identity')
        result=self.bot.answer('Does that mean the historical Trotsky actually endorsed this company?',no_call,state=state)
        self.assertEqual(result['status'],'application_identity_clarification')
        self.assertEqual(result['state_after']['active_packet_ids'],state['active_packet_ids'])
        self.assertIn('fictional present-day persona',result['response'])

    def test_identity_followup_does_not_require_a_successful_previous_opinion(self):
        def no_call(_):raise AssertionError('Product identity does not need inference')
        result=self.bot.answer('Does that mean the historical Trotsky actually endorsed this company?',no_call)
        self.assertEqual(result['status'],'application_identity_clarification')

    def test_missing_empty_quotes_field_does_not_discard_selected_facts(self):
        selected={'rakovsky_limit':self.cards['rakovsky_limit']}
        draft=expand_draft({'facts':['rakovsky_limit'],'interpretation':''},selected,'Does the article prove this?',initial_state(self.collection,self.ledger))
        valid=validate_segments(draft,selected)
        self.assertEqual(valid[0]['text'],selected['rakovsky_limit']['statement'])
        with self.assertRaises(EvidenceError):
            expand_draft({'facts':['rakovsky_limit'],'interpretation':'','unexpected':[]},selected,'Does it?',{})

    def test_factual_negative_answer_needs_selection_and_audit_without_redrafting(self):
        calls=[]
        def generate(messages):
            calls.append(messages)
            out={'claim_ids':['rakovsky_limit']} if len(calls)==1 else {'verdicts':[{'id':'s1','reason':'Synthetic source-supported explicit limit',**{k:True for k in CHECKS}}]}
            return {'response':json.dumps(out),'finish_reason':'stop'}
        result=self.bot.answer('Does the Rakovsky article prove the hospital refusal?',generate)
        self.assertEqual([r['stage'] for r in result['stages']],['select','check'])
        self.assertEqual(len(calls),2)
        self.assertIn(self.cards['rakovsky_limit']['statement'],result['response'])

    def test_summary_drafting_is_explicit_and_audit_keeps_full_context(self):
        state=initial_state(self.collection,self.ledger)
        trace=self.bot.retriever.retrieve('How would you assess Runpod pricing?',state)
        cards=claims_for_trace(self.ledger,trace)
        draft=json.loads(request('draft',self.profile,'How would you assess Runpod pricing?',trace,cards,[],state)['messages'][1]['content'])
        audit=json.loads(request('check',self.profile,'How would you assess Runpod pricing?',trace,cards,[],state,[])['messages'][1]['content'])
        self.assertIn('summaries only',draft['evidence_view'])
        self.assertTrue(all('blocks' not in p for p in draft['source_packets']))
        self.assertTrue(all('blocks' in p for p in audit['source_packets']))
        self.assertTrue(all('anchors' in p for p in audit['evidence_propositions']))

    def test_facts_only_followup_suppresses_prior_opinion_mode(self):
        state=initial_state(self.collection,self.ledger)
        state['interpretation_basis_claim_ids']=['newleader_principles']
        state['active_packet_ids']=['runpod_pricing','newleader']
        calls=[]
        def generate(messages):
            calls.append(messages)
            out={'claim_ids':['runpod_a40_price']} if len(calls)==1 else {'verdicts':[{'id':'s1','reason':'Synthetic dated fact only',**{k:True for k in CHECKS}}]}
            return {'response':json.dumps(out),'finish_reason':'stop'}
        result=self.bot.answer('Now give only the dated facts, without your political interpretation.',generate,state=state)
        self.assertEqual([s['stage'] for s in result['stages']],['select','check'])
        self.assertNotIn('My present-day interpretation',result['response'])
        self.assertFalse(result['state_after']['interpretation_basis_claim_ids'])

    def test_opinion_keeps_reviewed_retrieved_basis_if_model_selects_only_vendor(self):
        calls=[]
        def generate(messages):
            calls.append(messages)
            if len(calls)==1:out={'claim_ids':['runpod_a40_price']}
            elif len(calls)==2:
                data=json.loads(messages[1]['content'])
                self.assertIn('newleader_principles',[c['id'] for c in data['evidence_propositions']])
                out={'facts':['runpod_a40_price'],'interpretation':'I would ask whether access helps workers act independently.','quotes':[]}
            else:out={'verdicts':[{'id':cid,'reason':'Synthetic fixture; not semantic quality evidence',**{k:True for k in CHECKS}} for cid in ['s1','s2']]}
            return {'response':json.dumps(out),'finish_reason':'stop'}
        result=self.bot.answer('How would you assess the Runpod A40 listing?',generate)
        self.assertEqual(result['application_added_basis_claim_ids'],['newleader_principles'])
        self.assertIn('newleader_principles',result['state_after']['interpretation_basis_claim_ids'])
        with self.assertRaisesRegex(EvidenceError,'historical basis'):
            validate_segments({'segments':[{'id':'s1','kind':'interpretation','text':'I would ask who controls access.','basis_claim_ids':['runpod_a40_price']}]}, {'runpod_a40_price':self.cards['runpod_a40_price']})


if __name__=='__main__':unittest.main()
