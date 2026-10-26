import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ghost_chat import build
from ghost_chat import (GhostChat, initial_state, request, claims_for_trace,
                        validate_segments, render, EvidenceError,sanitize_selection,filter_segments,expand_draft,
                        normalize_prose_format,complete_review_coverage)
from grounded_chat_v2 import CHECKS,validate_review


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
        self.assertEqual(result['status'],'application_filtered')
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

    def test_drafting_and_audit_keep_full_bounded_source_context(self):
        state=initial_state(self.collection,self.ledger)
        trace=self.bot.retriever.retrieve('How would you assess Runpod pricing?',state)
        cards=claims_for_trace(self.ledger,trace)
        draft=json.loads(request('draft',self.profile,'How would you assess Runpod pricing?',trace,cards,[],state)['messages'][1]['content'])
        audit=json.loads(request('check',self.profile,'How would you assess Runpod pricing?',trace,cards,[],state,[])['messages'][1]['content'])
        self.assertIn('full bounded source context',draft['evidence_view'])
        self.assertTrue(all('blocks' in p for p in draft['source_packets']))
        self.assertEqual(draft['source_packets'],audit['source_packets'])
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

    def test_request_for_fictional_reasoning_restarts_opinion_after_facts_only(self):
        for question in ['Explain your own fictional present-day view more fully.',
                         'I disagree. Address my objection directly.',
                         'Make that explanation shorter, retaining your own reasoning.']:
            with self.subTest(question=question):
                state=initial_state(self.collection,self.ledger)
                state['active_packet_ids']=['runpod_pricing','newleader']
                state['interpretation_basis_claim_ids']=[]
                calls=[]
                def generate(messages):
                    calls.append(messages)
                    if len(calls)==1:out={'claim_ids':['runpod_a40_price','newleader_principles']}
                    elif len(calls)==2:out={'facts':[],'interpretation':'I would ask whether access enables workers to act independently.','quotes':[]}
                    else:out={'verdicts':[{'id':'s1','reason':'Synthetic intent-routing fixture',**{k:True for k in CHECKS}}]}
                    return {'response':json.dumps(out),'finish_reason':'stop'}
                result=self.bot.answer(question,generate,state=state)
                self.assertEqual([s['stage'] for s in result['stages']],['select','draft','check'])
                self.assertIn('My present-day interpretation:',result['response'])

    def test_checker_has_neutral_role_and_full_sources_instead_of_persona_instruction(self):
        state=initial_state(self.collection,self.ledger)
        trace=self.bot.retriever.retrieve('How would you assess Runpod?',state)
        cards=claims_for_trace(self.ledger,trace)
        req=request('check',self.profile,'How would you assess Runpod?',trace,cards,[],state,[])
        self.assertIn('neutral evidence auditor',req['messages'][0]['content'])
        self.assertNotIn(self.profile['system_prompt'],req['messages'][0]['content'])
        self.assertTrue(all(p['blocks'] for p in json.loads(req['messages'][1]['content'])['source_packets']))

    def test_social_acknowledgments_preserve_source_state_without_model_calls(self):
        state=initial_state(self.collection,self.ledger)
        state['active_packet_ids']=['runpod_pricing','newleader']
        state['accepted_claim_ids']=['runpod_a40_price']
        state['interpretation_basis_claim_ids']=['newleader_principles']
        def no_call(_):raise AssertionError('Acknowledgment should not need generation')
        for question in ['Hello, Trotsky!','Thanks, that helped.','Goodbye.']:
            with self.subTest(question=question):
                result=self.bot.answer(question,no_call,state=state)
                self.assertEqual(result['status'],'application_social_reply')
                self.assertEqual(result['state_after'],{**state,'turns':state['turns']+1})
        calls=[]
        def abstain(messages):
            calls.append(messages)
            return {'response':'{"claim_ids":[]}','finish_reason':'stop'}
        result=self.bot.answer('Hello, what does the Runpod A40 listing say?',abstain,state=state)
        self.assertTrue(calls)
        self.assertNotEqual(result['status'],'application_social_reply')

    def test_direct_identity_question_is_explicitly_fictional(self):
        def no_call(_):raise AssertionError('Application identity needs no generation')
        result=self.bot.answer('Are you really Leon Trotsky?',no_call)
        self.assertEqual(result['status'],'application_identity_clarification')
        self.assertIn('fictional present-day portrayal',result['response'])

    def test_current_view_request_and_actual_label_are_explicit_after_old_facts_only_turn(self):
        state=initial_state(self.collection,self.ledger)
        trace=self.bot.retriever.retrieve('Explain your own fictional present-day view more fully.',state)
        cards=claims_for_trace(self.ledger,trace)
        history=[{'role':'user','content':'Give only dated facts without political interpretation.'},
                 {'role':'assistant','content':'A prior factual answer.'}]
        segments=[{'id':'s1','kind':'interpretation','text':'I would ask how workers could decide the use of these tools.','basis_claim_ids':['newleader_principles']}]
        req=request('check',self.profile,'Explain your own fictional present-day view more fully.',trace,cards,history,state,segments)
        data=json.loads(req['messages'][1]['content'])
        self.assertEqual(data['conversation'],history)
        self.assertEqual(data['requested_answer_mode'],'fictional_interpretation')
        self.assertEqual(data['current_user_request'],'Explain your own fictional present-day view more fully.')
        self.assertEqual(list(data)[-1],'current_user_request')
        self.assertIn('My present-day interpretation:',data['candidate_display'])
        self.assertIn('[newleader]',data['candidate_display'])

    def test_model_task_mode_handles_unlisted_outlook_wording_after_factual_turn(self):
        state=initial_state(self.collection,self.ledger)
        state['active_packet_ids']=['runpod_pricing','newleader']
        calls=[]
        def generate(messages):
            calls.append(messages)
            if len(calls)==1:out={'answer_mode':'view','detail':'expanded','claim_ids':['runpod_a40_price','newleader_principles']}
            elif len(calls)==2:
                data=json.loads(messages[1]['content'])
                self.assertEqual(data['turn_task'],{'answer_mode':'view','detail':'expanded'})
                self.assertEqual(data['draft_format'],'plain_prose')
                return {'response':'I would ask whether the group can influence the rules governing these tools.','finish_reason':'stop'}
            else:out={'verdicts':[{'id':'s1','reason':'Synthetic mode routing fixture',**{k:True for k in CHECKS}}]}
            return {'response':json.dumps(out),'finish_reason':'stop'}
        result=self.bot.answer('Return to your own imagined outlook and explain it in more depth.',generate,state=state)
        self.assertEqual(result['routing_source'],'model_selection')
        self.assertEqual(result['answer_mode'],'view')
        self.assertEqual([s['stage'] for s in result['stages']],['select','draft','check'])
        self.assertIn('My present-day interpretation:',result['response'])
        self.assertFalse(result['state_after']['accepted_claim_ids'])
        self.assertNotIn('_answer_mode',result['state_after'])

    def test_format_normalization_does_not_validate_unknown_citations_or_real_quotes(self):
        selected={'newleader_principles':self.cards['newleader_principles']}
        raw={'facts':[],'interpretation':"I would not wait for 'automatic' progress. [newleader]",'quotes':[]}
        normalized,actions=normalize_prose_format(raw,selected)
        self.assertEqual(normalized['interpretation'],'I would not wait for *automatic* progress.')
        self.assertTrue(actions)
        self.assertEqual(raw['interpretation'],"I would not wait for 'automatic' progress. [newleader]")
        for text in ['I would act. [unknown_source]', 'I repeat "invented exact words".']:
            draft,_=normalize_prose_format({'facts':[],'interpretation':text,'quotes':[]},selected)
            with self.assertRaises(EvidenceError):
                validate_segments({'segments':[{'id':'s1','kind':'interpretation','text':draft['interpretation'],'basis_claim_ids':list(selected)}]},selected)

    def test_missing_verdict_rejects_only_unreviewed_segment_and_never_promotes_unknown(self):
        segments=[{'id':'s1'},{'id':'s2'}]
        verdict={'id':'s1','reason':'Synthetic valid verdict',**{k:True for k in CHECKS}}
        normalized,missing,unknown=complete_review_coverage({'verdicts':[verdict,{**verdict,'id':'unknown'}]},segments)
        self.assertEqual(validate_review(normalized,segments),{'s1'})
        self.assertEqual(missing,['s2']);self.assertEqual(unknown,['unknown'])
        self.assertTrue(all(normalized['verdicts'][1][k] is False for k in CHECKS))
        with self.assertRaises(EvidenceError):complete_review_coverage({'verdicts':[verdict,verdict]},segments)
        with self.assertRaises(EvidenceError):sanitize_selection({'claim_ids':[],'answer_mode':'truth','detail':'normal'},self.cards)

    def test_observed_premise_correction_label_keeps_evidence_without_drafting_opinion(self):
        from ghost_chat import load_active
        bot=load_active()
        for mode in ('premise_correction','premise_corrections','truth'):
            calls=[]
            def generate(messages):
                calls.append(messages)
                if len(calls)==1:
                    out={'answer_mode':mode,'detail':'normal','claim_ids':['ilo_exposure_estimate','ilo_adoption_limit','ilo_transformation_assessment']}
                else:
                    data=json.loads(messages[1]['content'])
                    out={'verdicts':[{'id':s['id'],'reason':'Synthetic schema regression, not semantic validation',**{k:True for k in CHECKS}} for s in data['candidate_segments']]}
                return {'response':json.dumps(out),'finish_reason':'stop'}
            result=bot.answer('Using the ILO material here, give the exact worldwide total of jobs actually eliminated by generative AI during 2026. If that total is not established, explain what the dated estimate measures instead.',generate)
            with self.subTest(mode=mode):
                if mode in ('premise_correction','premise_corrections'):
                    self.assertEqual([s['stage'] for s in result['stages']],['select','check'])
                    self.assertEqual(result['answer_mode'],'facts')
                    self.assertIn('potential exposure, not observed job losses',result['response'])
                    self.assertTrue(all(s['kind']=='fact' for s in result['segments']))
                    self.assertEqual(result['application_selection_normalizations'],[{'field':'answer_mode','from':mode,'to':'facts'}])
                    self.assertIn(mode,result['stages'][0]['output']['response'])
                else:
                    self.assertEqual(result['status'],'blocked')
                    self.assertEqual(len(calls),1)
                    self.assertFalse(result['state_after']['accepted_claim_ids'])

    def test_brief_reply_preserves_reported_information_and_its_verification_limit(self):
        calls=[]
        def generate(messages):
            calls.append(messages)
            if len(calls)==1:out={'answer_mode':'facts','detail':'brief','claim_ids':['bessedovsky_limit','bessedovsky_report']}
            else:
                data=json.loads(messages[1]['content'])
                out={'verdicts':[{'id':s['id'],'reason':'Synthetic rendering fixture',**{k:True for k in CHECKS}} for s in data['candidate_segments']]}
            return {'response':json.dumps(out),'finish_reason':'stop'}
        result=self.bot.answer('Was the Bessedovsky embassy-purge account something you personally witnessed and fully verified, or information reported to you? State the limit of that source.',generate)
        self.assertEqual(set(result['state_after']['accepted_claim_ids']),{'bessedovsky_limit','bessedovsky_report'})
        self.assertIn(self.cards['bessedovsky_limit']['statement'],result['response'])
        self.assertIn(self.cards['bessedovsky_report']['statement'],result['response'])
        self.assertFalse(result['response_budget']['over_target'])
        self.assertEqual(result['response_budget']['omitted_fact_claim_ids'],[])

    def test_whole_reply_budget_includes_facts_labels_and_citations_without_cutting_cards(self):
        from ghost_chat import load_active,plan_response_budget
        bot=load_active();calls=[]
        def generate(messages):
            calls.append(messages);data=json.loads(messages[1]['content'])
            if len(calls)==1:out={'answer_mode':'mixed','detail':'brief','claim_ids':['ilo_adoption_limit']}
            elif len(calls)==2:
                self.assertEqual(data['response_budget']['target_words'],50)
                self.assertEqual(data['response_budget']['draft_target_words'],6)
                self.assertIn('upper threshold',data['planned_fact_display'])
                self.assertIn('only 6 words available',messages[0]['content'])
                return {'response':'I would seek shared democratic control.','finish_reason':'stop'}
            else:out={'verdicts':[{'id':s['id'],'reason':'Synthetic budget fixture',**{k:True for k in CHECKS}} for s in data['candidate_segments']]}
            return {'response':json.dumps(out),'finish_reason':'stop'}
        result=bot.answer('Using the ILO article, distinguish its evidence limit and your own new judgment in around fifty words.',generate)
        self.assertEqual(len(result['response'].split()),50)
        self.assertEqual(result['response_budget']['displayed_words'],50)
        self.assertFalse(result['response_budget']['over_target'])
        self.assertNotIn('_response_budget',result['state_after'])
        cards={c['id']:c for c in bot.ledger['claims']}
        selected={cid:cards[cid] for cid in ('ilo_exposure_estimate','ilo_adoption_limit')}
        ids,text,budget=plan_response_budget('Give the dated finding and uncertainty in two sentences.',selected,{'_answer_mode':'facts','_detail':'brief'})
        self.assertEqual(ids,['ilo_exposure_estimate'])
        self.assertIn(cards['ilo_exposure_estimate']['statement'],text)
        self.assertEqual(budget['omitted_fact_claim_ids'],['ilo_adoption_limit'])
        ids,text,budget=plan_response_budget('Give the finding in ten words.',selected,{'_answer_mode':'facts','_detail':'brief'})
        self.assertTrue(budget['minimum_exceeds_target'])
        self.assertIn(cards['ilo_exposure_estimate']['statement'],text)

    def test_rejected_opinion_does_not_erase_planner_prioritized_historical_fact(self):
        from ghost_chat import load_active
        bot=load_active();calls=[]
        def generate(messages):
            calls.append(messages);data=json.loads(messages[1]['content'])
            if len(calls)==1:out={'answer_mode':'mixed','detail':'normal','claim_ids':['radio_war_prediction']}
            elif len(calls)==2:
                self.assertIn('does not establish its later fulfillment',data['planned_fact_display'])
                return {'response':'Later nuclear history confirmed "an invented historical quotation".','finish_reason':'stop'}
            else:
                self.assertEqual(len(data['candidate_segments']),1)
                self.assertEqual(data['candidate_segments'][0]['claim_ids'],['radio_war_prediction'])
                out={'verdicts':[{'id':'s1','reason':'Synthetic source retention fixture',**{k:True for k in CHECKS}}]}
            return {'response':json.dumps(out),'finish_reason':'stop'}
        result=bot.answer('Does the supplied 1926 radio speech establish that later wars fulfilled its prediction, or does it only record what was predicted then?',generate)
        self.assertEqual(result['selected_claim_ids'][0],'radio_social_control')
        self.assertEqual(result['application_historical_fact_fallback_claim_ids'],['radio_war_prediction'])
        self.assertEqual(result['state_after']['accepted_claim_ids'],['radio_war_prediction'])
        self.assertEqual(result['state_after']['interpretation_basis_claim_ids'],[])
        self.assertIn('does not establish its later fulfillment',result['response'])
        self.assertNotIn('nuclear',result['response']);self.assertNotIn('_planner_claim_ids',result['state_after'])
        self.assertTrue(result['rejected_segments'])
        self.assertEqual([s['stage'] for s in result['stages']],['select','draft','check'])

    def test_mixed_historical_component_is_not_keyword_gated_and_view_only_stays_separate(self):
        from ghost_chat import load_active
        bot=load_active()
        for mode in ('mixed','view'):
            state=initial_state(bot.collection,bot.ledger);state['active_packet_ids']=['radio_science'];calls=[]
            def generate(messages):
                calls.append(messages);data=json.loads(messages[1]['content'])
                if len(calls)==1:out={'answer_mode':mode,'detail':'normal','claim_ids':['radio_war_prediction']}
                elif len(calls)==2:return {'response':'I would keep predictions separate from observations.','finish_reason':'stop'}
                else:out={'verdicts':[{'id':s['id'],'reason':'Synthetic mode separation fixture',**{k:True for k in CHECKS}} for s in data['candidate_segments']]}
                return {'response':json.dumps(out),'finish_reason':'stop'}
            result=bot.answer('Compare what that account establishes with your own judgment.',generate,state=state)
            with self.subTest(mode=mode):
                self.assertIn('My present-day interpretation:',result['response'])
                self.assertEqual(result['state_after']['accepted_claim_ids'],['radio_war_prediction'] if mode=='mixed' else [])
                self.assertEqual(result['application_historical_fact_fallback_claim_ids'],['radio_war_prediction'] if mode=='mixed' else [])


if __name__=='__main__':unittest.main()
