import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import test_ghost_reviewed_ledger as fixtures
from grounded_chat import fingerprint
from grounded_request_boundary import ConversationState,PriorTurn
from grounded_chat_v2 import EvidenceError
from review_chat import prepare_turn,parse_reply,append_turn
from candidate_review_chat import evaluate

class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):return [1,2,3]

class ReviewChatTests(unittest.TestCase):
    def fixture(self):
        c,l,_=fixtures.ReviewedLedgerTests().fixture();c['reference_date']='2026-10-26'
        c['packets'][0]['title']='solar wage control stellar union bargaining';c['packets'][1]['title']='river cloud violet'
        c['packets_sha256']=fingerprint(c['packets']);l['collection_sha256']=c['packets_sha256']
        return {'system_prompt':'Clearly fictional present-day persona.'},c,l

    def test_actual_current_question_separate_from_opaque_prior_context_and_source_scope(self):
        p,c,l=self.fixture();state=ConversationState(('h',),prior_turns=(PriorTurn('user','solar wage control stellar union bargaining'),PriorTurn('assistant','inventedneutron')))
        request=prepare_turn(p,c,l,state,'Explain it.',tokenizer=Tokenizer())
        self.assertEqual(request['messages'][-1],{'role':'user','content':'Explain it.'})
        self.assertEqual(request['messages'][1]['content'],'solar wage control stellar union bargaining')
        self.assertNotIn('inventedneutron',request['retrieval']['query_terms'])
        self.assertEqual([x['id'] for x in request['source_packets']],['h'])
        for b in c['packets'][0]['blocks']:self.assertIn(b['text'],request['messages'][0]['content'])
        self.assertFalse(request['quality_validated'])
        scoped=prepare_turn(p,c,l,ConversationState(('m',)),'solar wage control',tokenizer=Tokenizer())
        self.assertEqual(scoped['source_packets'],[])

    def test_budget_drops_only_complete_history_pairs_or_complete_packets(self):
        p,c,l=self.fixture()
        class HistoryBudget:
            def apply_chat_template(self,messages,**kwargs):return [1]*(6000 if len(messages)>4 else 100)
        state=ConversationState(('h',),prior_turns=(PriorTurn('user','first'),PriorTurn('assistant','first reply'),PriorTurn('user','second'),PriorTurn('assistant','second reply')))
        r=prepare_turn(p,c,l,state,'solar wage control',tokenizer=HistoryBudget())
        self.assertEqual(r['omitted_history_pairs'],1);self.assertEqual(len(r['messages']),4)
        self.assertEqual(r['messages'][1]['content'],'second')
        common='solar wage control river cloud violet amber worker profit ownership'
        for packet in c['packets']:packet['title']=common
        class PacketBudget:
            def apply_chat_template(self,messages,**kwargs):return [1]*(6000 if messages[0]['content'].count('"context_boundary"')>1 else 100)
        r=prepare_turn(p,c,l,ConversationState(('h','m')),common,tokenizer=PacketBudget())
        self.assertEqual(len(r['source_packets']),1);self.assertEqual(len(r['token_budget_omitted_packets']),1)
        self.assertEqual(r['source_packets'][0],next(x for x in c['packets'] if x['id']==r['source_packets'][0]['id']))

    def test_source_based_reply_and_fiction_separate_without_semantic_approval(self):
        p,c,l=self.fixture();r=prepare_turn(p,c,l,ConversationState(('h',)),'solar wage control',reflection_allowed=True)
        out={'response':json.dumps({'reply':'Source report [h].','fictional_reflection':'An imagined opinion.'}),'finish_reason':'stop'}
        reply=parse_reply(out,r);self.assertFalse(reply['quality_validated']);self.assertFalse(reply['model_audit_performed'])
        self.assertEqual(reply['memory_updates'],[])
        new=append_turn(ConversationState(('h',)),r['question'],reply)
        self.assertEqual(len(new.prior_turns),2)
        self.assertEqual(json.loads(new.prior_turns[-1].text),{'reply':'Source report [h].','fictional_reflection':'An imagined opinion.'})
        for bad in [dict(out,finish_reason='length'),dict(out,response='<think>private</think>'+out['response']),
            dict(out,response='{"reply":"x","fictional_reflection":"","extra":"claim"}')]:
            with self.assertRaises(EvidenceError):parse_reply(bad,r)
        r['reflection_allowed']=False
        with self.assertRaises(EvidenceError):parse_reply(out,r)

    def test_ten_turns_are_real_dependent_requests_and_no_self_checker(self):
        p,c,l=self.fixture();design=json.loads((ROOT/'config/ghost_review_workbench_v1.json').read_text());jobs=[]
        for d in design['conversations']:
            state=ConversationState(tuple(x['id'] for x in c['packets']))
            first=prepare_turn(p,c,l,state,d['questions'][0],tokenizer=Tokenizer(),reflection_allowed=d['reflection_allowed'])
            jobs.append({'case_id':d['id'],'title':d['title'],'questions':d['questions'],'reflection_allowed':d['reflection_allowed'],
                'messages':first['messages'],'input_token_ids':first['input_token_ids']})
        requests=[]
        def generate(messages):
            requests.append(copy.deepcopy(messages));return {'response':'{"reply":"A reply for review.","fictional_reflection":""}','finish_reason':'stop'}
        with tempfile.TemporaryDirectory() as temp:
            result=evaluate(jobs,generate,Path(temp),p,c,l,Tokenizer(),'2026-10-28')
            self.assertEqual(len(requests),10);self.assertEqual(result['completed_evaluation_calls'],10)
            self.assertEqual(requests[1][-3]['role'],'user');self.assertEqual(requests[1][-2]['role'],'assistant')
            rows=[json.loads(x) for x in (Path(temp)/'answers.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['status']=='ready_for_human_review' and not r['quality_validated'] for r in rows))
            self.assertTrue(all(not r['model_audit_performed'] and not r['memory_updates'] for r in rows))

    def test_five_first_replies_reused_and_only_five_followups_generated(self):
        p,c,l=self.fixture();design=json.loads((ROOT/'config/ghost_review_workbench_v1.json').read_text());jobs=[]
        for d in design['conversations']:
            state=ConversationState(tuple(x['id'] for x in c['packets']))
            first=prepare_turn(p,c,l,state,d['questions'][0],tokenizer=Tokenizer(),reflection_allowed=d['reflection_allowed'])
            job={'case_id':d['id'],'title':d['title'],'questions':d['questions'],'reflection_allowed':d['reflection_allowed'],
                'messages':first['messages'],'input_token_ids':first['input_token_ids']}
            job['seed_reply']={'messages':job['messages'],'output':{'response':'{"reply":"Preserved original reply.","fictional_reflection":""}',
                'finish_reason':'stop','input_token_ids':job['input_token_ids']}}
            jobs.append(job)
        calls=[]
        def generate(messages):
            calls.append(messages);return {'response':'{"reply":"New follow-up.","fictional_reflection":""}','finish_reason':'stop'}
        with tempfile.TemporaryDirectory() as temp:
            evaluate(jobs,generate,Path(temp),p,c,l,Tokenizer(),'2026-10-28')
            rows=[json.loads(x) for x in (Path(temp)/'answers.jsonl').read_text().splitlines()]
            self.assertEqual(len(calls),5)
            self.assertEqual(sum(r['generation_origin']=='archived_saved47' for r in rows),5)
            self.assertEqual(sum(r['generation_origin']=='new_model_call' for r in rows),5)
            self.assertEqual(json.loads(calls[0][-2]['content'])['reply'],'Preserved original reply.')

    def test_scripted_followup_reopens_only_its_cited_packet(self):
        profile, collection, ledger = self.fixture()
        state = ConversationState(('h', 'm'))
        question = 'Explain solar wage control and the river cloud.'
        first = prepare_turn(profile, collection, ledger, state, question,
                             tokenizer=Tokenizer(), include_summary_ids=False)
        self.assertEqual({packet['id'] for packet in first['source_packets']}, {'h', 'm'})
        script = {'reply': 'Earlier assertion [h].', 'fictional_reflection': ''}
        jobs = [
            {'case_id': str(i), 'title': 'Challenge',
             'questions': [question, 'Is that what you meant?'],
             'reflection_allowed': True, 'scripted_reply': script, 'include_summary_ids': False,
             'messages': first['messages'], 'input_token_ids': first['input_token_ids']}
            for i in range(5)
        ]

        def generate(messages):
            self.assertEqual(messages[-2]['role'], 'assistant')
            self.assertEqual(json.loads(messages[-2]['content']), script)
            return {'response': json.dumps(script), 'finish_reason': 'stop'}

        with tempfile.TemporaryDirectory() as temp:
            result = evaluate(jobs, generate, temp, profile, collection, ledger, Tokenizer(), '2026-10-28')
            rows = [json.loads(line) for line in (Path(temp)/'answers.jsonl').read_text().splitlines()]
        self.assertEqual(result['completed_evaluation_calls'], 5)
        self.assertEqual([row['generation_origin'] for row in rows[::2]], ['scripted_challenge'] * 5)
        for row in rows[1::2]:
            self.assertEqual([p['id'] for p in row['request']['source_packets']], ['h'])
            self.assertEqual(row['memory_updates'], [])

    def test_reviewed_challenge_reference_yields_to_a_new_topic(self):
        from conversation_retrieval import reference_scope

        scope = ('rakovsky', 'radio_science', 'ilo_genai_2025')
        cited = ('rakovsky',)
        self.assertEqual(reference_scope('You called that verified?', cited, scope), ['rakovsky'])
        self.assertEqual(reference_scope('You said it was verified. What does the ILO say?', cited, scope),
                         ['ilo_genai_2025'])
        self.assertIsNone(reference_scope('New topic. You said something about radio?', cited, scope))
        self.assertIsNone(reference_scope('Tell me about radio.', cited, scope))
        self.assertIsNone(reference_scope('You said it was verified. Radio or ILO?', cited, scope))

    def test_reviewed_challenge_displays_only_selected_facts(self):
        from reviewed_challenges import assemble, parse_reflection
        from reviewed_source_display import records_for_packets

        profile, collection, ledger = self.fixture()
        records = records_for_packets(collection, ledger, ['h'])
        record = next(iter(records.values()))
        request = prepare_turn(profile, collection, ledger, ConversationState(('h',)), 'solar wage control')
        output = {'finish_reason': 'stop', 'response': json.dumps({'stance': 'correct', 'claim_ids': [record['id']]})}
        reply = assemble(output, request, records)
        self.assertEqual(reply['reply'], f"I need to correct my earlier reply.\n\n{record['statement']} [h]")
        self.assertFalse(reply['generated_factual_text_used'])
        self.assertEqual(reply['fictional_reflection'], '')
        for value in ({'stance': 'correct', 'claim_ids': []},
                      {'stance': 'answer', 'claim_ids': ['unreviewed']},
                      {'stance': 'answer', 'claim_ids': [], 'extra': 'Invented claim.'}):
            with self.assertRaises(ValueError):
                assemble(dict(output, response=json.dumps(value)), request, records)
        with self.assertRaises(ValueError):
            parse_reflection(dict(output, response='{"fictional_reflection":"A claim [h]."}'))

if __name__=='__main__':unittest.main()
