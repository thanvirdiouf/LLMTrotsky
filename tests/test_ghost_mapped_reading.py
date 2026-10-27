import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from ghost_chat import load_active
from mapped_source_reading import MappedSourceReadingChat,validate_map,validate_composition
from evaluate_ghost_comparison import FixedRetriever
from evaluate_ghost_factual_comparison import evaluate
from build_ghost_smoke import model_cases
from grounded_chat_v2 import CHECKS,EvidenceError,initial_state

class MappedReadingTest(unittest.TestCase):
    def setUp(self):
        old=load_active();self.bot=MappedSourceReadingChat(old.profile,old.collection,old.ledger)
        self.bot.retriever=FixedRetriever(self.bot,['ilo_genai_2025'])
        self.trace=self.bot.retriever.retrieve('ILO',initial_state(self.bot.collection,self.bot.ledger))
        p=self.trace['matches'][0]['packet'];b=p['blocks'][0]
        self.mapping={'units':[{'id':'u1','task':'Explain exposure denominator',
            'block_refs':[{'packet_id':p['id'],'block_id':b['block_id']}],'missing_evidence':''},
            {'id':'u2','task':'Explain missing local data','block_refs':[],
             'missing_evidence':'Local occupational data are not supplied.'}]}

    def test_block_membership_metadata_and_missing_support_contract(self):
        value=validate_map(self.mapping,self.trace);a=value['units'][0]['block_refs'][0]
        block=self.trace['matches'][0]['packet']['blocks'][0]
        self.assertEqual(a['speaker'],block['speaker'])
        self.assertIn('block_text_sha256',a)
        self.assertFalse(value['task_coverage_verified']);self.assertFalse(value['units'][0]['semantic_support_verified'])
        variants=[]
        v=copy.deepcopy(self.mapping);v['units'][0]['block_refs'][0]['packet_id']='unknown';variants.append(v)
        v=copy.deepcopy(self.mapping);v['units'][0]['block_refs'][0]['block_id']=99999;variants.append(v)
        v=copy.deepcopy(self.mapping);v['units'][0]['block_refs'][0]['speaker']='invented';variants.append(v)
        v=copy.deepcopy(self.mapping);v['units'][0]['block_refs'][0]['block_id']=True;variants.append(v)
        v=copy.deepcopy(self.mapping);v['units'][0]['block_refs']*=2;variants.append(v)
        v=copy.deepcopy(self.mapping);v['units'][0]['id']='u2';variants.append(v)
        v=copy.deepcopy(self.mapping);v['units'][0]['block_refs']=[];variants.append(v)
        v=copy.deepcopy(self.mapping);v['units'][1]['missing_evidence']='';variants.append(v)
        v=copy.deepcopy(self.mapping);v['units'][0]['answer']='invented proposal';variants.append(v)
        for v in variants:
            with self.subTest(v=v),self.assertRaises(EvidenceError):validate_map(v,self.trace)

    def test_composition_cannot_silently_drop_duplicate_or_reorder_tasks(self):
        evidence=validate_map(self.mapping,self.trace)
        valid={'units':[{'id':'u1','text':'These are exposure estimates.'},{'id':'u2','text':'Local data are missing.'}]}
        parts,_=validate_composition(valid,evidence,self.trace,'Explain exposure and missing local data.')
        self.assertEqual([s['id'] for s in parts],['u1','u2'])
        for units in [valid['units'][:1],valid['units'][::-1],[valid['units'][0]]*2,
                      [{'id':'u1','text':'Unsupported "invented wording".'},valid['units'][1]]]:
            with self.subTest(units=units),self.assertRaises(EvidenceError):validate_composition({'units':units},evidence,self.trace,'Explain.')

    def test_pipeline_rejects_missing_verdict_and_does_not_store_generated_facts(self):
        calls=[]
        def generate(messages):
            data=json.loads(messages[1]['content']);calls.append(data)
            if len(calls)==1:value=self.mapping
            elif len(calls)==2:
                self.assertIn('provisional_task_map',data);self.assertIn('source_packets',data)
                value={'units':[{'id':'u1','text':'The source measures exposure, not observed losses.'},{'id':'u2','text':'Local data are missing.'}]}
            else:
                self.assertFalse(data['provisional_task_map']['task_coverage_verified'])
                value={'verdicts':[{'id':'u1','reason':'Synthetic fixture, not semantic validation',**{k:True for k in CHECKS}}]}
            return {'response':json.dumps(value),'finish_reason':'stop'}
        row=self.bot.answer('Explain the exposure denominator and local data limits.',generate)
        self.assertEqual([s['stage'] for s in row['stages']],['map','draft','check'])
        self.assertEqual(row['missing_verdict_ids'],['u2']);self.assertEqual(row['status'],'partial')
        self.assertNotIn('Local data are missing.',row['response']);self.assertEqual(row['state_after']['accepted_claim_ids'],[])
        self.assertFalse(row['semantic_guarantee'])
        with self.assertRaises(EvidenceError):self.bot.answer('Follow up',generate,state=row['state_after'])

    def test_truncated_or_malformed_map_blocks_before_composition(self):
        for out in [{'response':json.dumps(self.mapping),'finish_reason':'length'},
                    {'response':'{"units":[],"units":[]}','finish_reason':'stop'}]:
            calls=[]
            def generate(messages):calls.append(messages);return out
            row=self.bot.answer('Explain exposure',generate)
            self.assertEqual(len(calls),1);self.assertEqual(row['status'],'blocked')
            self.assertFalse(row['accepted_segment_ids'])

    def test_new_comparison_budget_metadata_and_exact_replay(self):
        spec=model_cases({'factual_comparison':{'arms':['reading','mapped']},'conversation':[],
            'cases':[{'id':f'c{i}','question':f'Explain the ILO estimate, case {i}.','source_packet_ids':['ilo_genai_2025'],'checks':['HIDDEN_EXPECTATION']} for i in range(5)]})
        self.assertNotIn('HIDDEN_EXPECTATION',json.dumps(spec))
        def generate(messages):
            data=json.loads(messages[1]['content']);system=messages[0]['content']
            if system.startswith('Build a task-to-evidence map'):value=self.mapping
            elif system.startswith('Compose a concise factual answer'):value={'units':[{'id':'u1','text':'Exposure is not actual loss.'},{'id':'u2','text':'Local data are missing.'}]}
            elif 'candidate_segments' in data:value={'verdicts':[{'id':s['id'],'reason':'Synthetic only',**{k:True for k in CHECKS}} for s in data['candidate_segments']]}
            else:return {'response':'Exposure is not actual loss.','finish_reason':'stop'}
            return {'response':json.dumps(value),'finish_reason':'stop'}
        with tempfile.TemporaryDirectory() as out:
            rows,calls=evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,spec,generate,out,{'test':True})
            self.assertEqual(len(rows),10);self.assertEqual(len(calls),25)
            replay,_=evaluate(self.bot.profile,self.bot.collection,self.bot.ledger,spec,lambda _:self.fail('Replay inferred'),out,{'test':True})
            self.assertEqual(rows,replay)

if __name__=='__main__':unittest.main()
