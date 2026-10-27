import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from reviewed_source_display import records_for_packets,render_selection,select_ids
from grounded_chat import fingerprint
from grounded_chat_v2 import EvidenceError
from ghost_chat import plan_response_budget


class ReviewedDisplayTests(unittest.TestCase):
    def fixture(self):
        date={'earliest':'1930-01-05','latest':'1930-01-05','precision':'day'}
        packet={'id':'source','author':'Source author','title':'Source title','domain':'historical_record','url':'https://example.org/source',
            'composition_date_interval':date,'composition_date_evidence':[{'text':'Recorded date'}],'context_boundary':'Bounded context',
            'blocks':[{'block_id':1,'speaker':'Actual speaker','role':'body_candidate','text':'Anchored source text.'}]}
        collection={'packets':[packet],'packets_sha256':fingerprint([packet])}
        ledger={'collection_sha256':collection['packets_sha256'],'topic_aliases':{'source':[]},'claims':[
            {'id':'c'+str(i),'packet_id':'source','statement':'Reviewed statement '+str(i)+'.','attribution':'Author reports this; not independently verified',
                'status':'reported_allegation','domain':'historical_record','composition_date_interval':date,
                'anchors':[{'block_id':1,'speaker':'Actual speaker','text':'Anchored source text.'}]} for i in range(6)]}
        return collection,ledger

    def test_preserves_six_selected_statements_and_provenance_without_four_fact_packing_loss(self):
        collection,ledger=self.fixture();records=records_for_packets(collection,ledger,['source']);ids=list(records)
        old,_,budget=plan_response_budget('Explain the source fully.',{c['id']:c for c in ledger['claims']},{'_answer_mode':'facts','_detail':'normal'})
        self.assertEqual(len(old),4);self.assertEqual(budget['omitted_fact_claim_ids'],[])
        result=render_selection(ids,records)
        self.assertEqual(result['rendered_claim_ids'],ids);self.assertEqual(result['omitted_selected_claim_ids'],[])
        self.assertEqual(result['source_records'],list(records.values()))
        for record in records.values():
            self.assertIn(record['statement'],result['response']);self.assertEqual(record['status'],'reported_allegation')
            self.assertEqual(record['source']['author'],'Source author');self.assertEqual(record['source']['composition_date_interval']['earliest'],'1930-01-05')
            self.assertEqual(record['source']['block_refs'][0]['speaker'],'Actual speaker')
        self.assertFalse(result['generated_factual_text_used']);self.assertFalse(result['model_audit_performed'])
        self.assertFalse(result['requested_task_coverage_verified']);self.assertFalse(result['semantic_guarantee'])
        self.assertEqual(result['accepted_claim_ids'],[]);self.assertEqual(result['memory_updates'],[])

    def test_unknown_ids_extra_model_text_and_duplicate_selection_fail_closed(self):
        collection,ledger=self.fixture();records=records_for_packets(collection,ledger,['source'])
        for selection in [{'claim_ids':['unknown']},{'claim_ids':['c0','c0']},{'claim_ids':['c0'],'text':'New unsourced assertion'},
            {'claim_ids':'c0'}]:
            with self.subTest(selection=selection),self.assertRaises(EvidenceError):select_ids(json.dumps(selection),records)
        with self.assertRaises(EvidenceError):select_ids('{"claim_ids":[],"claim_ids":["c0"]}',records)
        empty=render_selection([],records)
        self.assertIn('passages may contain',empty['response']);self.assertFalse(empty['requested_task_coverage_verified'])

    def test_source_statement_status_date_and_speaker_changes_are_detected(self):
        collection,ledger=self.fixture();records=records_for_packets(collection,ledger,['source'])
        for field in ('statement','status','attribution'):
            altered=copy.deepcopy(records);altered['c0'][field]='changed'
            with self.subTest(field=field),self.assertRaises(EvidenceError):render_selection(['c0'],altered)
        for change in ('block','date','speaker'):
            c,l=copy.deepcopy(collection),copy.deepcopy(ledger)
            if change=='block':c['packets'][0]['blocks'][0]['text']='changed'
            elif change=='date':l['claims'][0]['composition_date_interval']={'earliest':'1940-01-01','latest':'1940-01-01'}
            else:l['claims'][0]['anchors'][0]['speaker']='Wrong speaker'
            with self.subTest(change=change),self.assertRaises(EvidenceError):records_for_packets(c,l,['source'])
        with self.assertRaises(EvidenceError):records_for_packets(collection,ledger,['absent'])

    def test_observation_and_publication_dates_remain_separate(self):
        c,l=self.fixture();p=c['packets'][0];p['domain']='contemporary_evidence';p['observed_date']='2026-10-26'
        c['packets_sha256']=fingerprint(c['packets']);l['collection_sha256']=c['packets_sha256']
        for claim in l['claims']:claim['domain']='contemporary_evidence'
        records=records_for_packets(c,l,['source']);source=records['c0']['source']
        self.assertEqual(source['composition_date_interval']['earliest'],'1930-01-05');self.assertEqual(source['observed_date'],'2026-10-26')


if __name__=='__main__':unittest.main()
