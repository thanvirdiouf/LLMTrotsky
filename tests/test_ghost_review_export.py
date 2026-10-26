import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from export_ghost_review import build_packet
from grounded_chat import fingerprint


class ReviewExportTest(unittest.TestCase):
    def fixture(self,root):
        packet={'id':'source','title':'Fixture source','author':'Document author','domain':'historical_record',
            'url':'https://example.invalid/source','composition_date_interval':{'earliest':'1926-03-01','latest':'1926-03-01'},
            'composition_date_evidence':[{'kind':'delivery_date','text':'1 March 1926'}],
            'context_boundary':'Complete fixture paragraph','blocks':[{'block_id':1,'speaker':'Document author','text':'A dated prediction, not a later outcome.'}]}
        spec={'cases':[{'id':'case','question':'What does this source establish?'}],'conversation':['Explain its limit.','Make that shorter.']}
        payload={n:json.dumps(v).encode() for n,v in {'collection.json':{'reference_date':'2026-10-26','packets':[packet]},'model_cases.json':spec}.items()}
        rows=[]
        for i,(cid,turn) in enumerate([('case',1),('ghost_conversation',1),('ghost_conversation',2)]):
            response='Exact displayed reply '+str(i)
            rows.append({'case_id':cid,'turn':turn,'key':f'{cid}/{turn}','response':response,'response_sha256':fingerprint(response),
                'status':'SECRET_PRIOR_STATUS','stages':[{'output':{'response':'SECRET_MODEL_VERDICT'}}],
                'retrieval':{'matches':[{'packet':copy.deepcopy(packet)}]}})
        run=root/'evaluation/runs/ghost_smoke_v99';(run/'download/ghost_smoke_results').mkdir(parents=True)
        (run/'download/ghost_smoke_bundle').mkdir()
        for n,b in payload.items():(run/'download/ghost_smoke_bundle'/n).write_bytes(b)
        self.write_rows(run,rows)
        return run,payload,rows

    def write_rows(self,run,rows):
        raw=('\n'.join(json.dumps(r) for r in rows)+'\n').encode()
        (run/'download/ghost_smoke_results/answers.jsonl').write_bytes(raw)
        (run/'verification.json').write_text(json.dumps({'status':'verified','answers_sha256':hashlib.sha256(raw).hexdigest()}))

    def test_blinding_preserves_exact_sources_and_actual_conversation_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,payload,rows=self.fixture(root)
            with patch('export_ghost_review.frozen_payloads',return_value=payload):packet,mapping=build_packet('ghost_smoke_v99',root)
            text=json.dumps(packet)
            self.assertNotIn('SECRET_PRIOR_STATUS',text);self.assertNotIn('SECRET_MODEL_VERDICT',text)
            self.assertEqual(len(packet['groups']),2)
            self.assertEqual([t['displayed_response'] for t in packet['groups'][1]['turns']],[rows[1]['response'],rows[2]['response']])
            self.assertEqual(packet['source_packets'][0]['blocks'][0]['speaker'],'Document author')
            self.assertEqual(mapping['mapping'][2]['run_key'],'ghost_conversation/2')

    def test_tampered_reply_or_source_cannot_be_exported_as_verified(self):
        for mutation in ('journal','response_hash','source','calibration'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);run,payload,rows=self.fixture(root)
                if mutation=='source':rows[0]['retrieval']['matches'][0]['packet']['author']='Wrong author'
                elif mutation=='calibration':rows[0]['mode']='reviewer_calibration'
                else:rows[0]['response']='changed text'
                self.write_rows(run,rows)
                if mutation=='journal':(run/'download/ghost_smoke_results/answers.jsonl').write_text('tampered after verification')
                with patch('export_ghost_review.frozen_payloads',return_value=payload),self.assertRaises(ValueError):build_packet('ghost_smoke_v99',root)
