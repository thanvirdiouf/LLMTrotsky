import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from review_chat_server import Workbench
from grounded_chat import fingerprint
from build_ghost_candidate_probe import enc,sha
import test_ghost_reviewed_ledger as fixtures

class WorkbenchTests(unittest.TestCase):
    def setup_workbench(self,root):
        collection,ledger,_=fixtures.ReviewedLedgerTests().fixture();collection['reference_date']='2026-10-26'
        collection['packets'][0]['title']='solar wage control stellar union';collection['packets_sha256']=fingerprint(collection['packets']);ledger['collection_sha256']=collection['packets_sha256']
        snapshot=root/'snapshot';snapshot.mkdir()
        for name,value in [('profile.json',{'system_prompt':'A fictional persona.'}),('collection.json',collection),('ledger.json',ledger),('build_manifest.json',{'ledger_fingerprint':fingerprint(ledger)})]:
            (snapshot/name).write_bytes(enc(value))
        return Workbench(snapshot,root/'run',root/'feedback.jsonl')
    def test_source_lookup_never_calls_a_model_and_live_requires_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=self.setup_workbench(Path(tmp));r=w.ask('session',{'question':'solar wage control stellar union','source_ids':['h']})
            self.assertEqual(r['model_calls'],0);self.assertEqual(r['kind'],'source_lookup');self.assertTrue(r['source_packets'])
            self.assertFalse(w.bootstrap()['live_available'])
            with self.assertRaises(ValueError):w.ask('session',{'question':'A live task'},live=True)
            with self.assertRaises(ValueError):w.ask('session',{'question':'Task','base_url':'https://untrusted.example'})
    def test_only_hash_verified_review_records_exposed_and_feedback_saved_locally(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);w=self.setup_workbench(root);w.review_run.mkdir()
            self.assertEqual(w.review_rows(),[])
            row={'case_id':'sample','title':'Example','turn':1,'question':'Review this','status':'ready_for_human_review',
                 'reply':{'reply':'Sample final reply','fictional_reflection':''},'validation_error':None,'model':{'model_id':'test'},'human_review_status':'unreviewed',
                 'request':{'source_packets':[],'retrieval':{},'omitted_history_pairs':0,'token_budget_omitted_packets':[],'messages':[{'role':'system','content':'DO_NOT_EXPOSE_PROMPT'}]},
                 'raw_reasoning':'DO_NOT_EXPOSE_REASONING'}
            output=w.review_run/'download/ghost_review_chat_results';output.mkdir(parents=True)
            raw=(json.dumps(row)+'\n').encode();(output/'answers.jsonl').write_bytes(raw)
            (w.review_run/'verification.json').write_bytes(enc({'status':'verified','frozen_pipeline_replayed':True,'answers_sha256':sha(raw)}))
            self.assertNotIn('DO_NOT_EXPOSE',json.dumps(w.review_rows()))
            visible=w.review_rows()[0]
            body={'case_id':'sample','turn':1,'support':'needs_checking','voice':'mixed','usefulness':'partial','notes':'My review',
                **{k:visible[k] for k in ('review_run','answers_sha256','reply_fingerprint')}}
            result=w.feedback(body)
            self.assertTrue(result['saved']);self.assertTrue((root/'feedback.jsonl').is_file())
            feedback=json.loads((root/'feedback.jsonl').read_text())
            self.assertEqual(feedback['answers_sha256'],sha(raw));self.assertEqual(feedback['review_run'],'run')
            self.assertEqual(feedback['reply_fingerprint'],fingerprint(row['reply']));self.assertFalse(feedback['independent_validation'])
            with self.assertRaises(ValueError):w.feedback({**body,'reply_fingerprint':'stale_reply'})
            (output/'answers.jsonl').write_text('changed')
            with self.assertRaises(ValueError):w.review_rows()

if __name__=='__main__':unittest.main()
