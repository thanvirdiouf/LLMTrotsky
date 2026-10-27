import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from ghost_chat import load_active
from ghost_evidence import evidence_view,render_evidence,literal
from grounded_chat_v2 import EvidenceError,initial_state

class GhostEvidenceTest(unittest.TestCase):
    def setUp(self):self.bot=load_active()

    def test_original_context_roles_and_certainty_retained_without_inference_or_memory(self):
        state=initial_state(self.bot.collection,self.bot.ledger);state['turns']=3;state['active_packet_ids']=['hearing']
        original=copy.deepcopy(state)
        view=evidence_view(self.bot,packet_ids=['hearing','ilo_genai_2025'],state=state)
        self.assertEqual(state,original);self.assertEqual(view['state_before'],view['state_after'])
        self.assertEqual(view['source_packets'],[next(p for p in self.bot.collection['packets'] if p['id']==pid) for pid in ['hearing','ilo_genai_2025']])
        self.assertEqual(view['model_calls'],0);self.assertIsNone(view['generated_answer']);self.assertFalse(view['semantic_verification'])
        output=render_evidence(view)
        for packet in view['source_packets']:
            for block in packet['blocks']:
                self.assertIn(block['text'],output);self.assertIn(block['speaker'],output)
        self.assertIn('1897, I should have said',output);self.assertIn('potential exposure',output)
        view['source_packets'][0]['blocks'][0]['text']='mutated copy'
        self.assertNotEqual(next(p for p in self.bot.collection['packets'] if p['id']=='hearing')['blocks'][0]['text'],'mutated copy')

    def test_retrieval_candidates_are_not_asserted_to_answer_question(self):
        view=evidence_view(self.bot,'What does Bessedovsky prove about an unnamed informant?')
        self.assertIn('bessedovsky',view['retrieval']['packet_ids'])
        self.assertIn('not a generated answer',view['notice']);self.assertFalse(view['semantic_verification'])
        for ids in [[],['unknown'],['hearing','hearing'],['hearing','eastman','blumkin'],[True]]:
            with self.subTest(ids=ids),self.assertRaises(EvidenceError):evidence_view(self.bot,packet_ids=ids)
        with self.assertRaises(EvidenceError):evidence_view(self.bot)

    def test_literal_source_markup_cannot_close_display_fence(self):
        text='source\n~~~\n<script>untrusted source text</script>\n~~~~~'
        result=literal(text);fence=result.splitlines()[0][:-4]
        self.assertEqual(fence,'~~~~~~');self.assertEqual(result.splitlines()[-1],fence)
        self.assertIn(text,result)

    def test_cli_source_inspection_requires_no_endpoint_and_keeps_json_clean(self):
        out=subprocess.run([sys.executable,'scripts/ghost_chat.py','--evidence-only','--source-id','hearing','--source-format','json'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(out.returncode,0,out.stderr)
        view=json.loads(out.stdout);self.assertEqual(view['model_calls'],0)
        self.assertEqual([p['id'] for p in view['source_packets']],['hearing'])
        out=subprocess.run([sys.executable,'scripts/ghost_chat.py','--source-id','hearing'],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(out.returncode,0);self.assertIn('requires --evidence-only',out.stderr)

if __name__=='__main__':unittest.main()
