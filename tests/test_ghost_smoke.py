import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from ghost_chat import load_active
from evaluate_ghost_smoke import evaluate


class GhostSmokeTest(unittest.TestCase):
    def test_replay_reuses_stage_calls_and_real_displayed_history(self):
        bot=load_active();spec={'cases':[],'conversation':['What does Runpod list?','Does that prove a meeting never happened?']}
        calls=[]
        def generate(messages):
            calls.append(messages)
            return {'response':'{"answerable":false,"claim_ids":[]}','finish_reason':'stop'}
        with tempfile.TemporaryDirectory() as temp:
            answers,records=evaluate(bot.profile,bot.collection,bot.ledger,spec,generate,Path(temp),{'test':True})
            self.assertEqual(len(answers),2);self.assertEqual(len(calls),2)
            data=json.loads(calls[1][1]['content'])
            self.assertEqual(data['conversation'][1]['content'],answers[0]['response'])
            again,_=evaluate(bot.profile,bot.collection,bot.ledger,spec,generate,Path(temp),{'test':True})
            self.assertEqual(again,answers);self.assertEqual(len(calls),2)
            records[0]['output']['response']='tampered'
            path=Path(temp)/'model_calls.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(ValueError):evaluate(bot.profile,bot.collection,bot.ledger,spec,generate,Path(temp),{'test':True})

    def test_budget_block_is_recorded_without_becoming_a_factual_answer(self):
        bot=load_active();spec={'cases':[{'id':'memory','question':'What does Runpod list for the A40?'}],'conversation':[]}
        with tempfile.TemporaryDirectory() as temp:
            answers,calls=evaluate(bot.profile,bot.collection,bot.ledger,spec,
                lambda _:{'response':'','finish_reason':'input_budget','model_call_made':False},Path(temp),{})
            self.assertEqual(answers[0]['status'],'blocked')
            self.assertFalse(calls[0]['output']['model_call_made'])
            self.assertFalse(answers[0]['state_after']['accepted_claim_ids'])

    def test_runtime_has_no_training_or_adapter_attachment_calls(self):
        tree=ast.parse((ROOT/'scripts/kaggle_ghost_smoke.py').read_text())
        calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(calls & {'backward','step','get_peft_model','save_pretrained','load_checkpoint'})

    def test_frozen_bundle_has_no_expected_answers_and_versions_cannot_escape_archive(self):
        from verify_ghost_smoke import frozen_payloads
        payload=frozen_payloads();cases=json.loads(payload['model_cases.json'])
        self.assertFalse(any('checks' in c for c in cases['cases']))
        self.assertEqual(json.loads(payload['protocol.json'])['answers'],10)
        with self.assertRaises(ValueError):frozen_payloads(version='../ghost_smoke_v1')


if __name__=='__main__':unittest.main()
