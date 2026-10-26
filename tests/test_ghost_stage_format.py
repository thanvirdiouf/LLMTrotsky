import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from ghost_chat import load_active,parse_stage_object,EvidenceError
from grounded_chat_v2 import CHECKS,parse_object
from grounded_chat import fingerprint


class StageFormatTest(unittest.TestCase):
    def test_planner_postface_is_discarded_without_promoting_it_to_an_answer(self):
        plan={'answer_mode':'facts','detail':'normal','claim_ids':['runpod_a40_price']}
        extra='\n\nPlanning note: the entire bill and tomorrow availability are guaranteed.'
        calls=[]
        def generate(messages):
            calls.append(messages)
            if len(calls)==1:return {'response':json.dumps(plan)+extra,'finish_reason':'stop'}
            data=json.loads(messages[1]['content'])
            self.assertNotIn('tomorrow availability',json.dumps(data))
            verdicts=[{'id':s['id'],'reason':'Synthetic fixture only',**{k:True for k in CHECKS}} for s in data['candidate_segments']]
            return {'response':json.dumps({'verdicts':verdicts})+'`','finish_reason':'stop'}
        result=load_active().answer('What hourly A40 compute rate does the supplied Runpod listing document?',generate)
        self.assertEqual(len(calls),2)
        self.assertIn('$0.59',result['response'])
        self.assertNotIn('guaranteed',result['response'])
        self.assertEqual([s['structured_output_normalization']['kind'] for s in result['stages']],
                         ['discarded_selection_postface','discarded_trailing_markdown_delimiter'])
        event=result['stages'][0]['structured_output_normalization']
        self.assertEqual(event['discarded_suffix_fingerprint'],fingerprint(extra))
        self.assertEqual(result['stages'][0]['output']['response'],json.dumps(plan)+extra)
        self.assertFalse(result['semantic_guarantee'])

    def test_check_normalization_cannot_turn_rejection_into_approval(self):
        calls=[]
        def generate(messages):
            calls.append(messages)
            if len(calls)==1:return {'response':'{"answer_mode":"facts","detail":"normal","claim_ids":["runpod_a40_price"]}','finish_reason':'stop'}
            data=json.loads(messages[1]['content'])
            return {'response':json.dumps({'verdicts':[{'id':s['id'],'reason':'Rejected synthetic candidate',**{k:False for k in CHECKS}} for s in data['candidate_segments']]})+'```','finish_reason':'stop'}
        result=load_active().answer('What hourly A40 rate is documented in the Runpod listing?',generate)
        self.assertEqual(result['accepted_segment_ids'],[])
        self.assertEqual(result['state_after']['accepted_claim_ids'],[])
        self.assertNotIn('$0.59',result['response'])

    def test_ambiguous_or_invalid_json_remains_blocked(self):
        invalid=[
            '{"answer_mode":"facts","answer_mode":"view"}\nPlanner note follows here.',
            '{"answer_mode":NaN}\nPlanner note follows here.',
            '{"answer_mode":"facts"',
            '{"answer_mode":"facts"}\n{"answer_mode":"view"}',
            '{"answer_mode":"facts"}\nAnother array ["view"] follows.',
            '{"answer_mode":"facts"} false',
            'Before the object {"answer_mode":"facts"}',
            '[{"answer_mode":"facts"}]`',
        ]
        for text in invalid:
            with self.subTest(text=text),self.assertRaises(EvidenceError):parse_stage_object(text,'select')
        for stage in ('check','draft'):
            with self.subTest(stage=stage),self.assertRaises(EvidenceError):
                parse_stage_object('{"verdicts":[]}\nActually reject every candidate now.',stage)
        with self.assertRaises(EvidenceError):parse_stage_object('{"facts":[]}`','draft')

    def test_original_strict_parser_and_object_strings_are_unchanged(self):
        text='{"value":"A literal } and [ character within a string"}'
        self.assertEqual(parse_stage_object(text,'check'),(parse_object(text),None))
        self.assertEqual(parse_stage_object(text+'`','check')[0],parse_object(text))
        with self.assertRaises(EvidenceError):parse_object(text+'`')
        with self.assertRaises(EvidenceError):parse_stage_object('{"verdicts":[],"verdicts":[]}`','check')

if __name__=='__main__':unittest.main()
