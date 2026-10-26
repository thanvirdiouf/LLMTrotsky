import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from selection_constraints import SelectionTrie,candidate_texts,selection_ids

class CharacterTokenizer:
    def __call__(self,texts,add_special_tokens=False):return {'input_ids':[[ord(c) for c in text] for text in texts]}

class SelectionConstraintTest(unittest.TestCase):
    def test_all_valid_paths_preserve_subset_choice_and_first_priority(self):
        ids=['fact_a','fact_b','fact_c'];texts=list(candidate_texts(ids))
        trie=SelectionTrie(CharacterTokenizer(),ids,0)
        self.assertEqual(trie.metadata['candidate_count'],len(texts))
        for text in texts:
            value=json.loads(text)
            self.assertIn(value['answer_mode'],('facts','view','mixed'))
            self.assertIn(value['detail'],('brief','normal','expanded'))
            self.assertEqual(len(value['claim_ids']),len(set(value['claim_ids'])))
            self.assertLessEqual(len(value['claim_ids']),4)
            path=[ord(c) for c in text]+[0]
            self.assertTrue(trie.accepts(path));prefix=[]
            for token in path:
                self.assertIn(token,trie.allowed(prefix));prefix.append(token)
            self.assertFalse(trie.accepts(path[:-1]))
        for first in ids:self.assertTrue(any(json.loads(t)['claim_ids']==[first]+[c for c in ids if c!=first] for t in texts))
        self.assertTrue(any(json.loads(t)['claim_ids']==[] for t in texts))

    def test_unknown_labels_ids_duplicates_early_eos_and_trailing_prose_are_impossible(self):
        trie=SelectionTrie(CharacterTokenizer(),['fact_a','fact_b'],0)
        good={'answer_mode':'facts','detail':'brief','claim_ids':['fact_a']}
        for mutation in ('mode','detail','unknown_id','duplicate','extra_field'):
            value=copy.deepcopy(good)
            if mutation=='mode':value['answer_mode']='premise_correction'
            elif mutation=='detail':value['detail']='unlimited'
            elif mutation=='unknown_id':value['claim_ids']=['invented']
            elif mutation=='duplicate':value['claim_ids']=['fact_a','fact_a']
            else:value['answerable']=True
            with self.subTest(mutation=mutation):self.assertFalse(trie.accepts([ord(c) for c in json.dumps(value,separators=(',',':'))]+[0]))
        self.assertNotIn(0,trie.allowed([]))
        valid=json.dumps(good,separators=(',',':'))
        self.assertFalse(trie.accepts([ord(c) for c in valid+' extra']+[0]))
        with self.assertRaises(ValueError):trie.allowed([0])
        callback=trie.callback(3)
        self.assertEqual(callback(0,[101,102,103]),trie.allowed([]))
        with self.assertRaises(ValueError):callback(1,[101,102,103])
        for bad in (['x','x'],[f'id_{i}' for i in range(13)],[None]):
            with self.assertRaises(ValueError):list(candidate_texts(bad))
        five=[f'fact_{i}' for i in range(5)]
        bounded=SelectionTrie(CharacterTokenizer(),five,0)
        too_many=json.dumps({**good,'claim_ids':five},separators=(',',':'))
        self.assertFalse(bounded.accepts([ord(c) for c in too_many]+[0]))

    def test_only_trusted_selection_stage_gets_a_constraint(self):
        messages=[{'role':'system','content':'Select relevant supplied claim IDs.'},
            {'role':'user','content':json.dumps({'evidence_propositions':[{'id':'known'}]})}]
        self.assertEqual(selection_ids(messages),['known'])
        messages[0]['content']='Audit every segment'
        self.assertIsNone(selection_ids(messages))
        messages[1]['content']='Select relevant supplied claim IDs.'
        self.assertIsNone(selection_ids(messages))
