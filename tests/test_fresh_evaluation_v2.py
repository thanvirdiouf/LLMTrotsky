"""Freeze/exclusion integrity tests on small independent source fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.freeze_fresh_evaluation_v2 import prepare,freeze,verify,CATEGORIES,DIRECTORY

ROOT=Path(__file__).resolve().parents[1]


def fixture(root):
    root=Path(root)
    def write(name,value,jsonl=False):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in value) if jsonl else json.dumps(value))
    persona=json.loads((ROOT/'config/persona.json').read_text())
    write('config/persona.json',persona)
    docs=[];catalog=[]
    for i in range(9):
        key=f'doc{i}';family=f'family{i}'
        docs.append({'document_id':key,'url':f'https://primary.example/{key}','source_sha256':str(i)*64,'composition_date_interval':{'earliest':'1931-01-01','latest':'1931-01-01','precision':'day','raw':'1January1931'},'composition_date_evidence':[{'kind':'signed_dateline','text':'1January1931'}],'blocks':[{'block_id':0,'role':'heading','text':key},{'block_id':1,'role':'body_candidate','text':' '.join(f'{key}token{n}' for n in range(100))}]})
        catalog.append({'document_id':key,'source_family_id':family,'url':docs[-1]['url']})
    write('corpus_mia/documents.jsonl',docs,True);write('corpus_mia/catalog.jsonl',catalog,True)
    write('corpus_mia/duplicate_edges.jsonl',[],True)
    messages=[{'role':'system','content':persona['system_prompt']},{'role':'user','content':'Earlier unique question?'}]
    write('datasets/seed_records.jsonl',[{'source_family_id':'family6','messages':messages}],True)
    write('datasets/train_seed.jsonl',[{'example_id':'seed','messages':messages+[{'role':'assistant','content':'Earlier unique answer.'}]}],True)
    write('evaluation/cases.jsonl',[{'source_family_id':'family7','messages':messages}],True)
    write('evaluation/fresh_holdout_v1/cases.jsonl',[{'source_family_id':'family8','messages':messages}],True)
    v1=json.loads((ROOT/'evaluation/fresh_holdout_v1/protocol.json').read_text())
    write('evaluation/fresh_holdout_v1/protocol.json',v1)
    write('training/runs/kaggle_361474126/run_manifest.json',{'identity':v1['pilot_identity']})
    write('training/pilot_lock.json',{'original':'preserved'})
    specs=[]
    for i in range(6):
        for cat in sorted(CATEGORIES):
            control='positive'
            if cat=='chronology':control='positive' if i<3 else 'conditional' if i<5 else 'negative'
            elif cat in ['false_premise','invented_memory_citation']:control='positive' if i<3 else 'negative'
            specs.append({'case_id':f'case{i}_{cat}','work':f'work{i}','source_document_id':f'doc{i}','source_block_ids':[1],'category':cat,'control_type':control,'question':f'New question{i} {cat}?','reference_checks':['Answer the factual criterion.']})
    write(DIRECTORY+'/specifications.json',{'cases':specs})
    (root/DIRECTORY/'source_review.md').write_text('Manual source review fixture only.\n')
    return docs,catalog,specs


class FreshEvaluationV2Tests(unittest.TestCase):
    def test_freeze_uses_model_projection_and_positive_controls_then_is_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            p=freeze(root)
            self.assertEqual(p['control_counts'],{'positive':15,'conditional':2,'negative':7})
            self.assertEqual(p['conditions'],['fresh_base','original_step_50','revised_step_50'])
            inputs=[json.loads(s) for s in (root/DIRECTORY/'model_inputs.jsonl').read_text().splitlines()]
            self.assertEqual(len(inputs),24)
            self.assertTrue(all(set(r)=={'case_id','category','split','messages'} for r in inputs))
            self.assertTrue(all(set(m)=={'role','content'} for r in inputs for m in r['messages']))
            self.assertEqual(verify(root),p)
            before=(root/DIRECTORY/'protocol.json').read_bytes()
            with self.assertRaisesRegex(FileExistsError,'frozen'):freeze(root)
            self.assertEqual((root/DIRECTORY/'protocol.json').read_bytes(),before)

    def test_exact_source_family_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,cat,_=fixture(root)
            cat[0]['source_family_id']='family6'
            (root/'corpus_mia/catalog.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cat))
            with self.assertRaisesRegex(ValueError,'overlap'):prepare(root)

    def test_detected_duplicate_text_in_training_family_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);docs,_,_=fixture(root)
            docs[6]['blocks'][1]['text']=docs[0]['blocks'][1]['text']
            (root/'corpus_mia/documents.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in docs))
            with self.assertRaisesRegex(ValueError,'Near-duplicate'):prepare(root)

    def test_known_edge_closure_cannot_enter_old_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            (root/'corpus_mia/duplicate_edges.jsonl').write_text(json.dumps({'left':'doc0','right':'doc7'})+'\n')
            with self.assertRaisesRegex(ValueError,'Near-duplicate'):prepare(root)

    def test_unreviewed_editorial_block_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);docs,_,_=fixture(root)
            docs[0]['blocks'][1]['role']='editorial'
            (root/'corpus_mia/documents.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in docs))
            with self.assertRaisesRegex(ValueError,'non-authored'):prepare(root)

    def test_checksum_corruption_and_preserved_training_change_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root);freeze(root)
            file=root/DIRECTORY/'model_inputs.jsonl';before=file.read_bytes();file.write_bytes(before+b'\n')
            with self.assertRaisesRegex(ValueError,'checksum'):verify(root)
            file.write_bytes(before)
            (root/'datasets/train_seed.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Preserved input'):verify(root)

if __name__=='__main__':unittest.main()
