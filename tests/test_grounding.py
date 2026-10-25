import copy, json, sys, unittest, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_grounding_cases import verify, OUT, sha
from kaggle_grounding import validate_payloads, read_evaluation_journal, CONDITIONS
from fresh_holdout_guard import guard_training_project

class GroundingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=verify();cls.b=(OUT/'model_inputs.jsonl').read_bytes()
        cls.c=json.loads((ROOT/'evaluation/runs/kaggle_data_revision_v2_v1/results/run_manifest.json').read_bytes())
    def test_frozen_and_training_exclusion(self):
        self.assertEqual(len(validate_payloads(self.b,self.p,self.c)),48)
        self.assertIn('evaluation/fresh_holdout_grounding_v1/source_exclusions.jsonl',guard_training_project(ROOT)['fresh_holdout_manifests'])
    def test_prompt_only_treatment_and_no_references(self):
        rows=validate_payloads(self.b,self.p,self.c)
        self.assertEqual(sum(r['treatment']=='with_passages' for r in rows),24)
        changed=copy.deepcopy(rows);changed[1]['messages'][1]['content']+=' Another instruction'
        payload=('\n'.join(json.dumps(r) for r in changed)+'\n').encode();p=copy.deepcopy(self.p);p['files']['model_inputs.jsonl']=sha(payload)
        with self.assertRaises(ValueError):validate_payloads(payload,p,self.c)
    def test_checksum_and_adapter_identity_fail_closed(self):
        with self.assertRaises(ValueError):validate_payloads(self.b+b'\n',self.p,self.c)
        c=copy.deepcopy(self.c);c['identity']['settings']['learning_rate']=1e-5
        with self.assertRaises(ValueError):validate_payloads(self.b,self.p,c)
    def test_no_training_and_all_four_arms(self):
        self.assertEqual(len(CONDITIONS),4)
        import ast
        tree=ast.parse((ROOT/'scripts/kaggle_grounding.py').read_text())
        calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(set(calls)&{'AdamW','backward','step','get_peft_model','save_pretrained','load_checkpoint'})
    def test_critical_rule_is_explicit(self):
        self.assertIn('0 if any critical flag',self.p['rubric']['primary_score'])

class ScoreValidationTest(unittest.TestCase):
    def test_critical_zero_preserves_separate_partial_credit(self):
        from review_grounding import map_judgment
        case={'category':'chronology','control_type':'positive','source_family_id':'test_family'}
        key={'condition':'neutral_test_condition','case_id':'test_case'}
        s={'dimension_score':3,'critical_types':['future_event_as_known'],'critical_zero_score':0}
        mapped=map_judgment(s,key,case)
        self.assertEqual(mapped['critical_zero_score'],0)
        self.assertEqual(mapped['dimension_score'],3)
        s={'dimension_score':3,'critical_types':[]}
        self.assertEqual(map_judgment(s,key,case)['critical_zero_score'],3)

    def test_checkpoint_hash_preserves_its_newline_schema(self):
        from review_grounding import checkpoint_identity_fingerprint, fingerprint
        p=json.loads((OUT/'protocol.json').read_bytes())
        checkpoint=next((ROOT/'evaluation/runs/kaggle_data_revision_v2_v1/download').rglob('checkpoint-000050/checkpoint_manifest.json'))
        cp=json.loads(checkpoint.read_bytes())
        self.assertEqual(checkpoint_identity_fingerprint(p['control_identity']),cp['identity_sha256'])
        self.assertEqual(fingerprint(p['control_identity']),p['control_identity_sha256'])
        self.assertNotEqual(cp['identity_sha256'],p['control_identity_sha256'])

    def test_hashes_applicability_and_critical_flags(self):
        import review_grounding as r
        original=r.RUN
        with tempfile.TemporaryDirectory() as temp:
            try:
                r.RUN=Path(temp);review=r.RUN/'review';review.mkdir()
                cases=[json.loads(s) for s in (OUT/'cases.jsonl').read_text().splitlines()]
                (review/'cases.jsonl').write_bytes((OUT/'cases.jsonl').read_bytes())
                (review/'rubric.json').write_bytes((OUT/'rubric.json').read_bytes())
                neutral=[];scores=[]
                for i in range(96):
                    c=cases[i%24];rid=f'T{i:03d}';response_hash=sha(b'synthetic test only')
                    neutral.append(dict(review_id=rid,case_id=c['case_id'],response='synthetic test only',response_sha256=response_hash))
                    scores.append(dict(review_id=rid,dimension_score=3,critical_types=[],checklist_met=[],central_premise_corrected=False if c['category']=='false_premise' and c['control_type']=='negative' else None,fabrication_present=False,unsupported_replacement_present=False,supported_answer=True if c['control_type']=='positive' else None,blanket_refusal=False if c['control_type']=='positive' else None,notes='Synthetic validation fixture only',response_sha256=response_hash))
                from prepare_grounding_cases import jl
                (review/'neutral_answers.jsonl').write_bytes(jl(neutral))
                (review/'condition_key.jsonl').write_text('synthetic key, no condition mapping\n')
                (review/'review_manifest.json').write_text(json.dumps(dict(neutral_sha256=sha(jl(neutral)),key_sha256=sha((review/'condition_key.jsonl').read_bytes()))))
                def save():(review/'scores.jsonl').write_bytes(jl(scores))
                save();self.assertEqual(len(r.validate_scores()),96)
                scores[0]['critical_types']=['invented_private_scene'];save()
                with self.assertRaises(ValueError):r.validate_scores()
                scores[0]['fabrication_present']=True;save();self.assertEqual(len(r.validate_scores()),96)
                scores[0]['response_sha256']='bad';save()
                with self.assertRaises(ValueError):r.validate_scores()
            finally:r.RUN=original

if __name__=='__main__':unittest.main()
