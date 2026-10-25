"""The controlled LR test must fail closed on changes to its causal controls."""
import copy
import json
from pathlib import Path
import unittest
from scripts import lr_test_protocol as guard
ROOT=Path(__file__).resolve().parents[1]
class LRControlTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.p=json.loads((ROOT/'training/lr_1e5/protocol.json').read_bytes())
  cls.pilot=json.loads((ROOT/'training/runs/kaggle_361474126/run_manifest.json').read_bytes())
  cls.train=(ROOT/'datasets/revision_v2/train_slice.jsonl').read_bytes()
  cls.ev=(ROOT/'training/lr_1e5/model_inputs.jsonl').read_bytes()
  cls.lock=json.loads((ROOT/'datasets/revision_v2/training_lock.json').read_bytes())
 def test_valid_control(self):
  a,b=guard.validate_payloads(self.train,self.ev,self.lock,self.p,self.pilot)
  self.assertEqual((len(a),len(b)),(50,48))
  self.assertEqual(self.p['control_identity']['settings'],{**self.p['settings'],'learning_rate':1e-4})
 def test_unauthorized_settings_rejected(self):
  for key,value in [('learning_rate',1e-4),('steps',51),('seed',43),('rank',16),('weight_decay',.1),('fresh_adapter',False)]:
   with self.subTest(key=key),self.assertRaises(ValueError):
    s={**self.p['settings'],key:value};guard.validate_settings(s,self.pilot['identity']['settings'])
 def test_altered_control_identity_rejected(self):
  for key,value in [('dataset_sha256','changed'),('order_sha256','changed'),('settings',{**self.p['control_identity']['settings'],'seed':43})]:
   p=copy.deepcopy(self.p);p['control_identity'][key]=value;p['control_identity_sha256']=guard.fingerprint(p['control_identity'])
   with self.subTest(key=key),self.assertRaises(ValueError):guard.validate_payloads(self.train,self.ev,self.lock,p,self.pilot)
 def test_fresh_and_regression_cannot_be_relabelled(self):
  rows=[json.loads(s) for s in self.ev.splitlines()];rows[24]['split']='fresh_diagnostic_holdout_v3'
  ev=''.join(json.dumps(r)+'\n' for r in rows).encode();p=copy.deepcopy(self.p);p['evaluation']['sha256']=guard.digest(ev)
  with self.assertRaises(ValueError):guard.validate_payloads(self.train,ev,self.lock,p,self.pilot)
 def test_no_references_in_model_inputs(self):
  rows=[json.loads(s) for s in self.ev.splitlines()];rows[0]['reference_checks']=['leaked answer']
  ev=''.join(json.dumps(r)+'\n' for r in rows).encode();p=copy.deepcopy(self.p);p['evaluation']['sha256']=guard.digest(ev)
  with self.assertRaises(ValueError):guard.validate_payloads(self.train,ev,self.lock,p,self.pilot)
 def test_training_order_matches_control(self):
  exposure=json.loads((ROOT/'training/data_revision_v2/token_mask_preflight.json').read_bytes())
  ordered=exposure['ordered_examples'];by_id={r['example_id']:r for r in ordered}
  rows=[json.loads(s) for s in self.train.splitlines()]
  encoded=[dict(example_id=r['example_id'],token_count=by_id[r['example_id']]['input_tokens'],dropped=False) for r in rows]
  ids=[rows[i]['example_id'] for i in guard.build_order(encoded)]
  self.assertEqual(ids,self.p['training_order']['example_ids'])
  self.assertEqual(sum(r['supervised_tokens'] for r in ordered),4407)
if __name__=='__main__':unittest.main()
