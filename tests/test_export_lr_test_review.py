"""Synthetic delivery checks for the real frozen LR protocol; no model execution."""
import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from scripts import export_lr_test_review as ex
ROOT=Path(__file__).resolve().parents[1]
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v)+'\n')
def jl(p,values):p.write_text(''.join(json.dumps(v)+'\n' for v in values))
class LRDeliveryTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name);self.run=self.base/'results';self.run.mkdir();self.archive=self.base/'download/lr_test_delivery.zip';self.archive.parent.mkdir()
  p=json.loads((ROOT/'training/lr_1e5/protocol.json').read_bytes());self.p=p
  raw={'protocol.json':ROOT/'training/lr_1e5/protocol.json','model_inputs.jsonl':ROOT/'training/lr_1e5/model_inputs.jsonl','training_lock.json':ROOT/'datasets/revision_v2/training_lock.json','inputs/train_slice.jsonl':ROOT/'datasets/revision_v2/train_slice.jsonl','inputs/pilot_run_manifest.json':ROOT/'training/runs/kaggle_361474126/run_manifest.json','inputs/control_run_manifest.json':ROOT/'evaluation/runs/kaggle_data_revision_v2_v1/results/run_manifest.json'}
  raw.update({'inputs/'+n:ROOT/'scripts'/n for n in p['frozen_helpers']})
  for name,source in raw.items():dest=self.run/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
  self.inputs=[json.loads(s) for s in raw['model_inputs.jsonl'].read_text().splitlines()];self.tokens={r['case_id']:[1,2,n] for n,r in enumerate(self.inputs)}
  identity={'settings':p['settings'],'packages':p['packages'],'model_id':p['settings']['model_id'],'revision':p['settings']['revision'],'dataset_sha256':p['training']['sha256'],'pilot_lock_sha256':ex.digest(raw['training_lock.json'].read_bytes()),'protocol_sha256':ex.digest(raw['protocol.json'].read_bytes()),'evaluation_inputs_sha256':p['evaluation']['sha256'],'chat_template_sha256':p['chat_template_sha256'],'order_sha256':p['training_order']['ordered_example_ids_sha256'],'helpers':p['frozen_helpers'],'gpu':p['pilot_identity']['gpu'],'dtype':'torch.float16','gpu_ids_used':[0],'purpose':'controlled_lr_1e5_50_updates','evaluation_prompt_token_ids_sha256':ex.fingerprint(self.tokens)}
  self.identity=identity;ident=ex.fingerprint(identity)
  write(self.run/'run_manifest.json',dict(status='complete',stage='complete',completed_steps=50,identity=identity,identity_sha256=ident,resume_checkpoint=None,checks=dict.fromkeys(ex.REQUIRED_CHECKS,True)))
  self.generation={**p['evaluation']['generation'],'temperature':1.,'top_k':0,'top_p':1.}
  write(self.run/'comparison_manifest.json',dict(identity_sha256=ident,conditions=ex.CONDITIONS,generation=self.generation,generation_sha256=ex.fingerprint(self.generation),evaluation_inputs_sha256=p['evaluation']['sha256'],evaluation_protocol_sha256=p['evaluation']['source_protocol_sha256'],enable_thinking=False,fresh_base_before_any_evaluation_adapter=True,original_checkpoint_manifest_sha256=p['original_checkpoint_manifest_sha256']))
  prior=ROOT/'evaluation/runs/kaggle_data_revision_v2_v1/results'
  for name in ['training_metrics.jsonl','tokenization_report.json','training_summary.json','trained_probe.json','reloaded_probe.json','training_order.json']:(self.run/name).write_bytes((prior/name).read_bytes())
  (self.run/'final_adapter').mkdir();(self.run/'final_adapter/adapter_config.json').write_bytes((prior/'final_adapter/adapter_config.json').read_bytes());(self.run/'final_adapter/adapter_model.safetensors').write_bytes(b'Synthetic test file; no model.')
  for step,h in [(0,p['control_initial_adapter_sha256']),(50,ex._file_hash(self.run/'final_adapter/adapter_model.safetensors'))]:
   cp=self.base/f'checkpoints/checkpoint-{step:06d}';write(cp/'checkpoint_manifest.json',dict(identity=identity,completed_steps=step,order_ids=p['training_order']['example_ids'],files={'adapter/adapter_model.safetensors':h}));write(cp/'COMPLETE',dict(manifest_sha256=ex._file_hash(cp/'checkpoint_manifest.json')))
  for condition in ex.CONDITIONS:
   values=[dict(case_id=r['case_id'],category=r['category'],split=r['split'],condition=condition,identity_sha256=ident,prompt_sha256=ex.fingerprint(r['messages']),generation_sha256=ex.fingerprint(self.generation),input_token_ids=self.tokens[r['case_id']],input_sha256=ex.fingerprint(self.tokens[r['case_id']]),token_ids=[41,248046],token_count=2,finish_reason='stop',response='Synthetic response only.') for r in self.inputs];jl(self.run/f'{condition}_responses.jsonl',values)
  write(self.run/'evaluation_summary.json',dict(complete=True,expected_per_condition=48,completed_per_condition=dict.fromkeys(ex.CONDITIONS,48),truncations_by_condition=dict.fromkeys(ex.CONDITIONS,0),quality_scores_assigned=False))
  self.seal()
 def seal(self):
  with zipfile.ZipFile(self.archive,'w',zipfile.ZIP_DEFLATED) as z:
   for p in sorted(self.run.rglob('*')):
    if p.is_file() and 'review' not in p.relative_to(self.run).parts:z.write(p,p.relative_to(self.run))
 def test_full_export_has144_blank_condition_hidden_rows(self):
  target=ex.export_review(self.run,project=ROOT,archive_path=self.archive)
  rows=[json.loads(s) for s in (target/'review_rows.jsonl').read_text().splitlines()]
  self.assertEqual(len(rows),144);self.assertEqual(sum(r['split']=='fresh_diagnostic_holdout_v3' for r in rows),72)
  self.assertTrue(all('condition' not in r and r['critical_failure']=='' for r in rows))
  self.assertEqual([len((target/f'review_shard_{i:02d}.jsonl').read_text().splitlines()) for i in range(1,4)],[48]*3)
 def test_change_in_actual_exposure_is_rejected(self):
  p=self.run/'training_metrics.jsonl';rs=[json.loads(s) for s in p.read_text().splitlines()];rs[0]['supervised_tokens']+=1;jl(p,rs);self.seal()
  with self.assertRaisesRegex(ValueError,'token exposure'):ex.validate_run(self.run,project=ROOT,archive_path=self.archive)
 def test_wrong_initialization_is_rejected(self):
  p=self.base/'checkpoints/checkpoint-000000/checkpoint_manifest.json';m=json.loads(p.read_bytes());m['files']['adapter/adapter_model.safetensors']='f'*64;write(p,m);write(p.parent/'COMPLETE',{'manifest_sha256':ex._file_hash(p)})
  with self.assertRaisesRegex(ValueError,'Initial adapter'):ex.validate_run(self.run,project=ROOT,archive_path=self.archive)
 def test_archive_detects_post_extraction_edit(self):
  (self.run/'training_metrics.jsonl').write_text('{}\n')
  with self.assertRaisesRegex(ValueError,'archive mismatch'):ex.validate_run(self.run,project=ROOT,archive_path=self.archive)
if __name__=='__main__':unittest.main()
