"""Synthetic CPU checks for v2 result integrity, matched exposure and blinding."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts import export_data_revision_review as exporter
from scripts import data_revision_protocol as guards

PROJECT=Path(__file__).resolve().parents[1]

def encode(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n').encode()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encode(value))
def jsonlines(values):return ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values).encode()
def write_lines(path,values):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(jsonlines(values))

class DataRevisionReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.project=self.root/'project';self.run=self.root/'archive/results';self.run.mkdir(parents=True)
        self.archive=self.root/'archive/download/data_revision_delivery.zip';self.archive.parent.mkdir()
        self.data=self.project/'datasets/revision_v2';self.data.mkdir(parents=True)
        self.training=self.project/'training/data_revision_v2';self.training.mkdir(parents=True)
        self.fresh=self.project/'evaluation/fresh_holdout_v2';self.fresh.mkdir(parents=True)
        pilot_bytes=(PROJECT/'training/runs/kaggle_361474126/run_manifest.json').read_bytes();self.pilot=json.loads(pilot_bytes)
        pilot_path=self.project/'training/runs/kaggle_361474126/run_manifest.json';pilot_path.parent.mkdir(parents=True);pilot_path.write_bytes(pilot_bytes)
        self.training_rows=[{'example_id':f't{i:02d}','messages':[{'role':'system','content':'Fixture persona.'},{'role':'user','content':f'Fixture question {i}.'},{'role':'assistant','content':f'Fixture target {i}.'}]} for i in range(50)]
        train_bytes=jsonlines(self.training_rows);(self.data/'train_slice.jsonl').write_bytes(train_bytes)
        lock={'schema_version':1,'dataset':{'records':50,'sha256':exporter.digest(train_bytes),'example_sha256':{r['example_id']:guards.fingerprint(r) for r in self.training_rows}},'persona':{'system_prompt_sha256':exporter.digest(b'Fixture persona.')},'contamination_guard':{'span_words':16,'text_sha256':[],'span_sha256':[]}}
        lock_bytes=encode(lock);(self.data/'training_lock.json').write_bytes(lock_bytes)
        version={'files':{name:exporter.digest((self.data/name).read_bytes()) for name in ['train_slice.jsonl','training_lock.json']}}
        write(self.data/'version_manifest.json',version)
        self.order=[r['example_id'] for r in self.training_rows]
        self.exposure={'slice_supervised_tokens':500,'ordered_examples':[{'example_id':key,'update':i,'input_tokens':30,'supervised_tokens':10} for i,key in enumerate(self.order,1)]}
        write(self.training/'token_mask_preflight.json',self.exposure)
        categories=['chronology','false_premise','invented_memory_citation','answer_detail']
        self.cases=[]
        for i in range(24):
            evidence=f'Fixture primary evidence {i}.'
            self.cases.append({'case_id':f'fixture_{i}','category':categories[i//6],'control_type':['positive','negative','conditional'][i%3],
             'split':'fresh_diagnostic_holdout_v2','messages':[{'role':'system','content':'Fixture persona.'},{'role':'user','content':f'Fixture eval question {i}.'}],
             'source_url':'https://example.org/primary','source_block_ids':[1],'reference_checks':['Fixture check.'],'reference_evidence':evidence,
             'evidence_sha256':exporter.digest(evidence.encode()),'composition_date_interval':{'earliest':'1931-01-01','latest':'1931-01-01'}})
        self.inputs=[{k:r[k] for k in ['case_id','category','split','messages']} for r in self.cases]
        write_lines(self.fresh/'cases.jsonl',self.cases);write_lines(self.fresh/'model_inputs.jsonl',self.inputs)
        write(self.fresh/'specifications.json',{'cases':'Fixture checks'});(self.fresh/'source_review.md').write_text('Fixture review only.')
        self.fresh_protocol={'files':{n:exporter.digest((self.fresh/n).read_bytes()) for n in ['cases.jsonl','model_inputs.jsonl']},
         'specifications_sha256':exporter.digest((self.fresh/'specifications.json').read_bytes()),'source_review_sha256':exporter.digest((self.fresh/'source_review.md').read_bytes())}
        write(self.fresh/'protocol.json',self.fresh_protocol)
        old_cp=self.project/'training/kaggle_cli/pilot_checkpoints/run/checkpoints/checkpoint-000050'
        old={'identity':self.pilot['identity'],'completed_steps':50,'files':{'adapter/adapter_model.safetensors':'a'*64}}
        old_hash=self.write_checkpoint(old_cp,old)
        settings=dict(self.pilot['identity']['settings']);settings['steps']=50
        helper=b'# Fixture archived helper only\n';self.helpers={'fixture_helper.py':exporter.digest(helper)}
        self.protocol={'settings':settings,'packages':self.pilot['identity']['packages'],'conditions':guards.CONDITIONS,
         'pilot_identity':self.pilot['identity'],'pilot_identity_sha256':self.pilot['identity_sha256'],'pilot_run_manifest_sha256':exporter.digest(pilot_bytes),
         'chat_template_sha256':self.pilot['identity']['chat_template_sha256'],
         'training':{'sha256':exporter.digest(train_bytes),'records':50,'parent_records':526,'version_manifest_sha256':exporter.digest((self.data/'version_manifest.json').read_bytes())},
         'training_lock_fingerprint':guards.fingerprint(lock),'training_order':{'example_ids':self.order,'ordered_example_ids_sha256':guards.fingerprint(self.order)},
         'evaluation':{'sha256':exporter.digest((self.fresh/'model_inputs.jsonl').read_bytes()),'source_protocol_sha256':exporter.digest((self.fresh/'protocol.json').read_bytes()),'generation':guards.GENERATION},
         'exposure_report_sha256':exporter.digest((self.training/'token_mask_preflight.json').read_bytes()),'original_checkpoint_manifest_sha256':old_hash,'frozen_helpers':self.helpers}
        protocol_bytes=encode(self.protocol);(self.training/'protocol.json').write_bytes(protocol_bytes)
        for name,data in {'protocol.json':protocol_bytes,'training_lock.json':lock_bytes,'model_inputs.jsonl':(self.fresh/'model_inputs.jsonl').read_bytes(),
          'inputs/train_slice.jsonl':train_bytes,'inputs/pilot_run_manifest.json':pilot_bytes,'inputs/fixture_helper.py':helper}.items():
            path=self.run/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        self.tokens={r['case_id']:[1,2,i] for i,r in enumerate(self.inputs)}
        self.identity={'settings':settings,'packages':self.protocol['packages'],'model_id':settings['model_id'],'revision':settings['revision'],
         'dataset_sha256':exporter.digest(train_bytes),'pilot_lock_sha256':exporter.digest(lock_bytes),'protocol_sha256':exporter.digest(protocol_bytes),
         'evaluation_inputs_sha256':exporter.digest((self.fresh/'model_inputs.jsonl').read_bytes()),'chat_template_sha256':self.protocol['chat_template_sha256'],
         'order_sha256':guards.fingerprint(self.order),'helpers':self.helpers,'gpu':self.pilot['identity']['gpu'],'dtype':'torch.float16','gpu_ids_used':[0],
         'evaluation_prompt_token_ids_sha256':guards.fingerprint(self.tokens),'purpose':'data_revision_v2_50_updates'}
        self.identity_sha=guards.fingerprint(self.identity)
        self.manifest={'status':'complete','stage':'complete','completed_steps':50,'identity':self.identity,'identity_sha256':self.identity_sha,
         'resume_checkpoint':None,'checks':dict.fromkeys(exporter.REQUIRED_CHECKS,True)}
        self.generation={**guards.GENERATION,'temperature':1.0,'top_k':0,'top_p':1.0}
        self.comparison={'identity_sha256':self.identity_sha,'conditions':guards.CONDITIONS,'generation':self.generation,'generation_sha256':guards.fingerprint(self.generation),
          'enable_thinking':False,'evaluation_inputs_sha256':self.protocol['evaluation']['sha256'],'evaluation_protocol_sha256':self.protocol['evaluation']['source_protocol_sha256'],
          'fresh_base_before_any_evaluation_adapter':True,'original_checkpoint_manifest_sha256':old_hash}
        self.metrics=[{'step':i,'verified_optimizer_step':i,'example_id':key,'input_tokens':30,'supervised_tokens':10,'loss':3.0,
          'gradient_norm_before_clip':1.0,'grad_scaler_scale':1.0,'finite_gradients':True,'frozen_gradients_absent':True} for i,key in enumerate(self.order,1)]
        self.token_report={'total':50,'retained':50,'dropped':0,'supervised_tokens':500,'examples':[{'example_id':key,'token_count':30,'supervised_tokens':10} for key in self.order]}
        self.responses={c:[{'case_id':r['case_id'],'category':r['category'],'split':r['split'],'condition':c,'identity_sha256':self.identity_sha,
          'prompt_sha256':guards.fingerprint(r['messages']),'generation_sha256':guards.fingerprint(self.generation),
          'input_token_ids':self.tokens[r['case_id']],'input_sha256':guards.fingerprint(self.tokens[r['case_id']]),
          'token_ids':[41,248046],'token_count':2,'finish_reason':'stop','response':f'Fixture answer {ci}.'} for r in self.inputs] for ci,c in enumerate(guards.CONDITIONS)}
        write(self.run/'training_order.json',{'example_ids':self.order,'sha256':guards.fingerprint(self.order)})
        write(self.run/'training_summary.json',{'completed_optimizer_steps':50,'unique_training_examples':50,'session_changed_adapter_tensors':['fixture_lora'], 'quality_improvement_established':False})
        write(self.run/'trained_probe.json',{'token_ids':[41,248046],'input_sha256':'f'*64});write(self.run/'reloaded_probe.json',{'token_ids':[41,248046],'input_sha256':'f'*64})
        write(self.run/'final_adapter/adapter_config.json',{'base_model_name_or_path':settings['model_id'],'revision':settings['revision'],'r':8,'lora_alpha':16,'lora_dropout':0.0})
        (self.run/'final_adapter/adapter_model.safetensors').write_bytes(b'Fixture adapter bytes, not a model.')
        new_cp=self.run.parent/'checkpoints/checkpoint-000050'
        self.write_checkpoint(new_cp,{'identity':self.identity,'completed_steps':50,'order_ids':self.order,'files':{'adapter/adapter_model.safetensors':exporter._file_hash(self.run/'final_adapter/adapter_model.safetensors')}})
        for p,identity in [(self.project/'training/kaggle_cli/pilot_checkpoints/run/checkpoints/checkpoint-000000',self.pilot['identity']),
                           (self.run.parent/'checkpoints/checkpoint-000000',self.identity)]:
            self.write_checkpoint(p,{'identity':identity,'completed_steps':0,'files':{'adapter/adapter_model.safetensors':'c'*64}})
        self.write_run();self.seal_archive()

    def write_checkpoint(self,path,value):
        path.mkdir(parents=True,exist_ok=True);write(path/'checkpoint_manifest.json',value)
        h=exporter.digest((path/'checkpoint_manifest.json').read_bytes());write(path/'COMPLETE',{'manifest_sha256':h});return h

    def write_run(self):
        write(self.run/'run_manifest.json',self.manifest);write(self.run/'comparison_manifest.json',self.comparison)
        write_lines(self.run/'training_metrics.jsonl',self.metrics);write(self.run/'tokenization_report.json',self.token_report)
        for c,rs in self.responses.items():write_lines(self.run/f'{c}_responses.jsonl',rs)
        write(self.run/'evaluation_summary.json',{'complete':True,'expected_per_condition':24,'completed_per_condition':dict.fromkeys(guards.CONDITIONS,24),
          'truncations_by_condition':{c:sum(r['finish_reason']=='length' for r in rs) for c,rs in self.responses.items()},'quality_scores_assigned':False})

    def seal_archive(self):
        with zipfile.ZipFile(self.archive,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.run.rglob('*')):
                if path.is_file() and 'review' not in path.relative_to(self.run).parts:archive.write(path,path.relative_to(self.run))

    def validate(self):return exporter.validate_run(self.run,project=self.project)

    def test_full_72_export_is_blinded_reproducible_and_preserves_controls(self):
        cases,results,_,_,v=self.validate();self.assertEqual(len(cases),24);self.assertEqual(sum(map(len,results.values())),72)
        self.assertTrue(v['checkpoint_diagnostics']['initial_adapter_file_hash_equal']);self.assertTrue(v['checkpoint_diagnostics']['revised_step50_manifest_verified'])
        a=exporter.export_review(self.run,project=self.project,output_dir=self.root/'review_a');b=exporter.export_review(self.run,project=self.project,output_dir=self.root/'review_b')
        for name in ['review_rows.jsonl','review_key.jsonl','review_rows.csv','review_manifest.json']:self.assertEqual((a/name).read_bytes(),(b/name).read_bytes())
        rows=[json.loads(s) for s in (a/'review_rows.jsonl').read_text().splitlines()];keys=[json.loads(s) for s in (a/'review_key.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),72);self.assertEqual(len({r['review_id'] for r in rows}),72)
        self.assertTrue(all('condition' not in r and 'identity_sha256' not in r and r['control_type'] in ['positive','negative','conditional'] for r in rows))
        self.assertTrue(all(r[k]=='' for r in rows for k in exporter.SCORE_FIELDS))
        self.assertEqual({k['condition'] for k in keys},set(guards.CONDITIONS))
        for i in range(1,4):
            rs=[json.loads(s) for s in (a/f'review_shard_{i:02d}.jsonl').read_text().splitlines()]
            self.assertEqual(len(rs),24);self.assertEqual(len({r['case_id'] for r in rs}),8)
        with self.assertRaises(FileExistsError):exporter.export_review(self.run,project=self.project,output_dir=a)

    def test_extracted_response_or_helper_edit_is_detected_against_archive(self):
        path=self.run/'revised_step_50_responses.jsonl';original=path.read_bytes();rs=deepcopy(self.responses['revised_step_50']);rs[0]['response']='Tampered answer';write_lines(path,rs)
        with self.assertRaisesRegex(ValueError,'archive mismatch'):self.validate()
        path.write_bytes(original);(self.run/'inputs/fixture_helper.py').write_text('Changed helper')
        with self.assertRaisesRegex(ValueError,'archive mismatch'):self.validate()

    def test_resealed_manifest_settings_status_checks_and_freshness_are_rejected(self):
        good=deepcopy(self.manifest)
        changes=[lambda m:m.update(completed_steps=49),lambda m:m.update(status='running'),lambda m:m.update(resume_checkpoint='/checkpoint'),lambda m:m['checks'].update(evaluation_base_untouched=False),lambda m:m['identity']['settings'].update(learning_rate=0.5)]
        for change in changes:
            self.manifest=deepcopy(good);change(self.manifest);self.manifest['identity_sha256']=guards.fingerprint(self.manifest['identity']);self.write_run();self.seal_archive()
            with self.subTest(change=change),self.assertRaises(ValueError):self.validate()
        self.manifest=good

    def test_49_updates_wrong_counters_tokens_and_nonfinite_metrics_are_rejected(self):
        good=deepcopy(self.metrics)
        for mutation in ['missing','counter','tokens','nonfinite']:
            self.metrics=deepcopy(good)
            if mutation=='missing':self.metrics.pop()
            elif mutation=='counter':self.metrics[-1]['verified_optimizer_step']=49
            elif mutation=='tokens':self.metrics[-1]['supervised_tokens']=11
            else:self.metrics[-1]['loss']=float('nan')
            self.write_run();self.seal_archive()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.validate()
        self.metrics=good

    def test_prompt_generation_input_eos_and_coverage_tampering_rejected(self):
        good=deepcopy(self.responses)
        for update in [{'prompt_sha256':'x'},{'generation_sha256':'x'},{'input_token_ids':[9]}, {'token_ids':[248046,248046]},{'finish_reason':'length'},{'category':'wrong'}]:
            self.responses=deepcopy(good);self.responses['original_step_50'][0].update(update);self.write_run();self.seal_archive()
            with self.subTest(update=update),self.assertRaises(ValueError):self.validate()
        self.responses=deepcopy(good);self.responses['revised_step_50'].pop();self.write_run();self.seal_archive()
        with self.assertRaisesRegex(ValueError,'Incomplete'):self.validate()

    def test_frozen_reference_and_token_exposure_changes_are_rejected(self):
        original=(self.fresh/'cases.jsonl').read_bytes();(self.fresh/'cases.jsonl').write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError,'Frozen evaluation file'):self.validate()
        (self.fresh/'cases.jsonl').write_bytes(original);write(self.training/'token_mask_preflight.json',{'slice_supervised_tokens':999})
        with self.assertRaisesRegex(ValueError,'exposure'):self.validate()

    def test_final_adapter_changed_in_resealed_archive_is_caught_by_saved_checkpoint(self):
        (self.run/'final_adapter/adapter_model.safetensors').write_bytes(b'Changed adapter');self.seal_archive()
        with self.assertRaisesRegex(ValueError,'saved revised'):self.validate()

    def test_archive_escape_and_duplicate_entries_rejected(self):
        with zipfile.ZipFile(self.archive,'a') as archive:archive.writestr('../escape','not extracted')
        with self.assertRaisesRegex(ValueError,'Unsafe'):self.validate()
        self.seal_archive()
        with zipfile.ZipFile(self.archive,'a') as archive:archive.writestr('run_manifest.json','duplicate')
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.validate()

if __name__=='__main__':unittest.main()
