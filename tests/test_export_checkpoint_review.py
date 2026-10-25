"""Fresh comparison review exports verify identity and never assign quality scores."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import export_checkpoint_review as exporter
from scripts import build_kaggle_checkpoint_compare as builder
from scripts import kaggle_checkpoint_compare as runner

PROJECT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + '\n')


def write_jsonl(path, values):
    path.write_text(''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values))


class CheckpointReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.project=self.root/'project';self.run=self.root/'run'
        self.run.mkdir();fresh=self.project/'evaluation/fresh_holdout_v1';fresh.mkdir(parents=True)
        for name in ['protocol.json','cases.jsonl','model_inputs.jsonl','source_exclusions.jsonl',
                     'overlap_review.json','specifications.json']:
            (fresh/name).write_bytes((PROJECT/'evaluation/fresh_holdout_v1'/name).read_bytes())
        pilot_path=self.project/'training/runs/kaggle_361474126/run_manifest.json'
        pilot_path.parent.mkdir(parents=True)
        pilot_path.write_bytes((PROJECT/'training/runs/kaggle_361474126/run_manifest.json').read_bytes())
        payloads,self.runtime=builder.prepare_payloads(PROJECT)
        self.runtime_bytes=payloads['runtime_protocol.json']
        (self.run/'runtime_protocol.json').write_bytes(self.runtime_bytes)
        input_bytes=payloads['model_inputs.jsonl'];(self.run/'model_inputs.jsonl').write_bytes(input_bytes)
        self.inputs=[json.loads(line) for line in input_bytes.splitlines()]
        self.protocol=json.loads((fresh/'protocol.json').read_bytes())
        self.tokens={case['case_id']:[1,2,3,index] for index,case in enumerate(self.inputs)}
        self.generation={**runner.GENERATION,'temperature':1.0,'top_k':0,'top_p':1.0}
        identity={'conditions':list(runner.CONDITIONS),'model':self.protocol['model'],
            'packages':self.protocol['packages'],'chat_template_sha256':self.protocol['chat_template_sha256'],
            'pilot_identity_sha256':self.protocol['pilot_identity_sha256'],
            'source_protocol_sha256':runner.digest((fresh/'protocol.json').read_bytes()),
            'runtime_protocol_sha256':runner.digest(self.runtime_bytes),
            'model_inputs_sha256':runner.digest(input_bytes),'pilot_run_manifest_sha256':runner.digest(pilot_path.read_bytes()),
            'seed':42,'enable_thinking':False,'dtype':'torch.float16','gpu_ids_used':[0],
            'kbit_preparation_training':False,'attention_implementation':'sdpa',
            'gpu':self.protocol['pilot_identity']['gpu'],
            'quantization':{'load_in_4bit':True,'bnb_4bit_quant_type':'nf4',
                            'bnb_4bit_use_double_quant':True,'compute_dtype':'torch.float16'},
            'generation':self.generation,'generation_sha256':runner.fingerprint(self.generation),
            'prompt_token_ids_sha256':runner.fingerprint(self.tokens),
            'checkpoints':{kind:{'step':step,'manifest_sha256':'a'*64,'adapter_sha256':'b'*64,
                'identity_sha256':runner.digest((json.dumps(self.protocol['pilot_identity'], sort_keys=True,
                    separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode())} for kind,step in [('step_50',50),('step_500',500)]}}
        identity_sha=runner.fingerprint(identity)
        self.manifest={'status':'complete','stage':'complete','identity':identity,'identity_sha256':identity_sha,
            'checks':dict.fromkeys(exporter.REQUIRED_CHECKS,True),'quality_scores_assigned':False}
        self.comparison={'identity_sha256':identity_sha,'conditions':list(runner.CONDITIONS),
            'generation':self.generation,'generation_sha256':runner.fingerprint(self.generation),
            'model_inputs_sha256':runner.digest(input_bytes),
            'source_protocol_sha256':identity['source_protocol_sha256'],'enable_thinking':False,
            'fresh_base_before_adapter_attachment':True,'step_500_reloads_base':True,'quality_scores_assigned':False}
        self.responses={}
        for number,condition in enumerate(runner.CONDITIONS):
            self.responses[condition]=[{'case_id':case['case_id'],'category':case['category'],'split':case['split'],
                'condition':condition,'identity_sha256':identity_sha,
                'prompt_sha256':runner.fingerprint(case['messages']),
                'generation_sha256':runner.fingerprint(self.generation),
                'input_token_ids':self.tokens[case['case_id']],
                'input_sha256':runner.fingerprint(self.tokens[case['case_id']]),
                'response':f'Synthetic fixture answer {number}.',
                'token_ids':[40+number,248046],'token_count':2,'finish_reason':'stop'} for case in self.inputs]
        self.write_run()

    def write_run(self):
        write_json(self.run/'run_manifest.json',self.manifest)
        write_json(self.run/'comparison_manifest.json',self.comparison)
        for condition,values in self.responses.items():write_jsonl(self.run/f'{condition}_responses.jsonl',values)

    def test_full_export_is_blinded_reproducible_and_unscored(self):
        first=exporter.export_review(self.run,project=self.project,output_dir=self.run/'first',shards=3)
        second=exporter.export_review(self.run,project=self.project,output_dir=self.run/'second',shards=3)
        for name in ['review_rows.jsonl','review_key.jsonl','review_rows.csv','review_manifest.json']:
            self.assertEqual((first/name).read_bytes(),(second/name).read_bytes())
        rows=[json.loads(line) for line in (first/'review_rows.jsonl').read_text().splitlines()]
        keys=[json.loads(line) for line in (first/'review_key.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),72);self.assertEqual(len({r['review_id'] for r in rows}),72)
        self.assertTrue(all('condition' not in r and 'identity_sha256' not in r for r in rows))
        self.assertTrue(all(r[name]=='' for r in rows for name in exporter.SCORE_FIELDS))
        keyed={r['review_id']:r for r in keys}
        for row in rows:
            key=keyed[row['review_id']]
            original=next(r for r in self.responses[key['condition']] if r['case_id']==row['case_id'])
            self.assertEqual(row['response'],original['response'])
            self.assertEqual(key['response_sha256'],runner.digest(row['response'].encode()))
        shard_cases=[]
        for number in (1,2,3):
            shard=[json.loads(l) for l in (first/f'review_shard_{number:02d}.jsonl').read_text().splitlines()]
            self.assertEqual(len(shard),24)
            counts={row['case_id']:0 for row in shard}
            for row in shard:counts[row['case_id']]+=1
            self.assertEqual(set(counts.values()),{3});self.assertEqual(len(counts),8)
            shard_cases.append(set(counts))
        self.assertFalse(shard_cases[0]&shard_cases[1]);self.assertFalse(shard_cases[1]&shard_cases[2])
        review_manifest=json.loads((first/'review_manifest.json').read_text())
        self.assertFalse(review_manifest['quality_scores_assigned'])
        for name,expected in review_manifest['output_sha256'].items():
            self.assertEqual(runner.digest((first/name).read_bytes()),expected)
        with self.assertRaises(FileExistsError):exporter.export_review(self.run,project=self.project,output_dir=first)

    def test_tampered_run_identity_and_comparison_are_rejected(self):
        original=deepcopy(self.manifest)
        for change in [lambda m:m['identity'].update(seed=3),lambda m:m.update(status='running'),
                       lambda m:m['checks'].update(step_50_complete=False),
                       lambda m:m.update(identity_sha256='a'*64)]:
            self.manifest=deepcopy(original);change(self.manifest);self.write_run()
            with self.assertRaises(ValueError):exporter.validate_run(self.run,project=self.project)
        self.manifest=original;self.write_run()
        for name,value in [('source_protocol_sha256','x'),('generation_sha256','y'),
                           ('step_500_reloads_base',False),('model_inputs_sha256','z')]:
            old=deepcopy(self.comparison);self.comparison[name]=value;self.write_run()
            with self.assertRaises(ValueError):exporter.validate_run(self.run,project=self.project)
            self.comparison=old

    def test_prompt_generation_token_stop_and_coverage_tampering_are_rejected(self):
        good=deepcopy(self.responses)
        for mutation in [{'prompt_sha256':'a'*64},{'generation_sha256':'b'*64},
                         {'identity_sha256':'c'*64},{'input_token_ids':[9]},
                         {'finish_reason':'length'},{'token_ids':[248046,248046]},
                         {'token_ids':[100,101]},{'category':'changed'}]:
            self.responses=deepcopy(good);self.responses['step_50'][0].update(mutation);self.write_run()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                exporter.validate_run(self.run,project=self.project)
        self.responses=deepcopy(good);self.responses['step_500'].pop();self.write_run()
        with self.assertRaisesRegex(ValueError,'Incomplete'):exporter.validate_run(self.run,project=self.project)
        self.responses=deepcopy(good);self.responses['step_50'][0]['token_ids']=[100]*512
        self.responses['step_50'][0].update(token_count=512,finish_reason='length');self.write_run()
        cases,results,_,_=exporter.validate_run(self.run,project=self.project)
        self.assertEqual(len(cases),24);self.assertEqual(results['step_50'][0]['finish_reason'],'length')

    def test_frozen_reference_and_runtime_projection_changes_are_rejected(self):
        fresh=self.project/'evaluation/fresh_holdout_v1'
        (fresh/'cases.jsonl').write_bytes((fresh/'cases.jsonl').read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'Frozen fresh evaluation'):
            exporter.validate_run(self.run,project=self.project)
        (fresh/'cases.jsonl').write_bytes((PROJECT/'evaluation/fresh_holdout_v1/cases.jsonl').read_bytes())
        runtime=deepcopy(self.runtime);runtime['source_protocol_sha256']='altered'
        write_json(self.run/'runtime_protocol.json',runtime)
        with self.assertRaisesRegex(ValueError,'Runtime protocol'):
            exporter.validate_run(self.run,project=self.project)


if __name__=='__main__':unittest.main()
