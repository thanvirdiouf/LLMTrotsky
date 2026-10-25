"""CPU verification of immutable inputs, journal provenance and inference control flow."""
import ast
import base64
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

from scripts import kaggle_checkpoint_compare as runner
from scripts import build_kaggle_checkpoint_compare as builder

PROJECT = Path(__file__).resolve().parents[1]


def fixture():
    pilot = json.loads((PROJECT / 'training/runs/kaggle_361474126/run_manifest.json').read_bytes())
    pilot['identity']['chat_template_sha256'] = runner.digest(b'fake nonthinking template')
    pilot['identity_sha256'] = runner.fingerprint(pilot['identity'])
    inputs = [{'case_id': f'fresh_fixture_{i}', 'category': 'answer_detail',
               'split': 'fresh_diagnostic_holdout', 'messages': [
                   {'role': 'system', 'content': 'Historical simulation.'},
                   {'role': 'user', 'content': f'Explain case {i}.'}]} for i in range(24)]
    data = b''.join((json.dumps(r) + '\n').encode() for r in inputs)
    protocol = {'schema_version': 1, 'model': runner.MODEL, 'generation': runner.GENERATION,
                'conditions': list(runner.CONDITIONS), 'cases': 24, 'seed': 42,
                'enable_thinking': False, 'max_input_tokens': 8192,
                'pilot_kernel_source': runner.KERNEL_SOURCE, 'files': {'model_inputs.jsonl': runner.digest(data)},
                'pilot_identity': pilot['identity'], 'pilot_identity_sha256': pilot['identity_sha256'],
                'packages': pilot['identity']['packages'],
                'chat_template_sha256': pilot['identity']['chat_template_sha256'],
                'source_protocol_sha256': 'fixture-source-protocol'}
    return inputs, data, protocol, pilot


def journal_row(case, condition='fresh_base', identity='identity', input_tokens=None):
    tokens = [1, 2, 3] if input_tokens is None else input_tokens
    return {'case_id': case['case_id'], 'condition': condition,
            'identity_sha256': identity, 'prompt_sha256': runner.fingerprint(case['messages']),
            'generation_sha256': runner.fingerprint(runner.GENERATION),
            'input_token_ids': tokens, 'input_sha256': runner.fingerprint(tokens),
            'response': 'Synthetic fixture only.', 'token_ids': [41, 248046], 'token_count': 2,
            'finish_reason': 'stop'}


class PayloadAndJournalTests(unittest.TestCase):
    def test_input_projection_rejects_reference_leakage_duplicate_and_changed_settings(self):
        inputs, data, protocol, pilot = fixture()
        self.assertEqual(len(runner.validate_payloads(data, protocol, pilot)), 24)
        for mutation in ['reference', 'duplicate', 'identity', 'generation']:
            rows, p, saved = deepcopy(inputs), deepcopy(protocol), deepcopy(pilot)
            if mutation == 'reference': rows[0]['reference_evidence'] = 'Do not send this.'
            if mutation == 'duplicate': rows[1]['case_id'] = rows[0]['case_id']
            if mutation == 'identity': saved['identity']['settings']['learning_rate'] = 3
            if mutation == 'generation': p['generation']['eos_token_id'] = 248044
            payload = b''.join((json.dumps(r) + '\n').encode() for r in rows)
            p['files']['model_inputs.jsonl'] = runner.digest(payload)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.validate_payloads(payload, p, saved)

    def test_tokenizer_explicit_list_avoids_new_mapping_default_and_counts_tokens(self):
        calls=[]
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls.append(kwargs)
                return list(range(257)) if kwargs.get('return_dict') is False else {'input_ids': list(range(257))}
        inputs, _, _, _ = fixture()
        self.assertEqual(len(runner.tokenize_inputs(Tokenizer(), inputs, 300)[inputs[0]['case_id']]), 257)
        self.assertTrue(all(call['enable_thinking'] is False and call['return_dict'] is False for call in calls))
        with self.assertRaisesRegex(ValueError, 'budget'):
            runner.tokenize_inputs(Tokenizer(), inputs, 256)

    def test_resume_rejects_mixed_prompt_generation_input_eos_and_torn_rows(self):
        inputs, _, _, _ = fixture()
        token_map = {case['case_id']: [1, 2, 3] for case in inputs}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'responses.jsonl'
            good = journal_row(inputs[0])
            runner.atomic_journal(path, [good])
            self.assertEqual(runner.read_evaluation_journal(path, 'identity', 'fresh_base', inputs,
                                                           runner.GENERATION, token_map), [good])
            for key, value in [('prompt_sha256', 'wrong'), ('generation_sha256', 'wrong'),
                               ('input_token_ids', [9]), ('token_count', 9),
                               ('finish_reason', 'length'), ('condition', 'step_50')]:
                row = dict(good); row[key] = value
                runner.atomic_journal(path, [row])
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.read_evaluation_journal(path, 'identity', 'fresh_base', inputs, runner.GENERATION, token_map)
            runner.atomic_journal(path, [good, good])
            with self.assertRaises(ValueError):
                runner.read_evaluation_journal(path, 'identity', 'fresh_base', inputs, runner.GENERATION, token_map)
            path.write_bytes(b'{"case_id":')
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                runner.read_evaluation_journal(path, 'identity', 'fresh_base', inputs, runner.GENERATION, token_map)

    def test_checkpoint_selection_requires_unique_declared_steps_and_matching_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in (50, 500): (root / f'checkpoint-{step:06d}').mkdir()
            calls=[]
            def verify(path, expected_identity):
                calls.append((path.name, expected_identity))
                return {'completed_steps': int(path.name[-6:]), 'order_ids': ['a', 'b']}
            self.assertEqual(set(runner.find_checkpoints(root, {'id': 1}, verify)), {'step_50', 'step_500'})
            self.assertEqual(len(calls), 2)
            (root / 'duplicate' / 'checkpoint-000050').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'found 2'):
                runner.find_checkpoints(root, {'id': 1}, verify)

    def test_export_reports_partial_completion_without_inventing_quality_scores(self):
        inputs, _, _, _ = fixture()
        results={kind: [] for kind in runner.CONDITIONS}
        results['fresh_base']=[journal_row(inputs[0])]
        with tempfile.TemporaryDirectory() as temporary:
            summary=runner.export_results(temporary, results, inputs, 'identity')
            self.assertFalse(summary['complete'])
            self.assertFalse(summary['quality_scores_assigned'])
            import csv
            with (Path(temporary)/'paired_answers.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['fresh_base_critical_failure'], '')
            self.assertEqual(rows[0]['step_50_response'], '')


class BuilderTests(unittest.TestCase):
    def test_live_frozen_notebook_and_cli_stage_are_current(self):
        notebook=builder.make_notebook()
        expected=json.loads((PROJECT/'notebooks/kaggle_checkpoint_compare.ipynb').read_bytes())
        self.assertEqual(notebook,expected)
        staged=PROJECT/'training/kaggle_cli/checkpoint_compare'
        self.assertEqual(expected,json.loads((staged/'kaggle_checkpoint_compare.ipynb').read_bytes()))
        self.assertEqual(builder.kernel_metadata(),json.loads((staged/'kernel-metadata.json').read_bytes()))

    def test_metadata_is_private_version_pinned_and_inference_only(self):
        m=builder.kernel_metadata()
        self.assertTrue(m['is_private'])
        self.assertEqual(m['kernel_sources'], [runner.KERNEL_SOURCE])
        self.assertEqual(m['machine_shape'], 'NvidiaTeslaT4')
        self.assertTrue(m['docker_image'].endswith('9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'))
        self.assertEqual(m['dataset_sources'], [])
        self.assertNotIn('train_seed.jsonl', builder.BUNDLED_PATHS)
        self.assertNotIn('kaggle_pilot.py', builder.BUNDLED_PATHS)

    def test_runtime_projection_and_notebook_compile_with_reference_artifacts_excluded(self):
        inputs, data, p, pilot=fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name, rel in builder.BUNDLED_PATHS.items():
                path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
                if name=='model_inputs.jsonl': path.write_bytes(data)
                elif name=='pilot_run_manifest.json': path.write_text(json.dumps(pilot))
                else:path.write_bytes((PROJECT/rel).read_bytes())
            fresh=root/builder.FRESH
            (fresh/'cases.jsonl').write_text('reference evidence, never bundled\n')
            (fresh/'source_exclusions.jsonl').write_text('excluded source ids\n')
            (fresh/'specifications.json').write_text('scoring checklist\n')
            p['files'].update({name:runner.digest((fresh/name).read_bytes()) for name in ['cases.jsonl','source_exclusions.jsonl']})
            p['specifications_sha256']=runner.digest((fresh/'specifications.json').read_bytes())
            p['rubric']={'secret-reference': 'DO NOT BUNDLE'}
            (fresh/'protocol.json').write_text(json.dumps(p))
            payloads,runtime=builder.prepare_payloads(root)
            self.assertNotIn('rubric',runtime)
            self.assertEqual(set(runtime['files']), {'model_inputs.jsonl'})
            self.assertEqual(set(payloads), set(builder.BUNDLED_PATHS)|{'runtime_protocol.json'})
            notebook=builder.make_notebook(root)
            self.assertEqual(notebook,builder.make_notebook(root))
            ids=[]
            for cell in notebook['cells']:
                ids.append(cell['id'])
                if cell['cell_type']=='code' and cell['id']!='install':
                    compile(''.join(cell['source']),cell['id'],'exec')
            self.assertEqual(len(ids),len(set(ids)))
            bundle_source=next(''.join(c['source']) for c in notebook['cells'] if c['id']=='bundle')
            literal=bundle_source.split('BUNDLED_FILES = ',1)[1].split('\n',1)[0]
            packed=ast.literal_eval(literal)
            for name, item in packed.items():
                content=zlib.decompress(base64.b64decode(item['zlib_base64']))
                self.assertEqual(runner.digest(content),item['sha256'])
                self.assertEqual(content,payloads[name])
            (fresh/'cases.jsonl').write_text('changed reference')
            with self.assertRaisesRegex(ValueError,'Frozen holdout file'):
                builder.prepare_payloads(root)


class FakeRuntimeTests(unittest.TestCase):
    def test_complete_inference_path_base_precedes_adapters_and_step500_reloads(self):
        # This executes the real orchestration, journaling, tensor-check and
        # generation path. Everything resembling a GPU/model is a small fake.
        inputs,data,p,pilot=fixture()
        events=[]
        class Weight:
            dtype='fp32'
            def __init__(self,value=7):self.value=value
            def detach(self):return self
            def cpu(self):return self
            def to(self,dtype):return self
        class Vector:
            def __init__(self,value):self.value=value
            def tolist(self):return self.value
        class InputTensor:
            def __init__(self,value):self.value=value;self.shape=(1,len(value))
            def __getitem__(self,key):
                assert key==0
                return Vector(self.value)
        class OutputTensor:
            def __init__(self,value):self.value=value
            def __getitem__(self,key):
                assert key==(0,slice(3,None,None))
                return Vector(self.value)
        class Batch(dict):
            def to(self,device):
                assert device=='cuda:0';return self
        class Tokenizer:
            chat_template='fake nonthinking template';pad_token_id=248044
            def convert_tokens_to_ids(self,text):assert text=='<|im_end|>';return 248046
            def apply_chat_template(self,messages,**kw):
                assert kw['enable_thinking'] is False and kw['add_generation_prompt']
                if kw.get('return_tensors')=='pt':
                    assert kw['return_dict'] is True
                    return Batch(input_ids=InputTensor([1,2,3]))
                # Real new tokenizer versions can default to a mapping.
                return [1,2,3] if kw.get('return_dict') is False else {'input_ids':[1,2,3]}
            def decode(self,tokens,skip_special_tokens):
                return 'Synthetic fixture only.'
        class Config:
            use_cache=False
            def get_text_config(self,decoder):assert decoder;return self
        class Model:
            is_loaded_in_4bit=True
            def __init__(self):self.config=Config();self.condition='fresh_base';events.append('load_base')
            def parameters(self):return [SimpleNamespace(requires_grad=False,device='cuda:0')]
            def named_modules(self):return [('text',self)]
            def eval(self):return self
            def generate(self,**kw):
                assert kw['logits_to_keep']==1
                assert kw['generation_config'].eos_token_id==248046
                events.append('generate_'+self.condition)
                return OutputTensor([41,248046])
        class Gen:
            @classmethod
            def from_model_config(cls,config):return cls()
            def to_dict(self):return dict(self.__dict__)
        def module(name,**attrs):
            value=ModuleType(name)
            for k,v in attrs.items():setattr(value,k,v)
            return value
        def attach(model,path,is_trainable):
            assert is_trainable is False
            model.condition='step_'+str(int(path.parent.name[-6:]))
            model.peft_config={'adapter': True};events.append('attach_'+model.condition)
            return model
        gpu=pilot['identity']['gpu']
        torch=module('torch',float16='torch.float16',device=lambda kind,index:f'{kind}:{index}',
                     set_num_threads=lambda value:None,
                     equal=lambda x,y:x.value==y.value,
                     inference_mode=nullcontext,autocast=lambda *a,**kw:nullcontext(),
                     version=SimpleNamespace(cuda='fake-cuda'),
                     backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=None)),cudnn=SimpleNamespace(benchmark=None)),
                     cuda=SimpleNamespace(is_available=lambda:True,set_device=lambda index:None,
                         get_device_name=lambda index:gpu['name'],get_device_capability=lambda index:gpu['compute_capability'],
                         get_device_properties=lambda index:SimpleNamespace(total_memory=gpu['total_bytes']),
                         memory_allocated=lambda index:0,memory_reserved=lambda index:0,max_memory_allocated=lambda index:0,
                         synchronize=lambda:None,reset_peak_memory_stats=lambda index:None,empty_cache=lambda:None))
        def prepare(model,**kw):
            assert kw['training'] is False
            events.append('prepare_inference_only');return model
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);bundle=root/'bundle';bundle.mkdir();checkpoints=root/'input';checkpoints.mkdir()
            for helper in ['kbit_preparation.py','training_masks.py','pilot_checkpoints.py']:
                (bundle/helper).write_bytes((PROJECT/'scripts'/helper).read_bytes())
            for step in (50,500):
                cp=checkpoints/f'checkpoint-{step:06d}';ad=cp/'adapter';ad.mkdir(parents=True)
                (cp/'checkpoint_manifest.json').write_text('{}')
                cfg={'base_model_name_or_path':runner.MODEL['model_id'],'revision':runner.MODEL['revision'],
                     'r':pilot['identity']['settings']['rank'],'lora_alpha':pilot['identity']['settings']['lora_alpha'],
                     'lora_dropout':pilot['identity']['settings']['lora_dropout']}
                (ad/'adapter_config.json').write_text(json.dumps(cfg))
            def verify(path,expected_identity):
                assert expected_identity==pilot['identity']
                return {'completed_steps':int(path.name[-6:]),'order_ids':['one'],
                        'identity_sha256':'checkpoint','files':{'adapter/adapter_model.safetensors':'hash'}}
            snapshot=root/runner.MODEL['revision'];snapshot.mkdir()
            pilot_bytes=json.dumps(pilot).encode()
            p['pilot_run_manifest_sha256']=runner.digest(pilot_bytes)
            (bundle/'pilot_run_manifest.json').write_bytes(pilot_bytes)
            (bundle/'model_inputs.jsonl').write_bytes(data)
            (bundle/'runtime_protocol.json').write_text(json.dumps(p))
            modules={
                'torch':torch,
                'pilot_checkpoints':module('pilot_checkpoints',verify_checkpoint=verify),
                'kbit_preparation':module('kbit_preparation',prepare_frozen_quantized_base=prepare),
                'huggingface_hub':module('huggingface_hub',snapshot_download=lambda *a,**kw:str(snapshot)),
                'peft':module('peft',PeftModel=SimpleNamespace(from_pretrained=attach),prepare_model_for_kbit_training=object()),
                'peft.utils':module('peft.utils'),
                'peft.utils.save_and_load':module('peft.utils.save_and_load',get_peft_model_state_dict=lambda *a,**kw:{'weight':Weight()}),
                'safetensors':module('safetensors'),
                'safetensors.torch':module('safetensors.torch',load_file=lambda *a,**kw:{'weight':Weight()}),
                'transformers':module('transformers',AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a,**kw:Model()),
                    AutoTokenizer=SimpleNamespace(from_pretrained=lambda **kw:Tokenizer()),BitsAndBytesConfig=lambda **kw:kw,
                    GenerationConfig=Gen,set_seed=lambda seed:None,AutoConfig=SimpleNamespace(from_pretrained=lambda *a,**kw:Config()))}
            real_is_dir=Path.is_dir
            def is_dir(path):return True if str(path)=='/kaggle/working' else real_is_dir(path)
            with patch.dict(sys.modules,modules),patch.object(Path,'is_dir',is_dir),\
                 patch('importlib.metadata.version',side_effect=lambda name:p['packages'][name]),\
                 patch('subprocess.run',return_value=SimpleNamespace(stdout='fake packages\n')):
                result=runner.run_comparison(bundle,root/'output',input_root=checkpoints)
            manifest=json.loads((result/'run_manifest.json').read_bytes())
            self.assertEqual(manifest['status'],'complete')
            self.assertTrue(all(manifest['checks'].values()))
            self.assertEqual(events.count('load_base'),2)
            self.assertLess(events.index('generate_fresh_base'),events.index('attach_step_50'))
            self.assertEqual(events.count('generate_fresh_base'),24)
            self.assertEqual(events.count('generate_step_50'),24)
            self.assertEqual(events.count('generate_step_500'),24)
            step500_load=events.index('attach_step_500')-2
            self.assertEqual(events[step500_load:step500_load+2],['load_base','prepare_inference_only'])
            self.assertTrue((result/'comparison_delivery.zip').is_file())
            summary=json.loads((result/'evaluation_summary.json').read_bytes())
            self.assertTrue(summary['complete'])
            self.assertFalse(summary['quality_scores_assigned'])
            # A second execution with identical identity resumes all durable
            # rows and performs no further generation.
            before=len([e for e in events if e.startswith('generate_')])
            with patch.dict(sys.modules,modules),patch.object(Path,'is_dir',is_dir),\
                 patch('importlib.metadata.version',side_effect=lambda name:p['packages'][name]),\
                 patch('subprocess.run',return_value=SimpleNamespace(stdout='fake packages\n')):
                runner.run_comparison(bundle,root/'output',input_root=checkpoints,resume_dir=result)
            self.assertEqual(before,len([e for e in events if e.startswith('generate_')]))


if __name__=='__main__':
    unittest.main()
