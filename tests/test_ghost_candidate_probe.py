import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from candidate_base_probe import CANDIDATE,GIB,placement,resource_gate,run,validate_protocol,device_map_for
from build_ghost_candidate_probe import build,ASSETS,enc,sha
from verify_ghost_candidate_probe import payloads,verify
from grounded_chat import MODEL

class TokenizerFixture:
    chat_template='fixture-template'
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['enable_thinking'] is False
        count=json.loads(messages[-1]['content'])['memory_probe_only'].count('padding ')
        return [1]*25+[2]*count
    def decode(self,tokens,**kwargs):return 'fixture-probe-output'

class CandidateProbeTest(unittest.TestCase):
    def fixture(self,root):
        script=root/'scripts/candidate_base_probe.py';script.parent.mkdir(parents=True);script.write_bytes((ROOT/'scripts/candidate_base_probe.py').read_bytes())
        data=root/'evaluation/grounded_chat_v2';data.mkdir(parents=True)
        (data/'protocol.json').write_text(json.dumps({'packages':{'torch':'fixture'}}));(data/'preserved_v1_and_adapters.json').write_text('{}')
        meta=root/'training/kaggle_cli/grounded_chat_v1/kernel-metadata.json';meta.parent.mkdir(parents=True);meta.write_text('{}')
        assets=root/'evaluation/ghost_candidate_probe_v1/tokenizer_assets';assets.mkdir(parents=True)
        for name in ASSETS:(assets/name).write_text('fixture')
        build(root=root,tokenizer=TokenizerFixture());frozen=payloads('ghost_candidate_probe_v1',root);protocol=json.loads(frozen['protocol.json'])
        download=root/'evaluation/runs/ghost_candidate_probe_v1/download';bundle=download/'ghost_candidate_bundle';bundle.mkdir(parents=True)
        for name,raw in frozen.items():(bundle/name).write_bytes(raw)
        result={'schema_version':1,'purpose':protocol['purpose'],'model':CANDIDATE,'protocol_sha256':sha(frozen['protocol.json']),
            'status':'memory_probe_passed','phase':'complete','model_calls':1,'model_weights_loaded':True,'finite_logits':True,'training_steps':0,'adapters_loaded':[],
            'quality_evaluated':False,'active_model_changed':False,'enable_thinking':False,'packages':protocol['packages'],
            'devices_before':[{'index':i,'name':'Tesla T4','free_memory_bytes':15*GIB,'total_memory_bytes':16*GIB} for i in (0,1)],
            'resources_before':{'cache_free_bytes':128*GIB,'ram_available_bytes':24*GIB},'parameter_devices':['cuda:0','cuda:1'],
            'device_map':{'layer0':'0','layer1':'1'},'generation':{'do_sample':False,'num_beams':1,'max_new_tokens':16,'use_cache':True,'eos_token_id':7},
            'probe':{'input_token_ids':json.loads(frozen['memory_probe.json'])['input_token_ids'],'token_ids':[8]*15+[7],'response':'fixture-probe-output','finish_reason':'stop'},
            'peak_allocated_bytes':{'0':12*GIB,'1':12*GIB}}
        path=download/'ghost_candidate_results/result.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(result))
        return path,result,protocol

    def test_targeted_fixed_map_preserves_workload_and_memory_caps(self):
        mapping=device_map_for('fixed_34_30')
        layers=[v for k,v in mapping.items() if k.startswith('model.layers.')]
        self.assertEqual(layers.count(0),34);self.assertEqual(layers.count(1),30)
        self.assertEqual(mapping['model.embed_tokens'],0);self.assertEqual(mapping['lm_head'],1)
        for policy,count in [('unknown',64),('fixed_34_30',63)]:
            with self.assertRaises(ValueError):device_map_for(policy,count)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,old=self.fixture(root)
            assets=root/'evaluation/ghost_candidate_probe_v2/tokenizer_assets';assets.mkdir(parents=True)
            for name in ASSETS:(assets/name).write_text('fixture')
            build('ghost_candidate_probe_v2',root,TokenizerFixture(),placement_policy='fixed_34_30')
            frozen=payloads('ghost_candidate_probe_v2',root);new=json.loads(frozen['protocol.json'])
            self.assertEqual(new['placement_policy'],'fixed_34_30')
            for key in ('model','maximum_model_calls','max_input_tokens','max_output_tokens','per_gpu_weight_budget_bytes','runtime_timeout_seconds'):
                self.assertEqual(old[key],new[key])
            old_probe=json.loads(payloads('ghost_candidate_probe_v1',root)['memory_probe.json'])
            self.assertEqual(old_probe,json.loads(frozen['memory_probe.json']))

    def test_verifier_enforces_fixed_split_and_same_weight_memory_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,result,_=self.fixture(root)
            assets=root/'evaluation/ghost_candidate_probe_v2/tokenizer_assets';assets.mkdir(parents=True)
            for name in ASSETS:(assets/name).write_text('fixture')
            build('ghost_candidate_probe_v2',root,TokenizerFixture(),placement_policy='fixed_34_30')
            frozen=payloads('ghost_candidate_probe_v2',root)
            dest=root/'evaluation/runs/ghost_candidate_probe_v2/download';bundle=dest/'ghost_candidate_bundle';bundle.mkdir(parents=True)
            for name,raw in frozen.items():(bundle/name).write_bytes(raw)
            result.update(protocol_sha256=sha(frozen['protocol.json']),placement_policy='fixed_34_30',
                device_map={name:str(device) for name,device in device_map_for('fixed_34_30').items()},
                memory_after_load={str(i):{'allocated_bytes':12*GIB,'reserved_bytes':12*GIB} for i in (0,1)})
            path=dest/'ghost_candidate_results/result.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(result))
            self.assertTrue(verify('ghost_candidate_probe_v2',root,TokenizerFixture())['memory_success'])
            for variant in ('split','weight_memory'):
                bad=copy.deepcopy(result)
                if variant=='split':bad['device_map']['model.layers.33']='1'
                else:bad['memory_after_load']['0']['allocated_bytes']=14*GIB
                path.write_text(json.dumps(bad))
                with self.subTest(variant=variant),self.assertRaises(ValueError):verify('ghost_candidate_probe_v2',root,TokenizerFixture())

    def test_resource_and_placement_gates_do_not_allow_offload_training_or_adapters(self):
        devices=[{'name':'Tesla T4','free_memory_bytes':15*GIB} for _ in (0,1)]
        resource_gate(devices,128*GIB,24*GIB,55563022432)
        for gpu,cache,ram in [(devices[:1],128*GIB,24*GIB),(devices,20*GIB,24*GIB),(devices,128*GIB,4*GIB)]:
            with self.assertRaises(ValueError):resource_gate(gpu,cache,ram,55563022432)
        params=[('a',SimpleNamespace(device='cuda:0',requires_grad=False)),('b',SimpleNamespace(device='cuda:1',requires_grad=False))]
        model=SimpleNamespace(is_loaded_in_4bit=True,peft_config=None,named_parameters=lambda:params,hf_device_map={'a':0,'b':1})
        self.assertEqual(placement(model),['cuda:0','cuda:1'])
        for change in ('cpu','trainable','adapter','offload'):
            altered=copy.deepcopy(params);bad=copy.copy(model);bad.named_parameters=lambda:altered
            if change=='cpu':altered[1][1].device='cpu'
            elif change=='trainable':altered[1][1].requires_grad=True
            elif change=='adapter':bad.peft_config={'active':'adapter'}
            else:bad.hf_device_map={'a':'disk'}
            with self.subTest(change=change),self.assertRaises(ValueError):placement(bad)
        with patch('candidate_base_probe.Path.is_dir',return_value=False),self.assertRaises(RuntimeError):run('unused','unused')

    def test_exact_frozen_result_is_memory_evidence_and_never_quality_or_activation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,result,protocol=self.fixture(root)
            report=verify(root=root,tokenizer=TokenizerFixture());self.assertTrue(report['memory_success'])
            self.assertFalse(report['quality_evaluated']);self.assertFalse(report['active_model_changed']);self.assertEqual(report['model_calls'],1)
            for change in ('model','call_count','offload','input','peak','finite_logits','final_text'):
                bad=copy.deepcopy(result)
                if change=='model':bad['model']={'model_id':'other','revision':'other'}
                elif change=='call_count':bad['model_calls']=2
                elif change=='offload':bad['device_map']['layer1']='cpu'
                elif change=='input':bad['probe']['input_token_ids'][0]=999
                elif change=='peak':bad['peak_allocated_bytes']['0']=32*GIB
                elif change=='finite_logits':bad['finite_logits']=False
                else:bad['probe']['response']='changed'
                path.write_text(json.dumps(bad))
                with self.subTest(change=change),self.assertRaises(ValueError):verify(root=root,tokenizer=TokenizerFixture())
            path.write_text(json.dumps(result));(path.parent.parent/'ghost_candidate_bundle/candidate_base_probe.py').write_text('changed')
            with self.assertRaises(ValueError):verify(root=root)

    def test_blocked_probe_remains_failure_and_frozen_scope_is_narrow(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,result,protocol=self.fixture(root)
            result.update(status='blocked',error_type='OutOfMemoryError',error='Synthetic failure',model_calls=0,model_weights_loaded=False)
            path.write_text(json.dumps(result));self.assertFalse(verify(root=root)['memory_success'])
            for key,value in [('maximum_model_calls',2),('max_output_tokens',1024),('runtime_timeout_seconds',99999),('gpu_ids',[0]),('enable_thinking',True)]:
                bad=copy.deepcopy(protocol);bad[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):validate_protocol(bad)
            self.assertEqual(MODEL['model_id'],'Qwen/Qwen3.5-9B');self.assertNotEqual(CANDIDATE,MODEL)
            script=root/'scripts/candidate_base_probe.py';script.write_text(script.read_text()+'\n# changed\n')
            with self.assertRaises(ValueError):build(root=root,tokenizer=TokenizerFixture())
        with self.assertRaises(ValueError):payloads('../escape')

if __name__=='__main__':unittest.main()
