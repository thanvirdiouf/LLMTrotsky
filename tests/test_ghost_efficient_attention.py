from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from candidate_efficient_attention import make_backend,register_with,NAME
from candidate_base_probe import validate_protocol,ensure_fresh_output
from build_ghost_candidate_probe import build,ASSETS
from verify_ghost_candidate_probe import payloads,verify
import test_ghost_candidate_probe as probe_tests
from test_ghost_candidate_probe import TokenizerFixture

class EfficientAttentionTest(unittest.TestCase):
    def test_existing_probe_output_cannot_trigger_a_second_model_call(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);ensure_fresh_output(output)
            for status in ('initializing','blocked','memory_probe_passed'):
                (output/'result.json').write_text(json.dumps({'status':status}))
                with self.subTest(status=status),self.assertRaises(RuntimeError):ensure_fresh_output(output)

    def test_expanded_kv_uses_only_efficient_backend_and_preserves_mask_and_module(self):
        events=[];stats={};mask=object();module=SimpleNamespace(num_key_value_groups=6,is_causal=True,layer_idx=3)
        q=SimpleNamespace(shape=(1,24,64,256));k=SimpleNamespace(shape=(1,4,64,256));v=SimpleNamespace(shape=(1,4,64,256))
        def repeat(tensor,groups):
            events.append(('repeat',groups));shape=list(tensor.shape);shape[1]*=groups
            return SimpleNamespace(shape=tuple(shape))
        @contextmanager
        def context(backend):events.append(('backend',backend));yield
        def sdpa(proxy,query,key,value,actual_mask,**kwargs):
            self.assertEqual(proxy.num_key_value_groups,1);self.assertTrue(proxy.is_causal);self.assertEqual(proxy.layer_idx,3)
            self.assertIs(actual_mask,mask);self.assertIs(query,q)
            self.assertEqual(key.shape[1],24);self.assertEqual(value.shape[1],24)
            self.assertEqual(kwargs,{'dropout':0.0,'scaling':0.0625});return 'out',None
        backend=make_backend(sdpa,repeat,context,'EFFICIENT_ONLY',stats)
        self.assertEqual(backend(module,q,k,v,mask,dropout=0.0,scaling=0.0625),('out',None))
        self.assertEqual(module.num_key_value_groups,6);self.assertEqual(stats['calls'],1);self.assertEqual(stats['expanded_kv_calls'],1)
        self.assertFalse(stats['native_gqa']);self.assertEqual(events,[('repeat',6),('repeat',6),('backend','EFFICIENT_ONLY')])
        with self.assertRaises(ValueError):backend(module,q,k,v,mask,cache=object())
        with self.assertRaises(ValueError):backend(SimpleNamespace(num_key_value_groups=3),q,k,v,mask)

    def test_matching_mask_registry_is_installed_without_replacing_default_sdpa(self):
        class Registry:
            def __init__(self):self.values={'sdpa':'original'}
            def register(self,name,value):self.values[name]=value
        attn,mask=Registry(),Registry();fn=object();mask_fn=object()
        self.assertEqual(register_with(attn,mask,mask_fn,fn),NAME)
        self.assertIs(attn.values[NAME],fn);self.assertIs(mask.values[NAME],mask_fn)
        self.assertEqual(attn.values['sdpa'],'original');self.assertEqual(mask.values['sdpa'],'original')

    def test_policy_freezes_helper_and_verifier_requires_real_kernel_checks_and_coverage(self):
        from candidate_base_probe import device_map_for,GIB
        from build_ghost_candidate_probe import sha
        fixture=probe_tests.CandidateProbeTest()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,result,baseline=fixture.fixture(root)
            script=root/'scripts/candidate_efficient_attention.py';script.write_bytes((ROOT/'scripts/candidate_efficient_attention.py').read_bytes())
            assets=root/'evaluation/ghost_candidate_probe_v3/tokenizer_assets';assets.mkdir(parents=True)
            for name in ASSETS:(assets/name).write_text('fixture')
            build('ghost_candidate_probe_v3',root,TokenizerFixture(),placement_policy='fixed_34_30',attention_policy='repeat_kv_efficient_sdpa')
            frozen=payloads('ghost_candidate_probe_v3',root);protocol=json.loads(frozen['protocol.json']);validate_protocol(protocol)
            self.assertIn('candidate_efficient_attention.py',frozen)
            for key in ('model','maximum_model_calls','max_input_tokens','max_output_tokens','per_gpu_weight_budget_bytes'):
                self.assertEqual(baseline[key],protocol[key])
            dest=root/'evaluation/runs/ghost_candidate_probe_v3/download';bundle=dest/'ghost_candidate_bundle';bundle.mkdir(parents=True)
            for name,raw in frozen.items():(bundle/name).write_bytes(raw)
            result.update(protocol_sha256=sha(frozen['protocol.json']),placement_policy='fixed_34_30',attention_policy='repeat_kv_efficient_sdpa',
                device_map={k:str(v) for k,v in device_map_for('fixed_34_30').items()},
                memory_after_load={str(i):{'allocated_bytes':12*GIB,'reserved_bytes':12*GIB} for i in (0,1)},
                attention_equivalence_checks=[{'device':d,'case':c,'passed':True,'atol':0.002,'rtol':0.01,'max_abs_error':0.0001} for d in (0,1) for c in ('prefill_causal','decode','masked_prefill')],
                attention_runtime_stats={'backend':'EFFICIENT_ATTENTION','native_gqa':False,'mask_function':'sdpa_mask','calls':256,'expanded_kv_calls':256})
            path=dest/'ghost_candidate_results/result.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(result))
            self.assertTrue(verify('ghost_candidate_probe_v3',root,TokenizerFixture())['memory_success'])
            for change in ('checks','calls','mask','native_gqa'):
                bad=copy.deepcopy(result)
                if change=='checks':bad['attention_equivalence_checks'].pop()
                elif change=='calls':bad['attention_runtime_stats']['calls']=0
                elif change=='mask':bad['attention_runtime_stats']['mask_function']='missing'
                else:bad['attention_runtime_stats']['native_gqa']=True
                path.write_text(json.dumps(bad))
                with self.subTest(change=change),self.assertRaises(ValueError):verify('ghost_candidate_probe_v3',root,TokenizerFixture())
            bad=copy.deepcopy(protocol);bad['placement_policy']='auto'
            with self.assertRaises(ValueError):validate_protocol(bad)

if __name__=='__main__':unittest.main()
