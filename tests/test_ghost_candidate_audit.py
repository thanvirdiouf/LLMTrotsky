import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ghost_candidate_audit import build,freeze_inputs,PROBE
from build_ghost_candidate_probe import enc,sha
from candidate_audit import validate_protocol,input_ids,fresh_output,run,CANDIDATE,GIB
from verify_ghost_candidate_audit import payloads,verify_generated,score_rows,verify
from candidate_base_probe import device_map_for
from grounded_chat import MODEL


class Tokenizer:
    chat_template='fixture'
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['enable_thinking'] is False
        return [1,2,3]
    def decode(self,ids,**kwargs):return 'fixture output'


def cases_and_calls():
    cases=[{'id':'c'+str(i),'question':'Question '+str(i),'claim_ids':[],
        'segments':[{'id':'s1','kind':'source_analysis','text':'Candidate '+str(i)}],
        'expected_approved':i%2==0,'local_rationale':'LOCAL ONLY'} for i in range(10)]
    calls=[{'key':c['id']+'/1','step':1,'messages':[{'role':'system','content':'fixed auditor prompt'},
        {'role':'user','content':json.dumps({'current_user_request':c['question'],'candidate_segments':c['segments'],'source_packets':[{'text':'quoted source'}]})}]} for c in cases]
    pairs=[{'pair_id':'p'+str(i),'case_ids':['c'+str(2*i),'c'+str(2*i+1)]} for i in range(5)]
    return {'cases':cases,'conversation':[],'reviewer_only_pairs':pairs},calls


def protocol():
    return {'purpose':'candidate_neutral_audit','model':CANDIDATE,'evaluation_calls':10,'memory_probe_calls':1,
        'max_input_tokens':5120,'max_output_tokens':1024,'memory_probe_output_tokens':16,'enable_thinking':False,
        'training_steps':0,'adapters_loaded':[],'active_model_changed':False,'gpu_ids':[0,1],
        'placement_policy':'fixed_34_30','attention_policy':'repeat_kv_efficient_sdpa','per_gpu_weight_budget_bytes':13*GIB,
        'weight_file_bytes':55563022432,'runtime_timeout_seconds':2700,'installation_timeout_seconds':600,
        'generation':{'do_sample':False,'num_beams':1,'max_new_tokens':1024,'use_cache':True,'eos_token_id':9,'pad_token_id':0}}


class CandidateAuditTests(unittest.TestCase):
    def test_exact_control_messages_without_expected_label_leakage(self):
        spec,calls=cases_and_calls();jobs=freeze_inputs(spec,calls,Tokenizer())
        self.assertEqual([j['messages'] for j in jobs],[c['messages'] for c in calls])
        self.assertEqual(len(jobs),10)
        for job in jobs:self.assertEqual(set(job),{'case_id','messages','input_token_ids'})
        self.assertNotIn('LOCAL ONLY',json.dumps(jobs));self.assertNotIn('expected_approved',json.dumps(jobs))
        for change in ('order','question','candidate','extra'):
            bad=copy.deepcopy(calls)
            if change=='order':bad.reverse()
            elif change=='extra':bad.append(bad[0])
            else:
                data=json.loads(bad[0]['messages'][1]['content'])
                data['current_user_request' if change=='question' else 'candidate_segments']='changed'
                bad[0]['messages'][1]['content']=json.dumps(data)
            with self.subTest(change=change),self.assertRaises(ValueError):freeze_inputs(spec,bad,Tokenizer())

    def test_scope_local_inference_guard_and_no_automatic_resume(self):
        validate_protocol(protocol());self.assertNotEqual(MODEL,CANDIDATE)
        for key,value in [('model',MODEL),('evaluation_calls',30),('memory_probe_calls',2),('enable_thinking',True),
            ('gpu_ids',[0]),('max_output_tokens',4096),('runtime_timeout_seconds',99999),('training_steps',1),('active_model_changed',True)]:
            p=protocol();p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_protocol(p)
        with patch('candidate_audit.Path.is_dir',return_value=False),self.assertRaises(RuntimeError):run('unused','unused')
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);fresh_output(d)
            for name in ('result.json','memory_probe.json','judgments.jsonl'):
                path=d/name;path.write_text('{}')
                with self.subTest(name=name),self.assertRaises(ValueError):fresh_output(d)
                path.unlink()
        class Long(Tokenizer):
            def apply_chat_template(self,*args,**kwargs):return [1]*5121
        with self.assertRaises(ValueError):input_ids(Long(),[])

    def test_token_stop_finite_and_kernel_coverage_are_required(self):
        job={'messages':[],'input_token_ids':[1,2,3]}
        out={'input_token_ids':[1,2,3],'token_ids':[4,9],'token_count':2,'response':'fixture output','finish_reason':'stop',
            'finite_logits':True,'finite_logit_steps':2,'efficient_attention_calls':32,'expanded_kv_calls':32}
        verify_generated(job,out,1024,protocol()['generation'],Tokenizer())
        for key,value in [('input_token_ids',[1]),('token_ids',[4]),('response','invented'),('finish_reason','length'),
            ('finite_logits',False),('finite_logit_steps',1),('efficient_attention_calls',16),('expanded_kv_calls',0)]:
            bad=copy.deepcopy(out);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):verify_generated(job,bad,1024,protocol()['generation'],Tokenizer())

    def test_scoring_keeps_false_approval_rejection_and_blocks_separate(self):
        spec,_=cases_and_calls();rows=[]
        for i,c in enumerate(spec['cases']):
            approved=i not in (2,3)
            verdict={'id':'s1',**{k:approved for k in ('supported','certainty_preserved','attribution_preserved','temporal_safe','relevant','no_extra_claims')},'reason':'specific comparison'}
            rows.append({'case_id':c['id'],'segments':c['segments'],'status':'blocked' if i==4 else 'audit_approved' if approved else 'audit_rejected',
                'response':json.dumps({'verdicts':[verdict]}),'accepted_segment_ids':['s1'] if approved else []})
        # A completed but empty verdict list is unscored, never correct rejection.
        rows[5].update(response='{"verdicts":[]}',accepted_segment_ids=[],status='audit_rejected')
        report=score_rows(spec,rows)
        self.assertEqual(report['counts'],{'correct_approval':3,'false_acceptance':3,'false_rejection':1,'correct_rejection':1,'blocked':1,'unscored_judgment':1})
        self.assertEqual(report['pairs_both_correct'],0)

    def test_bundle_requires_success_and_freezes_assets_helpers_and_local_expectations(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);scripts=root/'scripts';scripts.mkdir()
            for name in ('candidate_audit.py','candidate_base_probe.py','candidate_efficient_attention.py','summarize_ghost_calibration.py'):
                (scripts/name).write_bytes((ROOT/'scripts'/name).read_bytes())
            assets=root/'evaluation'/PROBE/'tokenizer_assets';assets.mkdir(parents=True);(assets/'config.json').write_text('{}')
            stage=root/'training/kaggle_cli'/PROBE;stage.mkdir(parents=True);(stage/'kernel-metadata.json').write_text(json.dumps({'is_private':True,'id_no':140792344}))
            p=protocol();p.update(packages={'torch':'fixture'},tokenizer_files={'config.json':sha(b'{}')},chat_template_sha256=sha(b'fixture'),preserved_files={})
            probed={n:(scripts/n).read_bytes() for n in ('candidate_base_probe.py','candidate_efficient_attention.py')}
            probed.update({'protocol.json':enc(p),'memory_probe.json':enc({'messages':[],'input_token_ids':[1,2,3]})})
            spec,calls=cases_and_calls();expected=enc(spec)
            old={**p,'model':MODEL};binding={'local_version':'ghost_smoke_v30'}
            with patch('build_ghost_candidate_audit.probe_payloads',return_value=probed),patch('build_ghost_candidate_audit.control',return_value=({},old,spec,expected,calls,binding)),patch('build_ghost_candidate_audit.verify_probe') as verified:
                verified.return_value={'memory_success':False}
                with self.assertRaises(ValueError):build(root=root,tokenizer=Tokenizer())
                verified.return_value={'memory_success':True,'tokenizer_reconstructed_locally':True,'result_sha256':'fixture'}
                build(root=root,tokenizer=Tokenizer());frozen=payloads('ghost_candidate_audit_v1',root)
                self.assertNotIn('evaluation_spec.json',frozen);self.assertNotIn('LOCAL ONLY',frozen['audit_inputs.json'].decode())
                self.assertEqual((root/'evaluation/ghost_candidate_audit_v1/evaluation_spec.json').read_bytes(),expected)
                self.assertEqual(json.loads(frozen['protocol.json'])['model'],CANDIDATE)
                # Full verifier path: perfect synthetic decisions must remain
                # development evidence, with token and artifact checks enforced.
                dest=root/'evaluation/runs/ghost_candidate_audit_v1/download';bundle=dest/'ghost_candidate_audit_bundle';bundle.mkdir(parents=True)
                for name,raw in frozen.items():(bundle/name).write_bytes(raw)
                output=dest/'ghost_candidate_audit_results';output.mkdir()
                jobs=json.loads(frozen['audit_inputs.json']);outputs=[];replayed=[]
                out={'input_token_ids':[1,2,3],'token_ids':[4,9],'token_count':2,'response':'fixture output','finish_reason':'stop',
                    'finite_logits':True,'finite_logit_steps':2,'efficient_attention_calls':32,'expanded_kv_calls':32}
                for case,job in zip(spec['cases'],jobs):
                    digest=sha(json.dumps(job['messages'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
                    outputs.append({'case_id':case['id'],'model':CANDIDATE,'input_sha256':digest,'output':out})
                    yes=case['expected_approved'];verdict={'id':'s1',**{k:yes for k in ('supported','certainty_preserved','attribution_preserved','temporal_safe','relevant','no_extra_claims')},'reason':'specific comparison'}
                    replayed.append({'case_id':case['id'],'segments':case['segments'],'status':'audit_approved' if yes else 'audit_rejected','response':json.dumps({'verdicts':[verdict]}),'accepted_segment_ids':['s1'] if yes else []})
                (output/'judgments.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in outputs));(output/'memory_probe.json').write_text(json.dumps(out))
                frozen_p=json.loads(frozen['protocol.json'])
                result={k:frozen_p[k] for k in ('model','purpose','training_steps','adapters_loaded','active_model_changed','enable_thinking','packages','generation')}
                result.update(quality_validated=False,protocol_sha256=sha(frozen['protocol.json']),status='complete',evaluation_calls=10,memory_probe_calls=1,
                    parameter_devices=['cuda:0','cuda:1'],device_map={k:str(v) for k,v in device_map_for('fixed_34_30').items()},
                    memory_after_load={str(i):{'allocated_bytes':12*GIB} for i in (0,1)},
                    resources_before={'cache_free_bytes':128*GIB,'ram_available_bytes':24*GIB},
                    devices_before=[{'index':i,'name':'Tesla T4','free_memory_bytes':15*GIB,'total_memory_bytes':16*GIB} for i in (0,1)],
                    attention_equivalence_checks=[{'device':i,'case':c,'passed':True,'atol':0.002,'rtol':0.01} for i in (0,1) for c in ('prefill_causal','decode','masked_prefill')],
                    attention_runtime_stats={'backend':'EFFICIENT_ATTENTION','native_gqa':False,'mask_function':'sdpa_mask','calls':352,'expanded_kv_calls':352},
                    peak_allocated_bytes={'0':13*GIB,'1':13*GIB},judgments_sha256=sha((output/'judgments.jsonl').read_bytes()),memory_probe_sha256=sha((output/'memory_probe.json').read_bytes()))
                with patch('verify_ghost_candidate_audit.control',return_value=({},old,spec,expected,calls,binding)),patch('verify_ghost_candidate_audit.replay',return_value=replayed):
                    path=output/'result.json';path.write_text(json.dumps(result));report=verify(root=root,tokenizer=Tokenizer())
                    self.assertEqual(report['scores']['pairs_both_correct'],5);self.assertFalse(report['quality_validated']);self.assertFalse(report['active_model_changed'])
                    for change in ('model','call_count','offload','backend','generation','hash'):
                        bad=copy.deepcopy(result)
                        if change=='model':bad['model']=MODEL
                        elif change=='call_count':bad['evaluation_calls']=11
                        elif change=='offload':bad['device_map']['lm_head']='cpu'
                        elif change=='backend':bad['attention_runtime_stats']['calls']=351
                        elif change=='generation':bad['generation']['do_sample']=True
                        else:bad['judgments_sha256']='changed'
                        path.write_text(json.dumps(bad))
                        with self.subTest(change=change),self.assertRaises(ValueError):verify(root=root,tokenizer=Tokenizer())
                (scripts/'candidate_efficient_attention.py').write_text('changed')
                with self.assertRaises(ValueError):build(root=root,tokenizer=Tokenizer())
        with self.assertRaises(ValueError):payloads('../escape')


if __name__=='__main__':unittest.main()
